"""Replay the saved scene into a free-radius common section model.

Field ROIs below are explicitly selected from the saved images. This is not
automatic arbitrary-scene segmentation. Procurement sizes are comparisons
only. The hidden red contour is never reused as its own cylinder's tangent.
"""
from pathlib import Path
import csv
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'test_reports/20261021-projection-strip'))
from probe_projection import SOURCE, extract, read, robust_edge, sha
from pipe_twin.projection_diagnostics import fit_silhouettes, pixel_error_scenarios, projected_lines
from pipe_twin.section_diagnostics import section_basis
from pipe_twin.section_model import build_section_model, write_section_artifacts

OUT = ROOT/'outputs/section-model-20261021'
CATALOG = {'blue': 50., 'red': 40., 'white': 25.}


def red_edges(image, role, width=320):
    """Separate the front red silhouette from the back red occlusion boundary.

    The seam seed is between highlights, not the strongest edge in the whole
    red envelope. Its silhouette interpretation remains a field hypothesis.
    """
    center = 1260 if role == 'left' else 1140
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165)) & (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 40)
    red[:50] = False; red[280:] = False
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32), (5, 5), .9)
    gradient = np.gradient(gray, axis=0)
    points = {name: [] for name in ['rear_top', 'front_top', 'front_bottom']}
    for x in range(center-width//2, center+width//2):
        ys = np.flatnonzero(red[:, x])
        if len(ys) < 70:
            continue
        points['rear_top'].append([x, ys.min()-.5])
        points['front_bottom'].append([x, ys.max()+.5])
        prediction = 153.5-.062*(x-center)
        lo, hi = round(prediction-7), round(prediction+7)
        y = lo+int(np.argmax(gradient[lo:hi+1, x]))
        if y in (lo, hi) or gradient[y, x] < 3:
            continue
        f = gradient[y-1:y+2, x]
        offset = .5*(f[0]-f[2])/(f[0]-2*f[1]+f[2])
        points['front_top'].append([x, y+offset])
    observations, audit = {}, {}
    for name, pixels in points.items():
        retained, audit[name] = robust_edge(pixels)
        observations[name] = {'role': role, 'edge': 'bottom' if name == 'front_bottom' else 'top', 'points': retained}
    return observations, audit


def partial_end_rim(image, role):
    """Record a visible partial end contour; do not assert it is a true circle."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 0] < 15) | (hsv[:, :, 0] > 165)) & (hsv[:, :, 1] > 120) & (hsv[:, :, 2] > 70)
    x0, x1 = (295, 355) if role == 'left' else (195, 255)
    points = []
    for y in range(164, 216):
        xs = np.flatnonzero(mask[y, x0:x1])
        if len(xs) and 0 < xs[0] < x1-x0-1:
            points.append([x0+xs[0]-.5, y])
    points = np.asarray(points)
    # A highlight can erase the paint at the end and make the first red pixel
    # jump far into the body. Keep the local end-contour support; do not fill
    # the discarded rows or reinterpret a highlight edge as a circular rim.
    median_x = np.median(points[:, 0])
    keep = np.abs(points[:, 0]-median_x) <= 6.
    return {'role': role, 'points': points[keep], 'discarded_highlight_rows': int((~keep).sum())}


def partial_rim_trials(rims, observations, foreground, k, baseline, axis):
    """Spend a larger bounded budget testing free-radius end-circle hypotheses.

    Assumes a shared planar end rim normal to the axis; different seeds and
    visible-occlusion consistency expose failures of this extra assumption.
    Never selects a hypothesis for the section model merely for lowest RMS.
    """
    basis = section_basis(axis); inverse_k = np.linalg.inv(k)
    visible = [o for o in observations if o.get('boundary_kind') != 'occlusion']

    def residual(p):
        center = basis@p[:2]+axis*p[2]; radius = p[3]
        if radius <= 0 or center[2] <= radius:
            raise ValueError('INVALID_END_CIRCLE')
        values = []
        for obs in rims:
            camera = np.zeros(3) if obs['role'] == 'left' else np.array([baseline, 0., 0.])
            c = center-camera; h = axis@c
            matrix = h*np.eye(3)-np.outer(c, axis)
            conic = inverse_k.T@(matrix.T@matrix-radius*radius*np.outer(axis, axis))@inverse_k
            conic /= np.linalg.norm(conic)
            pixels = np.column_stack([obs['points'], np.ones(len(obs['points']))])
            gradient = 2*(pixels@conic)
            values.extend(np.sum((pixels@conic)*pixels, axis=1)/np.maximum(np.linalg.norm(gradient[:, :2], axis=1), 1e-12))
        for obs in visible:
            camera = np.zeros(3) if obs['role'] == 'left' else np.array([baseline, 0., 0.])
            line = projected_lines(p[:2], radius, basis, k, camera, 1260)[0 if obs['edge'] == 'top' else 1]
            values.extend(obs['points']@line[:2]+line[2])
        return np.asarray(values)

    lx, rx = [np.median(o['points'][:, 0]) for o in rims]
    yy = np.mean(rims[0]['points'][:, 1])
    z = k[0, 0]*baseline/(lx-rx)
    station = axis@np.array([(lx-k[0, 2])*z/k[0, 0], (yy-k[1, 2])*z/k[1, 1], z])
    trials = []
    for radius in [5., 8., 12., 18., 27., 40., 60., 90.]:
        # Numeric starts only, not catalogue sizes and not fixed in refinement.
        seed = fit_silhouettes(visible, k, baseline, axis, 1260, known_diameter_mm=2*radius)
        p = np.array([*seed['section_center_parameters_mm'], station, radius])
        damping = 1e-3
        for iteration in range(480):
            error = residual(p)
            jac = np.column_stack([(residual(p+np.eye(4)[j]*.001)-error)/.001 for j in range(4)])
            weights = 1/np.maximum(1, np.abs(error)/2.)
            normal = jac.T@(weights[:, None]*jac)
            step = np.linalg.solve(normal+damping*np.diag(np.maximum(np.diag(normal), 1e-9)), -jac.T@(weights*error))
            candidate = p+step
            try:
                improves = np.sum(weights*residual(candidate)**2) < np.sum(weights*error**2)
            except ValueError:
                improves = False
            if improves:
                p = candidate; damping = max(damping/3, 1e-9)
                if np.linalg.norm(step) < 1e-7:
                    break
            else:
                damping = min(damping*10, 1e12)
        consistency = []
        for role in ['left', 'right']:
            camera = np.zeros(3) if role == 'left' else np.array([baseline, 0., 0.])
            line = projected_lines(p[:2], p[3], basis, k, camera, 1260)[1]
            reference = 1260 if role == 'left' else 1140
            bottom = float(-(line[0]*reference+line[2])/line[1])
            bounds = []
            for edge in ['top', 'bottom']:
                obs = next(o for o in foreground if o['role'] == role and o['edge'] == edge)
                slope, intercept = np.polyfit(obs['points'][:, 0], obs['points'][:, 1], 1)
                bounds.append(float(slope*reference+intercept))
            consistency.append({'role': role, 'predicted_rear_bottom_v_px': bottom,
                                'foreground_top_bottom_v_px': bounds,
                                'hidden_inside_foreground_envelope': bool(bounds[0]-1 <= bottom <= bounds[1]+1)})
        trials.append({'initial_radius_mm': radius, 'iterations': iteration+1,
                       'hypothesis_diameter_mm': float(2*p[3]), 'rms_px': float(np.sqrt(np.mean(residual(p)**2))),
                       'section_center_uv_mm': p[:2].tolist(), 'visibility_checks': consistency})
    return {'status': 'AMBIGUOUS_END_RIM_HYPOTHESES_NOT_USED', 'max_iterations_per_seed': 480,
            'assumption': 'VISIBLE_END_EDGE_IS_ONE_CIRCULAR_RIM_NORMAL_TO_PIPE_AXIS',
            'observed_points': {o['role']: len(o['points']) for o in rims},
            'discarded_highlight_rows': {o['role']: o['discarded_highlight_rows'] for o in rims}, 'trials': trials,
            'accepted_diameter_mm': None}


def observations_for(images, width=320):
    observations = {name: [] for name in ['blue', 'red_front', 'red_rear']}
    audit = {}
    for role, image in images.items():
        observations['blue'] += extract(image, role, width)[0]
        red, audit[role] = red_edges(image, role, width)
        observations['red_front'] += [red['front_top'], red['front_bottom']]
        observations['red_rear'] += [red['rear_top'], {**red['front_top'], 'edge': 'bottom', 'boundary_kind': 'occlusion'}]
    return observations, audit


def stats(values):
    return {'mean': float(np.mean(values)), 'std': float(np.std(values, ddof=1)),
            'min': float(np.min(values)), 'max': float(np.max(values))}


def evidence_plot(images, observations, fits, rims):
    sys.path.insert(0, str(ROOT/'outputs/section-check-20261021/plot-deps'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 7), layout='constrained')
    for index, role in enumerate(['left', 'right']):
        ax = axes[index, 0]; ax.imshow(cv2.cvtColor(images[role], cv2.COLOR_BGR2RGB))
        for name, color in [('blue', '#00ffff'), ('red_front', '#ffff00'), ('red_rear', '#00ff00')]:
            for obs in observations[name]:
                if obs['role'] == role and obs.get('boundary_kind') != 'occlusion':
                    pixels = obs['points']; ax.scatter(pixels[:, 0], pixels[:, 1], s=3, c=color)
            if fits[name]['status'] == 'CONDITIONAL_FIT':
                for line in fits[name]['predicted_lines'][role]:
                    xs = np.array([1000, 1440])
                    ax.plot(xs, -(line[0]*xs+line[2])/line[1], '--', c=color, lw=1)
        center = 1260 if role == 'left' else 1140
        ax.set_xlim(center-180, center+180); ax.set_ylim(460, 60)
        ax.set_title(f'{role}：蓝管青色、前红黄色、后红可见上缘绿色；虚线为重投影')
        ax = axes[index, 1]; ax.imshow(cv2.cvtColor(images[role], cv2.COLOR_BGR2RGB))
        rim = next(o['points'] for o in rims if o['role'] == role)
        ax.scatter(rim[:, 0], rim[:, 1], s=7, c='#00ff00')
        ax.set_xlim((285, 380) if role == 'left' else (185, 280)); ax.set_ylim(235, 145)
        ax.set_title(f'{role}：后红局部端缘像素；未将侧视端缘当已恢复圆弧')
    fig.savefig(OUT/'observed-contours.png', dpi=150); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    previous = read(ROOT/'outputs/section-check-20261021/evidence.json')
    calibration = ROOT/'outputs/measurement_workbench/calibration_current.json'
    assert sha(calibration) == previous['calibration_sha256']
    for hashes in previous['source_image_hashes'].values():
        for name, digest in hashes.items():
            assert sha(SOURCE/name) == digest
    k = np.asarray(previous['rectified_intrinsic']); axis = np.asarray(previous['axis']['axis'])
    axis /= np.linalg.norm(axis); baseline = previous['active_baseline_mm']
    frames = []
    for index in range(20):
        images = {role: cv2.imdecode(np.fromfile(SOURCE/f'{index:03d}_{role}.png', np.uint8), 1) for role in ['left', 'right']}
        observations, audit = observations_for(images)
        fits = {name: fit_silhouettes(obs, k, baseline, axis, 1260) for name, obs in observations.items()}
        items = [{'id': name, 'fit': fit} for name, fit in fits.items()]
        per_frame = build_section_model(items, axis, origin_id='blue')
        frames.append({'frame': index, 'fits': fits, 'red_edge_audit': audit, 'pairs': per_frame['pairs']})
        print(index, {name: fit.get('trial_diameter_mm') for name, fit in fits.items()}, flush=True)
    errors = {name: pixel_error_scenarios(obs, k, baseline, axis, 1260) for name, obs in observations.items()}
    candidates = []
    for name, label, color, reference in [('blue', '蓝管 B1', '#2274bc', CATALOG['blue']),
                                         ('red_front', '前方红管 R1', '#e04432', CATALOG['red']),
                                         ('red_rear', '后方红管 R2（遮挡）', '#ad3c78', CATALOG['red'])]:
        group = [frame['fits'][name] for frame in frames]
        if all(f['status'] == 'CONDITIONAL_FIT' for f in group):
            fit = {'status': 'CONDITIONAL_FIT', 'axis': axis.tolist(), 'diameter_source': 'IMAGE_FIT',
                   'trial_diameter_mm': float(np.mean([f['trial_diameter_mm'] for f in group])),
                   'section_center_parameters_mm': np.mean([f['section_center_parameters_mm'] for f in group], axis=0).tolist()}
        else:
            fit = group[-1]
        candidates.append({'id': name, 'label': label, 'color': color, 'reference_diameter_mm': reference,
                           'fit': fit, 'pixel_error': errors[name]})
    rims = [partial_end_rim(images[role], role) for role in ['left', 'right']]
    rim_trials = partial_rim_trials(rims, observations['red_rear'], observations['red_front'], k, baseline, axis)
    variations = []
    for width in [160, 640]:
        selected, _ = observations_for(images, width)
        changed = [{'id': name, 'fit': fit_silhouettes(obs, k, baseline, axis, 1260)} for name, obs in selected.items()]
        variations.append({'width_px': width, 'section': build_section_model(changed, axis, origin_id='blue')})
    repeatability = {name: stats([f['fits'][name]['trial_diameter_mm'] for f in frames]) for name in ['blue', 'red_front']}
    repeatability['center_distance'] = stats([f['pairs'][0]['center_distance_mm'] for f in frames])
    repeatability['clearance'] = stats([f['pairs'][0]['surface_clearance_mm'] for f in frames])
    model = build_section_model(candidates, axis, origin_id='blue', station_mm=float(axis@np.mean(
        [f['fits']['blue']['center_at_reference_column_mm'] for f in frames], axis=0)), metadata={
            'source': str(SOURCE.relative_to(ROOT)), 'frames': 20, 'capture_date': '2026-10-20',
            'calibration_sha256': sha(calibration), 'calibration_validated': False, 'active_baseline_mm': baseline,
            'reported_baseline_mm': 60., 'reference_catalog_mm': CATALOG, 'reference_sizes_used_in_fitting': False,
            'white_pipe': 'NOT_IDENTIFIED_IN_THIS_CAPTURE', 'automatic_instance_segmentation': False,
            'aggregation': 'MEAN_OF_20_FREE_FITS_IN_THE_SAME_SECTION_BASIS',
            'pixel_error_evaluation': 'REPRESENTATIVE_LAST_FRAME; NO_SQRT_20_REDUCTION',
            'rear_red': rim_trials['status'], 'repeatability_not_absolute_accuracy': repeatability})
    write_section_artifacts(model, OUT)
    evidence = {'model': model, 'repeatability': repeatability, 'frames': frames,
                'source_image_hashes': previous['source_image_hashes'], 'partial_rim_trials': rim_trials,
                'window_sensitivity': variations, 'code_hashes': {str(p.relative_to(ROOT)): sha(p) for p in
                    [Path(__file__), ROOT/'pipe_twin/section_model.py', ROOT/'pipe_twin/projection_diagnostics.py']}}
    (OUT/'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (OUT/'rear-red-partial-contour.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream); writer.writerow(['role', 'u_px', 'v_px', 'interpretation'])
        for rim in rims:
            for point in rim['points']:
                writer.writerow([rim['role'], *point, 'OBSERVED_IMAGE_EDGE_NOT_A_3D_CIRCLE_POINT'])
    np.savez_compressed(OUT/'observed-edge-pixels.npz', **{f'{name}_{obs["role"]}_{obs["edge"]}': obs['points']
                                                       for name, obslist in observations.items() for obs in obslist})
    evidence_plot(images, observations, fits, rims)
    html = (OUT/'index.html').read_text(encoding='utf-8')
    html = html.replace('</html>', '<h2>实际可见轮廓</h2><p>黄色分界用于前方红管的上切线；它对后方红管是遮挡线，不参与后方半径拟合。后方红管另保存了双目局部端缘，扩大到 8 组初值、每组最多 480 步仍存在不同半径假设，尚未补成确定圆。白色 25 mm 仅在参照表中登记，本组照片未定位白管。</p><img src="observed-contours.png" style="max-width:100%" alt="可见轮廓与遮挡端缘"><p><a href="rear-red-partial-contour.csv">后红局部轮廓像素 CSV</a></p></html>')
    (OUT/'index.html').write_text(html, encoding='utf-8')
    print(json.dumps({'repeatability': repeatability, 'pairs': model['pairs'], 'output': str(OUT)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
