"""Axial pooling and conditional stereo silhouette experiment on one blue strip."""
from pathlib import Path
import hashlib
import json
import sys
from itertools import product

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.projection_diagnostics import fit_silhouettes, pool_strip_mask, pixel_error_scenarios

OUT = ROOT/'outputs/projection-strip-20261021'
SOURCE = ROOT/'outputs/retry-20261020-131151/active-retry/rectified'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def robust_edge(points):
    points = np.asarray(points, float)
    matrix = np.column_stack((points[:, 0], np.ones(len(points))))
    keep = np.ones(len(points), bool)
    for _ in range(4):
        slope, intercept = np.linalg.lstsq(matrix[keep], points[keep, 1], rcond=None)[0]
        error = points[:, 1]-matrix@[slope, intercept]
        scale = 1.4826*np.median(np.abs(error-np.median(error)))
        keep = np.abs(error) <= max(2., 3*scale)
    slope, intercept = np.linalg.lstsq(matrix[keep], points[keep, 1], rcond=None)[0]
    error = points[:, 1]-matrix@[slope, intercept]
    return points[keep], {'slope': float(slope), 'intercept': float(intercept),
                          'source_points': len(points), 'retained_points': int(keep.sum()),
                          'rms_px': float(np.sqrt(np.mean(error[keep]**2)))}


def extract(image, role, width=320):
    # Fixed visible, unoccluded blue patch selected from the saved image.
    # Right patch is a similar axial neighbourhood, not matched material ends.
    center = 1260 if role == 'left' else 1140
    x0, x1 = center-width//2, center+width//2
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 120) & (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 40)
    mask[:280] = False; mask[465:] = False
    points = {'top': [], 'bottom': []}
    for x in range(x0, x1):
        y = np.flatnonzero(mask[:, x])
        if len(y) >= 40:
            points['top'].append([x, y.min()-.5])
            points['bottom'].append([x, y.max()+.5])
    observations, edges = [], {}
    for edge in points:
        retained, audit = robust_edge(points[edge])
        observations.append({'role': role, 'edge': edge, 'points': retained})
        edges[edge] = audit
    slope = np.mean([v['slope'] for v in edges.values()])
    pool = pool_strip_mask(mask, x0, x1, 280, 465, slope)
    intervals = pool['intervals']
    pool['selected_blue_outer_envelope_px'] = intervals[-1]['high_q_px']-intervals[0]['low_q_px'] if intervals else None
    pool['interpretation'] = 'ONE_VISUALLY_SELECTED_BLUE_ENVELOPE_WITH_POSSIBLE_HIGHLIGHT_GAPS'
    return observations, {'role': role, 'width_along_image_x_px': width, 'edges': edges, 'pool': pool}


def plots(frame, views, observations, fit, frames):
    sys.path.insert(0, str(ROOT/'outputs/section-check-20261021/plot-deps'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), layout='constrained')
    for i, role in enumerate(['left', 'right']):
        ax = axes[i, 0]
        ax.imshow(cv2.cvtColor(frame[role], cv2.COLOR_BGR2RGB))
        for obs in observations:
            if obs['role'] == role:
                p = np.array(obs['points'])
                ax.scatter(p[:, 0], p[:, 1], s=2, c='yellow')
        x0, x1 = views[role]['pool']['columns']
        ax.set_xlim(x0-20, x1+20); ax.set_ylim(460, 280)
        ax.set_title(f'{role}：选取 {x1-x0} 列，黄色为色彩边缘实测像素')
        ax.set_xlabel('原图 u / px'); ax.set_ylabel('原图 v / px')
        profile = views[role]['pool']
        ax = axes[i, 1]
        ax.plot(profile['q_px'], profile['paint_fraction'], label='蓝色像素所占比例')
        ax.axhline(.5, color='gray', ls='--')
        for interval in profile['intervals']:
            ax.axvspan(interval['low_q_px'], interval['high_q_px'], color='tab:blue', alpha=.12)
        ax.set_title(f'{role} 汇聚外宽 {profile["selected_blue_outer_envelope_px"]:.2f} px；位置保留')
        ax.set_xlabel('横向投影坐标 q / px（未独立居中）'); ax.set_ylabel('沿轴向的支持比例')
        ax.set_xlim(280, 465); ax.set_ylim(-.02, 1.02); ax.grid(alpha=.3)
    fig.suptitle('一段管道 → 轴向汇聚剖面；内部缺口仍显示，不生成深度')
    fig.savefig(OUT/'pooled-profile.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout='constrained')
    for role in ['left', 'right']:
        axes[0].plot([r['frame'] for r in frames], [r['views'][role]['pool']['selected_blue_outer_envelope_px'] for r in frames], '.-', label=role)
    axes[0].set_title('汇聚宽度：图像量'); axes[0].set_ylabel('像素'); axes[0].legend()
    axes[1].plot([r['frame'] for r in frames], [r['fit'].get('trial_diameter_mm', np.nan) for r in frames], '.-')
    axes[1].set_title('条件试拟合外径：未验收'); axes[1].set_ylabel('mm')
    axes[2].plot([r['frame'] for r in frames], [r['fit'].get('center_at_reference_column_mm', [0,0,np.nan])[2] for r in frames], '.-')
    axes[2].set_title('条件轴心 Z：未验收'); axes[2].set_ylabel('mm')
    for ax in axes:
        ax.set_xlabel('帧序号'); ax.grid(alpha=.3)
    fig.suptitle('同一 320 px 段、20 帧独立处理；重复稳定不等于绝对精度')
    fig.savefig(OUT/'repeatability.png', dpi=160); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    previous = read(ROOT/'outputs/section-check-20261021/evidence.json')
    cal_path = ROOT/'outputs/measurement_workbench/calibration_current.json'
    assert sha(cal_path) == previous['calibration_sha256']
    for role in previous['source_image_hashes']:
        for name, value in previous['source_image_hashes'][role].items():
            assert sha(SOURCE/name) == value
    k = np.array(previous['rectified_intrinsic'])
    axis = np.array(previous['axis']['axis'])
    baseline = previous['active_baseline_mm']
    frames = []
    for index in range(20):
        images = {role: cv2.imdecode(np.fromfile(SOURCE/f'{index:03d}_{role}.png', np.uint8), cv2.IMREAD_COLOR)
                  for role in ['left', 'right']}
        observations, views = [], {}
        for role in images:
            edges, views[role] = extract(images[role], role)
            observations += edges
        fit = fit_silhouettes(observations, k, baseline, axis, 1260.)
        frames.append({'frame': index, 'views': views, 'fit': fit})
        print(index, fit.get('trial_diameter_mm'), fit.get('reprojection_rms_px'), flush=True)
    perturbations = []
    # Deliberate correlated edge offsets, not independent pixel noise divided
    # by sqrt(columns). These are stress tests, not statistical confidence CIs.
    for j, obs in enumerate(observations):
        for shift in [-1., -.5, .5, 1.]:
            changed = [{**o, 'points': o['points'].copy()} for o in observations]
            changed[j]['points'][:, 1] += shift
            perturbations.append({'role': obs['role'], 'edge': obs['edge'], 'shift_px': shift,
                                  'fit': fit_silhouettes(changed, k, baseline, axis, 1260.)})
    variations = []
    for role, record in previous['axis']['by_role'].items():
        variations.append({'axis_source': role, 'fit': fit_silhouettes(observations, k, baseline, record['axis'], 1260.)})
    variations.append({'axis_source': 'zero_camera_z', 'fit': fit_silhouettes(
        observations, k, baseline, [axis[0], axis[1], 0.], 1260.)})
    windows = []
    for width in [160, 640]:
        edges, by_role = [], {}
        for role in images:
            parts, by_role[role] = extract(images[role], role, width)
            edges += parts
        windows.append({'width_px': width, 'views': by_role, 'fit': fit_silhouettes(edges, k, baseline, axis, 1260.)})
    pixel_error = pixel_error_scenarios(observations, k, baseline, axis, 1260., sigma_px=.5)
    corner_stress = []
    for signs in product([-1, 1], repeat=4):
        changed = [{**o, 'points': o['points']+np.array([0., sign*.5])}
                   for o, sign in zip(observations, signs)]
        corner_stress.append({'edge_offsets_px': [sign*.5 for sign in signs],
                              'fit': fit_silhouettes(changed, k, baseline, axis, 1260.)})
    compact_frames = [{**row, 'views': {role: {**value, 'pool': {a:b for a,b in value['pool'].items()
                      if a not in ('q_px', 'paint_fraction')}} for role,value in row['views'].items()}} for row in frames]
    evidence = {'scope': 'OFFLINE_AXIAL_IMAGE_POOLING_AND_CONDITIONAL_PROJECTION',
                'boundary_source': 'BLUE_COLOR_ENVELOPE_NOT_YET_VALIDATED_AS_PHYSICAL_SILHOUETTE',
                'physical_length_known': False, 'nominal_radius_used': False, 'accepted_measurements': [],
                'calibration_sha256': sha(cal_path), 'calibration_validated': False,
                'active_baseline_mm': baseline, 'reported_baseline_mm': 60., 'fixed_axis': axis.tolist(),
                'source_image_hashes': previous['source_image_hashes'],
                'code_hashes': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT/'pipe_twin/projection_diagnostics.py']},
                'frames': compact_frames, 'last_profiles': views, 'edge_shift_stress_tests': perturbations,
                'axis_sensitivity': variations, 'window_sensitivity': windows,
                'pixel_error_scenarios': pixel_error, 'four_edge_half_pixel_corner_stress': corner_stress}
    (OUT/'evidence.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    np.savez_compressed(OUT/'last-edge-pixels.npz', **{f'{o["role"]}_{o["edge"]}':o['points'] for o in observations})
    plots(images, views, observations, fit, frames)
    print('output', OUT, flush=True)


if __name__ == '__main__':
    main()
