"""Replay local measured disparities into conditional section scatter plots.

Uses the hash-bound 2026-10-20 cache, verifies the final pair by recomputing
stereo, and repeats the current temporal reducer with bidirectional LR checks.
Images, point arrays and plots stay under ignored outputs/. No calibration,
GUI state, accepted report, nominal radius or physical count is modified.
"""
from pathlib import Path
import hashlib
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.image_regions import propose_regions
from pipe_twin.section_diagnostics import axis_from_image_lines, circle_trial, section_basis
from pipe_twin.stereo_analyzer import _calibration_from_manifest, _compute_stereo_depth
from pipe_twin.temporal_stereo import _fuse_disparities, REVISION

OUT = ROOT / 'outputs/section-check-20261021'
SOURCE = ROOT / 'outputs/retry-20261020-131151/active-retry/rectified'
CACHE = ROOT / 'outputs/static-algorithm-20261020/cache'
CAL = ROOT / 'outputs/measurement_workbench/calibration_current.json'
STATIONS = [400, 900, 1260, 1620]
WIDTHS = [80, 160]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def silhouette_axis(regions, k):
    records = []
    for role, proposals in regions.items():
        for index, region in enumerate(proposals):
            x, y, w, h = region['region_px']
            if w < 800 or region['appearance'] == 'neutral':
                continue
            for edge in ('top', 'bottom'):
                points = []
                for col in range(max(70, int(.12*w)), min(w-70, int(.88*w))):
                    rows = np.flatnonzero(region['mask'][:, col])
                    if len(rows):
                        points.append((x+col, y+(rows[0] if edge == 'top' else rows[-1])))
                points = np.asarray(points, float)
                matrix = np.column_stack((points[:, 0], np.ones(len(points))))
                keep = np.ones(len(points), bool)
                for _ in range(4):
                    slope, intercept = np.linalg.lstsq(matrix[keep], points[keep, 1], rcond=None)[0]
                    residual = points[:, 1]-matrix@[slope, intercept]
                    keep = np.abs(residual) < max(2., 3*np.median(np.abs(residual)))
                records.append({'role': role, 'candidate': index, 'edge': edge,
                                'line': [float(slope), -1., float(intercept)],
                                'residual_rms_px': float(np.sqrt(np.mean(residual[keep]**2)))})
    lines = np.array([r['line'] for r in records])
    result = axis_from_image_lines(lines, k)
    axis = np.array(result['axis'])
    leave_out = [axis_from_image_lines(np.delete(lines, i, axis=0), k)['axis'] for i in range(len(lines))]
    by_role = {role: axis_from_image_lines([r['line'] for r in records if r['role'] == role], k)
               for role in regions}
    result.update(lines=records, scope='CONDITIONAL_ON_PARALLEL_SILHOUETTES_AND_PROVISIONAL_CALIBRATION',
                  leave_one_line_out_max_degrees=float(np.degrees(np.arccos(np.clip(
                      np.min(np.abs(np.array(leave_out)@axis)), 0, 1)))), by_role=by_role)
    return result


def fused_maps(samples, limit):
    maps, audit = {}, {}
    for role in samples:
        print('fusing', role, flush=True)
        maps[role], audit[role] = _fuse_disparities(samples[role])
    yy, xx = np.indices(maps['left'].shape)
    valid = {}
    for role, other, sign in [('left', 'right', -1), ('right', 'left', 1)]:
        disparity = maps[role]
        target = np.rint(xx+sign*np.nan_to_num(disparity)).astype(int)
        mask = np.isfinite(disparity) & (target >= 0) & (target < disparity.shape[1])
        opposite = np.full(disparity.shape, np.nan, np.float32)
        opposite[mask] = maps[other][yy[mask], target[mask]]
        valid[role] = mask & np.isfinite(opposite) & (np.abs(disparity-opposite) <= limit)
    for role in maps:
        maps[role][~valid[role]] = np.nan
        audit[role]['post_consistency_valid_fraction'] = float(np.mean(valid[role]))
    return maps, audit


def candidate_points(disparity, region, k, baseline, role):
    x, y, w, h = region['region_px']
    # Exclude the same three-pixel boundary neighbourhood from every method.
    mask = cv2.erode(region['mask'].astype(np.uint8), np.ones((7, 7), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    local = disparity[y:y+h, x:x+w]
    rows, cols = np.nonzero(mask & np.isfinite(local) & (local > 0))
    d = local[rows, cols].astype(float)
    u, v = cols+x, rows+y
    z = k[0, 0]*baseline/d
    xyz = np.column_stack(((u-k[0, 2])*z/k[0, 0], (v-k[1, 2])*z/k[1, 1], z))
    if role == 'right':
        xyz[:, 0] += baseline
    # Shared left-image slab. This is a neighbourhood, not a known physical
    # station, and correspondence does not itself establish pipe identity.
    left_u = u if role == 'left' else u+d
    fraction = np.zeros(len(rows))
    for col in np.unique(cols):
        r = np.flatnonzero(region['mask'][:, col])
        selected = cols == col
        fraction[selected] = (rows[selected]-r.min())/max(1, r.max()-r.min())
    return xyz, np.column_stack((u, v)), left_u, fraction, mask


def make_plots(records, arrays, images, regions):
    sys.path.insert(0, str(OUT / 'plot-deps'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False})
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), layout='constrained')
    for ax, mode, label in zip(axes, ['frame19', 'fused'], ['末帧', '20帧多数一致性筛选']):
        ax.imshow(cv2.cvtColor(images['left'], cv2.COLOR_BGR2RGB))
        for index, region in enumerate(regions['left']):
            if region['appearance'] == 'neutral':
                continue
            x, y, w, h = region['region_px']
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, color='yellow', lw=1))
            ax.text(x+4, y-5, f'C{index} {region["appearance"]}', color='yellow', fontsize=9)
        for station in STATIONS:
            ax.axvspan(station-80, station+80, color='cyan', alpha=.12)
            ax.text(station, 40, str(station), color='cyan', ha='center')
        for row in records:
            if row['role'] == 'left' and row['method'] == mode and row['width_px'] == 160:
                uv = arrays[row['key']+'_pixels']
                ax.scatter(uv[:, 0], uv[:, 1], s=.6, c='lime', alpha=.6)
        ax.set_xlim(0, 1920); ax.set_ylim(540, 0)
        ax.set_title(label+'：绿点为原始实测视差支持；候选编号不代表管道身份或数量')
    fig.savefig(OUT/'sampling.png', dpi=150); plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 9), layout='constrained')
    for j, station in enumerate(STATIONS):
        for i, mode in enumerate(['frame19', 'fused']):
            ax = axes[i, j]
            selected = [r for r in records if r['method'] == mode and r['station_left_px'] == station
                        and r['width_px'] == 160 and r['appearance'] == 'hue-105']
            total = 0
            for row in selected:
                points = arrays[row['key']+'_section']
                ax.scatter(points[:, 0], points[:, 1], s=3, alpha=.5,
                           label=f'{row["role"]} C{row["candidate"]}: {len(points)}',
                           c='tab:blue' if row['role'] == 'left' else 'tab:orange')
                total += len(points)
            ax.set_title(f'x={station} ±80 px / {mode}\n{total} 个点（双目非独立真值）')
            ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.3)
            ax.set_xlabel('断面横向 U / mm'); ax.set_ylabel('断面深向 V / mm')
            ax.invert_yaxis()
            if selected:
                ax.legend(fontsize=8)
    fig.suptitle('蓝色候选：未补洞的断面散点，全范围显示；尺度与轴向依赖暂用标定')
    fig.savefig(OUT/'blue-sections.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for width, ax in zip(WIDTHS, axes):
        for station in STATIONS:
            selected = sorted([r for r in records if r['role'] == 'left' and r['appearance'] == 'hue-105'
                               and r['station_left_px'] == station and r['width_px'] == width
                               and r['method'].startswith('frame')], key=lambda r: int(r['method'][5:]))
            ax.plot([int(r['method'][5:]) for r in selected],
                    [r['circle'].get('trial_diameter_mm', np.nan) for r in selected], '.-', label=f'x={station}')
        ax.set_title(f'带宽 {width} px：试拟合直径，不是有效测量')
        ax.set_xlabel('帧序号'); ax.set_ylabel('诊断圆直径 / mm'); ax.grid(alpha=.3); ax.legend()
        ax.set_yscale('log')
    fig.suptitle('逐帧稳定性：每帧独立计算，未把重复像素叠加充当更多空间采样')
    fig.savefig(OUT/'temporal-trials.png', dpi=160); plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    meta = load_json(ROOT/'outputs/static-algorithm-20261020/probe.json')['inputs']
    assert sha(CAL) == meta['calibration_sha256'], 'Active calibration changed'
    cache_meta = {k: v for k, v in meta.items() if k != 'baseline_reducer_commit'}
    key = hashlib.sha256(json.dumps(cache_meta, sort_keys=True).encode()).hexdigest()[:16]
    paths = [CACHE/f'{key}-{i:02d}.npz' for i in range(20)]
    for role, hashes in meta['images'].items():
        for name, expected in hashes.items():
            assert sha(SOURCE/name) == expected, f'Image hash mismatch: {name}'
    images = {r: cv2.imdecode(np.fromfile(SOURCE/f'019_{r}.png', np.uint8), cv2.IMREAD_COLOR)
              for r in ['left', 'right']}
    bundle = load_json(CAL)
    cal = _calibration_from_manifest(bundle['stereo_calibration'])
    k = np.asarray(bundle['rectification_recipe']['P1'])[:, :3]
    assert np.allclose(k, np.asarray(bundle['rectification_recipe']['P2'])[:, :3])
    samples = {r: [] for r in images}
    for path in paths:
        with np.load(path) as data:
            for role in samples:
                samples[role].append(data[role])
    fresh = _compute_stereo_depth(images['left'], images['right'], cal, meta['analysis'])
    checks = {}
    for role in samples:
        expected = cal.left.fx*cal.baseline_mm/getattr(fresh, role+'_depth_mm')
        actual = samples[role][-1]
        assert np.array_equal(np.isfinite(expected), np.isfinite(actual))
        delta = float(np.nanmax(np.abs(actual-expected)))
        assert delta < 1e-4
        checks[role] = {'finite_masks_equal': True, 'maximum_disparity_difference_px': delta}
    print('40 image hashes, calibration and final fresh stereo match verified', flush=True)
    fused, fusion_audit = fused_maps(samples, meta['analysis']['left_right_consistency_px'])
    np.savez_compressed(OUT/'fused-disparities.npz', **fused)
    regions = {r: propose_regions(images[r])[0] for r in images}
    axis = silhouette_axis(regions, k)
    basis = section_basis(axis['axis'])
    records, arrays = [], {}
    for method in [f'frame{i:02d}' for i in range(20)]+['fused']:
        for role in samples:
            disparity = fused[role] if method == 'fused' else samples[role][int(method[5:])]
            for index, region in enumerate(regions[role]):
                if region['appearance'] == 'neutral':
                    continue
                xyz, uv, left_u, transverse, mask = candidate_points(disparity, region, k, cal.baseline_mm, role)
                for station in STATIONS:
                    for width in WIDTHS:
                        keep = (left_u >= station-width/2) & (left_u < station+width/2)
                        # Skip nonintersecting proposals, retaining intersecting
                        # empty samples as explicit failures. Right denominator
                        # is unknown without disparity and is never invented.
                        x, y, w, h = region['region_px']
                        if role == 'left' and (x+w <= station-width/2 or x >= station+width/2):
                            continue
                        if role == 'right':
                            # Include empty right samples whenever this image
                            # region could intersect the slab within the search
                            # range. Do not silently drop failed/empty windows.
                            matching = meta['analysis']['stereo_matching']
                            dmin = matching['min_disparity']
                            dmax = dmin+matching['num_disparities']
                            if x+w+dmax <= station-width/2 or x+dmin >= station+width/2:
                                continue
                        pts, pixels = xyz[keep], uv[keep]
                        section = pts @ basis
                        bins = np.minimum(9, (transverse[keep]*10).astype(int))
                        counts = np.bincount(bins, minlength=10)
                        denominator = None
                        if role == 'left':
                            cols = np.arange(x, x+w)
                            denominator = int(mask[:, (cols >= station-width/2) & (cols < station+width/2)].sum())
                        name = f'{method}_{role}_c{index}_x{station}_w{width}'
                        record = {'key': name, 'method': method, 'role': role, 'candidate': index,
                                  'appearance': region['appearance'], 'station_left_px': station, 'width_px': width,
                                  'point_count': len(pts), 'mask_pixels': denominator,
                                  'valid_fraction': len(pts)/denominator if denominator else None,
                                  'transverse_bins': counts.tolist(),
                                  'transverse_bins_with_three_points': int(np.count_nonzero(counts >= 3)),
                                  'depth_p05_p50_p95_mm': np.percentile(pts[:, 2], [5, 50, 95]).tolist() if len(pts) else None,
                                  'circle': circle_trial(section),
                                  'association': 'UNVERIFIED_MAY_CONTAIN_MULTIPLE_SEGMENTS'}
                        records.append(record)
                        if method in ('frame19', 'fused'):
                            arrays[name+'_section'], arrays[name+'_pixels'], arrays[name+'_xyz'] = section, pixels, pts
        print(method, 'done', flush=True)
    sensitivity = []
    axes_to_try = {r: info['axis'] for r, info in axis['by_role'].items()}
    axes_to_try['zero_camera_z'] = [axis['axis'][0], axis['axis'][1], 0.]
    for label, direction in axes_to_try.items():
        alternate_basis = section_basis(direction)
        fits = [circle_trial(arrays[r['key']+'_xyz']@alternate_basis)
                for r in records if r['method'] in ('frame19', 'fused')]
        sensitivity.append({'axis_source': label, 'axis': direction, 'trials': len(fits),
                            'geometric_only_passes': sum(f['geometry_passed'] for f in fits)})
    evidence = {'scope': 'OFFLINE_DIAGNOSTIC_NOT_ACCEPTED_MEASUREMENT',
                'reported_hardware': {'fov_degrees': 80, 'fov_user_description': '全方位的视场角',
                                      'fov_axis': 'UNSPECIFIED', 'lens_focal_length_mm': 3., 'baseline_mm': 60.},
                'physical_pipe_count': None, 'accepted_diameters': [], 'accepted_pair_distances': [],
                'calibration_sha256': sha(CAL), 'calibration_validated': cal.validated,
                'active_baseline_mm': cal.baseline_mm, 'rectified_intrinsic': k.tolist(),
                'raw_intrinsics': {r: bundle['rectification_recipe'][key]
                                   for r, key in [('left', 'K1'), ('right', 'K2')]},
                'diagnostic_source_hashes': {str(p.relative_to(ROOT)): sha(p) for p in [
                    Path(__file__), ROOT/'pipe_twin/section_diagnostics.py',
                    ROOT/'pipe_twin/temporal_stereo.py', ROOT/'pipe_twin/image_regions.py']},
                'source_image_hashes': meta['images'], 'cached_disparity_hashes': {p.name: sha(p) for p in paths},
                'fresh_final_pair_check': checks, 'fusion_revision': REVISION, 'fusion': fusion_audit,
                'axis': axis, 'basis_columns': basis.tolist(), 'axis_sensitivity': sensitivity,
                'candidates': {r: [{a: b for a, b in item.items() if a != 'mask'} for item in proposals]
                               for r, proposals in regions.items()},
                'windows': records}
    (OUT/'evidence.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    np.savez_compressed(OUT/'section-points.npz', **arrays)
    make_plots(records, arrays, images, regions)
    print('records', len(records), 'geometric-only passes', sum(r['circle']['geometry_passed'] for r in records), flush=True)
    print('output', OUT, flush=True)


if __name__ == '__main__':
    main()
