"""Read-only same-photo audit of the newly retained blue lower boundary."""
from pathlib import Path
import hashlib
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.axial_selection import band_sampling_masks
from pipe_twin.elevation_dataset import load_elevation_dataset
from pipe_twin.pooled_sections import aggregate_image
from pipe_twin.stereo_analyzer import _calibration_from_manifest


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stage(report):
    return (report.get('section_evaluations') or [report['section_evaluation']])[0]


def residuals(points, k, axis):
    ray = np.linalg.solve(k, np.r_[points.mean(axis=0), 1.])
    line = np.linalg.solve(k.T, np.cross(ray, axis))
    value = (np.c_[points, np.ones(len(points))]@line)/np.linalg.norm(line[:2])
    return {'p95_px': float(np.percentile(abs(value), 95)),
            'maximum_px': float(np.max(abs(value))),
            'above_2px': int(np.count_nonzero(abs(value) > 2.)),
            'signed_range_px': [float(value.min()), float(value.max())]}


def main():
    source = ROOT/'outputs/elevation_workbench/captures/elevation-20261030-183635-b6b7b875/manifest.json'
    old_path = ROOT/'outputs/ground-color-boundaries-20261030/report.json'
    new_path = ROOT/'outputs/geometry-path-20261031/report.json'
    photos = {r: source.parent/'photos'/f'{r}.png' for r in ('left', 'right')}
    protected = [source, source.parent/'report.json', old_path, new_path, *photos.values(),
        ROOT/'outputs/measurement_workbench/calibration_current.json',
        ROOT/'outputs/elevation_workbench/last_scene.json']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    old, new = [stage(json.loads(path.read_text(encoding='utf-8'))) for path in (old_path, new_path)]
    loaded = load_elevation_dataset(source)
    calibration = _calibration_from_manifest(loaded['calibration'])
    k = np.asarray(calibration.left.rectified_intrinsic)
    images = {r: cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR) for r, p in photos.items()}
    selection = loaded['registration_settings']['aggregation_band']
    masks, band = band_sampling_masks(selection, images['left'].shape, images['right'].shape,
                                     disparity_range=(0., 255.))
    items, _ = aggregate_image(images['left'], 'left', sampling_mask=masks['left'], axis_hint_image=band['axis_image'])
    current = next(i for i in items if i['candidate_id'] == 'POOL-L21')
    previous = next(i for i in old['views']['left'] if i['candidate_id'] == 'POOL-L21')
    edge = next(e for e in current['pooled_edges'] if e['edge'] == 'bottom')
    observed = next(e for e in current['aggregation']['boundaries'] if e['edge'] == 'bottom')
    old_edge = next(e for e in previous['aggregation']['boundaries'] if e['edge'] == 'bottom')
    points = edge['points']
    axis_image = np.asarray(edge['image_axis'])
    normal = np.array([-axis_image[1], axis_image[0]])
    u, v = np.rint(points).astype(int).T
    inner = np.rint(points-.75*normal).astype(int)
    hsv = cv2.cvtColor(images['left'], cv2.COLOR_BGR2HSV)
    colours = hsv[inner[:, 1], inner[:, 0]]
    hue_error = abs(colours[:, 0].astype(int)-105)
    painted = (np.minimum(hue_error, 180-hue_error) <= 15) & (colours[:, 1] >= 120) & (colours[:, 2] >= 40)
    old_axis, new_axis = [np.asarray(s['axis']['axis']) for s in (old, new)]
    measured = residuals(points, k, new_axis)
    recorded = next(row for row in new['ground_correction']['boundary_residuals'][12:]
                    if row['role'] == 'left' and row['edge'] == 'bottom')
    assert abs(measured['p95_px']-recorded['p95_px']) < 1e-8
    result = {
        'scope': 'SAME_IMAGE_SAMPLING_AUDIT_NOT_FIELD_ACCURACY_OR_NEW_ACCEPTANCE',
        'source': str(source.relative_to(ROOT)), 'new_capture': False,
        'candidate_id': current['candidate_id'], 'role': 'left', 'edge': 'bottom',
        'previous_boundary': {'status': old_edge['status'], 'holdout_p95_px': old_edge['holdout_p95_px'],
            'participated_in_ground_check': False, 'source_columns': old_edge['source_columns']},
        'new_boundary': {'status': observed['status'], 'holdout_p95_px': observed['holdout_p95_px'],
            'adaptive_requested_span_px': 384., 'observed_axial_interval_px': observed['observed_axial_interval_px'],
            'full_span_status': next(e for e in current['full_span_audit']['boundaries'] if e['edge'] == 'bottom')['status'],
            'source_axial_samples': observed['source_columns'], 'retained_samples': len(points),
            'actual_original_pixel_bounds': [points.min(axis=0).tolist(), points.max(axis=0).tolist()],
            'all_retained_samples_inside_original_user_band': bool(masks['left'][v, u].all()),
            'all_retained_inward_samples_supported_by_original_blue_pixels': bool(painted.all()),
            'retained_axial_runs_px': observed['retained_axial_runs_px']},
        'same_new_points_with_previous_axis': residuals(points, k, old_axis),
        'same_new_points_with_new_axis': measured,
        'axis_change_between_versions_deg': float(np.degrees(np.arccos(np.clip(old_axis@new_axis, -1, 1)))),
        'new_other_15_boundaries_max_p95_px': max(row['p95_px'] for index, row in
            enumerate(new['ground_correction']['boundary_residuals']) if index != 13),
        'ground_boundary_limit_px': 2., 'thresholds_changed': False,
        'interpretation': 'The additional blue-left-bottom evidence passes its local boundary holdout but conflicts with the independently ground-constrained common axis. Keep this rejection; do not discard the new evidence to recover the previous green check.',
        'source_sha256': hashes,
    }
    result['sources_unchanged'] = all(digest(p) == hashes[str(p.relative_to(ROOT))] for p in protected)
    assert result['sources_unchanged']
    target = Path(__file__).with_name('sampling-audit.json')
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k not in ('source_sha256', 'new_boundary')}, indent=2))


if __name__ == '__main__':
    main()
