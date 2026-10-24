"""Read-only five-pair field audit of ground support, ROI and angle stability."""
from pathlib import Path
import hashlib
import json
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.capture_gui import load_calibration_json
from pipe_twin.ground_check import check_ground
from pipe_twin.ground_visualization import decode_support, ground_evidence_image, tilt_angles
from pipe_twin.stereo_analyzer import _analysis_config, _calibration_from_manifest, _compute_stereo_depth, _quality

OUTPUT = ROOT/'outputs/tilt-effect-check-20261024'
CAPTURE = OUTPUT/'raw-20261024-083322/capture.json'
CALIBRATION = ROOT/'outputs/measurement_workbench/calibration_current.json'
REFERENCE = ROOT/'outputs/tilt-inspection-20261023/captures/elevation-20261023-084738-cbfc8998/manifest.json'
REGIONS = {'previous': [1050, 730, 420, 240], 'lower_clear': [1050, 820, 420, 180],
           'rear_large': [700, 40, 850, 200], 'left_clear': [80, 80, 350, 300]}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)


def main():
    raw = json.loads(CAPTURE.read_text(encoding='utf-8'))
    reference = json.loads(REFERENCE.read_text(encoding='utf-8'))
    calibration = load_calibration_json(CALIBRATION)
    parsed = _calibration_from_manifest(calibration)
    settings = _analysis_config(reference['analysis'])
    originals = {str(p.relative_to(ROOT)): sha(p) for p in (CAPTURE, CALIBRATION, REFERENCE)}
    rows, files = [], []
    start = time.perf_counter()
    for index, pair in enumerate(raw['captures'], 1):
        images = {role: read(OUTPUT/'rectified'/pair[role]['path']) for role in ('left', 'right')}
        quality = {role: _quality(image, settings) for role, image in images.items()}
        depth = _compute_stereo_depth(images['left'], images['right'], parsed, settings)
        healthy = bool(depth.audit['status'] == 'VALID' and pair['sync_delta_ms'] <= parsed.max_sync_delta_ms
                       and all(q['passed'] for q in quality.values()))
        group = {'depth': depth, 'pair_healthy': healthy}
        frame = {'index': index, 'capture_id': pair['capture_id'], 'captured_at': pair['left']['captured_at'],
                 'pair_healthy': healthy, 'quality': quality, 'depth_audit': depth.audit,
                 'image_health': pair['image_health'], 'regions': {}}
        for role in ('left', 'right'):
            path = CAPTURE.parent/pair[role]['path']
            assert sha(path) == pair[role]['sha256']
            files.append({'raw_path': str(path.relative_to(ROOT)), 'raw_sha256': pair[role]['sha256'],
                          'rectified_path': str((OUTPUT/'rectified'/pair[role]['path']).relative_to(ROOT)),
                          'rectified_sha256': sha(OUTPUT/'rectified'/pair[role]['path'])})
        for name, rect in REGIONS.items():
            region = {'image_size': [1920, 1080], 'rect_px': rect, 'selection_source': 'ASSISTANT_DIAGNOSTIC'}
            check = check_ground(group, parsed, region)
            directory = OUTPUT/'checks'/f'pair-{index:02d}'/name
            directory.mkdir(parents=True, exist_ok=True)
            (directory/'ground_check.json').write_text(json.dumps(check, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
            support = check.get('observed_support', {})
            metrics = {'rect_px': rect, 'status': check['status'], 'valid_depth_fraction': check.get('valid_depth_fraction'),
                       'sample_inlier_fraction': check.get('left_plane', {}).get('inlier_fraction'),
                       'left_plane': check.get('left_plane'), 'right_plane': check.get('right_plane'),
                       'left_right_normal_angle_deg': check.get('left_right_normal_angle_deg'),
                       'left_right_plane_offset_difference_mm': check.get('left_right_plane_offset_difference_mm'),
                       'candidate_angles_deg': tilt_angles(check), 'support_pixels': support.get('pixel_count'),
                       'valid_pixels': support.get('valid_pixel_count'),
                       'inferred_pixels': support.get('inferred_pixels'), 'pipe_axis_constrained_by_ground': check['pipe_axis_constrained_by_ground']}
            if support:
                _, mask = decode_support(check, (1920, 1080))
                assert int(mask.sum()) == support['pixel_count']
                metrics['support_fraction_of_valid'] = support['pixel_count']/max(1, support['valid_pixel_count'])
            frame['regions'][name] = metrics
            if index == len(raw['captures']):
                ok, data = cv2.imencode('.png', ground_evidence_image(images['left'], check)); assert ok
                (directory/'left_ground_overlay.png').write_bytes(data.tobytes())
        rows.append(frame)
        print(json.dumps({'pair': index, 'healthy': healthy, 'regions': {k: {'status': v['status'],
              'inliers': v['sample_inlier_fraction'], 'angles': v['candidate_angles_deg']} for k, v in frame['regions'].items()}}), flush=True)
    variability = {}
    for name in REGIONS:
        selected = [r['regions'][name] for r in rows]
        angles = np.array([r['candidate_angles_deg'] for r in selected if r['candidate_angles_deg'] is not None], dtype=float)
        variability[name] = {'statuses': [r['status'] for r in selected],
                             'candidate_angle_scope': 'DIAGNOSTIC_NOT_ACCEPTED_CAMERA_POSE'}
        if len(angles):
            values = [column[np.isfinite(column)] for column in angles.T]
            variability[name].update(
                observable_angle_counts=[len(column) for column in values],
                angle_min_deg=[float(np.min(column)) if len(column) else None for column in values],
                angle_max_deg=[float(np.max(column)) if len(column) else None for column in values],
                angle_std_deg=[float(np.std(column)) if len(column) else None for column in values])
    old_left = read(REFERENCE.parent/'photos/left.png')
    gray = lambda im: float(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).mean())
    assert originals == {str(p.relative_to(ROOT)): sha(p) for p in (CAPTURE, CALIBRATION, REFERENCE)}
    result = {'capture_manifest': str(CAPTURE.relative_to(ROOT)), 'new_physical_capture': True,
              'algorithm_commit': '882941447382d3545256ea90113133c38cfd2729',
              'calibration_id': parsed.calibration_id, 'calibration_validated': bool(parsed.validated),
              'baseline_mm': parsed.baseline_mm, 'raw_pair_count': raw['pair_count'], 'raw_usable_pair_count': raw['usable_pair_count'],
              'active_inputs_unchanged': True, 'source_sha256': originals, 'photo_identities': files,
              'rectified_left_gray_mean': {'previous_capture': gray(old_left), 'new_last_capture': gray(images['left'])},
              'processing_seconds': time.perf_counter()-start, 'regions': REGIONS, 'frames': rows,
              'candidate_variability': variability, 'full_pipe_evaluation_run': False,
              'all_ground_checks_rejected': all(v['status'] != 'CONSISTENT_GROUND_PLANE' for r in rows for v in r['regions'].values())}
    (Path(__file__).parent/'evidence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'processing_seconds': result['processing_seconds'], 'brightness': result['rectified_left_gray_mean'], 'variability': variability}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
