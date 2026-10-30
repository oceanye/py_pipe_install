"""Full production replay; optionally rectify and evaluate a real remote burst.

All outputs are isolated. Source scene, calibration, profile and default pointer
are protected by hashes. Never updates the user's active scene.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from pipe_twin.calibration_wizard import Rectifier
from pipe_twin.elevation_dataset import create_elevation_dataset, load_elevation_dataset
from pipe_twin.stereo_analyzer import analyze_stereo_capture

SCENE = ROOT / 'outputs/elevation_workbench/captures/elevation-20261030-161444-3ab7011e/manifest.json'
OUT = ROOT / 'outputs/function-review-20261030'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(report):
    ground = report.get('ground_check') or {}
    fusion = report.get('temporal_fusion') or {}
    sections = report.get('section_evaluations') or []
    return {
        'counts': report['counts'], 'registration': report['registration'],
        'position_counts': (report.get('position_assessment') or {}).get('counts'),
        'ground': {k: ground.get(k) for k in ['revision', 'status', 'valid_depth_fraction', 'left_plane',
                  'right_plane', 'correction_status', 'corrections']},
        'temporal_fusion': {k: fusion.get(k) for k in ['status', 'requested_frames', 'accepted_frames',
                  'reason_codes', 'left', 'right']},
        'temporal_frame_rejections': [{k: r.get(k) for k in ['capture_id', 'accepted', 'reason_codes']}
                                    for r in fusion.get('frames', [])],
        'solver_status': [{k: r.get(k) for k in ['label', 'candidate_id', 'solved', 'diameter_mm', 'pipe_id',
                           'reason_codes', 'temporal_consistency']} for r in report.get('solver_status', [])],
        'sections': [{
            'zone_id': s.get('zone_id'), 'status': s.get('status'), 'axis': s.get('axis'),
            'temporal_sections': s.get('temporal_sections'),
            'pairs': [{k: p.get(k) for k in ['ids', 'center_distance_mm', 'surface_clearance_mm',
                        'accepted_measurement']} for p in (s.get('model') or {}).get('pairs', [])],
        } for s in sections],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, help='Completed production remote capture.json')
    parser.add_argument('--variant', choices=['single', 'burst', 'both'], default='both')
    args = parser.parse_args()
    cv2.setNumThreads(2)
    protected = [SCENE, SCENE.with_name('report.json'), SCENE.parent/'photos/left.png',
                 SCENE.parent/'photos/right.png', ROOT/'outputs/elevation_workbench/last_scene.json',
                 ROOT/'outputs/measurement_workbench/calibration_current.json',
                 ROOT/'outputs/measurement_workbench/workbench_profile.json']
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    evidence = {'source_scene': str(SCENE.relative_to(ROOT)), 'source_sha256': before, 'runs': []}
    output = OUT / ('fresh' if args.raw else 'replay')
    output.mkdir(parents=True, exist_ok=True)
    if not args.raw:
        start = time.monotonic()
        report = analyze_stereo_capture(SCENE, report_output_path=output/'report.json',
                    evidence_dir=output/'evidence', progress=lambda s: print(s, flush=True))
        evidence['runs'].append({'variant': 'same_user_photos', 'seconds': time.monotonic()-start,
                                 'report': str((output/'report.json').relative_to(ROOT)), **summary(report)})
    else:
        raw_path = args.raw.resolve()
        capture = json.loads(raw_path.read_text(encoding='utf8'))
        assert capture['status'] == 'COMPLETED', 'Never evaluate a failed burst as complete'
        assert capture['capture_kind'] == 'remote_stereo_raw' and capture['rectified'] is False
        assert capture['pixel_policy'] == 'RAW_CAMERA_BGR_PIXELS_NO_TRANSFORM'
        assert len(capture['captures']) == capture['pair_count'] == capture['requested_pair_count']
        evidence['raw_capture'] = {'path': str(raw_path.relative_to(ROOT)), 'sha256': sha(raw_path),
            'pair_count': capture['pair_count'], 'status': capture['status'],
            'first_frame_at': capture['first_frame_at'], 'finished_at': capture['finished_at'],
            'usable_pair_count': capture['usable_pair_count'], 'requested_exposure_ms': capture['requested_exposure_ms']}
        loaded = load_elevation_dataset(SCENE)
        manifest = loaded['manifest']
        bundle = json.loads((ROOT/'outputs/measurement_workbench/calibration_current.json').read_text(encoding='utf8'))
        calibration, recipe = bundle['stereo_calibration'], bundle['rectification_recipe']
        assert calibration == manifest['stereo_calibration'], 'Different calibration requires range review'
        rectifier = Rectifier(recipe)
        corrected_dir = output/'rectified'
        corrected_dir.mkdir(exist_ok=True)
        records = []
        for item in capture['captures']:
            record = {'timestamp_sources': {r: item['timestamp_source'] for r in ('left', 'right')},
                      'camera_capture_provenance': item['provenance'], 'sha256': {}}
            for role in ('left', 'right'):
                source = raw_path.parent/item[role]['path']
                assert sha(source) == item[role]['sha256']
                image = cv2.imdecode(np.frombuffer(source.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                image = rectifier.rectify(role, image)
                ok, encoded = cv2.imencode('.png', image)
                assert ok
                target = corrected_dir/source.name
                target.write_bytes(encoded.tobytes())
                record[f'{role}_path'] = str(target)
                record[f'{role}_time'] = item[role]['captured_at']
                record['sha256'][role] = sha(target)
            records.append(record)
        last = records[-1]
        analysis = {k: copy.deepcopy(v) for k, v in manifest['analysis'].items() if k not in ['mode', 'elevation_auto']}
        for variant in (['single', 'burst'] if args.variant == 'both' else [args.variant]):
            path = create_elevation_dataset(output_root=output/variant, calibration=calibration,
                left_path=Path(last['left_path']), right_path=Path(last['right_path']),
                left_time=last['left_time'], right_time=last['right_time'],
                pipe_specs=loaded['pipe_specs'], pair_confirmed=True,
                model_path=SCENE.parent/manifest['model']['path'], stl_unit='millimeter',
                rectification_recipe=recipe, camera_capture_provenance=last['camera_capture_provenance'],
                timestamp_sources=last['timestamp_sources'], analysis_settings=analysis,
                registration_settings=loaded['registration_settings'], scope_settings=loaded['scope_settings'],
                temporal_frames=records if variant == 'burst' else None)
            start = time.monotonic()
            report = analyze_stereo_capture(path, report_output_path=path.with_name('report.json'),
                evidence_dir=path.parent/'evidence', progress=lambda s: print(variant, s, flush=True))
            evidence['runs'].append({'variant': variant, 'seconds': time.monotonic()-start,
                                     'manifest': str(path.relative_to(ROOT)), **summary(report)})
    evidence['source_unchanged'] = before == {str(p.relative_to(ROOT)): sha(p) for p in protected}
    assert evidence['source_unchanged']
    target = Path(__file__).with_name('fresh-evidence.json' if args.raw else 'replay-evidence.json')
    target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print('SAVED', target, flush=True)


if __name__ == '__main__':
    main()
