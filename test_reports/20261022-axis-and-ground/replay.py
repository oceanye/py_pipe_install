"""Replay the exact 11:39 field pair without altering its band or source files."""
from pathlib import Path
import hashlib
import json
import math
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.elevation_dataset import load_elevation_dataset, create_elevation_dataset
from pipe_twin.elevation_auto import analyze_elevation_auto_manifest
from pipe_twin.pooled_sections import analyze_pooled_sections
from pipe_twin.stereo_analyzer import _calibration_from_manifest
from pipe_twin.capture_gui import _CAPTURE_PROVENANCE_KEYS
from pipe_twin.ground_check import ground_summary_lines
from pipe_twin.section_display import section_summary_lines

SOURCE = ROOT/'outputs/elevation_workbench/captures/elevation-20261022-113933-359f0666/manifest.json'
OUTPUT = ROOT/'outputs/axial-algorithm-audit-20261022'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def brief(stage):
    return {key: stage.get(key) for key in ('status', 'section_revision', 'axis', 'boundary_pair_count',
        'conditional_section_count', 'candidates', 'position_assessment', 'aggregation_band', 'proposal_audit')} | {
        'views': {role: [{key: item.get(key) for key in ('candidate_id', 'appearance', 'region_px',
            'width_px', 'boundary_status', 'reason_codes')} for item in items]
            for role, items in stage.get('views', {}).items()}}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    loaded = load_elevation_dataset(SOURCE)
    manifest = loaded['manifest']
    views = manifest['capture']['capture_groups'][-1]['views']
    paths = [SOURCE, SOURCE.parent/'report.json', ROOT/'outputs/measurement_workbench/calibration_current.json',
             loaded['left_path'], loaded['right_path']]
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    band = loaded['registration_settings']['aggregation_band']
    ground = {'coordinate_space': 'rectified_left', 'image_size': [1920, 1080],
              'rect_px': [1050, 730, 420, 240], 'selection_source': 'ASSISTANT_DIAGNOSTIC'}
    package = create_elevation_dataset(output_root=OUTPUT/'captures', calibration=loaded['calibration'],
        left_path=loaded['left_path'], right_path=loaded['right_path'], left_time=loaded['left_time'], right_time=loaded['right_time'],
        pipe_specs=loaded['pipe_specs'], pair_confirmed=True, model_path=SOURCE.parent/manifest['model']['path'],
        stl_unit=manifest['model'].get('source_unit'), rectification_recipe=loaded['rectification_recipe'],
        camera_capture_provenance={r: {k: v[k] for k in _CAPTURE_PROVENANCE_KEYS if k in v} for r, v in views.items()},
        timestamp_sources={r: v.get('timestamp_source', 'MANIFEST_OPERATOR_CONFIRMED') for r, v in views.items()},
        analysis_settings={k: v for k, v in manifest['analysis'].items() if k not in ('mode', 'elevation_auto')},
        registration_settings=loaded['registration_settings'] | {'ground_region': ground},
        scope_settings=loaded['scope_settings'])
    started = time.perf_counter()
    report = analyze_elevation_auto_manifest(package, report_output_path=package.parent/'report.json',
        evidence_dir=package.parent/'evidence', progress=lambda msg: print(msg, flush=True))
    elapsed = time.perf_counter()-started
    previous = json.loads((SOURCE.parent/'report.json').read_text(encoding='utf-8'))['section_evaluation']
    left, right = [cv2.imdecode(np.fromfile(loaded[r+'_path'], np.uint8), cv2.IMREAD_COLOR) for r in ('left', 'right')]
    calibration = _calibration_from_manifest(loaded['calibration'])
    endpoints = np.array(band['axis_segment_px'])
    center, vector = endpoints.mean(axis=0), endpoints[1]-endpoints[0]
    variants = []
    # Deliberate mouse errors change the ROI cut, not the metric pipe axis.
    for delta in (-2., 2.):
        angle = math.radians(delta)
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        half = rotation@vector/2
        modified = band | {'axis_segment_px': [(center-half).tolist(), (center+half).tolist()]}
        stage = analyze_pooled_sections(left, right, calibration, aggregation_band=modified,
            disparity_range=report['section_evaluation']['aggregation_band']['disparity_range_px'])
        variants.append({'mouse_angle_change_deg': delta, 'band': modified, 'result': brief(stage)})
    assert {str(p.relative_to(ROOT)): sha(p) for p in paths} == before
    evidence = {'source_manifest': str(SOURCE.relative_to(ROOT)), 'new_manifest': str(package.relative_to(ROOT)),
        'new_camera_capture': False, 'original_user_band_preserved': True, 'band': band,
        'originals_unchanged': True, 'original_sha256': before, 'production_seconds': elapsed,
        'legacy_counts': report['counts'], 'position_counts': report['position_assessment']['counts'],
        'previous': brief(previous), 'current': brief(report['section_evaluation']),
        'ground_check': report['ground_check'], 'mouse_sensitivity': variants}
    (Path(__file__).parent/'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (OUTPUT/'gui-summary.txt').write_text('\n'.join(section_summary_lines(report, detailed=True)+['']+
        ground_summary_lines(report['ground_check'])), encoding='utf-8')
    print(json.dumps({'manifest': str(package), 'seconds': elapsed, 'counts': report['counts'],
        'ground': report['ground_check'], 'mouse_checks': [{
            'delta': v['mouse_angle_change_deg'], 'pairs': v['result']['boundary_pair_count'],
            'diameters': [c['fit'].get('trial_diameter_mm') for c in v['result']['candidates']],
            'center_changes': [c['window_sensitivity'].get('center_difference_mm') for c in v['result']['candidates']]
        } for v in variants]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
