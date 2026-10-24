"""Replay saved field photographs through the production entry point.

No new camera capture, no calibration edits, no CAD dimensions in fitting.
Images/overlays stay under ignored outputs; publish only numeric evidence.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.elevation_auto import analyze_elevation_auto_manifest
from pipe_twin.elevation_dataset import load_elevation_dataset
from pipe_twin.stereo_analyzer import _calibration_from_manifest
from pipe_twin.pooled_sections import analyze_pooled_sections
from pipe_twin.section_display import section_summary_lines

DEFAULT = ROOT/'outputs/retry-20261020-131151/active-static-v2/burst/elevation-20261020-151239-63c7f244/manifest.json'
OUT = ROOT/'outputs/axial-aggregation-20261022'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(stage):
    return {key: stage.get(key) for key in ('revision', 'policy_sha256', 'policy', 'status', 'axis',
        'boundary_pair_count', 'conditional_section_count', 'occlusion_recovered', 'nominal_dimensions_used_in_fitting',
        'candidates', 'registration', 'position_assessment')} | {'boundaries': {
            role: [{k: item.get(k) for k in ('candidate_id', 'appearance', 'region_px', 'boundary_status',
                'reason_codes', 'highlight_band_count', 'span_attempts', 'aggregation')}
                   for item in items if item['boundary_stable']]
            for role, items in stage.get('views', {}).items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=DEFAULT)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    active = ROOT/'outputs/measurement_workbench/calibration_current.json'
    calibration_before = sha(active)
    started = time.perf_counter()
    report = analyze_elevation_auto_manifest(args.manifest, report_output_path=OUT/'report.json',
        evidence_dir=OUT/'evidence', progress=lambda text: print(text, flush=True))
    elapsed = time.perf_counter()-started
    (OUT/'gui-summary.txt').write_text('\n'.join(section_summary_lines(report, detailed=True)), encoding='utf-8')
    assert sha(active) == calibration_before
    evidence = {'source_manifest': str(args.manifest.relative_to(ROOT)), 'manifest_sha256': sha(args.manifest),
        'new_camera_capture': False, 'active_calibration_sha256': calibration_before, 'active_calibration_unchanged': True,
        'production_replay_seconds': elapsed, 'input_hashes': report['inputs'],
        'legacy_counts': report['counts'], 'position_counts': report['position_assessment']['counts'],
        'section_evaluation': compact(report['section_evaluation']), 'report_sha256': sha(OUT/'report.json'),
        'snapshot_checks': []}
    cal = _calibration_from_manifest(load_elevation_dataset(args.manifest)['calibration'])
    folder = ROOT/'outputs/retry-20261020-131151/active-retry/rectified'
    for number in (0, 9):
        paths = [folder/f'{number:03d}_{role}.png' for role in ('left', 'right')]
        images = [cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR) for p in paths]
        stage = analyze_pooled_sections(*images, cal)
        evidence['snapshot_checks'].append({'frame': number, 'hashes': [sha(p) for p in paths], 'section': compact(stage)})
        print('boundary replay', number, stage.get('boundary_pair_count'), stage.get('conditional_section_count'), flush=True)
    evidence['changed_source_sha256'] = {p.name: sha(p) for p in [ROOT/'pipe_twin'/name for name in (
        'axial_aggregation.py', 'pooled_sections.py', 'section_position.py', 'section_display.py', 'elevation_auto.py', 'elevation_gui.py')]}
    (Path(__file__).parent/'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('complete', evidence['legacy_counts'], evidence['position_counts'], round(elapsed, 2), flush=True)


if __name__ == '__main__':
    main()
