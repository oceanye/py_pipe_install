"""Replay the last ground ROI without changing its provenance or source data."""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.capture_gui import _CAPTURE_PROVENANCE_KEYS
from pipe_twin.elevation_dataset import load_elevation_dataset, create_elevation_dataset
from pipe_twin.elevation_auto import analyze_elevation_auto_manifest
from pipe_twin.ground_check import ground_summary_lines

SOURCE = ROOT/'outputs/axial-algorithm-audit-20261022/captures/elevation-20261022-134055-f2d29ec1/manifest.json'
OUTPUT = ROOT/'outputs/ground-correction-20261022'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    loaded = load_elevation_dataset(SOURCE); manifest = loaded['manifest']
    views = manifest['capture']['capture_groups'][-1]['views']
    paths = [SOURCE, SOURCE.parent/'report.json', loaded['left_path'], loaded['right_path'],
             ROOT/'outputs/measurement_workbench/calibration_current.json']
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    package = create_elevation_dataset(output_root=OUTPUT/'captures', calibration=loaded['calibration'],
        left_path=loaded['left_path'], right_path=loaded['right_path'], left_time=loaded['left_time'], right_time=loaded['right_time'],
        pipe_specs=loaded['pipe_specs'], pair_confirmed=True, model_path=SOURCE.parent/manifest['model']['path'],
        stl_unit=manifest['model'].get('source_unit'), rectification_recipe=loaded['rectification_recipe'],
        camera_capture_provenance={r: {k: v[k] for k in _CAPTURE_PROVENANCE_KEYS if k in v} for r, v in views.items()},
        timestamp_sources={r: v.get('timestamp_source', 'MANIFEST_OPERATOR_CONFIRMED') for r, v in views.items()},
        analysis_settings={k: v for k, v in manifest['analysis'].items() if k not in ('mode', 'elevation_auto')},
        registration_settings=loaded['registration_settings'], scope_settings=loaded['scope_settings'])
    started = time.perf_counter()
    report = analyze_elevation_auto_manifest(package, report_output_path=package.parent/'report.json',
        evidence_dir=package.parent/'evidence', progress=lambda msg: print(msg, flush=True))
    elapsed = time.perf_counter()-started
    old = json.loads((SOURCE.parent/'report.json').read_text(encoding='utf-8'))
    stage = report['section_evaluation']
    assert report['ground_check']['correction_status'] == 'NOT_APPLIED'
    assert not report['ground_check']['pipe_axis_constrained_by_ground']
    assert stage['ground_correction']['status'] == 'GROUND_CHECK_FAILED'
    assert stage['axis'] == old['section_evaluation']['axis']
    assert [c['fit'] for c in stage['candidates']] == [c['fit'] for c in old['section_evaluation']['candidates']]
    assert {str(p.relative_to(ROOT)): sha(p) for p in paths} == before
    evidence = {'source_manifest': str(SOURCE.relative_to(ROOT)), 'new_manifest': str(package.relative_to(ROOT)),
        'new_camera_capture': False, 'originals_unchanged': True, 'original_sha256': before,
        'production_seconds': elapsed, 'ground_check': report['ground_check'],
        'legacy_counts': report['counts'], 'position_counts': report['position_assessment']['counts'],
        'axis_and_all_fits_unchanged_when_ground_rejected': True, 'section_revision': stage['section_revision'],
        'candidates': [{'candidate_id': c['candidate_id'], 'diameter_mm': c['fit'].get('trial_diameter_mm'),
                        'window_sensitivity': c['window_sensitivity']} for c in stage['candidates']]}
    (Path(__file__).parent/'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (OUTPUT/'gui-ground-summary.txt').write_text('\n'.join(ground_summary_lines(report['ground_check'])), encoding='utf-8')
    print(json.dumps(evidence, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
