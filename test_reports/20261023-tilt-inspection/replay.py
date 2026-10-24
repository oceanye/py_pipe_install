"""Replay saved photographs through standalone tilt and full pipe evaluation."""
from pathlib import Path
import copy
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.capture_gui import _CAPTURE_PROVENANCE_KEYS
from pipe_twin.elevation_dataset import load_elevation_dataset, create_elevation_dataset
from pipe_twin.elevation_auto import analyze_elevation_auto_manifest
from pipe_twin.ground_inspection import inspect_ground_pair
from pipe_twin.ground_visualization import decode_support, tilt_angles

SOURCE = ROOT/'outputs/ground-correction-20261022/captures/elevation-20261022-151148-94f6eacb/manifest.json'
OUTPUT = ROOT/'outputs/tilt-inspection-20261023'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    loaded = load_elevation_dataset(SOURCE); manifest = loaded['manifest']
    views = manifest['capture']['capture_groups'][-1]['views']
    paths = [SOURCE, SOURCE.parent/'report.json', loaded['left_path'], loaded['right_path'],
             ROOT/'outputs/measurement_workbench/calibration_current.json']
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    settings = {k: v for k, v in manifest['analysis'].items() if k not in ('mode', 'elevation_auto')}
    started = time.perf_counter()
    inspection_path, check = inspect_ground_pair(
        paths={r: loaded[r+'_path'] for r in views}, expected_hashes={r: sha(loaded[r+'_path']) for r in views},
        capture_times={r: loaded[r+'_time'] for r in views}, calibration=loaded['calibration'],
        region=loaded['registration_settings']['ground_region'], analysis_settings=settings,
        output_root=OUTPUT/'ground_inspections', pair_confirmed=True)
    standalone_seconds = time.perf_counter()-started
    package = create_elevation_dataset(output_root=OUTPUT/'captures', calibration=loaded['calibration'],
        left_path=loaded['left_path'], right_path=loaded['right_path'], left_time=loaded['left_time'], right_time=loaded['right_time'],
        pipe_specs=loaded['pipe_specs'], pair_confirmed=True, model_path=SOURCE.parent/manifest['model']['path'],
        stl_unit=manifest['model'].get('source_unit'), rectification_recipe=loaded['rectification_recipe'],
        camera_capture_provenance={r: {k: v[k] for k in _CAPTURE_PROVENANCE_KEYS if k in v} for r, v in views.items()},
        timestamp_sources={r: v.get('timestamp_source', 'MANIFEST_OPERATOR_CONFIRMED') for r, v in views.items()},
        analysis_settings=settings, registration_settings=loaded['registration_settings'], scope_settings=loaded['scope_settings'])
    started = time.perf_counter()
    report = analyze_elevation_auto_manifest(package, report_output_path=package.parent/'report.json',
        evidence_dir=package.parent/'evidence', progress=lambda msg: print(msg, flush=True))
    elapsed = time.perf_counter()-started
    old = json.loads((SOURCE.parent/'report.json').read_text(encoding='utf-8'))
    stage = report['section_evaluation']
    assert report['ground_check']['correction_status'] == 'NOT_APPLIED'
    assert not report['ground_check']['pipe_axis_constrained_by_ground']
    assert stage['axis'] == old['section_evaluation']['axis']
    assert [c['fit'] for c in stage['candidates']] == [c['fit'] for c in old['section_evaluation']['candidates']]
    assert {str(p.relative_to(ROOT)): sha(p) for p in paths} == before
    assert check['status'] == report['ground_check']['status']
    assert check['observed_support'] == report['ground_check']['observed_support']
    rect, mask = decode_support(check, (1920, 1080))
    assert mask.sum() == check['observed_support']['pixel_count'] < mask.size
    compact = copy.deepcopy(check)
    compact['observed_support'].pop('data')
    evidence = {'source_manifest': str(SOURCE.relative_to(ROOT)), 'new_manifest': str(package.relative_to(ROOT)),
        'inspection_path': str(inspection_path.relative_to(ROOT)), 'new_camera_capture': False,
        'originals_unchanged': True, 'original_sha256': before, 'standalone_seconds': standalone_seconds,
        'full_evaluation_seconds': elapsed, 'ground_check_without_mask': compact,
        'candidate_angles_deg': tilt_angles(check), 'mask_roi_pixels': int(mask.size),
        'mask_support_fraction_of_roi': float(mask.mean()), 'legacy_counts': report['counts'],
        'position_counts': report['position_assessment']['counts'],
        'axis_and_all_fits_unchanged_when_ground_rejected': True,
        'standalone_and_full_support_identical': True, 'ground_correction_status': report['ground_check']['correction_status']}
    (Path(__file__).parent/'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(evidence, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
