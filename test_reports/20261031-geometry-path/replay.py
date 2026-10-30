"""Same-image production replay; never update the user's current scene or calibration."""
from pathlib import Path
import hashlib
import json
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
from pipe_twin.stereo_analyzer import analyze_stereo_capture
from pipe_twin.view_geometry import report_view_geometry
from pipe_twin.workflow_status import workflow_summary

SCENE = ROOT/'outputs/elevation_workbench/captures/elevation-20261030-183635-b6b7b875/manifest.json'
BASELINE = ROOT/'outputs/ground-color-boundaries-20261030/report.json'
OUT = ROOT/'outputs/geometry-path-20261031'
base_summary = runpy.run_path(str(ROOT/'test_reports/20261030-ground-color-boundaries/replay.py'))['summarize']


def summarize(report):
    result = base_summary(report)
    for values in result['boundary_views'].values():
        for value in values:
            audit = value.pop('aggregation') or {}
            value['aggregation_summary'] = {'revision': audit.get('revision'),
                'stable_boundary_count': audit.get('stable_boundary_count'),
                'boundaries': [{k: edge.get(k) for k in ('edge', 'status', 'reason_codes',
                    'accepted_windows', 'total_windows', 'retained_columns', 'holdout_p95_px')}
                    for edge in audit.get('boundaries', [])]}
            value.pop('source_extractions', None)
            extraction = value.get('edge_extraction') or {}
            value['edge_extraction'] = {k: extraction.get(k) for k in ('revision', 'policy_sha256', 'source', 'edges')}
    for role in ('left', 'right'):
        for value in result['pairing_audit'].get(role, []):
            neighbors = value.pop('nearby_candidates', [])
            value['nearby_candidate_count'] = len(neighbors)
    result['solver_status'] = [{k: value.get(k) for k in ('label', 'candidate_id', 'solved', 'diameter_mm',
                              'pipe_id', 'reason_codes', 'values')} for value in result['solver_status'] or []]
    return result


def diagnostic_summary(report):
    value = report.get('reference_pair_diagnostics')
    return {k: v for k, v in value.items() if k not in ('audit', 'image_candidates', 'rejected')} if value else None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cv2.setNumThreads(2)
    manifest = json.loads(SCENE.read_text(encoding='utf-8'))
    protected = {SCENE, SCENE.with_name('report.json'), BASELINE,
                 ROOT/'outputs/elevation_workbench/last_scene.json',
                 ROOT/'outputs/measurement_workbench/calibration_current.json',
                 ROOT/'outputs/measurement_workbench/workbench_profile.json'}
    for group in manifest['capture']['capture_groups']:
        for record in [group, *group.get('temporal_frames', [])]:
            for view in record['views'].values():
                path = SCENE.parent/view['path']
                assert sha(path) == view['sha256']
                protected.add(path)
    before = {str(p.relative_to(ROOT)): sha(p) for p in sorted(protected)}
    old = json.loads(BASELINE.read_text(encoding='utf-8'))
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    report = analyze_stereo_capture(SCENE, report_output_path=OUT/'report.json',
        evidence_dir=OUT/'evidence', progress=lambda value: print(value, flush=True))
    assert before == {str(p.relative_to(ROOT)): sha(p) for p in sorted(protected)}
    result = {'source_scene': str(SCENE.relative_to(ROOT)), 'source_sha256': before,
              'source_unchanged': True, 'seconds': time.monotonic()-start,
              'baseline_report': str(BASELINE.relative_to(ROOT)),
              'before': summarize(old), 'after': summarize(report),
              'view_geometry': report_view_geometry(report),
              'reference_pair_diagnostics': diagnostic_summary(report),
              'workflow_summary': workflow_summary(report, manifest['analysis']['elevation_auto']['registration']),
              'full_report': str((OUT/'report.json').relative_to(ROOT)),
              'scope': 'SAME_IMAGES_ALGORITHM_REPLAY_NOT_INDEPENDENT_FIELD_ACCURACY_VALIDATION'}
    destination = Path(__file__).with_name('evidence.json')
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('SAVED', destination, flush=True)
    print(result['workflow_summary'], flush=True)


if __name__ == '__main__':
    main()
