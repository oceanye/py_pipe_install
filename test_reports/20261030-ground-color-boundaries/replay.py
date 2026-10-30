"""Replay the user's fixed 18:32 burst without changing the active field scene."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
from pipe_twin.stereo_analyzer import analyze_stereo_capture
from pipe_twin.workflow_status import workflow_summary

SCENE = ROOT/'outputs/elevation_workbench/captures/elevation-20261030-183635-b6b7b875/manifest.json'
OUT = ROOT/'outputs/ground-color-boundaries-20261030'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(report):
    ground = report.get('ground_check') or {}
    stage = report.get('section_evaluation') or {}
    fusion = report.get('temporal_fusion') or {}
    candidates = []
    for item in stage.get('candidates', []):
        fit = item.get('fit') or {}
        candidates.append({
            'candidate_id': item['candidate_id'], 'reason_codes': item.get('reason_codes'),
            'diameter_mm': fit.get('trial_diameter_mm'), 'fit_status': fit.get('status'),
            'center_camera_mm': fit.get('center_at_reference_column_mm'),
            'baseline_perpendicular_to_axis_mm': fit.get('baseline_perpendicular_to_axis_mm'),
            'window_sensitivity': item.get('window_sensitivity'),
            'dimension_status': item.get('dimension_status'),
            'pixel_error_scenarios': (item.get('pixel_error') or {}).get('scenarios'),
        })
    return {
        'generated_at': report.get('generated_at'), 'counts': report['counts'],
        'position_counts': report.get('position_assessment', {}).get('counts'),
        'camera_side_world': report['registration']['camera_view']['camera_side_world'],
        'ground': {k: v for k, v in ground.items() if k != 'observed_support'},
        'ground_support': {k: v for k, v in ground.get('observed_support', {}).items() if k != 'data'},
        'fusion': {k: fusion.get(k) for k in ('status', 'accepted_frames', 'requested_frames', 'reason_codes')},
        'axis': stage.get('axis'), 'unconstrained_axis': stage.get('unconstrained_axis'),
        'epipolar_refinement': stage.get('epipolar_refinement'),
        'conditional_section_count': stage.get('conditional_section_count'),
        'edge_extraction_revision': stage.get('edge_extraction_revision'),
        'edge_extraction_policy_sha256': stage.get('edge_extraction_policy_sha256'),
        'section_revision': stage.get('section_revision'),
        'candidates': candidates,
        'pairs': (stage.get('model') or {}).get('pairs', []),
        'boundary_views': {role: [{k: v.get(k) for k in (
            'candidate_id', 'appearance', 'width_px', 'boundary_status', 'reason_codes',
            'aggregation', 'edge_extraction', 'source_extractions')} for v in items]
            for role, items in stage.get('views', {}).items()},
        'pairing_audit': stage.get('pairing_audit'),
        'temporal_sections': stage.get('temporal_sections'),
        'section_registration': stage.get('registration'),
        'ground_trial_comparison': stage.get('ground_trial_comparison'),
        'solver_status': report.get('solver_status'),
    }


def main():
    cv2.setNumThreads(2)
    manifest = json.loads(SCENE.read_text(encoding='utf-8'))
    protected = [SCENE, SCENE.with_name('report.json'),
                 ROOT/'outputs/elevation_workbench/last_scene.json',
                 ROOT/'outputs/measurement_workbench/calibration_current.json',
                 ROOT/'outputs/measurement_workbench/workbench_profile.json']
    for group in manifest['capture']['capture_groups']:
        for record in [group, *group.get('temporal_frames', [])]:
            for view in record['views'].values():
                path = SCENE.parent/view['path']
                assert sha(path) == view['sha256']
                protected.append(path)
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    old = json.loads(SCENE.with_name('report.json').read_text(encoding='utf-8'))
    OUT.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    report = analyze_stereo_capture(SCENE, report_output_path=OUT/'report.json',
        evidence_dir=OUT/'evidence', progress=lambda text: print(text, flush=True))
    unchanged = before == {str(p.relative_to(ROOT)): sha(p) for p in protected}
    assert unchanged
    result = {
        'source_scene': str(SCENE.relative_to(ROOT)), 'source_sha256': before,
        'source_unchanged': unchanged, 'seconds': time.monotonic()-start,
        'same_images_before': summarize(old), 'after': summarize(report),
        'workflow_summary': workflow_summary(report, manifest['analysis']['elevation_auto']['registration']),
        'full_report': str((OUT/'report.json').relative_to(ROOT)),
        'scope': 'SAME_IMAGES_ALGORITHM_REPLAY_NOT_INDEPENDENT_FIELD_ACCURACY_VALIDATION',
    }
    path = Path(__file__).with_name('evidence.json')
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('SAVED', path, flush=True)
    print(result['workflow_summary'], flush=True)


if __name__ == '__main__':
    main()
