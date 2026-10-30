from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from pipe_twin.elevation_auto import analyze_elevation_auto_groups, reference_pair_diagnostics, analyze_elevation_auto_manifest
from pipe_twin.observation_evidence import observation_evidence, metric_cylinder
from pipe_twin.ground_source import reference_ground_source
from test_elevation_auto import _groups, _package


def test_heuristic_strip_cannot_claim_circle_center_by_removing_parallel_flag():
    strip = {'diameter_source': 'STEREO_LOCAL_RADIAL_P95_AND_PROJECTED_CHORD'}
    evidence = observation_evidence(strip)
    assert evidence['grade'] == 'SURFACE_STRIP_CANDIDATE'
    assert not evidence['axis_center_resolved'] and not evidence['free_diameter_resolved']
    assert not metric_cylinder({'diameter_mm': 40., 'center_camera_mm': [0, 0, 1000]})


def test_reference_diagnostic_keeps_all_solved_section_labels_and_diameters():
    from pipe_twin.section_status import solver_status_rows
    candidates = [{'candidate_id': f'pipe{i}', 'fit': {'status': 'CONDITIONAL_FIT', 'trial_diameter_mm': 25.+i},
                   'window_sensitivity': {'stable': True}} for i in range(3)]
    report = {'section_evaluation': {'candidates': candidates, 'model': {'candidates': [
        {'id': f'pipe{i}', 'estimated_center_uv_mm': [i*60., 0.]} for i in range(3)]}},
        'reference_pair_diagnostics': {'valid_depth_fraction': {'left': .48, 'right': .47},
            'metric_cylinder_count': 0, 'rejected_reason_counts': {'CYLINDER_FIT_RESIDUAL_TOO_LARGE': 55}}}
    rows = solver_status_rows(report)
    assert [r['label'] for r in rows] == ['S1', 'S2', 'S3']
    assert all(r['solved'] for r in rows)
    assert [r['diameter_mm'] for r in rows] == [25., 26., 27.]
    assert all('不等于管面圆弧覆盖' in r['detail'] and '圆柱拟合残差过大 55 次' in r['detail'] for r in rows)


def test_provisional_calibration_cannot_establish_installed_or_repeated_absence():
    calibration, specs, groups = _groups(2)
    calibration = replace(calibration, validated=False, relative_elevation_provisional=True)
    report = analyze_elevation_auto_groups(groups, calibration=calibration, pipe_specs=specs)
    assert report['local_surface']['audit']['metric_cylinder_count'] == 3
    assert report['counts'] == {'INSTALLED': 0, 'NOT_INSTALLED': 0, 'UNKNOWN': 4}
    assert all(not row['installation_evidence_qualified'] for row in report['pipes'])
    assert all('CALIBRATION_NOT_VALIDATED' in row['reason_codes'] for row in report['pipes'])


def reference_fixture():
    from test_ground_source import inputs
    group, capture, frames = inputs()
    valid = np.ones(group['left'].shape[:2], bool)
    z = np.full(valid.shape, 1000., np.float32)
    depth = SimpleNamespace(left_depth_mm=z, right_depth_mm=z.copy(), left_valid=valid,
        right_valid=valid.copy(), audit={'status': 'VALID'})
    calls = []
    def compute(*_):
        calls.append(True)
        return depth
    group['reference_ground_pair'] = reference_ground_source(group, capture, frames, compute_depth=compute)
    return group, depth, calls


def test_current_pair_diagnostic_reuses_depth_without_ground_roi_or_state_authority(monkeypatch):
    import pipe_twin.recognition as recognizers
    group, reference_depth, calls = reference_fixture()
    original_depth = group['depth']
    seen = []
    def recognize(_mode, left, right, depth, *_a, **_kw):
        seen.append(depth)
        return {'observations': [], 'rejected': [{'reason': 'LOCAL_DEPTH_INSUFFICIENT'}], 'audit': {}}
    monkeypatch.setattr(recognizers, 'recognize_local_pipes', recognize)
    result = reference_pair_diagnostics(group, SimpleNamespace(validated=True), [], {}, {})
    assert result['status'] == 'DIAGNOSTIC_ONLY' and result['candidate_count'] == 0
    assert result['valid_depth_fraction'] == {'left': 1., 'right': 1.}
    assert result['rejected_reason_counts'] == {'LOCAL_DEPTH_INSUFFICIENT': 1}
    assert not result['installation_evidence'] and not result['accepted_measurement']
    assert result['source_capture_id'] == 'REF' and result['source_burst_status'] == 'INVALID'
    assert seen == [reference_depth] and calls == [True]
    assert group['depth'] is original_depth and not group['pair_healthy']
    assert group['depth'].audit['temporal_fusion']['status'] == 'INVALID'


@pytest.mark.parametrize('failure', ['hash', 'pixels', 'capture_id', 'burst'])
def test_diagnostic_rejects_wrong_reference_binding(failure):
    group, _, _ = reference_fixture()
    reference = group['reference_ground_pair']
    if failure == 'hash':
        reference['source_photo_sha256']['left'] = 'old'
    elif failure == 'pixels':
        reference['left'] = np.ones_like(group['left'])
    elif failure == 'capture_id':
        reference['source_capture_id'] = 'old'
    else:
        reference['source_burst_status'] = 'VALID'
    with pytest.raises(ValueError, match='LOCAL_DIAGNOSTIC_REFERENCE'):
        reference_pair_diagnostics(group, SimpleNamespace(validated=True), [], {}, {})


def test_manifest_creates_reference_diagnostic_even_without_ground_region(tmp_path, monkeypatch):
    import pipe_twin.temporal_stereo as temporal
    import pipe_twin.stereo_analyzer as stereo
    import pipe_twin.recognition as recognizers
    path, _ = _package(tmp_path)
    for role in ('left', 'right'):
        (path.parent/'photos'/f'burst_{role}.png').write_bytes((path.parent/'photos'/f'{role}.png').read_bytes())
    shape = (480, 640)
    bad = SimpleNamespace(left_depth_mm=np.full(shape, np.nan, np.float32),
        right_depth_mm=np.full(shape, np.nan, np.float32), left_valid=np.zeros(shape, bool),
        right_valid=np.zeros(shape, bool), audit={'status': 'INVALID', 'temporal_fusion': {
            'status': 'INVALID', 'reference_capture_id': 'REF', 'requested_frames': 10,
            'reason_codes': ['BURST_REFERENCE_OR_SCENE_UNSTABLE']}})
    good = SimpleNamespace(left_depth_mm=np.full(shape, 1000., np.float32),
        right_depth_mm=np.full(shape, 1000., np.float32), left_valid=np.ones(shape, bool),
        right_valid=np.ones(shape, bool), audit={'status': 'VALID'})
    monkeypatch.setattr(temporal, 'validate_burst_records',
        lambda group, *_a: [{'capture_id': 'REF', 'views': {
            r: dict(v, path=f'photos/burst_{r}.png') for r, v in group['views'].items()}}])
    monkeypatch.setattr(temporal, 'fuse_stereo_burst', lambda *_a, **_kw: bad)
    depths = []
    def compute(*_a, **_kw):
        depths.append(True)
        return good
    monkeypatch.setattr(stereo, '_compute_stereo_depth', compute)
    monkeypatch.setattr(stereo, '_quality', lambda *_a: {'passed': True})
    monkeypatch.setattr(recognizers, 'recognize_local_pipes', lambda *_a, **_kw: {
        'observations': [], 'audit': {}, 'point_cloud': {}})
    result = analyze_elevation_auto_manifest(path)
    assert result['ground_check']['status'] == 'NOT_CONFIGURED'
    assert result['reference_pair_diagnostics']['source_scope'] == 'REFERENCE_STEREO_PAIR_ONLY'
    assert result['reference_pair_diagnostics']['valid_depth_fraction']['left'] == 1.
    assert result['temporal_fusion']['status'] == 'INVALID'
    assert result['counts']['INSTALLED'] == result['counts']['NOT_INSTALLED'] == 0
    assert result['capture_audit']['groups'][-1]['pair_healthy'] is False
    assert np.isnan(bad.left_depth_mm).all() and depths == [True]
