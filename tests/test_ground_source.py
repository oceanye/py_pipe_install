"""Independent reference ground must not repair a rejected burst implicitly."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from pipe_twin.ground_source import reference_ground_source, check_current_ground


def inputs():
    views = {r: {'sha256': r+'-sha', 'captured_at': '2026-10-30T18:32:50+08:00'} for r in ('left', 'right')}
    frames = [{'capture_id': 'REF', 'views': copy.deepcopy(views)}]
    capture = {'sync_valid': True, 'views': copy.deepcopy(views)}
    rejected = SimpleNamespace(audit={'status': 'INVALID', 'temporal_fusion': {
        'status': 'INVALID', 'reference_capture_id': 'REF', 'reason_codes': ['BURST_REFERENCE_OR_SCENE_UNSTABLE']}})
    group = {'left': np.zeros((12, 16, 3), np.uint8), 'right': np.ones((12, 16, 3), np.uint8),
             'depth': rejected, 'pair_healthy': False,
             'photos': {r: {'actual_sha256': views[r]['sha256']} for r in views},
             'quality': {r: {'passed': True} for r in views}}
    return group, capture, frames


def test_independent_ground_preserves_rejected_burst_and_binds_current_photos(monkeypatch):
    import pipe_twin.ground_source as module
    group, capture, frames = inputs()
    original_depth = group['depth']
    reference_depth = SimpleNamespace(audit={'status': 'VALID'})
    seen = []
    def compute(left, right):
        assert left is group['left'] and right is group['right']
        return reference_depth
    source = reference_ground_source(group, capture, frames, compute_depth=compute)
    assert source['depth'] is reference_depth and source['pair_healthy']
    assert source['source_capture_id'] == 'REF'
    assert source['source_photo_sha256'] == {'left': 'left-sha', 'right': 'right-sha'}
    group['reference_ground_pair'] = source
    def inspect(g, calibration, region, **kwargs):
        seen.append(g)
        return {'status': 'CONSISTENT_GROUND_PLANE', 'accepted_measurement': False}
    monkeypatch.setattr(module.ground_check, 'check_ground', inspect)
    result = check_current_ground(group, None, {'configured': True})
    assert seen == [source]
    assert result['source_scope'] == 'REFERENCE_STEREO_PAIR_ONLY'
    assert result['source_burst_status'] == 'INVALID'
    assert result['source_burst_reason_codes'] == ['BURST_REFERENCE_OR_SCENE_UNSTABLE']
    assert not result['accepted_measurement']
    assert group['depth'] is original_depth and not group['pair_healthy']
    assert original_depth.audit['status'] == 'INVALID'


@pytest.mark.parametrize('failure', ['no_burst', 'valid_burst', 'healthy_group', 'sync', 'left_quality', 'right_quality', 'depth'])
def test_no_reference_ground_for_ineligible_or_unhealthy_inputs(failure):
    group, capture, frames = inputs()
    if failure == 'no_burst':
        frames = []
    elif failure == 'valid_burst':
        group['depth'].audit['temporal_fusion']['status'] = 'VALID'
    elif failure == 'healthy_group':
        group['pair_healthy'] = True
    elif failure == 'sync':
        capture['sync_valid'] = False
    elif failure.endswith('_quality'):
        group['quality'][failure.split('_')[0]]['passed'] = False
    calls = []
    def compute(*_):
        calls.append(True)
        return SimpleNamespace(audit={'status': 'INVALID'})
    assert reference_ground_source(group, capture, frames, compute_depth=compute) is None
    assert len(calls) == (1 if failure == 'depth' else 0)


@pytest.mark.parametrize('mismatch', ['capture_id', 'photo_hash', 'view_hash', 'time'])
def test_reference_ground_rejects_wrong_current_frame_before_computing(mismatch):
    group, capture, frames = inputs()
    if mismatch == 'capture_id':
        frames[-1]['capture_id'] = 'OLD'
    elif mismatch == 'photo_hash':
        group['photos']['right']['actual_sha256'] = 'OLD'
    elif mismatch == 'view_hash':
        capture['views']['left']['sha256'] = 'OLD'
    else:
        frames[-1]['views']['left']['captured_at'] = '2026-10-29T18:32:50+08:00'
    with pytest.raises(ValueError, match='GROUND_REFERENCE_'):
        reference_ground_source(group, capture, frames, compute_depth=lambda *_: pytest.fail('Must not compute'))


def test_no_ground_region_uses_regular_check(monkeypatch):
    import pipe_twin.ground_source as module
    group, _, _ = inputs()
    group['reference_ground_pair'] = {'depth': 'unused'}
    monkeypatch.setattr(module.ground_check, 'check_ground', lambda g, *args, **kwargs: {'same_group': g is group})
    assert check_current_ground(group, None, None) == {'same_group': True}


def test_ground_success_cannot_promote_rejected_burst_installation(monkeypatch):
    from pipe_twin.elevation_auto import analyze_elevation_auto_groups
    import pipe_twin.ground_source as module
    from test_elevation_auto import _groups, _scene
    calibration, specs, groups = _groups()
    group = groups[-1]
    group['pair_healthy'] = False
    group['depth'].audit['temporal_fusion'] = {
        'status': 'INVALID', 'requested_frames': 10, 'accepted_frames': 4,
        'reason_codes': ['BURST_REFERENCE_OR_SCENE_UNSTABLE']}
    group['reference_ground_pair'] = dict(group, pair_healthy=True,
        source_scope='REFERENCE_STEREO_PAIR_ONLY', source_burst_status='INVALID')
    _, _, rotation, _ = _scene()
    normal = np.cross(rotation[:, 2], [1., 0., 0.])
    normal /= np.linalg.norm(normal)
    monkeypatch.setattr(module.ground_check, 'check_ground', lambda *args, **kwargs: {
        'status': 'CONSISTENT_GROUND_PLANE', 'surface_identity_source': 'OPERATOR_MARKED_GROUND',
        'left_plane': {'normal_camera': normal.tolist()}, 'accepted_measurement': False,
        'calibration_validated': calibration.validated})
    result = analyze_elevation_auto_groups(groups, calibration=calibration, pipe_specs=specs,
        registration_settings={'ground_region': {'coordinate_space': 'rectified_left',
            'image_size': [calibration.left.width, calibration.left.height], 'rect_px': [0, 0, 100, 100]}})
    assert result['ground_check']['status'] == 'CONSISTENT_GROUND_PLANE'
    assert result['ground_check']['source_scope'] == 'REFERENCE_STEREO_PAIR_ONLY'
    assert result['counts']['INSTALLED'] == result['counts']['NOT_INSTALLED'] == 0
    assert result['counts']['UNKNOWN'] == len(specs)
    assert result['position_assessment']['counts']['IN_POSITION'] == 0
    assert result['section_evaluation']['temporal_sections']['status'] == 'BURST_REJECTED'
    assert not result['capture_audit']['groups'][-1]['pair_healthy']
