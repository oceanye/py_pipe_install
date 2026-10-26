import copy
from types import SimpleNamespace

import numpy as np
import pytest

from pipe_twin.ground_correction import correct_pipe_axis, finish_ground_correction
from pipe_twin.ground_check import ground_summary_lines
from pipe_twin.section_diagnostics import axis_from_image_lines
from pipe_twin.pooled_sections import analyze_pooled_sections
from test_axial_consistency import projected_samples
from test_axial_aggregation import synthetic_views
from pipe_twin.axial_aggregation import aggregate_boundaries


def ground_for(normal):
    normal = np.asarray(normal, float); normal /= np.linalg.norm(normal)
    return {'status': 'CONSISTENT_GROUND_PLANE', 'surface_identity_source': 'OPERATOR_MARKED_GROUND',
            'left_plane': {'normal_camera': normal.tolist()}, 'calibration_validated': False}


def free_axis(observations, k):
    lines = [[*np.polyfit(o['points'][:, 0], o['points'][:, 1], 1)] for o in observations]
    result = axis_from_image_lines([[m, -1, b] for m, b in lines], k)
    result['status'] = 'STABLE'
    return result


@pytest.mark.parametrize('normal', [(0, -1, 0), (.2, -.7, -.4), (0, 0, -1)])
def test_recovers_axis_in_tilted_ground_without_forcing_equal_depth(normal):
    normal = np.array(normal, float); normal /= np.linalg.norm(normal)
    direction = np.cross(normal, [0., .2, 1.]); direction /= np.linalg.norm(direction)
    obs, _, k, _, truth = projected_samples(direction)
    corrected, audit = correct_pipe_axis(obs, k, free_axis(obs, k), ground_for(normal))
    assert audit['applied']
    assert abs(np.dot(corrected['axis'], normal)) < 1e-10
    assert abs(np.dot(corrected['axis'], truth)) > .99999999
    assert max(r['p95_px'] for r in audit['boundary_residuals']) < .01
    if normal[0] != 0:
        assert abs(corrected['axis'][2]) > .01


def test_ground_constraint_changes_the_solver_and_preserves_free_radius(monkeypatch):
    import pipe_twin.pooled_sections as module
    views, k = synthetic_views()
    monkeypatch.setattr(module, 'aggregate_image', lambda image, role: (copy.deepcopy(views[role]), {'truncated': False}))
    # Simulate the direction degeneracy of the free solver. The ground solver
    # must refit the real observed lines, not project this erroneous prior.
    monkeypatch.setattr(module, 'axis_from_image_lines', lambda *_: {
        'axis': [1., 0., 0.], 'singular_values': [2., .03, .025]})
    cal = SimpleNamespace(left=SimpleNamespace(rectified_intrinsic=k), right=SimpleNamespace(rectified_intrinsic=k), baseline_mm=60.)
    original = analyze_pooled_sections(None, None, cal)
    assert original['model'] is None
    corrected = analyze_pooled_sections(None, None, cal, ground_result=ground_for([.35, 1., 0.]))
    assert corrected['ground_correction']['applied']
    assert corrected['unconstrained_axis']['status'] == 'COMMON_AXIS_UNSTABLE'
    assert corrected['axis']['source'] == 'GROUND_PLANE_CONSTRAINED'
    assert corrected['candidates'][0]['fit']['trial_diameter_mm'] == pytest.approx(50., abs=.02)
    np.testing.assert_allclose(corrected['model']['section']['normal_camera'], corrected['axis']['axis'])
    assert corrected['candidates'][0]['pixel_error']['ground_normal_uncertainty_included'] is False


@pytest.mark.parametrize('failure', ['missing', 'failed', 'unconfirmed', 'conflict', 'degenerate', 'one_eye'])
def test_unsupported_correction_preserves_the_original_axis(failure):
    obs, _, k, _, axis = projected_samples([1., -.2, .1])
    original = free_axis(obs, k)
    ground = ground_for([.2, 1., 0.])
    if failure == 'missing':
        ground = None
    elif failure == 'failed':
        ground['status'] = 'GROUND_PLANE_INCONSISTENT'
    elif failure == 'unconfirmed':
        ground['surface_identity_source'] = 'ASSISTANT_SUGGESTED_GROUND_NOT_CONFIRMED'
    elif failure == 'conflict':
        ground = ground_for(axis)
    elif failure == 'degenerate':
        line = np.polyfit(obs[0]['points'][:, 0], obs[0]['points'][:, 1], 1)
        ground = ground_for(np.array([line[0], -1, line[1]])@k)
        for o in obs:
            o['points'] = obs[0]['points'].copy()
    else:
        obs = [o for o in obs if o['role'] == 'left']
    corrected, audit = correct_pipe_axis(obs, k, original, ground)
    assert corrected is original and not audit['applied']
    assert audit['status'] != 'APPLIED'


def test_summary_distinguishes_partial_correction_and_constrained_zero_angle():
    g = ground_for([0, -1, 0]) | {'camera_to_ground_perpendicular_mm': 500.,
        'ground_depth_median_mm': 1800., 'optical_axis_to_ground_deg': 0.,
        'image_roll_relative_to_ground_deg': 0., 'left_right_normal_angle_deg': 0.,
        'left_right_plane_offset_difference_mm': 0.}
    before = copy.deepcopy(g)
    result = finish_ground_correction(g, [
        {'zone_id': 'Z01', 'axis': {'status': 'STABLE', 'axis': [1, 0, 0]},
         'ground_correction': {'status': 'APPLIED', 'applied': True, 'axis_change_deg': 2.}},
        {'zone_id': 'Z02', 'ground_correction': {'status': 'GROUND_AXIS_CONFLICT', 'applied': False}}])
    assert g == before
    assert result['correction_status'] == 'PARTIALLY_APPLIED'
    assert result['pipe_axis_constrained_by_ground']
    text = '\n'.join(ground_summary_lines(result))
    assert '已应用地面修正' in text and '未应用地面修正' in text and '不是独立核验' in text


def test_candidate_trial_applies_conflicting_hypothesis_without_promoting_ground():
    from pipe_twin.ground_visualization import tilt_label_lines, checked_ground
    obs, _, k, _, _ = projected_samples([1., -.2, .1])
    original = free_axis(obs, k)
    ground = {'status': 'GROUND_PLANE_INCONSISTENT', 'surface_identity_source': 'ASSISTANT_SUGGESTED_GROUND_NOT_CONFIRMED',
              'left_plane': {'status': 'GROUND_PLANE_INCONSISTENT', 'candidate_plane': {'normal_camera': [.5, 1., 0.]}},
              'candidate_trial_enabled': True}
    before = copy.deepcopy(ground)
    corrected, audit = correct_pipe_axis(obs, k, original, ground)
    assert audit['status'] == 'TRIAL_APPLIED' and audit['applied']
    assert audit['boundary_consistency_bypassed'] and not audit['boundary_check_passed']
    assert abs(np.dot(corrected['axis'], [.5, 1., 0.])) < 1e-10
    assert ground == before
    result = finish_ground_correction(ground, [{'axis': corrected, 'ground_correction': audit}])
    assert result['correction_status'] == 'TRIAL_APPLIED'
    assert result['status'] == 'GROUND_PLANE_INCONSISTENT' and not checked_ground(result)
    text = '\n'.join(tilt_label_lines(result, compact=True))
    assert '试验' in text and '不用于修正' not in text
    strict = dict(ground, candidate_trial_enabled=False)
    unchanged, rejected = correct_pipe_axis(obs, k, original, strict)
    assert unchanged is original and not rejected['applied']


@pytest.mark.parametrize('normal', [[0, 0, 0], [float('nan'), 0, 1], [1, 2]])
def test_trial_does_not_invent_a_missing_or_invalid_plane(normal):
    obs, _, k, _, _ = projected_samples([1., -.2, .1])
    original = free_axis(obs, k)
    ground = {'status': 'GROUND_PLANE_INCONSISTENT', 'candidate_trial_enabled': True,
              'left_plane': {'candidate_plane': {'normal_camera': normal}}}
    corrected, audit = correct_pipe_axis(obs, k, original, ground)
    assert corrected is original and not audit['applied']
    assert audit['status'] == 'GROUND_NORMAL_INVALID'


def test_trial_free_radius_fits_remain_diagnostic_even_with_validated_calibration(monkeypatch):
    from pipe_twin.section_position import assess_pooled_sections
    import pipe_twin.pooled_sections as module
    views, k = synthetic_views()
    monkeypatch.setattr(module, 'aggregate_image', lambda image, role: (copy.deepcopy(views[role]), {'truncated': False}))
    cal = SimpleNamespace(left=SimpleNamespace(rectified_intrinsic=k), right=SimpleNamespace(rectified_intrinsic=k), baseline_mm=60., validated=True)
    g = {'status': 'GROUND_PLANE_INCONSISTENT', 'candidate_trial_enabled': True,
         'left_plane': {'candidate_plane': {'normal_camera': [.35, 1., 0.]}}}
    stage = analyze_pooled_sections(None, None, cal, ground_result=g)
    assert stage['ground_correction']['applied']
    assert stage['candidates'][0]['fit']['trial_diameter_mm'] == pytest.approx(50., abs=.02)
    assert 'GROUND_CORRECTION_EXPERIMENTAL' in stage['candidates'][0]['reason_codes']
    spec = {'pipe_id': 'P1', 'centerline_world_mm': [[0,0,0], [0,0,300]], 'nominal_diameter_mm': 50.}
    assess_pooled_sections(stage, [spec], cal, capture_healthy=True, settings={}, matching={})
    assert stage['position_assessment']['pipes'][0]['state'] == 'UNKNOWN'
    assert 'GROUND_CORRECTION_EXPERIMENTAL' in stage['position_assessment']['pipes'][0]['reason_codes']


@pytest.mark.parametrize('zones', [False, True])
def test_real_ground_plane_flows_before_section_fit_in_full_and_cropped_pipeline(monkeypatch, zones):
    from test_elevation_auto import _groups
    from test_ground_check import scene
    from pipe_twin.elevation_auto import analyze_elevation_auto_groups, analyze_elevation_auto_zones
    import pipe_twin.pooled_sections as module
    calibration, specs, groups = _groups()
    ground_group, _, region, _ = scene([-.35, -1., 0.])
    ground_group['depth'].audit = {'status': 'VALID'}
    groups[0]['depth'] = ground_group['depth']
    views, source_k = synthetic_views()
    target_k = calibration.left.rectified_intrinsic
    # Supply independently projected pipe silhouettes at the actual K. The
    # plane fitting, ground constraint, cylinder refit and zone plumbing are real.
    def observed(image, role, **_kwargs):
        candidates = copy.deepcopy(views[role])
        offset = np.array([60 if role == 'left' else 0, 40]) if zones else np.zeros(2)
        for item in candidates:
            for edge in item['pooled_edges']:
                edge['points'] = (edge['points']-source_k[:2, 2])*(target_k[0, 0]/source_k[0, 0])+target_k[:2, 2]-offset
            item['aggregation'], item['pooled_edges'] = aggregate_boundaries(item['pooled_edges'])
            points = np.vstack([e['points'] for e in item['pooled_edges']])
            low, high = np.floor(points.min(axis=0)), np.ceil(points.max(axis=0))
            item['region_px'] = [*low.astype(int).tolist(), *(high-low+1).astype(int).tolist()]
            item['width_px'] *= target_k[0, 0]/source_k[0, 0]
        return candidates, {'truncated': False}
    monkeypatch.setattr(module, 'aggregate_image', observed)
    kwargs = {'calibration': calibration, 'pipe_specs': specs, 'registration_settings': {'ground_region': region}}
    if zones:
        result = analyze_elevation_auto_zones(groups, **kwargs, scope_settings={'scope': 'zones', 'zones': [
            {'zone_id': 'Z01', 'label': 'pipe', 'enabled': True, 'coordinate_space': 'rectified_left',
             'roi_rect_px': [60, 40, 520, 250]}]})
        stage = result['section_evaluations'][0]
    else:
        result = analyze_elevation_auto_groups(groups, **kwargs)
        stage = result['section_evaluation']
    assert result['ground_check']['status'] == 'CONSISTENT_GROUND_PLANE'
    assert result['ground_check']['correction_status'] == 'APPLIED'
    assert result['ground_check']['pipe_axis_constrained_by_ground']
    assert stage['candidates'][0]['fit']['trial_diameter_mm'] == pytest.approx(50., abs=.02)
    assert stage['axis']['source'] == 'GROUND_PLANE_CONSTRAINED'
