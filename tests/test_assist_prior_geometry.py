import copy
from types import SimpleNamespace

import numpy as np
import pytest

from pipe_twin.axial_aggregation import aggregate_boundaries
from pipe_twin.assist_prior_geometry import prior_positions, _line_samples
from pipe_twin.projection_diagnostics import fit_silhouettes
from test_directional_projection import analytic_samples


def fixture(degrees=60., offsets=None):
    observations, k, axis, _, image_axis, _ = analytic_samples(degrees, count=1000)
    views = {}
    offsets = offsets or {}
    for role in ('left', 'right'):
        local = [{**o, 'points': o['points']-np.array([offsets.get(role+'_x', 0), offsets.get(role+'_y', 0)])}
                 for o in observations if o['role'] == role]
        audit, _ = aggregate_boundaries(local)
        assert audit['stable_boundary_count'] == 2
        views[role] = [{'candidate_id': role, 'image_axis': image_axis.tolist(), 'appearance': 'hue-0',
                        'aggregation': audit}]
    stage = {'axis': {'status': 'STABLE', 'axis': axis.tolist()}, 'views': views, 'zone_offsets': offsets,
             'candidates': [{'candidate_id': 'candidate', 'left_candidate_id': 'left', 'right_candidate_id': 'right',
                 'fit': {'status': 'INVALID_FREE_FIT', 'trial_diameter_mm': 5., 'predicted_lines': {}}}]}
    calibration = SimpleNamespace(left=SimpleNamespace(rectified_intrinsic=k),
                                  right=SimpleNamespace(rectified_intrinsic=k.copy()), baseline_mm=60.)
    return stage, calibration, observations, axis


@pytest.mark.parametrize('degrees', [0, 60, 90, 135])
def test_known_diameter_center_uses_observed_lines_when_free_fit_failed(degrees):
    stage, calibration, _, axis = fixture(degrees)
    before = copy.deepcopy(stage)
    value = prior_positions(stage, calibration, {'RED': 40.})['candidate']
    assert value['status'] == 'PRIOR_CONDITIONAL_CENTER'
    center = np.array([0., 0., 1500.])
    expected = center-axis*(axis@center)
    np.testing.assert_allclose(value['center_camera_mm'], expected, atol=1e-5)
    assert value['prior_diameter_mm'] == 40. and not value['accepted_measurement']
    assert not value['predicted_cylinder_lines_used'] and not value['independent_pixel_samples']
    assert stage == before


def test_crop_offsets_restore_full_camera_coordinates_and_hint_source_is_explicit():
    stage, calibration, _, axis = fixture(90, {'left_x': 204, 'left_y': 53, 'right_x': 170, 'right_y': 29})
    value = prior_positions(stage, calibration, {'RED': {'diameter_mm': 40., 'source': 'MODEL_DIAMETER_PRIOR'}})['candidate']
    assert value['status'] == 'PRIOR_CONDITIONAL_CENTER'
    center = np.array([0., 0., 1500.]); center -= axis*(axis@center)
    np.testing.assert_allclose(value['center_camera_mm'], center, atol=1e-5)
    assert value['diameter_source'] == 'MODEL_DIAMETER_PRIOR'


def test_missing_observation_support_cannot_be_replaced_by_predicted_lines():
    stage, calibration, _, _ = fixture()
    for role in ('left', 'right'):
        for edge in stage['views'][role][0]['aggregation']['boundaries']:
            edge.pop('retained_axial_runs_px')
    value = prior_positions(stage, calibration, {'RED': 40.})['candidate']
    assert value['status'] == 'PRIOR_CENTER_UNAVAILABLE'
    assert 'OBSERVED_BOUNDARY_SUPPORT_UNAVAILABLE' in value['reason_codes']
    assert 'section_center_uv_mm' not in value


def test_occluded_edge_is_not_recreated_and_remaining_edges_can_use_prior():
    stage, calibration, _, _ = fixture()
    stage['views']['left'][0]['aggregation']['boundaries'][0]['boundary_kind'] = 'occlusion'
    result = prior_positions(stage, calibration, {'RED': 40.})['candidate']
    assert result['status'] == 'PRIOR_CONDITIONAL_CENTER'
    assert len(result['boundary_support']) == 3


@pytest.mark.parametrize('failure', ['no_prior', 'bad_prior', 'no_calibration', 'bad_axis', 'epipolar', 'unequal_intrinsics'])
def test_unavailable_evidence_fails_locally_without_inventing_centers(failure):
    stage, calibration, _, _ = fixture()
    hints = {'RED': 40.}
    if failure == 'no_prior':
        hints = {}
    elif failure == 'bad_prior':
        hints = {'RED': float('nan')}
    elif failure == 'no_calibration':
        calibration = None
    elif failure == 'bad_axis':
        stage['axis']['status'] = 'COMMON_AXIS_UNRESOLVED'
    elif failure == 'epipolar':
        stage['epipolar_refinement'] = {'applied': True}
    else:
        calibration.right.rectified_intrinsic[0, 2] += 10
    value = prior_positions(stage, calibration, hints)['candidate']
    assert value['status'] == 'PRIOR_CENTER_UNAVAILABLE' and value['reason_codes']
    assert 'section_center_uv_mm' not in value


def test_trial_ground_axis_prefers_available_unconstrained_image_axis():
    stage, calibration, _, axis = fixture()
    stage['unconstrained_axis'] = copy.deepcopy(stage['axis'])
    stage['axis']['axis'] = [1., 0., 0.]
    stage['ground_correction'] = {'trial_mode': True, 'boundary_check_passed': False}
    value = prior_positions(stage, calibration, {'hue-0': 40.})['candidate']
    assert value['status'] == 'PRIOR_CONDITIONAL_CENTER'
    assert value['axis_source'] == 'FREE_IMAGE_AXIS'
    np.testing.assert_allclose(value['axis_camera'], axis)


def test_observation_line_representatives_do_not_bridge_retained_run_gaps():
    boundary = {'status': 'STABLE', 'line_homogeneous': [0., 1., -20.], 'image_axis': [1., 0.],
                'retained_axial_runs_px': [[0, 200], [400, 500]]}
    points, _, runs = _line_samples(boundary, {}, np.array([13., 27.]))
    original = points-[13., 27.]
    assert len(points) == 128 and runs == 2
    assert np.all(((original[:, 0] >= 0) & (original[:, 0] < 200)) |
                  ((original[:, 0] >= 400) & (original[:, 0] < 500)))
    np.testing.assert_allclose(original[:, 1], 20.)


def test_one_bad_candidate_does_not_remove_other_prior_centers():
    stage, calibration, _, _ = fixture()
    stage['candidates'].append({'candidate_id': 'bad', 'left_candidate_id': 'missing', 'right_candidate_id': 'right'})
    values = prior_positions(stage, calibration, {'RED': 40.})
    assert values['candidate']['status'] == 'PRIOR_CONDITIONAL_CENTER'
    assert values['bad']['status'] == 'PRIOR_CENTER_UNAVAILABLE'


def test_known_diameter_resolves_parallel_baseline_but_free_radius_stays_unobservable():
    # The independent physical tangent generator constructs real cylinder
    # surface points and projects them, not the solver's predicted lines.
    observations, k, axis, center, image_axis, _ = analytic_samples(0., axial_z=0., radius=20.)
    free = fit_silhouettes(observations, k, 60., axis, image_axis=image_axis, reference_pixel=k[:2, 2])
    assert free['status'] == 'UNOBSERVABLE_PARALLEL_BASELINE'
    assert 'trial_diameter_mm' not in free
    for boundaries in (observations, [observations[0], observations[3]]):
        fitted = fit_silhouettes(boundaries, k, 60., axis, image_axis=image_axis,
                                reference_pixel=k[:2, 2], known_diameter_mm=40.)
        assert fitted['status'] == 'CONDITIONAL_FIT'
        np.testing.assert_allclose(fitted['center_at_reference_pixel_mm'], center, atol=1e-5)
        assert fitted['baseline_perpendicular_to_axis_mm'] == 0.
        assert fitted['diameter_source'] == 'USER_KNOWN_CONSTRAINT'
        assert fitted['accepted_diameter_mm'] is None
    wrong = fit_silhouettes(observations, k, 60., axis, image_axis=image_axis,
                            reference_pixel=k[:2, 2], known_diameter_mm=50.)
    assert wrong['status'] == 'CONDITIONAL_FIT'
    assert wrong['known_diameter_mm'] == 50. and wrong['accepted_diameter_mm'] is None
    assert wrong['center_at_reference_pixel_mm'][2] == pytest.approx(center[2]*1.25, abs=1e-5)
    assert wrong['diameter_source'] == 'USER_KNOWN_CONSTRAINT'
