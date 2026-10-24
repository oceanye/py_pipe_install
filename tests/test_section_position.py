import copy

import pytest

from pipe_twin.section_position import compare_section_position, merge_position_assessments


def scene():
    specs = [{'pipe_id': str(i), 'centerline_world_mm': [[x, 0., -100.], [x, 0., 200.]],
              'nominal_diameter_mm': d} for i, (x, d) in enumerate([(0., 50.), (150., 40.), (300., 25.)])]
    candidates = [{'candidate_id': str(i), 'boundary_status': 'STABLE', 'reason_codes': [],
                   'window_sensitivity': {'stable': True}, 'pixel_error': {'scenarios': {
                       'COHERENT_EDGE_OFFSETS': {'diameter_sigma_mm': .1}}}} for i in range(2)]
    model = {'candidates': [{'id': str(i), 'diameter_mm': d} for i, d in enumerate([50., 40.])],
             'pairs': [{'ids': ['0', '1'], 'center_distance_mm': 150., 'surface_clearance_mm': 105.,
                        'pixel_error': {'scenarios': {'COHERENT_EDGE_OFFSETS': {
                            'center_distance_normal_95_halfwidth_mm': .5,
                            'surface_clearance_normal_95_halfwidth_mm': .6}}}}]}
    registration = {'status': 'MATCHED', 'matches': [{'pipe_id': str(i), 'observation_id': str(i)} for i in range(2)]}
    return model, candidates, specs, registration


def evaluate(data, **kwargs):
    return compare_section_position(*data, **({'calibration_validated': True, 'capture_healthy': True} | kwargs))


def test_relative_position_passes_without_camera_pose_length_or_hidden_pipe_inference():
    data = scene()
    first = evaluate(data)
    assert first['counts'] == {'IN_POSITION': 2, 'OUT_OF_POSITION': 0, 'UNKNOWN': 1}
    assert first['pipes'][-1]['state'] == 'UNKNOWN'
    for spec in data[2]:
        spec['centerline_world_mm'][0][2] -= 10000
        spec['centerline_world_mm'][1][2] += 5000
        for endpoint in spec['centerline_world_mm']:
            endpoint[0] += 9000; endpoint[1] -= 3000
    assert evaluate(data)['counts'] == first['counts']
    assert not first['absence_assessed']
    assert not first['absolute_installation_pose_assessed']


@pytest.mark.parametrize('offset,halfwidth,state', [(10., .5, 'OUT_OF_POSITION'), (4.8, .5, 'UNKNOWN'), (2., .5, 'IN_POSITION')])
def test_uncertainty_interval_controls_position_decision(offset, halfwidth, state):
    data = scene(); pair = data[0]['pairs'][0]
    pair['center_distance_mm'] += offset; pair['surface_clearance_mm'] += offset
    pair['pixel_error']['scenarios']['COHERENT_EDGE_OFFSETS']['center_distance_normal_95_halfwidth_mm'] = halfwidth
    result = evaluate(data)
    assert result['pipes'][0]['state'] == state


@pytest.mark.parametrize('kwargs,reason', [({'calibration_validated': False}, 'CALIBRATION_NOT_VALIDATED'),
    ({'capture_healthy': False}, 'CAPTURE_UNHEALTHY'), ({'truncated': True}, 'SECTION_SEARCH_TRUNCATED')])
def test_insufficient_quality_cannot_be_promoted(kwargs, reason):
    result = evaluate(scene(), **kwargs)
    assert result['counts']['UNKNOWN'] == 3
    assert reason in result['pipes'][0]['reason_codes']


def test_missing_error_estimate_unstable_segment_or_repeated_assignment_stays_unknown():
    data = scene(); data[1][0]['pixel_error'] = {}
    assert evaluate(data)['pipes'][0]['state'] == 'UNKNOWN'
    data = scene(); data[1][0]['window_sensitivity']['stable'] = False
    assert evaluate(data)['pipes'][0]['state'] == 'UNKNOWN'
    data = scene(); data[3]['matches'][1]['observation_id'] = '0'
    assert evaluate(data)['counts']['UNKNOWN'] == 3


def test_zone_contradiction_remains_unknown_instead_of_selecting_a_positive_vote():
    good = evaluate(scene()); bad = copy.deepcopy(good)
    bad['pipes'][0]['state'] = 'OUT_OF_POSITION'
    merged = merge_position_assessments([good, bad], scene()[2])
    assert merged['pipes'][0]['state'] == 'UNKNOWN'
    assert merged['pipes'][0]['reason_codes'] == ['ZONE_POSITION_CONFLICT']
