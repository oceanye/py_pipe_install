import copy
import numpy as np

from pipe_twin.relative_scene import scene_sections, model_states, draw_scene
from unittest import mock


def report_fixture():
    return {'section_evaluation': {
        'model': {'candidates': [
            {'id': 'a', 'estimated_center_uv_mm': [100, 2100], 'diameter_mm': 42., 'color': '#DD3333',
             'diameter_source': 'IMAGE_FIT'},
            {'id': 'b', 'estimated_center_uv_mm': [180, 2200], 'diameter_mm': 48., 'color': '#3355DD',
             'diameter_source': 'IMAGE_FIT'}], 'section': {'normal_camera': [.9, .3, .3]}, 'pairs': []},
        'candidates': [{'candidate_id': 'a', 'boundary_status': 'STABLE', 'reason_codes': [],
                        'window_sensitivity': {'stable': True}},
                       {'candidate_id': 'b', 'reason_codes': ['POOLED_DIMENSIONS_UNSTABLE']}],
        'registration': {'status': 'AMBIGUOUS', 'matches': [{'observation_id': 'a', 'pipe_id': 'P001'}]}}}


def test_unregistered_visible_sections_remain_visible_without_changing_measured_dimensions():
    report = report_fixture(); original = copy.deepcopy(report)
    scene = scene_sections(report)[0]
    assert [i['diameter_mm'] for i in scene['items']] == [42., 48.]
    assert [i['center_uv_mm'] for i in scene['items']] == [[0., 0.], [80., 100.]]
    assert all(i['pipe_id'] is None for i in scene['items'])
    assert scene['independent_of_ground']
    assert scene['items'][0]['subset_stable']
    assert not scene['items'][1]['subset_stable']
    assert report == original
    assert not model_states(report)


def test_only_unique_correspondence_links_observation_to_cad():
    report = report_fixture(); registration = report['section_evaluation']['registration']
    registration['status'] = 'MATCHED'
    assert scene_sections(report)[0]['items'][0]['pipe_id'] == 'P001'
    registration['matches'].append({'observation_id': 'b', 'pipe_id': 'P001'})
    assert all(i['pipe_id'] is None for i in scene_sections(report)[0]['items'])


def test_presentation_does_not_promote_candidates_to_installed_or_unseen_to_absent():
    report = {'pipes': [{'pipe_id': 'P001', 'installation_state': 'INSTALLED'},
                        {'pipe_id': 'P002', 'installation_state': 'UNKNOWN'},
                        {'pipe_id': 'P003', 'installation_state': 'UNKNOWN'},
                        {'pipe_id': 'P004', 'installation_state': 'NOT_INSTALLED'}],
              'position_assessment': {'pipes': [{'pipe_id': 'P001', 'state': 'UNKNOWN'},
                                                {'pipe_id': 'P002', 'state': 'IN_POSITION'},
                                                {'pipe_id': 'P003', 'state': 'OUT_OF_POSITION'}]}}
    assert model_states(report) == {'P001': 'UNKNOWN', 'P002': 'IN_POSITION', 'P003': 'OUT_OF_POSITION', 'P004': 'UNKNOWN'}
    report['pipes'][0]['installation_evidence_qualified'] = True
    assert model_states(report)['P001'] == 'INSTALLED'


def test_unrelated_zones_have_separate_origins_and_covariance_is_relative():
    report = report_fixture(); stage = report.pop('section_evaluation')
    for candidate in stage['model']['candidates']:
        candidate['pixel_error'] = {'scenarios': {'COHERENT_EDGE_OFFSETS': {
            'section_parameter_covariance_mm2': np.eye(3).tolist()}}}
    other = copy.deepcopy(stage); other['zone_id'] = 'Z2'
    for candidate in other['model']['candidates']:
        candidate['estimated_center_uv_mm'][1] += 500
    report['section_evaluations'] = [stage, other]
    scenes = scene_sections(report)
    assert scenes[0]['items'] == scenes[1]['items']
    np.testing.assert_array_equal(scenes[0]['items'][0]['center_covariance_mm2'], np.zeros((2, 2)))
    np.testing.assert_array_equal(scenes[0]['items'][1]['center_covariance_mm2'], 2*np.eye(2))


def test_clockwise_display_rotates_centers_and_uncertainty_but_preserves_measurements():
    scene = scene_sections(report_fixture())[0]
    scene['items'][1]['center_covariance_mm2'] = [[9., 0.], [0., 4.]]
    original = copy.deepcopy(scene)
    canvas = mock.Mock()
    canvas.winfo_width.return_value = canvas.winfo_height.return_value = 800
    unrotated = draw_scene(canvas, scene)
    old_ellipse = np.asarray(canvas.create_line.call_args.args).reshape(-1, 2)
    canvas.reset_mock()
    rotated = draw_scene(canvas, scene, clockwise_quarters=1)
    new_ellipse = np.asarray(canvas.create_line.call_args.args).reshape(-1, 2)
    before = np.subtract(unrotated[1][1:3], unrotated[0][1:3])
    after = np.subtract(rotated[1][1:3], rotated[0][1:3])
    np.testing.assert_allclose(after / np.linalg.norm(after), [-before[1], before[0]] / np.linalg.norm(before))
    assert np.ptp(old_ellipse[:, 1]) > np.ptp(old_ellipse[:, 0])
    assert np.ptp(new_ellipse[:, 0]) > np.ptp(new_ellipse[:, 1])
    assert scene == original
    assert [hit[0]['diameter_mm'] for hit in rotated] == [42., 48.]
