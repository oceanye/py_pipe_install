import copy
from types import SimpleNamespace

import numpy as np

from pipe_twin.axial_aggregation import aggregate_boundaries
from pipe_twin.pooled_sections import analyze_pooled_sections, _observed_edges
from pipe_twin.section_diagnostics import section_basis
from pipe_twin.section_model import build_section_model
from pipe_twin.section_position import assess_pooled_sections
from test_axial_aggregation import synthetic_views
from test_elevation_registration import _specs, _observations


def run_views(monkeypatch, views, k):
    import pipe_twin.pooled_sections as module
    monkeypatch.setattr(module, 'aggregate_image', lambda image, role: (copy.deepcopy(views[role]), {'truncated': False}))
    cal = SimpleNamespace(left=SimpleNamespace(rectified_intrinsic=k), right=SimpleNamespace(rectified_intrinsic=k), baseline_mm=60.)
    return analyze_pooled_sections(None, None, cal)


def test_stable_2d_edges_do_not_certify_an_unstable_3d_axis(monkeypatch):
    import pipe_twin.pooled_sections as module
    views, k = synthetic_views()
    monkeypatch.setattr(module, 'axis_from_image_lines', lambda *_: {
        'axis': [1., 0., .2], 'singular_values': [2., .03, .025], 'line_residual_degrees': [.5]*4})
    result = run_views(monkeypatch, views, k)
    assert result['boundary_pair_count'] == 1
    assert result['conditional_section_count'] == 0
    assert result['model'] is None
    assert 'COMMON_AXIS_UNSTABLE' in result['candidates'][0]['reason_codes']


def test_ambiguous_stereo_proposals_are_not_assigned_by_order(monkeypatch):
    views, k = synthetic_views()
    views['right'].append(copy.deepcopy(views['right'][0]))
    views['right'][-1]['candidate_id'] += '-other'
    result = run_views(monkeypatch, views, k)
    assert result['boundary_pair_count'] == 0
    assert not result['candidates']


def test_three_stable_tangents_keep_conditional_section_without_filling_fourth(monkeypatch):
    views, k = synthetic_views()
    right = views['right'][0]
    right['aggregation']['boundaries'][0]['status'] = 'UNSTABLE'
    right['aggregation']['stable_boundary_count'] = 1
    right['boundary_stable'] = False
    right['pooled_edges'] = right['pooled_edges'][1:]
    result = run_views(monkeypatch, views, k)
    assert result['conditional_section_count'] == 1
    candidate = result['candidates'][0]
    assert candidate['boundary_status'] == 'PARTIAL'
    assert candidate['fit']['observed_boundary_count'] == 3
    assert 'PARTIAL_STEREO_BOUNDARY_SUPPORT' in candidate['reason_codes']
    assert not result['occlusion_recovered']
    assert candidate['fit']['diameter_source'] == 'IMAGE_FIT'


def test_two_remaining_tangents_do_not_create_a_free_radius(monkeypatch):
    views, k = synthetic_views()
    for role in views:
        item = views[role][0]
        item['aggregation']['stable_boundary_count'] = 1
        item['aggregation']['boundaries'][0]['status'] = 'UNSTABLE'
        item['boundary_stable'] = False
        item['pooled_edges'] = item['pooled_edges'][1:]
    result = run_views(monkeypatch, views, k)
    assert result['boundary_pair_count'] == 0


def test_separated_pieces_of_the_same_outline_do_not_create_an_inter_pipe_gap(monkeypatch):
    views, k = synthetic_views()
    for role in views:
        original = views[role][0]
        items = []
        for index, (low, high) in enumerate([(400, 800), (1000, 1400)]):
            item = copy.deepcopy(original)
            item['candidate_id'] += str(index)
            item['region_px'] = [low, 200, high-low, 400]
            edges = [{**e, 'points': e['points'][(e['points'][:, 0] >= low) & (e['points'][:, 0] < high)]} for e in item['pooled_edges']]
            item['aggregation'], item['pooled_edges'] = aggregate_boundaries(edges)
            items.append(item)
        views[role] = items
    result = run_views(monkeypatch, views, k)
    assert result['boundary_pair_count'] == 2
    assert all('POSSIBLE_SHARED_PIPE_INSTANCE' in c['reason_codes'] for c in result['candidates'])
    assert len(result['model']['candidates']) == 1
    assert not result['model']['pairs']


def test_compound_outer_envelope_never_gets_a_single_cylinder_radius(monkeypatch):
    views, k = synthetic_views()
    views['right'][0]['reason_codes'] = ['MULTIPLE_BANDS_OR_OCCLUDED_INSTANCES']
    result = run_views(monkeypatch, views, k)
    assert result['boundary_pair_count'] == 1
    assert result['conditional_section_count'] == 0


def test_separate_bright_bands_are_a_reason_to_withhold_outline_identity():
    image = np.zeros((200, 800, 3), np.uint8)
    image[50:160, 50:750] = (0, 0, 220)
    image[75:82, 50:750] = 245; image[120:128, 50:750] = 245
    proposal = {'region_px': [50, 50, 700, 110], 'appearance': 'hue-0',
                'mask': np.full((110, 700), 255, np.uint8), 'width_px': 110,
                'axis_image': [1, 0], 'sampled_color_srgb': '#DD0000'}
    item = _observed_edges(image, proposal, 'left')
    assert item['highlight_band_count'] == 2
    assert 'MULTIPLE_BANDS_OR_OCCLUDED_INSTANCES' in item['reason_codes']


def test_section_model_registers_and_assesses_relative_placement_without_prefilled_matches():
    specs = _specs()
    observations = _observations(specs)
    axis = np.array([0., 0., 1.]); basis = section_basis(axis)
    items, candidates = [], []
    for o in observations:
        uncertainty = {'scenarios': {'COHERENT_EDGE_OFFSETS': {
            'section_parameter_covariance_mm2': (np.eye(3)*.01).tolist(), 'diameter_sigma_mm': .2}}}
        fit = {'status': 'CONDITIONAL_FIT', 'axis': axis.tolist(), 'diameter_source': 'IMAGE_FIT',
               'section_center_parameters_mm': (basis.T@o['center_camera_mm']).tolist(), 'trial_diameter_mm': o['diameter_mm']}
        items.append({'id': o['observation_id'], 'fit': fit, 'pixel_error': uncertainty})
        candidates.append({'candidate_id': o['observation_id'], 'boundary_status': 'STABLE', 'reason_codes': [],
                           'window_sensitivity': {'stable': True}, 'fit': fit, 'pixel_error': uncertainty})
    pooled = {'model': build_section_model(items, axis), 'candidates': candidates}
    result = assess_pooled_sections(pooled, specs, SimpleNamespace(validated=True), capture_healthy=True,
                                    settings={'axis_world': [0, 0, 1]}, matching={})
    assert result['registration']['status'] == 'MATCHED'
    assert result['position_assessment']['counts']['IN_POSITION'] == 4
    # Once one section loses support, it cannot borrow a valid neighbour's fit.
    pooled['candidates'][0]['window_sensitivity']['stable'] = False
    result = assess_pooled_sections(pooled, specs, SimpleNamespace(validated=True), capture_healthy=True,
                                    settings={'axis_world': [0, 0, 1]}, matching={})
    assert result['position_assessment']['counts']['IN_POSITION'] == 3
    assert result['position_assessment']['pipes'][0]['state'] == 'UNKNOWN'
