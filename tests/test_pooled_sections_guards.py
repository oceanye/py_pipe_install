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
    row = result['pairing_audit']['left'][0]
    assert row['status'] == 'INSUFFICIENT_INDEPENDENT_SILHOUETTES'
    assert row['stable_boundary_count'] == 1
    assert row['compatible_candidate_ids'] == []
    assert row['nearby_candidates'][0]['combined_stable_boundary_count'] == 2
    assert row['nearby_candidates'][0]['reason_codes'] == ['INSUFFICIENT_INDEPENDENT_SILHOUETTES']


def test_partial_disjoint_sections_do_not_crash_or_merge_from_one_shared_edge(monkeypatch):
    views, k = synthetic_views()
    for role in views:
        original = views[role][0]
        items = []
        for index, (low, high) in enumerate([(400, 800), (1000, 1400)]):
            item = copy.deepcopy(original)
            item['candidate_id'] += str(index)
            item['region_px'] = [low, 200, high-low, 400]
            edges = [{**e, 'points': e['points'][(e['points'][:, 0] >= low) & (e['points'][:, 0] < high)]}
                     for e in item['pooled_edges']]
            # Different second boundaries are not known on the left. They
            # cannot be replaced by either the first edge or a failed fit.
            if role == 'left':
                edges[1]['points'] = np.empty((0, 2))
            item['aggregation'], item['pooled_edges'] = aggregate_boundaries(edges)
            items.append(item)
        views[role] = items
    result = run_views(monkeypatch, views, k)
    assert result['boundary_pair_count'] == 2
    assert all(c['boundary_status'] == 'PARTIAL' for c in result['candidates'])
    assert all(c['included_in_section_model'] for c in result['candidates'])
    assert not any('POSSIBLE_SHARED_PIPE_INSTANCE' in c['reason_codes'] for c in result['candidates'])
    assert all(not c['accepted_measurement'] for c in result['model']['candidates'])


def test_pairing_audit_preserves_ambiguity_instead_of_assigning_by_order(monkeypatch):
    views, k = synthetic_views()
    views['right'].append(copy.deepcopy(views['right'][0]))
    views['right'][1]['candidate_id'] += '-duplicate'
    result = run_views(monkeypatch, views, k)
    assert result['boundary_pair_count'] == 0
    assert result['pairing_audit']['left'][0]['status'] == 'AMBIGUOUS_STEREO_CANDIDATES'
    assert len(result['pairing_audit']['left'][0]['compatible_candidate_ids']) == 2
    assert all(row['status'] == 'AMBIGUOUS_STEREO_CANDIDATES' for row in result['pairing_audit']['right'])


def test_unobservable_pixel_perturbation_is_a_candidate_failure_not_pipeline_crash(monkeypatch):
    from test_projection_diagnostics import independent_contours
    axis = np.array([1., .0003, 0.])
    observations, k = independent_contours(axis, [0., 50., 1800.], 20.)
    views = {}
    for role in ('left', 'right'):
        edges = []
        for observation in observations:
            if observation['role'] != role:
                continue
            line = np.polyfit(observation['points'][:, 0], observation['points'][:, 1], 1)
            x = np.arange(400, 1400)
            edges.append({**observation, 'points': np.c_[x, np.polyval(line, x)]})
        audit, pooled = aggregate_boundaries(edges)
        views[role] = [{'candidate_id': 'POOL-'+role, 'role': role, 'region_px': [400, 200, 1000, 400],
                       'width_px': 34., 'appearance': 'hue-105', 'color_srgb': '#0066CC',
                       'reason_codes': [], 'boundary_status': 'STABLE', 'boundary_stable': True,
                       'aggregation': audit, 'pooled_edges': pooled}]
    result = run_views(monkeypatch, views, k)
    candidate = result['candidates'][0]
    assert candidate['fit']['status'] == 'CONDITIONAL_FIT'
    assert candidate['pixel_error']['status'] == 'PIXEL_UNCERTAINTY_UNRESOLVED'
    assert candidate['pixel_error']['scenarios'] == {}
    assert 'PIXEL_UNCERTAINTY_UNRESOLVED' in candidate['reason_codes']
    assert not result['model']['candidates'][0]['accepted_measurement']


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
