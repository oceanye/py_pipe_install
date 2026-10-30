"""Identity assistance does not require measurement acceptance or invent absence."""
import copy

import pytest

import pipe_twin.installation_assist as assist

HINTS = {'RED': 40., 'WHITE': 25., 'BLUE': 50.}
COLORS = {41: '#FF2222', 26: '#FFFFFF', 51: '#1166EE'}


def spec(identity, x, y, diameter, color='#FFFFFF', source='stl_synthetic_by_diameter'):
    return {'pipe_id': identity, 'nominal_diameter_mm': diameter, 'color_srgb': color, 'color_source': source,
            'centerline_world_mm': [[x, y, 0.], [x, y, 500.]]}


def image_report(rows, *, solved=False):
    views, candidates, model = [], [], []
    for index, (q, width, color) in enumerate(rows):
        views.append({'candidate_id': f'L{index}', 'region_px': [10., q-width/2, 600., width],
                      'width_px': width, 'color_srgb': color, 'image_axis': [1., 0.],
                      'aggregation': {'stable_boundary_count': 1, 'boundaries': []}, 'reason_codes': []})
        candidates.append({'candidate_id': f'O{index}', 'left_candidate_id': f'L{index}',
                           'fit': {'status': 'CONDITIONAL_FIT' if solved else 'UNRESOLVED'},
                           'reason_codes': ['POOLED_DIMENSIONS_UNSTABLE', 'GROUND_AXIS_CONFLICT']})
        model.append({'id': f'O{index}', 'estimated_center_uv_mm': [0., index*100.] if solved else None})
    return {'section_evaluation': {'views': {'left': views, 'right': []}, 'candidates': candidates,
                                  'model': {'candidates': model}},
            'counts': {'INSTALLED': 0, 'UNKNOWN': 3}, 'temporal_fusion': {'status': 'INVALID'}}


def settings():
    return {'camera_side_world': [0., 1., 0.], 'installation_assist': {
        'appearance_diameter_hints': HINTS, 'hint_source': 'USER_STATED_PURCHASE_DIAMETERS'}}


def test_colour_size_class_and_image_layout_work_without_any_metric_fit():
    models = [spec('RED', 0, 0, 41, '#1166EE'), spec('WHITE', 100, 0, 26, '#FF2222'),
              spec('BLUE', 220, 0, 51, '#FFFFFF')]
    report = image_report([(100., 61.5, COLORS[41]), (250., 39., COLORS[26]), (430., 76.5, COLORS[51])])
    original = copy.deepcopy((report, models))
    value = assist.analyze_installation_assist(report, models, settings())
    assert value['status'] == 'LIKELY'
    assert [o['suggested_pipe_id'] for o in value['observations']] == ['RED', 'WHITE', 'BLUE']
    assert value['synthetic_model_colors_ignored'] and not value['known_diameter_used_for_localization']
    assert value['method'] == 'IMAGE_LAYOUT_ONLY'
    assert value['advisory_only'] and not value['installation_acceptance'] and not value['accepted_measurement']
    assert (report, models) == original


def test_repeated_identical_row_keeps_ambiguity_even_if_only_one_hypothesis_displayed(monkeypatch):
    models = [spec(f'P{i}', i*100., 0., 41) for i in range(4)]
    report = image_report([(100.+i*100, 41., COLORS[41]) for i in range(3)])
    monkeypatch.setitem(assist.POLICY, 'maximum_displayed_hypotheses', 1)
    value = assist.analyze_installation_assist(report, models, settings())
    assert value['plausible_hypothesis_count'] > 1 and len(value['ranked_hypotheses']) == 1
    assert all(o['status'] == 'AMBIGUOUS' and len(o['candidate_pipe_ids']) > 1 for o in value['observations'])
    assert all(o['suggested_pipe_id'] is None for o in value['observations'])


def test_no_synthetic_colour_assumed_and_unknown_view_does_not_invent_world_height():
    models = [spec(f'P{i}', i*100., 0., 41) for i in range(4)]
    report = image_report([(100.+i*100, 41., COLORS[41]) for i in range(3)])
    value = assist.analyze_installation_assist(report, models)
    assert value['status'] == 'AMBIGUOUS'
    assert all(p['model_layer_index'] is None for p in value['pipes'])
    assert all(not pair['world_vertical_order_known'] for pair in value['pairwise_relations'])


def test_one_observation_is_candidate_only_and_unseen_model_is_not_absent():
    models = [spec('P1', 0, 0, 41), spec('P2', 100, 0, 26)]
    result = assist.analyze_installation_assist(image_report([(100., 41., COLORS[41])]), models, settings())
    assert result['status'] == 'INSUFFICIENT'
    assert result['observations'][0]['candidate_pipe_ids'] == ['P1']
    assert all(p['status'] == 'NO_OBSERVATION' for p in result['pipes'])
    assert 'NOT_INSTALLED' not in str(result)


def prior_fixture(monkeypatch, *, missing=()):
    models = [spec('A', 0, 100, 41), spec('B', 100, 0, 41), spec('C', 100, 200, 26),
              spec('D', 250, 100, 51), spec('UNSEEN', 400, 300, 51)]
    report = image_report([(100.+m['centerline_world_mm'][0][0]-.2*m['centerline_world_mm'][0][1],
                            m['nominal_diameter_mm']*1.5, COLORS[m['nominal_diameter_mm']]) for m in models[:4]], solved=True)
    values = {}
    for index, m in enumerate(models[:4]):
        if index in missing:
            continue
        x, y, _ = m['centerline_world_mm'][0]
        values[f'O{index}'] = {'status': 'PRIOR_CONDITIONAL_CENTER', 'section_center_uv_mm': [x+50., 1000.-y],
                              'axis_camera': [1., 0., 0.], 'diameter_source': 'USER_STATED_PRIOR'}
    monkeypatch.setattr('pipe_twin.assist_prior_geometry.prior_positions', lambda *args: values)
    return models, report


def test_coarse_prior_relative_geometry_identifies_permutation_and_full_model_layers(monkeypatch):
    models, report = prior_fixture(monkeypatch)
    result = assist.analyze_installation_assist(report, list(reversed(models)), settings(), calibration=object())
    assert [o['suggested_pipe_id'] for o in result['observations']] == ['A', 'B', 'C', 'D']
    assert result['known_diameter_used_for_localization']
    assert result['counts'] == {'LIKELY': 4, 'AMBIGUOUS': 0, 'NO_OBSERVATION': 1}
    # Y=300 unseen pipe still defines nearest layer; selected Y=200 is layer 2.
    assert result['observations'][2]['model_layer_index_from_camera'] == 2
    assert result['observations'][2]['model_layer_count'] == 4
    assert result['ranked_hypotheses'][0]['difference_score'] < 1e-9


def test_missing_prior_never_gets_identity_by_eliminating_other_assignments(monkeypatch):
    models, report = prior_fixture(monkeypatch, missing=(3,))
    result = assist.analyze_installation_assist(report, models, settings(), calibration=object())
    last = result['observations'][3]
    assert last['status'] == 'INSUFFICIENT' and last['suggested_pipe_id'] is None
    assert last['reason_codes'] == ['PRIOR_POSITION_UNAVAILABLE']


def test_duplicate_occluded_piece_is_not_counted_as_an_extra_pipe(monkeypatch):
    models, report = prior_fixture(monkeypatch)
    stage = report['section_evaluation']
    duplicate = copy.deepcopy(stage['views']['left'][0]); duplicate['candidate_id'] = 'DUPLICATE'
    stage['views']['left'].append(duplicate)
    stage['candidates'].append({'candidate_id': 'duplicate-piece', 'left_candidate_id': 'DUPLICATE',
                                'included_in_section_model': False, 'reason_codes': ['POSSIBLE_SHARED_PIPE_INSTANCE']})
    result = assist.analyze_installation_assist(report, models, settings(), calibration=object())
    assert len(result['observations']) == 4 and result['counts']['LIKELY'] == 4


def test_search_budget_counts_collision_branches_and_never_promotes_partial_search(monkeypatch):
    monkeypatch.setitem(assist.POLICY, 'maximum_assignments', 12)
    models = [spec(f'P{i}', i*100., 0., 41) for i in range(6)]
    report = image_report([(100.+i*100, 41., COLORS[41]) for i in range(3)])
    result = assist.analyze_installation_assist(report, models, settings())
    assert result['search_truncated'] and result['search_branches'] == 12
    assert result['counts']['LIKELY'] == 0


def test_more_observations_than_model_returns_without_combinatorial_work():
    models = [spec(f'P{i}', i*100., 0., 41) for i in range(3)]
    report = image_report([(100.+i*100, 41., COLORS[41]) for i in range(6)])
    result = assist.analyze_installation_assist(report, models, settings())
    assert result['reason_codes'] == ['MORE_OBSERVATIONS_THAN_MODEL_MEMBERS']
    assert result['ranked_hypotheses'] == []


def test_image_axis_reversal_does_not_reverse_identity_assignment():
    models = [spec('R', 0, 0, 41), spec('W', 100, 0, 26), spec('B', 220, 0, 51)]
    report = image_report([(100., 41., COLORS[41]), (200., 26., COLORS[26]), (320., 51., COLORS[51])])
    before = assist.analyze_installation_assist(report, models, settings())
    for view in report['section_evaluation']['views']['left']:
        view['image_axis'] = [-1., 0.]
    after = assist.analyze_installation_assist(report, models, settings())
    assert before['ranked_hypotheses'] == after['ranked_hypotheses']


def test_independent_historical_regions_are_not_joined():
    models = [spec('P1', 0, 0, 41)]
    report = {'section_evaluations': [image_report([(100., 41., COLORS[41])])['section_evaluation']]*2}
    result = assist.analyze_installation_assist(report, models, settings())
    assert result['status'] == 'INSUFFICIENT'
    assert result['reason_codes'] == ['NO_SINGLE_CURRENT_IMAGE_SECTION']


def test_right_observation_box_restores_refinement_then_crop_offset():
    models = [spec('P1', 0, 0, 41)]
    report = image_report([(100., 41., COLORS[41])])
    stage = report['section_evaluation']
    stage['candidates'][0]['right_candidate_id'] = 'R1'
    stage['views']['right'] = [{'candidate_id': 'R1', 'region_px': [10., 20., 100., 20.]}]
    stage['epipolar_refinement'] = {'applied': True,
                                   'right_refined_to_original': [[1., 0., 3.], [0., 1., 5.], [0., 0., 1.]]}
    stage['zone_offsets'] = {'right_x': 7., 'right_y': 11.}
    result = assist.analyze_installation_assist(report, models, settings())
    assert result['observations'][0]['right_region_px'] == pytest.approx([20., 36., 100., 20.])
