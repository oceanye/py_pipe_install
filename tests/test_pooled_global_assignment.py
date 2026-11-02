import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pipe_twin.axial_aggregation import aggregate_boundaries
from pipe_twin.pooled_sections import analyze_pooled_sections, _assign_pairs, _compatibility_reasons
from pipe_twin.stereo_assignment import global_assignment, POLICY
from test_projection_diagnostics import independent_contours


def assign(costs, *, compatible=None, cue=lambda a, b: 0):
    costs = np.asarray(costs, float)
    compatible = np.ones(costs.shape, bool) if compatible is None else np.asarray(compatible, bool)
    evidence = {(i, j): {'score': float(costs[i, j])} for i, j in zip(*np.nonzero(compatible))}
    return global_assignment(compatible, evidence, cue)


def test_joint_assignment_resolves_competing_same_colour_edges_without_reusing_right_pipe():
    pairs, audit = assign([[.015, .113], [.164, .044]])
    assert pairs == [(0, 0), (1, 1)]
    assert {p['basis'] for p in audit['selected_pairs']} == {'GLOBAL_ONE_TO_ONE'}
    assert not audit['nominal_dimensions_used']
    assert audit['components'][0]['edge_decisions'][0]['score_margin'] == pytest.approx(.109)
    json.dumps(audit, allow_nan=False)


def test_global_assignment_is_invariant_to_candidate_enumeration():
    costs = np.array([[.015, .113], [.164, .044]])
    pairs, _ = assign(costs[:, ::-1])
    assert pairs == [(0, 1), (1, 0)]


def test_missing_pipe_can_be_unmatched_and_identical_duplicates_stay_ambiguous():
    pairs, audit = assign([[0., 0., .2], [.2, .2, 0.]])
    assert pairs == [(1, 2)]
    assert len(audit['components'][0]['best_pairs']) == 2
    # There is no arbitrary tie-break selecting either identical observation.
    pairs, _ = assign([[0., 0.]])
    assert not pairs


def test_image_order_alone_does_not_choose_identical_depth_layer_candidates():
    pairs, _ = assign([[0., 0.], [0., 0.]], cue=lambda a, b: int((a[0]-b[0])*(a[1]-b[1]) < 0))
    assert not pairs


def test_soft_order_does_not_forbid_a_strongly_supported_crossing():
    pairs, _ = assign([[.2, .0], [.0, .2]], cue=lambda a, b: int((a[0]-b[0])*(a[1]-b[1]) < 0))
    assert pairs == [(0, 1), (1, 0)]


def test_ambiguous_edges_do_not_hide_an_independent_unique_pipe():
    pairs, _ = assign(np.zeros((3, 3)), compatible=[[1, 1, 0], [1, 1, 0], [0, 0, 1]])
    assert pairs == [(2, 2)]


def test_missing_observations_prefer_one_good_pair_to_two_weak_forced_pairs():
    pairs, audit = assign([[0., .2], [.2, 0.]], compatible=[[1, 1], [1, 0]])
    assert pairs == [(0, 0)]
    assert audit['components'][0]['best_pairs'] == [[0, 0]]
    assert audit['unmatched_allowed']


def test_low_margin_pair_is_not_promoted_just_to_increase_pair_count():
    pairs, _ = assign([[.2, .24]])
    assert pairs == []


def test_search_limit_withholds_new_pairs_instead_of_using_incomplete_best(monkeypatch):
    monkeypatch.setitem(POLICY, 'maximum_search_nodes', 1)
    pairs, audit = assign([[0., .2], [.2, 0.]])
    assert not pairs
    assert audit['components'][0]['status'] == 'SEARCH_INCOMPLETE'


def projected_views():
    views = {'left': [], 'right': []}
    for index, center in enumerate(([70., 0., 1200.], [0., 0., 1000.])):
        observations, k = independent_contours([0., 1., 0.], center, 20.)
        for role in views:
            contours = [o for o in observations if o['role'] == role]
            contours.sort(key=lambda o: -np.mean(o['points'][:, 0]))
            edges = []
            for label, observation in zip(('top', 'bottom'), contours):
                y = np.arange(200., 800.)
                line = np.polyfit(observation['points'][:, 1], observation['points'][:, 0], 1)
                edges.append({'role': role, 'edge': label, 'image_axis': [0., 1.],
                              'points': np.c_[np.polyval(line, y), y]})
            points = np.vstack([e['points'] for e in edges])
            width = float(np.ptp(points[:, 0]))
            aggregation, pooled = aggregate_boundaries(edges)
            views[role].append({'candidate_id': f'{role}-{index}', 'role': role,
                'region_px': [float(points[:, 0].min()), 200., width, 600.],
                'image_axis': [0., 1.], 'width_px': width,
                'appearance': 'hue-0', 'color_srgb': '#EE2222', 'reason_codes': [],
                'boundary_status': 'STABLE', 'boundary_stable': True,
                'aggregation': aggregation, 'pooled_edges': pooled})
    return views, k


def test_independently_projected_repeated_red_pipes_pair_globally_and_recover_free_radii(monkeypatch):
    import pipe_twin.pooled_sections as module
    views, k = projected_views()
    comparisons = np.array([[not _compatibility_reasons(a, b) for b in views['right']] for a in views['left']])
    assert comparisons.all()  # Neither left candidate has a local unique choice.
    monkeypatch.setattr(module, 'aggregate_image', lambda image, role: (copy.deepcopy(views[role]), {'truncated': False}))
    cal = SimpleNamespace(left=SimpleNamespace(rectified_intrinsic=k), right=SimpleNamespace(rectified_intrinsic=k), baseline_mm=60.)
    result = analyze_pooled_sections(None, None, cal)
    assert result['boundary_pair_count'] == result['conditional_section_count'] == 2
    assert all(c['pairing_basis'] == 'GLOBAL_ONE_TO_ONE' for c in result['candidates'])
    assert [c['fit']['trial_diameter_mm'] for c in result['candidates']] == pytest.approx([40., 40.], abs=.02)
    assert not result['nominal_dimensions_used_in_fitting']
    assert all(row['status'] == 'PAIRED' for row in result['pairing_audit']['left'])
    assert [row['selected_candidate_id'] for row in result['pairing_audit']['left']] == ['right-0', 'right-1']


def test_occluded_width_failure_is_not_relaxed_by_global_assignment():
    views, _ = projected_views()
    views['right'][0]['width_px'] *= .4
    comparisons = np.array([[not _compatibility_reasons(a, b) for b in views['right']] for a in views['left']])
    assert not comparisons[:, 0].any()
    pairs, _ = _assign_pairs(views, comparisons, (0., 0.), (0., 256.))
    assert not any(j == 0 for _, j in pairs)
