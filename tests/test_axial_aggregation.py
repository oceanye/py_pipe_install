import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from pipe_twin.axial_aggregation import aggregate_boundaries
from pipe_twin.pooled_sections import analyze_pooled_sections, _compatible
from pipe_twin.section_display import boundary_overlay_segments, section_summary_lines
from test_projection_diagnostics import independent_contours


def edge(x=None):
    x = np.arange(1024) if x is None else np.asarray(x)
    return {'role': 'left', 'edge': 'top', 'points': np.c_[x, 300-.07*x+.15*np.sin(x)]}


def test_pooling_preserves_image_coordinates_and_rejects_duplicates_as_extra_support():
    audit, pooled = aggregate_boundaries([edge()])
    assert audit['stable_boundary_count'] == 1
    assert audit['boundaries'][0]['accepted_windows'] == 8
    np.testing.assert_allclose(pooled[0]['points'], edge()['points'])
    original = edge(np.arange(100))
    original['points'] = np.tile(original['points'], (20, 1))
    audit, pooled = aggregate_boundaries([original])
    assert audit['stable_boundary_count'] == 0
    assert not pooled


def test_occlusion_and_empty_windows_are_never_filled():
    a = edge(np.r_[np.arange(384), np.arange(640, 1024)])
    b = edge(); b.update(edge='bottom', boundary_kind='occlusion')
    audit, pooled = aggregate_boundaries([a, b])
    assert audit['occluded_boundary_count'] == 1
    assert len(pooled) == 1
    assert not np.any((pooled[0]['points'][:, 0] >= 384) & (pooled[0]['points'][:, 0] < 640))
    assert sum(not w['accepted'] for w in audit['boundaries'][0]['windows']) == 2


def test_real_column_runs_preserve_gaps_inside_otherwise_accepted_windows():
    columns = np.concatenate([np.r_[np.arange(start, start+45), np.arange(start+75, start+128)]
        for start in range(0, 384, 128)])
    audit, pooled = aggregate_boundaries([edge(columns)])
    boundary = audit['boundaries'][0]
    assert boundary['status'] == 'STABLE'
    # Adjacent source windows form one continuous observed run.
    expected = [[0, 45], [75, 173], [203, 301], [331, 384]]
    assert boundary['observed_column_runs_px'] == expected
    assert boundary['retained_column_runs_px'] == expected
    reconstructed = np.concatenate([np.arange(low, high) for low, high in expected])
    np.testing.assert_array_equal(reconstructed, pooled[0]['points'][:, 0])


def test_more_columns_do_not_certify_a_bent_boundary():
    a = edge(); a['points'][:, 1] += 12*np.sin(a['points'][:, 0]/170)
    audit, pooled = aggregate_boundaries([a])
    assert not pooled
    assert audit['stable_boundary_count'] == 0


def test_adaptive_span_preserves_one_stable_edge_when_other_is_unusable(monkeypatch):
    import pipe_twin.pooled_sections as module
    top = edge()
    top['points'][450:, 1] += .0002*(top['points'][450:, 0]-450)**2
    bottom = edge(); bottom['edge'] = 'bottom'
    bottom['points'][:, 1] += 40+12*np.sin(bottom['points'][:, 0]/30)
    proposal = {'role': 'left', 'region_px': [0, 200, 1024, 200], 'appearance': 'hue-0',
                'width_px': 40., 'edges': [top, bottom], 'reason_codes': []}
    monkeypatch.setattr(module, 'propose_regions', lambda *a, **kw: ([{}], {}))
    monkeypatch.setattr(module, '_observed_edges', lambda *a: copy.deepcopy(proposal))
    items, _ = module.aggregate_image(None, 'left')
    row = items[0]
    assert row['full_span_audit']['stable_boundary_count'] == 0
    assert row['aggregation']['stable_boundary_count'] == 1
    assert len(row['pooled_edges']) == 1 and row['pooled_edges'][0]['edge'] == 'top'
    assert not row['boundary_stable']
    assert np.max(row['pooled_edges'][0]['points'][:, 0]) < 640
    assert row['aggregation']['boundaries'][0]['holdout_p95_px'] <= 2


def synthetic_views():
    axis = np.array([1., -.35, .15]); axis /= np.linalg.norm(axis)
    observations, k = independent_contours(axis, [100., 80., 1800.], 25.)
    views = {}
    for role in ('left', 'right'):
        edges = []
        for o in observations:
            if o['role'] != role:
                continue
            line = np.polyfit(o['points'][:, 0], o['points'][:, 1], 1)
            x = np.arange(400, 1400)
            edges.append({**o, 'points': np.c_[x, np.polyval(line, x)]})
        audit, pooled = aggregate_boundaries(edges)
        views[role] = [{'candidate_id': 'POOL-'+role, 'role': role, 'region_px': [400, 200, 1000, 400],
                        'width_px': 34., 'appearance': 'hue-105', 'color_srgb': '#0066CC',
                        'reason_codes': [], 'boundary_status': 'STABLE', 'boundary_stable': True,
                        'aggregation': audit, 'pooled_edges': pooled}]
    return views, k


def test_independent_geometry_flows_from_stable_boundaries_to_free_radius(monkeypatch):
    import pipe_twin.pooled_sections as module
    views, k = synthetic_views()
    monkeypatch.setattr(module, 'aggregate_image', lambda image, role: (copy.deepcopy(views[role]), {'truncated': False}))
    cal = SimpleNamespace(left=SimpleNamespace(rectified_intrinsic=k), right=SimpleNamespace(rectified_intrinsic=k), baseline_mm=60.)
    result = analyze_pooled_sections(None, None, cal)
    candidate = result['candidates'][0]
    assert candidate['fit']['trial_diameter_mm'] == pytest.approx(50., abs=.02)
    assert candidate['window_sensitivity']['stable']
    assert not candidate['reason_codes']
    assert result['nominal_dimensions_used_in_fitting'] is False
    assert result['occlusion_recovered'] is False
    json.dumps(result, allow_nan=False)


def test_asymmetric_zone_crop_keeps_shared_image_coordinate_system(monkeypatch):
    import pipe_twin.pooled_sections as module
    views, k = synthetic_views()
    shift = np.array([170., 35.])
    right = views['right'][0]
    for o in right['pooled_edges']:
        o['points'] -= shift
    right['aggregation'], right['pooled_edges'] = aggregate_boundaries(right['pooled_edges'])
    right['region_px'][:2] = (np.array(right['region_px'][:2])-shift).tolist()
    assert _compatible(views['left'][0], right, shift)
    monkeypatch.setattr(module, 'aggregate_image', lambda image, role: (copy.deepcopy(views[role]), {'truncated': False}))
    kr = k.copy(); kr[:2, 2] -= shift
    cal = SimpleNamespace(left=SimpleNamespace(rectified_intrinsic=k), right=SimpleNamespace(rectified_intrinsic=kr), baseline_mm=60.)
    result = analyze_pooled_sections(None, None, cal)
    assert result['candidates'][0]['fit']['trial_diameter_mm'] == pytest.approx(50., abs=.02)


def test_gui_boundary_overlay_restores_zone_offsets_and_skips_missing_blocks():
    views, _ = synthetic_views()
    report = {'section_evaluations': [{'zone_id': 'Z01', 'zone_offsets': {'left_x': 100, 'left_y': 30},
                                     'views': views, 'candidates': []}]}
    segments = boundary_overlay_segments(report, 'left')
    assert segments[0][0] == 500
    a = views['left'][0]['aggregation']['boundaries'][0]
    assert segments[0][1] == pytest.approx(400*a['line_slope_intercept'][0]+a['line_slope_intercept'][1]+30)
    assert '不会恢复遮挡' in '\n'.join(section_summary_lines(report, detailed=True))
    assert '⑤保存并评估' in '\n'.join(section_summary_lines(None))
