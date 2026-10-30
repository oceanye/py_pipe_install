"""Analytic image truth for axial/normal sampling, including vertical pipes."""
import copy
import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pipe_twin.axial_aggregation import aggregate_boundaries
from pipe_twin.image_line_geometry import canonical_image_axis, normal_from_axis
from pipe_twin.pooled_sections import (_observed_edges, _join_disjoint, _compatible, _possible_same_instance,
                                     aggregate_image, analyze_pooled_sections)
from pipe_twin.section_display import boundary_overlay_segments


def strip(degrees, *, gaps=False, neutral=False):
    axis = canonical_image_axis([math.cos(math.radians(degrees)), math.sin(math.radians(degrees))])
    normal = normal_from_axis(axis)
    center = np.array([610., 610.])
    yy, xx = np.indices((1220, 1220))
    s = (xx-center[0])*axis[0]+(yy-center[1])*axis[1]
    t = (xx-center[0])*normal[0]+(yy-center[1])*normal[1]
    coverage = np.clip(30.-abs(t)+.5, 0, 1)*np.clip(450.-abs(s)+.5, 0, 1)
    colour = np.array([230, 230, 230] if neutral else [235, 105, 15])
    image = np.clip(90.+coverage[:, :, None]*(colour-90.), 0, 255).astype(np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), .55)
    mask = ((abs(s) <= 450) & (abs(t) <= 30)).astype(np.uint8)
    rows, cols = np.nonzero(mask)
    x, y, w, h = cols.min(), rows.min(), np.ptp(cols)+1, np.ptp(rows)+1
    proposal = {'region_px': [int(x), int(y), int(w), int(h)], 'axis_image': axis.tolist(),
        'mask': mask[y:y+h, x:x+w], 'width_px': 60., 'length_px': 900.,
        'appearance': 'neutral' if neutral else 'hue-105', 'sampled_color_srgb': '#0F69EB'}
    selection = abs(s) < 420
    if gaps:
        image[(s > -80) & (s < -30)] = 90  # An actual missing/occluded interval.
        selection &= ~((s > 130) & (s < 170))
    return image, proposal, selection, center, axis, normal


@pytest.mark.parametrize('degrees', [0, 25, -35, 60, 89.99, 90, 110])
@pytest.mark.parametrize('neutral', [False, True])
def test_arbitrary_axis_measures_transverse_width_and_all_axial_stations(degrees, neutral):
    image, proposal, selection, center, axis, normal = strip(degrees, neutral=neutral)
    original = image.copy()
    item = _observed_edges(image, proposal, 'left', selection)
    assert not item['reason_codes']
    assert item['width_px'] == 60
    for edge, sign in zip(item['edges'], [-1, 1]):
        points = edge['points']
        assert len(points) > 800  # Image width is irrelevant at vertical orientation.
        assert np.ptp(points@axis) > 810
        # Neutral V>=100 is deliberately not a calibrated subpixel edge: with
        # blur on a 90-gray background it expands the visible band. Keep that
        # bounded threshold bias visible rather than asserting metric accuracy.
        assert np.median((points-center)@normal) == pytest.approx(sign*30, abs=1.5 if neutral else 1.)
        assert np.std((points-center)@normal) < .65
        u, v = np.rint(points).astype(int).T
        assert selection[v, u].all()
    audit, pooled = aggregate_boundaries(item['edges'])
    assert audit['stable_boundary_count'] == 2
    assert all(e['holdout_p95_px'] <= 2 for e in audit['boundaries'])
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize('degrees', [0, 38, 90, 112])
def test_occlusion_and_selection_holes_stay_empty_in_points_and_original_image_overlay(degrees):
    image, proposal, selection, center, axis, normal = strip(degrees, gaps=True)
    item = _observed_edges(image, proposal, 'left', selection)
    audit, pooled = aggregate_boundaries(item['edges'])
    assert audit['stable_boundary_count'] == 2
    for edge in pooled:
        station = (edge['points']-center)@axis
        assert not np.any((station > -78) & (station < -32))
        assert not np.any((station > 131) & (station < 169))
    report = {'section_evaluation': {'views': {'left': [{'aggregation': audit}]},
                                    'zone_offsets': {'left_x': 17, 'left_y': 23}}}
    segments = boundary_overlay_segments(report, 'left')
    assert len(segments) >= 6
    for x0, y0, x1, y1 in segments:
        pixels = np.linspace([x0-17, y0-23], [x1-17, y1-23], 50)
        station = (pixels-center)@axis
        assert not np.any((station > -78) & (station < -32))
        assert not np.any((station > 131) & (station < 169))
        assert np.std((pixels-center)@normal) < 1.


def test_vertical_transverse_cut_never_becomes_a_missing_outer_contour():
    image, proposal, selection, center, axis, normal = strip(90)
    yy, xx = np.indices(selection.shape)
    transverse = (xx-center[0])*normal[0]+(yy-center[1])*normal[1]
    selection &= transverse <= 10
    item = _observed_edges(image, proposal, 'left', selection)
    assert all(len(edge['points']) == 0 for edge in item['edges'])


def test_vertical_proposal_and_disjoint_join_keep_gap_and_transverse_separation():
    image, proposal, selection, center, axis, normal = strip(90, gaps=True)
    items, _ = aggregate_image(image, 'left', sampling_mask=selection, axis_hint_image=axis)
    assert items and any(item['aggregation']['stable_boundary_count'] == 2 for item in items)
    assert all('AXIAL_IMAGE_PARAMETERIZATION_UNSUPPORTED' not in i['reason_codes'] for i in items)
    # Explicit pieces ensure no dependence on the image proposal's split choice.
    full = _observed_edges(image, proposal, 'left', selection)
    pieces = []
    for low, high in [(-410, -90), (-20, 410)]:
        item = copy.deepcopy(full)
        for edge in item['edges']:
            s = (edge['points']-center)@axis
            edge['points'] = edge['points'][(s >= low) & (s <= high)]
        pieces.append(item)
    joined = _join_disjoint(pieces)
    assert len(joined) == 1
    for edge in joined[0]['edges']:
        s = (edge['points']-center)@axis
        assert not np.any((s > -90) & (s < -20))


def test_vertical_stereo_pairing_allows_horizontal_disparity_without_mixing_geometry():
    image, proposal, selection, _, axis, _ = strip(90)
    first = _observed_edges(image, proposal, 'left', selection)
    first['aggregation'], first['pooled_edges'] = aggregate_boundaries(first['edges'])
    second = copy.deepcopy(first)
    second['role'] = 'right'
    second['region_px'][0] -= 90
    for edge in second['edges']:
        edge.update(role='right', points=edge['points']-[90., 0.])
    second['aggregation'], second['pooled_edges'] = aggregate_boundaries(second['edges'])
    assert _compatible(first, second)
    # A negative disparity outside the normal search direction remains rejected.
    for edge in second['edges']:
        edge['points'] += [190., 0.]
    second['aggregation'], second['pooled_edges'] = aggregate_boundaries(second['edges'])
    assert not _compatible(first, second)


def test_crossing_same_colour_axes_do_not_merge_or_abort_with_unobservable_station():
    items = []
    for axis, origin in [(np.array([1., 0.]), [0, 200]), (np.array([0., 1.]), [500, 0])]:
        normal = normal_from_axis(axis)
        s = np.arange(384)
        edges = [{'role': 'left', 'edge': edge, 'image_axis': axis.tolist(),
                  'points': np.asarray(origin)+s[:, None]*axis+offset*normal}
                 for edge, offset in [('top', 0), ('bottom', 40)]]
        audit, pooled = aggregate_boundaries(edges)
        items.append({'image_axis': axis.tolist(), 'appearance': 'hue-105', 'width_px': 40.,
                      'region_px': [*origin, 384, 40], 'edges': edges, 'aggregation': audit,
                      'pooled_edges': pooled, 'reason_codes': []})
    assert not _possible_same_instance(*items)
    assert not _compatible(*items)
    assert len(_join_disjoint(items)) == 2


def test_vertical_pipe_clipped_across_width_keeps_diagnostic_without_crashing():
    image, _, _, _, _, _ = strip(90)
    cropped = image[:, 565:615].copy()
    k = np.array([[800., 0., 10.], [0., 800., 610.], [0., 0., 1.]])
    calibration = SimpleNamespace(left=SimpleNamespace(rectified_intrinsic=k),
                                  right=SimpleNamespace(rectified_intrinsic=k), baseline_mm=60.)
    result = analyze_pooled_sections(cropped, cropped, calibration)
    assert any('TRANSVERSE_BOUNDARY_CLIPPED' in item['reason_codes'] for item in result['views']['left'])
    assert all(c.get('dimension_status') == 'UNRESOLVED' for c in result['candidates'])


@pytest.mark.parametrize('degrees', [0, -35, 60, 90])
def test_reverse_mouse_axis_does_not_flip_visible_edge_identity(degrees):
    image, proposal, selection, _, _, _ = strip(degrees)
    positive = _observed_edges(image, proposal, 'left', selection)
    proposal['axis_image'] = (-np.asarray(proposal['axis_image'])).tolist()
    negative = _observed_edges(image, proposal, 'left', selection)
    for a, b in zip(positive['edges'], negative['edges']):
        assert a['edge'] == b['edge']
        np.testing.assert_allclose(a['points'], b['points'])
