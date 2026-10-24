from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from pipe_twin.axial_selection import (
    normalize_axial_band, band_geometry, band_overlay, move_band_limit, band_sampling_masks,
)
from pipe_twin.image_regions import propose_regions
from pipe_twin.pooled_sections import _observed_edges, analyze_pooled_sections


def selection(points=((200, 240), (500, 240)), size=(640, 480)):
    return normalize_axial_band({'image_size': list(size), 'axis_segment_px': points})


@pytest.mark.parametrize('points', [((200, 240), (500, 240)), ((320, 80), (320, 400)),
                                    ((100, 100), (500, 350)), ((500, 350), (100, 100))])
def test_clipped_limits_are_perpendicular_and_strip_contains_only_between_them(points):
    band = selection(points)
    origin, axis, length = band_geometry(band)
    overlay = band_overlay(band)
    for station, line in zip((0, length), overlay['boundary_lines_px']):
        assert len(line) == 2
        np.testing.assert_allclose((np.asarray(line)-origin)@axis, station, atol=1e-8)
        assert abs((np.asarray(line[1])-line[0])@axis) < 1e-8
    stations = (np.asarray(overlay['polygon_px'])-origin)@axis
    assert stations.min() >= -1e-8 and stations.max() <= length+1e-8
    masks, audit = band_sampling_masks(band, (480, 640), (480, 640))
    yy, xx = np.indices((480, 640))
    reference = (xx-origin[0])*axis[0]+(yy-origin[1])*axis[1]
    np.testing.assert_array_equal(masks['left'], (reference >= -1e-8) & (reference <= length+1e-8))
    assert audit['supplies_metric_length_or_3d_axis'] is False


@pytest.mark.parametrize('changes', [
    {'axis_segment_px': [[0, 0], [10, 0]]}, {'axis_segment_px': [[-1, 0], [200, 0]]},
    {'axis_segment_px': [[0, 0], [640, 480]]}, {'axis_segment_px': [[0, 0], [float('nan'), 200]]},
    {'image_size': [640., 480]}, {'coordinate_space': 'raw_left'}, {'extra': True},
])
def test_bad_or_unbound_selection_rejected(changes):
    with pytest.raises(ValueError):
        normalize_axial_band(selection() | changes)


def test_selection_is_bound_to_original_resolution():
    with pytest.raises(ValueError, match='尺寸'):
        normalize_axial_band(selection(), image_size=(1280, 960))


@pytest.mark.parametrize('index,point', [(0, [-1000, 30]), (0, [900, 900]), (1, [-90, -40]), (1, [900, 900])])
def test_handles_move_along_fixed_axis_without_crossing_or_leaving_image(index, point):
    band = selection(((100, 100), (500, 300)))
    moved = move_band_limit(band, index, point)
    np.testing.assert_allclose(band_geometry(moved)[1], band_geometry(band)[1], atol=1e-8)
    assert band_geometry(moved)[2] >= 32-1e-9
    assert moved['axis_segment_px'][1-index] == band['axis_segment_px'][1-index]
    coordinates = np.asarray(moved['axis_segment_px'])
    assert (coordinates >= 0).all() and (coordinates <= [639, 479]).all()


@pytest.mark.parametrize('reverse', [False, True])
def test_right_envelope_contains_every_allowed_disparity_and_zone_masks_use_full_image_coordinates(reverse):
    points = [[200, 140], [500, 300]]
    band = selection(points[::-1] if reverse else points)
    full, audit = band_sampling_masks(band, (480, 640), (480, 640), disparity_range=(-16, 128), right_vertical_offset=4)
    rows, columns = np.nonzero(full['left'])
    for disparity in (-16, 0, 40, 128):
        ry, rx = rows-4, columns-disparity
        valid = (ry >= 0) & (ry < 480) & (rx >= 0) & (rx < 640)
        assert full['right'][ry[valid], rx[valid]].all()
    offsets = {'left_x': 180, 'left_y': 80, 'right_x': 60, 'right_y': 70}
    cropped, _ = band_sampling_masks(band, (250, 350), (260, 440), offsets=offsets,
                                    disparity_range=(-16, 128), right_vertical_offset=4)
    np.testing.assert_array_equal(cropped['left'], full['left'][80:330, 180:530])
    np.testing.assert_array_equal(cropped['right'], full['right'][70:330, 60:500])
    assert audit['right_scope'] == 'EPIPOLAR_SEARCH_ENVELOPE'


def test_manual_cuts_cannot_turn_into_observed_pipe_silhouettes():
    image = np.zeros((300, 800, 3), np.uint8)
    image[80:180, 10:790] = (0, 0, 220)
    band = selection(((250, 150), (600, 250)), (800, 300))
    mask = band_sampling_masks(band, image.shape, image.shape)[0]['left']
    proposal = {'region_px': [10, 80, 780, 100], 'appearance': 'hue-0', 'width_px': 100,
                'mask': np.ones((100, 780), np.uint8), 'axis_image': [1, 0], 'sampled_color_srgb': '#DC0000'}
    item = _observed_edges(image, proposal, 'left', mask)
    origin, axis, length = band_geometry(band)
    for edge, expected_y in zip(item['edges'], (79.5, 179.5)):
        assert len(edge['points']) > 200
        # Real pipe edges remain horizontal even though selection ends are oblique.
        np.testing.assert_allclose(edge['points'][:, 1], expected_y)
        station = (edge['points']-origin)@axis
        assert (station > 1).all() and (station < length-1).all()


def test_mask_separates_white_pipe_from_connected_boards_before_proposals():
    image = np.zeros((350, 1000, 3), np.uint8)
    image[150:190, 100:900] = 220
    image[40:310, 80:220] = 220
    image[40:310, 780:920] = 220
    original = image.copy()
    band = selection(((300, 170), (700, 170)), (1000, 350))
    mask = band_sampling_masks(band, image.shape, image.shape)[0]['left']
    full, _ = propose_regions(image)
    limited, _ = propose_regions(image, sampling_mask=mask, axis_hint_image=[1, 0])
    assert not any(p['appearance'] == 'neutral' and p['width_px'] < 50 for p in full)
    pipes = [p for p in limited if p['appearance'] == 'neutral']
    assert len(pipes) == 1 and pipes[0]['width_px'] == pytest.approx(39)
    assert pipes[0]['region_px'][0] >= 300 and sum(pipes[0]['region_px'][::2]) <= 701
    np.testing.assert_array_equal(image, original)


def test_pooled_stage_applies_masks_and_keeps_image_hint_out_of_3d_axis(monkeypatch):
    import pipe_twin.pooled_sections as module
    seen = {}
    def empty(image, role, **kwargs):
        seen[role] = kwargs
        return [], {'truncated': False}
    monkeypatch.setattr(module, 'aggregate_image', empty)
    axis_solver = Mock(side_effect=AssertionError('no 3-D evidence'))
    monkeypatch.setattr(module, 'axis_from_image_lines', axis_solver)
    kl = np.array([[500., 0, 140], [0, 500, 160], [0, 0, 1]])
    kr = kl.copy(); kr[0, 2] += 120; kr[1, 2] += 10
    cal = SimpleNamespace(left=SimpleNamespace(rectified_intrinsic=kl), right=SimpleNamespace(rectified_intrinsic=kr), baseline_mm=60.)
    band = selection(((200, 140), (500, 300)))
    offsets = {'left_x': 180, 'left_y': 80, 'right_x': 60, 'right_y': 70}
    result = analyze_pooled_sections(np.zeros((250, 350, 3), np.uint8), np.zeros((260, 440, 3), np.uint8), cal,
                                    aggregation_band=band, image_offsets=offsets, disparity_range=(0, 96))
    full, _ = band_sampling_masks(band, (480, 640), (480, 640), disparity_range=(0, 96))
    np.testing.assert_array_equal(seen['left']['sampling_mask'], full['left'][80:330, 180:530])
    np.testing.assert_array_equal(seen['right']['sampling_mask'], full['right'][70:330, 60:500])
    np.testing.assert_allclose(seen['left']['axis_hint_image'], band_geometry(band)[1])
    assert result['status'] == 'NO_STABLE_STEREO_BOUNDARIES' and result['model'] is None
    axis_solver.assert_not_called()


@pytest.mark.parametrize('use_zone', [False, True])
def test_production_entry_forwards_band_matcher_search_and_zone_offsets(monkeypatch, use_zone):
    from pipe_twin.elevation_auto import analyze_elevation_auto_groups, analyze_elevation_auto_zones
    from test_elevation_auto import _groups
    calibration, specs, groups = _groups()
    band = selection()
    groups[-1]['depth'].audit['matcher'] = {'min_disparity': -16, 'num_disparities': 128}
    class ReachedPooling(Exception):
        pass
    def inspect(left, right, cal, **kwargs):
        assert kwargs['aggregation_band'] == band
        assert kwargs['disparity_range'] == (-16, 111)
        if use_zone:
            assert kwargs['image_offsets']['left_x'] == 80
            assert kwargs['image_offsets']['right_x'] == 16
            assert left.shape[:2] == (300, 400)
            assert right.shape[:2] == (300, 528)
        else:
            assert kwargs['image_offsets'] is None
        raise ReachedPooling()
    monkeypatch.setattr('pipe_twin.pooled_sections.analyze_pooled_sections', inspect)
    with pytest.raises(ReachedPooling):
        kwargs = dict(calibration=calibration, pipe_specs=specs, registration_settings={'aggregation_band': band})
        if use_zone:
            analyze_elevation_auto_zones(groups, **kwargs, scope_settings={'scope': 'zones', 'right_roi_padding_px': 64,
                'zones': [{'zone_id': 'Z01', 'label': 'test', 'enabled': True, 'coordinate_space': 'rectified_left',
                           'roi_rect_px': [80, 60, 400, 300]}]})
        else:
            analyze_elevation_auto_groups(groups, **kwargs)
