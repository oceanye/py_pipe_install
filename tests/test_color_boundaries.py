"""Independent image-space checks of colour-supported contour localization."""
import copy

import cv2
import numpy as np
import pytest

import pipe_twin.pooled_sections as module
from pipe_twin.axial_aggregation import aggregate_boundaries


def painted_strip(colour=(235, 105, 15), *, noise=8., highlight=False, bent=False):
    height, width = 250, 1024
    rows, columns = np.indices((height, width))
    top = 105-.065*columns
    if bent:
        top = top+9*np.sin(columns/90.)
    bottom = top+60
    coverage = np.clip(np.minimum(rows-top, bottom-rows)+.5, 0, 1)
    image = 95.+coverage[:, :, None]*(np.asarray(colour)-95.)
    if highlight:
        image[np.abs(rows-(top+28)) < 5] = 248.
    image = cv2.GaussianBlur(image.astype(np.float32), (5, 5), .8)
    image += np.random.default_rng(9021).normal(0, noise, image.shape)
    image = np.clip(image, 0, 255).astype(np.uint8)
    mask = ((rows >= top) & (rows <= bottom)).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    y0, y1 = int(ys.min()), int(ys.max())+1
    hue = int(cv2.cvtColor(np.array([[colour]], np.uint8), cv2.COLOR_BGR2HSV)[0, 0, 0])
    hue = 0 if hue < 15 or hue > 165 else 105
    proposal = {'region_px': [0, y0, width, y1-y0], 'appearance': f'hue-{hue}',
        'mask': mask[y0:y1].copy(), 'width_px': 60., 'axis_image': [1., -.065],
        'sampled_color_srgb': '#%02X%02X%02X' % tuple(colour[::-1])}
    return image, proposal


@pytest.mark.parametrize('colour', [(235, 105, 15), (20, 40, 235)])
def test_noisy_painted_edges_gain_precision_without_setting_the_width(monkeypatch, colour):
    image, proposal = painted_strip(colour)
    original = image.copy()
    refined = module._observed_edges(image, proposal, 'left')
    monkeypatch.setattr(module, '_colour_gradient', lambda *args: None)
    threshold = module._observed_edges(image, proposal, 'left')
    np.testing.assert_array_equal(image, original)
    for edge, truth in [('top', 105.), ('bottom', 165.)]:
        a = next(e['points'] for e in refined['edges'] if e['edge'] == edge)
        b = next(e['points'] for e in threshold['edges'] if e['edge'] == edge)
        np.testing.assert_array_equal(a[:, 0], b[:, 0])
        true_y = truth-.065*a[:, 0]
        assert np.std(a[:, 1]-true_y) < .8*np.std(b[:, 1]-true_y)
        assert abs(np.median(a[:, 1]-true_y)) < 1.
        assert np.max(np.abs(a[:, 1]-b[:, 1])) <= 3.
    top, bottom = [e['points'] for e in refined['edges']]
    assert np.median(bottom[:, 1]-top[:, 1]) == pytest.approx(60., abs=1.2)


def test_bright_interior_hole_does_not_move_outer_colour_contours():
    base, proposal = painted_strip(highlight=False)
    bright, _ = painted_strip(highlight=True)
    first = module._observed_edges(base, proposal, 'left')
    second = module._observed_edges(bright, proposal, 'left')
    for a, b in zip(first['edges'], second['edges']):
        np.testing.assert_allclose(a['points'], b['points'], atol=1e-5)
    assert second['highlight_band_count'] == 1


def test_true_occluder_and_sampling_cut_never_gain_boundary_samples():
    image, proposal = painted_strip(highlight=True)
    image[:, 205:235] = 245  # Physical neutral occluder across the whole pipe.
    mask = np.ones(image.shape[:2], bool)
    mask[:, 550:590] = False
    item = module._observed_edges(image, proposal, 'left', mask)
    for edge in item['edges']:
        x = edge['points'][:, 0]
        assert not np.any((205 <= x) & (x < 235))
        assert not np.any((550 <= x) & (x < 590))
    audit, pooled = aggregate_boundaries(item['edges'])
    for edge in audit['boundaries']:
        assert all(high <= 205 or low >= 235 for low, high in edge['observed_column_runs_px'])
        assert all(high <= 550 or low >= 590 for low, high in edge['observed_column_runs_px'])


def test_colour_filter_does_not_turn_a_bent_outline_into_a_straight_cylinder():
    image, proposal = painted_strip(bent=True, noise=2.)
    item = module._observed_edges(image, proposal, 'left')
    audit, pooled = aggregate_boundaries(item['edges'])
    assert audit['stable_boundary_count'] == 0
    assert pooled == []


def test_neutral_surface_retains_original_pixel_boundary():
    image, proposal = painted_strip(colour=(230, 230, 230), noise=0.)
    proposal = copy.deepcopy(proposal)
    proposal['appearance'] = 'neutral'
    item = module._observed_edges(image, proposal, 'left')
    assert item['edge_extraction']['source'] == 'ORIGINAL_NEUTRAL_THRESHOLD'
    assert all(e['refined_columns'] == 0 for e in item['edge_extraction']['edges'].values())
