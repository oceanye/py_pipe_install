"""Independent cylinder projections: tilt, shared stations and mouse error."""
import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pipe_twin.image_regions import propose_regions
from pipe_twin.projection_diagnostics import fit_silhouettes
from pipe_twin.section_diagnostics import section_basis
from pipe_twin.pooled_sections import _compatible
from test_axial_aggregation import synthetic_views


def projected_samples(axis, radius=20.):
    """Find silhouette generators on a dense surface, then project real stations."""
    axis = np.array(axis, float); axis /= np.linalg.norm(axis)
    center = np.array([100., 80., 1800.])
    u = np.cross(axis, [0., 0., 1.]); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    theta = np.linspace(0, 2*np.pi, 12000, endpoint=False)
    ring = radius*(np.cos(theta)[:, None]*u+np.sin(theta)[:, None]*v)
    cloud = np.vstack([ring-600*axis, ring+600*axis])+center
    k = np.array([[1200., 0, 960], [0, 1200, 540], [0, 0, 1]])
    result, truth = [], []
    intervals = [(-440, 440), (-430, 450), (-570, 340), (-550, 350)]
    for role, camera in [('left', np.zeros(3)), ('right', np.array([60., 0., 0.]))]:
        pixels = cv2.projectPoints(cloud, np.zeros(3), -camera, k, None)[0].reshape(-1, 2)
        hull = cv2.convexHull(pixels.astype(np.float32), returnPoints=False).ravel()
        next_indices = np.roll(hull, -1)
        longest = np.argsort(np.linalg.norm(pixels[hull]-pixels[next_indices], axis=1))[-2:]
        longest = sorted(longest, key=lambda i: (pixels[hull[i], 1]+pixels[next_indices[i], 1])/2)
        for edge, index in zip(('top', 'bottom'), longest):
            point = cloud[hull[index]]
            transverse = point-center-axis*((point-center)@axis)
            relative = np.linspace(*intervals[len(result)], 1000)
            actual = center+transverse+relative[:, None]*axis
            samples = cv2.projectPoints(actual, np.zeros(3), -camera, k, None)[0].reshape(-1, 2)
            result.append({'role': role, 'edge': edge, 'points': samples})
            truth.append(actual@axis)
    return result, truth, k, center, axis


@pytest.mark.parametrize('angle,z', [(-7., 0.), (25., 0.), (-40., 0.), (-7., .2)])
def test_tilt_and_different_axial_reference_positions_do_not_change_section(angle, z):
    a = [math.cos(math.radians(angle)), math.sin(math.radians(angle)), z]
    obs, _, k, center, axis = projected_samples(a)
    results = [fit_silhouettes(obs, k, 60., axis, ref) for ref in (750., 1000., 1250.)]
    for result in results:
        assert result['status'] == 'CONDITIONAL_FIT'
        assert result['trial_diameter_mm'] == pytest.approx(40., abs=.02)
        np.testing.assert_allclose(result['section_center_parameters_mm'], section_basis(axis).T@center, atol=.2)
    # Positions along the physical axis differ, but their common section is invariant.
    assert np.linalg.norm(np.array(results[0]['center_at_reference_column_mm'])-results[-1]['center_at_reference_column_mm']) > 300
    np.testing.assert_allclose(results[0]['section_center_parameters_mm'], results[-1]['section_center_parameters_mm'], atol=1e-5)


@pytest.mark.parametrize('angle', [-7., 25., -40.])
def test_shared_stations_recover_material_positions_and_pair_disjoint_blocks(angle):
    from pipe_twin.axial_consistency import conditional_axial_stations, synchronize_axial_blocks
    obs, truth, k, center, axis = projected_samples([math.cos(math.radians(angle)), math.sin(math.radians(angle)), .1])
    fit = fit_silhouettes(obs, k, 60., axis, 1000.)
    for o, expected in zip(obs, truth):
        station = conditional_axial_stations(o, k, 60., fit)
        np.testing.assert_allclose(station, expected, atol=.2)
    shared, halves, audit = synchronize_axial_blocks(obs, k, 60., fit)
    assert audit['status'] == 'COMMON_AXIAL_SUPPORT'
    assert audit['accepted_windows'] >= 3
    common_low = max(t.min() for t in truth); common_high = min(t.max() for t in truth)
    np.testing.assert_allclose(audit['common_interval_mm'], [common_low, common_high], atol=.2)
    for i, source in enumerate(obs):
        # Every retained pixel is one of the original measurements, not an interpolation.
        originals = {tuple(p) for p in source['points']}
        assert set(map(tuple, shared[i]['points'])) <= originals
        assert not set(map(tuple, halves[0][i]['points'])) & set(map(tuple, halves[1][i]['points']))
        for parity in (0, 1):
            locations = conditional_axial_stations(halves[parity][i], k, 60., fit)
            assigned = np.clip(np.searchsorted(audit['window_edges_mm'], locations, side='right')-1, 0, audit['total_windows']-1)
            assert np.all(assigned % 2 == parity)
    for half in halves:
        recovered = fit_silhouettes(half, k, 60., axis, 1000.)
        assert recovered['trial_diameter_mm'] == pytest.approx(40., abs=.02)
        np.testing.assert_allclose(recovered['section_center_parameters_mm'], section_basis(axis).T@center, atol=.2)


def test_short_or_nonoverlapping_stations_cannot_certify_repeatability():
    from pipe_twin.axial_consistency import synchronize_axial_blocks
    obs, _, k, _, axis = projected_samples([1., -.2, .1])
    fit = fit_silhouettes(obs, k, 60., axis, 1000.)
    for o in obs:
        o['points'] = o['points'][:50] if o['role'] == 'left' else o['points'][-50:]
    shared, halves, audit = synchronize_axial_blocks(obs, k, 60., fit)
    assert not shared and not halves and audit['status'] == 'NO_COMMON_AXIAL_SUPPORT'


@pytest.mark.parametrize('error_deg', [-4., 0., 4.])
def test_mouse_hint_does_not_override_observed_pipe_direction(error_deg):
    image = np.full((280, 900, 3), 40, np.uint8)
    cv2.fillConvexPoly(image, np.array([[60, 140], [840, 40], [840, 105], [60, 205]], np.int32), (20, 40, 230))
    actual = math.atan2(-100, 780)
    hint = actual+math.radians(error_deg)
    proposals, audit = propose_regions(image, axis_hint_image=[math.cos(hint), math.sin(hint)])
    red = next(p for p in proposals if p['appearance'] == 'hue-0')
    assert red['width_px'] == pytest.approx(64.5, abs=1.5)
    direction = next(d for d in audit['direction_estimates'] if d['appearance'] == 'hue-0')
    assert direction['source'] == 'OBSERVED_NEAR_HINT'
    assert abs(direction['observed_angle_deg']-math.degrees(actual)) < .5


def test_small_bright_patch_between_red_pipes_is_not_a_continuous_highlight():
    image = np.full((260, 800, 3), 40, np.uint8)
    image[50:110, 30:770] = (20, 30, 230)
    image[124:184, 30:770] = (20, 30, 230)
    image[110:124, 100:180] = 240  # A short bright support, not a pipe highlight.
    proposals, _ = propose_regions(image, axis_hint_image=[1., 0.])
    red = [p for p in proposals if p['appearance'] == 'hue-0']
    assert len(red) == 2
    assert all(p['width_px'] < 65 for p in red)


def test_adjacent_hue_labels_can_pair_but_distinct_colours_cannot():
    views, _ = synthetic_views()
    a, b = views['left'][0], views['right'][0]
    a['appearance'] = 'hue-120'; b['appearance'] = 'hue-105'
    assert _compatible(a, b)
    b['color_srgb'] = '#00DD44'; b['appearance'] = 'hue-60'
    assert not _compatible(a, b)
