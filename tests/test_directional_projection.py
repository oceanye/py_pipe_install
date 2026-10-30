"""Analytic physical tangent generators, independent of projection fitter code."""
import copy
import math

import cv2
import numpy as np
import pytest

from pipe_twin.image_line_geometry import (
    canonical_image_axis, fit_image_line, normal_from_axis, point_at_station, sampling_coordinates,
)
from pipe_twin.projection_diagnostics import fit_silhouettes, pixel_error_scenarios
from pipe_twin.axial_consistency import synchronize_axial_blocks, conditional_axial_stations
from pipe_twin.ground_correction import correct_pipe_axis
from pipe_twin.section_diagnostics import axis_from_image_lines


def analytic_samples(degrees, *, axial_z=.18, baseline=60., radius=20., count=1000):
    """Construct real points whose radial normal is perpendicular to each sight ray.

    Tangency is solved in 3D as q·radial=-r²; projectPoints performs the
    independent image projection. No fitted line, section basis or forward
    projection helper supplies the truth.
    """
    angle = math.radians(degrees)
    axis = np.array([math.cos(angle), math.sin(angle), axial_z])
    axis /= np.linalg.norm(axis)
    center = np.array([0., 0., 1500.])
    k = np.array([[1350., 0., 960.], [0., 1200., 540.], [0., 0., 1.]])
    image_axis = np.array([k[0, 0]*axis[0], k[1, 1]*axis[1]])
    image_axis /= np.linalg.norm(image_axis)
    if image_axis[np.argmax(abs(image_axis))] < 0:
        image_axis = -image_axis
    image_normal = np.array([-image_axis[1], image_axis[0]])
    observations, true_stations = [], []
    for role, camera in [('left', np.zeros(3)), ('right', np.array([baseline, 0., 0.]))]:
        transverse = center-camera-axis*((center-camera)@axis)
        distance = np.linalg.norm(transverse)
        toward_axis = transverse/distance
        side = np.cross(axis, toward_axis)
        tangent_radials = [-(radius**2/distance)*toward_axis
            + sign*radius*math.sqrt(1-(radius/distance)**2)*side for sign in (-1, 1)]
        rows = []
        for radial in tangent_radials:
            # Unequal per-eye intervals exercise actual common axial station alignment.
            t = np.linspace(-510 if role == 'left' else -460, 510 if role == 'left' else 540, count)
            points = center+radial+t[:, None]*axis
            sight = points-camera
            np.testing.assert_allclose(sight@radial, 0., atol=1e-8)
            pixels = cv2.projectPoints(points, np.zeros(3), -camera, k, None)[0].reshape(-1, 2)
            rows.append((float(pixels.mean(axis=0)@image_normal), pixels, points@axis))
        for edge, (_, pixels, stations) in zip(('top', 'bottom'), sorted(rows, key=lambda row: row[0])):
            observations.append({'role': role, 'edge': edge, 'points': pixels, 'image_axis': image_axis.tolist()})
            true_stations.append(stations)
    return observations, k, axis, center, image_axis, true_stations


@pytest.mark.parametrize('degrees', [0., 30., 60., 85., 90., -30., -60., 135.])
@pytest.mark.parametrize('missing', [None, 0, 1, 2, 3])
def test_physical_tangent_generators_recover_free_radius_at_any_image_direction(degrees, missing):
    observations, k, axis, center, image_axis, _ = analytic_samples(degrees)
    original = copy.deepcopy(observations)
    if missing is not None:
        observations[missing]['boundary_kind'] = 'occlusion'
        observations[missing]['points'] += 100*normal_from_axis(image_axis)
    fit = fit_silhouettes(observations, k, 60., axis, reference_pixel=k[:2, 2], image_axis=image_axis)
    assert fit['status'] == 'CONDITIONAL_FIT'
    assert fit['trial_diameter_mm'] == pytest.approx(40., abs=1e-5)
    np.testing.assert_allclose(fit['center_at_reference_pixel_mm'], center, atol=1e-4)
    assert fit['reprojection_rms_px'] < 1e-7
    assert fit['reference_definition'] == 'IMAGE_TRANSVERSE_LINE'
    assert fit['physical_baseline_mm'] == 60.
    assert fit['observed_boundary_count'] == (4 if missing is None else 3)
    for before, after in zip(original, observations):
        if after.get('boundary_kind') != 'occlusion':
            np.testing.assert_array_equal(before['points'], after['points'])


@pytest.mark.parametrize('degrees', [0., 60., 90.])
def test_axis_sign_does_not_change_metric_solution_or_image_side_labels(degrees):
    obs, k, axis, center, image_axis, _ = analytic_samples(degrees)
    for direction, image_direction in [(axis, image_axis), (-axis, -image_axis)]:
        fit = fit_silhouettes(obs, k, 60., direction, image_axis=image_direction, reference_pixel=k[:2, 2])
        assert fit['trial_diameter_mm'] == pytest.approx(40., abs=1e-5)
        np.testing.assert_allclose(fit['center_at_reference_pixel_mm'], center, atol=1e-4)


@pytest.mark.parametrize('degrees', [30., 85., 90.])
def test_asymmetric_crops_restore_shared_coordinates_without_rotating_original_images(degrees):
    obs, k, axis, center, image_axis, _ = analytic_samples(degrees)
    left_crop = np.array([207., 51.]); right_crop = np.array([154., 32.])
    cropped = []
    for item in obs:
        origin = left_crop if item['role'] == 'left' else right_crop
        points = item['points']-origin
        if item['role'] == 'right':
            points += right_crop-left_crop
        cropped.append({**item, 'points': points})
    cropped_k = k.copy(); cropped_k[:2, 2] -= left_crop
    fit = fit_silhouettes(cropped, cropped_k, 60., axis,
                          image_axis=image_axis, reference_pixel=cropped_k[:2, 2])
    assert fit['status'] == 'CONDITIONAL_FIT'
    assert fit['trial_diameter_mm'] == pytest.approx(40., abs=1e-5)
    np.testing.assert_allclose(fit['center_at_reference_pixel_mm'], center, atol=1e-4)


def test_exact_vertical_edges_have_horizontal_noise_and_finite_nonzero_uncertainty():
    obs, k, axis, _, image_axis, _ = analytic_samples(90., axial_z=0., count=80)
    value = pixel_error_scenarios(obs, k, 60., axis, image_axis=image_axis, reference_pixel=k[:2, 2])
    np.testing.assert_allclose(value['pixel_noise_direction_image'], [-1., 0.], atol=1e-14)
    assert value['assumed_normal_pixel_sigma_px'] == .5
    for scenario in value['scenarios'].values():
        assert np.isfinite(scenario['diameter_sigma_mm']) and scenario['diameter_sigma_mm'] > .001
        assert np.isfinite(scenario['center_z_sigma_mm']) and scenario['center_z_sigma_mm'] > .01
    assert not value['total_accuracy_known']


@pytest.mark.parametrize('degrees', [30., 85., 90.])
def test_common_station_partition_works_when_x_is_constant_or_nearly_constant(degrees):
    obs, k, axis, center, image_axis, truth = analytic_samples(degrees, axial_z=0.)
    fit = fit_silhouettes(obs, k, 60., axis, image_axis=image_axis, reference_pixel=k[:2, 2])
    for item, stations in zip(obs, truth):
        np.testing.assert_allclose(conditional_axial_stations(item, k, 60., fit), stations, atol=1e-6)
    common, halves, audit = synchronize_axial_blocks(obs, k, 60., fit)
    assert audit['status'] == 'COMMON_AXIAL_SUPPORT'
    assert audit['accepted_windows'] >= 3
    for first, second in zip(halves[0], halves[1]):
        assert not set(map(tuple, first['points'])) & set(map(tuple, second['points']))
    for half in halves:
        recovered = fit_silhouettes(half, k, 60., axis, image_axis=image_axis, reference_pixel=k[:2, 2])
        assert recovered['trial_diameter_mm'] == pytest.approx(40., abs=1e-5)
        np.testing.assert_allclose(recovered['center_at_reference_pixel_mm'], center, atol=1e-4)


@pytest.mark.parametrize('degrees', [0., 60., 85., 90.])
def test_ground_axis_constraint_uses_homogeneous_lines_without_horizontal_span_requirement(degrees):
    obs, k, axis, _, _, _ = analytic_samples(degrees)
    lines = [fit_image_line(o['points']) for o in obs]
    free = axis_from_image_lines(lines, k) | {'status': 'STABLE'}
    ground_normal = np.cross(axis, [.2, -.3, 1.]); ground_normal /= np.linalg.norm(ground_normal)
    ground = {'status': 'CONSISTENT_GROUND_PLANE', 'surface_identity_source': 'OPERATOR_MARKED_GROUND',
              'left_plane': {'normal_camera': ground_normal.tolist()}}
    corrected, audit = correct_pipe_axis(obs, k, free, ground)
    assert audit['applied'] and audit['boundary_check_passed']
    assert abs(np.dot(corrected['axis'], axis)) > 1-1e-10
    assert abs(np.dot(corrected['axis'], ground_normal)) < 1e-10
    assert max(item['p95_px'] for item in audit['boundary_residuals']) < 1e-7


def test_true_parallel_baseline_still_rejected_even_with_many_exact_outline_points():
    obs, k, axis, _, image_axis, _ = analytic_samples(0., axial_z=0.)
    fit = fit_silhouettes(obs, k, 60., axis, image_axis=image_axis, reference_pixel=k[:2, 2])
    assert fit['status'] == 'UNOBSERVABLE_PARALLEL_BASELINE'
    assert fit['physical_baseline_mm'] == 60.
    assert 'trial_diameter_mm' not in fit


def test_vertical_tls_and_sampling_round_trip_and_canonical_sign():
    points = np.column_stack((np.full(100, 42.), np.linspace(10, 220, 100)))
    axis = canonical_image_axis([0., -1.])
    np.testing.assert_array_equal(axis, [0., 1.])
    line = fit_image_line(points, axis)
    np.testing.assert_allclose(points@line[:2]+line[2], 0., atol=1e-12)
    coords = sampling_coordinates(points, axis)
    recovered = np.array([point_at_station(line, axis, s) for s in coords[:, 0]])
    np.testing.assert_allclose(recovered, points)
