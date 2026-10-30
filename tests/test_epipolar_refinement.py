import cv2
import numpy as np
import pytest

from pipe_twin.epipolar_refinement import estimate_rotation, transform_points, refine_right_image


def observations(pitch=.012, roll=-.1, noise=.12):
    rng = np.random.default_rng(174)
    k = np.array([[4400., 0, 960], [0, 4400., 540], [0, 0, 1]])
    left = rng.uniform([300, 100], [1750, 950], (700, 2))
    right = left - np.c_[rng.uniform(70, 170, len(left)), np.zeros(len(left))]
    r = cv2.Rodrigues(np.radians([pitch, 0., roll]))[0]
    h = k@r@np.linalg.inv(k)
    observed = transform_points(right, np.linalg.inv(h)) + rng.normal(0, noise, right.shape)
    return k, left, observed, h


def test_known_small_pitch_roll_recovered_without_changing_baseline_or_intrinsics():
    k, left, right, h = observations()
    result = estimate_rotation(left, right, k, (1920, 1080))
    assert result['applied']
    np.testing.assert_allclose(result['rotation_pitch_roll_deg'], [.012, -.1], atol=.003)
    corrected = transform_points(right, result['right_original_to_refined'])
    assert np.percentile(abs(corrected[:, 1]-left[:, 1]), 95) < .3
    assert not result['calibration_updated']
    assert not result['yaw_estimated']
    np.testing.assert_allclose(transform_points(corrected, result['right_refined_to_original']), right, atol=1.e-8)


def test_correct_rectification_is_not_warped_just_to_fit_noise():
    k, left, right, _ = observations(pitch=0, roll=0)
    assert not estimate_rotation(left, right, k, (1920, 1080))['applied']


def test_heldout_spatially_inconsistent_correction_is_rejected():
    k, left, right, _ = observations()
    cells = np.floor(left/[240, 180]).astype(int)
    test = (cells[:, 0]+cells[:, 1]) % 2 != 0
    right[test, 1] = left[test, 1]
    result = estimate_rotation(left, right, k, (1920, 1080))
    assert not result['applied']
    assert result['status'] == 'NO_VALIDATED_IMPROVEMENT'


def test_large_rotation_and_narrow_support_are_rejected():
    k, left, right, _ = observations(roll=.4)
    assert not estimate_rotation(left, right, k, (1920, 1080))['applied']
    assert not estimate_rotation(left[:30], right[:30], k, (1920, 1080))['applied']
    left[:, 1] = 100; left[:, 0] = np.linspace(800, 900, len(left))
    assert estimate_rotation(left, right, k, (1920, 1080))['status'] == 'INSUFFICIENT_SPATIAL_COVERAGE'


def test_textureless_images_keep_original_object_unchanged():
    image = np.full((480, 640, 3), 127, np.uint8)
    actual, audit = refine_right_image(image, image, np.eye(3))
    assert actual is image
    assert not audit['applied']


def test_refined_boundary_overlay_is_mapped_back_to_original_image():
    from pipe_twin.section_display import boundary_overlay_segments
    h = np.array([[1., 0, 2], [0, 1., -3], [0, 0, 1.]])
    stage = {'epipolar_refinement': {'applied': True, 'right_refined_to_original': h.tolist()},
             'views': {'right': [{'aggregation': {'boundaries': [{'status': 'STABLE',
                 'retained_column_runs_px': [[10, 51]],
                 'line_slope_intercept': [.1, 20], 'windows': [{'accepted': True, 'axial_interval_px': [10, 51]}]}]}}]}}
    assert boundary_overlay_segments({'section_evaluation': stage}, 'right') == [(12., 18., 52., 22.)]
