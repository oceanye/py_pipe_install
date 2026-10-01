"""Known disparity, independent sensor noise, and fail-closed low-light tests."""
from dataclasses import replace

import cv2
import numpy as np
import pytest

from pipe_twin.stereo_analyzer import (
    CameraCalibration, StereoCalibration, StereoAnalysisError, _analysis_config,
    _compute_stereo_depth, _quality,
)


def noisy_plane():
    rng = np.random.default_rng(419)
    width, height, disparity = 640, 320, 24
    texture = cv2.GaussianBlur(rng.integers(0, 256, (height, width + disparity), np.uint8), (5, 5), 1.0)
    images = {}
    for role, start, gain, offset in (("left", 0, .25, 3), ("right", disparity, .35, 8)):
        gray = np.clip(texture[:, start:start + width] * gain + offset + rng.normal(0, 2.5, (height, width)), 0, 255).astype(np.uint8)
        images[role] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    K = np.array([[500., 0, width / 2], [0, 500., height / 2], [0, 0, 1]])
    left = CameraCalibration("left", "left", width, height, K, np.zeros(5), np.eye(3), np.zeros(3), None, None)
    right = replace(left, role="right", camera_id="right", center_world_mm=np.array([60., 0, 0]))
    return images, StereoCalibration("NOISY-PLANE", True, False, True, 60., 1., left, right), disparity


def test_low_light_reconstruction_has_known_scale_and_preserves_source():
    images, calibration, disparity = noisy_plane()
    original = {r: image.copy() for r, image in images.items()}
    config = _analysis_config({"stereo_matching": {"num_disparities": 64, "preprocessing": "low_light"}})
    result = _compute_stereo_depth(images["left"], images["right"], calibration, config)
    assert result.audit["algorithm_revision"] == "sgbm-low-light-v1"
    assert result.audit["matcher"]["preprocessing"] == "low_light"
    for role in ("left", "right"):
        assert np.array_equal(images[role], original[role])
        valid = getattr(result, role + "_valid")[20:-20, 90:-90]
        measured = getattr(result, role + "_depth_mm")[20:-20, 90:-90]
        assert valid.mean() > .7
        recovered = calibration.left.fx * calibration.baseline_mm / measured[valid]
        assert np.percentile(np.abs(recovered - disparity), 90) < 1.0


@pytest.mark.parametrize("value", [0, 8, 255])
def test_flat_images_do_not_acquire_depth_from_enhancement(value):
    images, calibration, _ = noisy_plane()
    for image in images.values():
        image[:] = value
    config = _analysis_config({"stereo_matching": {"num_disparities": 64, "preprocessing": "low_light"}})
    result = _compute_stereo_depth(images["left"], images["right"], calibration, config)
    assert result.audit["status"] == "INVALID"
    assert not result.left_valid.any() and not result.right_valid.any()
    assert np.isnan(result.left_depth_mm).all()


def test_signal_diagnostics_detect_dark_roi_despite_bright_board():
    image = np.full((200, 300, 3), 8, np.uint8)
    image[30:160, 90:180] = 170
    config = _analysis_config({})
    full = _quality(image, config)
    roi = _quality(image[:, :60], config)
    assert full["luminance_p95"] == 170
    assert "LOW_LIGHT" in full["warning_codes"]
    assert "LOW_TEXTURE" in roi["warning_codes"]
    assert roi["dark_pixel_fraction"] == 1
    # Diagnostics must not claim enhancement changed the acquired exposure.
    assert full["luminance_p50"] == 8


@pytest.mark.parametrize("bad", [None, True, "magic", 1, {}])
def test_invalid_preprocessing_rejected(bad):
    with pytest.raises(StereoAnalysisError, match="preprocessing"):
        _analysis_config({"stereo_matching": {"preprocessing": bad}})


def test_min_disparity_plus_range_must_fit_image():
    images, calibration, _ = noisy_plane()
    config = _analysis_config({"stereo_matching": {"min_disparity": 600, "num_disparities": 64}})
    result = _compute_stereo_depth(images["left"], images["right"], calibration, config)
    assert result.audit["reason_codes"] == ["DISPARITY_RANGE_EXCEEDS_IMAGE_WIDTH"]
