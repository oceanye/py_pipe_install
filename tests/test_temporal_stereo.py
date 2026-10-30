"""Known numerical truth and refusal cases, not physical-camera acceptance."""
import copy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pipe_twin.temporal_stereo import _fuse_disparities, fuse_stereo_burst, static_pair_check
from pipe_twin.stereo_analyzer import StereoDepthResult


def _burst(count=9, *, missing=False, bias=0.):
    rng = np.random.default_rng(83)
    base = rng.integers(35, 220, (120, 160, 3), dtype=np.uint8)
    frames = []
    for index in range(count):
        image = np.clip(base.astype(float)+rng.normal(0, 1, base.shape), 0, 255).astype(np.uint8)
        depths = {}
        for role in ("left", "right"):
            disparity = 12+bias+rng.normal(0, .38, base.shape[:2])
            if missing:
                disparity[rng.random(disparity.shape) < .2] = np.nan
                disparity[25:35, 30:40] = np.nan  # unobserved in every frame
            depths[role] = (12000/disparity).astype(np.float32)
        frames.append({"capture_id": str(index), "left": image, "right": np.roll(image, -12, axis=1),
                       "depth": StereoDepthResult(depths["left"], depths["right"],
                           np.isfinite(depths["left"]), np.isfinite(depths["right"]),
                           {"status": "VALID", "left_right_consistency_px": 1.})})
    cal = SimpleNamespace(left=SimpleNamespace(fx=200.), right=SimpleNamespace(fx=200.), baseline_mm=60.)
    return frames, cal


def _run(frames, cal):
    by_bytes = {f["left"].tobytes(): f["depth"] for f in frames}
    return fuse_stereo_burst(frames, load_images=lambda f: f,
        compute_depth=lambda left, right: by_bytes[left.tobytes()],
        quality_check=lambda image: {"passed": True}, calibration=cal)


def test_random_noise_reduced_without_inventing_unseen_depth():
    frames, cal = _burst(missing=True)
    before = [f["depth"].left_depth_mm.copy() for f in frames]
    result = _run(frames, cal)
    audit = result.audit["temporal_fusion"]
    assert audit["accepted_frames"] == 9
    mask = result.left_valid
    single_error = np.nanmedian(np.abs(frames[-1]["depth"].left_depth_mm[mask]-1000))
    fused_error = np.nanmedian(np.abs(result.left_depth_mm[mask]-1000))
    assert fused_error < single_error*.65
    assert not mask[25:35, 30:40].any()
    assert np.isnan(result.left_depth_mm[25:35, 30:40]).all()
    for old, frame in zip(before, frames):
        np.testing.assert_equal(old, frame["depth"].left_depth_mm)


def test_duplicates_are_not_independent_samples():
    frames, cal = _burst(3)
    frames = [dict(frames[0], capture_id=str(i)) for i in range(3)]
    result = _run(frames, cal)
    assert result.audit["status"] == "INVALID"
    assert result.audit["temporal_fusion"]["accepted_frames"] == 1
    assert not result.left_valid.any()


def test_motion_invalidates_whole_burst_and_local_change_is_detected():
    frames, cal = _burst(4)
    frames[1]["left"] = np.roll(frames[1]["left"], 3, axis=1)
    result = _run(frames, cal)
    assert result.audit["status"] == "INVALID"
    assert "BURST_REFERENCE_OR_SCENE_UNSTABLE" in result.audit["reason_codes"]
    reference = frames[-1]["left"]
    altered = reference.copy()
    altered[30:60, 50:90] = 250
    assert not static_pair_check(reference, altered)["passed"]


def test_uniform_scene_cannot_certify_static_camera():
    image = np.full((120, 160, 3), 100, np.uint8)
    assert static_pair_check(image, image)["reason"] == "BURST_MOTION_UNVERIFIABLE"


def _textured_stationary_image():
    rng = np.random.default_rng(3010)
    gray = cv2.GaussianBlur(rng.normal(120, 10, (240, 320)).astype(np.float32), (5, 5), .8)
    return cv2.cvtColor(np.rint(gray).astype(np.uint8), cv2.COLOR_GRAY2BGR)


@pytest.mark.parametrize('offset', [-9, 9])
def test_uniform_brightness_offset_does_not_invent_motion_or_modify_images(offset):
    reference = _textured_stationary_image()
    changed = np.clip(reference.astype(int)+offset, 0, 255).astype(np.uint8)
    before = [reference.copy(), changed.copy()]
    check = static_pair_check(reference, changed)
    assert check['passed']
    assert check['median_motion_px'] < .001 and check['p95_motion_px'] < .001
    assert check['brightness_shift'] == offset
    assert check['tracking_brightness_offset_removed'] == offset
    np.testing.assert_array_equal(reference, before[0])
    np.testing.assert_array_equal(changed, before[1])


@pytest.mark.parametrize('displacement', [1., 2.])
def test_brightness_normalization_does_not_remove_actual_spatial_motion(displacement):
    reference = _textured_stationary_image()
    transformed = cv2.warpAffine(reference, np.array([[1., 0., displacement], [0., 1., 0.]]),
        (reference.shape[1], reference.shape[0]), borderMode=cv2.BORDER_REFLECT)
    changed = np.clip(transformed.astype(int)-9, 0, 255).astype(np.uint8)
    check = static_pair_check(reference, changed)
    assert not check['passed'] and check['reason'] == 'BURST_SCENE_MOVED'
    assert check['median_motion_px'] == pytest.approx(displacement, abs=.05)


def test_persistent_bias_is_not_reported_as_accuracy():
    frames, cal = _burst(bias=3.)
    result = _run(frames, cal)
    assert result.audit["status"] == "VALID"
    assert np.nanmedian(result.left_depth_mm) < 850  # stable but wrong by >150 mm
    assert result.audit["temporal_fusion"]["validation_scope"] == "TEMPORAL_REPEATABILITY_NOT_ABSOLUTE_ACCURACY"


def test_intermittent_outliers_and_insufficient_support():
    samples = [np.full((8, 8), 12., np.float32) for _ in range(7)]
    samples[0][:] = 30
    for sample in samples[:5]:
        sample[0, 0] = np.nan
    result, audit = _fuse_disparities(samples)
    assert np.isnan(result[0, 0])
    assert np.all(result[1:] == 12)
    assert audit["required_support_frames"] == 4


def test_one_sided_outliers_do_not_displace_a_supported_static_surface():
    # Eight real measurements form one 1.5px-wide cluster; four gross errors
    # pull the ordinary median to its far edge, losing three good samples.
    values = [11.25, 11.3, 11.4, 11.5, 12.4, 12.5, 12.6, 12.75, 30, 31, 32, 33]
    samples = [np.full((2, 3), v, np.float32) for v in values]
    # A never-observed point and a too-sparse point must remain missing.
    for sample in samples:
        sample[0, 0] = np.nan
    for sample in samples[:8]:
        sample[0, 1] = np.nan
    fused, audit = _fuse_disparities(samples)
    assert fused[1, 1] == pytest.approx(11.95, abs=.01)
    assert np.isnan(fused[0, :2]).all()
    assert audit["majority_cluster_recovered_pixels"] == 4


@pytest.mark.parametrize("upper", [[13, 13, 13, 13], [13, 13.1, 13.2, 13.3]])
def test_equal_competing_depth_modes_are_not_a_static_surface(upper):
    samples = [np.full((2, 3), value, np.float32) for value in [11, 11, 11, 11]+upper]
    fused, _ = _fuse_disparities(samples)
    # The ordinary median 12 was never measured: accepting +/-1 around it
    # would turn two equally supported alternatives into an invented surface.
    assert np.isnan(fused).all()


def test_majority_recovery_is_order_independent_and_never_uses_other_pixels():
    values = [11.25, 11.3, 11.4, 11.5, 12.4, 12.5, 12.6, 12.75, 30, 31, 32, 33]
    samples = [np.full((2, 3), value, np.float32) for value in values]
    for sample in samples:
        sample[0, 0] = np.nan
    before = [sample.copy() for sample in samples]
    forward, _ = _fuse_disparities(samples)
    backward, _ = _fuse_disparities(list(reversed(samples)))
    np.testing.assert_array_equal(forward, backward)
    assert np.isnan(forward[0, 0])
    for a, b in zip(samples, before):
        np.testing.assert_array_equal(a, b)


def test_inconsistent_eyes_remain_invalid_after_fusion():
    frames, cal = _burst(5)
    for frame in frames:
        frame["depth"].right_depth_mm[:] = 500
    result = _run(frames, cal)
    assert not result.left_valid.any()
    assert not result.right_valid.any()


def test_exposure_jump_is_excluded_and_bad_reference_fails():
    frames, cal = _burst(5)
    frames[0]["left"] = np.clip(frames[0]["left"].astype(int)+25, 0, 255).astype(np.uint8)
    result = _run(frames, cal)
    assert result.audit["temporal_fusion"]["accepted_frames"] == 4
    assert result.audit["temporal_fusion"]["frames"][0]["reason_codes"] == ["BURST_EXPOSURE_UNSTABLE"]
    result = fuse_stereo_burst(frames, load_images=lambda f: f, compute_depth=lambda l, r: None,
        quality_check=lambda image: {"passed": False}, calibration=cal)
    assert result.audit["status"] == "INVALID"


def test_pixelwise_fusion_rejects_over_budget_sequence():
    frames, cal = _burst(25)
    with pytest.raises(ValueError):
        _run(frames, cal)
