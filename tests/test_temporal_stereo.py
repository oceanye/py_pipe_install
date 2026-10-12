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
