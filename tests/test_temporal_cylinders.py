"""Temporal fusion must improve actual cylinder/relative-layout extraction."""
import numpy as np

from pipe_twin.temporal_stereo import fuse_stereo_burst
from pipe_twin.stereo_analyzer import StereoDepthResult
from pipe_twin.recognition import recognize_local_pipes
from test_elevation_auto import _groups


def numerical_cylinder_comparison():
    cal, specs, groups = _groups()
    source = groups[0]
    rng = np.random.default_rng(431)
    texture = rng.integers(-24, 25, source["left"].shape)
    frames = []
    for index in range(20):
        frame = {"capture_id": str(index)}
        depths = {}
        for role in ("left", "right"):
            image = np.clip(source[role].astype(int)+texture, 0, 255).astype(np.uint8)
            image[0, 0] = [index+20, 40, 60]
            frame[role] = image
            truth = getattr(source["depth"], role+"_depth_mm")
            d = cal.left.fx*cal.baseline_mm/truth + rng.normal(0, .12, truth.shape)
            depths[role] = (cal.left.fx*cal.baseline_mm/d).astype(np.float32)
        frame["depth"] = StereoDepthResult(depths["left"], depths["right"],
            np.ones(depths["left"].shape, bool), np.ones(depths["right"].shape, bool),
            {"status": "VALID", "left_right_consistency_px": 1.})
        frames.append(frame)
    by_bytes = {f["left"].tobytes(): f["depth"] for f in frames}
    fused = fuse_stereo_burst(frames, load_images=lambda f: f,
        compute_depth=lambda l, r: by_bytes[l.tobytes()], quality_check=lambda im: {"passed": True}, calibration=cal)
    results = {}
    for name, depth in (("single", frames[-1]["depth"]), ("fused", fused)):
        results[name] = recognize_local_pipes("image_guided", frames[-1]["left"], frames[-1]["right"],
            depth, cal, specs, matching_settings={"color_filter_enabled": False})
    return results, fused.audit["temporal_fusion"]


def test_temporal_data_reaches_cylinder_fit_before_acceptance():
    results, audit = numerical_cylinder_comparison()
    assert audit["status"] == "VALID", audit
    assert len(results["fused"]["observations"]) > len(results["single"]["observations"])
    measured = sorted(o["diameter_mm"] for o in results["fused"]["observations"])
    np.testing.assert_allclose(measured, [24, 36, 50], atol=1.)
    observations = sorted(results["fused"]["observations"], key=lambda o: o["diameter_mm"])
    axes = np.asarray([o["axis_camera"] for o in observations])
    axis = np.linalg.eigh(axes.T@axes)[1][:, -1]
    for first, second, truth in ((0, 1, 100.), (0, 2, np.hypot(15, 165)), (1, 2, np.hypot(75, 85))):
        delta = np.asarray(observations[second]["center_camera_mm"])-observations[first]["center_camera_mm"]
        distance = np.linalg.norm(delta-np.dot(delta, axis)*axis)
        assert abs(distance-truth) < 1.5
