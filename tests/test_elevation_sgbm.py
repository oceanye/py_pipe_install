"""Actual OpenCV SGBM on independently rendered, textured pipe surfaces.

The synthetic rig is deliberately close-range and well textured.  It tests
the reconstruction chain, not an accuracy promise for untextured field pipes.
Analytic depth is used only to measure matcher error, never as its input.
"""
from __future__ import annotations

import cv2
import numpy as np

from pipe_twin.local_surface import extract_local_pipes
from pipe_twin.elevation_auto import analyze_elevation_auto_groups
from pipe_twin.stereo_analyzer import CameraCalibration, StereoCalibration, _analysis_config, _compute_stereo_depth, _quality


def textured_cylinder_scene():
    width, height, focal, baseline = 1280, 800, 1400.0, 120.0
    intrinsic = np.asarray([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.]])
    cameras = [CameraCalibration(role, role, width, height, intrinsic, np.zeros(5), np.eye(3),
                                 np.asarray([offset, 0., 0.]), None, None)
               for role, offset in (("left", 0.), ("right", baseline))]
    calibration = StereoCalibration("TEXTURED-SYNTHETIC-REGRESSION", True, False, True, baseline, 5., *cameras)
    specs = []
    # In model space the common longitudinal direction is +Z.  Three unequal
    # diameters and non-collinear transverse centres constrain the layout.
    rotation = np.asarray([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    translation = np.asarray([0., 0., 850.])
    for index, (x, y, diameter) in enumerate(((-110., 0., 50.), (0., 150., 70.), (110., 50., 90.))):
        specs.append({"pipe_id": f"P{index + 1}", "nominal_diameter_mm": diameter,
                      "centerline_world_mm": [[x, y, -800.], [x, y, 800.]],
                      "color_srgb": ("#C900C9", "#00C900", "#00C9C9")[index]})
    rng = np.random.default_rng(291)
    textures = [cv2.GaussianBlur(rng.integers(0, 256, (1024, 2048), dtype=np.uint8), (3, 3), 0.6)
                for _ in range(4)]
    images, truth = {}, {}
    for role in ("left", "right"):
        camera = getattr(calibration, role)
        yy, xx = np.indices((height, width), dtype=np.float64)
        rays = np.stack(((xx - width / 2) / focal, (yy - height / 2) / focal, np.ones(xx.shape)), axis=-1)
        origin = camera.center_world_mm
        z = np.full(xx.shape, 1500.)
        background = origin + rays * z[..., None]
        gray = cv2.remap(textures[0], ((background[..., 0] + 1000.) / 1.25).astype(np.float32),
                         ((background[..., 1] + 600.) / 1.25).astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_WRAP)
        image = np.repeat(gray[..., None], 3, axis=2)
        for index, spec in enumerate(specs):
            centre = rotation @ np.mean(spec["centerline_world_mm"], axis=0) + translation
            delta = origin - centre
            # Cylinder axis is camera X; solve the Y/Z ray intersection.
            qa = rays[..., 1]**2 + 1.
            qb = 2. * (rays[..., 1] * delta[1] + delta[2])
            qc = delta[1]**2 + delta[2]**2 - (spec["nominal_diameter_mm"] / 2.)**2
            discriminant = qb*qb - 4.*qa*qc
            hit = discriminant > 0
            t = np.full(xx.shape, np.inf)
            t[hit] = (-qb[hit] - np.sqrt(discriminant[hit])) / (2.*qa[hit])
            points = origin + rays * np.where(hit, t, 0)[..., None]
            station = points[..., 0] - centre[0]
            hit &= (t > 0) & (t < z) & (np.abs(station) < 800.)
            theta = np.arctan2(points[..., 1] - centre[1], centre[2] - points[..., 2])
            arc = theta * spec["nominal_diameter_mm"] / 2.
            pigment = cv2.remap(textures[index + 1], (station / 1.25 + 600.).astype(np.float32),
                               (arc / 1.25 + 300.).astype(np.float32), cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_WRAP)
            tint = np.asarray([int(spec["color_srgb"][i:i+2], 16) for i in (5, 3, 1)], dtype=float)
            tint /= tint.max()
            colour = np.clip((150. + 0.4*pigment[..., None]) * tint, 0, 255).astype(np.uint8)
            image[hit] = colour[hit]
            z[hit] = t[hit]
        images[role], truth[role] = image, z
    return images, truth, calibration, specs


def test_real_sgbm_reconstructs_visible_local_pipe_cylinders():
    images, truth, calibration, specs = textured_cylinder_scene()
    # The maximum physical disparity is about 205 px.  Only the search range
    # is enlarged from the normal matcher defaults to cover this metric rig.
    config = _analysis_config({"stereo_matching": {"num_disparities": 256}})
    depth = _compute_stereo_depth(images["left"], images["right"], calibration, config)
    surface = extract_local_pipes(images["left"], images["right"], depth, calibration, specs)
    assert len(surface["observations"]) == 3, surface["rejected"]
    assert depth.audit["status"] == "VALID"
    assert surface["audit"]["truncated"] is False
    for observation in surface["observations"]:
        assert observation["fit_rms_mm"] <= 1.5
        assert observation["left_right_center_difference_mm"] <= 5.0
        assert observation["candidate_pipe_ids"]
    # Truth is compared only after extraction.  It never replaces a depth
    # sample or masks a bad point before the reconstruction chain.
    for role in ("left", "right"):
        pipe_mask = truth[role] < 1499.
        measured_z = getattr(depth, f"{role}_depth_mm")
        valid = getattr(depth, f"{role}_valid") & pipe_mask
        assert np.count_nonzero(valid) / np.count_nonzero(pipe_mask) > .65
        assert np.percentile(np.abs(measured_z[valid] - truth[role][valid]), 90) < 3.
    quality = {role: _quality(images[role], config) for role in ("left", "right")}
    capture = {"left": images["left"], "right": images["right"], "depth": depth,
               "pair_healthy": all(row["passed"] for row in quality.values()) and depth.audit["status"] == "VALID",
               "quality": quality, "capture_id": "SGBM-CAPTURE-1",
               "captured_at": "2026-09-12T10:00:00+08:00"}
    result = analyze_elevation_auto_groups([capture], calibration=calibration, pipe_specs=specs)
    assert result["registration"]["status"] == "MATCHED", result["registration"]
    assert result["counts"] == {"INSTALLED": 3, "NOT_INSTALLED": 0, "UNKNOWN": 0}
