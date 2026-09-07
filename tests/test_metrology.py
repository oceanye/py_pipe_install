from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pipe_twin.metrology import analyze_local_geometry, fit_local_cylinder, measurement_settings, pair_geometry
from pipe_twin.stereo_analyzer import CameraCalibration, PipePrior, PipeProjection, StereoCalibration, StereoDepthResult, _analysis_config, _compute_stereo_depth


def camera(role="left", x=0.0):
    return CameraCalibration(role, role, 480, 240, np.array([[600., 0., 240.], [0., 600., 120.], [0., 0., 1.]]),
                             np.zeros(5), np.eye(3), np.array([x, 0., 0.]), None, None)


def cylinder(radius=20., center=(0., 0., 800.), rotation=None, arc=140.):
    angles, axial = np.meshgrid(np.linspace(-math.radians(arc / 2), math.radians(arc / 2), 40), np.linspace(-90, 90, 45))
    points = np.column_stack((axial.ravel(), radius * np.sin(angles.ravel()), -radius * np.cos(angles.ravel())))
    if rotation is not None:
        points = points @ rotation.T
    return points + np.array(center)


def segment(identity, y, z, diameter=20., start=-100., end=100.):
    return {"pipe_id": identity, "status": "MEASURED", "diameter_mm": diameter,
            "center_world_mm": [0., y, z], "axis_direction_world": [1., 0., 0.],
            "observed_segment_world_mm": [[start, y, z], [end, y, z]]}


def analytic_scene():
    left, right = camera(), camera("right", 50.)
    calibration = StereoCalibration("known", True, True, True, 50., 5., left, right)
    specs = [("A", -45., 800., 20., "#FF0000"), ("B", 45., 900., 15., "#0000FF")]
    pipes = [PipePrior(identity, identity, None, i + 1, "test", "test", color, 20.,
                       np.array([[-180., y, z], [180., y, z]])) for i, (identity, y, z, _, color) in enumerate(specs)]
    images, depths, valid, projections = {}, {}, {}, {"left": {}, "right": {}}
    for role in ("left", "right"):
        cam = getattr(calibration, role)
        rows, cols = np.indices((cam.height, cam.width))
        slope_y, slope_x = (rows - 120.) / 600., (cols - 240.) / 600.
        depth = np.full(rows.shape, np.inf)
        image = np.full((*rows.shape, 3), 80, np.uint8)
        for identity, y, z, radius, color in specs:
            a = 1 + slope_y ** 2
            b = -2 * (slope_y * y + z)
            c = y * y + z * z - radius * radius
            discriminant = b * b - 4 * a * c
            hit_z = (-b - np.sqrt(np.maximum(discriminant, 0))) / (2 * a)
            hit_x = slope_x * hit_z + cam.center_world_mm[0]
            mask = (discriminant > 0) & (np.abs(hit_x) <= 180) & (hit_z < depth)
            depth[mask] = hit_z[mask]
            image[mask] = [int(color[i:i + 2], 16) for i in (5, 3, 1)]
            # Deliberately project a WRONG 20 mm nominal diameter for both.
            centerline = [[600 * (x - cam.center_world_mm[0]) / z + 240, 600 * y / z + 120] for x in (-180, 180)]
            projections[role][identity] = PipeProjection(np.zeros(rows.shape, bool), z, centerline, 600 * 20 / z, 1, 1, "FULLY_VISIBLE")
        valid[role] = np.isfinite(depth)
        depth[~valid[role]] = np.nan
        images[role], depths[role] = image, depth
    depth = StereoDepthResult(depths["left"], depths["right"], valid["left"], valid["right"], {})
    return pipes, projections, images, depth, calibration


class MetrologyTests(unittest.TestCase):
    def test_cylinder_fit_does_not_require_nominal_diameter(self):
        fit = fit_local_cylinder(cylinder(radius=25.))
        self.assertAlmostEqual(fit["diameter_mm"], 50., places=4)
        self.assertAlmostEqual(fit["center_world_mm"][2], 800., places=4)
        self.assertLess(fit["fit_rms_mm"], 1e-5)

    def test_tilted_observed_axis_and_translated_cylinder(self):
        angle = .3
        rotation = np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])
        fit = fit_local_cylinder(cylinder(center=(31, 80, 950), rotation=rotation))
        self.assertAlmostEqual(fit["diameter_mm"], 40., places=4)
        self.assertGreater(abs(np.dot(fit["axis_direction_world"], rotation[:, 0])), .99999)
        np.testing.assert_allclose(fit["center_world_mm"], [31, 80, 950], atol=1e-4)

    def test_flat_or_small_arc_is_not_reported_as_a_diameter(self):
        points = cylinder()
        points[:, 2] = 780.
        with self.assertRaises(ValueError):
            fit_local_cylinder(points)
        with self.assertRaisesRegex(ValueError, "ARC"):
            fit_local_cylinder(cylinder(arc=20))

    def test_tiny_occlusion_fragment_cannot_be_accepted_as_a_smaller_pipe(self):
        points = cylinder(radius=5)
        points[:, 0] *= 26 / 180
        with self.assertRaisesRegex(ValueError, "SUPPORT_TOO_SHORT"):
            fit_local_cylinder(points)

    def test_center_distance_clear_gap_and_front_back(self):
        pair = pair_geometry(segment("A", 0, 800, 20), segment("B", 30, 840, 30), camera())
        self.assertEqual(pair["status"], "MEASURED")
        self.assertAlmostEqual(pair["center_distance_mm"], 50)
        self.assertAlmostEqual(pair["clear_gap_mm"], 25)
        self.assertEqual(pair["front_pipe_id"], "A")
        self.assertAlmostEqual(pair["depth_delta_b_minus_a_mm"], 40)

    def test_order_is_unresolved_within_tolerance_and_reverses_with_camera(self):
        pair = pair_geometry(segment("A", 0, 800), segment("B", 50, 800.8), camera(), 1)
        self.assertIsNone(pair["front_pipe_id"])
        cam = camera()
        reverse = SimpleNamespace(rotation_world_to_camera=np.diag([1, -1, -1]), center_world_mm=np.array([0, 0, 1600]))
        self.assertEqual(pair_geometry(segment("A", 0, 800), segment("B", 50, 850), reverse)["front_pipe_id"], "B")

    def test_no_extrapolation_or_missing_measurement_fallback(self):
        pair = pair_geometry(segment("A", 0, 800), segment("B", 50, 800, start=150, end=250), camera())
        self.assertEqual(pair["status"], "UNKNOWN")
        missing = segment("A", 0, 800) | {"status": "UNKNOWN"}
        self.assertNotIn("center_distance_mm", pair_geometry(missing, segment("B", 50, 800), camera()))

    def test_negative_gap_is_preserved_for_review(self):
        pair = pair_geometry(segment("A", 0, 800), segment("B", 15, 800), camera())
        self.assertEqual(pair["clear_gap_mm"], -5)
        self.assertIn("NEGATIVE_GAP_CHECK_GEOMETRY", pair["reason_codes"])

    def test_two_views_recover_actual_diameter_and_pair_geometry_without_cad_mask(self):
        result = analyze_local_geometry(*analytic_scene(), healthy=True)
        self.assertEqual([r["status"] for r in result["pipes"]], ["MEASURED", "MEASURED"], result)
        for pipe, diameter in zip(result["pipes"], (40., 30.)):
            self.assertAlmostEqual(pipe["diameter_mm"], diameter, places=3)
            self.assertEqual(pipe["nominal_diameter_mm"], 20.)
        pair = result["pairs"][0]
        self.assertAlmostEqual(pair["center_distance_mm"], math.hypot(90, 100), places=3)
        self.assertAlmostEqual(pair["clear_gap_mm"], math.hypot(90, 100) - 35, places=3)
        self.assertEqual(pair["front_pipe_id"], "A")

    def test_unhealthy_calibration_and_single_view_do_not_produce_measurements(self):
        args = analytic_scene()
        result = analyze_local_geometry(*args, healthy=False)
        self.assertTrue(all(r["status"] == "UNKNOWN" for r in result["pipes"]))
        args[3].right_valid[:] = False
        result = analyze_local_geometry(*args, healthy=True)
        self.assertTrue(all(r["status"] == "UNKNOWN" for r in result["pipes"]))

    def test_right_matcher_invalid_negative_sentinel_is_never_depth(self):
        _, _, images, _, calibration = analytic_scene()
        class FakeMatcher:
            def __init__(self, disparity):
                self.disparity = disparity
            def compute(self, left, right):
                return np.full(left.shape, self.disparity * 16, np.int16)
        config = _analysis_config({"stereo_matching": {"num_disparities": 32}})
        with patch("pipe_twin.stereo_analyzer.cv2.StereoSGBM_create", side_effect=[FakeMatcher(31), FakeMatcher(-33)]):
            result = _compute_stereo_depth(images["left"], images["right"], calibration, config)
        self.assertFalse(result.left_valid.any())
        self.assertFalse(result.right_valid.any())

    def test_measurement_settings_reject_nonfinite_and_invalid_roi(self):
        for value in (0, -1, float("nan"), True):
            with self.assertRaises(ValueError):
                measurement_settings({"tolerance_mm": value})
        with self.assertRaises(ValueError):
            measurement_settings({"rois": {"A": {"left": [0, 0, 1, 1]}}})


if __name__ == "__main__":
    unittest.main()
