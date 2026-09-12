from __future__ import annotations

import copy
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pipe_twin.local_surface import LocalSurfaceError, extract_local_pipes, _CloudBudget
from pipe_twin.metrology import fit_local_cylinder


def _camera(center_x: float = 0.0):
    k = np.asarray([[500.0, 0.0, 160.0], [0.0, 500.0, 120.0], [0.0, 0.0, 1.0]])
    return SimpleNamespace(
        fx=500.0, fy=500.0, width=320, height=240, rectified_intrinsic=k,
        intrinsic=k, center_world_mm=np.asarray([center_x, 0.0, 0.0]),
        rotation_world_to_rectified_camera=np.eye(3),
    )


def _pair(axis=(1.0, 0.0, 0.0), noisy: bool = True, right_y_shift: int = 0,
          right_radius: float = 25.0, right_axis=None):
    h, w = 240, 320
    left = np.full((h, w, 3), 70, dtype=np.uint8)
    right = np.full((h, w, 3), 70, dtype=np.uint8)
    zl = np.full((h, w), np.nan, dtype=np.float32)
    zr = np.full((h, w), np.nan, dtype=np.float32)
    vl = np.zeros((h, w), dtype=bool)
    vr = np.zeros((h, w), dtype=bool)
    a = np.asarray(axis, dtype=float); a /= np.linalg.norm(a)
    centre = np.asarray([0.0, 0.0, 900.0])
    rng = np.random.default_rng(7)

    def render(camera_x: float, image: np.ndarray, depth: np.ndarray, valid: np.ndarray, y_shift: int = 0):
        direction = a if camera_x == 0 or right_axis is None else np.asarray(right_axis, dtype=float)
        direction = direction / np.linalg.norm(direction)
        radius = 25.0 if camera_x == 0 else right_radius
        yy, xx = np.indices((h, w), dtype=float)
        rays = np.stack(((xx - 160.0) / 500.0, (yy - 120.0) / 500.0, np.ones((h, w))), axis=-1)
        origin = np.asarray([camera_x, 0.0, 0.0])
        cp = origin - centre
        rp = rays - np.sum(rays * direction, axis=-1, keepdims=True) * direction
        cp_perp = cp - direction * float(np.dot(cp, direction))
        qa = np.sum(rp * rp, axis=-1)
        qb = 2.0 * np.sum(rp * cp_perp, axis=-1)
        qc = float(np.dot(cp_perp, cp_perp) - radius**2)
        disc = qb * qb - 4.0 * qa * qc
        hit = disc > 0
        t = np.full((h, w), np.nan)
        t[hit] = (-qb[hit] - np.sqrt(disc[hit])) / (2.0 * qa[hit])
        points = origin + rays * t[..., None]
        axial = np.sum((points - centre) * direction, axis=-1)
        hit &= np.isfinite(t) & (t > 0) & (np.abs(axial) <= 180.0)
        if y_shift:
            hit = np.roll(hit, y_shift, axis=0)
            t = np.roll(t, y_shift, axis=0)
        image[hit] = (0, 0, 255)
        value = t * rays[..., 2]
        if noisy:
            value = value + rng.normal(0.0, 0.4, value.shape)
        depth[hit] = value[hit]
        valid[hit] = True

    render(0.0, left, zl, vl)
    render(60.0, right, zr, vr, right_y_shift)
    calibration = SimpleNamespace(validated=True, rectified=True, baseline_mm=60.0, left=_camera(0.0), right=_camera(60.0))
    depth = SimpleNamespace(left_depth_mm=zl, right_depth_mm=zr, left_valid=vl, right_valid=vr)
    return left, right, depth, calibration


class LocalSurfaceTests(unittest.TestCase):
    def test_no_pose_local_cylinder_is_reconstructed_and_paired(self):
        left, right, depth, calibration = _pair()
        result = extract_local_pipes(
            left, right, depth, calibration,
            [{"pipe_id": "P1", "nominal_diameter_mm": 50.0, "color_srgb": "#FF0000"}],
        )
        self.assertEqual(result["audit"]["coordinate_frame"], "LEFT_RECTIFIED_CAMERA_MM")
        self.assertLessEqual(result["audit"]["point_count"], 6000)
        self.assertEqual(len(result["observations"]), 1, result)
        observation = result["observations"][0]
        self.assertNotIn("pipe_id", observation)
        self.assertEqual(observation["candidate_pipe_ids"], ["P1"])
        self.assertAlmostEqual(observation["diameter_mm"], 50.0, delta=3.0)
        self.assertLess(observation["fit_rms_mm"], 2.0)
        self.assertGreater(observation["point_count"], 150)

    def test_tilted_axis_survives_noisy_depth(self):
        left, right, depth, calibration = _pair(axis=(1.0, 0.18, 0.12))
        result = extract_local_pipes(
            left, right, depth, calibration,
            [{"pipe_id": "P1", "nominal_diameter_mm": 50.0, "color_srgb": "#FF0000"}],
        )
        self.assertEqual(len(result["observations"]), 1, result)
        self.assertLess(result["observations"][0]["left_right_axis_angle_deg"], 3.0)

    def test_bad_right_view_is_rejected_instead_of_fabricating_pair(self):
        left, right, depth, calibration = _pair(right_y_shift=100)
        result = extract_local_pipes(
            left, right, depth, calibration,
            [{"pipe_id": "P1", "nominal_diameter_mm": 50.0, "color_srgb": "#FF0000"}],
        )
        self.assertEqual(result["observations"], [])
        self.assertTrue(result["rejected"])

    def test_flat_colour_plane_cannot_be_called_a_pipe(self):
        left, right, depth, calibration = _pair()
        mask = np.all(left == (0, 0, 255), axis=2)
        left[mask] = (0, 0, 255); right[mask] = (0, 0, 255)
        depth.left_depth_mm[mask] = 900.0; depth.right_depth_mm[mask] = 840.0
        result = extract_local_pipes(
            left, right, depth, calibration,
            [{"pipe_id": "P1", "nominal_diameter_mm": 50.0, "color_srgb": "#FF0000"}],
        )
        self.assertEqual(result["observations"], [])
        self.assertTrue(any("SURFACE" in str(row.get("reason", "")) or "CYLINDER" in str(row.get("reason", "")) for row in result["rejected"]))

    def test_invalid_inputs_fail_closed(self):
        left, right, depth, calibration = _pair()
        with self.assertRaises(LocalSurfaceError):
            extract_local_pipes(left.astype(np.float32), right, depth, calibration, [{"nominal_diameter_mm": 50, "color_srgb": "#FF0000"}])

    def test_nearby_same_colour_pipe_cannot_replace_the_right_counterpart(self):
        specs = [{"pipe_id": "P1", "nominal_diameter_mm": 50, "color_srgb": "#FF0000"}]
        # These image shifts correspond to about 10, 20 and 30 mm at this
        # depth.  Both views still contain well-fit cylinders.
        for pixels in (6, 12, 18):
            with self.subTest(pixels=pixels):
                result = extract_local_pipes(*_pair(right_y_shift=pixels), specs)
                self.assertEqual(result["observations"], [])
                self.assertTrue(any(row["reason"] == "NO_UNIQUE_LEFT_RIGHT_PAIR" for row in result["rejected"]))
                self.assertGreaterEqual(len(result["audit"]["components"]), 2)

    def test_different_diameter_or_axis_cannot_be_paired(self):
        specs = [{"pipe_id": "P1", "nominal_diameter_mm": 50, "color_srgb": "#FF0000"}]
        for options in ({"right_radius": 28.0}, {"right_axis": (1.0, 0.14, 0.0)}):
            with self.subTest(options=options):
                result = extract_local_pipes(*_pair(**options), specs)
                self.assertEqual(result["observations"], [])

    def test_opposite_pca_signs_do_not_cancel_observed_segment(self):
        args = _pair()
        specs = [{"pipe_id": "P1", "nominal_diameter_mm": 50, "color_srgb": "#FF0000"}]
        baseline = extract_local_pipes(*args, specs)["observations"][0]
        calls = []

        def sign_changed(points):
            fit = fit_local_cylinder(points)
            calls.append(True)
            # Depth and RGB fits in the right view return the equivalent
            # cylinder parameterisation with its axial sign reversed.
            if len(calls) > 2:
                fit["axis_direction_world"] = (-np.asarray(fit["axis_direction_world"])).tolist()
                fit["observed_segment_world_mm"] = fit["observed_segment_world_mm"][::-1]
            return fit

        with patch("pipe_twin.local_surface.fit_local_cylinder", side_effect=sign_changed):
            observation = extract_local_pipes(*args, specs)["observations"][0]
        segment = np.asarray(observation["observed_segment_camera_mm"])
        self.assertGreater(np.linalg.norm(segment[1] - segment[0]), 300)
        np.testing.assert_allclose(segment, baseline["observed_segment_camera_mm"], atol=1e-6)
        self.assertAlmostEqual(np.linalg.norm(segment[1] - segment[0]), observation["axial_support_mm"], places=5)

    def test_fusion_uses_only_real_common_axial_support(self):
        left, right, depth, calibration = _pair()
        depth.left_valid[:, :120] = False
        depth.right_valid[:, 175:] = False
        specs = [{"pipe_id": "P1", "nominal_diameter_mm": 50, "color_srgb": "#FF0000"}]
        result = extract_local_pipes(left, right, depth, calibration, specs)
        self.assertEqual(len(result["observations"]), 1, result["rejected"])
        observation = result["observations"][0]
        points = np.asarray(observation["observed_segment_camera_mm"])
        self.assertGreater(observation["axial_support_mm"], 100)
        self.assertLess(observation["axial_support_mm"], 165)
        self.assertTrue(np.all(points[:, 0] >= -72))
        self.assertTrue(np.all(points[:, 0] <= 88))

    def test_disjoint_axial_support_does_not_pair(self):
        left, right, depth, calibration = _pair()
        depth.left_valid[:, :175] = False
        depth.right_valid[:, 120:] = False
        result = extract_local_pipes(left, right, depth, calibration,
                                     [{"pipe_id": "P1", "nominal_diameter_mm": 50, "color_srgb": "#FF0000"}])
        self.assertEqual(result["observations"], [])

    def test_same_diameter_candidates_are_not_assigned_to_model_id(self):
        specs = [{"pipe_id": identity, "nominal_diameter_mm": 50, "color_srgb": "#FF0000"}
                 for identity in ("P1", "P2")]
        observation = extract_local_pipes(*_pair(), specs)["observations"][0]
        self.assertNotIn("pipe_id", observation)
        self.assertEqual(observation["candidate_pipe_ids"], ["P1", "P2"])
        self.assertIs(observation["identity_assigned"], False)

    def test_display_colour_does_not_prevent_depth_surface_reconstruction(self):
        left, right, depth, calibration = _pair()
        for image, valid in ((left, depth.left_valid), (right, depth.right_valid)):
            image[valid] = (100, 110, 120)
        result = extract_local_pipes(left, right, depth, calibration,
                                     [{"pipe_id": "P1", "nominal_diameter_mm": 50, "color_srgb": "#FF0000"}])
        self.assertEqual(len(result["observations"]), 1, result["rejected"])
        observation = result["observations"][0]
        self.assertEqual(observation["segmentation_source"], "depth_connected_surface")
        self.assertEqual(observation["measured_color_srgb"], "#786E64")
        self.assertEqual(observation["color_srgb"], "#786E64")

    def test_depth_jump_separates_unpainted_pipe_from_same_colour_plane(self):
        left, right, depth, calibration = _pair()
        for image, z, valid in ((left, depth.left_depth_mm, depth.left_valid),
                                (right, depth.right_depth_mm, depth.right_valid)):
            image[:] = 100
            z[~valid] = 1300
            valid[:] = True
        result = extract_local_pipes(left, right, depth, calibration,
                                     [{"pipe_id": "P1", "nominal_diameter_mm": 50, "color_srgb": "#FF0000"}])
        self.assertEqual(len(result["observations"]), 1, result["rejected"])
        self.assertTrue(any(row["reason"] != "NO_UNIQUE_LEFT_RIGHT_PAIR" for row in result["rejected"]))

    def test_zero_valid_depth_cannot_generate_a_pipe(self):
        left, right, depth, calibration = _pair()
        depth.left_valid[:] = depth.right_valid[:] = False
        result = extract_local_pipes(left, right, depth, calibration,
                                     [{"pipe_id": "P1", "nominal_diameter_mm": 50, "color_srgb": "#FF0000"}])
        self.assertEqual(result["observations"], [])
        self.assertEqual(result["point_cloud"]["points_camera_mm"], [])
        self.assertTrue(any(row["reason"] == "INSUFFICIENT_VALID_DEPTH" for row in result["rejected"]))

    def test_rectified_intrinsic_only_provider_and_arbitrary_world_pose(self):
        args = _pair()
        specs = [{"pipe_id": "P1", "nominal_diameter_mm": 50, "color_srgb": "#FF0000"}]
        baseline = extract_local_pipes(*args, specs)["observations"][0]
        calibration = args[-1]
        rotation = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        origin = np.asarray([500000., -200000., 10000.])
        for role, offset in (("left", 0.), ("right", 60.)):
            camera = getattr(calibration, role)
            del camera.intrinsic
            camera.center_world_mm = origin + rotation.T @ np.asarray([offset, 0., 0.])
            camera.rotation_world_to_rectified_camera = rotation
        observation = extract_local_pipes(*args, specs)["observations"][0]
        np.testing.assert_allclose(observation["center_camera_mm"], baseline["center_camera_mm"], atol=1e-9)

    def test_invalid_metric_calibration_and_depth_types_are_rejected(self):
        args = _pair()
        specs = [{"pipe_id": "P1", "nominal_diameter_mm": 50, "color_srgb": "#FF0000"}]
        for field, value in (("baseline_mm", 0), ("baseline_mm", float("nan")), ("baseline_mm", 55),
                             ("baseline_mm", True), ("validated", "true")):
            changed = copy.deepcopy(args)
            setattr(changed[-1], field, value)
            with self.subTest(field=field, value=value), self.assertRaises(LocalSurfaceError):
                extract_local_pipes(*changed, specs)
        for field, value in (("fx", float("nan")), ("fy", float("inf")), ("fx", True), ("height", 123)):
            changed = copy.deepcopy(args)
            setattr(changed[-1].left, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(LocalSurfaceError):
                extract_local_pipes(*changed, specs)
        for dtype in (object, bool, complex, "U5"):
            changed = copy.deepcopy(args)
            changed[2].left_depth_mm = np.zeros(changed[2].left_depth_mm.shape, dtype=dtype)
            with self.subTest(dtype=dtype), self.assertRaises(LocalSurfaceError):
                extract_local_pipes(*changed, specs)

    def test_point_budget_is_applied_before_json_lists(self):
        points = np.arange(3_000_000, dtype=np.float64).reshape(-1, 3)
        cloud = _CloudBudget(6000)
        cloud.add(points, "#123456", len(points))
        self.assertIsInstance(cloud.colors, np.ndarray)
        self.assertEqual(cloud.points.shape, (6000, 3))
        cloud.add(points, "#ABCDEF", len(points))
        self.assertEqual(len(cloud.points), 6000)
        self.assertEqual(cloud.source_count, 2_000_000)
        public = cloud.public()
        self.assertEqual(len(public["points_camera_mm"]), 6000)
        self.assertEqual(len(public["colors_srgb"]), 6000)
        json.dumps(public, allow_nan=False)

    def test_depth_component_fraction_counts_internal_holes(self):
        left, right, depth, calibration = _pair()
        for valid in (depth.left_valid, depth.right_valid):
            valid[116:125, 145:170] = False
        result = extract_local_pipes(left, right, depth, calibration,
                                     [{"pipe_id": "P1", "nominal_diameter_mm": 50, "color_srgb": "#FF0000"}])
        self.assertEqual(len(result["observations"]), 1)
        geometry = [row for row in result["audit"]["components"]
                    if row["valid_fraction_basis"] == "ENCLOSED_COMPONENT_INCLUDING_HOLES"]
        self.assertEqual(len(geometry), 2)
        for row in geometry:
            self.assertGreater(row["support_region_pixels"], row["valid_region_pixels"])
            self.assertLess(row["valid_fraction"], 0.99)
            self.assertGreater(row["valid_fraction"], 0.90)

    def test_candidate_work_is_bounded_and_truncation_is_explicit(self):
        left, right, depth, calibration = _pair()
        for image, z, valid in ((left, depth.left_depth_mm, depth.left_valid),
                                (right, depth.right_depth_mm, depth.right_valid)):
            image[:] = 70
            z[:] = np.nan
            valid[:] = False
            for y in range(10, 200, 20):
                for x in range(10, 280, 20):
                    image[y:y+10, x:x+10] = (0, 0, 255)
                    z[y:y+10, x:x+10] = 900.
                    valid[y:y+10, x:x+10] = True
        result = extract_local_pipes(left, right, depth, calibration,
                                     [{"pipe_id": "P1", "nominal_diameter_mm": 50, "color_srgb": "#FF0000"}],
                                     config={"maximum_component_candidates": 3})
        self.assertIs(result["audit"]["truncated"], True)
        self.assertEqual(result["audit"]["status"], "TRUNCATED")
        self.assertEqual(result["observations"], [])
        for role in ("left", "right"):
            self.assertEqual(result["audit"]["candidate_budgets"][role]["attempted"], 3)
        self.assertTrue(any(row.get("skipped_components", 0) > 0 for row in result["rejected"]))


if __name__ == "__main__":
    unittest.main()
