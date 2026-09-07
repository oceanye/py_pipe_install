from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path

import numpy as np

from pipe_twin.camera_pose import apply_camera_pose, calibration_pose
from pipe_twin.stereo_analyzer import _calibration_from_manifest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "test_model" / "field_stereo_demo_manifest.json"


class CameraPoseTests(unittest.TestCase):
    def setUp(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.calibration = manifest["stereo_calibration"]

    def test_keep_returns_an_independent_unchanged_copy(self):
        before = copy.deepcopy(self.calibration)
        result = apply_camera_pose(self.calibration, {"mode": "keep"})
        self.assertEqual(result, before)
        self.assertIsNot(result, self.calibration)
        result["left_camera"]["center_world_mm"][0] += 1
        self.assertEqual(self.calibration, before)

    def test_current_pose_roll_moves_both_cameras_as_one_rigid_rig(self):
        original = calibration_pose(self.calibration)
        adjustment = {
            "mode": "adjust_current",
            "center_world_mm": [10.0, 20.0, 30.0],
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "roll_deg": 12.0,
            "registration_validated": False,
        }
        result = apply_camera_pose(self.calibration, adjustment)
        parsed = _calibration_from_manifest(result)

        midpoint = (parsed.left.center_world_mm + parsed.right.center_world_mm) / 2
        self.assertTrue(np.allclose(midpoint, [10.0, 20.0, 30.0], atol=1e-9))
        baseline_camera = parsed.left.rotation_world_to_camera @ (
            parsed.right.center_world_mm - parsed.left.center_world_mm
        )
        self.assertTrue(
            np.allclose(baseline_camera, [parsed.baseline_mm, 0.0, 0.0], atol=1e-8)
        )
        old_rotation = np.asarray(original["rotation_world_to_camera"])
        old_camera_x_world = old_rotation.T @ np.array([1.0, 0.0, 0.0])
        rotated_x = parsed.left.rotation_world_to_camera @ old_camera_x_world
        self.assertAlmostEqual(
            math.degrees(math.atan2(rotated_x[1], rotated_x[0])), 12.0, places=8
        )
        self.assertFalse(parsed.registration_validated)
        self.assertNotEqual(result["calibration_id"], self.calibration["calibration_id"])
        self.assertEqual(result["registration_adjustment"]["roll_deg"], 12.0)

    def test_direction_presets_point_at_the_model_from_each_side(self):
        expected = {
            "positive_x": [-1.0, 0.0, 0.0],
            "negative_x": [1.0, 0.0, 0.0],
            "positive_y": [0.0, -1.0, 0.0],
            "negative_y": [0.0, 1.0, 0.0],
            "positive_z": [0.0, 0.0, -1.0],
            "negative_z": [0.0, 0.0, 1.0],
        }
        for mode, forward in expected.items():
            with self.subTest(mode=mode):
                result = apply_camera_pose(
                    self.calibration,
                    {
                        "mode": mode,
                        "center_world_mm": [100.0, 200.0, 300.0],
                        "registration_validated": True,
                    },
                )
                pose = calibration_pose(result)
                self.assertTrue(np.allclose(pose["center_world_mm"], [100, 200, 300]))
                self.assertTrue(np.allclose(pose["forward_world"], forward, atol=1e-9))
                self.assertTrue(pose["registration_validated"])

    def test_invalid_adjustments_fail_closed(self):
        cases = (
            {"mode": "sideways"},
            {"mode": "adjust_current", "yaw_deg": 181},
            {"mode": "adjust_current", "center_world_mm": [1, 2]},
            {"mode": "adjust_current", "center_world_mm": [1, 2, float("nan")]},
            {"mode": "adjust_current", "center_world_mm": [True, 2, 3]},
            {"mode": "adjust_current", "registration_validated": "yes"},
        )
        for adjustment in cases:
            with self.subTest(adjustment=adjustment), self.assertRaises(ValueError):
                apply_camera_pose(self.calibration, adjustment)


if __name__ == "__main__":
    unittest.main()
