from __future__ import annotations

import unittest

import cv2
import numpy as np

from pipe_twin.calibration_adapter import (
    CalibrationAdapterError,
    adapt_opencv_stereo_calibration,
    validate_calibration,
)
from pipe_twin.stereo_analyzer import _calibration_from_manifest, _project_points
from pipe_twin.camera_pose import apply_camera_pose
from pipe_twin.qr_registration import QrPoseEstimate, register_calibration_from_qr


class CalibrationAdapterTests(unittest.TestCase):
    @staticmethod
    def source() -> dict:
        return {
            "K1": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            "K2": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
            "D1": [0.01, -0.02, 0.0, 0.0, 0.0],
            "D2": [0.01, -0.02, 0.0, 0.0, 0.0],
            "image_size": [640, 480],
            "R": np.eye(3).tolist(),
            "T": [-0.095, 0.0, 0.0],
            "translation_unit": "m",
            "left_camera_pose": {
                "rotation_world_to_camera": np.eye(3).tolist(),
                "center_world_mm": [0.0, 0.0, 0.0],
            },
        }

    def test_opencv_rectification_is_manifest_compatible(self) -> None:
        calibration = adapt_opencv_stereo_calibration(
            self.source(), calibration_id="field-v1", validated=True, registration_validated=True
        )
        parsed = _calibration_from_manifest(calibration)
        self.assertTrue(parsed.rectified)
        self.assertAlmostEqual(parsed.baseline_mm, 95.0, places=5)
        self.assertIsNotNone(parsed.left.rectification_matrix)
        self.assertIsNotNone(parsed.right.projection_matrix)
        self.assertTrue(validate_calibration(calibration)["valid"])

    def test_missing_unit_is_rejected_instead_of_guessing(self) -> None:
        source = self.source()
        source.pop("translation_unit")
        with self.assertRaisesRegex(CalibrationAdapterError, "translation_unit"):
            adapt_opencv_stereo_calibration(source, calibration_id="field-v1")

    def test_unvalidated_registration_is_reported_as_not_ready(self) -> None:
        calibration = adapt_opencv_stereo_calibration(self.source(), calibration_id="field-v1")
        diagnostics = validate_calibration(calibration)
        self.assertTrue(diagnostics["valid"])
        self.assertFalse(diagnostics["ready_for_field_analysis"])

    def test_rectified_projection_uses_right_baseline_once(self) -> None:
        """A CAD point on the optical axis has exactly f*B/Z disparity."""
        calibration = adapt_opencv_stereo_calibration(
            self.source(), calibration_id="field-v1", validated=True, registration_validated=True
        )
        parsed = _calibration_from_manifest(calibration)
        world = np.asarray([[0.0, 0.0, 1000.0]], dtype=np.float64)
        left_px, _ = _project_points(world, parsed.left, rectified=True)
        right_px, _ = _project_points(world, parsed.right, rectified=True)
        # Distortion changes the common rectified principal point slightly;
        # the disparity itself must still equal the P2 baseline encoding.
        expected = float(parsed.right.projection_matrix[0, 0]) * 95.0 / 1000.0
        self.assertAlmostEqual(float(left_px[0, 0] - right_px[0, 0]), expected, places=5)

    def test_rectified_projection_matches_opencv_for_rotated_rig(self) -> None:
        """CAD projection agrees with OpenCV's independent undistortPoints path."""
        source = self.source()
        rvec = np.asarray([0.0, np.deg2rad(7.0), 0.0], dtype=np.float64)
        source["R"] = cv2.Rodrigues(rvec)[0].tolist()
        calibration = adapt_opencv_stereo_calibration(
            source, calibration_id="field-rot", validated=True, registration_validated=True
        )
        parsed = _calibration_from_manifest(calibration)
        world = np.asarray([[120.0, 35.0, 1100.0]], dtype=np.float64)
        ours_l, _ = _project_points(world, parsed.left, rectified=True)
        ours_r, _ = _project_points(world, parsed.right, rectified=True)
        # OpenCV's stereo convention: x2 = R*x1 + T.  Compare against the
        # exact R1/P1 and R2/P2 remap outputs, independently of our helper.
        x1 = world
        # Adapter explicitly converts the source metre translation to mm.
        x2 = (np.asarray(source["R"]) @ x1.T).T + np.asarray(source["T"])[None, :] * 1000.0
        expected_l = cv2.undistortPoints(
            cv2.projectPoints(x1, np.zeros(3), np.zeros(3), np.asarray(source["K1"]), np.asarray(source["D1"]))[0],
            np.asarray(source["K1"]), np.asarray(source["D1"]),
            R=np.asarray(parsed.left.rectification_matrix), P=np.asarray(parsed.left.projection_matrix),
        ).reshape(-1, 2)
        expected_r = cv2.undistortPoints(
            cv2.projectPoints(x2, np.zeros(3), np.zeros(3), np.asarray(source["K2"]), np.asarray(source["D2"]))[0],
            np.asarray(source["K2"]), np.asarray(source["D2"]),
            R=np.asarray(parsed.right.rectification_matrix), P=np.asarray(parsed.right.projection_matrix),
        ).reshape(-1, 2)
        np.testing.assert_allclose(ours_l, expected_l, rtol=0, atol=1e-5)
        np.testing.assert_allclose(ours_r, expected_r, rtol=0, atol=1e-5)

    def test_pose_adjustment_preserves_rectified_frame_for_nonparallel_rig(self) -> None:
        source = self.source()
        source["R"] = cv2.Rodrigues(np.asarray([0.0, np.deg2rad(7.0), 0.0]))[0].tolist()
        calibration = adapt_opencv_stereo_calibration(
            source, calibration_id="pose-rot", validated=True, registration_validated=True
        )
        before = _calibration_from_manifest(calibration)
        adjusted = apply_camera_pose(
            calibration,
            {"mode": "adjust_current", "roll_deg": 4.0, "registration_validated": True},
        )
        after = _calibration_from_manifest(adjusted)
        expected = cv2.Rodrigues(np.asarray([0.0, 0.0, np.deg2rad(4.0)]))[0] @ before.left.rotation_world_to_rectified_camera
        np.testing.assert_allclose(after.left.rotation_world_to_rectified_camera, expected, atol=1e-6)
        np.testing.assert_allclose(
            after.right.rotation_world_to_rectified_camera,
            cv2.Rodrigues(np.asarray([0.0, 0.0, np.deg2rad(4.0)]))[0]
            @ before.right.rotation_world_to_rectified_camera,
            atol=1e-6,
        )

    def test_qr_registration_stores_raw_rotations_without_double_rectification(self) -> None:
        source = self.source()
        source["R"] = cv2.Rodrigues(np.asarray([0.0, np.deg2rad(7.0), 0.0]))[0].tolist()
        calibration = adapt_opencv_stereo_calibration(
            source, calibration_id="qr-rot", validated=True, registration_validated=False
        )
        target_rotation = cv2.Rodrigues(np.asarray([0.0, np.deg2rad(3.0), 0.0]))[0]
        estimate = QrPoseEstimate(
            decoded_payload="QR", marker_edge_mm=100.0,
            corners_px=np.zeros((4, 2)), rotation_marker_to_camera=target_rotation,
            translation_marker_to_camera_mm=np.asarray([0.0, 0.0, 1000.0]),
            camera_center_marker_mm=np.asarray([0.0, 0.0, -1000.0]),
            camera_distance_mm=1000.0, reprojection_rms_px=0.1,
            source_image_sha256="a" * 64,
        )
        result = register_calibration_from_qr(
            calibration, estimate, marker_center_world_mm=[0, 0, 0],
            print_right_world="+X", print_up_world="+Y",
            registration_validated=True,
        )
        parsed = _calibration_from_manifest(result)
        np.testing.assert_allclose(parsed.left.rotation_world_to_rectified_camera, target_rotation, atol=1e-6)
        np.testing.assert_allclose(parsed.right.rotation_world_to_rectified_camera, target_rotation, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
