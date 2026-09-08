from __future__ import annotations

import unittest

import numpy as np

from pipe_twin.calibration_adapter import (
    CalibrationAdapterError,
    adapt_opencv_stereo_calibration,
    validate_calibration,
)
from pipe_twin.stereo_analyzer import _calibration_from_manifest


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


if __name__ == "__main__":
    unittest.main()
