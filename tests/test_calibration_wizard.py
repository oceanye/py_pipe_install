import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from pipe_twin.calibration_wizard import (
    BoardObservation,
    CalibrationWizardError,
    calibrate_stereo_from_folders,
)


class CalibrationWizardTests(unittest.TestCase):
    def test_missing_folder_has_actionable_error(self):
        with self.assertRaisesRegex(CalibrationWizardError, "目录不存在"):
            calibrate_stereo_from_folders("missing-left", "missing-right")

    def test_mismatched_folder_counts_are_rejected_before_cv(self):
        with tempfile.TemporaryDirectory() as root:
            left = Path(root) / "left"
            right = Path(root) / "right"
            left.mkdir()
            right.mkdir()
            (left / "0001.png").write_bytes(b"not-an-image")
            (right / "0001.png").write_bytes(b"not-an-image")
            (right / "0002.png").write_bytes(b"not-an-image")
            with self.assertRaisesRegex(CalibrationWizardError, "数量不同"):
                calibrate_stereo_from_folders(left, right)

    def test_offline_folder_calibration_uses_one_detector_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            left = Path(root) / "left"
            right = Path(root) / "right"
            left.mkdir()
            right.mkdir()
            for index in range(8):
                (left / f"{index:02}.png").write_bytes(b"x")
                (right / f"{index:02}.png").write_bytes(b"x")
            observation = BoardObservation(
                np.zeros((54, 2), dtype=float), 100.0, (1, 1), "classic"
            )
            rejected = SimpleNamespace(
                validated=False,
                rejection_reasons=["测试质量门禁"],
            )
            with (
                mock.patch(
                    "pipe_twin.calibration_wizard.cv2.imread",
                    return_value=np.zeros((480, 640, 3), dtype=np.uint8),
                ),
                mock.patch(
                    "pipe_twin.calibration_wizard.detect_board_corners",
                    return_value=observation,
                ) as detector,
                mock.patch(
                    "pipe_twin.calibration_wizard.solve_stereo_calibration",
                    return_value=rejected,
                ),
            ):
                with self.assertRaisesRegex(CalibrationWizardError, "质量门禁未通过"):
                    calibrate_stereo_from_folders(left, right)
            self.assertTrue(detector.call_args_list)
            self.assertTrue(
                all(call.kwargs.get("use_sb") is False for call in detector.call_args_list)
            )


if __name__ == "__main__":
    unittest.main()
