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
                    "pipe_twin.calibration_wizard.read_calibration_image",
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

    def test_calibration_image_reader_supports_unicode_windows_paths(self):
        import cv2

        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "原厂标定照片.png"
            image = np.full((7, 11, 3), 127, dtype=np.uint8)
            ok, encoded = cv2.imencode(".png", image)
            self.assertTrue(ok)
            path.write_bytes(bytes(encoded))
            from pipe_twin.calibration_wizard import read_calibration_image

            loaded = read_calibration_image(path)
            self.assertIsNotNone(loaded)
            self.assertEqual(tuple(loaded.shape), (7, 11, 3))

    def test_offline_sb_fallback_uses_one_detector_for_each_pair(self):
        with tempfile.TemporaryDirectory() as root:
            left = Path(root) / "left"
            right = Path(root) / "right"
            left.mkdir()
            right.mkdir()
            for index in range(8):
                (left / f"{index:02}.png").write_bytes(b"x")
                (right / f"{index:02}.png").write_bytes(b"x")
            sb_observation = BoardObservation(
                np.zeros((54, 2), dtype=float), 100.0, (1, 1), "sb"
            )
            rejected = SimpleNamespace(
                validated=False,
                rejection_reasons=["测试质量门禁"],
            )
            detector_modes = []

            def detector(_image, *, use_sb=True, **_kwargs):
                detector_modes.append(use_sb)
                return sb_observation if use_sb else None

            with (
                mock.patch(
                    "pipe_twin.calibration_wizard.read_calibration_image",
                    return_value=np.zeros((480, 640, 3), dtype=np.uint8),
                ),
                mock.patch(
                    "pipe_twin.calibration_wizard.detect_board_corners",
                    side_effect=detector,
                ),
                mock.patch(
                    "pipe_twin.calibration_wizard.solve_stereo_calibration",
                    return_value=rejected,
                ),
            ):
                with self.assertRaisesRegex(CalibrationWizardError, "质量门禁未通过"):
                    calibrate_stereo_from_folders(left, right)
            self.assertEqual(detector_modes, [False] * 16 + [True] * 16)


if __name__ == "__main__":
    unittest.main()
