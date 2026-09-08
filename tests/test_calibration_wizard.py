import tempfile
import unittest
from pathlib import Path

from pipe_twin.calibration_wizard import CalibrationWizardError, calibrate_stereo_from_folders


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


if __name__ == "__main__":
    unittest.main()
