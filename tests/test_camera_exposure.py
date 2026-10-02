from __future__ import annotations

import unittest
from unittest.mock import Mock

import cv2

from pipe_twin.camera_exposure import configure_exposure, exposure_preset_label, exposure_summary
from pipe_twin.capture_gui import _capture_provenance


class ExposureTests(unittest.TestCase):
    @staticmethod
    def camera(backend, readback, *, accepted=True, auto=1):
        capture = Mock()
        capture.set.return_value = accepted
        capture.get.side_effect = lambda key: {
            cv2.CAP_PROP_BACKEND: backend,
            cv2.CAP_PROP_EXPOSURE: readback,
            cv2.CAP_PROP_AUTO_EXPOSURE: auto,
        }[key]
        return capture

    def test_directshow_requests_manual_1_256_and_reports_readback(self):
        capture = self.camera(cv2.CAP_DSHOW, -8)
        result = configure_exposure(capture, cv2.CAP_DSHOW)
        self.assertEqual(result["status"], "DRIVER_REPORTED")
        self.assertEqual(result["reported_ms"], 3.90625)
        self.assertEqual(capture.set.call_args_list, [
            unittest.mock.call(cv2.CAP_PROP_AUTO_EXPOSURE, 0),
            unittest.mock.call(cv2.CAP_PROP_EXPOSURE, -8),
        ])
        capture.get.assert_called_once_with(cv2.CAP_PROP_EXPOSURE)
        self.assertIn("1/256", exposure_summary({0: result}))

    def test_one_thirtieth_preset_maps_to_requested_milliseconds(self):
        self.assertEqual(exposure_preset_label(1000 / 30), "1/30 秒（Windows 1/32）")

    def test_directshow_maps_one_thirtieth_request_to_one_thirty_second_step(self):
        result = configure_exposure(self.camera(cv2.CAP_DSHOW, -5), cv2.CAP_DSHOW, 1000 / 30)
        self.assertEqual(result["status"], "DRIVER_REPORTED")
        self.assertEqual(result["requested_native"], -5)
        self.assertEqual(result["reported_ms"], 31.25)

    def test_v4l2_detects_backend_and_uses_100_microsecond_units(self):
        capture = self.camera(cv2.CAP_V4L2, 50)
        result = configure_exposure(capture, cv2.CAP_ANY)
        self.assertEqual(result["status"], "DRIVER_REPORTED")
        self.assertEqual(result["reported_ms"], 5.0)
        self.assertEqual(capture.set.call_args_list, [
            unittest.mock.call(cv2.CAP_PROP_MODE, 0),
            unittest.mock.call(cv2.CAP_PROP_AUTO_EXPOSURE, 1),
            unittest.mock.call(cv2.CAP_PROP_EXPOSURE, 50),
        ])

    def test_unknown_backend_is_not_sent_guessed_exposure_units(self):
        capture = self.camera(cv2.CAP_AVFOUNDATION, 0.005)
        result = configure_exposure(capture, cv2.CAP_ANY)
        self.assertEqual(result["status"], "UNCONFIRMED")
        capture.set.assert_not_called()

    def test_ignored_or_invalid_readback_is_not_reported_as_applied(self):
        for value in (-6, -1, 0, float("nan"), float("inf")):
            with self.subTest(value=value):
                result = configure_exposure(self.camera(cv2.CAP_DSHOW, value), cv2.CAP_DSHOW)
                self.assertEqual(result["status"], "UNCONFIRMED")
                self.assertIn("未确认", exposure_summary({0: result}))

    def test_rejected_manual_command_or_remaining_auto_mode_is_unconfirmed(self):
        for capture, backend in (
            (self.camera(cv2.CAP_DSHOW, -8, accepted=False), cv2.CAP_DSHOW),
            (self.camera(cv2.CAP_V4L2, 50, auto=3), cv2.CAP_V4L2),
        ):
            with self.subTest(backend=backend):
                self.assertEqual(configure_exposure(capture, backend)["status"], "UNCONFIRMED")

    def test_driver_exception_keeps_preview_available_with_explicit_status(self):
        capture = self.camera(cv2.CAP_DSHOW, -8)
        capture.set.side_effect = cv2.error("unsupported control")
        result = configure_exposure(capture, cv2.CAP_DSHOW)
        self.assertEqual(result["status"], "UNCONFIRMED")
        self.assertIn("unsupported control", result["reason"])

    def test_provenance_rejects_nonfinite_or_malformed_exposure_values(self):
        for key, values in (
            ("capture_exposure_ms", [True, -1, 0, float("nan"), float("inf"), "5"]),
            ("capture_exposure_status", [[], {}, "success"]),
        ):
            for value in values:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    _capture_provenance({"left": {key: value}}, "left")


if __name__ == "__main__":
    unittest.main()
