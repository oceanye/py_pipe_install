"""Closed camera windows release ownership only after late opens are disposed."""
from types import SimpleNamespace
import unittest
from unittest import mock

from pipe_twin.capture_gui import StereoCameraDialog
from pipe_twin.calibration_wizard import ChessboardWizardDialog
from pipe_twin.stereo_camera import LAYOUT_SIDE_BY_SIDE_LR


class PendingCameraOwnershipTests(unittest.TestCase):
    def dialog(self, kind):
        dialog = object.__new__(kind)
        dialog._opening = dialog._closing = dialog._probing_modes = False
        dialog.open_after_id = dialog.solve_after_id = None
        dialog._detector_error_keys, dialog._sb_disabled_roles = set(), set()
        dialog._indices = lambda: (0, None)
        dialog._layout = lambda: LAYOUT_SIDE_BY_SIDE_LR
        dialog._eye_size = lambda: (640, 480)
        dialog.calibration = SimpleNamespace(left=SimpleNamespace(width=640, height=480))
        dialog.exposure_preset = mock.Mock(get=mock.Mock(return_value='自动曝光'))
        for name in ('message', 'exposure_status', 'left_status', 'right_status', 'window'):
            setattr(dialog, name, mock.Mock())
        dialog.stop = mock.Mock()
        return dialog

    def run_case(self, kind, order):
        dialog = self.dialog(kind)
        target = ('pipe_twin.capture_gui.run_in_background' if kind is StereoCameraDialog
                  else 'pipe_twin.stereo_camera.run_in_background')
        with mock.patch(target) as run:
            self.assertTrue(dialog.start())
        callback = run.call_args.args[1]
        session = mock.Mock()
        # Holding ownership until close() returns prevents monitor/device overlap.
        session.close.side_effect = lambda: self.assertTrue(dialog._opening)
        if order == 'ready_then_close':
            callback(session, None)
            self.assertTrue(dialog._opening)
            dialog.close()
        elif order == 'close_then_ready':
            dialog.close()
            self.assertTrue(dialog._opening)
            callback(session, None)
        else:
            dialog.close()
            callback(None, OSError('open failed'))
        self.assertFalse(dialog._opening)
        self.assertTrue(dialog._closing)
        if order != 'failed_after_close':
            session.close.assert_called_once()
        dialog.window.destroy.assert_called_once()

    def test_pending_grab_camera_lifetime(self):
        for order in ('ready_then_close', 'close_then_ready', 'failed_after_close'):
            with self.subTest(order=order):
                self.run_case(StereoCameraDialog, order)

    def test_pending_calibration_camera_lifetime(self):
        for order in ('ready_then_close', 'close_then_ready', 'failed_after_close'):
            with self.subTest(order=order):
                self.run_case(ChessboardWizardDialog, order)


if __name__ == '__main__':
    unittest.main()
