from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock, call

import cv2
import numpy as np
import pytest

from pipe_twin import stereo_camera
from pipe_twin.camera_exposure import configure_exposure, exposure_summary
from pipe_twin.camera_lock import CameraBusyError, CameraLease
from pipe_twin.calibration_wizard import ChessboardWizardDialog
from pipe_twin.capture_gui import StereoCameraDialog, _capture_provenance
from pipe_twin.stereo_camera import StereoCameraSession, probe_video_modes
from pipe_twin.workbench_profile import default_profile, validate_profile


@pytest.mark.parametrize("duration,native", [(31.25, -5), (15.625, -6), (7.8125, -7),
                                           (5.0, -8), (3.90625, -8), (1.953125, -9), (0.9765625, -10)])
def test_directshow_manual_presets_are_adjustable(duration, native):
    capture = Mock()
    capture.set.return_value = True
    capture.get.return_value = native
    result = configure_exposure(capture, cv2.CAP_DSHOW, duration)
    assert result["status"] == "DRIVER_REPORTED"
    assert result["target_ms"] == duration
    assert result["reported_ms"] == 1000 * 2 ** native
    assert capture.set.call_args_list == [call(cv2.CAP_PROP_AUTO_EXPOSURE, 0),
                                          call(cv2.CAP_PROP_EXPOSURE, native)]
    if duration > 5:
        assert "慢于 1/200" in exposure_summary({0: result})


@pytest.mark.parametrize("backend,auto_value", [(cv2.CAP_DSHOW, 1), (cv2.CAP_V4L2, 3)])
def test_auto_does_not_report_previous_manual_shutter(backend, auto_value):
    capture = Mock()
    capture.set.return_value = True
    capture.get.side_effect = lambda key: auto_value if key == cv2.CAP_PROP_AUTO_EXPOSURE else -8
    result = configure_exposure(capture, backend, None)
    assert result["status"] == "AUTO"
    assert result["reported_ms"] is None
    assert result["target_ms"] is None
    expected = [call(cv2.CAP_PROP_AUTO_EXPOSURE, auto_value)]
    if backend == cv2.CAP_V4L2:
        expected.insert(0, call(cv2.CAP_PROP_MODE, 0))
    assert capture.set.call_args_list == expected
    assert call(cv2.CAP_PROP_EXPOSURE) not in capture.get.call_args_list


@pytest.mark.parametrize("value", [True, 0, -1, float("nan"), float("inf"), 2000.1, "auto"])
def test_invalid_shutter_does_not_touch_device(value):
    capture = Mock()
    with pytest.raises(ValueError):
        configure_exposure(capture, cv2.CAP_DSHOW, value)
    capture.set.assert_not_called()


class Capture:
    def __init__(self, *, opened=True):
        self.opened = opened
        self.released = False
        self.props = {}

    def isOpened(self):
        return self.opened

    def set(self, key, value):
        self.props[key] = value
        return True

    def get(self, key):
        return self.props.get(key, 0)

    def read(self):
        return True, np.full((4, 12, 3), 90, dtype=np.uint8)

    def release(self):
        self.released = True


def session(factory, **kwargs):
    return StereoCameraSession(layout="side_by_side_left_right", left_index=0, right_index=None,
                               eye_width=6, eye_height=4, backend=cv2.CAP_DSHOW,
                               capture_factory=factory, **kwargs)


def test_live_shutter_switch_keeps_handle_and_updates_capture_metadata(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPE_TWIN_CAMERA_LOCK_DIR", str(tmp_path))
    capture = Capture()
    factory = Mock(return_value=capture)
    with session(factory) as camera:
        camera.set_exposure(None)
        pair = camera.read_pair()
        for role in ("left", "right"):
            record = _capture_provenance(pair.provenance, role)
            assert record["capture_exposure_status"] == "AUTO"
            assert "capture_exposure_ms" not in record
            assert "capture_exposure_target_ms" not in record
        camera.set_exposure(31.25)
        pair = camera.read_pair()
        assert pair.provenance["left"]["capture_exposure_ms"] == 31.25
        assert pair.provenance["right"]["capture_exposure_status"] == "DRIVER_REPORTED"
        assert not capture.released
        factory.assert_called_once()
    assert capture.released


def test_failed_native_open_releases_handle_before_retry(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPE_TWIN_CAMERA_LOCK_DIR", str(tmp_path))
    monkeypatch.setattr(stereo_camera.time, "sleep", lambda _: None)
    failed, ready = Capture(opened=False), Capture()
    attempts = []

    def factory(*_args):
        attempts.append(1)
        if len(attempts) == 1:
            return failed
        assert failed.released
        return ready

    with session(factory) as camera:
        assert camera.read_pair().left.shape == (4, 6, 3)
    assert len(attempts) == 2
    assert ready.released


def test_timed_out_probe_retains_lease_until_native_handle_released(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPE_TWIN_CAMERA_LOCK_DIR", str(tmp_path))
    unblock = threading.Event()
    released = threading.Event()
    opened_preview = threading.Event()
    errors = []

    class BlockingCapture(Capture):
        def read(self):
            assert unblock.wait(3)
            return super().read()

        def release(self):
            super().release()
            released.set()

    def preview_factory(*_args):
        assert released.is_set(), "preview must not race a still-open probe"
        opened_preview.set()
        return Capture()

    def open_preview():
        try:
            with session(preview_factory):
                pass
        except Exception as error:
            errors.append(error)

    try:
        assert probe_video_modes(index=0, capture_factory=lambda *_: BlockingCapture(),
                                 candidates=[(12, 4), (24, 8)], per_mode_timeout_s=0.1) == []
        with pytest.raises(CameraBusyError):
            with CameraLease([0]):
                pass
        worker = threading.Thread(target=open_preview)
        worker.start()
        assert not opened_preview.wait(0.15)
        unblock.set()
        worker.join(3)
        assert not worker.is_alive()
        assert not errors
        assert opened_preview.is_set()
    finally:
        unblock.set()
        released.wait(3)


def test_mode_probe_reuses_one_device_handle(tmp_path, monkeypatch):
    monkeypatch.setenv("PIPE_TWIN_CAMERA_LOCK_DIR", str(tmp_path))
    capture = Capture()
    factory = Mock(return_value=capture)
    assert probe_video_modes(index=0, capture_factory=factory,
                             candidates=[(12, 4), (24, 8)]) == [(12, 4)]
    factory.assert_called_once()
    assert capture.released


@pytest.mark.parametrize("dialog_type", [ChessboardWizardDialog, StereoCameraDialog])
def test_both_dialogs_apply_selected_shutter_without_reopening(dialog_type):
    dialog = object.__new__(dialog_type)
    dialog.session = SimpleNamespace(set_exposure=Mock(), exposure_summary="自动曝光")
    dialog.exposure_preset = Mock(get=Mock(return_value="自动曝光"))
    dialog.exposure_status = Mock()
    dialog._apply_exposure()
    dialog.session.set_exposure.assert_called_once_with(None)
    dialog.exposure_status.set.assert_called_once_with("自动曝光")


@pytest.mark.parametrize("duration", [None, 5.0, 31.25, 2000.0])
def test_profile_retains_shutter_selection(duration):
    profile = default_profile()
    profile["camera"]["exposure_ms"] = duration
    assert validate_profile(profile)["camera"]["exposure_ms"] == duration
