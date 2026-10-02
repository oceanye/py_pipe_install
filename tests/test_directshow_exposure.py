from unittest.mock import Mock

import cv2
import pytest

from pipe_twin.camera_exposure import configure_exposure
from pipe_twin.directshow_controls import DirectShowControlError


class NativeControl:
    device_path = "test-device"

    def __init__(self, *, flags=None, value=None, step=1):
        self.override_flags, self.override_value, self.step = flags, value, step
        self.writes = []
        self.current = {"value": -2, "flags": 1}
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def exposure_range(self):
        return {"minimum": -11, "maximum": -2, "step": self.step, "default": -6, "capabilities": 3}

    def exposure(self):
        return dict(self.current)

    def set_exposure(self, value, flags):
        self.writes.append((value, flags))
        self.current = {"value": value if self.override_value is None else self.override_value,
                        "flags": flags if self.override_flags is None else self.override_flags}


def apply(control, duration):
    capture = Mock()
    indices = []
    def factory(index):
        indices.append(index)
        return control
    result = configure_exposure(capture, cv2.CAP_DSHOW, duration, device_index=2, native_factory=factory)
    assert indices == [2]
    capture.set.assert_not_called()
    assert control.closed
    return result


def test_native_manual_writes_value_and_manual_flag_together():
    control = NativeControl()
    result = apply(control, 5.0)
    assert control.writes == [(-8, 2)]
    assert result["status"] == "DRIVER_REPORTED"
    assert result["readback_flags"] == 2
    assert result["reported_ms"] == 3.90625
    assert result["native_range"]["maximum"] == -2


def test_native_auto_confirms_mode_without_reusing_manual_time():
    control = NativeControl(value=-2)
    result = apply(control, None)
    assert control.writes == [(-6, 1)]
    assert result["status"] == "AUTO"
    assert result["reported_ms"] is None


@pytest.mark.parametrize("duration", [0.1, 500.0, 1000.0, 2000.0])
def test_out_of_driver_range_is_not_applied_or_claimed(duration):
    control = NativeControl()
    result = apply(control, duration)
    assert control.writes == []
    assert result["status"] == "UNCONFIRMED"
    assert result["reported_ms"] is None
    assert result["observed_state"] == {"value": -2, "flags": 1}


@pytest.mark.parametrize("duration,flags,value", [(5., 1, -8), (5., 2, -6), (None, 2, -6), (None, 3, -6)])
def test_ignored_mode_or_value_is_never_reported_as_success(duration, flags, value):
    result = apply(NativeControl(flags=flags, value=value), duration)
    assert result["status"] == "UNCONFIRMED"


def test_native_failure_does_not_fall_back_to_unverified_flags():
    capture = Mock()
    def unavailable(index):
        raise DirectShowControlError("device unavailable")
    result = configure_exposure(capture, cv2.CAP_DSHOW, None, device_index=0, native_factory=unavailable)
    assert result["status"] == "UNCONFIRMED"
    assert "device unavailable" in result["reason"]
    capture.set.assert_not_called()


def test_native_coarse_step_never_selects_a_slower_shutter():
    control = NativeControl(step=2)
    result = apply(control, 5.)
    assert control.writes == [(-9, 2)]
    assert result["reported_ms"] <= 5.


def test_unavailable_backend_is_not_reported_as_camera_busy(monkeypatch):
    from pipe_twin.stereo_camera import StereoCameraSession, StereoCameraError
    monkeypatch.setattr(cv2.videoio_registry, "hasBackend", lambda backend: False)
    session = StereoCameraSession(layout="side_by_side_left_right", left_index=0, right_index=None,
                                  eye_width=1920, eye_height=1080, backend=cv2.CAP_MSMF)
    with pytest.raises(StereoCameraError, match="OpenCV.*1400"):
        session._open_one(0, 3840, 1080)
