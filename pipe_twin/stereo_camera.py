"""Direct, auditable capture from one side-by-side or two UVC cameras."""

from __future__ import annotations

import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import cv2
import numpy as np


LAYOUT_SIDE_BY_SIDE_LR = "side_by_side_left_right"
LAYOUT_SIDE_BY_SIDE_RL = "side_by_side_right_left"
LAYOUT_SEPARATE = "separate_devices"
SUPPORTED_LAYOUTS = {
    LAYOUT_SIDE_BY_SIDE_LR,
    LAYOUT_SIDE_BY_SIDE_RL,
    LAYOUT_SEPARATE,
}


class StereoCameraError(ValueError):
    """Raised when a camera pair cannot satisfy the capture contract."""


@dataclass(frozen=True)
class CapturedStereoPair:
    left: np.ndarray
    right: np.ndarray
    left_captured_at: str
    right_captured_at: str
    sync_delta_ms: float
    timestamp_source: str
    provenance: dict[str, dict[str, Any]]


def _host_timestamp(nanoseconds: int) -> str:
    return datetime.fromtimestamp(
        nanoseconds / 1_000_000_000,
        tz=timezone.utc,
    ).astimezone().isoformat(timespec="milliseconds")


def _default_backend() -> int:
    return cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY


def probe_video_devices(
    *,
    maximum_index: int = 7,
    backend: int | None = None,
    capture_factory: Callable[..., Any] = cv2.VideoCapture,
    per_index_timeout_s: float = 5.0,
    progress: Callable[[int, str], None] | None = None,
    cancelled: Callable[[], bool] | None = None,
) -> list[dict[str, Any]]:
    """Return OpenCV camera indices that can be opened, without retaining them.

    A single misbehaving capture driver (commonly a virtual camera) can block
    inside ``VideoCapture`` forever, so every index is probed on a worker
    thread bounded by ``per_index_timeout_s``.  A timed-out index is skipped
    and reported via ``progress``; its worker keeps running as a daemon and
    releases the device whenever the driver finally answers.  ``cancelled``
    is polled between indices so the UI can abort a long sweep.
    """
    if type(maximum_index) is not int or not 0 <= maximum_index <= 32:
        raise StereoCameraError("maximum_index must be an integer between 0 and 32")
    selected_backend = _default_backend() if backend is None else backend
    devices: list[dict[str, Any]] = []

    def _probe_one(index: int) -> dict[str, Any] | None:
        capture = capture_factory(index, selected_backend)
        try:
            if not capture.isOpened():
                return None
            return {
                "index": index,
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": float(capture.get(cv2.CAP_PROP_FPS)),
                "backend": int(selected_backend),
            }
        except Exception:
            # Driver errors behave like "no device here"; never kill the worker.
            return None
        finally:
            capture.release()

    for index in range(maximum_index + 1):
        if cancelled is not None and cancelled():
            break
        if progress is not None:
            progress(index, "opening")
        outcome: dict[str, Any] = {}
        worker = threading.Thread(
            target=lambda i=index: outcome.setdefault("device", _probe_one(i)),
            daemon=True,
        )
        worker.start()
        worker.join(per_index_timeout_s)
        if worker.is_alive() and "device" not in outcome:
            if progress is not None:
                progress(index, "timeout")
            continue
        device = outcome.get("device")
        if device is not None:
            devices.append(device)
        elif progress is not None:
            progress(index, "empty")
    return devices


def run_in_background(
    task: Callable[[], Any],
    on_result: Callable[[Any, Exception | None], None],
) -> None:
    """Run ``task`` on a daemon thread and deliver its outcome once.

    Opening a UVC device or probing indices can block for seconds on
    DirectShow, so GUI callers must not run it on the Tk thread.  ``on_result``
    executes on the worker thread and must not touch Tk widgets — store the
    outcome and let the UI thread poll for it.  Native ``cv2.error`` exceptions
    are reported here instead of escaping into the Tk callback.
    """

    def _worker() -> None:
        try:
            result = task()
        except Exception as error:  # noqa: BLE001 - reported to the caller
            on_result(None, error)
            return
        on_result(result, None)

    threading.Thread(target=_worker, daemon=True).start()


class StereoCameraSession:
    """Own live camera handles and produce paired, unmodified BGR frames."""

    def __init__(
        self,
        *,
        layout: str,
        left_index: int,
        right_index: int | None,
        eye_width: int,
        eye_height: int,
        backend: int | None = None,
        capture_factory: Callable[..., Any] = cv2.VideoCapture,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if layout not in SUPPORTED_LAYOUTS:
            raise StereoCameraError(f"unsupported stereo camera layout: {layout!r}")
        for name, value in (("left_index", left_index), ("eye_width", eye_width), ("eye_height", eye_height)):
            if type(value) is not int or value < 0 or (name != "left_index" and value <= 0):
                raise StereoCameraError(f"{name} must be a valid integer")
        if layout == LAYOUT_SEPARATE:
            if type(right_index) is not int or right_index < 0:
                raise StereoCameraError("right_index must be a non-negative integer")
            if right_index == left_index:
                raise StereoCameraError("left and right cameras must use different indices")
        self.layout = layout
        self.left_index = left_index
        self.right_index = right_index
        self.eye_width = eye_width
        self.eye_height = eye_height
        self.backend = _default_backend() if backend is None else backend
        self.capture_factory = capture_factory
        self.clock_ns = clock_ns
        self.left_capture: Any | None = None
        self.right_capture: Any | None = None

    def _open_one(self, index: int, width: int, height: int) -> Any:
        capture = self.capture_factory(index, self.backend)
        if not capture.isOpened():
            capture.release()
            raise StereoCameraError(f"无法打开相机索引 {index}")
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def open(self) -> None:
        self.close()
        try:
            if self.layout in {LAYOUT_SIDE_BY_SIDE_LR, LAYOUT_SIDE_BY_SIDE_RL}:
                self.left_capture = self._open_one(
                    self.left_index,
                    self.eye_width * 2,
                    self.eye_height,
                )
            else:
                self.left_capture = self._open_one(
                    self.left_index, self.eye_width, self.eye_height
                )
                self.right_capture = self._open_one(
                    int(self.right_index), self.eye_width, self.eye_height
                )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        for capture in (self.left_capture, self.right_capture):
            if capture is not None:
                capture.release()
        self.left_capture = None
        self.right_capture = None

    def _side_by_side_pair(self) -> CapturedStereoPair:
        capture = self.left_capture
        if capture is None:
            raise StereoCameraError("相机预览尚未打开")
        started = self.clock_ns()
        ok, frame = capture.read()
        finished = self.clock_ns()
        if not ok or not isinstance(frame, np.ndarray):
            raise StereoCameraError(f"相机索引 {self.left_index} 读取失败")
        expected_shape = (self.eye_height, self.eye_width * 2)
        if frame.ndim != 3 or frame.shape[:2] != expected_shape or frame.shape[2] != 3:
            raise StereoCameraError(
                f"并排双目流实际为 {frame.shape[1] if frame.ndim >= 2 else '?'}×"
                f"{frame.shape[0] if frame.ndim >= 2 else '?'}；标定要求 "
                f"{self.eye_width * 2}×{self.eye_height}"
            )
        first = frame[:, : self.eye_width].copy()
        second = frame[:, self.eye_width :].copy()
        left, right = (
            (first, second)
            if self.layout == LAYOUT_SIDE_BY_SIDE_LR
            else (second, first)
        )
        timestamp = _host_timestamp((started + finished) // 2)
        order = "LEFT_THEN_RIGHT" if self.layout == LAYOUT_SIDE_BY_SIDE_LR else "RIGHT_THEN_LEFT"
        common = {
            "capture_backend": "OpenCV",
            "capture_layout": "SINGLE_FRAME_SIDE_BY_SIDE",
            "capture_device_index": self.left_index,
            "side_by_side_order": order,
            "capture_sync_method": "SAME_UVC_FRAME",
        }
        return CapturedStereoPair(
            left=left,
            right=right,
            left_captured_at=timestamp,
            right_captured_at=timestamp,
            sync_delta_ms=0.0,
            timestamp_source="HOST_SYSTEM_CLOCK",
            provenance={"left": dict(common), "right": dict(common)},
        )

    def _grab_with_time(self, capture: Any) -> tuple[bool, int]:
        started = self.clock_ns()
        ok = bool(capture.grab())
        finished = self.clock_ns()
        return ok, (started + finished) // 2

    def _separate_pair(self) -> CapturedStereoPair:
        if self.left_capture is None or self.right_capture is None:
            raise StereoCameraError("左右相机预览尚未打开")
        with ThreadPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(self._grab_with_time, self.left_capture)
            right_future = executor.submit(self._grab_with_time, self.right_capture)
            left_ok, left_time = left_future.result()
            right_ok, right_time = right_future.result()
        left_retrieved, left = self.left_capture.retrieve()
        right_retrieved, right = self.right_capture.retrieve()
        if not left_ok or not left_retrieved or not isinstance(left, np.ndarray):
            raise StereoCameraError(f"左相机索引 {self.left_index} 读取失败")
        if not right_ok or not right_retrieved or not isinstance(right, np.ndarray):
            raise StereoCameraError(f"右相机索引 {self.right_index} 读取失败")
        expected_shape = (self.eye_height, self.eye_width)
        for role, frame in (("左", left), ("右", right)):
            if frame.ndim != 3 or frame.shape[:2] != expected_shape or frame.shape[2] != 3:
                raise StereoCameraError(
                    f"{role}相机实际为 {frame.shape[1] if frame.ndim >= 2 else '?'}×"
                    f"{frame.shape[0] if frame.ndim >= 2 else '?'}；标定要求 "
                    f"{self.eye_width}×{self.eye_height}"
                )
        sync_delta_ms = abs(right_time - left_time) / 1_000_000
        common = {
            "capture_backend": "OpenCV",
            "capture_layout": "SEPARATE_UVC_DEVICES",
            "capture_sync_method": "PARALLEL_HOST_GRAB",
        }
        return CapturedStereoPair(
            left=left.copy(),
            right=right.copy(),
            left_captured_at=_host_timestamp(left_time),
            right_captured_at=_host_timestamp(right_time),
            sync_delta_ms=sync_delta_ms,
            timestamp_source="HOST_SYSTEM_CLOCK",
            provenance={
                "left": common | {"capture_device_index": self.left_index},
                "right": common | {"capture_device_index": self.right_index},
            },
        )

    def read_pair(self) -> CapturedStereoPair:
        if self.layout in {LAYOUT_SIDE_BY_SIDE_LR, LAYOUT_SIDE_BY_SIDE_RL}:
            return self._side_by_side_pair()
        return self._separate_pair()

    def __enter__(self) -> "StereoCameraSession":
        self.open()
        return self

    def __exit__(self, *_error: object) -> None:
        self.close()


__all__ = [
    "CapturedStereoPair",
    "LAYOUT_SEPARATE",
    "LAYOUT_SIDE_BY_SIDE_LR",
    "LAYOUT_SIDE_BY_SIDE_RL",
    "StereoCameraError",
    "StereoCameraSession",
    "probe_video_devices",
    "run_in_background",
]
