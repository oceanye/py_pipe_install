"""Request a short manual shutter using each capture backend's native units."""

from __future__ import annotations

import math
from typing import Any

import cv2


TARGET_EXPOSURE_MS = 5.0  # 1/200 s maximum, to limit motion blur.


def configure_exposure(capture: Any, backend: int) -> dict[str, Any]:
    """Report driver acceptance/readback; this is not a sensor timing measurement.

    DirectShow exposure is integer log2(seconds), so -8 = 1/256 s is
    the closest supported step no slower than 1/200 s. V4L2 uses 100 us.
    Never send either unit convention to an unknown backend.
    """
    result: dict[str, Any] = {
        "target_ms": TARGET_EXPOSURE_MS,
        "status": "UNCONFIRMED",
        "reported_ms": None,
    }
    try:
        if backend == cv2.CAP_ANY:
            backend = int(capture.get(cv2.CAP_PROP_BACKEND))
        result["backend"] = backend
        if backend == cv2.CAP_DSHOW:
            native_value, manual_value = -8, 0
        elif backend == cv2.CAP_V4L2:
            # Disable OpenCV's optional [0, 1] property normalization.
            if not capture.set(cv2.CAP_PROP_MODE, 0):
                result["reason"] = "无法切换到原生曝光单位"
                return result
            native_value, manual_value = 50, 1
        else:
            result["reason"] = "当前相机接口暂不支持程序设置快门"
            return result
        result["requested_native"] = native_value
        result["manual_accepted"] = bool(capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual_value))
        result["shutter_accepted"] = bool(capture.set(cv2.CAP_PROP_EXPOSURE, native_value))
        value = float(capture.get(cv2.CAP_PROP_EXPOSURE))
        if math.isfinite(value):
            result["readback_native"] = value
            if backend == cv2.CAP_DSHOW and -30 <= value <= -2:
                result["reported_ms"] = 1000 * 2 ** value
            elif backend == cv2.CAP_V4L2 and value > 0:
                result["reported_ms"] = value / 10
        manual_confirmed = result["manual_accepted"]
        if backend == cv2.CAP_V4L2:
            manual_confirmed = manual_confirmed and capture.get(cv2.CAP_PROP_AUTO_EXPOSURE) == manual_value
        # DirectShow does not expose an AUTO_EXPOSURE getter in OpenCV 4.13.
        # Its successful manual command plus exposure readback is all we can
        # confirm through this API; do not claim hardware timing verification.
        reported_ms = result["reported_ms"]
        if (
            manual_confirmed
            and result["shutter_accepted"]
            and reported_ms is not None
            and 0 < reported_ms <= TARGET_EXPOSURE_MS
        ):
            result["status"] = "DRIVER_REPORTED"
        else:
            result["reason"] = "驱动未确认手动快门不慢于 1/200 秒"
    except (cv2.error, OSError, TypeError, ValueError, OverflowError) as error:
        result["reason"] = f"快门设置失败：{error}"
    return result


def exposure_summary(settings: dict[int, dict[str, Any]]) -> str:
    if not settings:
        return "快门目标 ≤1/200 秒（5 ms）；待连接"
    values = []
    for index, record in settings.items():
        if record["status"] == "DRIVER_REPORTED":
            duration = record["reported_ms"]
            values.append(f"设备 {index}：驱动回报 1/{1000 / duration:g} 秒（{duration:.3f} ms）")
        else:
            values.append(f"设备 {index}：未确认，请在相机设置中手动调整快门")
    return "快门目标 ≤1/200 秒；" + "；".join(values)
