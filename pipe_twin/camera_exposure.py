"""Apply automatic or selected manual exposure in backend-native units."""

from __future__ import annotations

import math
from typing import Any

import cv2

from .directshow_controls import DirectShowCameraControl


TARGET_EXPOSURE_MS = 5.0  # 1/200 s maximum, to limit motion blur.
MIN_EXPOSURE_MS = 0.1
MAX_EXPOSURE_MS = 2000.0
EXPOSURE_PRESETS = {
    "自动曝光": None,
    "1 秒": 1000.0,
    "2 秒": 2000.0,
    "1/30 秒（Windows 1/32）": 1000 / 30,
    "1/32 秒": 1000 / 32,
    "1/64 秒": 1000 / 64,
    "1/128 秒": 1000 / 128,
    "1/200 秒（Windows 1/256）": TARGET_EXPOSURE_MS,
    "1/256 秒": 1000 / 256,
    "1/512 秒": 1000 / 512,
    "1/1024 秒": 1000 / 1024,
}
DEFAULT_EXPOSURE_PRESET = "1/200 秒（Windows 1/256）"


def exposure_preset_label(duration: float | None) -> str:
    return next((label for label, value in EXPOSURE_PRESETS.items() if value == duration),
                DEFAULT_EXPOSURE_PRESET)


def configure_exposure(capture: Any, backend: int,
                       exposure_ms: float | None = TARGET_EXPOSURE_MS, *,
                       device_index: int | None = None,
                       native_factory: Any = DirectShowCameraControl) -> dict[str, Any]:
    """Report driver acceptance/readback; this is not a sensor timing measurement.

    DirectShow exposure is integer log2(seconds), so -8 = 1/256 s is
    the closest supported step no slower than 1/200 s; native 0 and 1
    correspond to 1 s and 2 s. V4L2 uses 100 us.
    Never send either unit convention to an unknown backend.
    """
    if exposure_ms is not None and (
        type(exposure_ms) not in (int, float) or not math.isfinite(exposure_ms)
        or not MIN_EXPOSURE_MS <= exposure_ms <= MAX_EXPOSURE_MS
    ):
        raise ValueError(
            f"手动快门必须在 {MIN_EXPOSURE_MS:g} 到 {MAX_EXPOSURE_MS:g} ms 之间；None 表示自动曝光"
        )
    result: dict[str, Any] = {
        "target_ms": exposure_ms,
        "status": "UNCONFIRMED",
        "reported_ms": None,
    }
    try:
        if backend == cv2.CAP_ANY:
            backend = int(capture.get(cv2.CAP_PROP_BACKEND))
        result["backend"] = backend
        if backend == cv2.CAP_DSHOW and device_index is not None:
            result["control_api"] = "IAMCameraControl"
            with native_factory(device_index) as control:
                limits = control.exposure_range()
                result["native_range"] = limits
                result["device_path"] = control.device_path
                mode = 1 if exposure_ms is None else 2
                if not limits["capabilities"] & mode:
                    result["reason"] = "驱动不支持所选曝光模式"
                    return result
                if exposure_ms is None:
                    value = limits["default"]
                else:
                    value = math.floor(math.log2(exposure_ms / 1000))
                    result["requested_native"] = value
                    if not limits["minimum"] <= value <= limits["maximum"]:
                        result["reason"] = "所选快门超出驱动声明范围；未写入不支持的值"
                        result["observed_state"] = control.exposure()
                        return result
                    step = limits["step"]
                    if step <= 0:
                        result["reason"] = "驱动曝光步长无效"
                        return result
                    value = limits["minimum"] + ((value - limits["minimum"]) // step) * step
                    result["requested_native"] = value
                control.set_exposure(value, mode)
                state = control.exposure()
                result.update(readback_native=state["value"], readback_flags=state["flags"])
                if exposure_ms is None:
                    result["auto_accepted"] = state["flags"] == 1
                    if result["auto_accepted"]:
                        result["status"] = "AUTO"
                else:
                    result["manual_accepted"] = state["flags"] == 2
                    result["shutter_accepted"] = state["value"] == value
                    if -30 <= state["value"] <= 1 and result["manual_accepted"]:
                        result["reported_ms"] = 1000 * 2 ** state["value"]
                    if result["manual_accepted"] and result["shutter_accepted"] and result["reported_ms"] is not None:
                        result["status"] = "DRIVER_REPORTED"
                if result["status"] == "UNCONFIRMED":
                    result["reason"] = "驱动曝光值或自动/手动模式读回不一致"
                return result
        if backend not in {cv2.CAP_DSHOW, cv2.CAP_V4L2}:
            result["reason"] = "当前相机接口暂不支持程序设置快门"
            return result
        if backend == cv2.CAP_V4L2:
            # Both manual shutter and auto mode use native control values.
            if not capture.set(cv2.CAP_PROP_MODE, 0):
                result["reason"] = "无法切换到原生曝光单位"
                return result
        if exposure_ms is None:
            auto_value = 1 if backend == cv2.CAP_DSHOW else 3
            accepted = bool(capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_value))
            if backend == cv2.CAP_V4L2:
                accepted = accepted and capture.get(cv2.CAP_PROP_AUTO_EXPOSURE) == auto_value
            result["auto_accepted"] = accepted
            if accepted:
                result["status"] = "AUTO"
            else:
                result["reason"] = "驱动未确认自动曝光设置"
            # Some drivers keep returning the previous manual shutter in auto.
            # That value is not the live exposure duration.
            return result
        if backend == cv2.CAP_DSHOW:
            native_value, manual_value = math.floor(math.log2(exposure_ms / 1000)), 0
        elif backend == cv2.CAP_V4L2:
            native_value, manual_value = max(1, math.floor(exposure_ms * 10)), 1
        result["requested_native"] = native_value
        result["manual_accepted"] = bool(capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual_value))
        result["shutter_accepted"] = bool(capture.set(cv2.CAP_PROP_EXPOSURE, native_value))
        if backend == cv2.CAP_DSHOW:
            # OpenCV's exposure-value setter sends flags=0. Reassert MANUAL
            # after the value for callers without a native device binding.
            result["manual_accepted"] = bool(capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, manual_value)) and result["manual_accepted"]
        value = float(capture.get(cv2.CAP_PROP_EXPOSURE))
        if math.isfinite(value):
            result["readback_native"] = value
            if backend == cv2.CAP_DSHOW and -30 <= value <= 1:
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
            and 0 < reported_ms <= exposure_ms
            and math.isclose(value, native_value, abs_tol=1e-6)
        ):
            result["status"] = "DRIVER_REPORTED"
        else:
            result["reason"] = "驱动未确认所选手动快门"
    except (cv2.error, OSError, TypeError, ValueError, OverflowError) as error:
        result["reason"] = f"快门设置失败：{error}"
    return result


def exposure_summary(settings: dict[int, dict[str, Any]]) -> str:
    if not settings:
        return "快门待连接；可选自动曝光或手动档位"
    values = []
    for index, record in settings.items():
        if record["status"] == "AUTO":
            values.append(f"设备 {index}：自动曝光；移动棋盘时注意运动模糊")
        elif record["status"] == "DRIVER_REPORTED":
            duration = record["reported_ms"]
            values.append(f"设备 {index}：驱动回报 1/{1000 / duration:g} 秒（{duration:.3f} ms）")
            if duration > TARGET_EXPOSURE_MS:
                values[-1] += "，慢于 1/200 秒，请保持目标静止"
        else:
            values.append(f"设备 {index}：未确认，请在相机设置中手动调整快门")
    return "；".join(values)
