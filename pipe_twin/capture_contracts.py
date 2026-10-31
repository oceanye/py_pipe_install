"""Pure contracts and model helpers for field capture.

The capture dialogs keep their window and camera lifecycle code separate from
the validation of timestamps, calibration provenance and CAD pipe catalogs.
This module is intentionally free of Tk imports.
"""

from __future__ import annotations

import copy
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .cad_model import load_cad_scene


_CAPTURE_PROVENANCE_KEYS = {
    "capture_backend",
    "capture_layout",
    "capture_device_index",
    "side_by_side_order",
    "capture_sync_method",
    "capture_exposure_target_ms",
    "capture_exposure_ms",
    "capture_exposure_status",
}

_NON_FIELD_CALIBRATION_MARKERS = (
    "SYNTHETIC",
    "DEMO",
    "EXAMPLE",
    "REPLACE_WITH_REAL",
)


def normalize_capture_time(value: str, *, field: str = "拍摄时间") -> tuple[str, bool]:
    """Accept common operator input and return millisecond ISO-8601 with an offset.

    The boolean indicates that the workstation's local time zone was added.
    A date without a clock time is never sufficient for stereo synchronization.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}不能为空；请选择照片后核对自动填写的时间。")
    text = value.strip()
    text = (
        text.replace("年", "-")
        .replace("月", "-")
        .replace("日", " ")
        .replace("时", ":")
        .replace("分", ":")
        .replace("秒", "")
        .replace("/", "-")
        .strip()
        .rstrip(":")
    )
    text = re.sub(r"\s+", " ", text)
    if not re.search(r"[T ]\d{1,2}:\d{1,2}", text):
        raise ValueError(f"{field}必须包含时、分，例如 2026-09-07 15:30:20。")
    match = re.fullmatch(
        r"(\d{4})-(\d{1,2})-(\d{1,2})([T ])(\d{1,2}):(\d{1,2})"
        r"(?::(\d{1,2})(\.\d{1,6})?)?([+-]\d{2}:?\d{2}|[Zz])?",
        text,
    )
    if match:
        year, month, day, separator, hour, minute, second, fraction, offset = match.groups()
        second = second or "00"
        if offset and offset.lower() == "z":
            offset = "+00:00"
        elif offset and ":" not in offset:
            offset = offset[:3] + ":" + offset[3:]
        text = (
            f"{int(year):04d}-{int(month):02d}-{int(day):02d}{separator}"
            f"{int(hour):02d}:{int(minute):02d}:{int(second):02d}"
            f"{fraction or ''}{offset or ''}"
        )
    elif text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(
            f"{field}格式无法识别；可输入 2026-09-07 15:30:20 或 "
            "2026-09-07T15:30:20.000+08:00。"
        ) from error
    assumed_local_zone = parsed.tzinfo is None or parsed.utcoffset() is None
    if assumed_local_zone:
        parsed = parsed.astimezone()
    return parsed.isoformat(timespec="milliseconds"), assumed_local_zone


def photo_file_time(path: str | Path) -> str:
    """Return a photo file's modification time in the workstation time zone."""
    timestamp = Path(path).stat().st_mtime
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="milliseconds")


def load_calibration_json(path_value: str | Path) -> dict[str, Any]:
    """Load a calibration object and reject blank/directory paths clearly."""
    text = str(path_value).strip()
    if not text:
        raise ValueError("请先选择真实双目标定 JSON，再连接相机。")
    path = Path(text)
    if not path.is_file():
        raise ValueError(f"双目标定 JSON 不存在或不是文件：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"双目标定 JSON 格式无效：{error}") from error
    if not isinstance(payload, dict):
        raise ValueError("双目标定文件应为 JSON 对象。")
    calibration = payload.get("stereo_calibration", payload)
    if not isinstance(calibration, dict):
        raise ValueError("双目标定 JSON 缺少 stereo_calibration 对象。")
    return calibration


def field_calibration_problem(calibration: Any) -> str | None:
    """Explain why a calibration must not be used for a live field capture."""
    if not isinstance(calibration, dict):
        return "双目标定内容不是有效对象。"
    calibration_id = str(calibration.get("calibration_id", "")).strip()
    upper_id = calibration_id.upper()
    if not calibration_id:
        return "双目标定缺少 calibration_id，无法确认其来源。"
    if any(marker in upper_id for marker in _NON_FIELD_CALIBRATION_MARKERS):
        return (
            f"当前标定 ID“{calibration_id}”属于合成演示或格式示例，不能用于现场 USB 双目相机。"
            "请选择这台相机的真实标定 JSON；每目应为实际分辨率，并包含实测内参、畸变、毫米基线和极线校正状态。"
        )
    from .calibration_quality import provisional_calibration_allowed

    if calibration.get("validated") is not True and not provisional_calibration_allowed(calibration):
        return "当前双目标定尚未标记为已验证（validated=true），不能用于现场测量。"
    if calibration.get("rectified") is not True:
        return "当前输入未声明为已极线校正（rectified=true），不能直接进行现场双目匹配。"
    return None


def capture_provenance(
    payload: dict[str, dict[str, Any]] | None,
    role: str,
) -> dict[str, Any]:
    """Validate and copy the camera-side provenance record for one view."""
    if payload is None:
        return {}
    if not isinstance(payload, dict) or set(payload) - {"left", "right"}:
        raise ValueError("camera_capture_provenance must contain only left/right records")
    record = payload.get(role, {})
    if not isinstance(record, dict) or set(record) - _CAPTURE_PROVENANCE_KEYS:
        raise ValueError(f"camera_capture_provenance.{role} contains unsupported fields")
    if "capture_device_index" in record and (
        type(record["capture_device_index"]) is not int or record["capture_device_index"] < 0
    ):
        raise ValueError(f"camera_capture_provenance.{role}.capture_device_index is invalid")
    for key, value in record.items():
        if key in {"capture_exposure_target_ms", "capture_exposure_ms"}:
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"camera_capture_provenance.{role}.{key} is invalid")
            continue
        if key == "capture_exposure_status" and (
            not isinstance(value, str) or value not in {"DRIVER_REPORTED", "UNCONFIRMED", "AUTO"}
        ):
            raise ValueError(f"camera_capture_provenance.{role}.{key} is invalid")
        if key != "capture_device_index" and (
            not isinstance(value, str) or not value.strip() or len(value) > 128
        ):
            raise ValueError(f"camera_capture_provenance.{role}.{key} is invalid")
    return copy.deepcopy(record)


def catalog_from_model(
    path: str | Path,
    *,
    stl_unit: str | None = None,
) -> tuple[list[dict], list[str]]:
    """Suggest straight cylindrical CAD objects; these are DESIGN values only."""
    scene = load_cad_scene(path, stl_unit=stl_unit)
    pipes, skipped = [], []
    for item in scene.objects:
        vertices = np.unique(item.vertices_world_mm, axis=0)
        origin = np.mean(vertices, axis=0)
        _, _, vectors = np.linalg.svd(vertices - origin, full_matrices=False)
        axis = vectors[0]
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        axial = (vertices - origin) @ axis
        radial = vertices - origin - np.outer(axial, axis)
        radii = np.linalg.norm(radial, axis=1)
        radius = float(np.quantile(radii, 0.75))
        length = float(np.ptp(axial))
        outer = radii[radii > radius * 0.5]
        if (
            radius <= 0
            or length < 4 * radius
            or not len(outer)
            or np.max(np.abs(outer - radius)) > 0.05 * radius
        ):
            skipped.append(item.object_id)
            continue
        index = len(pipes) + 1
        pipes.append(
            {
                "instance_id": index,
                "pipe_id": f"P{index:03d}",
                "cad_object_id": item.object_id,
                "cad_uuid": item.guid,
                "layer_id": item.layer_path or "现场",
                "color_class": "cad_color",
                "color_srgb": item.color_srgb,
                "nominal_diameter_mm": round(2 * radius, 6),
                "centerline_world_mm": [
                    (origin + axis * axial.min()).tolist(),
                    (origin + axis * axial.max()).tolist(),
                ],
            }
        )
    return pipes, skipped


__all__ = [
    "capture_provenance",
    "catalog_from_model",
    "field_calibration_problem",
    "load_calibration_json",
    "normalize_capture_time",
    "photo_file_time",
]
