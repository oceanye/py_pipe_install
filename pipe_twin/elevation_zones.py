"""Validation and image geometry helpers for optional elevation zones.

Zones are group regions on the rectified left image.  They are deliberately
separate from per-pipe ROIs: the recognizer still finds pipes and the global
diameter/colour policy still owns identity.  The right image is expanded by a
bounded disparity margin and the cropped calibration principal points are
shifted with the pixels.
"""

from __future__ import annotations

import copy
import re
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np

from .stereo_analyzer import CameraCalibration, StereoCalibration


ZONE_SCHEMA_VERSION = 1
MAX_ZONES = 8
MIN_ZONE_WIDTH_PX = 32
MIN_ZONE_HEIGHT_PX = 32
DEFAULT_RIGHT_ROI_PADDING_PX = 128
_ZONE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$")


class ElevationZoneError(ValueError):
    """Raised when a persisted or interactive zone is malformed."""


def _integer(value: Any, field: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if type(value) is not int:
        raise ElevationZoneError(f"{field}必须是整数")
    if minimum is not None and value < minimum:
        raise ElevationZoneError(f"{field}不能小于{minimum}")
    if maximum is not None and value > maximum:
        raise ElevationZoneError(f"{field}不能大于{maximum}")
    return value


def _rect(value: Any, field: str, image_size: tuple[int, int] | None) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ElevationZoneError(f"{field}必须是[x,y,width,height]")
    x, y, width, height = (_integer(item, f"{field}[{index}]") for index, item in enumerate(value))
    if x < 0 or y < 0:
        raise ElevationZoneError(f"{field}的x/y不能为负数")
    if width < MIN_ZONE_WIDTH_PX or height < MIN_ZONE_HEIGHT_PX:
        raise ElevationZoneError(
            f"{field}至少为{MIN_ZONE_WIDTH_PX}×{MIN_ZONE_HEIGHT_PX}像素"
        )
    if image_size is not None:
        image_width, image_height = image_size
        if x + width > image_width or y + height > image_height:
            raise ElevationZoneError(f"{field}超出左目矫正图范围{image_width}×{image_height}")
    return [x, y, width, height]


def _zone(raw: Any, index: int, image_size: tuple[int, int] | None) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ElevationZoneError(f"zones[{index}]必须是对象")
    allowed = {"zone_id", "label", "enabled", "coordinate_space", "roi_rect_px"}
    unknown = set(raw) - allowed
    if unknown:
        raise ElevationZoneError(f"zones[{index}]包含未知字段：{sorted(unknown)}")
    zone_id = raw.get("zone_id")
    if not isinstance(zone_id, str) or _ZONE_ID.fullmatch(zone_id) is None:
        raise ElevationZoneError(f"zones[{index}].zone_id必须是ASCII标识符")
    label = raw.get("label", zone_id)
    if not isinstance(label, str) or not label.strip() or len(label.strip()) > 64:
        raise ElevationZoneError(f"zones[{index}].label必须是1到64字符")
    enabled = raw.get("enabled", True)
    if type(enabled) is not bool:
        raise ElevationZoneError(f"zones[{index}].enabled必须是布尔值")
    coordinate_space = raw.get("coordinate_space", "rectified_left")
    if coordinate_space != "rectified_left":
        raise ElevationZoneError("分区坐标系只支持rectified_left")
    return {
        "zone_id": zone_id,
        "label": label.strip(),
        "enabled": enabled,
        "coordinate_space": "rectified_left",
        "roi_rect_px": _rect(raw.get("roi_rect_px"), f"zones[{index}].roi_rect_px", image_size),
    }


def normalize_zone_settings(payload: Mapping[str, Any] | None = None,
                            *, image_size: tuple[int, int] | None = None) -> dict[str, Any]:
    """Return a strict, portable full-frame or group-zone scope policy."""

    if payload is not None and not isinstance(payload, Mapping):
        raise ElevationZoneError("scope配置必须是对象")
    value = dict(payload or {})
    allowed = {"scope", "zone_schema_version", "right_roi_padding_px", "zones"}
    unknown = set(value) - allowed
    if unknown:
        raise ElevationZoneError(f"scope配置包含未知字段：{sorted(unknown)}")
    scope = value.get("scope", "full_frame")
    if scope not in {"full_frame", "zones"}:
        raise ElevationZoneError("scope只能是full_frame或zones")
    version = value.get("zone_schema_version", ZONE_SCHEMA_VERSION)
    if type(version) is not int or version != ZONE_SCHEMA_VERSION:
        raise ElevationZoneError(f"zone_schema_version必须是{ZONE_SCHEMA_VERSION}")
    padding = _integer(value.get("right_roi_padding_px", DEFAULT_RIGHT_ROI_PADDING_PX),
                       "right_roi_padding_px", minimum=0, maximum=1024)
    raw_zones = value.get("zones", [])
    if not isinstance(raw_zones, list):
        raise ElevationZoneError("zones必须是数组")
    if len(raw_zones) > MAX_ZONES:
        raise ElevationZoneError(f"分区最多支持{MAX_ZONES}个")
    zones = [_zone(item, index, image_size) for index, item in enumerate(raw_zones)]
    ids = [item["zone_id"] for item in zones]
    if len(set(ids)) != len(ids):
        raise ElevationZoneError("zone_id必须唯一")
    if scope == "zones" and not any(item["enabled"] for item in zones):
        raise ElevationZoneError("分区模式至少需要一个启用分区")
    if scope == "full_frame" and zones:
        raise ElevationZoneError("full_frame模式不能携带分区")
    return {
        "scope": scope,
        "zone_schema_version": ZONE_SCHEMA_VERSION,
        "right_roi_padding_px": padding,
        "zones": zones,
    }


def enabled_zones(settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    normalized = normalize_zone_settings(settings)
    return [copy.deepcopy(zone) for zone in normalized["zones"] if zone["enabled"]]


def _cropped_camera(camera: CameraCalibration, x: int, y: int, width: int, height: int) -> CameraCalibration:
    intrinsic = np.asarray(camera.intrinsic, dtype=np.float64).copy()
    intrinsic[0, 2] -= float(x)
    intrinsic[1, 2] -= float(y)
    projection = None
    if camera.projection_matrix is not None:
        projection = np.asarray(camera.projection_matrix, dtype=np.float64).copy()
        projection[0, 2] -= float(x)
        projection[1, 2] -= float(y)
    return replace(camera, width=width, height=height, intrinsic=intrinsic, projection_matrix=projection)


def crop_stereo_group_for_zone(group: Mapping[str, Any], calibration: StereoCalibration,
                               zone: Mapping[str, Any], *, right_padding_px: int) -> tuple[dict[str, Any], dict[str, int]]:
    """Crop one capture and shift the calibration for a rectified-left zone."""

    left = np.asarray(group["left"])
    right = np.asarray(group["right"])
    depth = group["depth"]
    if left.ndim < 2 or right.ndim < 2 or left.shape[:2] != (calibration.left.height, calibration.left.width):
        raise ElevationZoneError("分区照片尺寸与标定不一致")
    if right.shape[:2] != (calibration.right.height, calibration.right.width):
        raise ElevationZoneError("右目照片尺寸与标定不一致")
    x, y, width, height = [int(item) for item in zone["roi_rect_px"]]
    if x + width > calibration.left.width or y + height > calibration.left.height:
        raise ElevationZoneError("分区超出左目矫正图")
    pad = _integer(right_padding_px, "right_padding_px", minimum=0, maximum=1024)
    right_x = max(0, x - pad)
    right_end = min(calibration.right.width, x + width + pad)
    right_width = right_end - right_x
    if right_width < MIN_ZONE_WIDTH_PX:
        raise ElevationZoneError("右目扩展后的分区过窄")
    if y + height > calibration.right.height:
        raise ElevationZoneError("分区超出右目矫正图")
    left_offset = {"x": x, "y": y}
    right_offset = {"x": right_x, "y": y}
    cropped_left = left[y:y + height, x:x + width].copy()
    cropped_right = right[y:y + height, right_x:right_end].copy()
    cropped_values = {
        "left_depth_mm": np.asarray(depth.left_depth_mm)[y:y + height, x:x + width].copy(),
        "right_depth_mm": np.asarray(depth.right_depth_mm)[y:y + height, right_x:right_end].copy(),
        "left_valid": np.asarray(depth.left_valid)[y:y + height, x:x + width].copy(),
        "right_valid": np.asarray(depth.right_valid)[y:y + height, right_x:right_end].copy(),
        "audit": {**dict(getattr(depth, "audit", {})), "zone_id": zone["zone_id"]},
    }
    if hasattr(depth, "__dataclass_fields__"):
        cropped_depth = replace(depth, **cropped_values)
    else:
        cropped_depth = SimpleNamespace(**cropped_values)
    cropped_group = dict(group)
    cropped_group.update({"left": cropped_left, "right": cropped_right, "depth": cropped_depth,
                          "zone_id": zone["zone_id"], "zone_label": zone.get("label", zone["zone_id"])})
    return cropped_group, {"left_x": x, "left_y": y, "right_x": right_x, "right_y": y,
                           "width": width, "height": height, "right_width": right_width}


def crop_calibration_for_zone(calibration: StereoCalibration, zone: Mapping[str, Any],
                              *, right_padding_px: int) -> StereoCalibration:
    """Return the rectified calibration for the same crop as a zone group."""

    x, y, width, height = [int(item) for item in zone["roi_rect_px"]]
    pad = _integer(right_padding_px, "right_padding_px", minimum=0, maximum=1024)
    right_x = max(0, x - pad)
    right_end = min(calibration.right.width, x + width + pad)
    right_width = right_end - right_x
    if x < 0 or y < 0 or x + width > calibration.left.width or y + height > calibration.left.height:
        raise ElevationZoneError("分区超出左目矫正图")
    if y + height > calibration.right.height or right_width < MIN_ZONE_WIDTH_PX:
        raise ElevationZoneError("分区超出右目矫正图或右目扩展区域过窄")
    return replace(
        calibration,
        left=_cropped_camera(calibration.left, x, y, width, height),
        right=_cropped_camera(calibration.right, right_x, y, right_width, height),
    )


def offset_region_box(value: Any, offset: Mapping[str, int]) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x, y, width, height = [int(item) for item in value]
    except (TypeError, ValueError):
        return None
    return [x + int(offset["x"]), y + int(offset["y"]), width, height]


def offset_observation_pixels(observation: Mapping[str, Any], offsets: Mapping[str, int]) -> dict[str, Any]:
    """Copy an observation and map its two cropped image boxes to full pixels."""

    result = copy.deepcopy(dict(observation))
    result["left_region_px"] = offset_region_box(result.get("left_region_px"),
                                                  {"x": offsets["left_x"], "y": offsets["left_y"]})
    result["right_region_px"] = offset_region_box(result.get("right_region_px"),
                                                   {"x": offsets["right_x"], "y": offsets["right_y"]})
    return result


__all__ = [
    "DEFAULT_RIGHT_ROI_PADDING_PX",
    "ElevationZoneError",
    "MAX_ZONES",
    "MIN_ZONE_HEIGHT_PX",
    "MIN_ZONE_WIDTH_PX",
    "ZONE_SCHEMA_VERSION",
    "crop_stereo_group_for_zone",
    "crop_calibration_for_zone",
    "enabled_zones",
    "normalize_zone_settings",
    "offset_observation_pixels",
]
