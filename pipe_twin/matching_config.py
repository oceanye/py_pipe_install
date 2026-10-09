"""Validated diameter/colour matching settings for elevation recognition.

Diameter is the physical identity gate and is always enabled.  Colour is an
optional candidate hint because DXF colours can describe the painted pipes,
while an STL normally has no reliable colour at all.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


DEFAULT_MATCHING_SETTINGS: dict[str, Any] = {
    "diameter_filter_enabled": True,
    "diameter_tolerance_mm": 3.0,
    "diameter_tolerance_ratio": 0.10,
    "color_filter_enabled": False,
    "color_filter_mode": "hint",
    "color_delta_lab": 45.0,
}


def normalize_matching_settings(payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Validate the portable matching policy used by all elevation scopes."""

    if payload is not None and not isinstance(payload, Mapping):
        raise ValueError("matching配置必须是对象")
    value = dict(payload or {})
    unknown = set(value) - set(DEFAULT_MATCHING_SETTINGS)
    if unknown:
        raise ValueError(f"matching配置包含未知字段：{sorted(unknown)}")
    result = {**DEFAULT_MATCHING_SETTINGS, **value}
    if result["diameter_filter_enabled"] is not True:
        raise ValueError("直径筛选是自动识别的必需主筛选，不能关闭")
    if type(result["color_filter_enabled"]) is not bool:
        raise ValueError("matching.color_filter_enabled必须是布尔值")
    if result["color_filter_mode"] != "hint":
        raise ValueError("当前颜色筛选只支持hint辅助模式")
    for key in ("diameter_tolerance_mm", "diameter_tolerance_ratio", "color_delta_lab"):
        item = result[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) or float(item) <= 0:
            raise ValueError(f"matching.{key}必须是正数")
        result[key] = float(item)
    if result["diameter_tolerance_ratio"] > 1:
        raise ValueError("matching.diameter_tolerance_ratio不能超过1")
    if result["color_delta_lab"] > 150:
        raise ValueError("matching.color_delta_lab不能超过150")
    result["diameter_filter_enabled"] = True
    result["color_filter_enabled"] = bool(result["color_filter_enabled"])
    result["color_filter_mode"] = "hint"
    return result


__all__ = ["DEFAULT_MATCHING_SETTINGS", "normalize_matching_settings"]
