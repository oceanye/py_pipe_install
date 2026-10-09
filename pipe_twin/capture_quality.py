"""Advisory pipe-patch diagnostics; never grants measurement or installation status."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any

import cv2
import numpy as np

from .local_surface import color_candidate_mask, DEFAULT_COLOR_DELTA_LAB

REVISION = "pipe-patch-quality-v1"
POLICY = {
    "color_rule": "OPENCV_UINT8_LAB_EUCLIDEAN_DISTANCE",
    "color_delta_lab": DEFAULT_COLOR_DELTA_LAB,
    "color_green_min_percent": 60.0, "color_amber_min_percent": 20.0,
    "highlight_channel_min": 250, "highlight_required_channels": 2,
    "highlight_green_max_percent": 1.0, "highlight_amber_max_percent": 5.0,
    "minimum_patch_pixels": 64,
    "dark_gray_max": 16, "low_light_median_below": 32,
    "dark_fraction_above": 0.5,
    "threshold_status": "PROVISIONAL_ADVISORY_NOT_FIELD_VALIDATED",
}
POLICY_SHA256 = hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()
LEVEL_TEXT = {"GREEN": "绿", "AMBER": "黄", "RED": "红", "PENDING": "待框选"}


def _color_level(value: float | None) -> str:
    if value is None:
        return "PENDING"
    return ("GREEN" if value >= POLICY["color_green_min_percent"] else
            "AMBER" if value >= POLICY["color_amber_min_percent"] else "RED")


def _highlight_level(value: float | None) -> str:
    if value is None:
        return "PENDING"
    return ("GREEN" if value <= POLICY["highlight_green_max_percent"] else
            "AMBER" if value <= POLICY["highlight_amber_max_percent"] else "RED")


def validate_color(value: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", value):
        raise ValueError("参考颜色必须是 #RRGGBB")
    return value.upper()


def patch_mask(image: np.ndarray, region: list[int]) -> np.ndarray:
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3):
        raise ValueError("需使用 8 位彩色照片")
    if not isinstance(region, (tuple, list)) or len(region) != 4 or any(type(v) is not int for v in region):
        raise ValueError("检查框需为整数 x、y、宽、高")
    x, y, w, h = region
    if x < 0 or y < 0 or w < 8 or h < 8 or x + w > image.shape[1] or y + h > image.shape[0]:
        raise ValueError("检查框超出照片范围，或小于 8×8 像素")
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[y:y+h, x:x+w] = True
    return mask


def measure_patch(image: np.ndarray, color: str, mask: np.ndarray) -> dict[str, Any]:
    """Measure an independent operator mask, not the color detector's own output."""
    color = validate_color(color)
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or not isinstance(mask, np.ndarray)
            or mask.dtype != np.bool_ or mask.shape != image.shape[:2]):
        raise ValueError("照片和管面掩码不匹配")
    if int(mask.sum()) < POLICY["minimum_patch_pixels"]:
        raise ValueError("管面取样至少需要 64 个像素")
    pixels = image[mask].reshape(-1, 1, 3)
    lab = cv2.cvtColor(pixels, cv2.COLOR_BGR2LAB)
    coverage = float(np.mean(color_candidate_mask(lab, color)) * 100)
    bright = (pixels[:, 0] >= POLICY["highlight_channel_min"]).sum(axis=1) >= POLICY["highlight_required_channels"]
    highlight = float(np.mean(bright) * 100)
    gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY).ravel()
    median = float(np.median(gray))
    dark_fraction = float(np.mean(gray <= POLICY["dark_gray_max"]))
    usable_color = pixels[:, 0][(~bright) & (gray > POLICY["dark_gray_max"])]
    sampled_color = None
    if len(usable_color) >= POLICY["minimum_patch_pixels"]:
        bgr = np.median(usable_color, axis=0).round().astype(int)
        sampled_color = "#" + "".join(f"{v:02X}" for v in bgr[::-1])
    return {
        "pixel_count": int(mask.sum()), "target_color_srgb": color,
        "color_coverage_percent": coverage,
        "color_level": _color_level(coverage),
        "highlight_risk_percent": highlight,
        "highlight_level": _highlight_level(highlight),
        "luminance_median": median, "dark_fraction": dark_fraction,
        "underexposed": median < POLICY["low_light_median_below"] or dark_fraction > POLICY["dark_fraction_above"],
        "sampled_color_srgb": sampled_color,
    }


def summarize_pair(eyes: dict[str, dict | None]) -> dict[str, Any]:
    present = [eyes.get(role) for role in ("left", "right") if eyes.get(role) is not None]
    complete = len(present) == 2
    color = min((v["color_coverage_percent"] for v in present), default=None) if complete else None
    highlight = max((v["highlight_risk_percent"] for v in present), default=None) if complete else None
    dark = any(v["underexposed"] for v in present)
    advice = []
    if not complete:
        advice.append("请在左右照片各框选同一根管的一小块可见管面，避开支架和背景。")
    if dark:
        advice.append("取样区域偏暗：先恢复普通管面亮度；高光占比低不代表照片可用。")
    if complete:
        if _highlight_level(highlight) == "RED":
            advice.append("先调灯光角度或柔光，必要时缩短曝光，降低亮部裁切；白管需结合画面判断。")
        elif _highlight_level(highlight) == "AMBER":
            advice.append("有高光风险：观察亮条，调整灯光角度并复拍比较。")
        if _color_level(color) != "GREEN":
            advice.append("颜色覆盖不足：照明稳定后，从非高光管面取色并核对参考色。")
        if _color_level(color) == "GREEN" and _highlight_level(highlight) == "GREEN" and not dark:
            advice.append("两项取样指标较好；仍需运行评估检查深度和圆柱拟合。")
    return {
        "status": "UNDEREXPOSED" if dark else "COMPLETE" if complete else "INCOMPLETE",
        "color_coverage_percent": color, "highlight_risk_percent": highlight,
        "color_level": _color_level(color),
        "highlight_level": _highlight_level(highlight),
        "advice": advice,
    }


def diagnostic_record(photo_hashes: dict[str, str], samples: list[dict]) -> dict:
    return {
        "revision": REVISION, "policy": copy.deepcopy(POLICY), "policy_sha256": POLICY_SHA256,
        "scope": "MANUAL_PIPE_PATCH_ADVICE_NOT_IDENTITY_OR_MEASUREMENT",
        "photo_sha256": dict(photo_hashes), "samples": copy.deepcopy(samples),
    }
