"""Adapters and diagnostics for real OpenCV/GLM stereo calibration files.

The recognition pipeline consumes a manifest-bound, CAD-referenced calibration.
This module converts the common OpenCV ``K/D/R/T`` result into that contract,
while keeping the raw calibration and the exact ``R1/R2/P1/P2`` rectification
matrices auditable.  It deliberately refuses to guess units or CAD pose.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


class CalibrationAdapterError(ValueError):
    """Raised when a source calibration is incomplete or inconsistent."""


def _matrix(value: Any, shape: tuple[int, ...], field: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise CalibrationAdapterError(f"{field} must be numeric") from error
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise CalibrationAdapterError(f"{field} must have finite shape {shape}")
    return result


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise CalibrationAdapterError(f"{field} must be a finite number")
    result = float(value)
    if positive and result <= 0:
        raise CalibrationAdapterError(f"{field} must be positive")
    return result


def _unit_scale_to_mm(unit: Any) -> float:
    if not isinstance(unit, str):
        raise CalibrationAdapterError(
            "translation_unit is required; refusing to guess whether T is m or mm"
        )
    units = {"mm": 1.0, "millimeter": 1.0, "millimeters": 1.0,
             "m": 1000.0, "meter": 1000.0, "meters": 1000.0,
             "cm": 10.0, "centimeter": 10.0, "centimeters": 10.0}
    try:
        return units[unit.strip().lower()]
    except KeyError as error:
        raise CalibrationAdapterError(f"unsupported translation_unit: {unit!r}") from error


def _value(payload: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in payload:
            return payload[name]
    raise CalibrationAdapterError(f"missing one of: {', '.join(names)}")


def _camera_pose(payload: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    pose = payload.get("left_camera_pose", payload.get("left_pose"))
    if not isinstance(pose, Mapping):
        raise CalibrationAdapterError(
            "left_camera_pose is required to bind calibration to CAD world coordinates"
        )
    rotation = _matrix(
        _value(pose, "rotation_world_to_camera", "R_world_to_camera"),
        (3, 3),
        "left_camera_pose.rotation_world_to_camera",
    )
    center = _matrix(
        _value(pose, "center_world_mm", "camera_center_world_mm"),
        (3,),
        "left_camera_pose.center_world_mm",
    )
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6
    ):
        raise CalibrationAdapterError(
            "left_camera_pose.rotation_world_to_camera must be a proper rotation"
        )
    return rotation, center


def adapt_opencv_stereo_calibration(
    source: Mapping[str, Any],
    *,
    calibration_id: str,
    left_camera_id: str = "camera-left",
    right_camera_id: str = "camera-right",
    max_sync_delta_ms: float = 5.0,
    validated: bool = False,
    registration_validated: bool = False,
) -> dict[str, Any]:
    """Convert OpenCV stereoCalibrate output plus a CAD left pose.

    ``R`` and ``T`` use OpenCV's cam1-to-cam2 convention.  ``T`` must carry an
    explicit unit.  The returned cameras retain raw K/D and store R1/R2/P1/P2
    so CAD projections use the same rectified image geometry as SGBM.
    """
    if not isinstance(source, Mapping):
        raise CalibrationAdapterError("calibration source must be an object")
    K1 = _matrix(_value(source, "K1", "camera_matrix_left", "left_K"), (3, 3), "K1")
    K2 = _matrix(_value(source, "K2", "camera_matrix_right", "right_K"), (3, 3), "K2")
    D1 = np.asarray(_value(source, "D1", "distortion_left", "left_D"), dtype=np.float64).reshape(-1)
    D2 = np.asarray(_value(source, "D2", "distortion_right", "right_D"), dtype=np.float64).reshape(-1)
    if len(D1) not in {4, 5, 8, 12, 14} or len(D2) not in {4, 5, 8, 12, 14}:
        raise CalibrationAdapterError("D1 and D2 must contain 4, 5, 8, 12, or 14 coefficients")
    size_value = _value(source, "image_size", "image_size_px", "size")
    if not isinstance(size_value, (list, tuple)) or len(size_value) != 2:
        raise CalibrationAdapterError("image_size must be [width, height]")
    width, height = int(size_value[0]), int(size_value[1])
    if width <= 0 or height <= 0:
        raise CalibrationAdapterError("image_size must be positive")
    R = _matrix(_value(source, "R", "rotation_left_to_right"), (3, 3), "R")
    T = _matrix(_value(source, "T", "translation_left_to_right"), (3,), "T")
    scale = _unit_scale_to_mm(source.get("translation_unit", source.get("T_unit")))
    T_mm = T * scale
    left_rotation, left_center = _camera_pose(source)
    right_rotation = R @ left_rotation
    right_center = left_center - right_rotation.T @ T_mm
    baseline_mm = float(np.linalg.norm(right_center - left_center))
    if baseline_mm <= 0:
        raise CalibrationAdapterError("R/T imply a zero stereo baseline")
    flags = int(source.get("stereo_rectify_flags", cv2.CALIB_ZERO_DISPARITY))
    alpha = float(source.get("stereo_rectify_alpha", 0.0))
    R1, R2, P1, P2, _Q, _roi1, _roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, (width, height), R, T_mm,
        flags=flags, alpha=alpha,
    )
    encoded_baseline = -float(P2[0, 3]) / float(P2[0, 0])
    if encoded_baseline <= 0 or not math.isclose(encoded_baseline, baseline_mm, rel_tol=1e-5, abs_tol=1e-3):
        raise CalibrationAdapterError("stereoRectify baseline is inconsistent with camera centres")

    def camera(camera_id: str, K: np.ndarray, D: np.ndarray, Rrect: np.ndarray, P: np.ndarray, rotation: np.ndarray, center: np.ndarray) -> dict[str, Any]:
        return {
            "camera_id": camera_id, "width": width, "height": height,
            "K": K.tolist(), "D": D.tolist(),
            "rotation_world_to_camera": rotation.tolist(),
            "center_world_mm": center.tolist(),
            "rectification_matrix": Rrect.tolist(),
            "projection_matrix": P.tolist(),
        }

    return {
        "calibration_id": str(calibration_id),
        "validated": bool(validated),
        "registration_validated": bool(registration_validated),
        "rectified": True,
        "baseline_mm": baseline_mm,
        "max_sync_delta_ms": _number(max_sync_delta_ms, "max_sync_delta_ms"),
        "left_camera": camera(left_camera_id, K1, D1, R1, P1, left_rotation, left_center),
        "right_camera": camera(right_camera_id, K2, D2, R2, P2, right_rotation, right_center),
        "source_audit": {
            "adapter": "opencv_stereoCalibrate_to_manifest_v1",
            "translation_unit": str(source.get("translation_unit", source.get("T_unit"))),
            "stereo_rectify_flags": flags,
            "stereo_rectify_alpha": alpha,
            "opencv_translation_left_to_right_mm": T_mm.tolist(),
            "rectification_outputs": ["R1", "R2", "P1", "P2"],
        },
    }


def validate_calibration(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a structured, user-facing validation result for a manifest calibration."""
    from .stereo_analyzer import StereoAnalysisError, _calibration_from_manifest

    candidate = payload.get("stereo_calibration", payload) if isinstance(payload, Mapping) else payload
    try:
        parsed = _calibration_from_manifest(candidate)
    except (StereoAnalysisError, TypeError, ValueError) as error:
        return {"valid": False, "error": str(error), "repair_hint": _repair_hint(str(error))}
    return {
        "valid": True,
        "calibration_id": parsed.calibration_id,
        "image_size": [parsed.left.width, parsed.left.height],
        "baseline_mm": parsed.baseline_mm,
        "validated": parsed.validated,
        "registration_validated": parsed.registration_validated,
        "rectified": parsed.rectified,
        "ready_for_field_analysis": parsed.validated and parsed.registration_validated,
    }


def _repair_hint(error: str) -> str:
    if "rectified must be true" in error:
        return "先调用适配器生成 R1/R2/P1/P2 并 remap 左右图，再把矫正图哈希写入 manifest。"
    if "rotation_world_to_camera" in error or "center_world_mm" in error:
        return "补充 CAD 世界坐标到 OpenCV 相机坐标的左目位姿；不要直接填 SDK 的 R|t。"
    if "baseline" in error:
        return "统一 T 与 CAD 坐标单位（建议 mm），并检查 OpenCV T 的 cam1→cam2 方向。"
    if "projection matrix" in error or "K values" in error:
        return "保留 stereoRectify 的 P1/P2；不要把原始左右 K 强行复制成相同数值。"
    return "按报错字段修正标定 JSON 后重新运行校验。"


def load_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise CalibrationAdapterError("calibration JSON must contain an object")
    return copy.deepcopy(value)


__all__ = [
    "CalibrationAdapterError",
    "adapt_opencv_stereo_calibration",
    "load_json",
    "validate_calibration",
]
