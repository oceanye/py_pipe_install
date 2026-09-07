"""Auditable stereo-rig placement and tilt adjustments in CAD coordinates."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


POSE_MODE_LABELS = {
    "keep": "沿用标定文件位姿",
    "adjust_current": "当前位姿 + 角度/位置修正",
    "positive_z": "相机位于 +Z，朝向 -Z",
    "negative_z": "相机位于 -Z，朝向 +Z",
    "positive_x": "侧立面：从右往左（CAD +X → -X）",
    "negative_x": "侧立面：从左往右（CAD -X → +X）",
    "positive_y": "侧立面：从上往下（CAD +Y → -Y）",
    "negative_y": "侧立面：从下往上（CAD -Y → +Y）",
}


SIDE_ELEVATION_VIEWS = {
    "top_down": {
        "label": "从上往下",
        "mode": "positive_y",
        "outward_world": [0.0, 1.0, 0.0],
    },
    "bottom_up": {
        "label": "从下往上",
        "mode": "negative_y",
        "outward_world": [0.0, -1.0, 0.0],
    },
    "left_right": {
        "label": "从左往右",
        "mode": "negative_x",
        "outward_world": [-1.0, 0.0, 0.0],
    },
    "right_left": {
        "label": "从右往左",
        "mode": "positive_x",
        "outward_world": [1.0, 0.0, 0.0],
    },
}


_PRESET_FORWARD_UP = {
    "positive_z": ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
    "negative_z": ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
    # Side-elevation presets keep CAD +Z (the pipe longitudinal direction in
    # the supplied STL) horizontal in the image. The opposite view naturally
    # reverses it while preserving a proper right-handed camera frame.
    "positive_x": ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    "negative_x": ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    "positive_y": ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0]),
    "negative_y": ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
}


def _finite(value: Any, field: str, *, bound: float | None = None) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if bound is not None and abs(result) > bound:
        raise ValueError(f"{field} must be between {-bound:g} and {bound:g}")
    return result


def _rotation(axis: str, degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]], dtype=float)
    if axis == "y":
        return np.array([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]], dtype=float)
    return np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]], dtype=float)


def _look_rotation(forward_world: list[float], up_world: list[float]) -> np.ndarray:
    forward = np.asarray(forward_world, dtype=float)
    forward /= np.linalg.norm(forward)
    up = np.asarray(up_world, dtype=float)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.vstack((right, down, forward))
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9) or np.linalg.det(rotation) < 0:
        raise ValueError("camera direction does not define a proper rotation")
    return rotation


def calibration_pose(calibration: Mapping[str, Any]) -> dict[str, Any]:
    """Return the current rig midpoint and world viewing direction."""
    from .stereo_analyzer import _calibration_from_manifest

    parsed = _calibration_from_manifest(calibration)
    midpoint = (parsed.left.center_world_mm + parsed.right.center_world_mm) / 2
    forward = parsed.left.rotation_world_to_camera.T @ np.array([0.0, 0.0, 1.0])
    return {
        "center_world_mm": midpoint.tolist(),
        "forward_world": forward.tolist(),
        "rotation_world_to_camera": parsed.left.rotation_world_to_camera.tolist(),
        "baseline_mm": parsed.baseline_mm,
        "registration_validated": parsed.registration_validated,
    }


def model_center_from_pipes(pipes: Any) -> np.ndarray:
    """Return the axis-aligned centre of all catalogued pipe centreline ends."""
    if not isinstance(pipes, list) or not pipes:
        raise ValueError("请先从 CAD 模型读取管件目录")
    try:
        points = np.asarray(
            [point for pipe in pipes for point in pipe["centerline_world_mm"]],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("管件目录缺少有效的 CAD 中心线") from error
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError("管件目录缺少有效的 CAD 中心线")
    return (points.min(axis=0) + points.max(axis=0)) / 2.0


def suggested_side_view(
    pipes: Any,
    view_key: str,
    distance_mm: float,
) -> dict[str, Any]:
    """Aim one named side-elevation view at the CAD pipe group centre."""
    if view_key not in SIDE_ELEVATION_VIEWS:
        raise ValueError("请选择有效的 CAD 侧立面观察方向")
    distance = _finite(distance_mm, "camera_distance_mm", bound=1_000_000)
    if distance <= 0:
        raise ValueError("相机到模型中心距离必须为正数")
    target = model_center_from_pipes(pipes)
    definition = SIDE_ELEVATION_VIEWS[view_key]
    center = target + distance * np.asarray(definition["outward_world"])
    return {
        "view_key": view_key,
        "view_label": definition["label"],
        "mode": definition["mode"],
        "target_world_mm": target.tolist(),
        "center_world_mm": center.tolist(),
        "distance_mm": distance,
    }


def _dominant_axis_degrees(
    angles_degrees: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    angles = np.mod(np.asarray(angles_degrees, dtype=np.float64), 180.0)
    weights = np.asarray(weights, dtype=np.float64)
    if angles.ndim != 1 or weights.shape != angles.shape or len(angles) < 2:
        raise ValueError("没有足够的直线用于自动倾斜校正")
    bins = np.zeros(180, dtype=np.float64)
    for angle, weight in zip(angles, weights):
        bins[int(round(angle)) % 180] += max(float(weight), 0.0)
    smoothed = sum(np.roll(bins, offset) for offset in range(-7, 8))
    peak = float(np.argmax(smoothed))
    delta = (angles - peak + 90.0) % 180.0 - 90.0
    selected = np.abs(delta) <= 10.0
    if np.count_nonzero(selected) < 2 or float(np.sum(weights[selected])) <= 0:
        raise ValueError("画面直线方向不集中，无法可靠自动校正倾斜")
    radians = np.deg2rad(angles[selected] * 2.0)
    vector = np.sum(weights[selected] * np.exp(1j * radians))
    angle = math.degrees(math.atan2(vector.imag, vector.real)) / 2.0
    angle %= 180.0
    concentration = float(np.sum(weights[selected]) / max(np.sum(weights), 1e-9))
    return angle, concentration


def estimate_pipe_roll_correction(
    image_path: str | Path,
    pipes: Any,
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    """Match the dominant observed pipe line to the CAD projection using roll."""
    from .stereo_analyzer import _calibration_from_manifest

    path = Path(image_path)
    if not path.is_file():
        raise ValueError("请先同步抓拍包含管件的左目图")
    image = cv2.imdecode(
        np.frombuffer(path.read_bytes(), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if image is None:
        raise ValueError("左目图无法解码")
    parsed = _calibration_from_manifest(calibration)
    if (image.shape[1], image.shape[0]) != (parsed.left.width, parsed.left.height):
        raise ValueError("左目图尺寸与当前标定不一致")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturated = cv2.inRange(hsv, (0, 55, 35), (179, 255, 255))
    saturated = cv2.morphologyEx(
        saturated, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)
    )
    edges = cv2.Canny(saturated, 40, 120)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 1800.0,
        threshold=max(30, image.shape[1] // 50),
        minLineLength=max(80, image.shape[1] // 12),
        maxLineGap=max(20, image.shape[1] // 40),
    )
    if lines is None or len(lines) < 2:
        raise ValueError("未检测到足够的彩色管件直线，无法自动校正倾斜")
    observed_angles: list[float] = []
    observed_weights: list[float] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = math.hypot(dx, dy)
        if length > 0:
            observed_angles.append(math.degrees(math.atan2(dy, dx)))
            observed_weights.append(length)
    observed, concentration = _dominant_axis_degrees(
        np.asarray(observed_angles), np.asarray(observed_weights)
    )
    if concentration < 0.30:
        raise ValueError("彩色管件直线方向不集中，无法可靠自动校正倾斜")

    model_angles: list[float] = []
    model_weights: list[float] = []
    for pipe in pipes:
        line = np.asarray(pipe.get("centerline_world_mm"), dtype=np.float64)
        if line.shape != (2, 3) or not np.all(np.isfinite(line)):
            continue
        world_vector = line[1] - line[0]
        camera_vector = parsed.left.rotation_world_to_camera @ world_vector
        projected_length = float(np.linalg.norm(camera_vector[:2]))
        if projected_length > 1e-6:
            model_angles.append(
                math.degrees(math.atan2(camera_vector[1], camera_vector[0]))
            )
            model_weights.append(projected_length)
    expected, _model_concentration = _dominant_axis_degrees(
        np.asarray(model_angles), np.asarray(model_weights)
    )
    correction = (observed - expected + 90.0) % 180.0 - 90.0
    if abs(correction) > 20.0:
        raise ValueError(
            f"自动估计需要修正 {correction:.2f}°，超过轻微倾斜的 ±20° 范围；"
            "请先选择正确的侧立面观察方向"
        )
    return {
        "roll_correction_deg": float(correction),
        "observed_pipe_angle_deg": float(observed),
        "expected_cad_angle_deg": float(expected),
        "line_count": int(len(observed_angles)),
        "direction_concentration": concentration,
    }


def apply_camera_pose(
    calibration: Mapping[str, Any],
    adjustment: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a calibration copy with a rigid stereo-rig pose adjustment.

    The input images remain rectified.  Both camera rotations and their world
    centres move as one rigid rig, so the baseline remains positive camera X.
    Yaw is about camera down, pitch about camera right, and roll about the
    optical axis.  Roll therefore corrects a tilted projection in the image.
    """
    from .stereo_analyzer import _calibration_from_manifest

    if not isinstance(calibration, Mapping):
        raise ValueError("stereo calibration must be an object")
    parsed = _calibration_from_manifest(calibration)
    if adjustment is None:
        return copy.deepcopy(dict(calibration))
    if not isinstance(adjustment, Mapping):
        raise ValueError("camera pose adjustment must be an object")
    mode = adjustment.get("mode", "keep")
    if mode not in POSE_MODE_LABELS:
        raise ValueError(f"unsupported camera position mode: {mode!r}")
    if mode == "keep":
        return copy.deepcopy(dict(calibration))

    yaw = _finite(adjustment.get("yaw_deg", 0.0), "yaw_deg", bound=180.0)
    pitch = _finite(adjustment.get("pitch_deg", 0.0), "pitch_deg", bound=180.0)
    roll = _finite(adjustment.get("roll_deg", 0.0), "roll_deg", bound=180.0)
    center_value = adjustment.get("center_world_mm")
    if center_value is None:
        center = (parsed.left.center_world_mm + parsed.right.center_world_mm) / 2
    else:
        if not isinstance(center_value, (list, tuple)) or len(center_value) != 3:
            raise ValueError("center_world_mm must contain three finite values within ±1000000 mm")
        center = np.asarray(
            [
                _finite(value, f"center_world_mm[{index}]", bound=1_000_000)
                for index, value in enumerate(center_value)
            ],
            dtype=float,
        )
    registration_validated = adjustment.get("registration_validated", False)
    if type(registration_validated) is not bool:
        raise ValueError("registration_validated must be a boolean")

    if mode == "adjust_current":
        base_rotation = parsed.left.rotation_world_to_camera
    else:
        forward, up = _PRESET_FORWARD_UP[mode]
        base_rotation = _look_rotation(forward, up)
    # Corrections are expressed in the camera image frame.  Pre-multiplication
    # rotates the camera axes while keeping the convention x=right, y=down,
    # z=forward.  The exact order is recorded in the manifest audit below.
    correction = _rotation("z", roll) @ _rotation("x", pitch) @ _rotation("y", yaw)
    rotation = correction @ base_rotation
    baseline_world = rotation.T @ np.array([parsed.baseline_mm, 0.0, 0.0])
    left_center = center - baseline_world / 2
    right_center = center + baseline_world / 2

    result = copy.deepcopy(dict(calibration))
    result["left_camera"]["rotation_world_to_camera"] = rotation.tolist()
    result["right_camera"]["rotation_world_to_camera"] = rotation.tolist()
    result["left_camera"]["center_world_mm"] = left_center.tolist()
    result["right_camera"]["center_world_mm"] = right_center.tolist()
    result["registration_validated"] = registration_validated
    audit = {
        "mode": mode,
        "mode_label_zh": POSE_MODE_LABELS[mode],
        "rig_center_world_mm": center.tolist(),
        "yaw_deg": yaw,
        "pitch_deg": pitch,
        "roll_deg": roll,
        "registration_validated": registration_validated,
        "rotation_order": "camera_Rz_roll @ camera_Rx_pitch @ camera_Ry_yaw @ base_world_to_camera",
        "definition": "Rigid CAD-to-rectified-stereo extrinsic adjustment; image pixels were not warped",
    }
    signature = hashlib.sha256(
        json.dumps(audit, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    result["calibration_id"] = f"{parsed.calibration_id}-pose-{signature}"
    result["registration_adjustment"] = audit
    _calibration_from_manifest(result)
    return result


__all__ = [
    "POSE_MODE_LABELS",
    "SIDE_ELEVATION_VIEWS",
    "apply_camera_pose",
    "calibration_pose",
    "estimate_pipe_roll_correction",
    "model_center_from_pipes",
    "suggested_side_view",
]
