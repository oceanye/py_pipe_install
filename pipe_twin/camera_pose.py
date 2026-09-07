"""Auditable stereo-rig placement and tilt adjustments in CAD coordinates."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np


POSE_MODE_LABELS = {
    "keep": "沿用标定文件位姿",
    "adjust_current": "当前位姿 + 角度/位置修正",
    "positive_z": "相机位于 +Z，朝向 -Z",
    "negative_z": "相机位于 -Z，朝向 +Z",
    "positive_x": "相机位于 +X，朝向 -X",
    "negative_x": "相机位于 -X，朝向 +X",
    "positive_y": "相机位于 +Y，朝向 -Y（俯视）",
    "negative_y": "相机位于 -Y，朝向 +Y（仰视）",
}


_PRESET_FORWARD_UP = {
    "positive_z": ([0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
    "negative_z": ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
    "positive_x": ([-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    "negative_x": ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    "positive_y": ([0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
    "negative_y": ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
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
    "apply_camera_pose",
    "calibration_pose",
]
