"""Optional model-coordinate viewing hemisphere, independent of pipe axis."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


CAMERA_SIDE_PRESETS = {
    "自动判断（未指定）": None,
    "从 +X 侧朝 -X 看": [1., 0., 0.],
    "从 -X 侧朝 +X 看": [-1., 0., 0.],
    "从 +Y 侧朝 -Y 看": [0., 1., 0.],
    "从 -Y 侧朝 +Y 看": [0., -1., 0.],
    "从 +Z 侧朝 -Z 看": [0., 0., 1.],
    "从 -Z 侧朝 +Z 看": [0., 0., -1.],
}
VIEW_POLICY = {
    "revision": "camera-side-hemisphere-v1",
    "coordinate_frame": "MODEL",
    "side_vector_convention": "SCENE_TO_CAMERA_VIEWING_HEMISPHERE",
    "minimum_facing_cosine_exclusive": 1.e-6,
}
VIEW_POLICY_SHA256 = hashlib.sha256(json.dumps(VIEW_POLICY, sort_keys=True).encode()).hexdigest()


def normalize_camera_side(value: Any) -> list[float] | None:
    if value is None:
        return None
    if (not isinstance(value, (list, tuple)) or len(value) != 3
            or any(type(v) not in (int, float) for v in value)):
        raise ValueError("相机观察侧必须为模型坐标中的三个数值")
    vector = np.asarray(value, float)
    length = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or not np.isfinite(length) or length < 1.e-9:
        raise ValueError("相机观察侧不能为零或非有限数")
    return (vector / length).tolist()


def camera_side_label(value: Any) -> str:
    side = normalize_camera_side(value)
    for label, preset in CAMERA_SIDE_PRESETS.items():
        if (side is None and preset is None) or (side is not None and preset is not None
                                                 and np.allclose(side, preset, atol=1.e-9, rtol=0)):
            return label
    return "自定义观察侧 " + str(np.round(side, 3).tolist())


def preview_rotation(side: list[float]) -> np.ndarray:
    """Right/up/towards-viewer rows for an orthographic model preview."""
    towards_viewer = np.asarray(normalize_camera_side(side), float)
    up = np.array([0., 0., 1.])
    if abs(float(up @ towards_viewer)) > .95:
        up = np.array([0., 1., 0.])
    right = np.cross(up, towards_viewer)
    right /= np.linalg.norm(right)
    return np.vstack((right, np.cross(towards_viewer, right), towards_viewer))
