"""Persistent workbench profile: one-time field setup that survives restarts.

The profile stores everything an operator configures once — model binding,
pipe catalog (identity/diameter/color), the current stereo calibration with
its QR/pose adjustment chain, the rectification recipe for wizard-made
calibrations, QR print geometry, camera device selection, and the last
manifest — so a new session starts from the previous state instead of
re-entering it.  Layout follows the workbook pattern in ``measurement_book``:
kind/schema constants, fail-closed validation, atomic writes.

The module deliberately imports neither tkinter nor OpenCV at module level;
the security-relevant calibration/pipe validators from ``stereo_analyzer``
are imported lazily inside the validation paths that need them.
"""

from __future__ import annotations

import copy
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .pipeline import atomic_write_text


KIND = "pipe-twin-workbench-profile"
SCHEMA_VERSION = "1.0"
CALIBRATION_BUNDLE_KIND = "pipe-twin-camera-calibration"
CALIBRATION_BUNDLE_SCHEMA_VERSION = "1.0"

# Keep in sync with measurement_gui.OUTPUT_ROOT; duplicated here so this data
# module does not pull in a Tk panel module.
_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "measurement_workbench"
PROFILE_NAME = "workbench_profile.json"
CALIBRATION_NAME = "calibration_current.json"
STL_UNITS = ("millimeter", "centimeter", "meter", "inch")
_MODEL_SUFFIXES = {".3dm", ".3mf", ".stl"}

_QR_SETTINGS_KEYS = {
    "marker_id",
    "marker_edge_mm",
    "measured_marker_edge_mm",
    "marker_center_world_mm",
    "print_right_world",
    "print_up_world",
    "max_reprojection_rms_px",
}
# The two live-capture safety confirmations must be re-ticked by a human on
# every QrRegistrationDialog open (capture_gui keeps them hard-coded False);
# they must never be restored from disk.
_QR_FORBIDDEN_KEYS = {"print_measured", "cad_confirmed"}
_POSE_MODES = {
    "keep",
    "adjust_current",
    "positive_x",
    "negative_x",
    "positive_y",
    "negative_y",
    "positive_z",
    "negative_z",
}
_POSE_KEYS = {
    "mode",
    "center_world_mm",
    "yaw_deg",
    "pitch_deg",
    "roll_deg",
    "registration_validated",
}
_CAMERA_LAYOUTS = {
    "side_by_side_left_right",
    "side_by_side_right_left",
    "separate_devices",
}
_SECTION_KEYS = (
    "model_path",
    "stl_unit",
    "pipes",
    "calibration_path",
    "calibration_current",
    "rectification_recipe",
    "qr_settings",
    "pose_adjustment",
    "side_view_distance_mm",
    "camera",
    "last_manifest_path",
    "chessboard",
    "wizard",
    "capture_history",
)
_OPERATOR_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")
_DIRECTION_PATTERN = re.compile(r"^[+-][XYZ]$")


def default_profile_path() -> Path:
    return _ROOT / PROFILE_NAME


def default_calibration_path() -> Path:
    return _ROOT / CALIBRATION_NAME


def default_profile() -> dict:
    return {
        "kind": KIND,
        "schema_version": SCHEMA_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "model_path": "",
        "stl_unit": "millimeter",
        "pipes": [],
        "calibration_path": "",
        "calibration_current": None,
        "rectification_recipe": None,
        "qr_settings": {},
        "pose_adjustment": {"mode": "keep"},
        "side_view_distance_mm": 1000.0,
        "camera": {"layout": "side_by_side_left_right", "left_index": 0, "right_index": 1},
        "last_manifest_path": "",
        "chessboard": {"square_mm": 20.0, "columns": 9, "rows": 7, "dpi": 300},
        "wizard": {
            "operator": "field",
            "max_reprojection_rms_px": 1.5,
            "min_pairs": 10,
            "expected_baseline_mm": 0.0,
            "nominal_fov_deg": 0.0,
            "nominal_focal_length_mm": 0.0,
        },
        "capture_history": False,
    }


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{label} 必须是{'正的' if positive else ''}有限数值")
    return float(value)


def _matrix(payload: Any, shape: tuple[int, int], label: str) -> list[list[float]]:
    if not isinstance(payload, list) or len(payload) != shape[0]:
        raise ValueError(f"{label} 必须是 {shape[0]}×{shape[1]} 数值矩阵")
    rows = []
    for row in payload:
        if not isinstance(row, list) or len(row) != shape[1]:
            raise ValueError(f"{label} 必须是 {shape[0]}×{shape[1]} 数值矩阵")
        rows.append([_number(value, label) for value in row])
    return rows


def _vector3(payload: Any, label: str) -> list[float]:
    if not isinstance(payload, list) or len(payload) != 3:
        raise ValueError(f"{label} 必须是三个有限数值")
    return [_number(value, label) for value in payload]


def _distortion(payload: Any, label: str) -> list[float]:
    if not isinstance(payload, list) or len(payload) not in {0, 4, 5, 8, 12, 14}:
        raise ValueError(f"{label} 必须是 0/4/5/8/12/14 个畸变系数")
    return [_number(value, label) for value in payload]


def _rotation_matrix(payload: Any, label: str) -> list[list[float]]:
    matrix = np.asarray(_matrix(payload, (3, 3), label), dtype=float)
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6) or not math.isclose(
        float(np.linalg.det(matrix)), 1.0, abs_tol=1e-6
    ):
        raise ValueError(f"{label} 必须是正交旋转矩阵")
    return matrix.tolist()


def validate_rectification_recipe(
    payload: Any, *, calibration: Mapping | None = None
) -> dict:
    """Fail-closed validation of a wizard rectification recipe (pure JSON)."""
    required_keys = {
        "calibration_id",
        "K1",
        "D1",
        "K2",
        "D2",
        "R1",
        "R2",
        "P1",
        "P2",
        "image_width_px",
        "image_height_px",
        "output_width_px",
        "output_height_px",
        "alpha",
        "created_at",
        "definition",
    }
    optional_keys = {"right_frame_transform"}
    if not isinstance(payload, dict):
        raise ValueError("极线矫正配方必须是 JSON 对象")
    if not required_keys.issubset(payload) or set(payload) - required_keys - optional_keys:
        missing = sorted(required_keys - set(payload))
        extra = sorted(set(payload) - required_keys - optional_keys)
        raise ValueError(f"极线矫正配方字段不匹配（缺少 {missing}，多余 {extra}）")
    if not str(payload["calibration_id"]).strip():
        raise ValueError("极线矫正配方缺少 calibration_id")
    recipe = json.loads(json.dumps(payload, allow_nan=False))
    frame_transform = recipe.get("right_frame_transform", "none")
    if frame_transform not in {"none", "flip_horizontal", "flip_vertical", "rotate_180"}:
        raise ValueError("极线矫正配方 right_frame_transform 无效")
    recipe["right_frame_transform"] = frame_transform
    for name in ("K1", "K2"):
        matrix = np.asarray(_matrix(recipe[name], (3, 3), name), dtype=float)
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not math.isclose(matrix[2, 2], 1.0, abs_tol=1e-9):
            raise ValueError(f"极线矫正配方 {name} 不是有效的内参矩阵")
        if abs(matrix[0, 1]) > 1e-6 or abs(matrix[1, 0]) > 1e-6:
            raise ValueError(f"极线矫正配方 {name} 不应包含轴倾斜项")
        recipe[name] = matrix.tolist()
    for name in ("D1", "D2"):
        recipe[name] = _distortion(recipe[name], name)
    for name in ("R1", "R2"):
        recipe[name] = _rotation_matrix(recipe[name], name)
    for name in ("P1", "P2"):
        matrix = np.asarray(_matrix(recipe[name], (3, 4), name), dtype=float)
        if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError(f"极线矫正配方 {name} 不是有效的投影矩阵")
        recipe[name] = matrix.tolist()
    p1 = np.asarray(recipe["P1"], dtype=float)
    p2 = np.asarray(recipe["P2"], dtype=float)
    if not np.allclose(p1[:, :3], p2[:, :3], rtol=1e-6, atol=1e-6):
        raise ValueError("极线矫正配方 P1/P2 必须共享同一个矫正后内参矩阵")
    if not np.allclose(p1[:, 3], 0.0, atol=1e-6):
        raise ValueError("极线矫正配方 P1 的平移列必须为零")
    if not np.allclose(p2[1:, 3], 0.0, atol=1e-6):
        raise ValueError("极线矫正配方仅支持水平双目，P2 不得包含垂直投影平移")
    encoded_baseline = -float(p2[0, 3]) / float(p2[0, 0])
    if not math.isfinite(encoded_baseline) or encoded_baseline <= 0:
        raise ValueError("极线矫正配方 P2 必须编码正的水平基线")
    for key in ("image_width_px", "image_height_px", "output_width_px", "output_height_px"):
        if type(recipe[key]) is not int or recipe[key] <= 0:
            raise ValueError(f"极线矫正配方 {key} 必须是正整数")
    if (recipe["output_width_px"], recipe["output_height_px"]) != (
        recipe["image_width_px"],
        recipe["image_height_px"],
    ):
        raise ValueError("极线矫正配方输出尺寸必须与图像尺寸一致")
    alpha = _number(recipe["alpha"], "极线矫正配方 alpha")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("极线矫正配方 alpha 必须在 0 到 1 之间")
    recipe["alpha"] = alpha
    for key in ("created_at", "definition"):
        if not isinstance(recipe[key], str) or not recipe[key].strip():
            raise ValueError(f"极线矫正配方缺少 {key}")
    if calibration is not None:
        from .stereo_analyzer import _calibration_from_manifest

        parsed = _calibration_from_manifest(calibration)
        if not calibration_ids_match(
            parsed.calibration_id, str(recipe["calibration_id"])
        ):
            raise ValueError("极线矫正配方不属于当前相机标定")
        if (recipe["image_width_px"], recipe["image_height_px"]) != (
            parsed.left.width,
            parsed.left.height,
        ):
            raise ValueError("极线矫正配方尺寸与当前标定的图像尺寸不一致")
        if not np.allclose(p1[:, :3], parsed.left.intrinsic, rtol=1e-6, atol=1e-6):
            raise ValueError("极线矫正配方 P1 与当前左目矫正内参不一致")
        if not np.allclose(p2[:, :3], parsed.right.intrinsic, rtol=1e-6, atol=1e-6):
            raise ValueError("极线矫正配方 P2 与当前右目矫正内参不一致")
        if not math.isclose(
            encoded_baseline, parsed.baseline_mm, rel_tol=1e-4, abs_tol=1e-3
        ):
            raise ValueError("极线矫正配方 P2 基线与当前标定不一致")
    return recipe


def _validate_qr_settings(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("qr_settings 必须是对象")
    forbidden = sorted(set(payload) & _QR_FORBIDDEN_KEYS)
    if forbidden:
        raise ValueError(f"qr_settings 不允许持久化现场安全确认项：{forbidden}")
    if set(payload) - _QR_SETTINGS_KEYS:
        raise ValueError("qr_settings 包含不支持的字段")
    result = json.loads(json.dumps(payload, allow_nan=False))
    if "marker_id" in result and not isinstance(result["marker_id"], str):
        raise ValueError("qr_settings.marker_id 必须是文本")
    for key in ("marker_edge_mm", "measured_marker_edge_mm", "max_reprojection_rms_px"):
        if key in result:
            result[key] = _number(result[key], f"qr_settings.{key}", positive=True)
    if "marker_center_world_mm" in result:
        result["marker_center_world_mm"] = _vector3(
            result["marker_center_world_mm"], "qr_settings.marker_center_world_mm"
        )
    for key in ("print_right_world", "print_up_world"):
        if key in result and (
            not isinstance(result[key], str) or not _DIRECTION_PATTERN.match(result[key])
        ):
            raise ValueError(f"qr_settings.{key} 必须是 +X/-X/+Y/-Y/+Z/-Z 之一")
    return result


def _validate_pose_adjustment(payload: Any) -> dict:
    if not isinstance(payload, dict) or "mode" not in payload:
        raise ValueError("pose_adjustment 必须包含 mode")
    if set(payload) - _POSE_KEYS:
        raise ValueError("pose_adjustment 包含不支持的字段")
    if payload["mode"] not in _POSE_MODES:
        raise ValueError("pose_adjustment.mode 无效")
    result = json.loads(json.dumps(payload, allow_nan=False))
    if "center_world_mm" in result:
        result["center_world_mm"] = _vector3(
            result["center_world_mm"], "pose_adjustment.center_world_mm"
        )
        if any(abs(value) > 1_000_000 for value in result["center_world_mm"]):
            raise ValueError("pose_adjustment.center_world_mm 超出支持范围")
    for key in ("yaw_deg", "pitch_deg", "roll_deg"):
        if key in result:
            angle = _number(result[key], f"pose_adjustment.{key}")
            if not -180.0 <= angle <= 180.0:
                raise ValueError(f"pose_adjustment.{key} 必须在 ±180 度内")
            result[key] = angle
    if "registration_validated" in result and type(result["registration_validated"]) is not bool:
        raise ValueError("pose_adjustment.registration_validated 必须是布尔值")
    return result


def _validate_camera(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("camera 必须是对象")
    if set(payload) - {
        "layout",
        "left_index",
        "right_index",
        "right_frame_transform",
    }:
        raise ValueError("camera 包含不支持的字段")
    legacy_transform = payload.get("right_frame_transform", "none")
    if legacy_transform not in {
        "none",
        "flip_horizontal",
        "flip_vertical",
        "rotate_180",
    }:
        raise ValueError("camera.right_frame_transform 无效")
    layout = payload.get("layout")
    if layout not in _CAMERA_LAYOUTS:
        raise ValueError("camera.layout 无效")
    left_index = payload.get("left_index")
    if type(left_index) is not int or left_index < 0:
        raise ValueError("camera.left_index 必须是非负整数")
    right_index = payload.get("right_index")
    if layout == "separate_devices":
        if type(right_index) is not int or right_index < 0:
            raise ValueError("独立设备模式下 camera.right_index 必须是非负整数")
        if right_index == left_index:
            raise ValueError("独立设备模式下左右相机索引不能相同")
    elif right_index is not None and (type(right_index) is not int or right_index < 0):
        raise ValueError("camera.right_index 必须是非负整数")
    return {"layout": layout, "left_index": left_index, "right_index": right_index}


def _validate_chessboard(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("chessboard 必须是对象")
    if set(payload) - {"square_mm", "columns", "rows", "dpi"}:
        raise ValueError("chessboard 包含不支持的字段")
    square_mm = _number(payload.get("square_mm"), "chessboard.square_mm", positive=True)
    if not 5.0 <= square_mm <= 200.0:
        raise ValueError("chessboard.square_mm 应在 5 到 200 mm 之间")
    for key in ("columns", "rows"):
        if type(payload.get(key)) is not int or payload[key] < 4:
            raise ValueError(f"chessboard.{key} 必须是不小于 4 的整数")
    if type(payload.get("dpi")) is not int or not 150 <= payload["dpi"] <= 1200:
        raise ValueError("chessboard.dpi 必须在 150 到 1200 之间")
    return {"square_mm": square_mm, "columns": payload["columns"], "rows": payload["rows"], "dpi": payload["dpi"]}


def _validate_wizard(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("wizard 必须是对象")
    if set(payload) - {
        "operator",
        "max_reprojection_rms_px",
        "min_pairs",
        "expected_baseline_mm",
        "nominal_fov_deg",
        "nominal_focal_length_mm",
    }:
        raise ValueError("wizard 包含不支持的字段")
    operator = payload.get("operator")
    if not isinstance(operator, str) or not _OPERATOR_PATTERN.match(operator):
        raise ValueError("wizard.operator 只能包含字母、数字、点、下划线和连字符（≤32 字符）")
    max_rms = _number(payload.get("max_reprojection_rms_px"), "wizard.max_reprojection_rms_px", positive=True)
    if type(payload.get("min_pairs")) is not int or payload["min_pairs"] < 3:
        raise ValueError("wizard.min_pairs 必须是不小于 3 的整数")
    expected_baseline = _number(
        payload.get("expected_baseline_mm", 0.0),
        "wizard.expected_baseline_mm",
    )
    if expected_baseline != 0.0 and not 20.0 <= expected_baseline <= 2000.0:
        raise ValueError("wizard.expected_baseline_mm 必须为 0 或在 20 到 2000 mm 之间")
    nominal_fov = _number(
        payload.get("nominal_fov_deg", 0.0), "wizard.nominal_fov_deg"
    )
    if nominal_fov != 0.0 and not 10.0 <= nominal_fov < 180.0:
        raise ValueError("wizard.nominal_fov_deg 必须为 0 或在 10 到 180 度之间")
    nominal_focal = _number(
        payload.get("nominal_focal_length_mm", 0.0),
        "wizard.nominal_focal_length_mm",
    )
    if nominal_focal != 0.0 and not 0.1 <= nominal_focal <= 100.0:
        raise ValueError("wizard.nominal_focal_length_mm 必须为 0 或在 0.1 到 100 mm 之间")
    return {
        "operator": operator,
        "max_reprojection_rms_px": max_rms,
        "min_pairs": payload["min_pairs"],
        "expected_baseline_mm": expected_baseline,
        "nominal_fov_deg": nominal_fov,
        "nominal_focal_length_mm": nominal_focal,
    }


def validate_profile(payload: Any) -> dict:
    """Validate a workbench profile and return a normalized deep copy."""
    from .stereo_analyzer import _calibration_from_manifest, _pipe_from_manifest

    if not isinstance(payload, dict) or payload.get("kind") != KIND or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"请选择工作台配置 JSON（{KIND}，版本 {SCHEMA_VERSION}）")
    if set(payload) - set(_SECTION_KEYS) - {"kind", "schema_version", "updated_at"}:
        raise ValueError("工作台配置包含不支持的段落")
    try:
        datetime.fromisoformat(str(payload.get("updated_at", "")))
    except ValueError as error:
        raise ValueError("工作台配置缺少有效的 updated_at 时间") from error
    result = json.loads(json.dumps(payload, allow_nan=False))

    model_path = result.get("model_path")
    if not isinstance(model_path, str):
        raise ValueError("model_path 必须是文本")
    if model_path and Path(model_path).suffix.lower() not in _MODEL_SUFFIXES:
        raise ValueError("model_path 需要 3DM、3MF 或 STL 文件")
    if result.get("stl_unit") not in STL_UNITS:
        raise ValueError("stl_unit 无效")
    pipes = result.get("pipes")
    if not isinstance(pipes, list):
        raise ValueError("pipes 必须是列表")
    pipe_ids = set()
    for index, pipe in enumerate(pipes):
        parsed = _pipe_from_manifest(pipe, index)
        if parsed.pipe_id in pipe_ids:
            raise ValueError("管件业务 ID 不能重复")
        pipe_ids.add(parsed.pipe_id)
    if not isinstance(result.get("calibration_path"), str):
        raise ValueError("calibration_path 必须是文本")
    calibration = result.get("calibration_current")
    if calibration is not None:
        _calibration_from_manifest(calibration)
    recipe = result.get("rectification_recipe")
    camera_payload = result.get("camera", {})
    legacy_transform = (
        camera_payload.get("right_frame_transform", "none")
        if isinstance(camera_payload, dict)
        else "none"
    )
    if isinstance(recipe, dict) and "right_frame_transform" not in recipe:
        recipe = dict(recipe)
        recipe["right_frame_transform"] = legacy_transform
    result["rectification_recipe"] = (
        None
        if recipe is None
        else validate_rectification_recipe(recipe, calibration=calibration)
    )
    result["qr_settings"] = _validate_qr_settings(result.get("qr_settings", {}))
    result["pose_adjustment"] = _validate_pose_adjustment(result.get("pose_adjustment", {}))
    result["side_view_distance_mm"] = _number(
        result.get("side_view_distance_mm"), "side_view_distance_mm", positive=True
    )
    result["camera"] = _validate_camera(result.get("camera", {}))
    if not isinstance(result.get("last_manifest_path"), str):
        raise ValueError("last_manifest_path 必须是文本")
    result["chessboard"] = _validate_chessboard(result.get("chessboard", {}))
    result["wizard"] = _validate_wizard(result.get("wizard", {}))
    if type(result.get("capture_history")) is not bool:
        raise ValueError("capture_history 必须是布尔值")
    return result


def load_profile(path: str | Path | None = None) -> tuple[dict | None, str]:
    """Load the profile; quarantine unreadable files and fail closed to empty.

    Returns ``(profile, "")`` on success, ``(None, "")`` when no profile file
    exists yet, and ``(None, message)`` when the file exists but is unusable.
    """
    target = Path(path) if path is not None else default_profile_path()
    if not target.exists():
        return None, ""
    try:
        profile = validate_profile(json.loads(target.read_text(encoding="utf-8-sig")))
        return profile, ""
    except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as error:
        message = f"工作台配置无法载入：{error}"
        recovery = target.with_name(
            f"{target.stem}_recovery_{datetime.now():%Y%m%d_%H%M%S_%f}{target.suffix}"
        )
        try:
            target.replace(recovery)
            message += f"；原文件已移至 {recovery.name}"
        except OSError:
            pass
        return None, message


def save_profile(profile: Mapping, *, path: str | Path | None = None) -> Path:
    """Validate, rotate the previous file to ``.bak``, and atomically write."""
    target = Path(path) if path is not None else default_profile_path()
    valid = validate_profile(profile)
    valid["updated_at"] = datetime.now(timezone.utc).isoformat()
    if target.exists():
        backup = target.with_name(target.name + ".bak")
        atomic_write_text(backup, target.read_text(encoding="utf-8"))
    return atomic_write_text(
        target, json.dumps(valid, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def reset_profile(*, path: str | Path | None = None) -> Path:
    """Move any existing profile aside and write a fresh default."""
    target = Path(path) if path is not None else default_profile_path()
    if target.exists():
        recovery = target.with_name(
            f"{target.stem}_recovery_{datetime.now():%Y%m%d_%H%M%S_%f}{target.suffix}"
        )
        try:
            target.replace(recovery)
        except OSError:
            pass
    return save_profile(default_profile(), path=target)


def update_profile(updates: Mapping[str, Any], *, path: str | Path | None = None) -> Path:
    """Merge partial section updates into the stored profile and save."""
    current, _problem = load_profile(path)
    profile = current or default_profile()
    unknown = sorted(set(updates) - set(_SECTION_KEYS))
    if unknown:
        raise ValueError(f"不支持的配置段落：{unknown}")
    merged = copy.deepcopy(profile) | copy.deepcopy(dict(updates))
    return save_profile(merged, path=path)


def capture_state_from_profile(profile: Mapping) -> dict:
    """Project a validated profile onto the capture dialog's flat state."""
    validated = validate_profile(profile)
    return {
        "model_path": validated["model_path"],
        "stl_unit": validated["stl_unit"],
        "pipes": copy.deepcopy(validated["pipes"]),
        "calibration_path": validated["calibration_path"],
        "calibration_current": copy.deepcopy(validated["calibration_current"]),
        "rectification_recipe": copy.deepcopy(validated["rectification_recipe"]),
        "qr_settings": copy.deepcopy(validated["qr_settings"]),
        "pose_adjustment": copy.deepcopy(validated["pose_adjustment"]),
        "side_view_distance_mm": validated["side_view_distance_mm"],
        "camera": copy.deepcopy(validated["camera"]),
        "last_manifest_path": validated["last_manifest_path"],
        "capture_history": validated["capture_history"],
    }


def profile_sections_from_state(state: Mapping[str, Any], *, sections: Iterable[str]) -> dict:
    """Build a partial profile update from the dialog's flat state keys."""
    unknown = sorted(set(sections) - set(_SECTION_KEYS))
    if unknown:
        raise ValueError(f"不支持的配置段落：{unknown}")
    update: dict[str, Any] = {}
    for section in sections:
        if section not in state:
            raise ValueError(f"状态缺少配置段落 {section}")
        update[section] = copy.deepcopy(state[section])
    return update


def calibration_ids_match(current_id: str, recipe_id: str) -> bool:
    """A recipe stays valid while the id carries -qr-/-pose- chain suffixes."""
    return bool(current_id) and bool(recipe_id) and (
        current_id == recipe_id or current_id.startswith(recipe_id + "-")
    )


def write_standalone_calibration(
    calibration: Mapping,
    *,
    path: str | Path | None = None,
    note: str = "由棋盘格标定向导生成；CAD 配准（二维码/方向）请用工作台继续配置。",
) -> Path:
    """Write a calibration JSON in the same shape as the dialog example file."""
    from .stereo_analyzer import _calibration_from_manifest

    _calibration_from_manifest(calibration)
    target = Path(path) if path is not None else default_calibration_path()
    payload = {
        "note": note,
        "stereo_calibration": copy.deepcopy(dict(calibration)),
    }
    return atomic_write_text(
        target, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def camera_calibration_bundle(
    calibration: Mapping,
    *,
    rectification_recipe: Mapping | None = None,
    qr_settings: Mapping | None = None,
) -> dict:
    """Build a portable camera calibration result with its remap recipe."""
    from .stereo_analyzer import _calibration_from_manifest

    calibration_copy = copy.deepcopy(dict(calibration))
    parsed = _calibration_from_manifest(calibration_copy)
    recipe = (
        None
        if rectification_recipe is None
        else validate_rectification_recipe(
            rectification_recipe, calibration=calibration_copy
        )
    )
    if recipe is not None and not calibration_ids_match(
        parsed.calibration_id, str(recipe.get("calibration_id", ""))
    ):
        raise ValueError("极线矫正配方不属于当前相机标定")
    if "-chess-" in parsed.calibration_id.casefold() and recipe is None:
        raise ValueError("棋盘格向导标定必须连同极线矫正配方一起保存")
    settings = _validate_qr_settings(dict(qr_settings or {}))
    return {
        "kind": CALIBRATION_BUNDLE_KIND,
        "schema_version": CALIBRATION_BUNDLE_SCHEMA_VERSION,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "stereo_calibration": calibration_copy,
        "rectification_recipe": recipe,
        "qr_settings": settings,
    }


def validate_camera_calibration_bundle(payload: Any) -> dict:
    """Validate a saved calibration bundle and normalize its JSON values."""
    if not isinstance(payload, dict):
        raise ValueError("相机标定结果必须是 JSON 对象")
    if payload.get("kind") != CALIBRATION_BUNDLE_KIND or payload.get(
        "schema_version"
    ) != CALIBRATION_BUNDLE_SCHEMA_VERSION:
        raise ValueError(
            f"请选择 {CALIBRATION_BUNDLE_KIND} 版本 "
            f"{CALIBRATION_BUNDLE_SCHEMA_VERSION} 的标定结果"
        )
    expected = {
        "kind",
        "schema_version",
        "saved_at",
        "stereo_calibration",
        "rectification_recipe",
        "qr_settings",
    }
    if set(payload) != expected:
        raise ValueError("相机标定结果字段不完整或包含未知字段")
    try:
        datetime.fromisoformat(str(payload.get("saved_at", "")))
    except ValueError as error:
        raise ValueError("相机标定结果缺少有效保存时间") from error
    normalized = camera_calibration_bundle(
        payload["stereo_calibration"],
        rectification_recipe=payload["rectification_recipe"],
        qr_settings=payload["qr_settings"],
    )
    normalized["saved_at"] = str(payload["saved_at"])
    return normalized


def save_camera_calibration_bundle(
    path: str | Path,
    calibration: Mapping,
    *,
    rectification_recipe: Mapping | None = None,
    qr_settings: Mapping | None = None,
) -> Path:
    """Atomically save a portable intrinsic + QR registration result."""
    target = Path(path)
    if target.suffix.lower() != ".json":
        target = target.with_suffix(".json")
    payload = camera_calibration_bundle(
        calibration,
        rectification_recipe=rectification_recipe,
        qr_settings=qr_settings,
    )
    return atomic_write_text(
        target, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def load_camera_calibration_bundle(path: str | Path) -> dict:
    """Read a saved portable camera calibration result."""
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"相机标定结果不存在或不是文件：{source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"相机标定结果 JSON 无效：{error}") from error
    return validate_camera_calibration_bundle(payload)


__all__ = [
    "CALIBRATION_BUNDLE_KIND",
    "CALIBRATION_BUNDLE_SCHEMA_VERSION",
    "CALIBRATION_NAME",
    "KIND",
    "PROFILE_NAME",
    "SCHEMA_VERSION",
    "STL_UNITS",
    "calibration_ids_match",
    "camera_calibration_bundle",
    "capture_state_from_profile",
    "default_calibration_path",
    "default_profile",
    "default_profile_path",
    "load_profile",
    "load_camera_calibration_bundle",
    "profile_sections_from_state",
    "reset_profile",
    "save_profile",
    "save_camera_calibration_bundle",
    "update_profile",
    "validate_profile",
    "validate_camera_calibration_bundle",
    "validate_rectification_recipe",
    "write_standalone_calibration",
]
