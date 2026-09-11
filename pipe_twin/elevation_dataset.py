"""Self-contained field packages for the registration-free elevation mode.

The elevation workflow only needs a rectified stereo calibration, one paired
left/right image and a small pipe-region catalogue.  This module deliberately
keeps that package independent from the CAD mesh and QR registration formats.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping
from uuid import uuid4

import cv2
import numpy as np

from .photo_capture import load_photo_snapshot, resolve_photo_path
from .pipeline import atomic_write_text
from .stereo_analyzer import _calibration_from_manifest

_TIMESTAMP_SOURCES = {
    "CAMERA_HARDWARE_CLOCK", "CAMERA_SYSTEM_CLOCK", "EXIF_DATETIME_ORIGINAL",
    "HOST_SYSTEM_CLOCK", "MANIFEST_OPERATOR_CONFIRMED",
}
_MARKERS = ("SYNTHETIC", "DEMO", "EXAMPLE", "REPLACE_WITH_REAL")
_PROVENANCE_KEYS = {"capture_backend", "capture_layout", "capture_device_index", "side_by_side_order", "capture_sync_method"}


def _timestamp(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}不能为空")
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{field}必须是 ISO-8601 时间") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        parsed = parsed.astimezone()
    return parsed.isoformat(timespec="milliseconds")


def _sync_delta_ms(left: str, right: str) -> float:
    return abs((datetime.fromisoformat(left) - datetime.fromisoformat(right)).total_seconds() * 1000.0)


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _specs(value: Any, width: int, height: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("pipe_specs必须是非空列表")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ValueError(f"pipe_specs[{i}]必须是对象")
        pipe_id = raw.get("pipe_id")
        if not isinstance(pipe_id, str) or not pipe_id.strip() or pipe_id in seen:
            raise ValueError("pipe_id必须是唯一非空字符串")
        color = raw.get("color_srgb")
        if not isinstance(color, str) or len(color) != 7 or color[0] != "#":
            raise ValueError(f"{pipe_id}.color_srgb必须是#RRGGBB")
        try:
            int(color[1:], 16)
        except ValueError as exc:
            raise ValueError(f"{pipe_id}.color_srgb必须是#RRGGBB") from exc
        diameter = raw.get("nominal_diameter_mm")
        if type(diameter) not in (int, float) or not math.isfinite(float(diameter)) or float(diameter) <= 0:
            raise ValueError(f"{pipe_id}.nominal_diameter_mm必须为正数")
        item: dict[str, Any] = {
            "pipe_id": pipe_id,
            "color_srgb": color.upper(),
            "nominal_diameter_mm": float(diameter),
        }
        for role in ("left", "right"):
            region = raw.get(f"{role}_region_px")
            if not isinstance(region, (list, tuple)) or len(region) != 4 or any(type(x) is not int for x in region):
                raise ValueError(f"{pipe_id}.{role}_region_px必须是整数[x,y,width,height]")
            x, y, w, h = region
            if x < 0 or y < 0 or w < 8 or h < 8 or x + w > width or y + h > height:
                raise ValueError(f"{pipe_id}.{role}_region_px超出图像范围")
            item[f"{role}_region_px"] = [x, y, w, h]
        if "expected_depth_mm" in raw:
            depth = raw["expected_depth_mm"]
            if type(depth) not in (int, float) or not math.isfinite(float(depth)) or float(depth) <= 0:
                raise ValueError(f"{pipe_id}.expected_depth_mm必须为正数")
            item["expected_depth_mm"] = float(depth)
        if "axis" in raw:
            axis = raw["axis"]
            if axis not in ("auto", "horizontal", "vertical"):
                raise ValueError(f"{pipe_id}.axis必须为auto、horizontal或vertical")
            item["axis"] = axis
        if raw.get("cad_object_id") is not None:
            if not isinstance(raw["cad_object_id"], str) or not raw["cad_object_id"].strip():
                raise ValueError(f"{pipe_id}.cad_object_id必须是非空字符串")
            item["cad_object_id"] = raw["cad_object_id"]
        # GUI previews may carry a design centreline or other JSON metadata.
        # Keep it in the package, while the elevation engine intentionally
        # consumes only the required colour/region/diameter fields.
        for key in ("centerline_world_mm", "source_label"):
            if key in raw:
                item[key] = copy.deepcopy(raw[key])
        if "extra" in raw:
            if not isinstance(raw["extra"], Mapping):
                raise ValueError(f"{pipe_id}.extra必须是对象")
            item["extra"] = copy.deepcopy(dict(raw["extra"]))
        seen.add(pipe_id)
        result.append(item)
    return result


def _read_photo(path: Path, role: str, calibration_camera: Any, timestamp: str,
                timestamp_source: str) -> tuple[bytes, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{role}照片不是有效三通道图像")
    h, w = image.shape[:2]
    if (w, h) != (calibration_camera.width, calibration_camera.height):
        raise ValueError(f"{role}照片尺寸{w}x{h}与标定尺寸不一致")
    if timestamp_source not in _TIMESTAMP_SOURCES:
        raise ValueError(f"{role}.timestamp_source不受支持")
    return data, {
        "camera_id": calibration_camera.camera_id,
        "path": f"photos/{role}{path.suffix.lower() or '.png'}",
        "sha256": _hash(data), "expected_width": w, "expected_height": h,
        "captured_at": _timestamp(timestamp, f"{role}拍摄时间"),
        "timestamp_source": timestamp_source,
        "orientation_policy": "RAW_PIXELS_NO_EXIF_TRANSFORM",
    }


def _calibration_ok(calibration: Mapping[str, Any]) -> Any:
    if not isinstance(calibration, Mapping):
        raise ValueError("calibration必须是对象")
    cid = str(calibration.get("calibration_id", ""))
    if any(marker in cid.upper() for marker in _MARKERS):
        raise ValueError("合成/演示标定不能用于基础现场模式")
    try:
        parsed = _calibration_from_manifest(dict(calibration))
    except Exception as exc:
        raise ValueError(f"标定不可用：{exc}") from exc
    if parsed.validated is not True:
        raise ValueError("基础模式需要validated=true的真实标定")
    return parsed


def _safe_old_view(manifest_path: Path, view: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
    # load_photo_snapshot performs portable-path, root and hash validation.
    _, integrity = load_photo_snapshot(manifest_path, view)
    source = resolve_photo_path(manifest_path, str(view["path"]))
    data = source.read_bytes()
    if _hash(data) != integrity["actual_sha256"]:
        raise ValueError("历史照片在复制前发生变化")
    return data, dict(view)


def _history(old_path: Path, old: Mapping[str, Any], root_name: str) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    capture = old.get("capture")
    groups = capture.get("capture_groups", []) if isinstance(capture, Mapping) else []
    if not isinstance(groups, list):
        raise ValueError("历史capture_groups无效")
    records: list[dict[str, Any]] = []
    assets: dict[str, bytes] = {}
    for index, group in enumerate(groups):
        if not isinstance(group, Mapping) or not isinstance(group.get("views"), Mapping):
            raise ValueError("历史capture_group格式无效")
        record = copy.deepcopy(dict(group))
        views = record["views"]
        for role in ("left", "right"):
            view = views.get(role)
            if not isinstance(view, Mapping):
                raise ValueError("历史左右照片不完整")
            data, copied = _safe_old_view(old_path, view)
            suffix = Path(str(view["path"])).suffix.lower() or ".png"
            name = f"history/{index:04d}_{role}{suffix}"
            assets[name] = data
            copied["path"] = name
            views[role] = copied
        records.append(record)
    return records, assets


def elevation_history_compatible(path: Path, calibration: Mapping[str, Any],
                                  pipe_specs: list[dict], model_path: Path | None = None,
                                  stl_unit: str | None = None,
                                  rectification_recipe: Mapping[str, Any] | None = None) -> bool:
    """Return whether a prior package can be reused under this configuration.

    A changed configuration returns ``False``.  A tampered or malformed prior
    package raises, because silently dropping altered evidence would make a
    recovery look valid.
    """
    old_path = Path(path).resolve()
    if old_path.is_dir():
        old_path /= "manifest.json"
    loaded = load_elevation_dataset(old_path)
    old = loaded["manifest"]
    parsed = _calibration_ok(calibration)
    old_specs = loaded["pipe_specs"]
    new_specs = _specs(pipe_specs, parsed.left.width, parsed.left.height)
    # load_elevation_dataset checked the old calibration, recipe, model and
    # every image against their own snapshot before comparing the new setup.
    old_recipe = loaded["rectification_recipe"]
    if rectification_recipe is not None:
        from .workbench_profile import validate_rectification_recipe
        current_recipe = validate_rectification_recipe(dict(rectification_recipe), calibration=dict(calibration))
    else:
        current_recipe = None
    old_model = old.get("model") or {}
    current_hash = None
    if model_path is not None:
        model = Path(model_path).resolve()
        if not model.is_file() or model.suffix.lower() != ".stl":
            raise ValueError("model_path必须是STL文件")
        current_hash = _hash(model.read_bytes())
    same_model = (old_model.get("sha256") == current_hash and
                  bool(old_model.get("path")) == bool(model_path) and
                  (model_path is None or old_model.get("source_unit") == stl_unit))
    return bool(old.get("stereo_calibration") == dict(calibration) and old_specs == new_specs and
                same_model and old_recipe == current_recipe)


def create_elevation_dataset(*, output_root: Path, calibration: dict, left_path: Path,
                             right_path: Path, left_time: str, right_time: str,
                             pipe_specs: list[dict], pair_confirmed: bool,
                             previous_manifest: Path | None = None, model_path: Path | None = None,
                             stl_unit: str | None = None,
                             rectification_recipe: dict | None = None,
                             camera_capture_provenance: dict | None = None,
                             timestamp_sources: dict | None = None) -> Path:
    """Validate and atomically create a minimal ``elevation_depth`` package."""
    if pair_confirmed is not True:
        raise ValueError("请确认左右照片来自同一次同步拍摄")
    parsed = _calibration_ok(calibration)
    recipe = None
    if rectification_recipe is not None:
        from .workbench_profile import validate_rectification_recipe
        try:
            recipe = validate_rectification_recipe(dict(rectification_recipe), calibration=dict(calibration))
        except Exception as exc:
            raise ValueError(f"矫正配方不可用：{exc}") from exc
    if timestamp_sources is not None and (not isinstance(timestamp_sources, Mapping) or set(timestamp_sources) - {"left", "right"}):
        raise ValueError("timestamp_sources只能包含left/right")
    source_left, view_left = _read_photo(Path(left_path).resolve(), "left", parsed.left, left_time, (timestamp_sources or {}).get("left", "MANIFEST_OPERATOR_CONFIRMED"))
    source_right, view_right = _read_photo(Path(right_path).resolve(), "right", parsed.right, right_time, (timestamp_sources or {}).get("right", "MANIFEST_OPERATOR_CONFIRMED"))
    sync_delta = _sync_delta_ms(view_left["captured_at"], view_right["captured_at"])
    if sync_delta > parsed.max_sync_delta_ms + 1e-6:
        raise ValueError(f"左右照片时间差{sync_delta:.3f}ms超过标定允许值{parsed.max_sync_delta_ms:.3f}ms")
    if _hash(source_left) == _hash(source_right):
        raise ValueError("左右照片内容相同")
    specs = _specs(pipe_specs, parsed.left.width, parsed.left.height)
    model_data: bytes | None = None
    model_record: dict[str, Any] = {"kind": "optional_reference"}
    if model_path is not None:
        model = Path(model_path).resolve()
        if not model.is_file() or model.suffix.lower() != ".stl":
            raise ValueError("model_path必须是STL文件")
        model_data = model.read_bytes()
        model_record.update({"path": f"model/{model.name}", "sha256": _hash(model_data), "unit": "millimeter"})
        if stl_unit is not None:
            model_record["source_unit"] = str(stl_unit)
    spec_hash = _hash(_canonical(specs).encode())
    run_id = f"elevation-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    manifest: dict[str, Any] = {
        "schema_version": "2.0", "dataset_id": run_id,
        "model_revision": (model_record.get("sha256", spec_hash))[:16],
        "validation_scope": "ELEVATION_DEPTH_CAPTURE_PENDING_ACCEPTANCE",
        "model": model_record, "stereo_calibration": copy.deepcopy(dict(calibration)),
        "analysis": {"mode": "elevation_depth", "elevation_depth": {"pipes": copy.deepcopy(specs)}},
        "capture": {"kind": "stereo_still_capture_set", "camera_layout": "stereo",
                     "capture_group_id": run_id, "interval_minutes": 5, "capture_groups": []},
    }
    if recipe is not None:
        manifest["rectification_recipe"] = copy.deepcopy(recipe)
    assets: dict[str, bytes] = {view_left["path"]: source_left, view_right["path"]: source_right}
    if model_data is not None:
        assets[model_record["path"]] = model_data
    if camera_capture_provenance is not None:
        if not isinstance(camera_capture_provenance, Mapping) or set(camera_capture_provenance) - {"left", "right"}:
            raise ValueError("camera_capture_provenance必须是对象")
        for role, view in (("left", view_left), ("right", view_right)):
            provenance = camera_capture_provenance.get(role, {})
            if not isinstance(provenance, Mapping) or set(provenance) - _PROVENANCE_KEYS:
                raise ValueError(f"camera_capture_provenance.{role}字段无效")
            view.update(copy.deepcopy(dict(provenance)))
    if previous_manifest is not None:
        old_path = Path(previous_manifest).resolve()
        if old_path.is_dir():
            old_path /= "manifest.json"
        loaded = load_elevation_dataset(old_path)
        old = loaded["manifest"]
        old_specs = loaded["pipe_specs"]
        old_model = old.get("model") or {}
        same_model = (
            old_model.get("sha256") == model_record.get("sha256")
            and bool(old_model.get("path")) == bool(model_record.get("path"))
            and old_model.get("source_unit") == model_record.get("source_unit")
        )
        if (old.get("stereo_calibration") != calibration or old_specs != specs or
                loaded["rectification_recipe"] != recipe or not same_model):
            raise ValueError("历史数据只能在标定、管道配置和模型完全相同时复用")
        old_records, old_assets = _history(old_path, old, run_id)
        manifest["capture"]["capture_groups"].extend(old_records)
        assets.update(old_assets)
        duplicate = bool(old_records and all(
            old_records[-1].get("views", {}).get(role, {}).get("sha256") == view["sha256"] and
            old_records[-1].get("views", {}).get(role, {}).get("captured_at") == view["captured_at"]
            for role, view in (("left", view_left), ("right", view_right))))
    else:
        duplicate = False
    if not duplicate:
        manifest["capture"]["capture_groups"].append({"capture_id": run_id, "sync_valid": True,
            "sync_delta_ms": sync_delta, "views": {"left": view_left, "right": view_right}})
    output_parent = Path(output_root).resolve()
    output_parent.mkdir(parents=True, exist_ok=True)
    target = output_parent / run_id
    if target.exists():
        raise FileExistsError(target)
    staging = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=str(output_parent)))
    try:
        for relative, data in assets.items():
            destination = staging / PurePosixPath(relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        atomic_write_text(staging / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        from .stereo_analyzer import _capture_groups_from_manifest
        _capture_groups_from_manifest(staging / "manifest.json", manifest["capture"], parsed)
        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return (target / "manifest.json").resolve()


def load_elevation_dataset(path: Path) -> dict[str, Any]:
    """Load and hash-check a package, returning the latest pair for the GUI."""
    manifest_path = Path(path).resolve()
    if manifest_path.is_dir():
        manifest_path /= "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "2.0" or (manifest.get("analysis") or {}).get("mode") != "elevation_depth":
        raise ValueError("不是schema2.0 elevation_depth数据包")
    calibration = _calibration_ok(manifest.get("stereo_calibration"))
    recipe = manifest.get("rectification_recipe")
    if recipe is not None:
        from .workbench_profile import validate_rectification_recipe
        try:
            recipe = validate_rectification_recipe(recipe, calibration=manifest["stereo_calibration"])
        except Exception as exc:
            raise ValueError(f"矫正配方不可用：{exc}") from exc
    from .stereo_analyzer import _capture_groups_from_manifest
    _capture_groups_from_manifest(manifest_path, manifest.get("capture"), calibration)
    model = manifest.get("model") or {}
    if not isinstance(model, Mapping):
        raise ValueError("model元数据无效")
    if model.get("path") is not None:
        model_rel = str(model.get("path"))
        portable = PurePosixPath(model_rel.replace("\\", "/"))
        windows = PureWindowsPath(model_rel)
        if portable.is_absolute() or windows.is_absolute() or windows.drive or ".." in portable.parts:
            raise ValueError("model.path不能逃出数据目录")
        model_file = (manifest_path.parent / portable).resolve()
        try:
            model_file.relative_to(manifest_path.parent)
        except ValueError as exc:
            raise ValueError("model.path不能逃出数据目录") from exc
        if not model_file.is_file():
            raise FileNotFoundError(model_file)
        expected_model_hash = model.get("sha256")
        if not isinstance(expected_model_hash, str) or _hash(model_file.read_bytes()) != expected_model_hash:
            raise ValueError("模型SHA-256不匹配")
    specs = ((manifest.get("analysis") or {}).get("elevation_depth") or {}).get("pipes")
    specs = _specs(specs, calibration.left.width, calibration.left.height)
    groups = (manifest.get("capture") or {}).get("capture_groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("数据包没有拍摄组")
    latest = groups[-1]
    views = latest.get("views") if isinstance(latest, Mapping) else None
    if not isinstance(views, Mapping):
        raise ValueError("最新拍摄组缺少左右照片")
    # Verify every historical byte, so recovery cannot silently use altered evidence.
    for group in groups:
        gviews = group.get("views", {})
        for role in ("left", "right"):
            load_photo_snapshot(manifest_path, gviews[role])
    left = views["left"]; right = views["right"]
    return {"manifest": manifest, "calibration": manifest["stereo_calibration"], "pipe_specs": specs,
            "left_path": resolve_photo_path(manifest_path, left["path"]),
            "right_path": resolve_photo_path(manifest_path, right["path"]),
            "left_time": left["captured_at"], "right_time": right["captured_at"],
            "rectification_recipe": recipe}


__all__ = ["create_elevation_dataset", "load_elevation_dataset", "elevation_history_compatible"]
