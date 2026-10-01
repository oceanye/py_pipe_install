"""Safe registration-free stereo elevation screening.

The elevation mode is deliberately a conservative ROI classifier.  It does not
project CAD or use a QR pose; each continuous pipe is represented by one ROI
per rectified eye and evidence is pooled over its length.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


class ElevationDepthError(ValueError):
    """Invalid or unsafe elevation-depth input."""


# These gates are safety invariants.  Manifest settings may only tighten them.
_HARD_MIN_COLOR = 0.10
_HARD_MIN_DEPTH = 0.25
_HARD_MIN_JOINT = 0.08
_HARD_MAX_WIDTH_ERROR = 0.60
_HARD_MIN_ELONGATION = 1.10


def _number(value: Any, field: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ElevationDepthError(f"{field} must be a finite number")
    value = float(value)
    if positive and value <= 0:
        raise ElevationDepthError(f"{field} must be positive")
    return value


def _hex_bgr(value: Any) -> np.ndarray:
    if not isinstance(value, str) or len(value) != 7 or value[0] != "#":
        raise ElevationDepthError("pipe color must be #RRGGBB")
    try:
        rgb = [int(value[i : i + 2], 16) for i in (1, 3, 5)]
    except ValueError as error:
        raise ElevationDepthError("pipe color must be #RRGGBB") from error
    return np.asarray([[rgb[::-1]]], dtype=np.uint8)


def _region(value: Any, label: str, width: int, height: int) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ElevationDepthError(f"{label} must be [x,y,width,height]")
    if any(type(item) is not int and not isinstance(item, np.integer) for item in value):
        raise ElevationDepthError(f"{label} must contain integers")
    try:
        x, y, w, h = (int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ElevationDepthError(f"{label} must contain integers") from error
    if w < 8 or h < 8 or x < 0 or y < 0 or x + w > width or y + h > height:
        raise ElevationDepthError(f"{label} is outside the image")
    return x, y, w, h


def validate_elevation_specs(payload: Any, width: int, height: int) -> list[dict[str, Any]]:
    """Validate and normalize the public per-pipe ROI contract."""
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        raise ElevationDepthError("image dimensions must be positive integers")
    if not isinstance(payload, list) or not payload:
        raise ElevationDepthError("elevation_depth.pipes must be non-empty")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        field = f"elevation_depth.pipes[{index}]"
        if not isinstance(item, Mapping):
            raise ElevationDepthError(f"{field} must be an object")
        pipe_id = item.get("pipe_id")
        if not isinstance(pipe_id, str) or not pipe_id.strip() or pipe_id in seen:
            raise ElevationDepthError("elevation_depth pipe_id values must be unique text")
        color = item.get("color_srgb")
        _hex_bgr(color)
        diameter = _number(item.get("nominal_diameter_mm"), f"{field}.nominal_diameter_mm", positive=True)
        left = _region(item.get("left_region_px"), f"{pipe_id}.left_region_px", width, height)
        # Explicit right ROI is required: silently copying the left ROI hides camera mapping errors.
        if "right_region_px" not in item:
            raise ElevationDepthError(f"{pipe_id}.right_region_px is required")
        right = _region(item.get("right_region_px"), f"{pipe_id}.right_region_px", width, height)
        expected = item.get("expected_depth_mm")
        if expected is not None:
            expected = _number(expected, f"{field}.expected_depth_mm", positive=True)
        axis = item.get("axis", "auto")
        if axis not in {"auto", "horizontal", "vertical"}:
            raise ElevationDepthError(f"{field}.axis must be auto, horizontal, or vertical")
        seen.add(pipe_id)
        result.append({"pipe_id": pipe_id, "color_srgb": str(color).upper(), "diameter_mm": diameter,
                       "left": left, "right": right, "expected_depth_mm": expected, "axis": axis})
    return result


def validate_elevation_config(settings: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Normalize configurable thresholds while preserving non-disableable gates."""
    if settings is not None and not isinstance(settings, Mapping):
        raise ElevationDepthError("elevation_depth settings must be an object")
    raw = dict(settings or {})
    defaults = {"color_delta_lab": 28.0, "minimum_color_support_fraction": _HARD_MIN_COLOR,
                "minimum_valid_depth_fraction": _HARD_MIN_DEPTH, "minimum_joint_support_fraction": _HARD_MIN_JOINT,
                "maximum_width_relative_error": _HARD_MAX_WIDTH_ERROR, "minimum_negative_captures": 2,
                "min_negative_interval_seconds": 1.0, "depth_tolerance_mm": 50.0,
                "free_space_margin_mm": 40.0, "left_right_depth_tolerance_mm": 80.0,
                "left_right_position_tolerance_fraction": 0.35}
    cfg = {**defaults, **{k: raw[k] for k in defaults if k in raw}}
    for key in ("color_delta_lab", "minimum_color_support_fraction", "minimum_valid_depth_fraction",
                "minimum_joint_support_fraction", "maximum_width_relative_error", "min_negative_interval_seconds",
                "depth_tolerance_mm", "free_space_margin_mm", "left_right_depth_tolerance_mm",
                "left_right_position_tolerance_fraction"):
        cfg[key] = _number(cfg[key], f"elevation_depth.{key}")
    if cfg["color_delta_lab"] <= 0 or cfg["min_negative_interval_seconds"] < 1.0:
        raise ElevationDepthError("color tolerance must be positive and negative interval must be at least 1 second")
    cfg["minimum_color_support_fraction"] = max(_HARD_MIN_COLOR, min(1.0, cfg["minimum_color_support_fraction"]))
    cfg["minimum_valid_depth_fraction"] = max(_HARD_MIN_DEPTH, min(1.0, cfg["minimum_valid_depth_fraction"]))
    cfg["minimum_joint_support_fraction"] = max(_HARD_MIN_JOINT, min(1.0, cfg["minimum_joint_support_fraction"]))
    cfg["maximum_width_relative_error"] = min(_HARD_MAX_WIDTH_ERROR, max(0.0, cfg["maximum_width_relative_error"]))
    negative_count = cfg["minimum_negative_captures"]
    if isinstance(negative_count, bool) or not isinstance(negative_count, (int, float)) or not np.isfinite(negative_count) or int(negative_count) != negative_count:
        raise ElevationDepthError("minimum_negative_captures must be an integer")
    cfg["minimum_negative_captures"] = int(negative_count)
    if cfg["minimum_negative_captures"] < 2:
        raise ElevationDepthError("minimum_negative_captures must be at least 2")
    return cfg


def _array_signature(left: np.ndarray, right: np.ndarray, depth: Any) -> str:
    digest = hashlib.sha256()
    for array in (left, right):
        arr = np.ascontiguousarray(array)
        digest.update(str(arr.dtype).encode()); digest.update(repr(arr.shape).encode()); digest.update(arr.tobytes())
    return digest.hexdigest()


def _validate_calibration(calibration: Any, left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    if calibration is None or not bool(getattr(calibration, "validated", False)) or not bool(getattr(calibration, "rectified", False)):
        raise ElevationDepthError("elevation depth requires validated stereo calibration")
    baseline = _number(getattr(calibration, "baseline_mm", None), "calibration.baseline_mm", positive=True)
    cameras = []
    for role, image in (("left", left), ("right", right)):
        camera = getattr(calibration, role, None)
        fx = _number(getattr(camera, "fx", None), f"calibration.{role}.fx", positive=True)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ElevationDepthError("elevation images must be BGR images")
        declared = (getattr(camera, "width", None), getattr(camera, "height", None))
        if all(isinstance(v, int) and v > 0 for v in declared) and declared != (image.shape[1], image.shape[0]):
            raise ElevationDepthError(f"{role} image dimensions do not match calibration")
        cameras.append({"role": role, "camera_id": getattr(camera, "camera_id", None), "fx_px": fx,
                        "image_size": [int(image.shape[1]), int(image.shape[0])]})
    return {"validated": True, "baseline_mm": baseline, "rectified": bool(getattr(calibration, "rectified", False)), "cameras": cameras}


def _depth_arrays(depth: Any, shape: tuple[int, int]) -> None:
    if depth is None:
        raise ElevationDepthError("elevation mode requires stereo depth")
    for name in ("left_depth_mm", "right_depth_mm", "left_valid", "right_valid"):
        array = getattr(depth, name, None)
        if not isinstance(array, np.ndarray) or array.shape != shape:
            raise ElevationDepthError(f"depth.{name} must match image dimensions")
        if name.endswith("_valid") and array.dtype != np.bool_:
            raise ElevationDepthError(f"depth.{name} must be boolean")
        if name.endswith("_depth_mm") and not np.issubdtype(array.dtype, np.number):
            raise ElevationDepthError(f"depth.{name} must be numeric")
    if not np.any(depth.left_valid) or not np.any(depth.right_valid):
        return


def _region_evidence(image: np.ndarray, depth_values: np.ndarray, valid: np.ndarray,
                     spec: Mapping[str, Any], role: str, config: Mapping[str, Any], fx: float) -> dict[str, Any]:
    x, y, w, h = spec[role]
    crop = image[y:y+h, x:x+w]
    from .stereo_analyzer import _image_signal_diagnostics
    signal = _image_signal_diagnostics(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(_hex_bgr(spec["color_srgb"]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    color_mask = np.linalg.norm(lab - target, axis=2) <= float(config["color_delta_lab"])
    valid_crop = np.asarray(valid[y:y+h, x:x+w], dtype=bool)
    depth_crop = np.asarray(depth_values[y:y+h, x:x+w], dtype=np.float64)
    finite = np.isfinite(depth_crop) & (depth_crop > 0)
    valid_crop &= finite
    joint = color_mask & valid_crop
    color_fraction = float(np.mean(color_mask))
    valid_fraction = float(np.mean(valid_crop))
    joint_fraction = float(np.mean(joint))
    median_depth = float(np.median(depth_crop[joint])) if np.any(joint) else None
    median_valid_depth = float(np.median(depth_crop[valid_crop])) if np.any(valid_crop) else None
    foreground_fraction = 0.0
    foreground = False
    free_space_fraction = 0.0
    expected = spec.get("expected_depth_mm")
    if expected is not None and np.any(valid_crop):
        foreground_fraction = float(np.mean(depth_crop[valid_crop] < float(expected) - float(config["depth_tolerance_mm"])))
        free_space_fraction = float(np.count_nonzero(valid_crop & (depth_crop >= float(expected) + float(config["free_space_margin_mm"]))) / valid_crop.size)
        foreground = foreground_fraction > 0.05
    components = cv2.connectedComponentsWithStats(joint.astype(np.uint8), 8)
    component = None
    if components[0] > 1:
        index = 1 + int(np.argmax(components[2][1:, cv2.CC_STAT_AREA]))
        area = int(components[2][index, cv2.CC_STAT_AREA])
        cx = float(components[3][index, 0]); cy = float(components[3][index, 1])
        cw = int(components[2][index, cv2.CC_STAT_WIDTH]); ch = int(components[2][index, cv2.CC_STAT_HEIGHT])
        major, minor = max(cw, ch), max(1, min(cw, ch))
        axis_ok = spec["axis"] == "auto" or (spec["axis"] == "horizontal" and cw >= ch) or (spec["axis"] == "vertical" and ch >= cw)
        component = {"area_px": area, "centroid_px": [cx, cy], "bbox_wh_px": [cw, ch],
                     "elongation": float(major / minor), "axis_ok": bool(axis_ok)}
        component_mask = components[1] == index
        median_depth = float(np.median(depth_crop[component_mask]))
        joint_fraction = float(area / joint.size)
    observed_width = float(min(component["bbox_wh_px"])) if component else None
    predicted_width = float(fx * float(spec["diameter_mm"]) / median_depth) if median_depth and median_depth > 0 else None
    width_error = abs(observed_width - predicted_width) / predicted_width if observed_width and predicted_width else None
    gates = {"color": color_fraction >= config["minimum_color_support_fraction"],
             "depth": valid_fraction >= config["minimum_valid_depth_fraction"],
             "joint": joint_fraction >= config["minimum_joint_support_fraction"],
             "component": bool(component and component["axis_ok"] and
                                component["elongation"] >= _HARD_MIN_ELONGATION),
             "width": bool(width_error is not None and width_error <= config["maximum_width_relative_error"])}
    return {"region_xywh": list(spec[role]), "image_signal": signal, "color_support_fraction": color_fraction,
            "valid_depth_fraction": valid_fraction, "joint_support_fraction": joint_fraction,
            "median_depth_mm": median_depth, "median_valid_depth_mm": median_valid_depth,
            "target_depth_median_mm": median_depth,
            "foreground_occlusion": foreground, "foreground_fraction": foreground_fraction,
            "free_space_fraction": free_space_fraction, "component": component,
            "observed_width_px": observed_width, "predicted_width_px": predicted_width,
            "width_relative_error": width_error, "gates": gates,
            "color_gate_passed": gates["color"], "depth_gate_passed": gates["depth"],
            "joint_gate_passed": gates["joint"], "width_gate_passed": gates["width"]}


def _one_capture(group: Mapping[str, Any], specs: list[dict[str, Any]], config: Mapping[str, Any], calibration_audit: Mapping[str, Any]) -> dict[str, Any]:
    left, right, depth = group["left"], group["right"], group["depth"]
    if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray) or left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3 or left.dtype != np.uint8 or right.dtype != np.uint8:
        raise ElevationDepthError("elevation images must be matching BGR images")
    for image, camera in ((left, calibration_audit["cameras"][0]), (right, calibration_audit["cameras"][1])):
        if camera["image_size"] != [int(image.shape[1]), int(image.shape[0])]:
            raise ElevationDepthError("capture image dimensions changed during elevation analysis")
    _depth_arrays(depth, left.shape[:2])
    signature = _array_signature(left, right, depth)
    pair_healthy = bool(group.get("pair_healthy", False))
    result = {"capture_id": str(group.get("capture_id", "")), "captured_at": group.get("captured_at"),
              "pair_signature": signature, "pair_healthy": pair_healthy, "pipes": {}}
    for spec in specs:
        lev = _region_evidence(left, depth.left_depth_mm, depth.left_valid, spec, "left", config, calibration_audit["cameras"][0]["fx_px"])
        rev = _region_evidence(right, depth.right_depth_mm, depth.right_valid, spec, "right", config, calibration_audit["cameras"][1]["fx_px"])
        if lev["component"]:
            lev["global_centroid_px"] = [spec["left"][0] + lev["component"]["centroid_px"][0], spec["left"][1] + lev["component"]["centroid_px"][1]]
        if rev["component"]:
            rev["global_centroid_px"] = [spec["right"][0] + rev["component"]["centroid_px"][0], spec["right"][1] + rev["component"]["centroid_px"][1]]
        observed_disparity = lev.get("global_centroid_px", [None])[0] - rev.get("global_centroid_px", [None])[0] if lev.get("global_centroid_px") and rev.get("global_centroid_px") else None
        depth_consistent = bool(lev["median_depth_mm"] is not None and rev["median_depth_mm"] is not None and
            abs(lev["median_depth_mm"] - rev["median_depth_mm"]) <= config["left_right_depth_tolerance_mm"])
        depth_for_disparity = float(np.mean([lev["median_depth_mm"], rev["median_depth_mm"]])) if depth_consistent else None
        expected_disparity = calibration_audit["cameras"][0]["fx_px"] * calibration_audit["baseline_mm"] / depth_for_disparity if depth_for_disparity else None
        observed_width = max(lev.get("observed_width_px") or 0.0, rev.get("observed_width_px") or 0.0)
        disparity_tolerance = max(2.0, min(max(spec["left"][2], spec["right"][2]) / 2.0, observed_width * 0.5 + 2.0))
        epipolar_tolerance = max(2.0, min(5.0, observed_width * 0.15 + 2.0))
        same_position = bool(observed_disparity is not None and expected_disparity is not None and abs(observed_disparity - expected_disparity) <= disparity_tolerance and abs(lev["global_centroid_px"][1] - rev["global_centroid_px"][1]) <= epipolar_tolerance)
        target_depth_match = bool(spec.get("expected_depth_mm") is None or (depth_for_disparity is not None and abs(depth_for_disparity - spec["expected_depth_mm"]) <= config["depth_tolerance_mm"]))
        installed = bool(pair_healthy and same_position and depth_consistent and not lev["foreground_occlusion"] and not rev["foreground_occlusion"] and
            target_depth_match and
            all(lev["gates"][k] and rev["gates"][k] for k in ("color", "depth", "joint", "component", "width")))
        reference = spec.get("expected_depth_mm")
        free = bool(pair_healthy and reference is not None and lev["gates"]["depth"] and rev["gates"]["depth"] and
            not lev["foreground_occlusion"] and not rev["foreground_occlusion"] and
            lev["free_space_fraction"] >= 0.65 and rev["free_space_fraction"] >= 0.65 and
            not lev["gates"]["color"] and not rev["gates"]["color"])
        reasons = []
        if installed: evidence = "ELEVATION_STEREO_COLOR_DEPTH"; reasons = ["COLOR_DEPTH_SAME_COMPONENT", "LEFT_RIGHT_CONSISTENT", "DIAMETER_WIDTH_MATCH"]
        elif free: evidence = "ELEVATION_REPEATED_FREE_SPACE_CANDIDATE"; reasons = ["REFERENCE_DEPTH_KNOWN", "VALID_FREE_SPACE", "NO_FOREGROUND_COLOR"]
        else:
            evidence = "INCONCLUSIVE"
            reasons = ["PAIR_UNHEALTHY" if not pair_healthy else "INSUFFICIENT_COLOR_DEPTH_GEOMETRY"]
            for warning in ("LOW_LIGHT", "DARK_REGION_DOMINANT", "LOW_TEXTURE"):
                if any(warning in ev["image_signal"]["warning_codes"] for ev in (lev, rev)):
                    reasons.append(warning)
        result["pipes"][spec["pipe_id"]] = {"left": lev, "right": rev, "installed_candidate": installed,
            "free_space_candidate": free, "evidence": evidence, "reason_codes": reasons,
            "left_right_position_consistent": same_position, "left_right_depth_consistent": depth_consistent}
    return result


def analyze_elevation_groups(groups: list[Mapping[str, Any]], *, calibration: Any, pipe_specs: list[Mapping[str, Any]], config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(groups, list) or not groups:
        raise ElevationDepthError("elevation depth requires at least one capture group")
    first_left, first_right = groups[0].get("left"), groups[0].get("right")
    if not isinstance(first_left, np.ndarray) or not isinstance(first_right, np.ndarray) or first_left.shape != first_right.shape:
        raise ElevationDepthError("elevation images must be matching arrays")
    calibration_audit = _validate_calibration(calibration, first_left, first_right)
    cfg = validate_elevation_config(config)
    specs = validate_elevation_specs(pipe_specs, first_left.shape[1], first_left.shape[0])
    ambiguous_ids: set[str] = set()
    for index, first in enumerate(specs):
        for second in specs[index + 1:]:
            if first["color_srgb"] != second["color_srgb"]:
                continue
            for eye in ("left", "right"):
                ax, ay, aw, ah = first[eye]; bx, by, bw, bh = second[eye]
                overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
                if overlap / max(1, min(aw * ah, bw * bh)) >= 0.30:
                    ambiguous_ids.update((first["pipe_id"], second["pipe_id"]))
    captures = [_one_capture(group, specs, cfg, calibration_audit) for group in groups]
    results = []
    for spec in specs:
        rows = [capture["pipes"][spec["pipe_id"]] for capture in captures]
        current = rows[-1]
        negatives = []
        # A stale/failed current pair invalidates negative inference even if
        # older captures were healthy; otherwise a camera outage can be
        # mistaken for an empty pipe location.
        if not bool(captures[-1]["pair_healthy"]) or not bool(current["free_space_candidate"]):
            rows_for_negative = []
        else:
            rows_for_negative = list(zip(reversed(captures), reversed(rows)))
        prior_time = None
        signatures: set[str] = set()
        for capture, row in rows_for_negative:
            if not row["free_space_candidate"]:
                # Negative evidence must be a contiguous healthy sequence;
                # do not reach through an occluded/installed capture.
                break
            if capture["pair_signature"] in signatures:
                continue
            when = capture.get("captured_at")
            if isinstance(when, str):
                try: timestamp = datetime.fromisoformat(when.replace("Z", "+00:00")).timestamp()
                except ValueError: timestamp = None
            else: timestamp = None
            # Without capture timestamps the temporal independence gate cannot
            # be demonstrated; capture_id alone is deliberately insufficient.
            if timestamp is None:
                continue
            if prior_time is not None and prior_time - timestamp < cfg["min_negative_interval_seconds"]:
                continue
            signatures.add(capture["pair_signature"]); negatives.append(capture["capture_id"]); prior_time = timestamp if timestamp is not None else prior_time
        if spec["pipe_id"] in ambiguous_ids:
            state, basis = "UNKNOWN", "ELEVATION_IDENTITY_AMBIGUOUS"
        elif current["installed_candidate"]:
            state, basis = "INSTALLED", "ELEVATION_STEREO_COLOR_DEPTH"
        elif spec.get("expected_depth_mm") is None:
            state, basis = "UNKNOWN", "REFERENCE_DEPTH_REQUIRED_FOR_NEGATIVE"
        elif len(negatives) >= cfg["minimum_negative_captures"]:
            state, basis = "NOT_INSTALLED", "ELEVATION_REPEATED_FREE_SPACE"
        else:
            state, basis = "UNKNOWN", "ELEVATION_INSUFFICIENT_OR_OCCLUDED_EVIDENCE"
        results.append({"pipe_id": spec["pipe_id"], "installation_state": state,
            "installation_state_zh": {"INSTALLED": "安装", "NOT_INSTALLED": "未安装", "UNKNOWN": "遮蔽不确定"}[state],
            "state_basis": basis, "identity_ambiguous": spec["pipe_id"] in ambiguous_ids,
            "reason_codes": (["ELEVATION_IDENTITY_AMBIGUOUS"] if spec["pipe_id"] in ambiguous_ids else current["reason_codes"]), "current_evidence": current,
            "independent_free_space_captures": list(reversed(negatives))})
    counts = {state: sum(row["installation_state"] == state for row in results) for state in ("INSTALLED", "NOT_INSTALLED", "UNKNOWN")}
    return {"mode": "elevation_depth", "registration_required": False, "production_authority": False,
            "counts": counts, "calibration_audit": calibration_audit, "capture_audit": {"count": len(captures), "groups": captures}, "pipes": results}


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream: stream.write(content); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists(): temporary.unlink()


def analyze_elevation_depth_manifest(manifest_path: str | Path, *, report_output_path: str | Path | None = None, evidence_dir: str | Path | None = None) -> dict[str, Any]:
    """Run elevation mode with the same snapshot, quality and capture checks as the CAD router."""
    from .stereo_analyzer import _analysis_config, _capture_groups_from_manifest, _calibration_from_manifest, _compute_stereo_depth, _rectify, _quality
    from .photo_capture import load_photo_snapshot
    from .pipeline import ensure_paths_distinct
    path = Path(manifest_path).resolve(); manifest = json.loads(path.read_text(encoding="utf-8"))
    calibration = _calibration_from_manifest(manifest.get("stereo_calibration"))
    if not calibration.validated: raise ElevationDepthError("elevation depth requires validated stereo calibration")
    _run_id, _interval, groups = _capture_groups_from_manifest(path, manifest.get("capture"), calibration)
    analysis = manifest.get("analysis")
    if not isinstance(analysis, Mapping) or analysis.get("mode") != "elevation_depth": raise ElevationDepthError("analysis.mode must be 'elevation_depth'")
    settings = dict(analysis.get("elevation_depth") or {})
    specs = settings.pop("pipes", None) or manifest.get("model", {}).get("pipes")
    if not specs: raise ElevationDepthError("elevation_depth.pipes is required")
    cfg = validate_elevation_config(settings); config = _analysis_config({**analysis, "stereo_matching": analysis.get("stereo_matching", {})})
    groups_for_engine = []; input_hashes = {"manifest": hashlib.sha256(path.read_bytes()).hexdigest(), "photos": {}}
    for group in groups:
        images = {}; integrities = {}; quality = {}; hashes = []
        for role in ("left", "right"):
            image, integrity = load_photo_snapshot(path, group["views"][role]); image = _rectify(image, getattr(calibration, role), calibration.rectified)
            images[role] = image; integrities[role] = integrity; hashes.append(integrity["actual_sha256"]); input_hashes["photos"][group["capture_id"] + "." + role] = integrity["actual_sha256"]; quality[role] = _quality(image, config)
        depth = _compute_stereo_depth(images["left"], images["right"], calibration, config)
        healthy = bool(group["sync_valid"] and all(quality[r]["passed"] for r in ("left", "right")) and depth.audit.get("status") == "VALID")
        groups_for_engine.append({"left": images["left"], "right": images["right"], "depth": depth, "pair_healthy": healthy, "capture_id": group["capture_id"], "captured_at": group["captured_at"], "sync_valid": group["sync_valid"], "quality": quality, "integrities": integrities})
    result = analyze_elevation_groups(groups_for_engine, calibration=calibration, pipe_specs=specs, config=cfg)
    engine_groups = result["capture_audit"]["groups"]
    result.update({"schema_version": "2.0", "report_type": "stereo-elevation-depth-pipe-installation-state", "generated_at": datetime.now(timezone.utc).isoformat(), "dataset_id": manifest.get("dataset_id"), "model_revision": manifest.get("model_revision"), "field_installation_state_inferred": True, "inputs": input_hashes, "capture_audit": {"groups": [{"capture_id": g["capture_id"], "captured_at": g["captured_at"], "sync_valid": g["sync_valid"], "quality": g.get("quality"), "photos": g.get("integrities"), "pair_healthy": g["pair_healthy"]} for g in groups_for_engine], "count": len(groups_for_engine)}, "analysis_config": cfg})
    for public, evidence in zip(result["capture_audit"]["groups"], engine_groups):
        public["pipes"] = evidence["pipes"]
        public["pair_signature"] = evidence["pair_signature"]
    for public, group in zip(result["capture_audit"]["groups"], groups_for_engine):
        public["depth_audit"] = group["depth"].audit
    result["stereo_analysis_config"] = {k: v for k, v in config.items()
                                        if k not in {"mode", "elevation_depth", "elevation_auto"}}
    output = Path(report_output_path).resolve() if report_output_path is not None else None
    input_paths = [path]
    model = manifest.get("model")
    if isinstance(model, Mapping) and model.get("path"):
        model_path = Path(model["path"])
        input_paths.append((model_path if model_path.is_absolute() else path.parent / model_path).resolve())
    for group in groups:
        for view in group["views"].values():
            view_path = Path(view["path"])
            input_paths.append((view_path if view_path.is_absolute() else path.parent / view_path).resolve())
    if output is not None:
        try:
            ensure_paths_distinct(report=output, **{f"input_{i}": p for i, p in enumerate(input_paths)})
        except ValueError as error:
            raise ElevationDepthError(str(error)) from error
    if evidence_dir is not None:
        root = Path(evidence_dir).resolve(); root.mkdir(parents=True, exist_ok=True)
        evidence_files = []
        overlay_specs = validate_elevation_specs(specs, groups_for_engine[0]["left"].shape[1], groups_for_engine[0]["left"].shape[0])
        for group in groups_for_engine:
            for role in ("left", "right"):
                target = (root / f"{group['capture_id']}_{role}_overlay.png").resolve()
                if not target.is_relative_to(root):
                    raise ElevationDepthError("capture_id must not escape the evidence directory")
                try:
                    protected = input_paths + ([output] if output is not None else [])
                    ensure_paths_distinct(output=target, **{f"input_{i}": p for i, p in enumerate(protected)})
                except ValueError as error:
                    raise ElevationDepthError(str(error)) from error
                overlay = group[role].copy()
                for spec in overlay_specs:
                    x, y, w, h = spec["left" if role == "left" else "right"]
                    cv2.rectangle(overlay, (x, y), (x + w - 1, y + h - 1), (0, 200, 255), 2)
                    cv2.putText(overlay, str(spec["pipe_id"]), (x, max(15, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
                ok, encoded = cv2.imencode(".png", overlay)
                if not ok: raise ElevationDepthError(f"failed to encode evidence overlay: {target}")
                _atomic_write(target, encoded.tobytes()); evidence_files.append(str(target))
        result["evidence_files"] = evidence_files
    if hashlib.sha256(path.read_bytes()).hexdigest() != input_hashes["manifest"]:
        raise ElevationDepthError("manifest changed during elevation analysis")
    if output is not None:
        _atomic_write(output, (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    return result


__all__ = ["ElevationDepthError", "validate_elevation_specs", "validate_elevation_config", "analyze_elevation_depth_manifest", "analyze_elevation_groups"]
