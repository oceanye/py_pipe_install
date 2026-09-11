"""Registration-free stereo elevation screening for continuous pipe runs.

This mode deliberately does not project a CAD mesh into the camera.  A field
manifest supplies one expected 2-D region per pipe in each rectified eye;
colour, local width and stereo depth are then pooled over that region.  It is
intended for a fixed elevation where a pipe has one state along its length.
The full CAD/QR analyser remains the authoritative mode for 3-D registration.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


class ElevationDepthError(ValueError):
    """Invalid or unsafe elevation-depth input."""


def _hex_bgr(value: Any) -> np.ndarray:
    if not isinstance(value, str) or len(value) != 7 or value[0] != "#":
        raise ElevationDepthError("pipe color must be #RRGGBB")
    try:
        rgb = [int(value[index : index + 2], 16) for index in (1, 3, 5)]
    except ValueError as error:
        raise ElevationDepthError("pipe color must be #RRGGBB") from error
    return np.asarray([[rgb[::-1]]], dtype=np.uint8)


def _region(value: Any, label: str, width: int, height: int) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ElevationDepthError(f"{label} must be [x,y,width,height]")
    try:
        x, y, w, h = (int(item) for item in value)
    except (TypeError, ValueError) as error:
        raise ElevationDepthError(f"{label} must contain integers") from error
    if w < 8 or h < 8 or x < 0 or y < 0 or x + w > width or y + h > height:
        raise ElevationDepthError(f"{label} is outside the image")
    return x, y, w, h


def _pipe_specs(payload: Any, width: int, height: int) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or not payload:
        raise ElevationDepthError("elevation_depth.pipes must be non-empty")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ElevationDepthError(f"elevation_depth.pipes[{index}] must be an object")
        pipe_id = item.get("pipe_id")
        if not isinstance(pipe_id, str) or not pipe_id or pipe_id in seen:
            raise ElevationDepthError("elevation_depth pipe_id values must be unique text")
        color = str(item.get("color_srgb", ""))
        diameter = item.get("nominal_diameter_mm")
        if not isinstance(diameter, (int, float)) or not np.isfinite(diameter) or diameter <= 0:
            raise ElevationDepthError(f"elevation_depth.pipes[{index}].nominal_diameter_mm is invalid")
        left = _region(item.get("left_region_px"), f"{pipe_id}.left_region_px", width, height)
        right = _region(item.get("right_region_px", left), f"{pipe_id}.right_region_px", width, height)
        _hex_bgr(color)
        seen.add(pipe_id)
        result.append({"pipe_id": pipe_id, "color_srgb": color.upper(), "diameter_mm": float(diameter), "left": left, "right": right})
    return result


def _color_support(image: np.ndarray, region: tuple[int, int, int, int], color: str, tolerance: float) -> tuple[float, float | None]:
    x, y, w, h = region
    crop = image[y : y + h, x : x + w]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(_hex_bgr(color), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    distance = np.linalg.norm(lab - target, axis=2)
    mask = distance <= float(tolerance)
    if not np.any(mask):
        return 0.0, None
    # A robust section width is useful as a diagnostic, but never creates a
    # positive state on its own.
    columns = np.flatnonzero(np.count_nonzero(mask, axis=0) >= 2)
    observed_width = float(columns[-1] - columns[0] + 1) if len(columns) >= 2 else None
    return float(np.count_nonzero(mask) / mask.size), observed_width


def _region_evidence(image: np.ndarray, depth: np.ndarray, valid: np.ndarray, spec: Mapping[str, Any], role: str, config: Mapping[str, Any]) -> dict[str, Any]:
    region = spec[role]
    x, y, w, h = region
    support, observed_width = _color_support(image, region, str(spec["color_srgb"]), float(config["color_delta_lab"]))
    valid_region = valid[y : y + h, x : x + w]
    depth_region = depth[y : y + h, x : x + w]
    valid_fraction = float(np.count_nonzero(valid_region) / valid_region.size)
    values = depth_region[valid_region]
    median_depth = float(np.median(values)) if len(values) else None
    color_passed = support >= float(config["minimum_color_support_fraction"])
    depth_passed = valid_fraction >= float(config["minimum_valid_depth_fraction"])
    return {
        "region_xywh": list(region),
        "color_support_fraction": support,
        "color_gate_passed": color_passed,
        "observed_width_px": observed_width,
        "valid_depth_fraction": valid_fraction,
        "median_depth_mm": median_depth,
        "depth_gate_passed": depth_passed,
    }


def _one_capture(left: np.ndarray, right: np.ndarray, depth: Any, specs: list[dict[str, Any]], config: Mapping[str, Any], *, pair_healthy: bool, capture_id: str, signature: str) -> dict[str, Any]:
    if left.shape != right.shape or left.ndim != 3 or left.shape[2] != 3:
        raise ElevationDepthError("elevation images must be matching BGR images")
    if depth is None or not hasattr(depth, "left_depth_mm"):
        raise ElevationDepthError("elevation mode requires stereo depth")
    result: dict[str, Any] = {"capture_id": capture_id, "pair_signature": signature, "pair_healthy": bool(pair_healthy), "pipes": {}}
    for spec in specs:
        left_ev = _region_evidence(left, depth.left_depth_mm, depth.left_valid, spec, "left", config)
        right_ev = _region_evidence(right, depth.right_depth_mm, depth.right_valid, spec, "right", config)
        installed = bool(pair_healthy and left_ev["color_gate_passed"] and right_ev["color_gate_passed"] and left_ev["depth_gate_passed"] and right_ev["depth_gate_passed"])
        free = bool(pair_healthy and left_ev["depth_gate_passed"] and right_ev["depth_gate_passed"] and not left_ev["color_gate_passed"] and not right_ev["color_gate_passed"])
        if installed:
            evidence, reasons = "ELEVATION_STEREO_COLOR_DEPTH", ["COLOR_MATCH", "VALID_STEREO_DEPTH"]
        elif free:
            evidence, reasons = "ELEVATION_FREE_SPACE_CANDIDATE", ["COLOR_ABSENT", "VALID_STEREO_DEPTH"]
        else:
            evidence, reasons = "INCONCLUSIVE", ["INSUFFICIENT_COLOR_OR_DEPTH"]
        result["pipes"][spec["pipe_id"]] = {"left": left_ev, "right": right_ev, "installed_candidate": installed, "free_space_candidate": free, "evidence": evidence, "reason_codes": reasons}
    return result


def analyze_elevation_groups(groups: list[Mapping[str, Any]], *, calibration: Any, pipe_specs: list[Mapping[str, Any]], config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Classify continuous pipes from already rectified stereo group arrays.

    ``groups`` contains ``left``, ``right``, ``depth``, ``pair_healthy`` and a
    stable ``capture_id``/``pair_signature``. It is intentionally array based
    so tests and camera adapters can use the same engine.
    """
    cfg = {"color_delta_lab": 28.0, "minimum_color_support_fraction": 0.10, "minimum_valid_depth_fraction": 0.25, "minimum_negative_captures": 2}
    cfg.update(dict(config or {}))
    if not pipe_specs:
        raise ElevationDepthError("elevation_depth requires at least one pipe")
    height, width = groups[0]["left"].shape[:2]
    specs = _pipe_specs(pipe_specs, width, height)
    ambiguous_ids: set[str] = set()
    for index, first in enumerate(specs):
        fx, fy, fw, fh = first["left"]
        for second in specs[index + 1 :]:
            if first["color_srgb"] != second["color_srgb"]:
                continue
            sx, sy, sw, sh = second["left"]
            overlap = max(0, min(fx + fw, sx + sw) - max(fx, sx)) * max(0, min(fy + fh, sy + sh) - max(fy, sy))
            if overlap / max(1, min(fw * fh, sw * sh)) >= 0.30:
                ambiguous_ids.update((first["pipe_id"], second["pipe_id"]))
    captures = []
    for group in groups:
        captures.append(_one_capture(group["left"], group["right"], group["depth"], specs, cfg, pair_healthy=bool(group.get("pair_healthy", True)), capture_id=str(group["capture_id"]), signature=str(group.get("pair_signature", group["capture_id"]))))
    results = []
    for spec in specs:
        rows = [capture["pipes"][spec["pipe_id"]] for capture in captures]
        current = rows[-1]
        positive = bool(current["installed_candidate"])
        unique_free = []
        signatures: set[str] = set()
        for capture, row in zip(reversed(captures), reversed(rows)):
            if not row["free_space_candidate"] or capture["pair_signature"] in signatures:
                continue
            signatures.add(capture["pair_signature"])
            unique_free.append(capture["capture_id"])
        if spec["pipe_id"] in ambiguous_ids:
            state, basis = "UNKNOWN", "ELEVATION_IDENTITY_AMBIGUOUS"
        elif positive:
            state, basis = "INSTALLED", "ELEVATION_STEREO_COLOR_DEPTH"
        elif len(unique_free) >= int(cfg["minimum_negative_captures"]):
            state, basis = "NOT_INSTALLED", "ELEVATION_REPEATED_FREE_SPACE"
        else:
            state, basis = "UNKNOWN", "ELEVATION_INSUFFICIENT_OR_OCCLUDED_EVIDENCE"
        results.append({"pipe_id": spec["pipe_id"], "installation_state": state, "installation_state_zh": {"INSTALLED": "安装", "NOT_INSTALLED": "未安装", "UNKNOWN": "遮蔽不确定"}[state], "state_basis": basis, "identity_ambiguous": spec["pipe_id"] in ambiguous_ids, "current_evidence": current, "independent_free_space_captures": list(reversed(unique_free))})
    counts = {state: sum(row["installation_state"] == state for row in results) for state in ("INSTALLED", "NOT_INSTALLED", "UNKNOWN")}
    return {"mode": "elevation_depth", "registration_required": False, "assignment_method": "color_diameter_depth_component", "production_authority": False, "counts": counts, "captures": captures, "pipes": results}


def analyze_elevation_depth_manifest(manifest_path: str | Path, *, report_output_path: str | Path | None = None) -> dict[str, Any]:
    """Run the simplified manifest path without loading a CAD mesh."""
    from .stereo_analyzer import _analysis_config, _capture_groups_from_manifest, _calibration_from_manifest, _compute_stereo_depth, _rectify, _hash_file, load_photo_snapshot

    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    calibration = _calibration_from_manifest(manifest.get("stereo_calibration"))
    if not calibration.validated:
        raise ElevationDepthError("elevation depth requires validated stereo calibration")
    _run_id, _interval, groups = _capture_groups_from_manifest(path, manifest.get("capture"), calibration)
    analysis = manifest.get("analysis")
    if not isinstance(analysis, Mapping) or analysis.get("mode") != "elevation_depth":
        raise ElevationDepthError("analysis.mode must be 'elevation_depth'")
    settings = dict(analysis.get("elevation_depth") or {})
    specs = settings.get("pipes") or manifest.get("model", {}).get("pipes")
    if not specs:
        raise ElevationDepthError("elevation_depth.pipes is required")
    config = _analysis_config({**analysis, "stereo_matching": analysis.get("stereo_matching", {})})
    groups_for_engine = []
    for group in groups:
        images = {}
        hashes = []
        quality_ok = True
        for role in ("left", "right"):
            image, integrity = load_photo_snapshot(path, group["views"][role])
            image = _rectify(image, getattr(calibration, role), calibration.rectified)
            images[role] = image
            hashes.append(integrity["actual_sha256"])
            quality_ok = quality_ok and bool(group["sync_valid"])
        depth = _compute_stereo_depth(images["left"], images["right"], calibration, config)
        groups_for_engine.append({"left": images["left"], "right": images["right"], "depth": depth, "pair_healthy": quality_ok and depth.audit.get("status") == "VALID", "capture_id": group["capture_id"], "pair_signature": ":".join(hashes)})
    result = analyze_elevation_groups(groups_for_engine, calibration=calibration, pipe_specs=specs, config=settings)
    result.update({"schema_version": "2.0", "report_type": "stereo-elevation-depth-pipe-installation-state", "dataset_id": manifest.get("dataset_id"), "model_revision": manifest.get("model_revision"), "capture_group_id": manifest.get("capture", {}).get("capture_group_id") if isinstance(manifest.get("capture"), Mapping) else None, "field_installation_state_inferred": True})
    if report_output_path is not None:
        output = Path(report_output_path).resolve()
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


__all__ = ["ElevationDepthError", "analyze_elevation_depth_manifest", "analyze_elevation_groups"]
