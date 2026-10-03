"""Observed pipe surfaces -> STL/DXF line layout -> elevation installation states.

Registration is expressed in the LEFT RECTIFIED CAMERA frame.  Pipe endpoints
are deliberately not used: translation along parallel pipe axes is not
observable from a local surface.  Free-space evidence is evaluated within the
common axial interval actually seen on the registration pipes.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .logging_config import get_logger, log_event
from .pipe_geometry import cylinder_section_geometry, pipe_measurement_record

LOGGER = get_logger("elevation_auto")
STATE_ZH = {"INSTALLED": "安装", "NOT_INSTALLED": "未安装", "UNKNOWN": "遮蔽不确定"}
STATE_BGR = {"INSTALLED": (60, 180, 60), "NOT_INSTALLED": (50, 50, 220), "UNKNOWN": (0, 190, 240)}


def cylinder_depth_grid(camera: Any, center: np.ndarray, axis: np.ndarray,
                        radius_mm: float, *, baseline_offset: float = 0,
                        step: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact pinhole rays against an infinite cylinder; values are camera Z.

    The right rectified eye has origin [baseline,0,0] in the left eye frame.
    No fronto-parallel assumption or scalar reference depth is involved.
    """
    center = np.asarray(center, float) - [baseline_offset, 0, 0]
    axis = np.array(axis, dtype=float, copy=True)
    if center.shape != (3,) or axis.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(axis).all():
        raise ValueError("Invalid cylinder geometry")
    length = np.linalg.norm(axis)
    if not math.isfinite(length) or length < 1e-9 or not math.isfinite(radius_mm) or radius_mm <= 0 or type(step) is not int or step < 1:
        raise ValueError("Invalid cylinder direction, radius or grid step")
    axis /= length
    ys, xs = np.meshgrid(np.arange(step // 2, camera.height, step),
                         np.arange(step // 2, camera.width, step), indexing="ij")
    k = camera.rectified_intrinsic
    rays = np.stack(((xs-k[0, 2])/k[0, 0], (ys-k[1, 2])/k[1, 1], np.ones(xs.shape)), axis=-1)
    transverse = rays - (rays @ axis)[..., None] * axis
    q = center - (center @ axis) * axis
    a = np.sum(transverse**2, axis=-1)
    b = transverse @ q
    discriminant = b*b - a*(q @ q - radius_mm**2)
    depth = np.full(xs.shape, np.nan)
    hit = (a > 1e-12) & (discriminant >= 0)
    z = np.zeros(xs.shape)
    z[hit] = (b[hit] - np.sqrt(discriminant[hit])) / a[hit]
    hit &= z > 1e-3
    depth[hit] = z[hit]
    return depth, ys, xs


def _line_geometry(spec: Mapping, registration: Mapping) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(registration["rotation_model_to_camera"], float)
    t = np.asarray(registration["translation_model_to_camera_mm"], float)
    line = np.asarray(spec["centerline_world_mm"], float) @ r.T + t
    axis = line[1] - line[0]
    return line.mean(axis=0), axis / np.linalg.norm(axis)


def _common_observed_interval(surface: Mapping, registration: Mapping) -> tuple[float, float] | None:
    observations = {o["observation_id"]: o for o in surface["observations"]}
    axis = np.asarray(registration["camera_axis"], float)
    ranges = []
    for pair in registration["matches"]:
        obs = observations[pair["observation_id"]]
        values = np.asarray(obs["observed_segment_camera_mm"], float) @ axis
        ranges.append((float(values.min()), float(values.max())))
    if not ranges:
        return None
    low, high = max(x[0] for x in ranges), min(x[1] for x in ranges)
    return (low, high) if high - low >= 40 else None


def _view_evidence(spec: Mapping, center: np.ndarray, axis: np.ndarray, camera: Any,
                   image: np.ndarray, measured: np.ndarray, valid: np.ndarray,
                   baseline: float, interval: tuple[float, float] | None,
                   common_axis: np.ndarray, step: int) -> dict:
    expected, ys, xs = cylinder_depth_grid(camera, center, axis, spec["nominal_diameter_mm"]/2,
                                          baseline_offset=baseline, step=step)
    mask = np.isfinite(expected)
    if interval is None:
        mask[:] = False
    else:
        k = camera.rectified_intrinsic
        positions = np.stack(((xs-k[0, 2])*expected/k[0, 0]+baseline,
                              (ys-k[1, 2])*expected/k[1, 1], expected), axis=-1)
        station = positions @ common_axis
        mask &= (station >= interval[0]) & (station <= interval[1])
    count = int(mask.sum())
    if not count:
        return {"region_xywh": None, "projected_samples": 0, "visibility": "OUT_OF_VIEW_OR_NO_COMMON_SECTION",
                "free_space_candidate": False, "foreground_fraction": 0., "valid_depth_fraction": 0.}
    rows, columns = ys[mask], xs[mask]
    x, y = int(columns.min()), int(rows.min())
    box = [x, y, min(camera.width-x, int(columns.max())-x+step), min(camera.height-y, int(rows.max())-y+step)]
    z = measured[rows, columns]
    good = valid[rows, columns] & np.isfinite(z) & (z > 0)
    predicted = expected[mask]
    # The thresholds express state screening tolerance, not measurement accuracy.
    tolerance = max(20., float(spec["nominal_diameter_mm"])*.3)
    foreground = good & (z < predicted-tolerance)
    target = good & (np.abs(z-predicted) <= tolerance)
    free = good & (z > predicted+max(30., tolerance))
    enough = count >= 40 and float(good.mean()) >= .7
    occlusion_fraction = float(foreground.mean())
    free_fraction = float(free.mean())
    visibility = "OCCLUDED" if occlusion_fraction > .05 else "DEPTH_INSUFFICIENT" if not enough else "OBSERVED_REGION"
    return {"region_xywh": box, "projected_samples": count, "expected_depth_mm": float(np.median(predicted)),
            "expected_depth_range_mm": [float(predicted.min()), float(predicted.max())],
            "target_tolerance_mm": tolerance, "free_space_margin_mm": max(30., tolerance),
            "valid_depth_fraction": float(good.mean()), "foreground_fraction": occlusion_fraction,
            "target_depth_fraction": float(target.mean()), "free_space_fraction": free_fraction,
            "visibility": visibility, "free_space_candidate": bool(enough and free_fraction >= .75 and occlusion_fraction <= .03)}


def evaluate_registered_capture(pipe_specs: list[dict], surface: Mapping, registration: Mapping,
                                images: Mapping, depth: Any, calibration: Any, *, healthy: bool) -> dict:
    """Compare current observed geometry to automatically placed model lines."""
    matched = registration.get("status") == "MATCHED"
    associations = {m["pipe_id"]: m for m in registration.get("matches", [])} if matched else {}
    observations = {str(item.get("observation_id")): item for item in surface.get("observations", [])
                    if isinstance(item, Mapping) and item.get("observation_id")}
    interval = _common_observed_interval(surface, registration) if matched else None
    common_axis = np.asarray(registration.get("camera_axis", [1, 0, 0]), float)
    step = max(1, math.ceil(max(calibration.left.width, calibration.left.height)/640))
    result = {}
    for spec in pipe_specs:
        evidence: dict[str, Any] = {}
        association = associations.get(spec["pipe_id"])
        observation = observations.get(str(association.get("observation_id"))) if association else None
        measured_section = None
        model_section = None
        if observation is not None and healthy:
            measured_diameter = float(observation["diameter_mm"])
            center_camera = np.asarray(observation["center_camera_mm"], dtype=float)
            evidence.update(
                measured_diameter_mm=measured_diameter,
                diameter_error_mm=measured_diameter - float(spec["nominal_diameter_mm"]),
                observed_center_camera_mm=center_camera.tolist(),
                observed_distance_to_left_camera_mm=float(np.linalg.norm(center_camera)),
                position_residual_mm=float(association.get("residual_mm")) if association.get("residual_mm") is not None else None,
                measured_color_srgb=str(observation.get("measured_color_srgb", observation.get("color_srgb", ""))).upper(),
                color_used_as="AUXILIARY_HINT_ONLY",
                diameter_match_basis=("STEREO_PARALLEL_LOCAL_STRIP_OUTER_SURFACE"
                                      if observation.get("parallel_local")
                                      else "STEREO_LOCAL_CYLINDER_OUTER_SURFACE"),
            )
            measured_section = cylinder_section_geometry(
                center_camera, observation["axis_camera"], measured_diameter,
                section_definition="MIDPOINT_OF_COMMON_STEREO_OBSERVED_AXIS_SEGMENT",
                intrinsic=calibration.left.rectified_intrinsic)
        if matched:
            center, axis = _line_geometry(spec, registration)
            if healthy and interval is not None:
                # Compare the model and observation at one transverse plane;
                # CAD midpoint/length is not observable from a local view.
                station = float(np.dot(np.asarray(observation["center_camera_mm"]), common_axis)) if observation is not None else sum(interval) / 2
                section_center = center + axis * ((station - float(center @ common_axis)) / float(axis @ common_axis))
                try:
                    model_section = cylinder_section_geometry(
                        section_center, axis, float(spec["nominal_diameter_mm"]),
                        section_definition="MODEL_AT_SAME_OBSERVED_TRANSVERSE_PLANE",
                        intrinsic=calibration.left.rectified_intrinsic)
                except ValueError:
                    # An unseen model member can lie behind the camera after
                    # registration; it must not abort visible-pipe analysis.
                    evidence["model_section_reason"] = "MODEL_SECTION_NOT_IN_FRONT_OF_CAMERA"
            if model_section is not None and measured_section is not None:
                evidence["model_center_camera_mm"] = section_center.tolist()
                evidence["model_distance_to_left_camera_mm"] = float(np.linalg.norm(section_center))
                evidence["center_distance_error_mm"] = float(np.linalg.norm(center_camera) - np.linalg.norm(section_center))
                evidence["distance_reference"] = "CENTERLINE_AT_SAME_OBSERVED_SECTION"
            for role in ("left", "right"):
                evidence[role] = _view_evidence(spec, center, axis, getattr(calibration, role), images[role],
                    getattr(depth, f"{role}_depth_mm"), getattr(depth, f"{role}_valid"),
                    calibration.baseline_mm if role == "right" else 0., interval, common_axis, step)
        positive = bool(healthy and matched and association is not None)
        free = bool(healthy and matched and not positive and all(evidence[r]["free_space_candidate"] for r in ("left", "right")))
        evidence.update(installed_candidate=positive, free_space_candidate=free,
                        observation_id=associations.get(spec["pipe_id"], {}).get("observation_id"))
        if not healthy:
            reasons = ["CAPTURE_OR_DEPTH_UNHEALTHY"]
        elif not matched:
            reasons = list(registration.get("reason_codes") or [registration.get("status", "REGISTRATION_REQUIRED")])
        elif positive:
            reasons = ["LOCAL_PARALLEL_STRIP_MATCHED_TO_MODEL" if observation is not None and observation.get("parallel_local")
                       else "LOCAL_CYLINDER_MATCHED_TO_STL"]
        elif free:
            reasons = ["OBSERVED_FREE_SPACE_REQUIRES_REPETITION"]
        elif any(evidence[r]["visibility"] == "OCCLUDED" for r in ("left", "right")):
            reasons = ["FOREGROUND_OCCLUSION"]
        else:
            reasons = ["OUT_OF_VIEW_OR_INSUFFICIENT_SURFACE_EVIDENCE"]
        evidence["reason_codes"] = reasons
        evidence["measurement"] = pipe_measurement_record(
            spec, status="MEASURED" if measured_section else "MODEL_PREDICTION_ONLY" if model_section else "UNKNOWN",
            measured=measured_section, model_prediction=model_section,
            surface_samples=observation.get("surface_samples", {}) if observation is not None else {},
            reason_codes=[] if measured_section else reasons)
        result[spec["pipe_id"]] = evidence
    return result


def _same_alignment(first: Mapping, second: Mapping, specs: list[dict]) -> bool:
    if first.get("status") != "MATCHED" or second.get("status") != "MATCHED":
        return False
    for spec in specs:
        ca, aa = _line_geometry(spec, first)
        cb, ab = _line_geometry(spec, second)
        if abs(float(aa @ ab)) < math.cos(math.radians(2)):
            return False
        offset = cb-ca
        if np.linalg.norm(offset-aa*(offset @ aa)) > 5:
            return False
    return True


def analyze_elevation_auto_groups(groups: list[Mapping], *, calibration: Any, pipe_specs: list[dict],
                                   registration_settings: Mapping | None = None) -> dict:
    """Pure in-memory entry point; every capture estimates its own alignment."""
    from .local_surface import extract_local_pipes
    from .parallel_local import extract_parallel_local_pipes
    from .elevation_registration import register_elevation
    from .elevation_dataset import normalize_registration_settings
    settings = normalize_registration_settings(registration_settings)
    if not groups:
        raise ValueError("至少需要一组双目照片")
    captures = []
    current_surface = None
    for group in groups:
        observation_mode = settings.get("local_observation_mode", "auto")
        cylinder_surface = None
        if observation_mode in {"auto", "cylinder"}:
            cylinder_surface = extract_local_pipes(group["left"], group["right"], group["depth"], calibration, pipe_specs)
        if observation_mode == "parallel_strip":
            surface = extract_parallel_local_pipes(group["left"], group["right"], group["depth"], calibration, pipe_specs)
        elif observation_mode == "auto" and (not cylinder_surface["observations"] or cylinder_surface.get("audit", {}).get("truncated")):
            # A board or foreground object can leave a long straight side strip
            # while hiding the curvature required by the cylinder fitter.  In
            # auto mode, retain the strict cylinder result when it is healthy;
            # otherwise use the bounded parallel-strip fallback and keep the
            # original audit for diagnosis.
            surface = extract_parallel_local_pipes(group["left"], group["right"], group["depth"], calibration, pipe_specs)
            surface.setdefault("audit", {})["fallback_from"] = "CYLINDER_SURFACE"
            surface["audit"]["cylinder_surface_audit"] = cylinder_surface.get("audit", {})
        else:
            surface = cylinder_surface
        anchors = settings["anchors"] if group.get("anchors_apply", group is groups[-1]) else {}
        registration = register_elevation(pipe_specs, surface["observations"], settings["axis_world"], anchors=anchors)
        present_count = settings.get("present_pipe_count")
        if present_count is not None and len(surface["observations"]) > present_count:
            registration = {"status": "SCENE_INVENTORY_CONFLICT", "matches": [],
                            "reason_codes": ["OBSERVATIONS_EXCEED_PRESENT_PIPE_COUNT"]}
        sensor_healthy = bool(group.get("pair_healthy", False))
        surface_truncated = bool(surface.get("audit", {}).get("truncated", False))
        healthy = sensor_healthy and not surface_truncated
        evidence = evaluate_registered_capture(pipe_specs, surface, registration, group, group["depth"], calibration, healthy=healthy)
        if surface_truncated:
            for item in evidence.values():
                item["reason_codes"] = ["LOCAL_SURFACE_SEARCH_TRUNCATED"]
        digest = hashlib.sha256()
        for role in ("left", "right"):
            digest.update(np.ascontiguousarray(group[role]).tobytes())
        capture = {"capture_id": group["capture_id"], "captured_at": group["captured_at"],
                   "pair_healthy": sensor_healthy, "analysis_healthy": healthy, "pair_signature": digest.hexdigest(), "registration": registration,
                   "surface_audit": surface.get("audit", {}), "pipes": evidence,
                   "local_observations": surface["observations"],
                   "quality": group.get("quality"), "depth_audit": group["depth"].audit,
                   "photos": group.get("photos")}
        captures.append(capture)
        current_surface = surface
    current = captures[-1]
    rows = []
    for spec in pipe_specs:
        pipe_id = spec["pipe_id"]
        evidence = current["pipes"][pipe_id]
        independent = []
        signatures = set()
        last_time = None
        for capture in reversed(captures):
            if not capture["pipes"][pipe_id]["free_space_candidate"] or not _same_alignment(current["registration"], capture["registration"], pipe_specs):
                break
            timestamp = datetime.fromisoformat(capture["captured_at"].replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                raise ValueError("拍摄时间必须包含时区")
            if capture["pair_signature"] in signatures or (last_time is not None and (last_time-timestamp).total_seconds() < 1):
                continue
            signatures.add(capture["pair_signature"])
            last_time = timestamp
            independent.append(capture["capture_id"])
        state = "INSTALLED" if evidence["installed_candidate"] else "NOT_INSTALLED" if len(independent) >= 2 else "UNKNOWN"
        reasons = ["REPEATED_REGISTERED_FREE_SPACE"] if state == "NOT_INSTALLED" else evidence["reason_codes"]
        rows.append({"pipe_id": pipe_id, "installation_state": state, "installation_state_zh": STATE_ZH[state],
                     "measurement": evidence["measurement"],
                     "reason_codes": reasons, "state_basis": reasons[0], "current_evidence": evidence,
                     "independent_free_space_captures": list(reversed(independent)),
                     "assessment_scope": "LOCAL_COMMON_AXIAL_SECTION"})
    return {"mode": "elevation_auto", "counts": {s: sum(r["installation_state"] == s for r in rows) for s in STATE_ZH},
            "pipes": rows, "registration": current["registration"], "local_surface": current_surface,
            "capture_audit": {"count": len(captures), "groups": captures},
            "registration_required": True, "qr_registration_required": False,
            "production_authority": False, "longitudinal_installation_segments_assessed": False,
            "counts_scope": "MODEL_CANDIDATE_POSITIONS_NOT_PHYSICAL_INVENTORY",
            "scene_inventory": {"present_pipe_count": settings.get("present_pipe_count"),
                                "model_candidate_count": len(pipe_specs),
                                "observed_cylinder_count": len(current_surface["observations"]),
                                "observed_local_strip_count": sum(bool(o.get("parallel_local")) for o in current_surface["observations"]),
                                "identity_or_absence_proof": False}}


def analyze_elevation_auto_manifest(manifest_path: str | Path, *, report_output_path: str | Path | None = None,
                                    evidence_dir: str | Path | None = None) -> dict:
    from .elevation_dataset import load_elevation_dataset
    from .stereo_analyzer import _analysis_config, _calibration_from_manifest, _capture_groups_from_manifest, _compute_stereo_depth, _quality
    from .photo_capture import load_photo_snapshot, resolve_photo_path
    from .pipeline import ensure_paths_distinct
    from .elevation_depth import _atomic_write

    path = Path(manifest_path).resolve()
    original = path.read_bytes()
    loaded = load_elevation_dataset(path)
    manifest = loaded["manifest"]
    if loaded["mode"] != "elevation_auto":
        raise ValueError("需要elevation_auto现场包")
    calibration = _calibration_from_manifest(loaded["calibration"])
    _, _, captures = _capture_groups_from_manifest(path, manifest["capture"], calibration)
    config = _analysis_config(manifest["analysis"])
    settings = manifest["analysis"]["elevation_auto"]
    inputs = {"manifest": path, "model": path.parent / manifest["model"]["path"]}
    for capture in captures:
        for role in ("left", "right"):
            inputs[f"{capture['capture_id']}_{role}"] = resolve_photo_path(path, capture["views"][role]["path"])
    outputs = {}
    if report_output_path is not None:
        outputs["report"] = Path(report_output_path).resolve()
    if evidence_dir is not None:
        root = Path(evidence_dir).resolve()
        for capture in captures:
            for role in ("left", "right"):
                target = (root / f"{capture['capture_id']}_{role}_overlay.png").resolve()
                if not target.is_relative_to(root):
                    raise ValueError("证据路径越界")
                outputs[f"overlay_{capture['capture_id']}_{role}"] = target
        outputs["point_cloud"] = root / "local_surface.ply"
    ensure_paths_distinct(**inputs, **outputs)
    hashes = {key: hashlib.sha256(p.read_bytes()).hexdigest() for key, p in inputs.items()}
    groups = []
    for capture in captures:
        images, photos, quality = {}, {}, {}
        for role in ("left", "right"):
            images[role], photos[role] = load_photo_snapshot(path, capture["views"][role])
            quality[role] = _quality(images[role], config)
        depth = _compute_stereo_depth(images["left"], images["right"], calibration, config)
        bound_anchors = settings.get("anchor_pair_sha256")
        anchors_apply = bool(bound_anchors and all(bound_anchors.get(r) == photos[r]["actual_sha256"] for r in ("left", "right")))
        groups.append(dict(images, depth=depth, capture_id=capture["capture_id"], captured_at=capture["captured_at"],
                           pair_healthy=bool(capture["sync_valid"] and depth.audit["status"] == "VALID" and all(q["passed"] for q in quality.values())),
                           quality=quality, photos=photos, anchors_apply=anchors_apply))
    if loaded["registration_settings"]["anchors"] and not any(g["anchors_apply"] for g in groups):
        raise ValueError("手工基准对应缺少匹配的照片哈希，请在当前照片上重新指定")
    result = analyze_elevation_auto_groups(groups, calibration=calibration, pipe_specs=loaded["pipe_specs"],
                                           registration_settings=loaded["registration_settings"])
    result.update(schema_version="2.0", report_type="stereo-auto-elevation-state", dataset_id=manifest["dataset_id"],
                  generated_at=datetime.now(timezone.utc).isoformat(), inputs=hashes,
                  calibration_audit={"calibration_id": calibration.calibration_id, "validated": calibration.validated,
                                     "coordinate_frame": "LEFT_RECTIFIED_CAMERA_MM"})
    # All inference used immutable decoded images. Recheck bound snapshots
    # before publishing any output, including the optional STL reference.
    if path.read_bytes() != original or any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[key] for key, p in inputs.items()):
        raise ValueError("分析期间现场输入发生变化，请重试")
    evidence_files = []
    latest_states = {row["pipe_id"]: row["installation_state"] for row in result["pipes"]}
    for group, audit in zip(groups, result["capture_audit"]["groups"]):
        for role in ("left", "right"):
            target = outputs.get(f"overlay_{group['capture_id']}_{role}")
            if target is None:
                continue
            overlay = group[role].copy()
            for pipe_id, evidence in audit["pipes"].items():
                box = evidence.get(role, {}).get("region_xywh")
                if box:
                    x, y, w, h = box
                    state = latest_states[pipe_id] if audit is result["capture_audit"]["groups"][-1] else "INSTALLED" if evidence["installed_candidate"] else "UNKNOWN"
                    color = STATE_BGR[state]
                    cv2.rectangle(overlay, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(overlay, pipe_id, (x, max(15, y-3)), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 1)
            if audit["registration"]["status"] != "MATCHED":
                for observation in audit["local_observations"]:
                    box = observation.get(f"{role}_region_px")
                    if box:
                        x, y, w, h = box
                        cv2.rectangle(overlay, (x, y), (x+w, y+h), (230, 180, 80), 2)
                        cv2.putText(overlay, observation["observation_id"], (x, max(15, y-3)), cv2.FONT_HERSHEY_SIMPLEX, .5, (230, 180, 80), 1)
            ok, encoded = cv2.imencode(".png", overlay)
            if not ok:
                raise ValueError("无法保存匹配叠图")
            _atomic_write(target, encoded.tobytes())
            evidence_files.append(str(target))
    if "point_cloud" in outputs:
        cloud = result["local_surface"].get("point_cloud", {})
        points = cloud.get("points_camera_mm", [])
        colors = cloud.get("colors_srgb", [])
        lines = ["ply", "format ascii 1.0", "comment frame LEFT_RECTIFIED_CAMERA_MM", f"element vertex {len(points)}",
                 "property float x", "property float y", "property float z", "property uchar red", "property uchar green", "property uchar blue", "end_header"]
        for index, point in enumerate(points):
            color = colors[index] if index < len(colors) else [160, 160, 160]
            if isinstance(color, str):
                color = [int(color[i:i+2], 16) for i in (1, 3, 5)]
            lines.append(" ".join([*(f"{v:.5f}" for v in point), *(str(int(v)) for v in color)]))
        _atomic_write(outputs["point_cloud"], ("\n".join(lines)+"\n").encode("utf-8"))
        evidence_files.append(str(outputs["point_cloud"]))
    result["evidence_files"] = evidence_files
    if "report" in outputs:
        _atomic_write(outputs["report"], (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8"))
    log_event(LOGGER, "automatic_elevation_finished", manifest=str(path), counts=result["counts"],
              registration_status=result["registration"]["status"], surface_audit=result["local_surface"].get("audit"))
    return result
