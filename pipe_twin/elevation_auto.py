"""Observed pipe surfaces -> STL/DXF line layout -> elevation installation states.

Registration is expressed in the LEFT RECTIFIED CAMERA frame.  Pipe endpoints
are deliberately not used: translation along parallel pipe axes is not
observable from a local surface.  Free-space evidence is evaluated within the
common axial interval actually seen on the registration pipes.
"""
from __future__ import annotations

import copy
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
from .elevation_zones import (
    crop_calibration_for_zone,
    crop_stereo_group_for_zone,
    enabled_zones,
    normalize_zone_settings,
    offset_observation_pixels,
)
from .model_zones import zone_model_specs

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
    relative_matches = {
        item["pipe_id"]: item
        for item in (registration.get("relative_layout", {}).get("matches", [])
                     if isinstance(registration.get("relative_layout"), Mapping) else [])
        if isinstance(item, Mapping) and item.get("pipe_id")
    }
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
                relative_cross_section_offset_mm=(relative_matches.get(spec["pipe_id"], {}).get("observed_offset_transverse_mm")
                                                  if relative_matches.get(spec["pipe_id"]) else None),
                relative_cross_section_error_mm=(float(relative_matches[spec["pipe_id"]]["offset_error_mm"])
                                                 if spec["pipe_id"] in relative_matches else None),
                measured_color_srgb=str(observation.get("measured_color_srgb", observation.get("color_srgb", ""))).upper(),
                measured_color_class=str(observation.get("measured_color_class", "UNKNOWN")),
                color_consistent=bool(observation.get("color_consistent", False)),
                color_used_as=("NONE_GEOMETRY_ONLY" if observation.get("observation_basis") == "DEPTH_ONLY_LOCAL_CYLINDER"
                               else "AUXILIARY_HINT_ONLY"),
                relative_layout_basis="TRANSVERSE_PLANE_NORMAL_TO_COMMON_PIPE_AXIS",
                diameter_match_basis=("STEREO_LOCAL_RADIAL_P95_AND_PROJECTED_CHORD"
                                      if observation.get("parallel_local")
                                      else "STEREO_LOCAL_CYLINDER_OUTER_SURFACE"),
            )
            if isinstance(observation.get("distance_geometry"), Mapping):
                geometry = observation["distance_geometry"]
                evidence["stereo_distance_geometry"] = dict(geometry)
                if geometry.get("nearest_visible_surface_range_mm") is not None:
                    evidence["nearest_visible_surface_range_mm"] = float(geometry["nearest_visible_surface_range_mm"])
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
            reasons = [
                "LOCAL_PARALLEL_STRIP_MATCHED_TO_MODEL"
                if observation is not None and observation.get("parallel_local")
                else "LOCAL_DEPTH_GEOMETRY_MATCHED_TO_MODEL"
                if observation is not None and observation.get("observation_basis") == "DEPTH_ONLY_LOCAL_CYLINDER"
                else "LOCAL_CYLINDER_MATCHED_TO_STL"
            ]
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
                                   registration_settings: Mapping | None = None,
                                   matching_settings: Mapping | None = None) -> dict:
    """Pure in-memory entry point; every capture estimates its own alignment."""
    from .elevation_registration import register_elevation
    from .elevation_dataset import normalize_registration_settings
    from .matching_config import normalize_matching_settings
    from .recognition import recognize_local_pipes
    settings = normalize_registration_settings(registration_settings)
    # Preserve the pre-policy Python API behaviour for callers that invoke the
    # in-memory analyzer directly.  Persisted GUI/manifests pass an explicit
    # normalized policy (whose colour filter defaults off); old integrations
    # with no argument continue to use the colour-aware detector.
    matching_payload = {"color_filter_enabled": True} if matching_settings is None else matching_settings
    matching = normalize_matching_settings(matching_payload)
    if not groups:
        raise ValueError("至少需要一组双目照片")
    captures = []
    current_surface = None
    for group in groups:
        observation_mode = settings.get("local_observation_mode", "auto")
        # The recognition package owns the detector choice and fallback
        # policy.  Registration, absence evidence, and historical refresh
        # remain independent of the chosen observation backend.
        surface = recognize_local_pipes(
            observation_mode, group["left"], group["right"], group["depth"],
            calibration, pipe_specs, matching_settings=matching,
        )
        surface.setdefault("audit", {}).update({
            "local_observation_mode": observation_mode,
            "fixed_camera_relative_layout": True,
            "model_axis_world": settings.get("axis_world"),
            "diameter_filter_enabled": matching["diameter_filter_enabled"],
            "diameter_tolerance_mm": matching["diameter_tolerance_mm"],
            "diameter_tolerance_ratio": matching["diameter_tolerance_ratio"],
            "color_filter_enabled": matching["color_filter_enabled"],
            "color_filter_mode": matching["color_filter_mode"],
        })
        anchors = settings["anchors"] if group.get("anchors_apply", group is groups[-1]) else {}
        registration = register_elevation(
            pipe_specs, surface["observations"], settings["axis_world"], anchors=anchors,
            config={
                "diameter_tolerance_mm": matching["diameter_tolerance_mm"],
                "diameter_tolerance_ratio": matching["diameter_tolerance_ratio"],
            },
        )
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
                   "status_refresh": group.get("status_refresh"),
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
            "status_refresh": current.get("status_refresh"),
            "capture_audit": {"count": len(captures), "groups": captures},
            "registration_required": True, "qr_registration_required": False,
            "production_authority": False, "longitudinal_installation_segments_assessed": False,
            "counts_scope": "MODEL_CANDIDATE_POSITIONS_NOT_PHYSICAL_INVENTORY",
            "matching": matching,
            "scene_inventory": {"present_pipe_count": settings.get("present_pipe_count"),
                                "model_candidate_count": len(pipe_specs),
                                "observed_cylinder_count": len(current_surface["observations"]),
                                "observed_local_strip_count": sum(bool(o.get("parallel_local")) for o in current_surface["observations"]),
                                "observed_geometry_only_count": sum(o.get("observation_basis") == "DEPTH_ONLY_LOCAL_CYLINDER"
                                                                     for o in current_surface["observations"]),
                                "identity_or_absence_proof": False}}


def _zone_observation_id(zone_id: str, value: Any) -> str:
    text = str(value or "")
    return text if text.startswith(f"{zone_id}:") else f"{zone_id}:{text}"


def _offset_evidence_boxes(evidence: Mapping[str, Any], offsets: Mapping[str, int]) -> dict[str, Any]:
    result = copy.deepcopy(dict(evidence))
    for role, key in (("left", "left_x"), ("right", "right_x")):
        view = result.get(role)
        if isinstance(view, Mapping):
            view = dict(view)
            view["region_xywh"] = (
                [int(view["region_xywh"][0]) + int(offsets[key]),
                 int(view["region_xywh"][1]) + int(offsets[f"{role}_y"]),
                 int(view["region_xywh"][2]), int(view["region_xywh"][3])]
                if isinstance(view.get("region_xywh"), (list, tuple)) and len(view["region_xywh"]) == 4
                else None
            )
            result[role] = view
    return result


def _qualify_zone_report(report: Mapping[str, Any], zone: Mapping[str, Any],
                         offsets: Mapping[str, int]) -> dict[str, Any]:
    """Make local IDs and boxes globally auditable after a zone crop."""

    zone_id = str(zone["zone_id"])
    result = copy.deepcopy(dict(report))
    surface = result.get("local_surface")
    if isinstance(surface, Mapping):
        surface = dict(surface)
        observations = []
        for raw in surface.get("observations", []):
            item = offset_observation_pixels(raw, offsets)
            item["observation_id"] = _zone_observation_id(zone_id, item.get("observation_id"))
            observations.append(item)
        surface["observations"] = observations
        rejected = []
        for raw in surface.get("rejected", []):
            item = copy.deepcopy(raw)
            for role, key in (("left", "left_x"), ("right", "right_x")):
                box_key = f"{role}_region_px"
                if isinstance(item.get(box_key), (list, tuple)) and len(item[box_key]) == 4:
                    item[box_key] = [int(item[box_key][0]) + int(offsets[key]),
                                     int(item[box_key][1]) + int(offsets[f"{role}_y"]),
                                     int(item[box_key][2]), int(item[box_key][3])]
            rejected.append(item)
        surface["rejected"] = rejected
        surface.setdefault("audit", {})["zone_id"] = zone_id
        result["local_surface"] = surface

    def qualify_registration(registration: Any) -> Any:
        if not isinstance(registration, Mapping):
            return registration
        value = copy.deepcopy(dict(registration))
        for item in value.get("matches", []) if isinstance(value.get("matches"), list) else []:
            if isinstance(item, Mapping) and item.get("observation_id"):
                item["observation_id"] = _zone_observation_id(zone_id, item["observation_id"])
        relative = value.get("relative_layout")
        if isinstance(relative, Mapping):
            relative = dict(relative)
            for item in relative.get("matches", []) if isinstance(relative.get("matches"), list) else []:
                if isinstance(item, Mapping) and item.get("observation_id"):
                    item["observation_id"] = _zone_observation_id(zone_id, item["observation_id"])
            value["relative_layout"] = relative
        return value

    result["registration"] = qualify_registration(result.get("registration"))
    for row in result.get("pipes", []) if isinstance(result.get("pipes"), list) else []:
        if not isinstance(row, Mapping):
            continue
        if row.get("current_evidence"):
            row["current_evidence"] = _offset_evidence_boxes(row["current_evidence"], offsets)
            if row["current_evidence"].get("observation_id"):
                row["current_evidence"]["observation_id"] = _zone_observation_id(
                    zone_id, row["current_evidence"]["observation_id"])
    audit = result.get("capture_audit")
    if isinstance(audit, Mapping):
        audit = dict(audit)
        groups = []
        for group in audit.get("groups", []) if isinstance(audit.get("groups"), list) else []:
            item = copy.deepcopy(group)
            item["zone_id"] = zone_id
            item["zone_source"] = zone.get("source", "manual")
            if zone.get("source") == "stl_parallel":
                item["zone_model_pipe_ids"] = list(zone.get("model_pipe_ids", []))
                item["zone_roi_source"] = zone.get("roi_source", "unmapped")
            item["registration"] = qualify_registration(item.get("registration"))
            item["local_observations"] = []
            for raw in group.get("local_observations", []) if isinstance(group, Mapping) else []:
                observation = offset_observation_pixels(raw, offsets)
                observation["observation_id"] = _zone_observation_id(zone_id, observation.get("observation_id"))
                item["local_observations"].append(observation)
            if isinstance(item.get("pipes"), Mapping):
                item["pipes"] = {
                    pipe_id: _offset_evidence_boxes(evidence, offsets)
                    for pipe_id, evidence in item["pipes"].items()
                }
                for evidence in item["pipes"].values():
                    if evidence.get("observation_id"):
                        evidence["observation_id"] = _zone_observation_id(zone_id, evidence["observation_id"])
            groups.append(item)
        audit["groups"] = groups
        audit["zone_id"] = zone_id
        result["capture_audit"] = audit
    result["zone_id"] = zone_id
    result["zone_label"] = zone.get("label", zone_id)
    result["zone_offsets"] = dict(offsets)
    result["zone_source"] = zone.get("source", "manual")
    if zone.get("source") == "stl_parallel":
        result["zone_model_pipe_ids"] = list(zone.get("model_pipe_ids", []))
        result["zone_model_catalog_sha256"] = zone.get("model_catalog_sha256")
        result["zone_roi_source"] = zone.get("roi_source", "unmapped")
    return result


def _merge_zone_registrations(registrations: list[Mapping[str, Any]]) -> dict[str, Any]:
    matched = [item for item in registrations if item.get("status") == "MATCHED"]
    base = copy.deepcopy(max(matched or registrations, key=lambda item: len(item.get("matches", [])))) if registrations else {
        "status": "INSUFFICIENT_OBSERVATIONS", "matches": []}
    by_pipe: dict[str, dict[str, Any]] = {}
    by_observation: dict[str, str] = {}
    conflicts: list[dict[str, Any]] = []
    for registration in matched:
        for raw in registration.get("matches", []):
            if not isinstance(raw, Mapping):
                continue
            item = copy.deepcopy(dict(raw))
            pipe_id = str(item.get("pipe_id", ""))
            observation_id = str(item.get("observation_id", ""))
            if not pipe_id or not observation_id:
                continue
            previous = by_pipe.get(pipe_id)
            if previous is None or float(item.get("residual_mm", math.inf)) < float(previous.get("residual_mm", math.inf)):
                by_pipe[pipe_id] = item
            old_pipe = by_observation.get(observation_id)
            if old_pipe is not None and old_pipe != pipe_id:
                conflicts.append({"observation_id": observation_id, "pipe_ids": [old_pipe, pipe_id]})
            by_observation[observation_id] = pipe_id
    if not matched:
        base["status"] = base.get("status", "INSUFFICIENT_OBSERVATIONS")
    elif conflicts:
        base["status"] = "ZONE_MATCH_CONFLICT"
        base["reason_codes"] = sorted(set((base.get("reason_codes") or []) + ["ZONE_MATCH_CONFLICT"]))
    else:
        base["status"] = "MATCHED"
    base["matches"] = list(by_pipe.values())
    base["zone_registrations"] = [
        {"zone_id": item.get("zone_id"), "status": item.get("status"),
         "match_count": len(item.get("matches", [])),
         "reason_codes": list(item.get("reason_codes") or [])}
        for item in registrations
    ]
    if conflicts:
        base["zone_conflicts"] = conflicts
    return base


def _merge_zone_rows(zone_reports: list[Mapping[str, Any]], pipe_specs: list[dict]) -> list[dict[str, Any]]:
    by_pipe: dict[str, list[tuple[str, Mapping[str, Any]]]] = {spec["pipe_id"]: [] for spec in pipe_specs}
    for report in zone_reports:
        zone_id = str(report.get("zone_id"))
        for row in report.get("pipes", []):
            if isinstance(row, Mapping) and row.get("pipe_id") in by_pipe:
                by_pipe[row["pipe_id"]].append((zone_id, row))
    rows = []
    for spec in pipe_specs:
        entries = by_pipe[spec["pipe_id"]]
        installed = [entry for entry in entries if entry[1].get("installation_state") == "INSTALLED"]
        negative = [entry for entry in entries if entry[1].get("installation_state") == "NOT_INSTALLED"]
        unknown = [entry for entry in entries if entry[1].get("installation_state") == "UNKNOWN"]
        if installed and negative:
            state = "UNKNOWN"
            chosen = installed[0][1]
            reasons = ["ZONE_RESULT_CONFLICT"]
        elif installed:
            state = "INSTALLED"
            chosen = max(installed, key=lambda item: bool(item[1].get("measurement", {}).get("measured_section")))[1]
            reasons = list(chosen.get("reason_codes") or ["LOCAL_CYLINDER_MATCHED_TO_STL"])
        elif negative and not unknown:
            state = "NOT_INSTALLED"
            chosen = negative[0][1]
            reasons = list(chosen.get("reason_codes") or ["REPEATED_REGISTERED_FREE_SPACE"])
        elif entries:
            state = "UNKNOWN"
            chosen = next((item[1] for item in entries if item[1].get("current_evidence")), entries[0][1])
            reasons = ["ZONE_PARTIAL_EVIDENCE"] if negative else list(chosen.get("reason_codes") or ["ZONE_NO_EVIDENCE"])
        else:
            state = "UNKNOWN"
            chosen = {"pipe_id": spec["pipe_id"], "measurement": {}, "current_evidence": {}}
            reasons = ["ZONE_NO_EVIDENCE"]
        row = copy.deepcopy(dict(chosen))
        row.update(pipe_id=spec["pipe_id"], installation_state=state,
                   installation_state_zh=STATE_ZH[state], reason_codes=reasons,
                   state_basis=reasons[0], zone_ids=[zone_id for zone_id, _ in entries],
                   zone_results=[{"zone_id": zone_id, "installation_state": item.get("installation_state"),
                                  "reason_codes": list(item.get("reason_codes") or [])}
                                 for zone_id, item in entries])
        rows.append(row)
    return rows


def _merge_zone_capture_groups(zone_reports: list[Mapping[str, Any]], group_count: int,
                               pipe_specs: list[dict]) -> list[dict[str, Any]]:
    merged = []
    for index in range(group_count):
        entries = [report.get("capture_audit", {}).get("groups", [])[index]
                   for report in zone_reports]
        base = copy.deepcopy(entries[0])
        base["zone_ids"] = [str(item.get("zone_id")) for item in entries]
        base["zone_audits"] = [{"zone_id": item.get("zone_id"),
                                 "source": item.get("zone_source", "manual"),
                                 "model_pipe_ids": list(item.get("zone_model_pipe_ids", [])),
                                 "roi_source": item.get("zone_roi_source", "user"),
                                 "surface_audit": item.get("surface_audit"),
                                 "registration": item.get("registration")} for item in entries]
        base["registration"] = _merge_zone_registrations([item.get("registration", {}) for item in entries])
        observations = []
        for item in entries:
            observations.extend(item.get("local_observations", []))
        base["local_observations"] = observations
        pipes = {}
        for spec in pipe_specs:
            candidates = [item.get("pipes", {}).get(spec["pipe_id"]) for item in entries
                          if spec["pipe_id"] in item.get("pipes", {})]
            candidates = [item for item in candidates if isinstance(item, Mapping)]
            if not candidates:
                continue
            chosen = max(candidates, key=lambda item: (bool(item.get("installed_candidate")),
                                                       bool(item.get("measured_diameter_mm") is not None),
                                                       bool(item.get("free_space_candidate"))))
            evidence = copy.deepcopy(dict(chosen))
            evidence["zone_ids"] = [str(item.get("zone_id")) for item in entries
                                     if spec["pipe_id"] in item.get("pipes", {})]
            evidence["zone_evidence"] = [copy.deepcopy(item.get("pipes", {}).get(spec["pipe_id"]))
                                          for item in entries if spec["pipe_id"] in item.get("pipes", {})]
            evidence["installed_candidate"] = any(bool(item.get("pipes", {}).get(spec["pipe_id"], {}).get("installed_candidate"))
                                                  for item in entries)
            evidence["free_space_candidate"] = any(bool(item.get("pipes", {}).get(spec["pipe_id"], {}).get("free_space_candidate"))
                                                for item in entries)
            pipes[spec["pipe_id"]] = evidence
        base["pipes"] = pipes
        base["surface_audit"] = {"scope": "zones", "zones": base["zone_audits"]}
        merged.append(base)
    return merged


def _merge_zone_reports(zone_reports: list[Mapping[str, Any]], *, groups: list[Mapping[str, Any]],
                        calibration: Any, pipe_specs: list[dict], matching: Mapping[str, Any],
                        scope: Mapping[str, Any]) -> dict[str, Any]:
    if not zone_reports:
        raise ValueError("至少需要一个启用分区")
    latest_surfaces = [report.get("local_surface", {}) for report in zone_reports]
    observations = [observation for surface in latest_surfaces for observation in surface.get("observations", [])]
    rejected = [item for surface in latest_surfaces for item in surface.get("rejected", [])]
    point_cloud = {"points_camera_mm": [], "colors_srgb": []}
    for surface in latest_surfaces:
        cloud = surface.get("point_cloud", {}) if isinstance(surface, Mapping) else {}
        point_cloud["points_camera_mm"].extend(cloud.get("points_camera_mm", []))
        point_cloud["colors_srgb"].extend(cloud.get("colors_srgb", []))
    point_cloud["points_camera_mm"] = point_cloud["points_camera_mm"][:6000]
    point_cloud["colors_srgb"] = point_cloud["colors_srgb"][:len(point_cloud["points_camera_mm"])]
    latest_regs = [report.get("registration", {}) for report in zone_reports]
    merged_groups = _merge_zone_capture_groups(zone_reports, len(groups), pipe_specs)
    latest_group_registrations = [group.get("registration", {}) for group in merged_groups]
    registration = _merge_zone_registrations(latest_group_registrations)
    rows = _merge_zone_rows(zone_reports, pipe_specs)
    current = zone_reports[0]
    audit = {"scope": "zones", "zone_count": len(zone_reports),
             "zones": [{"zone_id": report.get("zone_id"), "label": report.get("zone_label"),
                        "source": report.get("zone_source", "manual"),
                        "model_pipe_ids": list(report.get("zone_model_pipe_ids", [])),
                        "roi_source": report.get("zone_roi_source", "user"),
                        "offsets": report.get("zone_offsets"), "audit": report.get("local_surface", {}).get("audit", {})}
                       for report in zone_reports],
             "observation_count": len(observations), "rejected_count": len(rejected),
             "truncated": any(bool(surface.get("audit", {}).get("truncated")) for surface in latest_surfaces)}
    scene_inventory = {
        "present_pipe_count": current.get("scene_inventory", {}).get("present_pipe_count"),
        "model_candidate_count": len(pipe_specs),
        "observed_cylinder_count": len(observations),
        "observed_local_strip_count": sum(bool(item.get("parallel_local")) for item in observations),
        "observed_geometry_only_count": sum(item.get("observation_basis") == "DEPTH_ONLY_LOCAL_CYLINDER" for item in observations),
        "identity_or_absence_proof": False,
    }
    return {
        "mode": "elevation_auto", "scope": "zones", "scope_settings": copy.deepcopy(dict(scope)),
        "counts": {state: sum(row["installation_state"] == state for row in rows) for state in STATE_ZH},
        "pipes": rows, "registration": registration,
        "local_surface": {"observations": observations, "rejected": rejected, "point_cloud": point_cloud, "audit": audit},
        "status_refresh": current.get("status_refresh"),
        "capture_audit": {"count": len(merged_groups), "groups": merged_groups},
        "registration_required": True, "qr_registration_required": False,
        "production_authority": False, "longitudinal_installation_segments_assessed": False,
        "counts_scope": "MODEL_CANDIDATE_POSITIONS_NOT_PHYSICAL_INVENTORY",
        "matching": copy.deepcopy(dict(matching)), "scene_inventory": scene_inventory,
        "zone_registration_audit": {"registrations": latest_regs,
                                      "anchors_ignored": bool(current.get("registration", {}).get("anchors"))},
    }


def analyze_elevation_auto_zones(groups: list[Mapping], *, calibration: Any, pipe_specs: list[dict],
                                 registration_settings: Mapping | None = None,
                                 matching_settings: Mapping | None = None,
                                 scope_settings: Mapping | None = None) -> dict:
    """Analyze configured group zones and merge their independent evidence."""

    settings = normalize_zone_settings(scope_settings, image_size=(calibration.left.width, calibration.left.height))
    zones = enabled_zones(settings)
    if not groups:
        raise ValueError("至少需要一组双目照片")
    if not zones:
        raise ValueError("分区模式至少需要一个启用分区")
    from .elevation_dataset import normalize_registration_settings
    global_registration = normalize_registration_settings(registration_settings)
    zone_registration = dict(global_registration, anchors={})
    matching_payload = {"color_filter_enabled": True} if matching_settings is None else matching_settings
    from .matching_config import normalize_matching_settings
    matching = normalize_matching_settings(matching_payload)
    reports = []
    for zone in zones:
        zone_specs = zone_model_specs(zone, pipe_specs)
        cropped_groups = []
        offsets = None
        for group in groups:
            cropped, offsets = crop_stereo_group_for_zone(
                group, calibration, zone, right_padding_px=settings["right_roi_padding_px"])
            cropped_groups.append(cropped)
        report = analyze_elevation_auto_groups(
            cropped_groups, calibration=crop_calibration_for_zone(
                calibration, zone, right_padding_px=settings["right_roi_padding_px"]),
            pipe_specs=zone_specs, registration_settings=zone_registration,
            matching_settings=matching,
        )
        reports.append(_qualify_zone_report(report, zone, offsets or {}))
    result = _merge_zone_reports(reports, groups=groups, calibration=calibration,
                                 pipe_specs=pipe_specs, matching=matching, scope=settings)
    if global_registration["anchors"]:
        result["zone_registration_audit"]["anchors_ignored"] = True
        result["zone_registration_audit"]["reason"] = "ZONE_MODE_DOES_NOT_USE_PER_PIPE_ANCHORS"
    return result


def analyze_elevation_auto_manifest(manifest_path: str | Path, *, report_output_path: str | Path | None = None,
                                    evidence_dir: str | Path | None = None) -> dict:
    from .elevation_dataset import load_elevation_dataset
    from .stereo_analyzer import _analysis_config, _calibration_from_manifest, _capture_groups_from_manifest, _compute_stereo_depth, _quality
    from .photo_capture import load_photo_snapshot, resolve_photo_path
    from .pipeline import atomic_write_bytes, ensure_paths_distinct

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
                           status_refresh=capture.get("status_refresh"),
                           pair_healthy=bool(capture["sync_valid"] and depth.audit["status"] == "VALID" and all(q["passed"] for q in quality.values())),
                           quality=quality, photos=photos, anchors_apply=anchors_apply))
    if loaded["registration_settings"]["anchors"] and not any(g["anchors_apply"] for g in groups):
        raise ValueError("手工基准对应缺少匹配的照片哈希，请在当前照片上重新指定")
    scope_settings = loaded.get("scope_settings") or {"scope": "full_frame", "zones": []}
    if scope_settings.get("scope") == "zones":
        result = analyze_elevation_auto_zones(
            groups, calibration=calibration, pipe_specs=loaded["pipe_specs"],
            registration_settings=loaded["registration_settings"],
            matching_settings=config["matching"], scope_settings=scope_settings,
        )
    else:
        result = analyze_elevation_auto_groups(
            groups, calibration=calibration, pipe_specs=loaded["pipe_specs"],
            registration_settings=loaded["registration_settings"],
            matching_settings=config["matching"],
        )
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
            atomic_write_bytes(target, encoded.tobytes())
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
        atomic_write_bytes(outputs["point_cloud"], ("\n".join(lines)+"\n").encode("utf-8"))
        evidence_files.append(str(outputs["point_cloud"]))
    result["evidence_files"] = evidence_files
    if "report" in outputs:
        atomic_write_bytes(outputs["report"], (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode("utf-8"))
    log_event(LOGGER, "automatic_elevation_finished", manifest=str(path), counts=result["counts"],
              registration_status=result["registration"]["status"], surface_audit=result["local_surface"].get("audit"))
    return result
