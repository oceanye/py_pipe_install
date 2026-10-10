"""Metric, length-invariant registration for the basic elevation workflow.

The basic workflow only needs to identify the common cross-section of a set of
parallel pipes.  This module therefore registers pipe centre points in the
planes normal to their common axis.  Translation along that axis is a gauge,
not an observable quantity, and is reported as such.

This is intentionally separate from the QR/CAD registration used by the full
dashboard.  A small or symmetric visible subset returns an explicit
``AMBIGUOUS``/``INSUFFICIENT_OBSERVATIONS`` result instead of inventing a pose.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .camera_view import VIEW_POLICY, VIEW_POLICY_SHA256, normalize_camera_side


class ElevationRegistrationError(ValueError):
    """Malformed model or local-surface registration input."""


_EPS = 1.0e-9


def _finite_vector(value: Any, field: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ElevationRegistrationError(f"{field} must be a finite 3-D vector") from exc
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ElevationRegistrationError(f"{field} must be a finite 3-D vector")
    return result


def _unit(value: Any, field: str) -> np.ndarray:
    vector = _finite_vector(value, field)
    length = float(np.linalg.norm(vector))
    if length <= _EPS:
        raise ElevationRegistrationError(f"{field} must be non-zero")
    return vector / length


def _color(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) not in (7, 9) or value[0] != "#":
        raise ElevationRegistrationError(f"{field} must be #RRGGBB or #RRGGBBAA")
    try:
        int(value[1:], 16)
    except ValueError as exc:
        raise ElevationRegistrationError(f"{field} must be #RRGGBB or #RRGGBBAA") from exc
    return value[:7].upper()


def _diameter(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ElevationRegistrationError(f"{field} must be a positive number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ElevationRegistrationError(f"{field} must be a positive number")
    return result


def _basis(axis: np.ndarray) -> np.ndarray:
    """Return a right-handed [u,v,axis] orthonormal basis."""
    helper = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(helper, axis))) > 0.85:
        helper = np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, helper)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    v /= np.linalg.norm(v)
    return np.column_stack((u, v, axis))


def _model_inputs(pipe_specs: Sequence[Mapping[str, Any]], axis_world: Any) -> tuple[list[dict[str, Any]], np.ndarray]:
    if not isinstance(pipe_specs, Sequence) or isinstance(pipe_specs, (str, bytes)) or not pipe_specs:
        raise ElevationRegistrationError("pipe_specs must be a non-empty sequence")
    specs: list[dict[str, Any]] = []
    seen: set[str] = set()
    directions: list[np.ndarray] = []
    for index, raw in enumerate(pipe_specs):
        if not isinstance(raw, Mapping):
            raise ElevationRegistrationError(f"pipe_specs[{index}] must be an object")
        pipe_id = raw.get("pipe_id")
        if not isinstance(pipe_id, str) or not pipe_id.strip() or pipe_id in seen:
            raise ElevationRegistrationError("pipe_id values must be unique non-empty text")
        centreline = np.asarray(raw.get("centerline_world_mm"), dtype=np.float64)
        if centreline.shape != (2, 3) or not np.all(np.isfinite(centreline)):
            raise ElevationRegistrationError(f"pipe_specs[{index}].centerline_world_mm must be 2x3")
        direction = centreline[1] - centreline[0]
        length = float(np.linalg.norm(direction))
        if length <= _EPS:
            raise ElevationRegistrationError(f"pipe_specs[{index}] has a zero-length centreline")
        directions.append(direction / length)
        specs.append({
            "pipe_id": pipe_id,
            "center": centreline.mean(axis=0),
            "diameter_mm": _diameter(raw.get("nominal_diameter_mm"), f"pipe_specs[{index}].nominal_diameter_mm"),
            "color": _color(raw.get("color_srgb"), f"pipe_specs[{index}].color_srgb"),
        })
        seen.add(pipe_id)
    if axis_world is None:
        axis = directions[0].copy()
    else:
        axis = _unit(axis_world, "axis_world")
    for direction in directions:
        alignment = float(np.dot(direction, axis))
        if abs(alignment) < math.cos(math.radians(2)):
            raise ElevationRegistrationError("model pipe axes must be parallel to one common axis")
    return specs, axis


def _observations(values: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ElevationRegistrationError("observations must be a sequence")
    result, seen = [], set()
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            raise ElevationRegistrationError(f"observations[{index}] must be an object")
        oid = raw.get("observation_id")
        if not isinstance(oid, str) or not oid.strip() or oid in seen:
            raise ElevationRegistrationError("observation_id values must be unique non-empty text")
        validated_color = raw.get("color_identity_validated", False)
        if type(validated_color) is not bool:
            raise ElevationRegistrationError("color_identity_validated must be boolean")
        color_hints = raw.get("color_candidate_pipe_ids", [])
        if color_hints is None:
            color_hints = []
        if (not isinstance(color_hints, list)
                or any(not isinstance(value, str) or not value for value in color_hints)
                or len(set(color_hints)) != len(color_hints)):
            raise ElevationRegistrationError("color_candidate_pipe_ids must be a unique list of pipe IDs")
        result.append({"observation_id": oid,
            "center": _finite_vector(raw.get("center_camera_mm"), "center_camera_mm"),
            "axis": _unit(raw.get("axis_camera"), "axis_camera"),
            "diameter_mm": _diameter(raw.get("diameter_mm"), "diameter_mm"),
            "color": _color(raw.get("color_srgb"), "color_srgb"),
            "color_identity_validated": validated_color,
            "color_candidate_pipe_ids": color_hints})
        seen.add(oid)
    return result


def _common_axis(obs: list[dict]) -> tuple[np.ndarray, list[int], bool]:
    """Fit the largest angular consensus first so distant clutter cannot tilt it."""
    axes = np.asarray([o["axis"] for o in obs])
    compatible = np.abs(axes @ axes.T) >= math.cos(math.radians(4))
    sizes = compatible.sum(axis=1)
    best_sets = {tuple(np.flatnonzero(compatible[i])) for i in np.flatnonzero(sizes == sizes.max())}
    chosen = min(best_sets)
    # Two separate, equally supported directions need another observation.
    ambiguous = any(not set(chosen).intersection(other) for other in best_sets)
    _, vectors = np.linalg.eigh(axes[list(chosen)].T @ axes[list(chosen)])
    axis = vectors[:, -1]
    if axis[int(np.argmax(np.abs(axis)))] < 0:
        axis = -axis
    inliers = [i for i in chosen if abs(float(axes[i] @ axis)) >= math.cos(math.radians(4))]
    return axis, inliers, ambiguous


def _config(payload: Mapping | None) -> dict:
    defaults = {"max_residual_mm": 5., "ambiguity_delta_mm": 3., "diameter_tolerance_mm": 3.,
                "diameter_tolerance_ratio": 0.10,
                "max_angle_hypotheses": 2000, "max_hypotheses": 50000}
    if payload is not None and (not isinstance(payload, Mapping) or set(payload) - set(defaults)):
        raise ElevationRegistrationError("unknown registration config fields")
    cfg = defaults | dict(payload or {})
    for key, value in cfg.items():
        if key in {"max_angle_hypotheses", "max_hypotheses"}:
            valid = type(value) is int and value > 0
        else:
            valid = type(value) in (int, float) and math.isfinite(value) and (value >= 0 if key == "ambiguity_delta_mm" else value > 0)
        if not valid:
            raise ElevationRegistrationError(f"invalid registration config: {key}")
    if cfg["diameter_tolerance_ratio"] > 1:
        raise ElevationRegistrationError("diameter_tolerance_ratio cannot exceed 1")
    return cfg


def _candidates(specs: list[dict], observation: dict, tolerance: float, ratio: float = 0.10) -> list[int]:
    candidates = [i for i, spec in enumerate(specs)
        if abs(spec["diameter_mm"] - observation["diameter_mm"]) <= max(tolerance, ratio * max(spec["diameter_mm"], observation["diameter_mm"]))
        and (not observation["color_identity_validated"] or spec["color"] is None or observation["color"] is None or spec["color"] == observation["color"])]
    # Colour is a secondary hint after the metric diameter gate.  If the hint
    # has no compatible member, preserve the diameter candidates and let the
    # rigid cross-section layout decide instead of turning colour into an
    # identity assertion.
    hints = observation.get("color_candidate_pipe_ids")
    if isinstance(hints, list) and hints:
        hinted = {str(value) for value in hints}
        narrowed = [index for index in candidates if specs[index]["pipe_id"] in hinted]
        if narrowed:
            candidates = narrowed
    return candidates


def _noncollinear(points: np.ndarray, tolerance: float) -> bool:
    if len(points) < 3:
        return False
    singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    # Millimetre noise on an otherwise straight row is not a third layout constraint.
    return len(singular) >= 2 and singular[1] / math.sqrt(len(points)) > tolerance


def _rigid2(model: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mx, ox = model.mean(axis=0), observed.mean(axis=0)
    u, _, vt = np.linalg.svd((model-mx).T @ (observed-ox))
    correction = np.diag([1., 1. if np.linalg.det(vt.T @ u.T) > 0 else -1.])
    rotation = vt.T @ correction @ u.T
    translation = ox - rotation @ mx
    residuals = np.linalg.norm(model @ rotation.T + translation - observed, axis=1)
    return rotation, translation, residuals


def _relative_layout(specs: list[dict], observations: list[dict], model_xy: np.ndarray,
                     camera_xy: np.ndarray, ids: list[int], targets: list[int],
                     rotation2: np.ndarray) -> dict[str, Any]:
    """Describe the observed/model transverse layout without axial gauge.

    The fixed-camera measurement is the centre-to-centre geometry in the
    plane normal to the common pipe direction.  Pairwise distances survive a
    camera translation and the unobservable translation along the pipe axis,
    so they are the stable quantities to compare with a DXF/STL catalogue.
    """
    model_points = np.asarray(model_xy[targets], dtype=float)
    observed_points = np.asarray(camera_xy[ids], dtype=float)
    model_centroid = model_points.mean(axis=0)
    observed_centroid = observed_points.mean(axis=0)
    model_offsets = model_points - model_centroid
    observed_offsets = observed_points - observed_centroid
    aligned_model_offsets = model_offsets @ rotation2.T
    offset_errors = np.linalg.norm(aligned_model_offsets - observed_offsets, axis=1)
    matches = []
    for index, (observation_index, target_index) in enumerate(zip(ids, targets)):
        matches.append({
            "pipe_id": specs[target_index]["pipe_id"],
            "observation_id": observations[observation_index]["observation_id"],
            "model_offset_transverse_mm": model_offsets[index].tolist(),
            "observed_offset_transverse_mm": observed_offsets[index].tolist(),
            "offset_error_mm": float(offset_errors[index]),
        })
    pairwise = []
    for first in range(len(matches)):
        for second in range(first + 1, len(matches)):
            model_distance = float(np.linalg.norm(model_points[first] - model_points[second]))
            observed_distance = float(np.linalg.norm(observed_points[first] - observed_points[second]))
            pairwise.append({
                "pipe_id_a": matches[first]["pipe_id"],
                "pipe_id_b": matches[second]["pipe_id"],
                "observation_id_a": matches[first]["observation_id"],
                "observation_id_b": matches[second]["observation_id"],
                "model_distance_mm": model_distance,
                "observed_distance_mm": observed_distance,
                "error_mm": observed_distance - model_distance,
            })
    return {
        "coordinate_frame": "TRANSVERSE_PLANE_NORMAL_TO_COMMON_PIPE_AXIS",
        "measurement": "CENTRE_TO_CENTRE_RELATIVE_LAYOUT_MM",
        "axial_translation_excluded": True,
        "matches": matches,
        "pairwise": pairwise,
        "max_offset_error_mm": float(offset_errors.max()) if len(offset_errors) else 0.0,
        "max_pairwise_error_mm": max((abs(item["error_mm"]) for item in pairwise), default=0.0),
    }


def _result(reason: str, model_axis: np.ndarray, camera_axis: np.ndarray | None, *, status: str = "AMBIGUOUS") -> dict[str, Any]:
    return {"status": status, "rotation_model_to_camera": None, "translation_model_to_camera_mm": None,
            "model_axis": model_axis.tolist(), "camera_axis": camera_axis.tolist() if camera_axis is not None else None,
            "coordinate_frame": "LEFT_RECTIFIED_CAMERA_MM",
            "matches": [], "rms_mm": None, "reason_codes": [reason], "alternatives": [],
            "relative_layout": None,
            "axial_translation_observable": False, "scale": 1., "hypotheses_explored": 0,
            "rejected_observation_ids": []}


def register_elevation(pipe_specs: Sequence[Mapping[str, Any]], observations: Sequence[Mapping[str, Any]],
                       axis_world: Sequence[float] | None = None, *, anchors: Mapping[str, str] | None = None,
                       camera_side_world: Sequence[float] | None = None,
                       config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Metric cross-section registration with bounded maximum-consensus search.

    Both signs of the *observed* cylinder axis are searched even if a directed
    model axis was supplied.  All 3-D poses are proper rotations at fixed scale
    one.  Axial translation is a display gauge, never an installation station.
    Unmatched clutter is allowed; anchors cannot override measured geometry.
    A truncated search cannot establish either a match or absence.
    An optional camera-side vector constrains the viewing hemisphere in model
    coordinates; it supplies neither translation nor an observed pipe identity.
    """
    cfg = _config(config)
    specs, model_axis = _model_inputs(pipe_specs, axis_world)
    parsed_obs = _observations(observations)
    try:
        side = normalize_camera_side(camera_side_world)
    except ValueError as error:
        raise ElevationRegistrationError(str(error)) from error
    view_audit = {**VIEW_POLICY, "policy_sha256": VIEW_POLICY_SHA256,
                  "camera_side_world": side, "rejected_pose_count": 0}
    if anchors is not None and (not isinstance(anchors, Mapping) or any(not isinstance(k, str) or not isinstance(v, str) for k,v in anchors.items())):
        raise ElevationRegistrationError("anchors must map observation IDs to pipe IDs")
    anchor_map = dict(anchors or {})
    if (set(anchor_map) - {o["observation_id"] for o in parsed_obs} or
            set(anchor_map.values()) - {s["pipe_id"] for s in specs} or len(set(anchor_map.values())) != len(anchor_map)):
        raise ElevationRegistrationError("anchors must map unique observation IDs to pipe IDs")
    if not parsed_obs:
        output = _result("NO_LOCAL_SURFACE_OBSERVATIONS", model_axis, None, status="INSUFFICIENT_OBSERVATIONS")
        output["camera_view"] = dict(view_audit)
        return output
    camera_axis, inliers, multiple_axes = _common_axis(parsed_obs)
    rejected = [o["observation_id"] for i,o in enumerate(parsed_obs) if i not in inliers]

    def failure(reason, status="AMBIGUOUS"):
        output = _result(reason, model_axis, camera_axis, status=status)
        output["rejected_observation_ids"] = rejected.copy()
        output["settings"] = cfg
        output["camera_view"] = dict(view_audit)
        return output

    if multiple_axes:
        return failure("MULTIPLE_COMMON_AXIS_GROUPS")
    if set(anchor_map).intersection(rejected):
        return failure("ANCHOR_AXIS_MISMATCH")
    obs, candidates = [], []
    for i in inliers:
        item = parsed_obs[i]
        compatible = _candidates(specs, item, cfg["diameter_tolerance_mm"], cfg["diameter_tolerance_ratio"])
        target = anchor_map.get(item["observation_id"])
        if target is not None:
            compatible = [j for j in compatible if specs[j]["pipe_id"] == target]
            if not compatible:
                return failure("ANCHOR_DIAMETER_OR_COLOR_MISMATCH")
        if compatible:
            obs.append(item); candidates.append(compatible)
        else:
            rejected.append(item["observation_id"])
    min_count = 2 if len(anchor_map) >= 2 else 3
    if len(obs) < min_count:
        return failure("COLLINEAR_WITHOUT_TWO_ANCHORS", "INSUFFICIENT_OBSERVATIONS")
    model_basis = _basis(model_axis)
    model_centers = np.asarray([s["center"] for s in specs])
    observed_centers = np.asarray([o["center"] for o in obs])
    model_xy = (model_centers @ model_basis)[:, :2]
    residual_gate = cfg["max_residual_mm"]
    collinear_tolerance = min(2., residual_gate / 2)
    require_triangle = len(anchor_map) < 2
    if require_triangle and (not _noncollinear(model_xy, collinear_tolerance) or
            not _noncollinear((observed_centers @ _basis(camera_axis))[:, :2], collinear_tolerance)):
        return failure("COLLINEAR_WITHOUT_TWO_ANCHORS", "INSUFFICIENT_OBSERVATIONS")
    anchored = {i: candidates[i][0] for i,o in enumerate(obs) if o["observation_id"] in anchor_map}
    nodes, seed_count, limited, best_count = 0, 0, False, min_count
    solutions: dict[tuple, dict] = {}

    def consume() -> bool:
        nonlocal nodes, limited
        if nodes >= cfg["max_hypotheses"]:
            limited = True
            return False
        nodes += 1
        return True

    # A metric 2-D rigid pose is seeded by two candidate correspondences.
    # Pair checks and recursive assignment visits both consume the same budget.
    for sign in (1, -1):
        camera_basis = _basis(sign * camera_axis)
        camera_xy = (observed_centers @ camera_basis)[:, :2]
        for oi, oj in itertools.combinations(range(len(obs)), 2):
            if limited:
                break
            vo = camera_xy[oj] - camera_xy[oi]
            lo = float(np.linalg.norm(vo))
            if lo <= max(1., 2*residual_gate):
                continue
            for si, sj in itertools.product(candidates[oi], candidates[oj]):
                if not consume():
                    break
                if si == sj:
                    continue
                vm = model_xy[sj] - model_xy[si]
                lm = float(np.linalg.norm(vm))
                if lm < 1e-6 or abs(lm-lo) > 2*residual_gate:
                    continue
                if seed_count >= cfg["max_angle_hypotheses"]:
                    limited = True
                    break
                seed_count += 1
                angle = math.atan2(float(vm[0]*vo[1]-vm[1]*vo[0]), float(vm @ vo))
                co, sn = math.cos(angle), math.sin(angle)
                rotation2 = np.array([[co,-sn],[sn,co]])
                translation2 = (camera_xy[oi]+camera_xy[oj])/2 - rotation2 @ ((model_xy[si]+model_xy[sj])/2)
                transformed = model_xy @ rotation2.T + translation2
                nearby = [[j for j in candidates[i] if np.linalg.norm(transformed[j]-camera_xy[i]) <= 3*residual_gate] for i in range(len(obs))]
                forced = dict(anchored)
                if (oi in forced and forced[oi] != si) or (oj in forced and forced[oj] != sj):
                    continue
                forced.update({oi:si, oj:sj})
                if len(set(forced.values())) != len(forced) or any(j not in nearby[i] for i,j in forced.items()):
                    continue
                order = sorted((i for i in range(len(obs)) if i not in forced), key=lambda i: (len(nearby[i]),i))
                assignments = [-1]*len(obs)
                for i,j in forced.items(): assignments[i] = j
                used = set(forced.values())

                def visit(position: int, count: int) -> None:
                    nonlocal best_count
                    if not consume() or count+len(order)-position < best_count:
                        return
                    if position == len(order):
                        if count < min_count:
                            return
                        ids = [i for i,j in enumerate(assignments) if j >= 0]
                        targets = [assignments[i] for i in ids]
                        if require_triangle and (not _noncollinear(model_xy[targets], collinear_tolerance) or not _noncollinear(camera_xy[ids], collinear_tolerance)):
                            return
                        rot, trans, errors = _rigid2(model_xy[targets], camera_xy[ids])
                        if float(errors.max()) > residual_gate:
                            return
                        if side is not None:
                            rlocal = np.eye(3); rlocal[:2, :2] = rot
                            candidate_rotation = camera_basis @ rlocal @ model_basis.T
                            # Camera +Z points into the scene.  The selected
                            # outward side must point towards camera -Z.
                            facing = -float((candidate_rotation @ side)[2])
                            if facing <= VIEW_POLICY["minimum_facing_cosine_exclusive"]:
                                view_audit["rejected_pose_count"] += 1
                                return
                        if count > best_count:
                            solutions.clear(); best_count = count
                        key = (sign, tuple(assignments))
                        rms = float(np.sqrt(np.mean(errors**2)))
                        if key not in solutions or rms < solutions[key]["rms"]:
                            solutions[key] = {"sign":sign, "assignment":tuple(assignments), "ids":ids,
                                "rotation2":rot, "translation2":trans, "errors":errors, "rms":rms, "basis":camera_basis.copy()}
                        return
                    i = order[position]
                    for j in nearby[i]:
                        if j not in used:
                            assignments[i] = j; used.add(j)
                            visit(position+1,count+1)
                            used.remove(j); assignments[i] = -1
                            if limited: return
                    # Extra reconstructed cylinders need not belong to this STL.
                    visit(position+1,count)

                visit(0,len(forced))
                if limited:
                    break
        if limited:
            break
    if not solutions:
        reason = ("SEARCH_SPACE_TRUNCATED" if limited else "CAMERA_SIDE_CONFLICT"
                  if view_audit["rejected_pose_count"] else "NO_RIGID_MATCH_WITHIN_RESIDUAL_GATE")
        output = failure(reason, "SEARCH_LIMIT" if limited else "AMBIGUOUS")
        output.update(hypotheses_explored=nodes, pose_seeds=seed_count)
        return output
    ranked = sorted(solutions.values(), key=lambda s: (s["rms"],s["sign"],s["assignment"]))
    best = ranked[0]
    close = [s for s in ranked[1:] if s["rms"] <= best["rms"] + cfg["ambiguity_delta_mm"]]

    def pose(solution: dict) -> dict:
        b = solution["basis"]
        rlocal = np.eye(3); rlocal[:2,:2] = solution["rotation2"]
        rotation = b @ rlocal @ model_basis.T
        ids = solution["ids"]
        targets = [solution["assignment"][i] for i in ids]
        axial = float(np.mean((observed_centers[ids]-model_centers[targets] @ rotation.T) @ b[:,2]))
        translation = b[:,:2] @ solution["translation2"] + axial*b[:,2]
        camera_xy_solution = (observed_centers @ b)[:, :2]
        relative_layout = _relative_layout(specs, obs, model_xy, camera_xy_solution, ids, targets, solution["rotation2"])
        return {"rotation_model_to_camera":rotation.tolist(), "translation_model_to_camera_mm":translation.tolist(),
            "estimated_camera_side_world": (rotation.T @ np.array([0., 0., -1.])).tolist(),
            "camera_axis": b[:,2].tolist(), "rms_mm":solution["rms"],
            "matches":[{"pipe_id":specs[j]["pipe_id"],"observation_id":obs[i]["observation_id"],"residual_mm":float(e)} for i,j,e in zip(ids,targets,solution["errors"])],
            "relative_layout": relative_layout}

    status = "SEARCH_LIMIT" if limited else "AMBIGUOUS" if close else "MATCHED"
    reason = "SEARCH_SPACE_TRUNCATED" if limited else "ALTERNATIVE_POSES_WITHIN_AMBIGUITY_GATE" if close else "RIGID_CROSS_SECTION_MATCH"
    output = failure(reason,status)
    output.update(pose(best))
    output.update(alternatives=[pose(s) for s in close[:5]], hypotheses_explored=nodes, pose_seeds=seed_count,
        matched_observation_count=len(best["ids"]), rejected_observation_ids=sorted(set(rejected + [o["observation_id"] for i,o in enumerate(obs) if best["assignment"][i] < 0])))
    return output


__all__ = ["ElevationRegistrationError", "register_elevation"]
