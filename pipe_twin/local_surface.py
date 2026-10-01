"""Observed local cylinders in the left rectified camera frame.

Depth-connected surfaces and optional RGB components propose candidates.  Both
eyes must independently resolve cylindrical curvature, then agree on the same
observed axial interval.  CAD supplies candidate diameters; identity and pose
are left to registration.  STL display colours need not match physical paint.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import cv2
import numpy as np

from .metrology import fit_local_cylinder


class LocalSurfaceError(ValueError):
    """Invalid stereo arrays, calibration, or pipe catalogue."""


_DEFAULTS = {
    "color_delta_lab": 45.0,
    "minimum_component_pixels": 80,
    "minimum_valid_fraction": 0.10,
    "depth_discontinuity_mm": 8.0,
    "pair_center_tolerance_mm": 5.0,
    "pair_axis_tolerance_deg": 4.0,
    "pair_diameter_tolerance_fraction": 0.12,
    "pair_diameter_tolerance_mm": 3.0,
    "minimum_common_support_mm": 40.0,
    "pair_ambiguity_margin_mm": 1.0,
    "maximum_cloud_points": 6000,
    "maximum_component_candidates": 96,
}


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not np.isfinite(norm) or norm <= 1e-9:
        raise ValueError("DEGENERATE_AXIS")
    return vector / norm


def _positive(value: Any, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise LocalSurfaceError(f"{field} must be finite and positive")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise LocalSurfaceError(f"{field} must be finite and positive")
    return value


def _rect_rotation(camera: Any) -> np.ndarray:
    value = getattr(camera, "rotation_world_to_rectified_camera", None)
    if value is None:
        raw = np.asarray(camera.rotation_world_to_camera, dtype=np.float64)
        rect = getattr(camera, "rectification_matrix", None)
        value = raw if rect is None else np.asarray(rect, dtype=np.float64) @ raw
    return np.asarray(value, dtype=np.float64)


def _intrinsic(camera: Any) -> np.ndarray:
    # Do not evaluate an intrinsic fallback eagerly: providers may expose only
    # the already-rectified matrix.
    value = getattr(camera, "rectified_intrinsic", None)
    if value is None:
        value = getattr(camera, "intrinsic", None)
    return np.asarray(value, dtype=np.float64)


def _hex_bgr(value: Any) -> np.ndarray:
    if not isinstance(value, str) or len(value) not in (7, 9) or value[0] != "#":
        raise LocalSurfaceError("pipe color must be #RRGGBB")
    try:
        channels = [int(value[i:i + 2], 16) for i in range(1, len(value), 2)]
    except ValueError as error:
        raise LocalSurfaceError("pipe color must be #RRGGBB") from error
    return np.asarray([[channels[:3][::-1]]], dtype=np.uint8)


def _settings(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is not None and not isinstance(value, Mapping):
        raise LocalSurfaceError("config must be an object")
    unknown = set(value or {}) - set(_DEFAULTS)
    if unknown:
        raise LocalSurfaceError(f"Unknown local surface settings: {sorted(unknown)}")
    cfg = {**_DEFAULTS, **dict(value or {})}
    for key in set(cfg) - {"minimum_component_pixels", "maximum_cloud_points", "maximum_component_candidates"}:
        cfg[key] = _positive(cfg[key], f"config.{key}")
    for key in ("minimum_component_pixels", "maximum_cloud_points", "maximum_component_candidates"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise LocalSurfaceError(f"config.{key} must be a positive integer")
    if cfg["minimum_component_pixels"] < 80:
        raise LocalSurfaceError("minimum_component_pixels must be at least 80")
    if cfg["maximum_cloud_points"] > 6000:
        raise LocalSurfaceError("maximum_cloud_points cannot exceed 6000")
    if cfg["maximum_component_candidates"] > 384:
        raise LocalSurfaceError("maximum_component_candidates cannot exceed 384")
    if cfg["minimum_valid_fraction"] > 1 or cfg["pair_diameter_tolerance_fraction"] > 1:
        raise LocalSurfaceError("fraction settings must not exceed 1")
    if cfg["pair_axis_tolerance_deg"] > 30:
        raise LocalSurfaceError("pair_axis_tolerance_deg cannot exceed 30")
    return cfg


def _catalog(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise LocalSurfaceError("pipe_specs must be a non-empty list")
    result = []
    seen = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise LocalSurfaceError(f"pipe_specs[{index}] must be an object")
        pipe_id = raw.get("pipe_id", f"P{index + 1:03d}")
        if not isinstance(pipe_id, str) or not pipe_id.strip() or pipe_id in seen:
            raise LocalSurfaceError("pipe_specs pipe_id values must be unique text")
        diameter = _positive(raw.get("nominal_diameter_mm", raw.get("diameter_mm")), "nominal_diameter_mm")
        color = raw.get("color_srgb", "#FFFFFF")
        _hex_bgr(color)
        result.append({"pipe_id": pipe_id, "nominal_diameter_mm": diameter, "color_srgb": color[:7].upper()})
        seen.add(pipe_id)
    return result


def _validate_arrays(left: Any, right: Any, depth: Any, calibration: Any) -> tuple[np.ndarray, ...]:
    if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray) or left.dtype != np.uint8 or right.dtype != np.uint8:
        raise LocalSurfaceError("left and right must be uint8 BGR arrays")
    if left.ndim != 3 or left.shape[2] != 3 or right.shape != left.shape or min(left.shape[:2]) <= 0:
        raise LocalSurfaceError("left and right must be nonempty matching BGR images")
    arrays = tuple(getattr(depth, name, None) for name in ("left_depth_mm", "right_depth_mm", "left_valid", "right_valid"))
    if any(not isinstance(item, np.ndarray) or item.shape != left.shape[:2] for item in arrays):
        raise LocalSurfaceError("depth arrays must match image dimensions")
    if arrays[2].dtype != np.bool_ or arrays[3].dtype != np.bool_:
        raise LocalSurfaceError("depth valid masks must be boolean")
    if any(item.dtype.kind not in "fiu" for item in arrays[:2]):
        raise LocalSurfaceError("depth values must have a real numeric dtype")
    if getattr(calibration, "validated", None) is not True or getattr(calibration, "rectified", None) is not True:
        raise LocalSurfaceError("validated rectified calibration is required")
    baseline = _positive(getattr(calibration, "baseline_mm", None), "calibration.baseline_mm")
    rotations, centres, matrices = [], [], []
    for role in ("left", "right"):
        camera = getattr(calibration, role, None)
        try:
            fx = _positive(getattr(camera, "fx", None), f"calibration.{role}.fx")
            fy = _positive(getattr(camera, "fy", None), f"calibration.{role}.fy")
            rotation = _rect_rotation(camera)
            centre = np.asarray(camera.center_world_mm, dtype=np.float64)
            matrix = _intrinsic(camera)
        except (AttributeError, TypeError, ValueError) as error:
            raise LocalSurfaceError(f"calibration.{role} camera is invalid: {error}") from error
        if (getattr(camera, "width", None), getattr(camera, "height", None)) != (left.shape[1], left.shape[0]):
            raise LocalSurfaceError(f"calibration.{role} dimensions do not match images")
        if (rotation.shape != (3, 3) or centre.shape != (3,) or not np.all(np.isfinite(rotation))
                or not np.all(np.isfinite(centre)) or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
                or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6)):
            raise LocalSurfaceError(f"calibration.{role} pose is invalid")
        canonical = np.asarray([[fx, 0, matrix[0, 2] if matrix.shape == (3, 3) else 0],
                                [0, fy, matrix[1, 2] if matrix.shape == (3, 3) else 0], [0, 0, 1]])
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)) or not np.allclose(matrix, canonical, atol=1e-6):
            raise LocalSurfaceError(f"calibration.{role} rectified intrinsic is invalid")
        rotations.append(rotation)
        centres.append(centre)
        matrices.append(matrix)
    baseline_left = rotations[0] @ (centres[1] - centres[0])
    if (not np.allclose(rotations[0], rotations[1], atol=1e-5)
            or not np.allclose(matrices[0], matrices[1], rtol=1e-5, atol=1e-6)
            or not np.allclose(baseline_left, [baseline, 0, 0], rtol=1e-4, atol=1e-3)):
        raise LocalSurfaceError("calibration does not describe a horizontal rectified stereo pair")
    return arrays


def _points(depth: np.ndarray, valid: np.ndarray, camera: Any, mask: np.ndarray, offset=(0, 0)) -> np.ndarray:
    rows, cols = np.nonzero(mask & valid & np.isfinite(depth) & (depth > 0))
    if not len(rows):
        return np.empty((0, 3), dtype=np.float64)
    z = depth[rows, cols].astype(np.float64)
    k = _intrinsic(camera)
    return np.column_stack(((cols + offset[0] - k[0, 2]) * z / camera.fx,
                            (rows + offset[1] - k[1, 2]) * z / camera.fy, z))


def _to_left(points: np.ndarray, calibration: Any, role: str) -> np.ndarray:
    # The validation above establishes the shared rectified orientation and
    # positive-X baseline.  Using relative translation avoids cancellation
    # from large, arbitrary CAD-world origins.
    if role == "left":
        return points
    return points + np.asarray([calibration.baseline_mm, 0.0, 0.0])


def _sample(points: np.ndarray, limit: int) -> np.ndarray:
    if len(points) <= limit:
        return points.copy()
    return points[np.linspace(0, len(points) - 1, limit, dtype=np.int64)]


class _CloudBudget:
    """Bound NumPy storage before ever making per-point Python objects."""

    def __init__(self, limit: int):
        self.limit = limit
        self.points = np.empty((0, 3), dtype=np.float64)
        self.colors = np.empty((0, 3), dtype=np.uint8)
        self.source_count = 0

    def add(self, points: np.ndarray, color: str, source_count: int) -> None:
        if not len(points):
            return
        total = self.source_count + source_count
        old_count = min(len(self.points), int(round(self.limit * self.source_count / total)))
        new_count = min(len(points), self.limit - old_count)
        old_indexes = np.linspace(0, len(self.points) - 1, old_count, dtype=np.int64) if old_count else np.empty(0, dtype=np.int64)
        new_points = _sample(points, new_count) if new_count else np.empty((0, 3))
        rgb = _hex_bgr(color)[0, 0, ::-1]
        self.points = np.concatenate((self.points[old_indexes], new_points))
        self.colors = np.concatenate((self.colors[old_indexes], np.tile(rgb, (new_count, 1))))
        self.source_count = total

    def public(self) -> dict[str, Any]:
        return {"points_camera_mm": self.points.tolist(),
                "colors_srgb": ["#%02X%02X%02X" % tuple(map(int, rgb)) for rgb in self.colors]}


def _depth_surface_mask(depth: np.ndarray, valid: np.ndarray, jump_mm: float) -> np.ndarray:
    """Split valid surfaces at measured depth jumps, without filling holes."""
    finite = valid & np.isfinite(depth) & (depth > 0)
    surface = finite.copy()
    values = np.where(finite, depth, 0).astype(np.float64)
    horizontal = finite[:, :-1] & finite[:, 1:] & (np.abs(values[:, :-1] - values[:, 1:]) > jump_mm)
    vertical = finite[:-1] & finite[1:] & (np.abs(values[:-1] - values[1:]) > jump_mm)
    surface[:, :-1] &= ~horizontal
    surface[:, 1:] &= ~horizontal
    surface[:-1] &= ~vertical
    surface[1:] &= ~vertical
    return surface


def _fit_components(image: np.ndarray, depth: np.ndarray, valid: np.ndarray, mask: np.ndarray,
                    calibration: Any, role: str, source: str, cfg: Mapping[str, Any],
                    target_color: str | None = None, *, budget: dict[str, int] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    camera = getattr(calibration, role)
    if budget is not None and budget["remaining"] <= 0:
        return [], [{"role": role, "segmentation_source": source, "target_color_srgb": target_color,
                     "reason": "CANDIDATE_BUDGET_EXCEEDED", "source_skipped": True}]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    accepted, rejected = [], []
    eligible = np.flatnonzero(stats[1:, cv2.CC_STAT_AREA] >= cfg["minimum_component_pixels"]) + 1
    eligible = eligible[np.argsort(-stats[eligible, cv2.CC_STAT_AREA], kind="stable")]
    limit = budget["remaining"] if budget is not None else cfg["maximum_component_candidates"]
    if len(eligible) > limit:
        rejected.append({"role": role, "segmentation_source": source, "target_color_srgb": target_color,
                         "reason": "CANDIDATE_BUDGET_EXCEEDED", "skipped_components": int(len(eligible) - limit)})
    for label in eligible[:limit]:
        if budget is not None:
            budget["remaining"] -= 1
            budget["attempted"] += 1
        area = int(stats[label, cv2.CC_STAT_AREA])
        box = [int(stats[label, column]) for column in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]
        x, y, width, height = box
        # Candidate work scales with its own bounding box, not with full-image
        # pixels multiplied by the number of speckles.
        component = labels[y:y + height, x:x + width] == label
        local_depth = depth[y:y + height, x:x + width]
        local_image = image[y:y + height, x:x + width]
        local_valid = valid[y:y + height, x:x + width] & np.isfinite(local_depth) & (local_depth > 0)
        if source == "depth_connected_surface":
            # Depth labels already contain valid pixels only.  Fill enclosed
            # holes for the denominator alone; none of these pixels becomes
            # geometry, free space or an interpolated depth observation.
            contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            enclosed = np.zeros(component.shape, dtype=np.uint8)
            cv2.drawContours(enclosed, contours, -1, 1, cv2.FILLED)
            denominator = max(area, int(np.count_nonzero(enclosed)))
            core = component
        else:
            denominator = area
            # RGB/depth boundaries can mix foreground and background within
            # a stereo block.  Exclude a narrow measured image border without
            # synthesising or replacing depth.  Preserve small candidates.
            radius = min(3, max(0, min(width, height) // 8))
            core = cv2.erode(component.astype(np.uint8), np.ones((2*radius+1, 2*radius+1), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        valid_count = int(np.count_nonzero(component & local_valid))
        support = core & local_valid
        n_points = int(np.count_nonzero(support))
        fraction = valid_count / denominator
        audit = {"role": role, "segmentation_source": source, "target_color_srgb": target_color,
                 "region_px": box, "point_count": n_points, "valid_fraction": fraction,
                 "support_region_pixels": denominator, "valid_region_pixels": valid_count,
                 "fit_support_fraction": n_points / denominator,
                 "valid_fraction_basis": "ENCLOSED_COMPONENT_INCLUDING_HOLES" if source == "depth_connected_surface" else "RGB_COMPONENT"}
        if fraction < cfg["minimum_valid_fraction"] or n_points < 80:
            rejected.append(audit | {"reason": "INSUFFICIENT_VALID_DEPTH"})
            continue
        points = _to_left(_points(local_depth, local_valid, camera, core, offset=(x, y)), calibration, role)
        try:
            fit = fit_local_cylinder(points)
        except (ValueError, np.linalg.LinAlgError) as error:
            rejected.append(audit | {"reason": str(error)})
            continue
        median_bgr = np.rint(np.median(local_image[support], axis=0)).astype(np.uint8)
        measured_color = "#%02X%02X%02X" % tuple(map(int, median_bgr[::-1]))
        # Keep at most the export budget for this candidate while the full
        # point array is released at the next iteration.
        fit.update(audit)
        fit.update(measured_color_srgb=measured_color, segmentation_sources=[source],
                   depth_median_mm=float(np.median(local_depth[support])),
                   depth_mad_mm=float(np.median(np.abs(local_depth[support] - np.median(local_depth[support])))),
                   _cloud_points=_sample(points, cfg["maximum_cloud_points"]))
        accepted.append(fit)
    return accepted, rejected


def _axis_point(segment: np.ndarray, common_axis: np.ndarray, station: float) -> np.ndarray:
    direction = _unit(segment[1] - segment[0])
    divisor = float(direction @ common_axis)
    if abs(divisor) < 0.90:
        raise ValueError("AXIS_DIRECTION_CONFLICT")
    return segment[0] + direction * ((station - float(segment[0] @ common_axis)) / divisor)


def _common_geometry(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    aa, bb = _unit(np.asarray(left["axis_direction_world"])), _unit(np.asarray(right["axis_direction_world"]))
    cosine = float(np.clip(abs(aa @ bb), 0, 1))
    if float(aa @ bb) < 0:
        bb = -bb
    axis = _unit(aa + bb)
    if axis[int(np.argmax(np.abs(axis)))] < 0:
        axis = -axis
    sl, sr = np.asarray(left["observed_segment_world_mm"], dtype=float), np.asarray(right["observed_segment_world_mm"], dtype=float)
    low = max(float(np.min(sl @ axis)), float(np.min(sr @ axis)))
    high = min(float(np.max(sl @ axis)), float(np.max(sr @ axis)))
    if high <= low:
        raise ValueError("VIEWS_HAVE_NO_COMMON_AXIAL_SUPPORT")
    stations = (low, (low + high) / 2.0, high)
    pl = np.asarray([_axis_point(sl, axis, t) for t in stations])
    pr = np.asarray([_axis_point(sr, axis, t) for t in stations])
    fused = (pl + pr) / 2.0
    segment = fused[[0, 2]]
    return {"center_camera_mm": fused[1].tolist(), "axis_camera": _unit(segment[1] - segment[0]).tolist(),
            "observed_segment_camera_mm": segment.tolist(), "axial_support_mm": high - low,
            "left_right_center_difference_mm": float(np.max(np.linalg.norm(pl - pr, axis=1))),
            "left_right_axis_angle_deg": math.degrees(math.acos(cosine))}


def _deduplicate(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One observed axis stays one candidate even if RGB and depth found it."""
    result = []
    for candidate in sorted(candidates, key=lambda item: (-item["visible_arc_degrees"], item["fit_rms_mm"])):
        duplicate = None
        for existing in result:
            try:
                geometry = _common_geometry(existing, candidate)
            except ValueError:
                continue
            diameter = min(existing["diameter_mm"], candidate["diameter_mm"])
            if (geometry["left_right_center_difference_mm"] <= min(3.0, diameter * 0.1)
                    and geometry["left_right_axis_angle_deg"] <= 2.0
                    and abs(existing["diameter_mm"] - candidate["diameter_mm"]) <= min(3.0, diameter * 0.1)):
                duplicate = existing
                break
        if duplicate is None:
            result.append(candidate)
        else:
            duplicate["segmentation_sources"] = sorted(set(duplicate["segmentation_sources"] + candidate["segmentation_sources"]))
    return result


def _pair(left: list[dict[str, Any]], right: list[dict[str, Any]], cfg: Mapping[str, Any]) -> tuple[list[tuple], list[dict[str, Any]]]:
    candidates = []
    for li, a in enumerate(left):
        for ri, b in enumerate(right):
            try:
                geometry = _common_geometry(a, b)
            except ValueError:
                continue
            difference = abs(float(a["diameter_mm"]) - float(b["diameter_mm"]))
            fraction = difference / min(float(a["diameter_mm"]), float(b["diameter_mm"]))
            if (geometry["left_right_axis_angle_deg"] > cfg["pair_axis_tolerance_deg"]
                    or geometry["left_right_center_difference_mm"] > cfg["pair_center_tolerance_mm"]
                    or geometry["axial_support_mm"] < cfg["minimum_common_support_mm"]
                    or difference > cfg["pair_diameter_tolerance_mm"]
                    or fraction > cfg["pair_diameter_tolerance_fraction"]):
                continue
            score = geometry["left_right_center_difference_mm"] + difference
            candidates.append((score, li, ri, geometry))
    candidates.sort(key=lambda item: item[:3])
    selected, rejected = [], []
    used_l, used_r = set(), set()
    for score, li, ri, geometry in candidates:
        if li in used_l or ri in used_r:
            continue
        competing = [item[0] for item in candidates if (item[1] == li or item[2] == ri) and item[1:3] != (li, ri)]
        if competing and min(competing) <= score + cfg["pair_ambiguity_margin_mm"]:
            rejected.append({"reason": "AMBIGUOUS_STEREO_COMPONENT_PAIR", "left_index": li, "right_index": ri})
            continue
        used_l.add(li)
        used_r.add(ri)
        selected.append((left[li], right[ri], float(score), geometry))
    for role, fits, used in (("left", left, used_l), ("right", right, used_r)):
        for index, fit in enumerate(fits):
            if index not in used:
                rejected.append({"role": role, "region_px": fit["region_px"], "point_count": fit["point_count"],
                                 "reason": "NO_UNIQUE_LEFT_RIGHT_PAIR"})
    return selected, rejected


def _component_audit(fit: Mapping[str, Any]) -> dict[str, Any]:
    return {key: fit[key] for key in ("role", "region_px", "point_count", "valid_fraction", "fit_rms_mm",
                                     "visible_arc_degrees", "measured_color_srgb", "segmentation_sources", "depth_median_mm", "depth_mad_mm",
                                     "support_region_pixels", "valid_region_pixels", "fit_support_fraction", "valid_fraction_basis",
                                     "diameter_source", "diameter_includes_near_surface") if key in fit}


def extract_local_pipes(left: np.ndarray, right: np.ndarray, depth: Any, calibration: Any,
                        pipe_specs: list[Mapping[str, Any]], *, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Extract paired local cylinders, with no CAD-world identity assignment."""
    arrays = _validate_arrays(left, right, depth, calibration)
    specs, cfg = _catalog(pipe_specs), _settings(config)
    colors = sorted({spec["color_srgb"] for spec in specs})
    candidate_colors = {color: {} for color in colors}
    rejected, view_candidates = [], {}
    depth_audit = {}
    fit_budgets = {}
    for role, image, z, valid in (("left", left, arrays[0], arrays[2]), ("right", right, arrays[1], arrays[3])):
        budget = {"remaining": cfg["maximum_component_candidates"], "attempted": 0}
        # Geometry proposals work even when display colours differ from paint.
        mask = _depth_surface_mask(z, valid, cfg["depth_discontinuity_mm"])
        accepted, errors = _fit_components(image, z, valid, mask, calibration, role, "depth_connected_surface", cfg, budget=budget)
        rejected.extend(errors)
        depth_audit[role] = {"accepted_components": len(accepted), "rejected_components": len(errors),
                             "valid_fraction": float(np.mean(valid & np.isfinite(z) & (z > 0)))}
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        for color in colors[:32]:
            target = cv2.cvtColor(_hex_bgr(color), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
            color_mask = np.linalg.norm(lab - target, axis=2) <= cfg["color_delta_lab"]
            fits, errors = _fit_components(image, z, valid, color_mask, calibration, role, "rgb_component", cfg, color, budget=budget)
            candidate_colors[color][role] = {"accepted_components": len(fits), "rejected_components": len(errors)}
            accepted.extend(fits)
            rejected.extend(errors)
        view_candidates[role] = _deduplicate(accepted)
        fit_budgets[role] = budget
    if len(colors) > 32:
        rejected.append({"reason": "COLOR_CANDIDATE_BUDGET_EXCEEDED", "skipped_colors": len(colors) - 32})
    pairs, pair_errors = _pair(view_candidates["left"], view_candidates["right"], cfg)
    rejected.extend(pair_errors)
    observations = []
    for lf, rf, score, geometry in pairs:
        measured = (float(lf["diameter_mm"]) + float(rf["diameter_mm"])) / 2.0
        # Diameter is the primary physical identity gate.  Keep the same
        # tolerance used by registration; colour remains a segmentation hint
        # and is never allowed to replace a diameter/distance match.
        candidates = [spec for spec in specs if abs(spec["nominal_diameter_mm"] - measured) <= max(3.0, 0.10 * spec["nominal_diameter_mm"])]
        diameter_classes = sorted({spec["nominal_diameter_mm"] for spec in candidates})
        rgb = (np.asarray(_hex_bgr(lf["measured_color_srgb"])[0, 0], dtype=float)
               + np.asarray(_hex_bgr(rf["measured_color_srgb"])[0, 0], dtype=float)) / 2.0
        color = "#%02X%02X%02X" % tuple(map(int, np.rint(rgb[::-1])))
        sources = sorted(set(lf["segmentation_sources"] + rf["segmentation_sources"]))
        observations.append({
            "observation_id": f"OBS-{len(observations) + 1:04d}",
            "candidate_pipe_ids": [spec["pipe_id"] for spec in candidates],
            "nominal_diameter_mm": diameter_classes[0] if len(diameter_classes) == 1 else None,
            "identity_assigned": False, "diameter_mm": measured,
            "color_srgb": color, "measured_color_srgb": color,
            "segmentation_source": "depth_connected_surface" if "depth_connected_surface" in sources else "rgb_component",
            "segmentation_sources": sources,
            "fit_rms_mm": float(max(lf["fit_rms_mm"], rf["fit_rms_mm"])),
            "diameter_source": "ROBUST_OUTER_CYLINDER_FIT_VISIBLE_SURFACE",
            "diameter_includes_near_surface": True,
            "left_region_px": lf["region_px"], "right_region_px": rf["region_px"],
            "point_count": int(lf["point_count"] + rf["point_count"]),
            "left_right_diameter_difference_mm": abs(lf["diameter_mm"] - rf["diameter_mm"]),
            "visible_arc_degrees": float(min(lf["visible_arc_degrees"], rf["visible_arc_degrees"])),
            "pair_score": score, **geometry,
        })
    cloud = _CloudBudget(cfg["maximum_cloud_points"])
    component_audit = []
    for candidates in view_candidates.values():
        for fit in candidates:
            cloud.add(fit["_cloud_points"], fit["measured_color_srgb"], fit["point_count"])
            component_audit.append(_component_audit(fit))
    point_cloud = cloud.public()
    truncated = any(row.get("reason") in {"CANDIDATE_BUDGET_EXCEEDED", "COLOR_CANDIDATE_BUDGET_EXCEEDED"}
                    for row in rejected)
    return {"observations": observations, "rejected": rejected, "point_cloud": point_cloud,
            "audit": {"coordinate_frame": "LEFT_RECTIFIED_CAMERA_MM", "candidate_colors": candidate_colors,
                      "truncated": truncated, "status": "TRUNCATED" if truncated else "COMPLETE",
                      "depth_surfaces": depth_audit, "components": component_audit,
                      "candidate_budgets": fit_budgets,
                      "observation_count": len(observations), "rejected_count": len(rejected),
                      "point_count": len(point_cloud["points_camera_mm"]), "source_point_count": cloud.source_count,
                      "point_cloud_sampling": "BOUNDED_NUMPY_STREAM_BEFORE_JSON",
                      "identity_assigned": False, "registration_performed": False, "settings": cfg}}


__all__ = ["LocalSurfaceError", "extract_local_pipes"]
