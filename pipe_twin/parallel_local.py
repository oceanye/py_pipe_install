"""Local stereo observations for a set of mutually parallel pipes.

The ordinary elevation detector fits a visible cylindrical arc.  That is a
useful high-confidence path, but a board, clamp, or foreground object can
hide the middle of a pipe while leaving a long straight side strip visible.
This module deliberately uses that strip instead: stereo points are fit to a
local 3-D line, and the visible image width at the measured local depth
supplies a conservative diameter estimate.  The result is still only an *observation*; model identity
is assigned later by :mod:`elevation_registration` using diameter and metric
cross-section position.

The implementation is bounded and conservative.  Large colour/background
components are ranked but do not become observations unless they have enough
valid stereo points and a strongly elongated 3-D support.  A single local
strip cannot establish a model pose, so the existing registration layer keeps
such a scene ``INSUFFICIENT_OBSERVATIONS``/``UNKNOWN``.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping

import cv2
import numpy as np

from .local_surface import _CloudBudget, _points, _to_left, _validate_arrays


class ParallelLocalError(ValueError):
    """Malformed stereo input or local-strip settings."""


_DEFAULTS: dict[str, Any] = {
    "color_delta_lab": 72.0,
    "minimum_component_pixels": 120,
    "minimum_valid_points": 80,
    "minimum_valid_fraction": 0.015,
    "minimum_axis_elongation": 2.5,
    "minimum_axis_support_mm": 55.0,
    "axis_consensus_deg": 10.0,
    "pair_y_tolerance_px": 38.0,
    "pair_disparity_tolerance_px": 80.0,
    "maximum_pair_center_error_mm": 35.0,
    "maximum_pair_diameter_difference_fraction": 0.25,
    "maximum_pair_depth_difference_fraction": 0.12,
    "maximum_components_per_color": 32,
    "maximum_total_components": 96,
    "maximum_cloud_points": 6000,
    "maximum_diameter_mm": 200.0,
    "radial_diameter_percentile": 95.0,
}


def _positive(value: Any, field: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ParallelLocalError(f"{field} must be finite and positive")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ParallelLocalError(f"{field} must be finite and positive")
    return value


def _settings(value: Mapping[str, Any] | None) -> dict[str, Any]:
    if value is not None and not isinstance(value, Mapping):
        raise ParallelLocalError("config must be an object")
    unknown = set(value or {}) - set(_DEFAULTS)
    if unknown:
        raise ParallelLocalError(f"Unknown parallel local settings: {sorted(unknown)}")
    cfg = {**_DEFAULTS, **dict(value or {})}
    integer_fields = {"minimum_component_pixels", "minimum_valid_points", "maximum_components_per_color",
                      "maximum_total_components", "maximum_cloud_points"}
    for key, item in cfg.items():
        if key in integer_fields:
            if type(item) is not int or item <= 0:
                raise ParallelLocalError(f"config.{key} must be a positive integer")
        else:
            cfg[key] = _positive(item, f"config.{key}")
    if cfg["color_delta_lab"] > 150:
        raise ParallelLocalError("config.color_delta_lab cannot exceed 150")
    if cfg["minimum_valid_fraction"] > 1:
        raise ParallelLocalError("config.minimum_valid_fraction cannot exceed 1")
    if cfg["radial_diameter_percentile"] >= 100:
        raise ParallelLocalError("config.radial_diameter_percentile must be below 100")
    if cfg["axis_consensus_deg"] > 30 or cfg["axis_consensus_deg"] <= 0:
        raise ParallelLocalError("config.axis_consensus_deg must be in (0, 30]")
    if cfg["maximum_cloud_points"] > 6000:
        raise ParallelLocalError("config.maximum_cloud_points cannot exceed 6000")
    if cfg["maximum_total_components"] > 384:
        raise ParallelLocalError("config.maximum_total_components cannot exceed 384")
    if cfg["maximum_diameter_mm"] <= cfg["minimum_axis_support_mm"] * 0.01:
        raise ParallelLocalError("config.maximum_diameter_mm is too small")
    return cfg


def _hex_rgb(value: Any) -> np.ndarray:
    if not isinstance(value, str) or len(value) not in (7, 9) or value[0] != "#":
        raise ParallelLocalError("pipe color must be #RRGGBB")
    try:
        return np.asarray([int(value[i:i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8)
    except ValueError as exc:
        raise ParallelLocalError("pipe color must be #RRGGBB") from exc


def _color_class(rgb: Any) -> str:
    """Classify paint after exposure normalization; geometry remains primary."""
    values = np.asarray(rgb, dtype=np.float64).reshape(3)
    red, green, blue = values
    maximum, minimum = float(values.max()), float(values.min())
    chroma = maximum - minimum
    if maximum < 1 or chroma <= max(6.0, maximum * 0.16):
        return "WHITE"
    if red >= green * 1.12 and red >= blue * 1.12:
        return "RED"
    if blue >= red * 1.12 and blue >= green * 1.05:
        return "BLUE"
    if green >= red * 1.10 and green >= blue * 1.10:
        return "GREEN"
    if red >= blue * 1.10 and green >= blue * 1.10:
        return "YELLOW"
    return "UNKNOWN"


def _catalog(pipe_specs: Any) -> list[dict[str, Any]]:
    if not isinstance(pipe_specs, list) or not pipe_specs:
        raise ParallelLocalError("pipe_specs must be a non-empty list")
    result = []
    seen: set[str] = set()
    for index, raw in enumerate(pipe_specs):
        if not isinstance(raw, Mapping):
            raise ParallelLocalError(f"pipe_specs[{index}] must be an object")
        pipe_id = raw.get("pipe_id", f"P{index + 1:03d}")
        if not isinstance(pipe_id, str) or not pipe_id.strip() or pipe_id in seen:
            raise ParallelLocalError("pipe_specs pipe_id values must be unique text")
        diameter = raw.get("nominal_diameter_mm", raw.get("diameter_mm"))
        diameter = _positive(diameter, "nominal_diameter_mm")
        color = raw.get("color_srgb", "#FFFFFF")
        rgb = _hex_rgb(color)
        result.append({"pipe_id": pipe_id, "nominal_diameter_mm": diameter,
                       "color_srgb": "#%02X%02X%02X" % tuple(map(int, rgb))})
        seen.add(pipe_id)
    return result


def _unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if vector.shape != (3,) or not math.isfinite(norm) or norm <= 1e-9:
        raise ParallelLocalError("DEGENERATE_LOCAL_AXIS")
    return vector / norm


def _lab_mask(image: np.ndarray, rgb: np.ndarray, delta: float) -> np.ndarray:
    """Return a broad paint mask; geometry and depth perform the hard gates."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(np.asarray([[rgb[::-1]]], dtype=np.uint8), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    distance = np.sqrt(np.sum((lab - target) ** 2, axis=2))
    mask = distance <= float(delta)
    # Exposure and white balance can move a painted surface a long way from
    # the catalogue colour in Lab space (the office pair can be nearly black
    # while still retaining its blue/red channel ordering).  Add a hue/channel
    # dominance mask for chromatic colours; geometry remains the hard gate.
    b, g, r = [channel.astype(np.float32) for channel in cv2.split(image)]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    if int(rgb.max()) - int(rgb.min()) >= 28:
        dominant = int(np.argmax(rgb))
        channels = (r, g, b)  # RGB order
        primary = channels[dominant]
        others = [channels[index] for index in range(3) if index != dominant]
        dominance = (primary >= np.maximum(others[0], others[1]) * 1.04 + 1.5) & (primary >= 8)
        target_hue = int(cv2.cvtColor(np.asarray([[rgb[::-1]]], dtype=np.uint8), cv2.COLOR_BGR2HSV)[0, 0, 0])
        hue_delta = np.abs(hsv[:, :, 0].astype(np.int16) - target_hue)
        hue_delta = np.minimum(hue_delta, 180 - hue_delta)
        # A broad Lab threshold alone makes yellow include red/green under
        # low light.  Keep hue agreement explicit while retaining channel
        # dominance for desaturated dark paint.
        mask = (mask & (hue_delta <= 28)) | (dominance & (hue_delta <= 22))
    # The white/grey class otherwise consumes every bright chessboard square.
    # Keeping a modest neutral/value gate still allows a light pipe while
    # making the later elongated-component test do the scene discrimination.
    if int(rgb.max()) - int(rgb.min()) < 28:
        b, g, r = cv2.split(image.astype(np.int16))
        neutral = (np.max(image, axis=2).astype(np.int16) - np.min(image, axis=2).astype(np.int16)) < 55
        if int(rgb.mean()) > 180:
            mask &= neutral & (np.max(image, axis=2) >= 65)
        else:
            mask &= neutral
    return mask.astype(np.uint8)


def _component_candidates(image: np.ndarray, depth: np.ndarray, valid: np.ndarray,
                          calibration: Any, role: str, color: str, cfg: Mapping[str, Any],
                          target_index: int, budget: dict[str, int] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rgb = _hex_rgb(color)
    mask = _lab_mask(image, rgb, cfg["color_delta_lab"])
    # A one-pixel open removes isolated sensor/chessboard speckles without
    # erasing a pipe side strip.  No closing is used because it could bridge
    # two nearby pipes and create a false common surface.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    eligible = [int(i) for i in range(1, count)
                if int(stats[i, cv2.CC_STAT_AREA]) >= cfg["minimum_component_pixels"]]

    def priority(label: int) -> tuple[float, int, int]:
        area = max(1, int(stats[label, cv2.CC_STAT_AREA]))
        width = max(1, int(stats[label, cv2.CC_STAT_WIDTH]))
        height = max(1, int(stats[label, cv2.CC_STAT_HEIGHT]))
        aspect = max(width, height) / max(1.0, min(width, height))
        # Long, thin candidates are more useful than a large background slab.
        score = math.log1p(area) * (1.0 + 2.0 * math.log1p(aspect))
        return (-score, -area, label)

    ordered = sorted(eligible, key=priority)
    rejected: list[dict[str, Any]] = []
    available = int(cfg["maximum_components_per_color"])
    if budget is not None:
        available = min(available, max(0, int(budget.get("remaining", 0))))
    if len(ordered) > available:
        rejected.append({"role": role, "color_srgb": color, "target_index": target_index,
                         "reason": "PARALLEL_COMPONENT_BUDGET_EXCEEDED",
                         "skipped_components": len(ordered) - available})
    candidates: list[dict[str, Any]] = []
    camera = getattr(calibration, role)
    for label in ordered[:available]:
        if budget is not None:
            budget["remaining"] = max(0, int(budget.get("remaining", 0)) - 1)
        x, y, width, height, area = (int(stats[label, key]) for key in
                                      (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
                                       cv2.CC_STAT_HEIGHT, cv2.CC_STAT_AREA))
        component = labels == label
        yy, xx = np.nonzero(component)
        if len(xx) < cfg["minimum_component_pixels"]:
            continue
        # Fit the visible image strip first.  A high aspect ratio is not
        # mandatory because perspective can compress a horizontal pipe, but
        # its 3-D support below must still be clearly axial.
        pixels = np.column_stack((xx.astype(float), yy.astype(float)))
        _, pixel_s, pixel_vt = np.linalg.svd(pixels - pixels.mean(axis=0), full_matrices=False)
        pixel_axis = pixel_vt[0]
        valid_points_mask = component & valid & np.isfinite(depth) & (depth > 0)
        rows, cols = np.nonzero(valid_points_mask)
        valid_count = len(rows)
        valid_fraction = valid_count / max(1, len(xx))
        audit = {"role": role, "color_srgb": color, "target_index": target_index,
                 "region_px": [x, y, width, height], "component_pixels": int(len(xx)),
                 "valid_points": int(valid_count), "valid_fraction": float(valid_fraction),
                 "pixel_aspect": float(pixel_s[0] / max(pixel_s[1], 1e-9))}
        if valid_count < cfg["minimum_valid_points"] or valid_fraction < cfg["minimum_valid_fraction"]:
            rejected.append(audit | {"reason": "INSUFFICIENT_LOCAL_STEREO_POINTS"})
            continue
        points = _points(depth, valid, camera, component)
        points = _to_left(points, calibration, role)
        if len(points) < cfg["minimum_valid_points"]:
            rejected.append(audit | {"reason": "INSUFFICIENT_LOCAL_STEREO_POINTS"})
            continue
        centred = points - points.mean(axis=0)
        _, singular, vt = np.linalg.svd(centred, full_matrices=False)
        axis = _unit(vt[0])
        elongation = float(singular[0] / max(singular[1], 1e-9))
        support = float(np.ptp(centred @ axis))
        audit.update({"axis_elongation": elongation, "axis_support_mm": support,
                      "singular_values": singular.tolist()})
        if elongation < cfg["minimum_axis_elongation"]:
            rejected.append(audit | {"reason": "LOCAL_AXIS_NOT_ELONGATED"})
            continue
        if support < cfg["minimum_axis_support_mm"]:
            rejected.append(audit | {"reason": "LOCAL_AXIS_SUPPORT_TOO_SHORT"})
            continue
        median_rgb = np.rint(np.median(image[valid_points_mask], axis=0)[::-1]).astype(np.uint8)
        measured_color = "#%02X%02X%02X" % tuple(map(int, median_rgb))
        measured_class = _color_class(median_rgb)
        expected_class = _color_class(rgb)
        audit.update({"measured_color_srgb": measured_color, "measured_color_class": measured_class,
                      "expected_color_class": expected_class})
        if expected_class != "UNKNOWN" and measured_class != expected_class:
            rejected.append(audit | {"reason": "LOCAL_COLOR_CLASS_MISMATCH"})
            continue
        projections = centred @ axis
        radial = np.linalg.norm(centred - np.outer(projections, axis), axis=1)
        # Estimate a visible chord in the image normal direction and a robust
        # outer radius in reconstructed 3-D.  The radial percentile is the
        # primary diameter estimate; the image chord is retained as an
        # independent lower-bound/uncertainty signal.  No fixed scale factor
        # is allowed to manufacture a full diameter from a short strip.
        valid_pixels = np.column_stack((cols.astype(float), rows.astype(float)))
        pixel_centre = valid_pixels.mean(axis=0)
        pixel_normal = np.asarray([-pixel_axis[1], pixel_axis[0]], dtype=float)
        projected_width_px = float(np.ptp(np.percentile((valid_pixels - pixel_centre) @ pixel_normal, [5, 95])))
        projected_width_mm = projected_width_px * float(np.median(depth[valid_points_mask])) / float(camera.fy)
        radial_diameter = 2.0 * float(np.percentile(radial, cfg["radial_diameter_percentile"]))
        diameter = max(radial_diameter, projected_width_mm)
        if not math.isfinite(diameter) or diameter <= 0 or diameter > cfg["maximum_diameter_mm"]:
            rejected.append(audit | {"reason": "LOCAL_DIAMETER_INVALID"})
            continue
        audit.update({"diameter_mm": diameter, "radial_diameter_mm": radial_diameter,
                      "diameter_source": "STEREO_LOCAL_RADIAL_P95_AND_PROJECTED_CHORD",
                      "projected_width_px": projected_width_px,
                      "projected_width_mm": projected_width_mm,
                      "diameter_radial_percentile": float(cfg["radial_diameter_percentile"]),
                      "diameter_uncertainty_mm": max(2.0, abs(radial_diameter - projected_width_mm)),
                      "depth_median_mm": float(np.median(depth[valid_points_mask])),
                      "depth_mad_mm": float(np.median(np.abs(depth[valid_points_mask] - np.median(depth[valid_points_mask]))))})
        candidates.append({"role": role, "color_srgb": color, "target_index": target_index,
                           "region_px": [x, y, width, height], "component_mask": component,
                           "points": points, "center": points.mean(axis=0), "axis": axis,
                           "pixel_axis": pixel_axis, "pixel_center": pixels.mean(axis=0),
                           "diameter_mm": diameter, "measured_color_srgb": measured_color,
                           "measured_color_class": measured_class,
                           "audit": audit})
    return candidates, rejected


def _angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    value = float(np.clip(abs(_unit(first) @ _unit(second)), 0.0, 1.0))
    return math.degrees(math.acos(value))


def _pair_candidates(left: list[dict[str, Any]], right: list[dict[str, Any]],
                     calibration: Any, cfg: Mapping[str, Any]) -> tuple[list[tuple[dict, dict, float]], list[dict]]:
    pairs: list[tuple[dict, dict, float]] = []
    rejected: list[dict] = []
    used: set[int] = set()
    fx = float(getattr(calibration.left, "fx"))
    baseline = float(calibration.baseline_mm)
    # Pair by a predictable stereo score.  Rectification preserves the image
    # row, and the expected horizontal shift is baseline*fx/depth.
    for li, candidate in enumerate(sorted(left, key=lambda item: float(item["pixel_center"][1]))):
        options = []
        for ri, other in enumerate(right):
            if ri in used or candidate["target_index"] != other["target_index"]:
                continue
            y_error = abs(float(candidate["pixel_center"][1] - other["pixel_center"][1]))
            if y_error > cfg["pair_y_tolerance_px"]:
                continue
            disparity = float(candidate["pixel_center"][0] - other["pixel_center"][0])
            depth = max(1.0, float(np.median(candidate["points"][:, 2])))
            expected = fx * baseline / depth
            disparity_error = abs(disparity - expected)
            if disparity_error > cfg["pair_disparity_tolerance_px"]:
                continue
            axis_error = _angle_deg(candidate["axis"], other["axis"])
            score = y_error * 4.0 + disparity_error + axis_error * 2.0
            options.append((score, ri, axis_error, disparity, expected))
        if not options:
            rejected.append({"role": "pair", "region_px": candidate["region_px"],
                             "reason": "NO_STEREO_LOCAL_STRIP_PAIR"})
            continue
        score, ri, axis_error, disparity, expected = min(options)
        other = right[ri]
        used.add(ri)
        if axis_error > cfg["axis_consensus_deg"] + 5:
            rejected.append({"role": "pair", "region_px": candidate["region_px"],
                             "right_region_px": other["region_px"], "reason": "LOCAL_PAIR_AXIS_MISMATCH"})
            continue
        pairs.append((candidate, other, float(score)))
    return pairs, rejected


def _fuse_pair(left: Mapping[str, Any], right: Mapping[str, Any], score: float,
               calibration: Any, observation_index: int) -> dict[str, Any]:
    left_axis = _unit(np.asarray(left["axis"], dtype=float))
    right_axis = _unit(np.asarray(right["axis"], dtype=float))
    if float(left_axis @ right_axis) < 0:
        right_axis = -right_axis
    axis = _unit(left_axis + right_axis)
    if axis[int(np.argmax(np.abs(axis)))] < 0:
        axis = -axis
    centres = np.asarray([left["center"], right["center"]], dtype=float)
    center = centres.mean(axis=0)
    segments = []
    for candidate in (left, right):
        points = np.asarray(candidate["points"], dtype=float)
        station = points @ axis
        segments.append(np.asarray([points[int(np.argmin(station))], points[int(np.argmax(station))]]))
    lows = [float(np.min(segment @ axis)) for segment in segments]
    highs = [float(np.max(segment @ axis)) for segment in segments]
    low, high = max(lows), min(highs)
    if high <= low:
        low, high = min(lows), max(highs)
    observed_segment = np.asarray([center + axis * (low - float(center @ axis)),
                                   center + axis * (high - float(center @ axis))])
    diameter = (float(left["diameter_mm"]) + float(right["diameter_mm"])) / 2.0
    uncertainty = max(float(left["audit"]["diameter_uncertainty_mm"]),
                      float(right["audit"]["diameter_uncertainty_mm"]),
                      abs(float(left["diameter_mm"]) - float(right["diameter_mm"])))
    left_rgb = _hex_rgb(left["measured_color_srgb"]).astype(np.float64)
    right_rgb = _hex_rgb(right["measured_color_srgb"]).astype(np.float64)
    fused_rgb = np.rint((left_rgb + right_rgb) / 2.0).astype(np.uint8)
    measured_color = "#%02X%02X%02X" % tuple(map(int, fused_rgb))
    left_class = str(left.get("measured_color_class", _color_class(left_rgb)))
    right_class = str(right.get("measured_color_class", _color_class(right_rgb)))
    color_consistent = left_class == right_class and left_class != "UNKNOWN"
    depth_left = float(left["audit"]["depth_median_mm"])
    depth_right = float(right["audit"]["depth_median_mm"])
    center_distance = float(np.linalg.norm(center))
    surface_ranges = np.linalg.norm(np.concatenate((left["points"], right["points"]), axis=0), axis=1)
    sources = ["PARALLEL_LOCAL_STRIP", "COLOR_COMPONENT", "STEREO_DEPTH"]
    return {
        "observation_id": f"OBS-{observation_index:04d}",
        "candidate_pipe_ids": [], "nominal_diameter_mm": None, "identity_assigned": False,
        "diameter_mm": diameter, "diameter_uncertainty_mm": uncertainty,
        "diameter_source": "STEREO_LOCAL_RADIAL_P95_AND_PROJECTED_CHORD",
        "diameter_includes_near_surface": True,
        "color_srgb": measured_color, "measured_color_srgb": measured_color,
        "measured_color_class": left_class if color_consistent else "MIXED",
        "left_color_class": left_class, "right_color_class": right_class,
        "color_consistent": color_consistent, "color_identity_validated": False,
        "segmentation_source": "parallel_local_strip",
        "segmentation_sources": sources, "fit_rms_mm": uncertainty,
        "left_region_px": left["region_px"], "right_region_px": right["region_px"],
        "point_count": int(len(left["points"]) + len(right["points"])),
        "left_right_diameter_difference_mm": abs(float(left["diameter_mm"]) - float(right["diameter_mm"])),
        "visible_arc_degrees": None, "parallel_local": True,
        "distance_geometry": {"center_camera_mm": center.tolist(), "axis_camera": axis.tolist(),
                               "diameter_mm": diameter, "section_definition": "LOCAL_PARALLEL_STRIP",
                               "center_range_mm": center_distance,
                               "nearest_visible_surface_range_mm": float(np.percentile(surface_ranges, 5)),
                               "left_depth_z_median_mm": depth_left,
                               "right_depth_z_median_mm": depth_right},
        "surface_samples": {"left": {"depth_z_median_mm": left["audit"]["depth_median_mm"],
                                       "depth_z_mad_mm": left["audit"]["depth_mad_mm"],
                                       "point_count": len(left["points"]),
                                       "definition": "LOCAL_PARALLEL_STRIP_DEPTH_PIXELS"},
                            "right": {"depth_z_median_mm": right["audit"]["depth_median_mm"],
                                       "depth_z_mad_mm": right["audit"]["depth_mad_mm"],
                                       "point_count": len(right["points"]),
                                       "definition": "LOCAL_PARALLEL_STRIP_DEPTH_PIXELS"}},
        "pair_score": score, "center_camera_mm": center.tolist(), "axis_camera": axis.tolist(),
        "observed_segment_camera_mm": observed_segment.tolist(),
        "axial_support_mm": max(0.0, high - low),
        "left_right_center_difference_mm": float(np.linalg.norm(centres[0] - centres[1])),
        "left_right_axis_angle_deg": _angle_deg(left_axis, right_axis),
        "left_right_depth_difference_mm": abs(depth_left - depth_right),
    }


def extract_parallel_local_pipes(left: np.ndarray, right: np.ndarray, depth: Any, calibration: Any,
                                 pipe_specs: list[dict], config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Extract local straight-strip observations from a rectified stereo pair."""
    cfg = _settings(config)
    _validate_arrays(left, right, depth, calibration)
    specs = _catalog(pipe_specs)
    colours: list[str] = []
    for spec in specs:
        if spec["color_srgb"] not in colours:
            colours.append(spec["color_srgb"])
    rejected: list[dict[str, Any]] = []
    views: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    budgets = {role: {"remaining": int(cfg["maximum_total_components"])} for role in ("left", "right")}
    for index, colour in enumerate(colours):
        for role in ("left", "right"):
            image = left if role == "left" else right
            z = getattr(depth, f"{role}_depth_mm")
            valid = getattr(depth, f"{role}_valid")
            accepted, errors = _component_candidates(image, z, valid, calibration, role, colour, cfg, index,
                                                     budgets[role])
            views[role].extend(accepted)
            rejected.extend(errors)
    pairs, pair_errors = _pair_candidates(views["left"], views["right"], calibration, cfg)
    rejected.extend(pair_errors)
    observations: list[dict[str, Any]] = []
    for left_candidate, right_candidate, score in pairs:
        observation = _fuse_pair(left_candidate, right_candidate, score, calibration, len(observations) + 1)
        pair_diameter_delta = float(observation["left_right_diameter_difference_mm"])
        pair_diameter_limit = max(6.0, float(cfg["maximum_pair_diameter_difference_fraction"]) * float(observation["diameter_mm"]))
        pair_depth_delta = float(observation["left_right_depth_difference_mm"])
        pair_depth_scale = max(float(observation["distance_geometry"]["left_depth_z_median_mm"]),
                               float(observation["distance_geometry"]["right_depth_z_median_mm"]), 1.0)
        if float(observation["left_right_center_difference_mm"]) > float(cfg["maximum_pair_center_error_mm"]):
            rejected.append({"role": "pair", "left_region_px": left_candidate["region_px"],
                             "right_region_px": right_candidate["region_px"],
                             "center_error_mm": float(observation["left_right_center_difference_mm"]),
                             "reason": "STEREO_LOCAL_CENTER_DISAGREEMENT"})
            continue
        if pair_diameter_delta > pair_diameter_limit:
            rejected.append({"role": "pair", "left_region_px": left_candidate["region_px"],
                             "right_region_px": right_candidate["region_px"],
                             "diameter_difference_mm": pair_diameter_delta,
                             "reason": "STEREO_LOCAL_DIAMETER_DISAGREEMENT"})
            continue
        if pair_depth_delta / pair_depth_scale > float(cfg["maximum_pair_depth_difference_fraction"]):
            rejected.append({"role": "pair", "left_region_px": left_candidate["region_px"],
                             "right_region_px": right_candidate["region_px"],
                             "depth_difference_mm": pair_depth_delta,
                             "reason": "STEREO_LOCAL_DEPTH_DISAGREEMENT"})
            continue
        if not observation["color_consistent"]:
            rejected.append({"role": "pair", "left_region_px": left_candidate["region_px"],
                             "right_region_px": right_candidate["region_px"],
                             "left_color_class": observation["left_color_class"],
                             "right_color_class": observation["right_color_class"],
                             "reason": "STEREO_LOCAL_COLOR_DISAGREEMENT"})
            continue
        # Use the DXF/STL diameters as a screening gate, never as a measured
        # value.  A huge red/blue background slab can form a straight 3-D
        # direction, but its projected width is outside every model pipe and
        # must not be allowed to create a second common-axis group.
        compatible = [spec for spec in specs
                      if abs(float(spec["nominal_diameter_mm"]) - float(observation["diameter_mm"]))
                      <= max(6.0, 0.25 * float(spec["nominal_diameter_mm"]))]
        if not compatible:
            rejected.append({"role": "pair", "left_region_px": left_candidate["region_px"],
                             "right_region_px": right_candidate["region_px"],
                             "diameter_mm": float(observation["diameter_mm"]),
                             "reason": "LOCAL_DIAMETER_OUTSIDE_MODEL_RANGE"})
            continue
        observation["candidate_pipe_ids"] = [spec["pipe_id"] for spec in compatible]
        color_compatible = [spec for spec in compatible
                            if _color_class(_hex_rgb(spec["color_srgb"])) == observation["measured_color_class"]]
        if color_compatible:
            observation["color_candidate_pipe_ids"] = [spec["pipe_id"] for spec in color_compatible]
        else:
            observation["color_candidate_pipe_ids"] = []
        diameters = sorted({float(spec["nominal_diameter_mm"]) for spec in compatible})
        observation["nominal_diameter_mm"] = diameters[0] if len(diameters) == 1 else None
        observations.append(observation)

    # The field assertion says the pipes are mutually parallel.  Remove an
    # isolated clutter direction before registration; retaining it would make
    # the registration layer correctly report MULTIPLE_COMMON_AXIS_GROUPS but
    # would hide the useful visible strip in the audit.
    if observations:
        axes = np.asarray([o["axis_camera"] for o in observations], dtype=float)
        support = np.asarray([o["axial_support_mm"] for o in observations], dtype=float)
        seed = int(np.argmax(support))
        common = _unit(axes[seed])
        keep = np.abs(axes @ common) >= math.cos(math.radians(cfg["axis_consensus_deg"]))
        for index, accepted in enumerate(observations):
            if not bool(keep[index]):
                rejected.append({"observation_id": accepted["observation_id"],
                                 "reason": "LOCAL_AXIS_OUTSIDE_PARALLEL_CONSENSUS"})
        observations = [item for index, item in enumerate(observations) if bool(keep[index])]
        # Re-number only after filtering so the exported IDs remain unique and
        # deterministic.  The original pair ID is preserved in the audit.
        for index, item in enumerate(observations, 1):
            item["observation_id"] = f"OBS-{index:04d}"

    cloud = _CloudBudget(cfg["maximum_cloud_points"])
    component_audit = []
    for view in views.values():
        for candidate in view:
            cloud.add(candidate["points"], candidate["measured_color_srgb"], len(candidate["points"]))
            component_audit.append(candidate["audit"])
    point_cloud = cloud.public()
    rejection_counts = dict(Counter(str(item.get("reason", "UNKNOWN")) for item in rejected))
    color_class_counts = dict(Counter(str(item.get("measured_color_class", "UNKNOWN"))
                                      for item in component_audit if item.get("measured_color_class")))
    return {
        "observations": observations, "rejected": rejected, "point_cloud": point_cloud,
        "audit": {
            "coordinate_frame": "LEFT_RECTIFIED_CAMERA_MM",
            "observation_type": "PARALLEL_LOCAL_STRIP",
            "status": ("TRUNCATED_LOCAL_SEARCH" if any(item.get("reason") == "PARALLEL_COMPONENT_BUDGET_EXCEEDED" for item in rejected)
                       else "PARTIAL_LOCAL_SEARCH" if observations else "NO_LOCAL_STRIP_OBSERVATIONS"),
            "truncated": any(item.get("reason") == "PARALLEL_COMPONENT_BUDGET_EXCEEDED" for item in rejected),
            "candidate_colors": colours,
            "components": component_audit, "observation_count": len(observations),
            "rejected_count": len(rejected), "point_count": len(point_cloud["points_camera_mm"]),
            "rejection_counts": rejection_counts, "measured_color_class_counts": color_class_counts,
            "identity_assigned": False, "registration_performed": False,
            "occlusion_diagnostics": {"middle_section_may_be_occluded": True,
                                       "local_visible_regions_only": True},
            "settings": cfg,
            "candidate_budgets": budgets,
        },
    }


__all__ = ["ParallelLocalError", "extract_parallel_local_pipes"]
