"""Local, observed cylinder geometry and pairwise distances, in millimetres.

CAD chooses a search neighbourhood and an identity; it never supplies the
radius of the fitted cylinder.  Each view must independently support a curved
surface and agree with the other view.  These are measurement quality gates,
not a certificate of field accuracy.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any, Mapping

import cv2
import numpy as np


def measurement_settings(payload: Mapping[str, Any] | None = None) -> dict:
    result = {
        "tolerance_mm": 1.0,
        "maximum_fit_rms_mm": 1.5,
        "maximum_view_difference_mm": 3.0,
        "minimum_arc_degrees": 70.0,
        "association_distance_mm": 80.0,
        "rois": {},
    }
    if payload is not None:
        if not isinstance(payload, Mapping):
            raise ValueError("measurement_settings must be an object")
        unknown = set(payload) - set(result)
        if unknown:
            raise ValueError(f"Unknown measurement settings: {sorted(unknown)}")
        result.update(payload)
    for key in result.keys() - {"rois"}:
        value = result[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        result[key] = float(value)
    if not 30 <= result["minimum_arc_degrees"] <= 180:
        raise ValueError("minimum_arc_degrees must be between 30 and 180")
    if not isinstance(result["rois"], dict):
        raise ValueError("rois must be an object")
    for pipe_id, views in result["rois"].items():
        if not isinstance(pipe_id, str) or not isinstance(views, dict):
            raise ValueError("ROI needs a pipe identity and view rectangles")
        for role, box in views.items():
            if role not in {"left", "right"} or not isinstance(box, (list, tuple)) or len(box) != 4:
                raise ValueError("ROI must be left/right: [x, y, width, height]")
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
                raise ValueError("ROI coordinates must be finite numbers")
            if min(box[:2]) < 0 or min(box[2:]) < 8:
                raise ValueError("ROI must be inside the image and at least 8 pixels wide/high")
    return result


def _unit(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-9:
        raise ValueError("DEGENERATE_AXIS")
    return vector / length


def _basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = _unit(np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))]))
    return first, np.cross(axis, first)


def fit_local_cylinder(points_mm: np.ndarray, settings: Mapping | None = None) -> dict:
    """Fit five cylinder parameters without using a nominal radius/axis.

    A numerical robust least-squares refinement follows a PCA/circle seed.
    Split-section stability and angular coverage reject flat/partial surfaces.
    The returned segment is only the observed axial support, never CAD length.
    """
    config = measurement_settings(settings)
    points = np.asarray(points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 80 or not np.isfinite(points).all():
        raise ValueError("INSUFFICIENT_VALID_3D_POINTS")
    if len(points) > 2400:
        points = points[np.linspace(0, len(points) - 1, 2400, dtype=int)]
    origin = np.mean(points, axis=0)
    q = points - origin
    _, _, vectors = np.linalg.svd(q, full_matrices=False)
    axis0 = vectors[0]
    b1, b2 = _basis(axis0)
    uv = np.column_stack((q @ b1, q @ b2))
    circle_matrix = np.column_stack((2 * uv, np.ones(len(q))))
    seed, _, rank, _ = np.linalg.lstsq(circle_matrix, np.sum(uv * uv, axis=1), rcond=None)
    radius_squared = seed[2] + np.dot(seed[:2], seed[:2])
    if rank < 3 or radius_squared <= 0:
        raise ValueError("SURFACE_CURVATURE_NOT_RESOLVED")
    params = np.array([seed[0], seed[1], 0.0, 0.0, math.sqrt(radius_squared)])

    def geometry(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return origin + p[0] * b1 + p[1] * b2, _unit(axis0 + p[2] * b1 + p[3] * b2)

    def residual(p: np.ndarray) -> np.ndarray:
        center, axis = geometry(p)
        delta = points - center
        radial = delta - np.outer(delta @ axis, axis)
        return np.linalg.norm(radial, axis=1) - p[4]

    damping = 1e-3
    steps = np.array([0.001, 0.001, 1e-5, 1e-5, 0.001])
    for _ in range(35):
        errors = residual(params)
        scale = max(0.05, 1.4826 * float(np.median(np.abs(errors - np.median(errors)))))
        weights = np.minimum(1.0, 1.5 * scale / np.maximum(np.abs(errors), 1e-9))
        jacobian = np.column_stack([
            (residual(params + np.eye(5)[i] * step) - errors) / step
            for i, step in enumerate(steps)
        ])
        normal = jacobian.T @ (weights[:, None] * jacobian)
        try:
            change = np.linalg.solve(
                normal + damping * np.diag(np.maximum(np.diag(normal), 1e-6)),
                -(jacobian.T @ (weights * errors)),
            )
        except np.linalg.LinAlgError as error:
            raise ValueError("UNSTABLE_CYLINDER_FIT") from error
        candidate = params + change
        if candidate[4] > 0 and np.sum(weights * residual(candidate) ** 2) < np.sum(weights * errors ** 2):
            params = candidate
            damping = max(damping / 3, 1e-9)
            if np.linalg.norm(change) < 1e-6:
                break
        else:
            damping *= 10
    center, axis = geometry(params)
    errors = residual(params)
    inliers = np.abs(errors) <= max(0.1, 3 * float(np.median(np.abs(errors))))
    if np.mean(inliers) < 0.75:
        raise ValueError("SURFACE_OUTLIERS")
    rms = float(np.sqrt(np.mean(errors[inliers] ** 2)))
    if rms > min(config["maximum_fit_rms_mm"], float(params[4]) * 0.05):
        raise ValueError("CYLINDER_FIT_RESIDUAL_TOO_LARGE")
    diameter = float(2 * params[4])
    offsets = points[inliers] - center
    axial = offsets @ axis
    low, high = np.quantile(axial, [0.02, 0.98])
    if high - low < max(40.0, 2 * diameter):
        raise ValueError("LOCAL_AXIS_SUPPORT_TOO_SHORT")
    b1, b2 = _basis(axis)
    angles = np.sort(np.mod(np.arctan2(offsets @ b2, offsets @ b1), 2 * math.pi))
    arc = float(np.degrees(2 * math.pi - np.max(np.diff(np.r_[angles, angles[0] + 2 * math.pi]))))
    if arc < config["minimum_arc_degrees"]:
        raise ValueError("INSUFFICIENT_VISIBLE_CYLINDER_ARC")
    section_radii = []
    for indexes in np.array_split(np.argsort(axial), 3):
        section_uv = np.column_stack((offsets[indexes] @ b1, offsets[indexes] @ b2))
        matrix = np.column_stack((2 * section_uv, np.ones(len(indexes))))
        solution, _, section_rank, _ = np.linalg.lstsq(matrix, np.sum(section_uv ** 2, axis=1), rcond=None)
        square = solution[2] + np.dot(solution[:2], solution[:2])
        if section_rank < 3 or square <= 0:
            raise ValueError("UNSTABLE_SECTION_FIT")
        section_radii.append(math.sqrt(square))
    spread = float(2 * np.ptp(section_radii))
    if spread > min(config["maximum_view_difference_mm"], diameter * 0.1):
        raise ValueError("DIAMETER_VARIES_ACROSS_LOCAL_SECTIONS")
    return {
        "diameter_mm": diameter,
        "center_world_mm": (center + axis * ((low + high) / 2)).tolist(),
        "axis_direction_world": axis.tolist(),
        "observed_segment_world_mm": [(center + axis * low).tolist(), (center + axis * high).tolist()],
        "fit_rms_mm": rms,
        "diameter_section_spread_mm": spread,
        "visible_arc_degrees": arc,
        "point_count": int(len(points)),
    }


def _axis_point(segment: np.ndarray, direction: np.ndarray, station: float) -> np.ndarray:
    axis = _unit(segment[1] - segment[0])
    divisor = np.dot(axis, direction)
    if abs(divisor) < 0.5:
        raise ValueError("AXIS_DIRECTION_CONFLICT")
    return segment[0] + axis * ((station - np.dot(segment[0], direction)) / divisor)


def pair_geometry(first: Mapping, second: Mapping, camera: Any, tolerance_mm: float = 1.0) -> dict:
    """Distance of local axes and cylindrical surface clearance at that location.

    Near-parallel axes use the middle of their common axial support (a named
    common section).  Skew axes use closest approach only inside BOTH observed
    segments.  Depth order is relative to the left camera at this same location.
    """
    result = {"pipe_id_a": first["pipe_id"], "pipe_id_b": second["pipe_id"], "status": "UNKNOWN"}
    if first.get("status") != "MEASURED" or second.get("status") != "MEASURED":
        return result | {"reason_codes": ["PAIR_REQUIRES_TWO_MEASURED_PIPES"]}
    sa = np.asarray(first["observed_segment_world_mm"], dtype=float)
    sb = np.asarray(second["observed_segment_world_mm"], dtype=float)
    a, b = _unit(sa[1] - sa[0]), _unit(sb[1] - sb[0])
    if np.dot(a, b) < 0:
        b = -b
    cosine = float(np.clip(np.dot(a, b), -1, 1))
    angle = math.degrees(math.acos(cosine))
    if angle <= 1.0:
        direction = _unit(a + b)
        lo = max(min(sa @ direction), min(sb @ direction))
        hi = min(max(sa @ direction), max(sb @ direction))
        if hi <= lo:
            return result | {"reason_codes": ["NO_COMMON_OBSERVED_SECTION"]}
        station = (lo + hi) / 2
        pa, pb = _axis_point(sa, direction, station), _axis_point(sb, direction, station)
        definition = "COMMON_LOCAL_SECTION_NEAR_PARALLEL_AXES"
    else:
        delta = sb[0] - sa[0]
        parameters = np.linalg.lstsq(np.column_stack((a, -b)), delta, rcond=None)[0]
        pa, pb = sa[0] + a * parameters[0], sb[0] + b * parameters[1]
        for point, segment, axis in ((pa, sa, a), (pb, sb, b)):
            station = float(point @ axis)
            if not min(segment @ axis) <= station <= max(segment @ axis):
                return result | {"reason_codes": ["CLOSEST_APPROACH_OUTSIDE_OBSERVED_REGION"]}
        definition = "LOCAL_AXIS_CLOSEST_APPROACH"
    distance = float(np.linalg.norm(pb - pa))
    gap = distance - (first["diameter_mm"] + second["diameter_mm"]) / 2
    rect_rotation = getattr(
        camera, "rotation_world_to_rectified_camera", camera.rotation_world_to_camera
    )
    za = float((rect_rotation @ (pa - camera.center_world_mm))[2])
    zb = float((rect_rotation @ (pb - camera.center_world_mm))[2])
    order_threshold = max(tolerance_mm, first.get("view_position_difference_mm", 0), second.get("view_position_difference_mm", 0))
    front = None if abs(zb - za) <= order_threshold else first["pipe_id"] if za < zb else second["pipe_id"]
    return result | {
        "status": "MEASURED", "center_distance_mm": distance, "clear_gap_mm": gap,
        "depth_delta_b_minus_a_mm": zb - za, "front_pipe_id": front,
        "depth_order": "DEPTH_TOO_CLOSE" if front is None else "RESOLVED",
        "depth_order_threshold_mm": order_threshold,
        "section_points_world_mm": [pa.tolist(), pb.tolist()], "distance_definition": definition,
        "axes_angle_degrees": angle, "reason_codes": ["NEGATIVE_GAP_CHECK_GEOMETRY"] if gap < 0 else [],
    }


def _measure_view(pipe: Any, projection: Any, image_lab: np.ndarray, depth: np.ndarray,
                  valid: np.ndarray, camera: Any, config: dict, role: str, color_tolerance: float) -> dict:
    if projection.centerline_px is None:
        raise ValueError("NO_PROJECTED_LOCAL_AXIS")
    line = np.asarray(projection.centerline_px)
    axis_px = _unit(line[1] - line[0])
    normal_px = np.array([-axis_px[1], axis_px[0]])
    middle = np.mean(line, axis=0)
    diameter_px = projection.predicted_diameter_px or 10.0
    half_length = min(float(np.linalg.norm(line[1] - line[0])) * 0.35, max(100.0, 4 * diameter_px))
    half_width = max(32.0, 2.5 * diameter_px)
    corners = np.array([middle + a * half_length * axis_px + b * half_width * normal_px for a in (-1, 1) for b in (-1, 1)])
    custom = config["rois"].get(pipe.pipe_id, {}).get(role)
    h, w = depth.shape
    if custom is not None:
        x, y, rw, rh = custom
        if x + rw > w or y + rh > h:
            raise ValueError("ROI_OUTSIDE_IMAGE")
        x0, y0, x1, y1 = int(x), int(y), int(x + rw), int(y + rh)
    else:
        x0, y0 = np.maximum(np.floor(corners.min(axis=0)), 0).astype(int)
        x1, y1 = np.minimum(np.ceil(corners.max(axis=0)) + 1, [w, h]).astype(int)
    if x1 - x0 < 8 or y1 - y0 < 8:
        raise ValueError("LOCAL_REGION_OUT_OF_FRAME")
    yy, xx = np.indices((y1 - y0, x1 - x0))
    pixels = np.stack((xx + x0, yy + y0), axis=-1)
    offsets = pixels - middle
    search = np.ones(xx.shape, dtype=bool) if custom is not None else ((np.abs(offsets @ axis_px) <= half_length) & (np.abs(offsets @ normal_px) <= half_width))
    rgb = np.array([[[int(pipe.color_srgb[i:i + 2], 16) / 255 for i in (1, 3, 5)]]], dtype=np.float32)
    color = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[0, 0]
    support = search & (np.linalg.norm(image_lab[y0:y1, x0:x1] - color, axis=2) <= color_tolerance)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(support.astype(np.uint8), 8)
    candidates = []
    reasons = []
    design_axis = _unit(pipe.centerline_world_mm[1] - pipe.centerline_world_mm[0])
    design_middle = np.mean(pipe.centerline_world_mm, axis=0)
    for label in sorted(range(1, count), key=lambda i: int(stats[i, cv2.CC_STAT_AREA]), reverse=True)[:8]:
        mask = (labels == label) & valid[y0:y1, x0:x1]
        rows, cols = np.nonzero(mask)
        if len(rows) < 80:
            continue
        z = depth[y0 + rows, x0 + cols]
        intrinsic = getattr(camera, "rectified_intrinsic", camera.intrinsic)
        points_camera = np.column_stack(((cols + x0 - intrinsic[0, 2]) * z / camera.fx,
                                        (rows + y0 - intrinsic[1, 2]) * z / camera.fy, z))
        rect_rotation = getattr(
            camera, "rotation_world_to_rectified_camera", camera.rotation_world_to_camera
        )
        points = points_camera @ rect_rotation + camera.center_world_mm
        try:
            fit = fit_local_cylinder(points, config)
            center = np.array(fit["center_world_mm"])
            difference = center - design_middle
            distance = float(np.linalg.norm(difference - design_axis * np.dot(difference, design_axis)))
            if distance > config["association_distance_mm"]:
                raise ValueError("OBSERVED_AXIS_OUTSIDE_ASSOCIATION_RANGE")
            if abs(np.dot(fit["axis_direction_world"], design_axis)) < math.cos(math.radians(15)):
                raise ValueError("OBSERVED_AXIS_DIRECTION_MISMATCH")
            fit["roi_xywh"] = [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]
            fit["support_bbox_xywh"] = [int(x0 + cols.min()), int(y0 + rows.min()), int(np.ptp(cols) + 1), int(np.ptp(rows) + 1)]
            candidates.append((distance, fit))
        except ValueError as error:
            reasons.append(str(error))
    candidates.sort(key=lambda item: item[0])
    if not candidates:
        raise ValueError(reasons[0] if reasons else "INSUFFICIENT_LOCAL_COLOR_AND_DEPTH_SUPPORT")
    if len(candidates) > 1 and candidates[1][0] - candidates[0][0] <= 2 * config["maximum_view_difference_mm"]:
        raise ValueError("AMBIGUOUS_LOCAL_PIPE_IDENTITY")
    return candidates[0][1]


def analyze_local_geometry(pipes: Any, projections: Mapping, images: Mapping, depth: Any,
                           calibration: Any, healthy: bool, settings: Mapping | None = None,
                           color_tolerance: float = 50.0) -> dict:
    config = measurement_settings(settings)
    labs = {role: cv2.cvtColor(image.astype(np.float32) / 255, cv2.COLOR_BGR2LAB) for role, image in images.items()}
    results = []
    for pipe in pipes:
        row = {"pipe_id": pipe.pipe_id, "status": "UNKNOWN", "reason_codes": [],
               "nominal_diameter_mm": pipe.nominal_diameter_mm, "views": {}}
        if not healthy or not calibration.validated or not calibration.registration_validated:
            row["reason_codes"].append("CAPTURE_OR_CALIBRATION_NOT_READY")
            results.append(row)
            continue
        for role in ("left", "right"):
            try:
                row["views"][role] = _measure_view(pipe, projections[role][pipe.pipe_id], labs[role],
                    getattr(depth, f"{role}_depth_mm"), getattr(depth, f"{role}_valid"),
                    getattr(calibration, role), config, role, color_tolerance)
            except (ValueError, np.linalg.LinAlgError) as error:
                row["reason_codes"].append(f"{role.upper()}:{error}")
        if len(row["views"]) == 2:
            left, right = row["views"]["left"], row["views"]["right"]
            direction = _unit(np.array(left["axis_direction_world"]))
            ls, rs = np.array(left["observed_segment_world_mm"]), np.array(right["observed_segment_world_mm"])
            lo, hi = max(min(ls @ direction), min(rs @ direction)), min(max(ls @ direction), max(rs @ direction))
            if hi <= lo:
                row["reason_codes"].append("VIEWS_HAVE_NO_COMMON_SECTION")
            else:
                center_l, center_r = _axis_point(ls, direction, (lo + hi) / 2), _axis_point(rs, direction, (lo + hi) / 2)
                difference = float(np.linalg.norm(center_l - center_r))
                diameter_difference = abs(left["diameter_mm"] - right["diameter_mm"])
                if max(difference, diameter_difference) > config["maximum_view_difference_mm"]:
                    row["reason_codes"].append("LEFT_RIGHT_METRIC_DISAGREEMENT")
                elif abs(np.dot(direction, right["axis_direction_world"])) < math.cos(math.radians(2)):
                    row["reason_codes"].append("LEFT_RIGHT_AXIS_DISAGREEMENT")
                else:
                    segment = [((_axis_point(ls, direction, t) + _axis_point(rs, direction, t)) / 2).tolist() for t in (lo, hi)]
                    center = (center_l + center_r) / 2
                    row.update(status="MEASURED", diameter_mm=(left["diameter_mm"] + right["diameter_mm"]) / 2,
                        center_world_mm=center.tolist(), observed_segment_world_mm=segment,
                        axis_direction_world=_unit(np.diff(segment, axis=0)[0]).tolist(),
                        camera_depth_mm=float((calibration.left.rotation_world_to_rectified_camera @ (center - calibration.left.center_world_mm))[2]),
                        view_position_difference_mm=difference, view_diameter_difference_mm=diameter_difference)
        results.append(row)
    # One observed local axis must not be counted for two CAD identities.
    ambiguous = set()
    for a, b in combinations(results, 2):
        if a["status"] != "MEASURED" or b["status"] != "MEASURED":
            continue
        relation = pair_geometry(a, b, calibration.left, config["tolerance_mm"])
        if relation.get("center_distance_mm", math.inf) < min(a["diameter_mm"], b["diameter_mm"]) * 0.5:
            ambiguous.update((a["pipe_id"], b["pipe_id"]))
    for item in results:
        if item["pipe_id"] in ambiguous:
            item["status"] = "UNKNOWN"
            item["reason_codes"].append("SHARED_OR_OVERLAPPING_OBSERVATION")
    pairs = [pair_geometry(a, b, calibration.left, config["tolerance_mm"]) for a, b in combinations(results, 2)]
    return {"schema_version": "1.0", "method": "LOCAL_BIDIRECTIONAL_STEREO_CYLINDER_FIT",
            "coordinate_frame": "CAD_WORLD_MM", "accuracy_validated": False,
            "settings": config, "pipes": results, "pairs": pairs}
