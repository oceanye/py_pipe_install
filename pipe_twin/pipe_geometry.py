"""Explicit distance references for a circular pipe at one named cross-section."""
from __future__ import annotations

import math
from typing import Any

import numpy as np

GEOMETRY_REVISION = "pipe-section-distances-v1"


def _vector(value: Any, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must be a finite 3-D vector") from error
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3-D vector")
    return result


def cylinder_section_geometry(center_camera_mm: Any, axis_camera: Any, diameter_mm: float,
                              *, section_definition: str, intrinsic: Any = None) -> dict:
    """Distances relative to the left rectified optical centre, in millimetres.

    Every value refers to the SAME plane normal to the cylinder axis through
    ``center_camera_mm``.  This is neither the whole pipe's closest point nor
    a median of visible depth pixels.  Tangent points describe the silhouette
    of the lateral cylinder (end caps and occlusion are not inferred).
    """
    center = _vector(center_camera_mm, "center_camera_mm")
    axis = _vector(axis_camera, "axis_camera")
    length = float(np.linalg.norm(axis))
    if not math.isfinite(length) or length <= 1e-9:
        raise ValueError("axis_camera must be non-zero")
    axis = axis / length
    if type(diameter_mm) not in (int, float) or not math.isfinite(diameter_mm) or diameter_mm <= 0:
        raise ValueError("diameter_mm must be finite and positive")
    if not isinstance(section_definition, str) or not section_definition:
        raise ValueError("section_definition is required")
    radius = float(diameter_mm) / 2
    k = None
    if intrinsic is not None:
        k = np.asarray(intrinsic, dtype=float)
        if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0 or not np.allclose(k[2], [0, 0, 1]):
            raise ValueError("intrinsic must be a finite pinhole matrix")

    def point_record(point):
        result = {"point_camera_mm": point.tolist(), "depth_z_mm": float(point[2]),
                  "range_mm": float(np.linalg.norm(point))}
        if k is not None:
            projected = k @ point
            result["pixel_xy"] = (projected[:2] / projected[2]).tolist()
        return result

    # Z extent is radius projected onto the image-depth axis; it is not
    # always radius itself when the pipe is tilted relative to the camera.
    z_radial = np.array([0., 0., 1.]) - axis[2] * axis
    z_length = float(np.linalg.norm(z_radial))
    front_z = float(center[2] - radius * z_length)
    if front_z <= 0:
        raise ValueError("pipe section must be wholly in front of the camera")
    front = center - radius * z_radial / z_length if z_length > 1e-9 else None
    transverse = center - axis * float(center @ axis)
    distance_to_axis = float(np.linalg.norm(transverse))
    nearest = center - radius * transverse / distance_to_axis if distance_to_axis > 1e-9 else None
    nearest_range = math.hypot(float(center @ axis), distance_to_axis - radius)
    tangent_points = []
    if distance_to_axis > radius:
        direction = transverse / distance_to_axis
        side = np.cross(axis, direction)
        inward = -radius * radius / distance_to_axis * direction
        sideways = radius * math.sqrt(max(0., 1 - (radius / distance_to_axis) ** 2)) * side
        tangent_points = [point_record(center + inward + sign * sideways) for sign in (-1, 1)]
        if k is not None:
            tangent_points.sort(key=lambda item: tuple(reversed(item["pixel_xy"])))
    return {
        "revision": GEOMETRY_REVISION, "coordinate_frame": "LEFT_RECTIFIED_CAMERA_MM",
        "section_definition": section_definition, "diameter_mm": float(diameter_mm),
        "axis_camera": axis.tolist(), "centerline": point_record(center),
        "minimum_surface_depth_z_mm": front_z,
        "maximum_surface_depth_z_mm": float(center[2] + radius * z_length),
        "minimum_depth_point": point_record(front) if front is not None else None,
        "nearest_surface_range_mm": nearest_range,
        "nearest_surface_point": point_record(nearest) if nearest is not None else None,
        "silhouette_tangent_points": tangent_points,
        "silhouette_status": "DEFINED" if tangent_points else "CAMERA_ON_OR_INSIDE_INFINITE_CYLINDER",
        "distance_scope": "ONE_CROSS_SECTION_NOT_WHOLE_PIPE_OR_END_CAP",
    }


def pipe_measurement_record(spec: dict, *, status: str = "UNKNOWN", measured: dict | None = None,
                            model_prediction: dict | None = None, surface_samples: dict | None = None,
                            reason_codes: list[str] | None = None) -> dict:
    """Keep model sizes, fitted geometry and raw pixel depths distinct."""
    return {"revision": GEOMETRY_REVISION, "status": status,
            "nominal_diameter_mm": float(spec["nominal_diameter_mm"]),
            "nominal_diameter_source": "MODEL_CATALOG" if spec.get("cad_object_id") or spec.get("centerline_world_mm") is not None else "CONFIGURED_REFERENCE",
            "cad_object_id": spec.get("cad_object_id"),
            "measured_diameter_mm": measured.get("diameter_mm") if measured else None,
            "measured_section": measured, "model_predicted_section": model_prediction,
            "surface_samples": surface_samples or {}, "reason_codes": list(reason_codes or []),
            "accuracy_validated": False}
