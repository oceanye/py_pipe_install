"""Deterministic fixed-point maps for a model cross-section.

This module exposes one 2-D point map for a chosen observation direction.
It deliberately omits axial segments, depth layers, and installation state.
The AI can evaluate visibility, position, and diameter at ``P01`` ... ``P09``;
the stereo/state engines remain outside this contract.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any

import numpy as np

from .camera_view import preview_rotation


SCHEMA_VERSION = "section-point-map-v1"


def _unit(value: Any, *, field: str) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise ValueError(f"{field} must be a finite 3-vector")
    length = float(np.linalg.norm(vector))
    if length < 1e-9:
        raise ValueError(f"{field} must be non-zero")
    return vector / length


def _line(pipe: Mapping[str, Any], index: int) -> np.ndarray:
    try:
        line = np.asarray(pipe["centerline_world_mm"], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"pipes[{index}].centerline_world_mm must be a numeric 2x3 array") from error
    if line.shape != (2, 3) or not np.isfinite(line).all():
        raise ValueError(f"pipes[{index}].centerline_world_mm must be a finite 2x3 array")
    if float(np.linalg.norm(line[1] - line[0])) < 1e-9:
        raise ValueError(f"pipes[{index}].centerline_world_mm must have non-zero length")
    return line


def _derived_normal(lines: np.ndarray) -> np.ndarray:
    directions = lines[:, 1] - lines[:, 0]
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    _, _, vectors = np.linalg.svd(directions, full_matrices=True)
    normal = vectors[0]
    if normal[int(np.argmax(np.abs(normal)))] < 0:
        normal = -normal
    return _unit(normal, field="derived section normal")


def _finite_number(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not np.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _reference_rows(reference: Any) -> np.ndarray:
    if isinstance(reference, Mapping):
        right = reference.get("right_world")
        up = reference.get("up_world")
        normal = reference.get("normal_world", reference.get("pipe_axis_world"))
        if right is None or up is None or normal is None:
            raise ValueError("reference_basis must contain right_world, up_world and normal_world")
        reference = [right, up, normal]
    matrix = np.asarray(reference, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("reference_basis must be a finite 3x3 row basis")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms < 1e-9):
        raise ValueError("reference_basis rows must be non-zero")
    matrix = matrix / norms[:, None]
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError("reference_basis rows must be orthogonal")
    if float(np.linalg.det(matrix)) < 0:
        raise ValueError("reference_basis must be right-handed")
    return matrix


def _view_projection(view: Mapping[str, Any] | None, centres: np.ndarray) -> tuple[str, np.ndarray | None, dict[str, Any]]:
    if view is None:
        return "SECTION_VIEW", None, {}
    if not isinstance(view, Mapping):
        raise ValueError("observation_view must be a mapping")
    view_id = str(view.get("view_id", "OBSERVATION_VIEW"))
    rotation = np.asarray(view.get("rotation_world_to_view"), dtype=float)
    centre = np.asarray(view.get("center_world_mm", [0., 0., 0.]), dtype=float)
    principal = np.asarray(view.get("principal_point_px"), dtype=float)
    scale = _finite_number(view.get("scale_px_per_mm"), field="observation_view.scale_px_per_mm")
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all() or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
        raise ValueError("observation_view.rotation_world_to_view must be an orthonormal 3x3 matrix")
    if centre.shape != (3,) or principal.shape != (2,) or not np.isfinite(centre).all() or not np.isfinite(principal).all() or scale <= 0:
        raise ValueError("observation_view contains invalid centre, principal point, or scale")
    camera = (centres - centre) @ rotation.T
    pixels = camera[:, :2] * scale + principal
    metadata = {
        "view_id": view_id,
        "rotation_world_to_view": rotation.tolist(),
        "center_world_mm": centre.tolist(),
        "principal_point_px": principal.tolist(),
        "scale_px_per_mm": scale,
    }
    return view_id, pixels, metadata


def build_model_section_reference(
    pipe_specs: Sequence[Mapping[str, Any]],
    axis_world: Sequence[float] | None = None,
    *,
    observation_direction_world: Sequence[float] | None = None,
    section_up_world: Sequence[float] | None = None,
    section_right_world: Sequence[float] | None = None,
    reference_basis: Mapping[str, Sequence[float]] | Sequence[Sequence[float]] | None = None,
    observation_view: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fixed ``P01...Pn`` map for one observed 2-D section.

    Point order is the projected observation order (pixel ``v`` top-to-bottom,
    then ``u`` left-to-right) when ``observation_view`` is supplied. Otherwise
    it is the same order in section coordinates. No per-point axial or depth
    identity is emitted.
    """
    if not isinstance(pipe_specs, Sequence) or isinstance(pipe_specs, (str, bytes)) or not pipe_specs:
        raise ValueError("pipe_specs must be a non-empty sequence")
    lines = np.stack([_line(pipe, index) for index, pipe in enumerate(pipe_specs)])
    pipe_ids: list[str] = []
    for index, pipe in enumerate(pipe_specs):
        value = pipe.get("pipe_id") if isinstance(pipe, Mapping) else None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"pipes[{index}].pipe_id must be a non-empty string")
        if value in pipe_ids:
            raise ValueError(f"duplicate pipe_id: {value}")
        pipe_ids.append(value)

    if reference_basis is not None and (axis_world is not None or observation_direction_world is not None or section_up_world is not None or section_right_world is not None):
        raise ValueError("reference_basis cannot be combined with explicit section directions")
    reference_rows = _reference_rows(reference_basis) if reference_basis is not None else None
    if reference_rows is not None:
        basis = reference_rows
        normal_source = "REFERENCE_BASIS"
    else:
        normal_value = observation_direction_world if observation_direction_world is not None else axis_world
        normal = _unit(normal_value, field="observation_direction_world" if observation_direction_world is not None else "axis_world") if normal_value is not None else _derived_normal(lines)
        normal_source = "OBSERVATION_DIRECTION" if observation_direction_world is not None else "EXPLICIT_NORMAL" if axis_world is not None else "DERIVED_FROM_CENTERLINES"
        if section_up_world is None and section_right_world is None:
            basis = preview_rotation(normal.tolist())
        else:
            if section_up_world is not None:
                up = _unit(section_up_world, field="section_up_world")
                up = up - normal * float(up @ normal)
                up = _unit(up, field="section_up_world_projected")
            if section_right_world is not None:
                right = _unit(section_right_world, field="section_right_world")
                right = right - normal * float(right @ normal)
                right = _unit(right, field="section_right_world_projected")
            if section_up_world is None:
                up = _unit(np.cross(normal, right), field="derived section up")
            elif section_right_world is None:
                right = _unit(np.cross(up, normal), field="derived section right")
            else:
                right = _unit(np.cross(up, normal), field="derived section right")
                up = _unit(np.cross(normal, right), field="derived section up")
            basis = np.vstack((right, up, normal))

    centres = lines.mean(axis=1)
    origin = np.mean(centres, axis=0)
    section = (centres - origin) @ basis.T
    uv = section[:, :2]
    view_id, projected, view_metadata = _view_projection(observation_view, centres)
    if projected is None:
        order = sorted(range(len(pipe_specs)), key=lambda i: (-uv[i, 1], uv[i, 0], pipe_ids[i]))
        order_rule = "SECTION_V_DESC_THEN_U_ASC_THEN_PIPE_ID"
    else:
        order = sorted(range(len(pipe_specs)), key=lambda i: (projected[i, 1], projected[i, 0], pipe_ids[i]))
        order_rule = "PROJECTED_PIXEL_V_ASC_THEN_U_ASC_THEN_PIPE_ID"

    points: list[dict[str, Any]] = []
    for point_order, index in enumerate(order, 1):
        point = {
            "point_id": f"P{point_order:02d}",
            "pipe_id": pipe_ids[index],
            "instance_id": pipe_specs[index].get("instance_id"),
            "cad_object_id": pipe_specs[index].get("cad_object_id"),
            "section_uv_mm": [float(uv[index, 0]), float(uv[index, 1])],
            "projected_center_px": None if projected is None else projected[index].tolist(),
            "nominal_diameter_mm": _finite_number(pipe_specs[index].get("nominal_diameter_mm"), field=f"pipes[{index}].nominal_diameter_mm"),
            "point_order": point_order,
            "projection_ambiguity": "NONE",
            "mapping_source": "MODEL_SECTION_PLUS_OBSERVATION_DIRECTION",
        }
        points.append(point)

    if projected is not None:
        for first in range(len(points)):
            for second in range(first + 1, len(points)):
                a = np.asarray(points[first]["projected_center_px"], dtype=float)
                b = np.asarray(points[second]["projected_center_px"], dtype=float)
                if float(np.linalg.norm(a - b)) <= 1e-6:
                    points[first]["projection_ambiguity"] = "OVERLAPPING_MODEL_POINTS"
                    points[second]["projection_ambiguity"] = "OVERLAPPING_MODEL_POINTS"

    pairs = []
    for first in range(len(points)):
        for second in range(first + 1, len(points)):
            delta = np.asarray(points[second]["section_uv_mm"], dtype=float) - np.asarray(points[first]["section_uv_mm"], dtype=float)
            distance = float(np.linalg.norm(delta))
            clearance = distance - (points[first]["nominal_diameter_mm"] + points[second]["nominal_diameter_mm"]) / 2.0
            pairs.append({"point_ids": [points[first]["point_id"], points[second]["point_id"]], "pipe_ids": [points[first]["pipe_id"], points[second]["pipe_id"]], "delta_uv_mm": delta.tolist(), "center_distance_mm": distance, "surface_clearance_mm": clearance, "scope": "SAME_2D_SECTION_ONLY"})

    mapping_payload = {
        "view_id": view_id,
        "ordering_rule": order_rule,
        "basis_rows": {"right_world": basis[0].tolist(), "up_world": basis[1].tolist(), "normal_world": basis[2].tolist()},
        "points": [{"point_id": p["point_id"], "pipe_id": p["pipe_id"], "section_uv_mm": p["section_uv_mm"], "projected_center_px": p["projected_center_px"]} for p in points],
    }
    mapping_sha256 = hashlib.sha256(json.dumps(mapping_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "reference_kind": "FIXED_MODEL_SECTION_POINT_MAP",
        "scope": "CROSS_SECTION_EVALUATION_ONLY",
        "units": "millimetres",
        "point_count": len(points),
        "point_id_format": "P%02d",
        "view_id": view_id,
        "mapping_sha256": mapping_sha256,
        "normal_source": normal_source,
        "observation_direction_world": basis[2].tolist(),
        "origin_world_mm": origin.tolist(),
        "basis_rows": {"right_world": basis[0].tolist(), "up_world": basis[1].tolist(), "normal_world": basis[2].tolist()},
        "observation_view": view_metadata,
        "ordering_rule": order_rule,
        "points": points,
        "pairs": pairs,
        "installation_state": "OUT_OF_SCOPE_FOR_SECTION_REFERENCE",
    }


__all__ = ["SCHEMA_VERSION", "build_model_section_reference"]
