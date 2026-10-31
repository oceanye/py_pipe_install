"""Stable CAD cross-section coordinates for multimodal evidence.

The stereo/state engines remain authoritative for installation status.  This
module only turns the model pipe catalogue into a human- and model-readable
cross-section reference: every pipe receives deterministic coordinates and
coarse upper/middle/lower and left/centre/right labels.  The reference is
intended to accompany section overlays sent to an AI model; it is not a
replacement for metric registration or state classification.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .camera_view import preview_rotation


SCHEMA_VERSION = "model-section-reference-v1"
_BANDS = ("LOWER", "MIDDLE", "UPPER")
_HORIZONTAL_BANDS = ("LEFT", "CENTER", "RIGHT")


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


def _derived_axis(lines: np.ndarray) -> np.ndarray:
    directions = lines[:, 1] - lines[:, 0]
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    _, _, vectors = np.linalg.svd(directions, full_matrices=True)
    axis = vectors[0]
    # Fix the otherwise arbitrary SVD sign.  This gives repeatable labels when
    # the same catalogue is loaded in a different order.
    if axis[int(np.argmax(np.abs(axis)))] < 0:
        axis = -axis
    return _unit(axis, field="derived axis")


def _band(value: float, low: float, high: float, labels: tuple[str, str, str]) -> str:
    span = high - low
    if span <= 1e-9:
        return labels[1]
    ratio = (value - low) / span
    if ratio < 1.0 / 3.0:
        return labels[0]
    if ratio > 2.0 / 3.0:
        return labels[2]
    return labels[1]


def _finite_number(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not np.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _reference_rows(reference: Any) -> np.ndarray:
    """Validate a right/up/axis row basis supplied in world coordinates."""
    if isinstance(reference, Mapping):
        keys = ("right_world", "up_world", "pipe_axis_world")
        if any(key not in reference for key in keys):
            raise ValueError("reference_basis must contain right_world, up_world and pipe_axis_world")
        reference = [reference[key] for key in keys]
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


def build_model_section_reference(
    pipe_specs: Sequence[Mapping[str, Any]],
    axis_world: Sequence[float] | None = None,
    *,
    section_up_world: Sequence[float] | None = None,
    section_right_world: Sequence[float] | None = None,
    reference_basis: Mapping[str, Sequence[float]] | Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic cross-section reference from model pipe lines.

    ``section_uv_mm`` is the only per-pipe geometry exposed to an AI caller.
    The normal row only defines the plane; no axial segment or camera-depth
    identity is emitted.
    """
    if not isinstance(pipe_specs, Sequence) or isinstance(pipe_specs, (str, bytes)):
        raise ValueError("pipe_specs must be a sequence")
    if not pipe_specs:
        raise ValueError("pipe_specs must not be empty")
    lines = np.stack([_line(pipe, index) for index, pipe in enumerate(pipe_specs)])
    pipe_ids: list[str] = []
    for index, pipe in enumerate(pipe_specs):
        value = pipe.get("pipe_id") if isinstance(pipe, Mapping) else None
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"pipes[{index}].pipe_id must be a non-empty string")
        if value in pipe_ids:
            raise ValueError(f"duplicate pipe_id: {value}")
        pipe_ids.append(value)

    original_axis_world = axis_world
    if reference_basis is not None and (section_up_world is not None or section_right_world is not None):
        raise ValueError("reference_basis cannot be combined with section_up_world/section_right_world")
    reference_rows = _reference_rows(reference_basis) if reference_basis is not None else None
    if axis_world is None and reference_rows is not None:
        axis_world = reference_rows[2]
    axis_source = ("REFERENCE_BASIS" if reference_rows is not None and original_axis_world is None
                   else "EXPLICIT" if axis_world is not None else "DERIVED_FROM_CENTERLINES")
    axis = _unit(axis_world, field="axis_world") if axis_world is not None else _derived_axis(lines)
    if reference_rows is not None:
        if not np.allclose(reference_rows[2], axis, atol=1e-6, rtol=0):
            raise ValueError("reference_basis pipe_axis_world conflicts with axis_world")
        basis = np.vstack((reference_rows[0], reference_rows[1], axis))
        basis_source = "REFERENCE_BASIS"
    elif section_up_world is None and section_right_world is None:
        # preview_rotation returns right/up/pipe-axis rows, matching the
        # transform used by ElevationModelViewer for its model section canvas.
        basis = preview_rotation(axis.tolist())
        basis_source = "MODEL_PREVIEW"
    else:
        if section_up_world is not None:
            up = _unit(section_up_world, field="section_up_world")
            up = up - axis * float(up @ axis)
            if float(np.linalg.norm(up)) < 1e-9:
                raise ValueError("section_up_world must not be parallel to axis_world")
            up = _unit(up, field="section_up_world_projected")
        if section_right_world is not None:
            right = _unit(section_right_world, field="section_right_world")
            right = right - axis * float(right @ axis)
            if float(np.linalg.norm(right)) < 1e-9:
                raise ValueError("section_right_world must not be parallel to axis_world")
            right = _unit(right, field="section_right_world_projected")
        if section_up_world is None:
            up = _unit(np.cross(axis, right), field="derived section up")
        elif section_right_world is None:
            right = _unit(np.cross(up, axis), field="derived section right")
        else:
            right = _unit(np.cross(up, axis), field="derived section right")
            up = _unit(np.cross(axis, right), field="derived section up")
        basis = np.vstack((right, up, axis))
        basis_source = ("EXPLICIT_SECTION_UP" if section_up_world is not None and section_right_world is None
                        else "EXPLICIT_SECTION_RIGHT" if section_right_world is not None and section_up_world is None
                        else "EXPLICIT_SECTION_UP_RIGHT")
    origin = np.mean(lines.mean(axis=1), axis=0)
    centres = lines.mean(axis=1)
    section = (centres - origin) @ basis.T
    u_values, v_values = section[:, 0], section[:, 1]
    u_low, u_high = float(np.min(u_values)), float(np.max(u_values))
    v_low, v_high = float(np.min(v_values)), float(np.max(v_values))
    vertical_order = sorted(range(len(pipe_specs)), key=lambda i: (-v_values[i], u_values[i], pipe_ids[i]))
    horizontal_order = sorted(range(len(pipe_specs)), key=lambda i: (u_values[i], -v_values[i], pipe_ids[i]))
    vertical_rank = {index: rank + 1 for rank, index in enumerate(vertical_order)}
    horizontal_rank = {index: rank + 1 for rank, index in enumerate(horizontal_order)}
    pipes: list[dict[str, Any]] = []
    for index, pipe in enumerate(pipe_specs):
        row: dict[str, Any] = {
            "pipe_id": pipe_ids[index],
            "instance_id": pipe.get("instance_id"),
            "cad_object_id": pipe.get("cad_object_id"),
            "nominal_diameter_mm": _finite_number(pipe.get("nominal_diameter_mm"), field=f"pipes[{index}].nominal_diameter_mm"),
            "section_uv_mm": [float(u_values[index]), float(v_values[index])],
            "vertical_band": _band(float(v_values[index]), v_low, v_high, _BANDS),
            "horizontal_band": _band(float(u_values[index]), u_low, u_high, _HORIZONTAL_BANDS),
            "vertical_rank_top_to_bottom": vertical_rank[index],
            "horizontal_rank_left_to_right": horizontal_rank[index],
        }
        pipes.append(row)

    pairs = []
    for first in range(len(pipes)):
        for second in range(first + 1, len(pipes)):
            delta = np.asarray(pipes[second]["section_uv_mm"], dtype=float) - np.asarray(pipes[first]["section_uv_mm"], dtype=float)
            distance = float(np.linalg.norm(delta))
            clearance = distance - (pipes[first]["nominal_diameter_mm"] + pipes[second]["nominal_diameter_mm"]) / 2.0
            pairs.append({
                "pipe_ids": [pipes[first]["pipe_id"], pipes[second]["pipe_id"]],
                "delta_uv_mm": delta.tolist(),
                "center_distance_mm": distance,
                "surface_clearance_mm": clearance,
                "scope": "SAME_2D_SECTION_ONLY",
            })

    reference: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "reference_kind": "CAD_MODEL_CROSS_SECTION",
        "scope": "CROSS_SECTION_EVALUATION_ONLY",
        "units": "millimetres",
        "axis_source": axis_source,
        "basis_source": basis_source,
        "axis_world": axis.tolist(),
        "origin_world_mm": origin.tolist(),
        "basis_rows": {
            "right_world": basis[0].tolist(),
            "up_world": basis[1].tolist(),
            "normal_world": basis[2].tolist(),
        },
        "orientation": {
            "positive_u": "right in the supplied section basis",
            "positive_v": "up along the declared section basis",
            "section_up_world": basis[1].tolist(),
        },
        "bands": {
            "vertical": {"lower": v_low, "upper": v_high, "thresholds": [v_low + (v_high-v_low)/3.0, v_low + 2.0*(v_high-v_low)/3.0]},
            "horizontal": {"left": u_low, "right": u_high, "thresholds": [u_low + (u_high-u_low)/3.0, u_low + 2.0*(u_high-u_low)/3.0]},
        },
        "pipes": pipes,
        "pairs": pairs,
        "installation_state": "OUT_OF_SCOPE_FOR_SECTION_REFERENCE",
    }
    return reference


__all__ = ["SCHEMA_VERSION", "build_model_section_reference"]
