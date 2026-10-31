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


def _ordered_bands(indices: list[int], values: np.ndarray, labels: tuple[str, str, str]) -> dict[int, str]:
    """Assign relative bands; a singleton remains middle because no ordering exists."""
    ordered = sorted(indices, key=lambda i: (-values[i], i))
    if len(ordered) == 1:
        return {ordered[0]: labels[1]}
    low, high = float(np.min(values[indices])), float(np.max(values[indices]))
    return {index: _band(float(values[index]), low, high, labels) for index in indices}


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
    camera_side_world: Sequence[float] | None = None,
    section_up_world: Sequence[float] | None = None,
    section_right_world: Sequence[float] | None = None,
    reference_basis: Mapping[str, Sequence[float]] | Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic cross-section reference from model pipe lines.

    ``section_uv_mm`` uses the right/up rows of ``basis_rows``.  Positive V is
    the model's visual up direction and positive U is visual right when viewing
    along the pipe axis.  The third basis row is the pipe axis, so it is an
    axial coordinate rather than an occlusion-depth measurement.
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
    u_values, v_values, depth_values = section[:, 0], section[:, 1], section[:, 2]
    u_low, u_high = float(np.min(u_values)), float(np.max(u_values))
    v_low, v_high = float(np.min(v_values)), float(np.max(v_values))
    axial_order = sorted(range(len(pipe_specs)), key=lambda i: (depth_values[i], pipe_ids[i]))
    vertical_order = sorted(range(len(pipe_specs)), key=lambda i: (-v_values[i], u_values[i], pipe_ids[i]))
    horizontal_order = sorted(range(len(pipe_specs)), key=lambda i: (u_values[i], -v_values[i], pipe_ids[i]))
    vertical_rank = {index: rank + 1 for rank, index in enumerate(vertical_order)}
    horizontal_rank = {index: rank + 1 for rank, index in enumerate(horizontal_order)}
    axial_rank = {index: rank + 1 for rank, index in enumerate(axial_order)}
    layer_keys = [str(pipe.get("layer_id", pipe.get("layer_path")) or "__ALL__") for pipe in pipe_specs]
    layer_vertical_rank: dict[int, int] = {}
    layer_vertical_band: dict[int, str] = {}
    layer_vertical_bounds: dict[str, tuple[float, float]] = {}
    for layer in sorted(set(layer_keys)):
        members = [i for i, key in enumerate(layer_keys) if key == layer]
        ordered = sorted(members, key=lambda i: (-v_values[i], u_values[i], pipe_ids[i]))
        layer_vertical_rank.update({index: rank + 1 for rank, index in enumerate(ordered)})
        low, high = float(np.min(v_values[members])), float(np.max(v_values[members]))
        layer_vertical_bounds[layer] = (low, high)
        layer_vertical_band.update(_ordered_bands(members, v_values, _BANDS))
    camera_side = None if camera_side_world is None else _unit(camera_side_world, field="camera_side_world")
    camera_depth = None if camera_side is None else (centres - origin) @ camera_side
    camera_depth_order = None if camera_depth is None else sorted(
        range(len(pipe_specs)), key=lambda i: (-camera_depth[i], pipe_ids[i]))
    camera_depth_rank = None if camera_depth_order is None else {
        index: rank + 1 for rank, index in enumerate(camera_depth_order)}

    pipes: list[dict[str, Any]] = []
    for index, pipe in enumerate(pipe_specs):
        row: dict[str, Any] = {
            "pipe_id": pipe_ids[index],
            "instance_id": pipe.get("instance_id"),
            "cad_object_id": pipe.get("cad_object_id"),
            "layer_id": pipe.get("layer_id", pipe.get("layer_path")),
            "nominal_diameter_mm": _finite_number(pipe.get("nominal_diameter_mm"), field=f"pipes[{index}].nominal_diameter_mm"),
            "center_world_mm": centres[index].tolist(),
            "section_uv_mm": [float(u_values[index]), float(v_values[index])],
            "axial_offset_mm": float(depth_values[index]),
            "vertical_band": layer_vertical_band[index],
            "vertical_band_global": _band(float(v_values[index]), v_low, v_high, _BANDS),
            "horizontal_band": _band(float(u_values[index]), u_low, u_high, _HORIZONTAL_BANDS),
            "vertical_rank_top_to_bottom": vertical_rank[index],
            "vertical_rank_within_layer_top_to_bottom": layer_vertical_rank[index],
            "vertical_layer_bounds_mm": list(layer_vertical_bounds[layer_keys[index]]),
            "vertical_rank_global_top_to_bottom": vertical_rank[index],
            "horizontal_rank_left_to_right": horizontal_rank[index],
            "axial_rank": axial_rank[index],
        }
        if camera_depth is not None:
            row["camera_depth_mm"] = float(camera_depth[index])
            row["camera_depth_rank_near_to_far"] = camera_depth_rank[index]
        # Preserve an already determined installation state as context only;
        # callers must still treat the deterministic report as authoritative.
        if "installation_state" in pipe:
            row["deterministic_installation_state"] = pipe.get("installation_state")
        pipes.append(row)

    reference: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "reference_kind": "CAD_MODEL_CROSS_SECTION",
        "scope": "ORIENTATION_AND_IDENTITY_AID_ONLY",
        "units": "millimetres",
        "axis_source": axis_source,
        "basis_source": basis_source,
        "axis_world": axis.tolist(),
        "origin_world_mm": origin.tolist(),
        "basis_rows": {
            "right_world": basis[0].tolist(),
            "up_world": basis[1].tolist(),
            "pipe_axis_world": basis[2].tolist(),
        },
        "orientation": {
            "positive_u": "right when viewed along pipe_axis_world",
            "positive_v": "up along the declared section basis",
            "positive_axis": "along pipe_axis_world; axial offset is not occlusion depth",
            "section_up_world": basis[1].tolist(),
            "camera_side_world": None if camera_side is None else camera_side.tolist(),
        },
        "bands": {
            "vertical": {"lower": v_low, "upper": v_high, "thresholds": [v_low + (v_high-v_low)/3.0, v_low + 2.0*(v_high-v_low)/3.0]},
            "horizontal": {"left": u_low, "right": u_high, "thresholds": [u_low + (u_high-u_low)/3.0, u_low + 2.0*(u_high-u_low)/3.0]},
        },
        "pipes": pipes,
        "deterministic_state_authority": "STEREO_OR_ELEVATION_REPORT_ONLY",
    }
    return reference


__all__ = ["SCHEMA_VERSION", "build_model_section_reference"]
