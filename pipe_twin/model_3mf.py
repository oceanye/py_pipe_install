"""Read-only inspection of the current 3MF pipe fixture.

The importer intentionally reports geometry evidence without assigning business
meaning.  Stable ``pipe_id`` values remain in the sidecar manifest.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import numpy as np


MODEL_PART = "3D/3dmodel.model"


class ThreeMFError(ValueError):
    """Raised when a 3MF archive cannot be inspected safely."""


def _attribute_by_local_name(element: ElementTree.Element, name: str) -> str | None:
    for key, value in element.attrib.items():
        if key.rsplit("}", 1)[-1].lower() == name.lower():
            return value
    return None


def _as_color_srgb(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().upper()
    if not value.startswith("#") or len(value) not in (7, 9):
        return value
    return value[:7]


def _fit_circle(points: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Fit a circle and return center, radius and RMS radial residual."""

    design = np.column_stack((2.0 * points[:, 0], 2.0 * points[:, 1], np.ones(len(points))))
    rhs = np.square(points[:, 0]) + np.square(points[:, 1])
    center_u, center_v, constant = np.linalg.lstsq(design, rhs, rcond=None)[0]
    radius_squared = constant + center_u**2 + center_v**2
    if radius_squared <= 0:
        raise ThreeMFError("The mesh cross-section does not define a valid circle")
    center = np.array([center_u, center_v], dtype=float)
    radius = float(np.sqrt(radius_squared))
    residual = np.linalg.norm(points - center, axis=1) - radius
    return center, radius, float(np.sqrt(np.mean(np.square(residual))))


def _mesh_topology(triangles: np.ndarray) -> dict[str, int | bool]:
    edges: Counter[tuple[int, int]] = Counter()
    degenerate = 0
    for v1, v2, v3 in triangles.tolist():
        if len({v1, v2, v3}) != 3:
            degenerate += 1
        edges.update(
            (
                tuple(sorted((v1, v2))),
                tuple(sorted((v2, v3))),
                tuple(sorted((v3, v1))),
            )
        )

    boundary = sum(count == 1 for count in edges.values())
    non_manifold = sum(count > 2 for count in edges.values())
    return {
        "unique_edge_count": len(edges),
        "boundary_edge_count": boundary,
        "non_manifold_edge_count": non_manifold,
        "degenerate_triangle_count": degenerate,
        "watertight": boundary == 0 and non_manifold == 0 and degenerate == 0,
    }


def _inspect_mesh(vertices: np.ndarray, triangles: np.ndarray) -> dict[str, Any]:
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        raise ThreeMFError("A mesh must contain at least three 3-D vertices")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) < 1:
        raise ThreeMFError("A mesh must contain triangular faces")
    if triangles.min() < 0 or triangles.max() >= len(vertices):
        raise ThreeMFError("A triangle references a vertex outside the mesh")

    minimum = vertices.min(axis=0)
    maximum = vertices.max(axis=0)
    spans = maximum - minimum
    axis_index = int(np.argmax(spans))
    cross_indices = [index for index in range(3) if index != axis_index]
    cross_center, radius, circle_residual = _fit_circle(vertices[:, cross_indices])

    centerline_start = np.zeros(3, dtype=float)
    centerline_end = np.zeros(3, dtype=float)
    centerline_start[axis_index] = minimum[axis_index]
    centerline_end[axis_index] = maximum[axis_index]
    centerline_start[cross_indices] = cross_center
    centerline_end[cross_indices] = cross_center

    result: dict[str, Any] = {
        "vertex_count": int(len(vertices)),
        "triangle_count": int(len(triangles)),
        "bbox_min_mm": minimum.tolist(),
        "bbox_max_mm": maximum.tolist(),
        "principal_axis": "xyz"[axis_index],
        "length_mm": float(spans[axis_index]),
        "diameter_mm": float(2.0 * radius),
        "circle_fit_rms_mm": circle_residual,
        "centerline_world_mm": [centerline_start.tolist(), centerline_end.tolist()],
    }
    result.update(_mesh_topology(triangles))
    return result


def inspect_3mf(path: str | Path) -> dict[str, Any]:
    """Inspect a 3MF Core archive and return JSON-serializable mesh metrics."""

    asset_path = Path(path)
    if not asset_path.is_file():
        raise FileNotFoundError(asset_path)

    try:
        with ZipFile(asset_path) as archive:
            if MODEL_PART not in archive.namelist():
                raise ThreeMFError(f"Missing required model part: {MODEL_PART}")
            root = ElementTree.fromstring(archive.read(MODEL_PART))
    except (BadZipFile, ElementTree.ParseError) as exc:
        raise ThreeMFError(f"Invalid 3MF archive: {asset_path}") from exc

    unit = root.attrib.get("unit", "millimeter")
    if unit != "millimeter":
        raise ThreeMFError(
            f"Unsupported 3MF unit {unit!r}; this M0 inspector only emits millimetre fields"
        )
    if root.find(".//{*}components") is not None:
        raise ThreeMFError("Component objects are not supported by the M0 3MF inspector")
    build = root.find("./{*}build")
    if build is None:
        raise ThreeMFError("The M0 3MF inspector requires an explicit build section")
    build_items = build.findall("./{*}item")
    if not build_items:
        raise ThreeMFError("The 3MF build section contains no items")
    if any(item.attrib.get("transform") for item in build_items):
        raise ThreeMFError("Build transforms are not supported by the M0 3MF inspector")

    color_groups: dict[str, list[str | None]] = {}
    for group in root.findall(".//{*}colorgroup"):
        group_id = group.attrib.get("id")
        if group_id is None:
            continue
        color_groups[group_id] = [
            _as_color_srgb(color.attrib.get("color")) for color in group.findall("./{*}color")
        ]

    objects: list[dict[str, Any]] = []
    for element in root.findall(".//{*}object"):
        mesh = element.find("./{*}mesh")
        if mesh is None:
            continue

        vertices = np.asarray(
            [
                [float(vertex.attrib[axis]) for axis in ("x", "y", "z")]
                for vertex in mesh.findall("./{*}vertices/{*}vertex")
            ],
            dtype=float,
        )
        triangles = np.asarray(
            [
                [int(triangle.attrib[key]) for key in ("v1", "v2", "v3")]
                for triangle in mesh.findall("./{*}triangles/{*}triangle")
            ],
            dtype=int,
        )

        property_id = element.attrib.get("pid")
        property_index = int(element.attrib.get("pindex", "0"))
        colors = color_groups.get(property_id or "", [])
        color = colors[property_index] if 0 <= property_index < len(colors) else None

        object_result: dict[str, Any] = {
            "id": element.attrib.get("id"),
            "object_id": element.attrib.get("id"),
            "uuid": _attribute_by_local_name(element, "UUID"),
            "name": element.attrib.get("name"),
            "type": element.attrib.get("type"),
            "color_srgb": color,
        }
        object_result.update(_inspect_mesh(vertices, triangles))
        objects.append(object_result)

    if not objects:
        raise ThreeMFError("The 3MF archive contains no mesh objects")

    mesh_ids = {str(item["object_id"]) for item in objects}
    build_ids = [item.attrib.get("objectid") for item in build_items]
    if any(object_id is None for object_id in build_ids) or set(build_ids) != mesh_ids:
        raise ThreeMFError(
            "Every M0 mesh object must appear exactly as an untransformed build item"
        )
    if len(build_ids) != len(set(build_ids)):
        raise ThreeMFError("Duplicate build instances are not supported by the M0 inspector")

    return {
        "format": "3MF Core",
        "unit": unit,
        "title": next(
            (
                metadata.text
                for metadata in root.findall("./{*}metadata")
                if metadata.attrib.get("name") == "Title"
            ),
            None,
        ),
        "object_count": len(objects),
        "total_vertex_count": sum(item["vertex_count"] for item in objects),
        "total_triangle_count": sum(item["triangle_count"] for item in objects),
        "all_meshes_watertight": all(bool(item["watertight"]) for item in objects),
        "objects": objects,
    }
