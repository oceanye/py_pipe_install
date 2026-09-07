"""Unified, read-only triangle-mesh access for 3MF and Rhino 3DM models.

The vision and GUI layers should consume :class:`CadScene` rather than depend
on a CAD SDK directly.  Every coordinate exposed by this module is converted
to millimetres.  Selected Rhino block instances and geometry without an
existing mesh are rejected: silently dropping or inventing a bound pipe would
break identity, projection, and occlusion decisions.  Callers may supply the
manifest-bound object IDs so unrelated construction geometry is not selected.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from zipfile import BadZipFile, ZipFile

import numpy as np

from .model_3mf import MODEL_PART, ThreeMFError, inspect_3mf
from .logging_config import get_logger, log_event

_LOGGER = get_logger("cad_model")

try:  # Keep legacy 3MF users importable until the optional dependency is installed.
    import rhino3dm  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised through a patched dependency test.
    rhino3dm = None  # type: ignore[assignment]


_MAX_CAD_BYTES = 512 * 1024 * 1024
_NIL_GUID = uuid.UUID(int=0)


class CadModelError(ValueError):
    """Raised when a CAD file cannot provide an unambiguous mesh scene."""


def _readonly_array(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    array.setflags(write=False)
    return array


def _validated_mesh_arrays(
    vertices: Any,
    triangles: Any,
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        vertices_array = np.asarray(vertices, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise CadModelError(f"{context} vertices are not a numeric 3-D array") from error
    try:
        triangles_array = np.asarray(triangles, dtype=np.int32)
    except (TypeError, ValueError, OverflowError) as error:
        raise CadModelError(f"{context} triangles are not an integer index array") from error
    if (
        vertices_array.ndim != 2
        or vertices_array.shape[1:] != (3,)
        or len(vertices_array) < 3
    ):
        raise CadModelError(f"{context} must contain at least three 3-D vertices")
    if (
        triangles_array.ndim != 2
        or triangles_array.shape[1:] != (3,)
        or len(triangles_array) < 1
    ):
        raise CadModelError(f"{context} must contain at least one triangle")
    if not np.all(np.isfinite(vertices_array)):
        raise CadModelError(f"{context} contains non-finite vertices")
    if triangles_array.min() < 0 or triangles_array.max() >= len(vertices_array):
        raise CadModelError(f"{context} contains an out-of-range vertex index")
    if np.any(
        (triangles_array[:, 0] == triangles_array[:, 1])
        | (triangles_array[:, 1] == triangles_array[:, 2])
        | (triangles_array[:, 2] == triangles_array[:, 0])
    ):
        raise CadModelError(f"{context} contains a degenerate triangle index")

    return (
        _readonly_array(vertices_array, dtype=np.dtype(np.float64)),
        _readonly_array(triangles_array, dtype=np.dtype(np.int32)),
    )


def _mesh_is_watertight(triangles: np.ndarray) -> bool:
    edge_counts: dict[tuple[int, int], int] = {}
    for first, second, third in triangles.tolist():
        for edge in ((first, second), (second, third), (third, first)):
            key = tuple(sorted(edge))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    return bool(edge_counts) and all(count == 2 for count in edge_counts.values())


@dataclass(frozen=True)
class CadObject:
    """One identity-preserving CAD object represented as a triangle mesh."""

    object_id: str
    guid: str | None
    pipe_id: str | None
    name: str
    layer_path: str | None
    color_srgb: str
    geometry_type: str
    mesh_source: str
    vertices_world_mm: np.ndarray
    triangles: np.ndarray
    bbox_min_mm: tuple[float, float, float]
    bbox_max_mm: tuple[float, float, float]
    watertight: bool

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertices_world_mm))

    @property
    def triangle_count(self) -> int:
        return int(len(self.triangles))


@dataclass(frozen=True)
class CadScene:
    """A normalized CAD scene whose mesh coordinates are all millimetres."""

    source_path: Path
    source_format: str
    source_sha256: str
    source_unit: str
    unit_scale_to_mm: float
    objects: tuple[CadObject, ...]

    @property
    def object_count(self) -> int:
        return len(self.objects)

    @property
    def by_object_id(self) -> dict[str, CadObject]:
        return {item.object_id: item for item in self.objects}

    @property
    def by_guid(self) -> dict[str, CadObject]:
        return {item.guid: item for item in self.objects if item.guid is not None}


def _read_cad_snapshot(path: str | Path) -> tuple[Path, bytes, str]:
    source_path = Path(path).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    size = source_path.stat().st_size
    if size <= 0 or size > _MAX_CAD_BYTES:
        raise CadModelError(
            f"CAD file size must be between 1 and {_MAX_CAD_BYTES} bytes: {source_path}"
        )
    with source_path.open("rb") as stream:
        raw = stream.read(_MAX_CAD_BYTES + 1)
    if len(raw) != size or len(raw) > _MAX_CAD_BYTES:
        raise CadModelError(f"CAD file size changed while reading: {source_path}")
    return source_path, raw, hashlib.sha256(raw).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cad_object(
    *,
    object_id: str,
    guid: str | None,
    pipe_id: str | None,
    name: str,
    layer_path: str | None,
    color_srgb: str,
    geometry_type: str,
    mesh_source: str,
    vertices_world_mm: Any,
    triangles: Any,
    context: str,
    watertight_override: bool | None = None,
) -> CadObject:
    vertices_array, triangles_array = _validated_mesh_arrays(
        vertices_world_mm,
        triangles,
        context=context,
    )
    minimum = vertices_array.min(axis=0)
    maximum = vertices_array.max(axis=0)
    return CadObject(
        object_id=object_id,
        guid=guid,
        pipe_id=pipe_id,
        name=name,
        layer_path=layer_path,
        color_srgb=color_srgb.upper(),
        geometry_type=geometry_type,
        mesh_source=mesh_source,
        vertices_world_mm=vertices_array,
        triangles=triangles_array,
        bbox_min_mm=tuple(float(value) for value in minimum),
        bbox_max_mm=tuple(float(value) for value in maximum),
        watertight=(
            _mesh_is_watertight(triangles_array)
            if watertight_override is None
            else watertight_override
        ),
    )


def _validate_scene_identity(objects: list[CadObject], *, context: str) -> None:
    if not objects:
        raise CadModelError(f"{context} contains no supported mesh objects")
    object_ids = [item.object_id for item in objects]
    if any(not value for value in object_ids) or len(set(object_ids)) != len(object_ids):
        raise CadModelError(f"{context} object IDs must be unique and non-empty")
    guids = [item.guid for item in objects if item.guid is not None]
    if len(set(guids)) != len(guids):
        raise CadModelError(f"{context} object GUIDs must be unique")
    pipe_ids = [item.pipe_id for item in objects if item.pipe_id is not None]
    if len(set(pipe_ids)) != len(pipe_ids):
        raise CadModelError(f"{context} pipe_id user strings must be unique")


def _required_object_ids(values: Iterable[str] | None) -> set[str] | None:
    if values is None:
        return None
    normalized: set[str] = set()
    for index, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise CadModelError(
                f"required_object_ids[{index}] must be a non-empty string"
            )
        normalized.add(value.strip().casefold())
    if not normalized:
        raise CadModelError("required_object_ids must not be empty when supplied")
    return normalized


def load_3mf_scene(
    path: str | Path,
    *,
    required_object_ids: Iterable[str] | None = None,
) -> CadScene:
    """Load the direct, untransformed meshes accepted by the existing 3MF audit."""

    required_ids = _required_object_ids(required_object_ids)
    source_path, raw, source_sha256 = _read_cad_snapshot(path)
    try:
        inspection = inspect_3mf(source_path)
    except ThreeMFError as error:
        raise CadModelError(str(error)) from error
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        np.linalg.LinAlgError,
    ) as error:
        raise CadModelError(f"Invalid 3MF geometry: {source_path}") from error
    # ``inspect_3mf`` historically accepts a path and therefore opens it a
    # second time.  Confirm that the bytes it inspected are still the same
    # snapshot used below; otherwise metadata (IDs/colours) could be paired
    # with geometry from a different revision during a concurrent file copy.
    try:
        if _hash_file(source_path) != source_sha256:
            raise CadModelError(f"3MF file changed while being inspected: {source_path}")
    except OSError as error:
        raise CadModelError(f"Cannot re-check 3MF snapshot: {source_path}") from error

    all_metadata = {str(item["object_id"]): item for item in inspection["objects"]}
    metadata = {
        object_id: item
        for object_id, item in all_metadata.items()
        if required_ids is None or object_id.casefold() in required_ids
    }
    if required_ids is not None:
        missing = required_ids - {value.casefold() for value in metadata}
        if missing:
            raise CadModelError(
                f"3MF scene is missing required object IDs: {sorted(missing)}"
            )
    try:
        with ZipFile(BytesIO(raw)) as archive:
            root = ElementTree.fromstring(archive.read(MODEL_PART))
    except (BadZipFile, KeyError, ElementTree.ParseError) as error:
        raise CadModelError(f"Invalid 3MF archive: {source_path}") from error

    objects: list[CadObject] = []
    for element in root.findall(".//{*}object"):
        mesh = element.find("./{*}mesh")
        if mesh is None:
            continue
        object_id = str(element.attrib.get("id", ""))
        if required_ids is not None and object_id.casefold() not in required_ids:
            continue
        if object_id not in metadata:
            raise CadModelError(f"3MF mesh object {object_id!r} was not audited")
        try:
            vertices = np.asarray(
                [
                    [float(vertex.attrib[axis]) for axis in ("x", "y", "z")]
                    for vertex in mesh.findall("./{*}vertices/{*}vertex")
                ],
                dtype=np.float64,
            )
            triangles = np.asarray(
                [
                    [int(triangle.attrib[key]) for key in ("v1", "v2", "v3")]
                    for triangle in mesh.findall("./{*}triangles/{*}triangle")
                ],
                dtype=np.int32,
            )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise CadModelError(
                f"Invalid vertices or faces in 3MF object {object_id!r}"
            ) from error
        item = metadata[object_id]
        raw_guid = item.get("uuid")
        guid = str(raw_guid).lower() if raw_guid else None
        objects.append(
            _cad_object(
                object_id=object_id,
                guid=guid,
                pipe_id=None,
                name=str(item.get("name") or ""),
                layer_path=None,
                color_srgb=str(item.get("color_srgb") or "#B0B0B0"),
                geometry_type="Mesh",
                mesh_source="3mf_native_mesh",
                vertices_world_mm=vertices,
                triangles=triangles,
                context=f"3MF object {object_id}",
            )
        )

    _validate_scene_identity(objects, context="3MF scene")
    if set(metadata) != {item.object_id for item in objects}:
        raise CadModelError("3MF audited objects and loaded mesh objects do not match")
    return CadScene(
        source_path=source_path,
        source_format="3mf",
        source_sha256=source_sha256,
        source_unit="millimeter",
        unit_scale_to_mm=1.0,
        objects=tuple(objects),
    )


def _rhino_dependency() -> Any:
    if rhino3dm is None:
        raise CadModelError(
            "Rhino .3dm loading requires the pinned dependency rhino3dm==8.32.1"
        )
    return rhino3dm


def _unit_scale_to_mm(model: Any, sdk: Any) -> tuple[str, float]:
    source_unit = model.Settings.ModelUnitSystem
    source_name = getattr(source_unit, "name", str(source_unit).rsplit(".", 1)[-1])
    unsupported = {
        getattr(sdk.UnitSystem, "None"),
        sdk.UnitSystem.Unset,
        sdk.UnitSystem.CustomUnits,
    }
    if source_unit in unsupported:
        raise CadModelError(f"Unsupported Rhino model unit: {source_name}")
    try:
        scale = float(
            sdk.UnitSystem.UnitScale(source_unit, sdk.UnitSystem.Millimeters)
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise CadModelError(
            f"Cannot convert Rhino model unit {source_name!r} to millimetres"
        ) from error
    if not math.isfinite(scale) or scale <= 0:
        raise CadModelError(
            f"Invalid Rhino-to-millimetre unit scale for {source_name!r}: {scale}"
        )
    return str(source_name), scale


def _color_to_srgb(value: Any, *, context: str) -> str:
    if not isinstance(value, (tuple, list)) or len(value) < 3:
        raise CadModelError(f"{context} does not provide an RGB display color")
    channels: list[int] = []
    for component in value[:3]:
        if type(component) is not int or component < 0 or component > 255:
            raise CadModelError(f"{context} contains an invalid RGB display color")
        channels.append(component)
    return "#" + "".join(f"{component:02X}" for component in channels)


def _rhino_mesh_arrays(mesh: Any, *, context: str) -> tuple[np.ndarray, np.ndarray]:
    if mesh is None:
        raise CadModelError(f"{context} has no cached render mesh")
    try:
        vertices = np.asarray(
            [[vertex.X, vertex.Y, vertex.Z] for vertex in mesh.Vertices],
            dtype=np.float64,
        )
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise CadModelError(f"{context} contains invalid mesh vertices") from error
    triangles: list[tuple[int, int, int]] = []
    try:
        faces = list(mesh.Faces)
    except (AttributeError, TypeError, ValueError) as error:
        raise CadModelError(f"{context} does not expose mesh faces") from error
    for face in faces:
        try:
            if len(face) != 4:
                raise CadModelError(f"{context} contains an unsupported mesh face")
            first, second, third, fourth = (int(value) for value in face)
        except CadModelError:
            raise
        except (TypeError, ValueError, OverflowError) as error:
            raise CadModelError(f"{context} contains an invalid mesh face") from error
        triangles.append((first, second, third))
        if fourth != third:
            triangles.append((first, third, fourth))
    try:
        triangle_array = np.asarray(triangles, dtype=np.int32)
    except (TypeError, ValueError, OverflowError) as error:
        raise CadModelError(f"{context} contains invalid mesh face indices") from error
    return vertices, triangle_array


def _join_rhino_meshes(
    meshes: list[Any],
    *,
    context: str,
) -> tuple[np.ndarray, np.ndarray]:
    vertices_parts: list[np.ndarray] = []
    triangle_parts: list[np.ndarray] = []
    vertex_offset = 0
    for mesh_index, mesh in enumerate(meshes):
        vertices, triangles = _rhino_mesh_arrays(
            mesh,
            context=f"{context} mesh {mesh_index}",
        )
        vertices_parts.append(vertices)
        triangle_parts.append(triangles + vertex_offset)
        vertex_offset += len(vertices)
    if not vertices_parts:
        raise CadModelError(f"{context} has no cached render mesh")
    return np.vstack(vertices_parts), np.vstack(triangle_parts)


def _rhino_geometry_mesh(
    geometry: Any,
    *,
    sdk: Any,
    context: str,
) -> tuple[str, str, np.ndarray, np.ndarray, bool]:
    if isinstance(geometry, sdk.InstanceReference):
        raise CadModelError(
            f"{context} is a Rhino block instance; explode/flatten block instances first"
        )
    if isinstance(geometry, sdk.Mesh):
        vertices, triangles = _rhino_mesh_arrays(geometry, context=context)
        return "Mesh", "3dm_native_mesh", vertices, triangles, bool(geometry.IsClosed)
    if isinstance(geometry, sdk.Brep):
        meshes = [face.GetMesh(sdk.MeshType.Render) for face in geometry.Faces]
        if not meshes or any(mesh is None for mesh in meshes):
            raise CadModelError(
                f"{context} is a Brep without complete cached render meshes; "
                "mesh it in Rhino and save the .3dm, or export a bound 3MF"
            )
        vertices, triangles = _join_rhino_meshes(meshes, context=context)
        return (
            "Brep",
            "3dm_cached_render_mesh",
            vertices,
            triangles,
            bool(geometry.IsSolid),
        )
    if isinstance(geometry, sdk.Extrusion):
        mesh = geometry.GetMesh(sdk.MeshType.Render)
        if mesh is None:
            raise CadModelError(
                f"{context} is an Extrusion without a cached render mesh; "
                "mesh it in Rhino and save the .3dm, or export a bound 3MF"
            )
        vertices, triangles = _rhino_mesh_arrays(mesh, context=context)
        return (
            "Extrusion",
            "3dm_cached_render_mesh",
            vertices,
            triangles,
            bool(geometry.IsSolid),
        )
    raise CadModelError(
        f"{context} has unsupported Rhino geometry type {type(geometry).__name__!r}"
    )


def load_3dm_scene(
    path: str | Path,
    *,
    required_object_ids: Iterable[str] | None = None,
) -> CadScene:
    """Load selected Rhino objects using native or already-cached render meshes."""

    sdk = _rhino_dependency()
    required_ids = _required_object_ids(required_object_ids)
    source_path, raw, source_sha256 = _read_cad_snapshot(path)
    try:
        model = sdk.File3dm.FromByteArray(raw)
    except (RuntimeError, ValueError, TypeError) as error:
        raise CadModelError(f"Invalid Rhino 3DM file: {source_path}") from error
    if model is None:
        raise CadModelError(f"Invalid Rhino 3DM file: {source_path}")

    instance_definitions = list(model.InstanceDefinitions)
    if instance_definitions and required_ids is None:
        raise CadModelError(
            "Rhino block instance definitions are not supported; explode/flatten blocks first"
        )
    source_unit, unit_scale = _unit_scale_to_mm(model, sdk)
    layers = {int(layer.Index): layer for layer in model.Layers}

    objects: list[CadObject] = []
    for object_index, item in enumerate(model.Objects):
        attributes = item.Attributes
        context = f"Rhino object {object_index}"
        raw_guid = attributes.Id
        if not isinstance(raw_guid, uuid.UUID) or raw_guid == _NIL_GUID:
            if required_ids is not None:
                continue
            raise CadModelError(f"{context} must have a persistent non-nil GUID")
        guid = str(raw_guid).lower()
        if required_ids is not None and guid.casefold() not in required_ids:
            continue
        context = f"Rhino object {guid}"
        if bool(attributes.IsInstanceDefinitionObject):
            raise CadModelError(
                f"{context} belongs to a block definition; explode/flatten blocks first"
            )
        layer_index = int(attributes.LayerIndex)
        layer = layers.get(layer_index)
        if layer is None:
            raise CadModelError(
                f"{context} references missing layer index {layer_index}"
            )
        try:
            draw_color = attributes.DrawColor(model)
        except (RuntimeError, ValueError, TypeError) as error:
            raise CadModelError(f"Cannot resolve display color for {context}") from error
        color_srgb = _color_to_srgb(draw_color, context=context)
        (
            geometry_type,
            mesh_source,
            vertices,
            triangles,
            watertight,
        ) = _rhino_geometry_mesh(item.Geometry, sdk=sdk, context=context)
        vertices *= unit_scale
        raw_pipe_id = attributes.GetUserString("pipe_id")
        normalized_pipe_id = str(raw_pipe_id).strip() if raw_pipe_id else ""
        pipe_id = normalized_pipe_id or None
        objects.append(
            _cad_object(
                object_id=guid,
                guid=guid,
                pipe_id=pipe_id,
                name=str(attributes.Name or ""),
                layer_path=str(layer.FullPath or layer.Name or ""),
                color_srgb=color_srgb,
                geometry_type=geometry_type,
                mesh_source=mesh_source,
                vertices_world_mm=vertices,
                triangles=triangles,
                context=context,
                watertight_override=watertight,
            )
        )

    if required_ids is not None:
        missing = required_ids - {item.object_id.casefold() for item in objects}
        if missing:
            raise CadModelError(
                f"Rhino scene is missing required object GUIDs: {sorted(missing)}"
            )
    _validate_scene_identity(objects, context="Rhino scene")
    return CadScene(
        source_path=source_path,
        source_format="3dm",
        source_sha256=source_sha256,
        source_unit=source_unit,
        unit_scale_to_mm=unit_scale,
        objects=tuple(objects),
    )


def load_cad_scene(
    path: str | Path,
    *,
    required_object_ids: Iterable[str] | None = None,
) -> CadScene:
    """Load a supported CAD file into the common millimetre mesh contract.

    When ``required_object_ids`` is supplied, unbound auxiliary Rhino objects
    are ignored while every requested identity remains mandatory and strict.
    """

    suffix = Path(path).suffix.lower()
    log_event(_LOGGER, "cad_load_start", path=Path(path), format=suffix or "<none>")
    if suffix == ".3mf":
        scene = load_3mf_scene(path, required_object_ids=required_object_ids)
        log_event(_LOGGER, "cad_load_finished", path=scene.source_path, format=scene.source_format, object_count=scene.object_count)
        return scene
    if suffix == ".3dm":
        scene = load_3dm_scene(path, required_object_ids=required_object_ids)
        log_event(_LOGGER, "cad_load_finished", path=scene.source_path, format=scene.source_format, object_count=scene.object_count)
        return scene
    log_event(_LOGGER, "cad_load_rejected", path=Path(path), format=suffix or "<none>")
    raise CadModelError(
        f"Unsupported CAD model extension {suffix or '<none>'!r}; expected .3mf or .3dm"
    )


__all__ = [
    "CadModelError",
    "CadObject",
    "CadScene",
    "load_3dm_scene",
    "load_3mf_scene",
    "load_cad_scene",
]
