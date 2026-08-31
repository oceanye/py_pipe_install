"""Deterministic CAD truth rendering for stereo and occlusion experiments.

The renderer is intentionally small and CPU-only.  It produces exact synthetic
camera-Z depth and instance ownership for the supplied triangle meshes.  These
outputs are geometric test truth, not evidence that a field camera is calibrated.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree
from zipfile import ZipFile

import cv2
import numpy as np

from . import __version__
from .model_3mf import MODEL_PART, inspect_3mf
from .pipeline import (
    atomic_write_text,
    ensure_paths_distinct,
    load_json_snapshot,
    sha256_file,
)
from .state import (
    INSTALLATION_STATES,
    NegativeInstallationEvidence,
    classify_installation_state,
    installation_state_label_zh,
)


_GENERATED_FILENAMES = (
    "camera.json",
    "left_rgb.png",
    "left_rgb_textured.png",
    "left_depth_preview.png",
    "left_instance_preview.png",
    "left_truth.npz",
    "left_view_topology.json",
    "right_rgb.png",
    "right_rgb_textured.png",
    "right_depth_preview.png",
    "right_instance_preview.png",
    "right_truth.npz",
    "right_view_topology.json",
    "elevation_reference.png",
    "elevation_amodal_overlay.png",
    "elevation_reference.svg",
    "elevation_view_topology.json",
    "occlusion_topology_graph.svg",
    "occlusion_matrix.csv",
    "elevation_continuous_overlap.csv",
    "installation_status.json",
    "dataset_manifest.json",
)

_STEREO_FUSION_VIEW_IDS = ("left", "right")
_VIEW_EVIDENCE_TYPES = (
    "DIRECT_SYNTHETIC_INSTANCE_EVIDENCE",
    "NEGATIVE_EVIDENCE_CANDIDATE",
    "INCONCLUSIVE",
)


@dataclass(frozen=True)
class MeshGeometry:
    object_id: str
    name: str
    uuid: str
    color_srgb: str
    measured_diameter_mm: float
    measured_centerline_world_mm: np.ndarray
    vertices_world_mm: np.ndarray
    triangles: np.ndarray


@dataclass(frozen=True)
class SceneInstance:
    instance_id: int
    pipe_id: str
    layer_id: str
    color_class: str
    nominal_diameter_mm: float
    ground_truth_installation_state: str
    centerline_world_mm: np.ndarray
    mesh: MeshGeometry


@dataclass(frozen=True)
class CameraModel:
    camera_id: str
    width: int
    height: int
    rotation_world_to_camera: np.ndarray
    center_world_mm: np.ndarray
    near_mm: float
    far_mm: float
    projection: str
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    scale_px_per_mm: float | None = None

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation_world_to_camera, dtype=float)
        center = np.asarray(self.center_world_mm, dtype=float)
        if rotation.shape != (3, 3) or center.shape != (3,):
            raise ValueError("Camera rotation must be 3x3 and center must have three values")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9):
            raise ValueError("Camera rotation must be orthonormal")
        if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-9):
            raise ValueError("Camera rotation determinant must equal +1")
        if self.width <= 0 or self.height <= 0 or self.near_mm <= 0 or self.far_mm <= self.near_mm:
            raise ValueError("Invalid image or clipping range")
        if self.projection == "perspective":
            if any(value is None or value <= 0 for value in (self.fx, self.fy)):
                raise ValueError("Perspective cameras require positive fx and fy")
            if self.cx is None or self.cy is None:
                raise ValueError("Perspective cameras require cx and cy")
        elif self.projection == "orthographic":
            if self.scale_px_per_mm is None or self.scale_px_per_mm <= 0:
                raise ValueError("Orthographic cameras require a positive scale")
            if self.cx is None or self.cy is None:
                raise ValueError("Orthographic cameras require cx and cy")
        else:
            raise ValueError(f"Unsupported projection: {self.projection}")

    @property
    def transform_camera_world(self) -> np.ndarray:
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = self.rotation_world_to_camera
        transform[:3, 3] = -self.rotation_world_to_camera @ self.center_world_mm
        return transform

    def world_to_camera(self, points_world_mm: np.ndarray) -> np.ndarray:
        points = np.asarray(points_world_mm, dtype=float)
        return (self.rotation_world_to_camera @ (points - self.center_world_mm).T).T

    def project(self, points_world_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        camera_points = self.world_to_camera(points_world_mm)
        depth = camera_points[:, 2]
        if self.projection == "perspective":
            uv = np.column_stack(
                (
                    float(self.cx) + float(self.fx) * camera_points[:, 0] / depth,
                    float(self.cy) + float(self.fy) * camera_points[:, 1] / depth,
                )
            )
        else:
            uv = np.column_stack(
                (
                    float(self.cx) + float(self.scale_px_per_mm) * camera_points[:, 0],
                    float(self.cy) + float(self.scale_px_per_mm) * camera_points[:, 1],
                )
            )
        return uv, depth

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "camera_id": self.camera_id,
            "width": self.width,
            "height": self.height,
            "projection": self.projection,
            "rotation_world_to_camera": self.rotation_world_to_camera.tolist(),
            "center_world_mm": self.center_world_mm.tolist(),
            "T_camera_world": self.transform_camera_world.tolist(),
            "near_mm": self.near_mm,
            "far_mm": self.far_mm,
            "depth_convention": "camera_z_mm",
            "distortion_model": "none",
        }
        if self.projection == "perspective":
            payload["K"] = [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ]
        else:
            payload["scale_px_per_mm"] = self.scale_px_per_mm
            payload["principal_point_px"] = [self.cx, self.cy]
        return payload


@dataclass
class RenderResult:
    rgb_bgr: np.ndarray
    depth_z_mm: np.ndarray
    instance_id: np.ndarray


def _local_attribute(element: ElementTree.Element, name: str) -> str | None:
    for key, value in element.attrib.items():
        if key.rsplit("}", 1)[-1].lower() == name.lower():
            return value
    return None


def load_mesh_geometries(path: str | Path) -> dict[str, MeshGeometry]:
    """Load direct, untransformed millimetre meshes after the normal 3MF audit."""

    audit = inspect_3mf(path)
    metadata = {str(item["object_id"]): item for item in audit["objects"]}
    with ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read(MODEL_PART))

    geometries: dict[str, MeshGeometry] = {}
    for element in root.findall(".//{*}object"):
        mesh = element.find("./{*}mesh")
        if mesh is None:
            continue
        object_id = str(element.attrib["id"])
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
        item = metadata[object_id]
        geometries[object_id] = MeshGeometry(
            object_id=object_id,
            name=str(item.get("name") or ""),
            uuid=str(_local_attribute(element, "UUID") or ""),
            color_srgb=str(item.get("color_srgb") or "#B0B0B0"),
            measured_diameter_mm=float(item["diameter_mm"]),
            measured_centerline_world_mm=np.asarray(
                item["centerline_world_mm"], dtype=np.float64
            ),
            vertices_world_mm=vertices,
            triangles=triangles,
        )
    return geometries


def _hex_rgb(color: str) -> np.ndarray:
    value = color.removeprefix("#")[:6]
    return np.asarray([int(value[index : index + 2], 16) for index in (0, 2, 4)], dtype=float)


def _face_bgr(mesh: MeshGeometry, triangle: np.ndarray) -> np.ndarray:
    vertices = mesh.vertices_world_mm[triangle]
    normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    magnitude = np.linalg.norm(normal)
    if magnitude > 0:
        normal /= magnitude
    light_world = np.asarray([0.15, -0.2, 1.0])
    light_world /= np.linalg.norm(light_world)
    diffuse = abs(float(normal @ light_world))
    shade = 0.62 + 0.38 * diffuse
    rgb = np.clip(_hex_rgb(mesh.color_srgb) * shade, 0, 255)
    return rgb[::-1].astype(np.uint8)


def _rasterize_triangle(
    uv: np.ndarray,
    depths: np.ndarray,
    camera: CameraModel,
    zbuffer: np.ndarray,
    instance_buffer: np.ndarray,
    rgb: np.ndarray,
    instance_id: int,
    color_bgr: np.ndarray,
) -> None:
    min_x = max(0, int(math.floor(float(np.min(uv[:, 0])))))
    max_x = min(camera.width - 1, int(math.ceil(float(np.max(uv[:, 0])))))
    min_y = max(0, int(math.floor(float(np.min(uv[:, 1])))))
    max_y = min(camera.height - 1, int(math.ceil(float(np.max(uv[:, 1])))))
    if min_x > max_x or min_y > max_y:
        return

    x0, y0 = uv[0]
    x1, y1 = uv[1]
    x2, y2 = uv[2]
    denominator = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(float(denominator)) < 1e-12:
        return

    xs = np.arange(min_x, max_x + 1, dtype=np.float64) + 0.5
    ys = np.arange(min_y, max_y + 1, dtype=np.float64) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)
    weight0 = ((y1 - y2) * (grid_x - x2) + (x2 - x1) * (grid_y - y2)) / denominator
    weight1 = ((y2 - y0) * (grid_x - x2) + (x0 - x2) * (grid_y - y2)) / denominator
    weight2 = 1.0 - weight0 - weight1
    inside = (weight0 >= -1e-8) & (weight1 >= -1e-8) & (weight2 >= -1e-8)
    if not np.any(inside):
        return

    if camera.projection == "perspective":
        inverse_depth = weight0 / depths[0] + weight1 / depths[1] + weight2 / depths[2]
        interpolated_depth = 1.0 / inverse_depth
    else:
        interpolated_depth = weight0 * depths[0] + weight1 * depths[1] + weight2 * depths[2]

    zview = zbuffer[min_y : max_y + 1, min_x : max_x + 1]
    update = inside & (interpolated_depth < zview)
    if not np.any(update):
        return
    zview[update] = interpolated_depth[update]
    instance_buffer[min_y : max_y + 1, min_x : max_x + 1][update] = instance_id
    rgb[min_y : max_y + 1, min_x : max_x + 1][update] = color_bgr


def render_scene(
    instances: list[SceneInstance],
    camera: CameraModel,
    background_bgr: tuple[int, int, int] = (174, 174, 174),
) -> RenderResult:
    """Render RGB, camera-Z depth and instance ownership with a Z-buffer."""

    zbuffer = np.full((camera.height, camera.width), np.inf, dtype=np.float64)
    instance_buffer = np.zeros((camera.height, camera.width), dtype=np.uint16)
    rgb = np.empty((camera.height, camera.width, 3), dtype=np.uint8)
    rgb[:] = background_bgr

    for instance in instances:
        mesh = instance.mesh
        projected, depths = camera.project(mesh.vertices_world_mm)
        for triangle in mesh.triangles:
            triangle_depths = depths[triangle]
            if np.any(triangle_depths <= camera.near_mm) or np.any(
                triangle_depths >= camera.far_mm
            ):
                continue
            _rasterize_triangle(
                projected[triangle],
                triangle_depths,
                camera,
                zbuffer,
                instance_buffer,
                rgb,
                instance.instance_id,
                _face_bgr(mesh, triangle),
            )

    depth = zbuffer.astype(np.float32)
    depth[~np.isfinite(depth)] = np.nan
    return RenderResult(rgb_bgr=rgb, depth_z_mm=depth, instance_id=instance_buffer)


def _bbox(mask: np.ndarray) -> list[int] | None:
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        return None
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max() - columns.min() + 1),
        int(rows.max() - rows.min() + 1),
    ]


def _axis_coverage(full_mask: np.ndarray, visible_mask: np.ndarray, axis_uv: np.ndarray) -> float:
    direction = axis_uv[1] - axis_uv[0]
    length = float(np.linalg.norm(direction))
    if length <= 1e-9:
        return 0.0
    direction /= length
    full_rows, full_columns = np.nonzero(full_mask)
    visible_rows, visible_columns = np.nonzero(visible_mask)
    if len(full_rows) == 0 or len(visible_rows) == 0:
        return 0.0
    origin = axis_uv[0]
    full_bins = np.unique(
        np.rint(
            (np.column_stack((full_columns, full_rows)) - origin) @ direction
        ).astype(np.int32)
    )
    visible_bins = np.unique(
        np.rint(
            (np.column_stack((visible_columns, visible_rows)) - origin) @ direction
        ).astype(np.int32)
    )
    return float(len(np.intersect1d(full_bins, visible_bins)) / max(len(full_bins), 1))


def compute_view_topology(
    instances: list[SceneInstance],
    camera: CameraModel,
    scene_render: RenderResult | None = None,
) -> tuple[dict[str, Any], dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Compute amodal/visible masks and directed occluder-to-target relations."""

    scene = scene_render or render_scene(instances, camera)
    full_masks: dict[int, np.ndarray] = {}
    visible_masks: dict[int, np.ndarray] = {}
    pipe_records: list[dict[str, Any]] = []
    edge_records: list[dict[str, Any]] = []

    by_id = {instance.instance_id: instance for instance in instances}
    for instance in instances:
        solo = render_scene([instance], camera)
        full_mask = solo.instance_id == instance.instance_id
        visible_mask = scene.instance_id == instance.instance_id
        full_masks[instance.instance_id] = full_mask
        visible_masks[instance.instance_id] = visible_mask
        full_pixels = int(np.count_nonzero(full_mask))
        visible_pixels = int(np.count_nonzero(visible_mask))
        hidden_mask = full_mask & ~visible_mask
        hidden_pixels = int(np.count_nonzero(hidden_mask))
        visible_fraction = visible_pixels / full_pixels if full_pixels else 0.0

        owners, counts = np.unique(scene.instance_id[hidden_mask], return_counts=True)
        occluders: list[dict[str, Any]] = []
        for owner, count in zip(owners.tolist(), counts.tolist(), strict=True):
            if owner == 0 or owner == instance.instance_id:
                continue
            occluder = by_id[int(owner)]
            contribution = int(count)
            record = {
                "occluder_instance_id": int(owner),
                "occluder_pipe_id": occluder.pipe_id,
                "occluded_pixels": contribution,
                "fraction_of_target_amodal": contribution / max(full_pixels, 1),
            }
            occluders.append(record)
            edge_records.append(
                {
                    "from_instance_id": int(owner),
                    "from_pipe_id": occluder.pipe_id,
                    "to_instance_id": instance.instance_id,
                    "to_pipe_id": instance.pipe_id,
                    "occluded_pixels": contribution,
                    "fraction_of_target_amodal": contribution / max(full_pixels, 1),
                }
            )

        if full_pixels == 0:
            state = "OUT_OF_FRUSTUM"
        elif visible_pixels == 0 and hidden_pixels == full_pixels:
            state = "FULLY_OCCLUDED"
        elif hidden_pixels == 0:
            state = "FULLY_VISIBLE"
        else:
            state = "PARTIALLY_OCCLUDED"

        axis_uv, axis_depth = camera.project(instance.centerline_world_mm)
        component_pixels = 0
        if visible_pixels:
            component_count, _, stats, _ = cv2.connectedComponentsWithStats(
                visible_mask.astype(np.uint8), connectivity=8
            )
            if component_count > 1:
                component_pixels = int(np.max(stats[1:, cv2.CC_STAT_AREA]))
        assessable = (
            state in {"FULLY_VISIBLE", "PARTIALLY_OCCLUDED"}
            and component_pixels >= 50
        )
        if assessable:
            installation_evidence = "DIRECT_SYNTHETIC_INSTANCE_EVIDENCE"
            reason_codes = [
                "EXACT_MANIFEST_INSTANCE_BINDING",
                "VISIBLE_INSTANCE_PIXELS",
            ]
        else:
            installation_evidence = "INCONCLUSIVE"
            reason_codes = [state, "NO_QUALIFIED_NEGATIVE_EVIDENCE"]
        pipe_records.append(
            {
                "instance_id": instance.instance_id,
                "pipe_id": instance.pipe_id,
                "layer_id": instance.layer_id,
                "nominal_diameter_mm": instance.nominal_diameter_mm,
                "axis_uv": axis_uv.tolist(),
                "axis_depth_z_mm": axis_depth.tolist(),
                "amodal_bbox_xywh": _bbox(full_mask),
                "visible_bbox_xywh": _bbox(visible_mask),
                "amodal_pixels_in_frame": full_pixels,
                "visible_pixels": visible_pixels,
                "occluded_pixels": hidden_pixels,
                "visible_fraction": visible_fraction,
                "visible_axis_coverage": _axis_coverage(full_mask, visible_mask, axis_uv),
                "largest_visible_component_pixels": component_pixels,
                "occlusion_state": state,
                "assessable": assessable,
                "installation_evidence": installation_evidence,
                "installation_reason_codes": reason_codes,
                "negative_evidence_gate": None,
                "expected_region_assessable": None,
                "expected_region_occlusion_state": "NOT_EVALUATED",
                "ground_truth_installation_state": (
                    instance.ground_truth_installation_state
                ),
                "production_authority": False,
                "occluders": sorted(
                    occluders,
                    key=lambda item: item["fraction_of_target_amodal"],
                    reverse=True,
                ),
            }
        )

    return (
        {
            "schema_version": "1.0",
            "camera_id": camera.camera_id,
            "edge_direction": "occluder_to_occluded_target",
            "pipes": pipe_records,
            "occlusion_edges": sorted(
                edge_records,
                key=lambda item: (
                    item["from_instance_id"],
                    item["to_instance_id"],
                ),
            ),
        },
        full_masks,
        visible_masks,
    )


def _negative_evidence_from_record(
    record: dict[str, Any],
    *,
    view_id: str,
) -> NegativeInstallationEvidence | None:
    payload = record.get("negative_evidence_gate")
    evidence_type = record.get("installation_evidence")
    if payload is None:
        if evidence_type == "NEGATIVE_EVIDENCE_CANDIDATE":
            raise ValueError(
                f"{view_id} instance {record.get('instance_id')} has a negative "
                "candidate without a complete gate bundle"
            )
        return None
    if evidence_type != "NEGATIVE_EVIDENCE_CANDIDATE":
        raise ValueError(
            f"{view_id} instance {record.get('instance_id')} has a negative gate "
            "without NEGATIVE_EVIDENCE_CANDIDATE"
        )
    if not isinstance(payload, dict):
        raise ValueError("negative_evidence_gate must be an object or null")
    try:
        return NegativeInstallationEvidence(**payload)
    except TypeError as error:
        raise ValueError(
            f"Invalid negative evidence gate for {view_id} instance "
            f"{record.get('instance_id')}"
        ) from error


def _index_topology_records(
    topology: dict[str, Any],
    *,
    expected_camera_id: str,
    instances: list[SceneInstance],
) -> dict[int, dict[str, Any]]:
    if topology.get("camera_id") != expected_camera_id:
        raise ValueError(
            f"Expected topology camera_id {expected_camera_id!r}, got "
            f"{topology.get('camera_id')!r}"
        )
    records = topology.get("pipes")
    if not isinstance(records, list):
        raise ValueError(f"{expected_camera_id} topology pipes must be a list")

    expected_by_id = {instance.instance_id: instance for instance in instances}
    if len(expected_by_id) != len(instances):
        raise ValueError("Scene instance_id values must be unique")
    if len({instance.pipe_id for instance in instances}) != len(instances):
        raise ValueError("Scene pipe_id values must be unique")

    indexed: dict[int, dict[str, Any]] = {}
    seen_pipe_ids: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"{expected_camera_id} topology records must be objects")
        raw_instance_id = record.get("instance_id")
        if type(raw_instance_id) is not int:
            raise ValueError(f"{expected_camera_id} instance_id must be an integer")
        instance_id = raw_instance_id
        if instance_id in indexed:
            raise ValueError(
                f"Duplicate instance_id {instance_id} in {expected_camera_id} topology"
            )
        if instance_id not in expected_by_id:
            raise ValueError(
                f"Unexpected instance_id {instance_id} in {expected_camera_id} topology"
            )
        pipe_id = record.get("pipe_id")
        if pipe_id != expected_by_id[instance_id].pipe_id:
            raise ValueError(
                f"pipe_id mismatch for {expected_camera_id} instance {instance_id}"
            )
        if pipe_id in seen_pipe_ids:
            raise ValueError(
                f"Duplicate pipe_id {pipe_id!r} in {expected_camera_id} topology"
            )
        evidence_type = record.get("installation_evidence")
        if evidence_type not in _VIEW_EVIDENCE_TYPES:
            raise ValueError(
                f"Invalid installation evidence for {expected_camera_id} "
                f"instance {instance_id}"
            )
        if evidence_type == "DIRECT_SYNTHETIC_INSTANCE_EVIDENCE" and not (
            record.get("assessable") is True
            and type(record.get("visible_pixels")) is int
            and record["visible_pixels"] > 0
            and record.get("occlusion_state")
            in {"FULLY_VISIBLE", "PARTIALLY_OCCLUDED"}
        ):
            raise ValueError(
                f"Direct instance evidence is inconsistent for "
                f"{expected_camera_id} instance {instance_id}"
            )
        _negative_evidence_from_record(record, view_id=expected_camera_id)
        indexed[instance_id] = record
        seen_pipe_ids.add(pipe_id)

    missing_ids = set(expected_by_id) - set(indexed)
    if missing_ids:
        raise ValueError(
            f"Missing instance_id values in {expected_camera_id} topology: "
            f"{sorted(missing_ids)}"
        )
    if len(indexed) != len(expected_by_id):
        raise ValueError(f"Unexpected record count in {expected_camera_id} topology")
    return indexed


def _view_evidence_summary(
    record: dict[str, Any],
    *,
    view_id: str,
) -> tuple[dict[str, Any], NegativeInstallationEvidence | None]:
    negative = _negative_evidence_from_record(record, view_id=view_id)
    evidence_type = str(record["installation_evidence"])
    reason_codes = list(record["installation_reason_codes"])
    expected_region_clear = (
        record.get("expected_region_assessable") is True
        and record.get("expected_region_occlusion_state") == "UNOCCLUDED"
        and type(record.get("amodal_pixels_in_frame")) is int
        and record["amodal_pixels_in_frame"] > 0
        and type(record.get("visible_pixels")) is int
        and record["visible_pixels"] == 0
        and type(record.get("occluded_pixels")) is int
        and record["occluded_pixels"] == 0
        and isinstance(record.get("occluders"), list)
        and not record["occluders"]
        and record.get("occlusion_state") == "NOT_OBSERVED"
    )
    negative_qualified = bool(
        negative is not None
        and negative.is_qualified()
        and expected_region_clear
    )
    if negative is not None:
        if negative_qualified:
            evidence_type = "QUALIFIED_NEGATIVE_EVIDENCE"
            reason_codes = ["ALL_NEGATIVE_EVIDENCE_GATES_PASSED"]
        else:
            evidence_type = "UNQUALIFIED_NEGATIVE_CANDIDATE"
            reason_codes = [
                "NEGATIVE_EVIDENCE_GATE_INCOMPLETE"
                if not negative.is_qualified()
                else "NEGATIVE_EVIDENCE_TOPOLOGY_CONFLICT"
            ]
    return (
        {
            "occlusion_state": record["occlusion_state"],
            "assessable": record["assessable"],
            "installation_evidence": evidence_type,
            "reason_codes": reason_codes,
            "negative_evidence_qualified": negative_qualified,
            "expected_region_assessable": record.get(
                "expected_region_assessable"
            ),
            "expected_region_occlusion_state": record.get(
                "expected_region_occlusion_state"
            ),
        },
        negative if negative_qualified else None,
    )


def build_installation_assessment(
    instances: list[SceneInstance],
    stereo_topologies: dict[str, dict[str, Any]],
    reference_topology: dict[str, Any],
) -> dict[str, Any]:
    """Fuse exact left/right synthetic evidence into one state per pipe."""

    if set(stereo_topologies) != set(_STEREO_FUSION_VIEW_IDS):
        raise ValueError("Stereo fusion requires exactly the left and right views")
    stereo_records = {
        view_id: _index_topology_records(
            stereo_topologies[view_id],
            expected_camera_id=view_id,
            instances=instances,
        )
        for view_id in _STEREO_FUSION_VIEW_IDS
    }
    reference_records = _index_topology_records(
        reference_topology,
        expected_camera_id="reference_elevation",
        instances=instances,
    )
    pipe_results: list[dict[str, Any]] = []
    counts = {state: 0 for state in INSTALLATION_STATES}

    for instance in instances:
        per_view: dict[str, dict[str, Any]] = {}
        installed_view_ids: list[str] = []
        negative_view_ids: list[str] = []
        qualified_negative_evidence: NegativeInstallationEvidence | None = None
        for view_id in _STEREO_FUSION_VIEW_IDS:
            record = stereo_records[view_id][instance.instance_id]
            summary, negative = _view_evidence_summary(record, view_id=view_id)
            per_view[view_id] = summary
            if record["installation_evidence"] == "DIRECT_SYNTHETIC_INSTANCE_EVIDENCE":
                installed_view_ids.append(view_id)
            if negative is not None and negative.is_qualified():
                negative_view_ids.append(view_id)
                qualified_negative_evidence = negative

        reference_record = reference_records[instance.instance_id]
        reference_summary, _ = _view_evidence_summary(
            reference_record,
            view_id="reference_elevation",
        )
        reference_summary["included_in_stereo_fusion"] = False
        per_view["reference_elevation"] = reference_summary

        evidence_conflict = bool(installed_view_ids and negative_view_ids)
        installation_state = classify_installation_state(
            direct_instance_evidence=bool(installed_view_ids),
            negative_evidence=qualified_negative_evidence,
            conflict=evidence_conflict,
        )
        counts[installation_state] += 1
        if installation_state == "INSTALLED":
            basis = "DIRECT_SYNTHETIC_INSTANCE_EVIDENCE"
            reasons = ["VISIBLE_INSTANCE_PIXELS_IN_STEREO_VIEW"]
        elif installation_state == "NOT_INSTALLED":
            basis = "QUALIFIED_NEGATIVE_EVIDENCE"
            reasons = ["ALL_NEGATIVE_EVIDENCE_GATES_PASSED"]
        elif evidence_conflict:
            basis = "CONFLICTING_EVIDENCE"
            reasons = ["POSITIVE_AND_QUALIFIED_NEGATIVE_EVIDENCE_CONFLICT"]
        else:
            basis = "INSUFFICIENT_EVIDENCE"
            reasons = [
                "NO_STEREO_VIEW_HAS_DIRECT_INSTANCE_EVIDENCE",
                "NO_QUALIFIED_NEGATIVE_EVIDENCE",
            ]
        pipe_results.append(
            {
                "instance_id": instance.instance_id,
                "pipe_id": instance.pipe_id,
                "installation_state": installation_state,
                "installation_state_zh": installation_state_label_zh(
                    installation_state
                ),
                "state_basis": basis,
                "reason_codes": reasons,
                "positive_evidence_view_ids": installed_view_ids,
                "negative_evidence_view_ids": negative_view_ids,
                "ground_truth_installation_state": (
                    instance.ground_truth_installation_state
                ),
                "ground_truth_source": "SYNTHETIC_SCENE_TRUTH",
                "visibility_by_view": per_view,
            }
        )

    return {
        "schema_version": "1.0",
        "domain": "synthetic_cad_truth",
        "assessment_kind": "synthetic_reference_oracle",
        "assessment_scope": "synthetic_rectified_stereo_left_right",
        "fusion_view_ids": list(_STEREO_FUSION_VIEW_IDS),
        "reference_elevation_included_in_fusion": False,
        "production_authority": False,
        "field_calibration_validated": False,
        "field_installation_state_inferred": False,
        "synthetic_reference_state_computed": True,
        "allowed_states": [
            {
                "code": state,
                "label_zh": installation_state_label_zh(state),
            }
            for state in INSTALLATION_STATES
        ],
        "policy": {
            "installed": "At least one stereo view has exact synthetic instance-owned visible pixels and no qualified negative conflict.",
            "not_installed": "Requires a structured gate bundle with validated calibration, registration, in-frame unoccluded expected region, sensor health, free space, at least two repeated absences, and at least two independent evidence sources.",
            "unknown": "No qualified positive or negative evidence, or evidence is occluded, out of view, unhealthy, or conflicting.",
            "safety_rule": "FULLY_OCCLUDED, OUT_OF_FRUSTUM, and NOT_OBSERVED never imply NOT_INSTALLED.",
        },
        "counts": counts,
        "pipes": pipe_results,
    }


def _camera_from_dict(camera_id: str, config: dict[str, Any]) -> CameraModel:
    return CameraModel(
        camera_id=camera_id,
        width=int(config["width"]),
        height=int(config["height"]),
        rotation_world_to_camera=np.asarray(config["rotation_world_to_camera"], dtype=float),
        center_world_mm=np.asarray(config["center_world_mm"], dtype=float),
        near_mm=float(config["near_mm"]),
        far_mm=float(config["far_mm"]),
        projection=str(config["projection"]),
        fx=float(config["fx"]) if "fx" in config else None,
        fy=float(config["fy"]) if "fy" in config else None,
        cx=float(config["cx"]),
        cy=float(config["cy"]),
        scale_px_per_mm=(
            float(config["scale_px_per_mm"]) if "scale_px_per_mm" in config else None
        ),
    )


def _build_scene(
    manifest: dict[str, Any], geometries: dict[str, MeshGeometry]
) -> list[SceneInstance]:
    instances: list[SceneInstance] = []
    seen_ids: set[int] = set()
    seen_pipe_ids: set[str] = set()
    seen_object_ids: set[str] = set()
    for pipe in manifest["model"]["pipes"]:
        instance_id = int(pipe["instance_id"])
        if instance_id <= 0 or instance_id > np.iinfo(np.uint16).max or instance_id in seen_ids:
            raise ValueError("instance_id values must be unique uint16 values above zero")
        seen_ids.add(instance_id)
        pipe_id = str(pipe["pipe_id"])
        if not pipe_id or pipe_id in seen_pipe_ids:
            raise ValueError("pipe_id values must be unique non-empty strings")
        seen_pipe_ids.add(pipe_id)
        object_id = str(pipe["cad_object_id"])
        if object_id in seen_object_ids:
            raise ValueError("Each CAD object may be bound to only one scene instance")
        seen_object_ids.add(object_id)
        mesh = geometries.get(object_id)
        if mesh is None:
            raise ValueError(f"Missing mesh object {object_id}")
        if mesh.uuid != pipe["cad_uuid"]:
            raise ValueError(f"UUID mismatch for {pipe['pipe_id']}")
        if mesh.color_srgb.upper() != str(pipe["color_srgb"]).upper():
            raise ValueError(f"CAD color mismatch for {pipe['pipe_id']}")
        centerline = np.asarray(pipe["centerline_world_mm"], dtype=float)
        if centerline.shape != (2, 3) or not np.all(np.isfinite(centerline)):
            raise ValueError(f"Invalid centerline for {pipe['pipe_id']}")
        diameter = float(pipe["nominal_diameter_mm"])
        if diameter <= 0:
            raise ValueError(f"Invalid nominal diameter for {pipe['pipe_id']}")
        if not math.isclose(
            diameter,
            mesh.measured_diameter_mm,
            rel_tol=0.0,
            abs_tol=0.05,
        ):
            raise ValueError(f"CAD diameter mismatch for {pipe['pipe_id']}")
        if not np.allclose(
            centerline,
            mesh.measured_centerline_world_mm,
            rtol=0.0,
            atol=0.05,
        ):
            raise ValueError(f"CAD centerline mismatch for {pipe['pipe_id']}")
        ground_truth_state = str(pipe["ground_truth_installation_state"])
        if ground_truth_state not in INSTALLATION_STATES:
            raise ValueError(
                f"Invalid ground-truth installation state for {pipe['pipe_id']}"
            )
        if ground_truth_state != "INSTALLED":
            raise ValueError(
                "Every entry in the current rendered scene must have "
                "ground_truth_installation_state=INSTALLED; a future missing-pipe "
                "fixture must separate the design catalog from active scene instances"
            )
        instances.append(
            SceneInstance(
                instance_id=instance_id,
                pipe_id=pipe_id,
                layer_id=str(pipe["layer_id"]),
                color_class=str(pipe["color_class"]),
                nominal_diameter_mm=diameter,
                ground_truth_installation_state=ground_truth_state,
                centerline_world_mm=centerline,
                mesh=mesh,
            )
        )
    return instances


def _validate_rectified_rig(
    rig: dict[str, Any], left: CameraModel, right: CameraModel
) -> None:
    if left.projection != "perspective" or right.projection != "perspective":
        raise ValueError("Stereo cameras must use perspective projection")
    scalar_fields = ("width", "height", "fx", "fy", "cx", "cy")
    for field in scalar_fields:
        if not math.isclose(
            float(getattr(left, field)),
            float(getattr(right, field)),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"Rectified stereo cameras must share {field}")
    if not np.allclose(
        left.rotation_world_to_camera,
        right.rotation_world_to_camera,
        atol=1e-9,
    ):
        raise ValueError("Rectified stereo cameras must share orientation")

    baseline_camera = left.rotation_world_to_camera @ (
        right.center_world_mm - left.center_world_mm
    )
    baseline_mm = float(rig["baseline_mm"])
    if not np.allclose(baseline_camera, [baseline_mm, 0.0, 0.0], atol=1e-9):
        raise ValueError(
            "Right camera center must be baseline_mm along the left camera +X axis"
        )
    if not bool(rig.get("rectified")):
        raise ValueError("Synthetic stereo rig must explicitly be rectified")


def _atomic_write_bytes(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise
    return path


def _write_png(path: Path, image: np.ndarray) -> Path:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError(f"Could not encode PNG: {path}")
    return _atomic_write_bytes(path, encoded.tobytes())


def _write_npz(path: Path, **arrays: np.ndarray) -> Path:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return _atomic_write_bytes(path, buffer.getvalue())


def _depth_preview(depth: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth)
    output = np.full((*depth.shape, 3), 174, dtype=np.uint8)
    if not np.any(valid):
        return output
    values = depth[valid]
    near, far = np.percentile(values, (1, 99))
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    normalized[valid] = np.clip((far - depth[valid]) * 255.0 / max(far - near, 1e-6), 0, 255)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    output[valid] = colored[valid]
    return output


def _instance_preview(instance_map: np.ndarray) -> np.ndarray:
    output = np.full((*instance_map.shape, 3), 174, dtype=np.uint8)
    for instance_id in np.unique(instance_map):
        if instance_id == 0:
            continue
        hue = int((int(instance_id) * 37) % 180)
        hsv = np.uint8([[[hue, 210, 245]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        output[instance_map == instance_id] = bgr
    return output


def _disparity_truth(
    depth_z_mm: np.ndarray, fx_px: float, baseline_mm: float
) -> np.ndarray:
    disparity = np.full(depth_z_mm.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(depth_z_mm) & (depth_z_mm > 0)
    disparity[valid] = fx_px * baseline_mm / depth_z_mm[valid]
    return disparity


def _procedural_textured_rgb(render: RenderResult, camera: CameraModel) -> np.ndarray:
    """Apply deterministic world-anchored texture for stereo matcher experiments."""

    if camera.projection != "perspective":
        raise ValueError("Procedural stereo texture requires a perspective camera")
    output = render.rgb_bgr.copy()
    rows, columns = np.nonzero(render.instance_id)
    if len(rows) == 0:
        return output
    depth = render.depth_z_mm[rows, columns].astype(np.float64)
    camera_points = np.column_stack(
        (
            (columns + 0.5 - float(camera.cx)) * depth / float(camera.fx),
            (rows + 0.5 - float(camera.cy)) * depth / float(camera.fy),
            depth,
        )
    )
    world_points = (
        camera.rotation_world_to_camera.T @ camera_points.T
    ).T + camera.center_world_mm
    cells = np.floor(world_points / 2.5).astype(np.int64) + 1_000_000
    cells_unsigned = cells.astype(np.uint64)
    hashes = (
        (cells_unsigned[:, 0] * np.uint64(73_856_093))
        ^ (cells_unsigned[:, 1] * np.uint64(19_349_663))
        ^ (cells_unsigned[:, 2] * np.uint64(83_492_791))
    )
    hashes ^= hashes >> np.uint64(13)
    hashes *= np.uint64(1_274_126_177)
    hashes ^= hashes >> np.uint64(16)
    texture_luma = 20.0 + 215.0 * ((hashes & np.uint64(255)).astype(np.float64) / 255.0)
    base = output[rows, columns].astype(np.float64)
    textured = 0.45 * base + 0.55 * texture_luma[:, None]
    output[rows, columns] = np.clip(textured, 0, 255).astype(np.uint8)
    return output


def _stereo_correspondence_valid(
    source: RenderResult,
    source_camera: CameraModel,
    target: RenderResult,
    target_camera: CameraModel,
    depth_tolerance_mm: float = 2.0,
) -> np.ndarray:
    """Mark source pixels whose reconstructed point is visible in the paired view."""

    valid_output = np.zeros(source.instance_id.shape, dtype=bool)
    rows, columns = np.nonzero(source.instance_id)
    if len(rows) == 0:
        return valid_output
    depth = source.depth_z_mm[rows, columns].astype(np.float64)
    camera_points = np.column_stack(
        (
            (columns + 0.5 - float(source_camera.cx))
            * depth
            / float(source_camera.fx),
            (rows + 0.5 - float(source_camera.cy))
            * depth
            / float(source_camera.fy),
            depth,
        )
    )
    world_points = (
        source_camera.rotation_world_to_camera.T @ camera_points.T
    ).T + source_camera.center_world_mm
    target_uv, target_depth = target_camera.project(world_points)
    target_columns = np.rint(target_uv[:, 0] - 0.5).astype(np.int64)
    target_rows = np.rint(target_uv[:, 1] - 0.5).astype(np.int64)
    inside = (
        (target_columns >= 0)
        & (target_columns < target_camera.width)
        & (target_rows >= 0)
        & (target_rows < target_camera.height)
    )
    candidate_indices = np.nonzero(inside)[0]
    if len(candidate_indices) == 0:
        return valid_output
    paired_ids = target.instance_id[
        target_rows[candidate_indices], target_columns[candidate_indices]
    ]
    paired_depth = target.depth_z_mm[
        target_rows[candidate_indices], target_columns[candidate_indices]
    ]
    source_ids = source.instance_id[
        rows[candidate_indices], columns[candidate_indices]
    ]
    correspond = (paired_ids == source_ids) & (
        np.abs(paired_depth - target_depth[candidate_indices]) <= depth_tolerance_mm
    )
    source_indices = candidate_indices[correspond]
    valid_output[rows[source_indices], columns[source_indices]] = True
    return valid_output


def _continuous_elevation_overlaps(instances: list[SceneInstance]) -> list[dict[str, Any]]:
    """Return nominal continuous cross-section overlaps, independent of pixels."""

    records: list[dict[str, Any]] = []
    front = [item for item in instances if item.layer_id == "front"]
    back = [item for item in instances if item.layer_id == "back"]
    for occluder in front:
        occluder_y = float(np.mean(occluder.centerline_world_mm[:, 1]))
        occluder_radius = occluder.nominal_diameter_mm / 2.0
        for target in back:
            target_y = float(np.mean(target.centerline_world_mm[:, 1]))
            target_radius = target.nominal_diameter_mm / 2.0
            lower = max(occluder_y - occluder_radius, target_y - target_radius)
            upper = min(occluder_y + occluder_radius, target_y + target_radius)
            overlap_mm = max(0.0, upper - lower)
            if overlap_mm <= 0:
                continue
            full_margin_mm = (
                occluder_radius - abs(occluder_y - target_y) - target_radius
            )
            fully_covered = math.isclose(
                overlap_mm,
                target.nominal_diameter_mm,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            if fully_covered:
                overlap_mm = target.nominal_diameter_mm
            if math.isclose(full_margin_mm, 0.0, rel_tol=0.0, abs_tol=1e-6):
                full_margin_mm = 0.0
            records.append(
                {
                    "occluder_instance_id": occluder.instance_id,
                    "occluder_pipe_id": occluder.pipe_id,
                    "target_instance_id": target.instance_id,
                    "target_pipe_id": target.pipe_id,
                    "center_offset_y_mm": abs(occluder_y - target_y),
                    "vertical_overlap_mm": overlap_mm,
                    "fraction_of_target_diameter": overlap_mm
                    / target.nominal_diameter_mm,
                    "full_coverage_margin_mm": full_margin_mm,
                    "continuous_state": (
                        "FULLY_COVERED" if fully_covered else "PARTIALLY_COVERED"
                    ),
                }
            )
    return records


def _amodal_elevation_overlay(
    elevation: RenderResult,
    instances: list[SceneInstance],
    full_masks: dict[int, np.ndarray],
    topology: dict[str, Any],
) -> np.ndarray:
    output = elevation.rgb_bgr.copy()
    records = {int(item["instance_id"]): item for item in topology["pipes"]}
    for instance in instances:
        mask = full_masks[instance.instance_id].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hue = int((instance.instance_id * 37) % 180)
        color = cv2.cvtColor(
            np.uint8([[[hue, 220, 250]]]), cv2.COLOR_HSV2BGR
        )[0, 0].tolist()
        cv2.drawContours(output, contours, -1, (20, 20, 20), 5, cv2.LINE_AA)
        cv2.drawContours(output, contours, -1, color, 2, cv2.LINE_AA)

    cv2.rectangle(output, (1190, 70), (1590, 585), (245, 245, 245), -1)
    cv2.rectangle(output, (1190, 70), (1590, 585), (40, 40, 40), 2)
    cv2.putText(
        output,
        "Amodal CAD outlines / visibility",
        (1205, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    for index, instance in enumerate(instances):
        record = records[instance.instance_id]
        y = 145 + index * 46
        hue = int((instance.instance_id * 37) % 180)
        color = cv2.cvtColor(
            np.uint8([[[hue, 220, 250]]]), cv2.COLOR_HSV2BGR
        )[0, 0].tolist()
        cv2.rectangle(output, (1207, y - 15), (1229, y + 7), color, -1)
        label = (
            f"{instance.instance_id} {instance.layer_id[0].upper()} "
            f"{record['occlusion_state']} {100.0 * record['visible_fraction']:.1f}%"
        )
        cv2.putText(
            output,
            label,
            (1240, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return output


def _mask_svg_path(mask: np.ndarray) -> str:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    parts: list[str] = []
    for contour in contours:
        epsilon = max(0.5, 0.001 * cv2.arcLength(contour, True))
        points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(points) < 3:
            continue
        parts.append("M " + " L ".join(f"{int(x)} {int(y)}" for x, y in points) + " Z")
    return " ".join(parts)


def build_elevation_svg(
    camera: CameraModel,
    instances: list[SceneInstance],
    full_masks: dict[int, np.ndarray],
    visible_masks: dict[int, np.ndarray],
    topology: dict[str, Any],
) -> str:
    """Create a vector elevation with dashed amodal and solid visible regions."""

    records = {int(item["instance_id"]): item for item in topology["pipes"]}
    ordered = sorted(
        instances,
        key=lambda item: float(np.mean(camera.project(item.centerline_world_mm)[1])),
        reverse=True,
    )
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{camera.width}" height="{camera.height}" viewBox="0 0 {camera.width} {camera.height}">',
        '<rect width="100%" height="100%" fill="#AEAEAE"/>',
        '<g>',
    ]
    for instance in ordered:
        color = instance.mesh.color_srgb
        visible_path = _mask_svg_path(visible_masks[instance.instance_id])
        if visible_path:
            elements.append(
                f'<path d="{visible_path}" fill="{color}" stroke="#202020" stroke-width="0.8" opacity="0.9"/>'
            )
    elements.append("</g><g>")
    for instance in ordered:
        full_path = _mask_svg_path(full_masks[instance.instance_id])
        if not full_path:
            continue
        elements.append(
            f'<path d="{full_path}" fill="none" stroke="#151515" stroke-width="5" '
            'stroke-dasharray="9 6" opacity="0.8"/>'
        )
        elements.append(
            f'<path d="{full_path}" fill="none" stroke="{instance.mesh.color_srgb}" stroke-width="2" '
            'stroke-dasharray="9 6"/>'
        )
    elements.append('</g><g font-family="Arial, sans-serif" font-size="14">')
    elements.append(
        '<rect x="1190" y="70" width="400" height="515" fill="#F5F5F5" stroke="#222"/>'
    )
    elements.append('<text x="1205" y="101" font-size="17">Instance / layer / state / visible</text>')
    for index, instance in enumerate(instances):
        record = records[instance.instance_id]
        y = 142 + index * 46
        fraction = 100.0 * float(record["visible_fraction"])
        elements.append(
            f'<text x="1208" y="{y}" fill="#111">{instance.instance_id} / {instance.layer_id} / '
            f'{record["occlusion_state"]} / {fraction:.1f}%</text>'
        )
    elements.extend(
        [
            "</g>",
            '<g font-family="Arial, sans-serif" font-size="16" fill="#111">',
            '<text x="24" y="30">Solid fill = visible; dashed outline = amodal/occluded extent</text>',
            '<text x="24" y="54">Reference elevation looks along world -Z; identity comes from instance_id</text>',
            "</g>",
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def build_topology_graph_svg(topology: dict[str, Any], width: int = 1800, height: int = 700) -> str:
    """Draw a simple two-layer directed occlusion graph without Graphviz."""

    pipes = topology["pipes"]
    front = sorted(
        [item for item in pipes if item["layer_id"] == "front"],
        key=lambda item: item["instance_id"],
    )
    back = sorted(
        [item for item in pipes if item["layer_id"] == "back"],
        key=lambda item: item["instance_id"],
    )
    positions: dict[int, tuple[float, float]] = {}
    for row, items in ((0, front), (1, back)):
        y = 180.0 if row == 0 else 500.0
        for index, item in enumerate(items):
            x = (index + 1) * width / (len(items) + 1)
            positions[int(item["instance_id"])] = (x, y)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="#444"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#FAFAFA"/>',
        '<text x="30" y="55" font-family="Arial" font-size="24">Front layer (closer to reference camera)</text>',
        '<text x="30" y="380" font-family="Arial" font-size="24">Back layer</text>',
    ]
    for edge in topology["occlusion_edges"]:
        source = positions.get(int(edge["from_instance_id"]))
        target = positions.get(int(edge["to_instance_id"]))
        if source is None or target is None:
            continue
        fraction = 100.0 * float(edge["fraction_of_target_amodal"])
        elements.append(
            f'<line x1="{source[0]:.1f}" y1="{source[1]+35:.1f}" x2="{target[0]:.1f}" y2="{target[1]-35:.1f}" stroke="#444" stroke-width="2" marker-end="url(#arrow)"/>'
        )
        elements.append(
            f'<text x="{(source[0]+target[0])/2:.1f}" y="{(source[1]+target[1])/2:.1f}" font-family="Arial" font-size="14" fill="#111">{fraction:.1f}%</text>'
        )
    for item in pipes:
        instance_id = int(item["instance_id"])
        x, y = positions[instance_id]
        state = item["occlusion_state"]
        fraction = 100.0 * float(item["visible_fraction"])
        fill = "#DFF2DF" if state == "FULLY_VISIBLE" else "#FFE8B8" if state == "PARTIALLY_OCCLUDED" else "#F5C2C2"
        elements.append(
            f'<rect x="{x-120:.1f}" y="{y-35:.1f}" width="240" height="70" rx="10" fill="{fill}" stroke="#333"/>'
        )
        elements.append(
            f'<text x="{x:.1f}" y="{y-5:.1f}" text-anchor="middle" font-family="Arial" font-size="15">{instance_id}:{item["pipe_id"]}</text>'
        )
        elements.append(
            f'<text x="{x:.1f}" y="{y+18:.1f}" text-anchor="middle" font-family="Arial" font-size="14">{state} / visible {fraction:.1f}%</text>'
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def _file_record(path: Path, root: Path, array: np.ndarray | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    if array is not None:
        record["shape"] = list(array.shape)
        record["dtype"] = str(array.dtype)
    return record


def generate_synthetic_stereo(
    manifest_path: str | Path,
    output_directory: str | Path,
) -> dict[str, Any]:
    """Generate one exact synthetic stereo capture and reference elevation."""

    manifest_file = Path(manifest_path).resolve()
    manifest, manifest_sha = load_json_snapshot(manifest_file)
    if manifest.get("domain") != "synthetic_cad_truth":
        raise ValueError("Synthetic stereo manifest must use domain='synthetic_cad_truth'")
    model_path = (manifest_file.parent / manifest["model"]["path"]).resolve()
    if sha256_file(model_path) != str(manifest["model"]["sha256"]).lower():
        raise ValueError("3MF hash does not match the synthetic manifest")

    reference_video = manifest.get("reference_video", {})
    reference_video_path = None
    if reference_video.get("path"):
        reference_video_path = (manifest_file.parent / reference_video["path"]).resolve()
        if not reference_video_path.is_file():
            raise FileNotFoundError(reference_video_path)
        if sha256_file(reference_video_path) != str(reference_video["sha256"]).lower():
            raise ValueError("Reference video hash does not match the synthetic manifest")

    output_root = Path(output_directory).resolve()
    source_parents = {manifest_file.parent, model_path.parent}
    if output_root in source_parents:
        raise ValueError("Synthetic products require a dedicated output subdirectory")
    if output_root.exists() and not output_root.is_dir():
        raise ValueError("Synthetic output path exists and is not a directory")
    guarded_paths: dict[str, str | Path | None] = {
        "manifest": manifest_file,
        "model": model_path,
        "reference_video": reference_video_path,
    }
    guarded_paths.update(
        {
            f"generated_{index}": output_root / filename
            for index, filename in enumerate(_GENERATED_FILENAMES)
        }
    )
    ensure_paths_distinct(**guarded_paths)
    output_root.mkdir(parents=True, exist_ok=True)

    geometries = load_mesh_geometries(model_path)
    instances = _build_scene(manifest, geometries)
    expected_object_count = int(manifest["model"]["expected_object_count"])
    if len(geometries) != expected_object_count or len(instances) != expected_object_count:
        raise ValueError("3MF/manifest object count does not match expected_object_count")
    if {item.mesh.object_id for item in instances} != set(geometries):
        raise ValueError("Manifest must bind every CAD mesh exactly once")
    rig = manifest["virtual_stereo_rig"]
    left_camera = _camera_from_dict("left", rig["left_camera"])
    right_camera = _camera_from_dict("right", rig["right_camera"])
    elevation_camera = _camera_from_dict("reference_elevation", manifest["reference_elevation"])
    _validate_rectified_rig(rig, left_camera, right_camera)

    left = render_scene(instances, left_camera)
    right = render_scene(instances, right_camera)
    elevation = render_scene(instances, elevation_camera)
    left_topology, _, _ = compute_view_topology(instances, left_camera, left)
    right_topology, _, _ = compute_view_topology(instances, right_camera, right)
    elevation_topology, elevation_full, elevation_visible = compute_view_topology(
        instances, elevation_camera, elevation
    )
    installation_assessment = build_installation_assessment(
        instances,
        {
            "left": left_topology,
            "right": right_topology,
        },
        elevation_topology,
    )
    expected_fully_occluded = set(
        int(value)
        for value in manifest["reference_elevation"].get(
            "expected_fully_occluded_instance_ids", []
        )
    )
    actual_fully_occluded = {
        int(item["instance_id"])
        for item in elevation_topology["pipes"]
        if item["occlusion_state"] == "FULLY_OCCLUDED"
    }
    if actual_fully_occluded != expected_fully_occluded:
        raise ValueError(
            "Reference elevation fully-occluded regression mismatch: "
            f"expected {sorted(expected_fully_occluded)}, got {sorted(actual_fully_occluded)}"
        )

    camera_payload = {
        "schema_version": "1.0",
        "rig_id": rig["rig_id"],
        "calibration_id": rig["calibration_id"],
        "camera_calibration_kind": "synthetic_exact",
        "field_calibration_validated": False,
        "rectified": True,
        "baseline_mm": float(rig["baseline_mm"]),
        "coordinate_convention": {
            "camera": "OpenCV X-right Y-down Z-forward",
            "transform": "T_target_source",
            "pixel_sample": "pixel center at integer+0.5 during rasterization",
        },
        "left_camera": left_camera.as_dict(),
        "right_camera": right_camera.as_dict(),
        "reference_elevation": elevation_camera.as_dict(),
    }
    camera_path = output_root / "camera.json"
    atomic_write_text(camera_path, json.dumps(camera_payload, indent=2) + "\n")
    installation_status_path = output_root / "installation_status.json"
    atomic_write_text(
        installation_status_path,
        json.dumps(installation_assessment, ensure_ascii=False, indent=2) + "\n",
    )

    products: dict[str, dict[str, Any]] = {}
    for name, camera, render, topology, paired_camera, paired_render in (
        ("left", left_camera, left, left_topology, right_camera, right),
        ("right", right_camera, right, right_topology, left_camera, left),
    ):
        rgb_path = _write_png(output_root / f"{name}_rgb.png", render.rgb_bgr)
        textured_rgb = _procedural_textured_rgb(render, camera)
        textured_rgb_path = _write_png(
            output_root / f"{name}_rgb_textured.png", textured_rgb
        )
        depth_preview_path = _write_png(
            output_root / f"{name}_depth_preview.png", _depth_preview(render.depth_z_mm)
        )
        instance_preview_path = _write_png(
            output_root / f"{name}_instance_preview.png", _instance_preview(render.instance_id)
        )
        truth_path = _write_npz(
            output_root / f"{name}_truth.npz",
            depth_z_mm=render.depth_z_mm.astype(np.float32),
            instance_id=render.instance_id.astype(np.uint16),
            disparity_px=_disparity_truth(
                render.depth_z_mm,
                float(camera.fx),
                float(rig["baseline_mm"]),
            ),
            stereo_correspondence_valid=_stereo_correspondence_valid(
                render,
                camera,
                paired_render,
                paired_camera,
            ),
        )
        topology_path = output_root / f"{name}_view_topology.json"
        atomic_write_text(topology_path, json.dumps(topology, indent=2) + "\n")
        products[name] = {
            "camera_id": camera.camera_id,
            "rgb": _file_record(rgb_path, output_root, render.rgb_bgr),
            "rgb_textured": _file_record(
                textured_rgb_path,
                output_root,
                textured_rgb,
            ),
            "truth_npz": _file_record(truth_path, output_root),
            "depth_preview": _file_record(depth_preview_path, output_root),
            "instance_preview": _file_record(instance_preview_path, output_root),
            "view_topology": _file_record(topology_path, output_root),
        }

    elevation_rgb_path = _write_png(output_root / "elevation_reference.png", elevation.rgb_bgr)
    elevation_overlay_path = _write_png(
        output_root / "elevation_amodal_overlay.png",
        _amodal_elevation_overlay(
            elevation,
            instances,
            elevation_full,
            elevation_topology,
        ),
    )
    elevation_topology_path = output_root / "elevation_view_topology.json"
    atomic_write_text(
        elevation_topology_path, json.dumps(elevation_topology, indent=2) + "\n"
    )
    elevation_svg_path = output_root / "elevation_reference.svg"
    atomic_write_text(
        elevation_svg_path,
        build_elevation_svg(
            elevation_camera,
            instances,
            elevation_full,
            elevation_visible,
            elevation_topology,
        ),
    )
    topology_graph_path = output_root / "occlusion_topology_graph.svg"
    atomic_write_text(topology_graph_path, build_topology_graph_svg(elevation_topology))

    matrix_path = output_root / "occlusion_matrix.csv"
    matrix = np.zeros((len(instances), len(instances)), dtype=float)
    id_to_index = {instance.instance_id: index for index, instance in enumerate(instances)}
    for edge in elevation_topology["occlusion_edges"]:
        matrix[
            id_to_index[int(edge["from_instance_id"])],
            id_to_index[int(edge["to_instance_id"])],
        ] = float(edge["fraction_of_target_amodal"])
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(["occluder\\target", *[item.pipe_id for item in instances]])
    for row_index, instance in enumerate(instances):
        writer.writerow([instance.pipe_id, *[f"{value:.8f}" for value in matrix[row_index]]])
    atomic_write_text(matrix_path, stream.getvalue())

    continuous_overlap_path = output_root / "elevation_continuous_overlap.csv"
    continuous_records = _continuous_elevation_overlaps(instances)
    stream = io.StringIO(newline="")
    continuous_fields = [
        "occluder_instance_id",
        "occluder_pipe_id",
        "target_instance_id",
        "target_pipe_id",
        "center_offset_y_mm",
        "vertical_overlap_mm",
        "fraction_of_target_diameter",
        "full_coverage_margin_mm",
        "continuous_state",
    ]
    writer = csv.DictWriter(stream, fieldnames=continuous_fields, lineterminator="\n")
    writer.writeheader()
    for record in continuous_records:
        writer.writerow(record)
    atomic_write_text(continuous_overlap_path, stream.getvalue())

    elevation_products = {
        "rgb": _file_record(elevation_rgb_path, output_root, elevation.rgb_bgr),
        "amodal_overlay": _file_record(elevation_overlay_path, output_root),
        "view_topology": _file_record(elevation_topology_path, output_root),
        "vector_reference": _file_record(elevation_svg_path, output_root),
        "topology_graph": _file_record(topology_graph_path, output_root),
        "occlusion_matrix": _file_record(matrix_path, output_root),
        "continuous_overlap": _file_record(continuous_overlap_path, output_root),
    }

    dataset_manifest = {
        "schema_version": "1.0",
        "dataset_id": manifest["dataset_id"],
        "domain": "synthetic_cad_truth",
        "generated_by": {
            "package": "pipe_twin",
            "version": __version__,
            "renderer": "cpu_triangle_zbuffer_v1",
            "random_seed": None,
        },
        "source": {
            "manifest_path": manifest_file.name,
            "manifest_sha256": manifest_sha,
            "model_path": manifest["model"]["path"],
            "model_sha256": manifest["model"]["sha256"],
            "reference_video": {
                "path": reference_video.get("path"),
                "sha256": reference_video.get("sha256"),
                "role": reference_video.get("role"),
                "verified_but_not_used_for_rendering": reference_video_path is not None,
            },
        },
        "calibration": _file_record(camera_path, output_root),
        "truth_semantics": {
            "depth": "float32 camera optical-axis Z in millimetres; NaN is background",
            "disparity": "float32 rectified magnitude fx*baseline/depth in pixels; NaN is background",
            "stereo_correspondence_valid": "bool source pixel has a same-instance, depth-consistent visible sample in the paired view",
            "instance": "uint16 stable instance_id; 0 is background",
            "identity_source": "manifest instance_id, never RGB color",
            "field_calibration_validated": False,
            "installation_state_inferred": False,
            "installation_state_inferred_scope": "field_observation_only",
            "field_installation_state_inferred": False,
            "synthetic_installation_assessment_generated": True,
            "synthetic_installation_assessment_production_authority": False,
        },
        "instance_catalog": [
            {
                "instance_id": item.instance_id,
                "pipe_id": item.pipe_id,
                "cad_object_id": item.mesh.object_id,
                "cad_uuid": item.mesh.uuid,
                "layer_id": item.layer_id,
                "color_class": item.color_class,
                "color_srgb": item.mesh.color_srgb,
                "nominal_diameter_mm": item.nominal_diameter_mm,
                "cad_measured_diameter_mm": item.mesh.measured_diameter_mm,
                "ground_truth_installation_state": (
                    item.ground_truth_installation_state
                ),
                "centerline_world_mm": item.centerline_world_mm.tolist(),
            }
            for item in instances
        ],
        "capture": {
            "capture_group_id": "pipe-group2-static-all-active",
            "scene_state_id": "all-nine-active",
            "synthetic_sync_error_ns": 0,
            "products": products,
        },
        "reference_elevation": elevation_products,
        "installation_assessment": _file_record(
            installation_status_path, output_root
        ),
        "limitations": [
            "This dataset validates CAD projection, stereo geometry, depth ownership, and occlusion topology only.",
            "It does not validate field camera calibration, reflective material response, sensor noise, D22/D50 physical accuracy, or installation state.",
            "Flat RGB is not suitable for validating dense stereo matching; *_rgb_textured.png adds deterministic artificial world-anchored texture for OpenCV matcher experiments.",
            "The rasterizer does not clip triangles crossing near/far planes; this fixture stays fully inside its clipping range.",
            "Occlusion matrices are per-render pixel truth; elevation_continuous_overlap.csv separately reports nominal continuous cross-section overlap.",
            "Synthetic installation states are non-production reference assessments; NOT_INSTALLED requires qualified negative evidence that this all-active fixture does not contain.",
            "The current renderer accepts only active INSTALLED scene instances; a missing-pipe fixture must separate the complete design catalog from the rendered active-instance set.",
            "FULLY_OCCLUDED and NOT_OBSERVED never imply NOT_INSTALLED.",
        ],
    }
    dataset_manifest_path = output_root / "dataset_manifest.json"
    atomic_write_text(
        dataset_manifest_path,
        json.dumps(dataset_manifest, ensure_ascii=False, indent=2) + "\n",
    )
    return dataset_manifest
