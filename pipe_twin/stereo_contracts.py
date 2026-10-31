"""Stable data contracts shared by stereo analysis and elevation workflows.

The analyzer keeps validation and algorithms behind these small immutable
records.  Keeping them in a dependency-light module prevents GUI, zone and
temporal helpers from importing the full stereo implementation just to build
or inspect a calibration/depth value.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cad_model import CadObject, CadScene as MeshCadScene


class StereoAnalysisError(ValueError):
    """Raised when a stereo manifest or bound asset fails closed."""


@dataclass(frozen=True)
class PipePrior:
    """One stable business identity bound to CAD geometry."""

    pipe_id: str
    cad_object_id: str
    cad_uuid: str | None
    instance_id: int | None
    layer_id: str
    color_class: str
    color_srgb: str
    nominal_diameter_mm: float
    centerline_world_mm: np.ndarray


@dataclass(frozen=True)
class CameraCalibration:
    """One camera expressed in the CAD/world coordinate system."""

    role: str
    camera_id: str
    width: int
    height: int
    intrinsic: np.ndarray
    distortion: np.ndarray
    rotation_world_to_camera: np.ndarray
    center_world_mm: np.ndarray
    rectification_matrix: np.ndarray | None
    projection_matrix: np.ndarray | None

    @property
    def fx(self) -> float:
        if self.projection_matrix is not None:
            return float(self.projection_matrix[0, 0])
        return float(self.intrinsic[0, 0])

    @property
    def fy(self) -> float:
        if self.projection_matrix is not None:
            return float(self.projection_matrix[1, 1])
        return float(self.intrinsic[1, 1])

    @property
    def rectified_intrinsic(self) -> np.ndarray:
        """Intrinsic matrix for the pixels supplied to the analyzer."""
        if self.projection_matrix is not None:
            return np.asarray(self.projection_matrix[:, :3], dtype=np.float64)
        return self.intrinsic

    @property
    def rotation_world_to_rectified_camera(self) -> np.ndarray:
        """CAD-world to the rectified image camera frame."""
        if self.rectification_matrix is not None:
            return self.rectification_matrix @ self.rotation_world_to_camera
        return self.rotation_world_to_camera


@dataclass(frozen=True)
class StereoCalibration:
    calibration_id: str
    validated: bool
    registration_validated: bool
    rectified: bool
    baseline_mm: float
    max_sync_delta_ms: float
    left: CameraCalibration
    right: CameraCalibration
    # Derived from the ID-bound operator record; never changes validated or
    # registration_validated. Zone crops retain this relative-use permission.
    relative_elevation_provisional: bool = False


@dataclass(frozen=True)
class CadScene:
    model_path: Path
    model_format: str
    expected_sha256: str
    actual_sha256: str
    pipes: tuple[PipePrior, ...]
    object_binding_validation: str
    mesh_scene: MeshCadScene
    objects_by_pipe_id: dict[str, CadObject]


@dataclass(frozen=True)
class StereoDepthResult:
    """Rectified depth in each camera coordinate system."""

    left_depth_mm: np.ndarray
    right_depth_mm: np.ndarray
    left_valid: np.ndarray
    right_valid: np.ndarray
    audit: dict[str, Any]


@dataclass(frozen=True)
class PipeProjection:
    mask: np.ndarray
    expected_depth_mm: float | None
    centerline_px: list[list[float]] | None
    predicted_diameter_px: float | None
    amodal_pixels: int
    nominal_visible_pixels: int
    nominal_occlusion_state: str


__all__ = [
    "CadScene",
    "CameraCalibration",
    "PipePrior",
    "PipeProjection",
    "StereoAnalysisError",
    "StereoCalibration",
    "StereoDepthResult",
]
