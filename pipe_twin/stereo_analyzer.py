"""Manifest-bound stereo still-photo analysis against a CAD pipe prior.

The public entry point in this module is deliberately independent from the
legacy M0 pipeline.  It accepts explicitly paired left/right photographs,
projects stable CAD pipe identities into both views, and emits one conservative
installation state per pipe.

This is an engineering prototype, not a field acceptance certificate.  In
particular, invalid disparity is never treated as free space and a pipe that is
fully occluded in both current views is always ``UNKNOWN``.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import cv2
import numpy as np

from . import __version__
from .cad_model import (
    CadModelError,
    CadObject,
    CadScene as MeshCadScene,
    load_cad_scene as load_mesh_cad_scene,
)
from .photo_capture import load_photo_snapshot
from .pipeline import ensure_paths_distinct
from .state import (
    INSTALLATION_STATES,
    NegativeInstallationEvidence,
    classify_installation_state,
    installation_state_label_zh,
)
from .logging_config import get_logger, log_event

_LOGGER = get_logger("stereo_analyzer")


_SHA256_PATTERN = frozenset("0123456789abcdefABCDEF")
_ORIENTATION_POLICY = "RAW_PIXELS_NO_EXIF_TRANSFORM"
_VIEW_ROLES = ("left", "right")
_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_IMAGE_PIXELS = 50_000_000
_MAX_PROJECTED_PIPE_PIXELS = 250_000_000
_MAX_FOCAL_LENGTH_PIXELS = 10_000_000.0
_MAX_WORLD_COORD_MM = 1_000_000_000.0
_MAX_SGBM_DISPARITIES = 8192
_MAX_SGBM_BLOCK_SIZE = 255
_MAX_SGBM_SPECKLE_WINDOW = 1_000_000
_MAX_SGBM_DISP12_DIFF = 4096
_MAX_MORPH_RADIUS_PIXELS = 1024
STEREO_ALGORITHM_REVISION = "sgbm-low-light-v1"


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


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StereoAnalysisError(f"{field} must be a non-empty string")
    return value


def _safe_identifier(value: object, field: str) -> str:
    result = _require_string(value, field)
    if _SAFE_IDENTIFIER.fullmatch(result) is None:
        raise StereoAnalysisError(
            f"{field} must use 1-128 ASCII letters, digits, '.', '_', or '-'"
        )
    return result


def _finite_number(
    value: object,
    field: str,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> float:
    if type(value) not in (int, float):
        raise StereoAnalysisError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise StereoAnalysisError(f"{field} must be a finite number") from error
    if not math.isfinite(result):
        raise StereoAnalysisError(f"{field} must be a finite number")
    if positive and result <= 0:
        raise StereoAnalysisError(f"{field} must be positive")
    if non_negative and result < 0:
        raise StereoAnalysisError(f"{field} must be non-negative")
    return result


def _positive_integer(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise StereoAnalysisError(f"{field} must be a positive integer")
    return value


def _sha256_text(value: object, field: str) -> str:
    result = _require_string(value, field)
    if len(result) != 64 or any(character not in _SHA256_PATTERN for character in result):
        raise StereoAnalysisError(f"{field} must contain 64 hexadecimal characters")
    return result.lower()


def _parse_timestamp(value: object, field: str) -> datetime:
    text = _require_string(value, field)
    normalized = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise StereoAnalysisError(f"{field} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise StereoAnalysisError(f"{field} must include a UTC offset or Z")
    return parsed


def _relative_asset_path(manifest_file: Path, value: object, field: str) -> Path:
    relative = _require_string(value, field)
    if "\\" in relative:
        raise StereoAnalysisError(f"{field} must use portable forward slashes")
    portable = PurePosixPath(relative)
    windows = PureWindowsPath(relative)
    if portable.is_absolute() or windows.is_absolute() or windows.drive:
        raise StereoAnalysisError(f"{field} must be relative to the manifest directory")
    if ".." in portable.parts:
        raise StereoAnalysisError(f"{field} must not leave the manifest directory")
    root = manifest_file.resolve().parent
    resolved = (root / Path(*portable.parts)).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise StereoAnalysisError(f"{field} resolves outside the manifest directory") from error
    return resolved


def _hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size <= 0 or size > _MAX_MANIFEST_BYTES:
        raise StereoAnalysisError(
            f"Manifest size must be between 1 and {_MAX_MANIFEST_BYTES} bytes"
        )
    raw = path.read_bytes()
    if len(raw) != size:
        raise StereoAnalysisError("Manifest changed while it was being read")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise StereoAnalysisError(f"Invalid UTF-8 JSON manifest: {path}") from error
    if not isinstance(payload, dict):
        raise StereoAnalysisError("Stereo manifest must be a JSON object")
    return payload, _hash_bytes(raw)


def _matrix(value: object, shape: tuple[int, ...], field: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise StereoAnalysisError(f"{field} must be a numeric matrix") from error
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise StereoAnalysisError(f"{field} must be finite with shape {shape}")
    return result


def _camera_from_manifest(
    role: str,
    payload: object,
    *,
    rectified: bool,
) -> CameraCalibration:
    if not isinstance(payload, dict):
        raise StereoAnalysisError(f"stereo_calibration.{role}_camera must be an object")
    camera_id = _require_string(payload.get("camera_id", role), f"{role}.camera_id")
    width = _positive_integer(payload.get("width"), f"{role}.width")
    height = _positive_integer(payload.get("height"), f"{role}.height")
    if width * height > _MAX_IMAGE_PIXELS:
        raise StereoAnalysisError(
            f"{role} image dimensions exceed {_MAX_IMAGE_PIXELS} pixels"
        )

    if "K" in payload:
        intrinsic = _matrix(payload["K"], (3, 3), f"{role}.K")
    else:
        fx = _finite_number(payload.get("fx"), f"{role}.fx", positive=True)
        fy = _finite_number(payload.get("fy"), f"{role}.fy", positive=True)
        cx = _finite_number(payload.get("cx"), f"{role}.cx")
        cy = _finite_number(payload.get("cy"), f"{role}.cy")
        intrinsic = np.asarray([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    canonical_intrinsic = np.asarray(
        [
            [intrinsic[0, 0], 0.0, intrinsic[0, 2]],
            [0.0, intrinsic[1, 1], intrinsic[1, 2]],
            [0.0, 0.0, 1.0],
        ]
    )
    if (
        intrinsic[0, 0] <= 0
        or intrinsic[1, 1] <= 0
        or intrinsic[0, 0] > _MAX_FOCAL_LENGTH_PIXELS
        or intrinsic[1, 1] > _MAX_FOCAL_LENGTH_PIXELS
        or abs(intrinsic[0, 2]) > _MAX_WORLD_COORD_MM
        or abs(intrinsic[1, 2]) > _MAX_WORLD_COORD_MM
        or not np.allclose(intrinsic, canonical_intrinsic, atol=1e-9)
    ):
        raise StereoAnalysisError(f"{role}.K is not a valid pinhole intrinsic matrix")

    distortion_value = payload.get("distortion_coefficients", payload.get("D", []))
    try:
        distortion = np.asarray(distortion_value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError, OverflowError) as error:
        raise StereoAnalysisError(f"{role}.distortion_coefficients must be numeric") from error
    if len(distortion) not in {0, 4, 5, 8, 12, 14} or not np.all(np.isfinite(distortion)):
        raise StereoAnalysisError(
            f"{role}.distortion_coefficients must contain 0, 4, 5, 8, 12, or 14 values"
        )
    if len(distortion) == 0:
        distortion = np.zeros(5, dtype=np.float64)

    rotation = _matrix(
        payload.get("rotation_world_to_camera"),
        (3, 3),
        f"{role}.rotation_world_to_camera",
    )
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) or not math.isclose(
        float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6
    ):
        raise StereoAnalysisError(f"{role}.rotation_world_to_camera must be a rotation")
    center = _matrix(payload.get("center_world_mm"), (3,), f"{role}.center_world_mm")
    if np.any(np.abs(center) > _MAX_WORLD_COORD_MM):
        raise StereoAnalysisError(
            f"{role}.center_world_mm exceeds the supported coordinate bound"
        )

    rectification_matrix: np.ndarray | None = None
    projection_matrix: np.ndarray | None = None
    has_rectified_projection = (
        payload.get("rectification_matrix") is not None
        or payload.get("projection_matrix") is not None
    )
    if not rectified or has_rectified_projection:
        rectification_matrix = _matrix(
            payload.get("rectification_matrix"),
            (3, 3),
            f"{role}.rectification_matrix",
        )
        projection_matrix = _matrix(
            payload.get("projection_matrix"),
            (3, 4),
            f"{role}.projection_matrix",
        )
        if projection_matrix[0, 0] <= 0 or projection_matrix[1, 1] <= 0:
            raise StereoAnalysisError(f"{role}.projection_matrix must have positive focal lengths")

    return CameraCalibration(
        role=role,
        camera_id=camera_id,
        width=width,
        height=height,
        intrinsic=intrinsic,
        distortion=distortion,
        rotation_world_to_camera=rotation,
        center_world_mm=center,
        rectification_matrix=rectification_matrix,
        projection_matrix=projection_matrix,
    )


def _calibration_from_manifest(payload: object) -> StereoCalibration:
    if not isinstance(payload, dict):
        raise StereoAnalysisError("stereo_calibration must be an object")
    calibration_id = _require_string(payload.get("calibration_id"), "calibration_id")
    if type(payload.get("validated")) is not bool:
        raise StereoAnalysisError("stereo_calibration.validated must be a boolean")
    if type(payload.get("registration_validated")) is not bool:
        raise StereoAnalysisError(
            "stereo_calibration.registration_validated must be a boolean"
        )
    if type(payload.get("rectified")) is not bool:
        raise StereoAnalysisError("stereo_calibration.rectified must be a boolean")
    rectified = payload["rectified"]
    # The depth gate below deliberately uses the rectified pinhole relation
    # ``Z = fx * B / disparity`` and a horizontal left/right matcher.  Raw
    # camera K/D/extrinsics cannot safely be fed into that relation unless the
    # exact output of ``stereoRectify`` (including a common P matrix, baseline
    # sign and epipolar residual) has also been audited.  The first field
    # release accepts only an already-rectified pair and fails closed rather
    # than silently producing a biased depth map from plausible-looking raw
    # calibration fields.  A future raw-input adapter can add the complete
    # stereoRectify audit without changing the state semantics.
    if not rectified:
        raise StereoAnalysisError(
            "stereo_calibration.rectified must be true; raw/non-rectified input "
            "requires a separately audited stereoRectify preprocessing step"
        )
    baseline = _finite_number(
        payload.get("baseline_mm"), "stereo_calibration.baseline_mm", positive=True
    )
    max_sync_delta = _finite_number(
        payload.get("max_sync_delta_ms"),
        "stereo_calibration.max_sync_delta_ms",
        non_negative=True,
    )
    left = _camera_from_manifest("left", payload.get("left_camera"), rectified=rectified)
    right = _camera_from_manifest("right", payload.get("right_camera"), rectified=rectified)
    if (left.width, left.height) != (right.width, right.height):
        raise StereoAnalysisError("Left/right calibrated image dimensions must match")
    measured_baseline = float(np.linalg.norm(right.center_world_mm - left.center_world_mm))
    if not math.isclose(measured_baseline, baseline, rel_tol=1e-4, abs_tol=1e-3):
        raise StereoAnalysisError(
            f"baseline_mm={baseline} disagrees with camera centres ({measured_baseline})"
        )
    if rectified:
        if left.rectification_matrix is not None or right.rectification_matrix is not None:
            if left.rectification_matrix is None or right.rectification_matrix is None:
                raise StereoAnalysisError("Rectified cameras must provide both rectification matrices")
            if left.projection_matrix is None or right.projection_matrix is None:
                raise StereoAnalysisError("Rectified cameras must provide both projection matrices")
            for role, camera in (("left", left), ("right", right)):
                if not np.allclose(
                    camera.rectification_matrix @ camera.rectification_matrix.T,
                    np.eye(3),
                    atol=1e-5,
                ) or not math.isclose(
                    float(np.linalg.det(camera.rectification_matrix)), 1.0, abs_tol=1e-5
                ):
                    raise StereoAnalysisError(f"{role}.rectification_matrix must be a rotation")
                if camera.projection_matrix[0, 0] <= 0 or camera.projection_matrix[1, 1] <= 0:
                    raise StereoAnalysisError(f"{role}.projection_matrix must have positive focal lengths")
            left_p, right_p = left.projection_matrix, right.projection_matrix
            if not np.allclose(left_p[:2, :3], right_p[:2, :3], rtol=1e-5, atol=1e-6):
                raise StereoAnalysisError(
                    "Rectified left/right projection matrices must share fx, fy, cx, and cy"
                )
            encoded_baseline = -float(right_p[0, 3]) / float(right_p[0, 0])
            if encoded_baseline <= 0 or not math.isclose(
                encoded_baseline, baseline, rel_tol=1e-4, abs_tol=1e-3
            ):
                raise StereoAnalysisError(
                    "Rectified projection matrix baseline disagrees with baseline_mm"
                )
        else:
            if not np.allclose(
                left.rotation_world_to_camera,
                right.rotation_world_to_camera,
                atol=1e-5,
            ):
                raise StereoAnalysisError("Rectified camera rotations must match")
            baseline_camera = left.rotation_world_to_camera @ (
                right.center_world_mm - left.center_world_mm
            )
            if baseline_camera[0] <= 0 or np.linalg.norm(baseline_camera[1:]) > max(
                1e-3, baseline * 1e-4
            ):
                raise StereoAnalysisError(
                    "Rectified stereo baseline must point along positive camera X"
                )
            if not np.allclose(
                left.intrinsic,
                right.intrinsic,
                rtol=1e-5,
                atol=1e-6,
            ):
                raise StereoAnalysisError(
                    "Rectified left/right K values must use the same fx, fy, cx, and cy"
                )
    return StereoCalibration(
        calibration_id=calibration_id,
        validated=payload["validated"],
        registration_validated=payload["registration_validated"],
        rectified=rectified,
        baseline_mm=baseline,
        max_sync_delta_ms=max_sync_delta,
        left=left,
        right=right,
    )


def _pipe_from_manifest(payload: object, index: int) -> PipePrior:
    field = f"model.pipes[{index}]"
    if not isinstance(payload, dict):
        raise StereoAnalysisError(f"{field} must be an object")
    pipe_id = _require_string(payload.get("pipe_id"), f"{field}.pipe_id")
    cad_object_id = _require_string(payload.get("cad_object_id"), f"{field}.cad_object_id")
    raw_cad_uuid = payload.get("cad_uuid")
    cad_uuid = (
        _require_string(raw_cad_uuid, f"{field}.cad_uuid")
        if raw_cad_uuid is not None
        else None
    )
    raw_instance = payload.get("instance_id")
    if raw_instance is not None and (type(raw_instance) is not int or raw_instance <= 0):
        raise StereoAnalysisError(f"{field}.instance_id must be a positive integer or null")
    layer_id = _require_string(payload.get("layer_id", "L0"), f"{field}.layer_id")
    color_class = _require_string(
        payload.get("color_class", payload.get("appearance_color")),
        f"{field}.color_class",
    )
    color_srgb = _require_string(
        payload.get("color_srgb", payload.get("nominal_color_srgb")),
        f"{field}.color_srgb",
    ).upper()
    if (
        len(color_srgb) not in {7, 9}
        or not color_srgb.startswith("#")
        or any(character not in _SHA256_PATTERN for character in color_srgb[1:])
    ):
        raise StereoAnalysisError(f"{field}.color_srgb must be #RRGGBB or #RRGGBBAA")
    diameter = _finite_number(
        payload.get("nominal_diameter_mm"),
        f"{field}.nominal_diameter_mm",
        positive=True,
    )
    centerline = _matrix(
        payload.get("centerline_world_mm"),
        (2, 3),
        f"{field}.centerline_world_mm",
    )
    if np.any(np.abs(centerline) > _MAX_WORLD_COORD_MM):
        raise StereoAnalysisError(
            f"{field}.centerline_world_mm exceeds the supported coordinate bound"
        )
    if float(np.linalg.norm(centerline[1] - centerline[0])) <= 0:
        raise StereoAnalysisError(f"{field}.centerline_world_mm must have non-zero length")
    return PipePrior(
        pipe_id=pipe_id,
        cad_object_id=cad_object_id,
        cad_uuid=cad_uuid,
        instance_id=raw_instance,
        layer_id=layer_id,
        color_class=color_class,
        color_srgb=color_srgb[:7],
        nominal_diameter_mm=diameter,
        centerline_world_mm=centerline,
    )


def _load_cad_scene(manifest_file: Path, model: object) -> CadScene:
    if not isinstance(model, dict):
        raise StereoAnalysisError("model must be an object")
    if model.get("unit") != "millimeter":
        raise StereoAnalysisError("model.unit must be 'millimeter'")
    model_path = _relative_asset_path(manifest_file, model.get("path"), "model.path")
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    expected_hash = _sha256_text(model.get("sha256"), "model.sha256")
    actual_hash = _hash_file(model_path)
    if actual_hash != expected_hash:
        raise StereoAnalysisError(
            f"CAD SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )
    raw_pipes = model.get("pipes")
    if not isinstance(raw_pipes, list) or not raw_pipes:
        raise StereoAnalysisError("model.pipes must be a non-empty list")
    pipes = tuple(_pipe_from_manifest(item, index) for index, item in enumerate(raw_pipes))
    pipe_ids = [item.pipe_id for item in pipes]
    cad_ids = [item.cad_object_id.casefold() for item in pipes]
    instance_ids = [item.instance_id for item in pipes if item.instance_id is not None]
    if len(pipe_ids) != len(set(pipe_ids)):
        raise StereoAnalysisError("model pipe_id values must be unique")
    if len(cad_ids) != len(set(cad_ids)):
        raise StereoAnalysisError("Each CAD object may be bound to only one pipe_id")
    if len(instance_ids) != len(set(instance_ids)):
        raise StereoAnalysisError("model instance_id values must be unique")

    try:
        mesh_scene = load_mesh_cad_scene(
            model_path,
            required_object_ids={pipe.cad_object_id for pipe in pipes},
            stl_unit=model.get("source_unit") if model_path.suffix.lower() == ".stl" else None,
        )
    except CadModelError as error:
        raise StereoAnalysisError(str(error)) from error
    if mesh_scene.source_sha256 != actual_hash:
        raise StereoAnalysisError("CAD loader did not consume the verified model snapshot")

    objects_by_pipe: dict[str, CadObject] = {}
    for pipe in pipes:
        if mesh_scene.source_format == "3dm":
            cad_object = mesh_scene.by_guid.get(pipe.cad_object_id.lower())
            if cad_object is None:
                cad_object = mesh_scene.by_object_id.get(pipe.cad_object_id.lower())
        else:
            cad_object = mesh_scene.by_object_id.get(pipe.cad_object_id)
        if cad_object is None:
            raise StereoAnalysisError(
                f"CAD model is missing mesh-bound object {pipe.cad_object_id!r} "
                f"for pipe {pipe.pipe_id!r}"
            )
        if pipe.cad_uuid is not None:
            if cad_object.guid is None or pipe.cad_uuid.casefold() != cad_object.guid.casefold():
                raise StereoAnalysisError(
                    f"CAD GUID for pipe {pipe.pipe_id!r} does not match the bound object"
                )
        if cad_object.pipe_id is not None and cad_object.pipe_id != pipe.pipe_id:
            raise StereoAnalysisError(
                f"CAD object {pipe.cad_object_id!r} pipe_id user string does not "
                f"match manifest pipe_id {pipe.pipe_id!r}"
            )
        if cad_object.vertex_count < 3 or cad_object.triangle_count < 1:
            raise StereoAnalysisError(
                f"CAD object {pipe.cad_object_id!r} has no usable triangle mesh"
            )
        if np.any(np.abs(np.asarray(cad_object.bbox_min_mm)) > _MAX_WORLD_COORD_MM) or np.any(
            np.abs(np.asarray(cad_object.bbox_max_mm)) > _MAX_WORLD_COORD_MM
        ):
            raise StereoAnalysisError(
                f"CAD object {pipe.cad_object_id!r} exceeds the supported world-coordinate bound"
            )
        # The sidecar centerline is a semantic label/width prior, never the
        # source of the projected silhouette.  Still require its endpoints to
        # lie in the verified mesh bounding box so a stale or malicious sidecar
        # cannot silently move an identity to another part of the scene.
        tolerance_mm = max(2.0, 0.25 * pipe.nominal_diameter_mm)
        endpoints = pipe.centerline_world_mm
        minimum = np.asarray(cad_object.bbox_min_mm) - tolerance_mm
        maximum = np.asarray(cad_object.bbox_max_mm) + tolerance_mm
        if np.any(endpoints < minimum) or np.any(endpoints > maximum):
            raise StereoAnalysisError(
                f"Manifest centerline for pipe {pipe.pipe_id!r} is outside "
                f"the bound CAD mesh {pipe.cad_object_id!r}"
            )
        objects_by_pipe[pipe.pipe_id] = cad_object
    binding_validation = (
        "3DM_GUID_AND_MESH_VALIDATED"
        if mesh_scene.source_format == "3dm"
        else "STL_COMPONENT_ID_AND_MESH_VALIDATED"
        if mesh_scene.source_format == "stl"
        else "3MF_OBJECT_ID_AND_MESH_VALIDATED"
    )
    return CadScene(
        model_path=model_path,
        model_format=mesh_scene.source_format,
        expected_sha256=expected_hash,
        actual_sha256=actual_hash,
        pipes=pipes,
        object_binding_validation=binding_validation,
        mesh_scene=mesh_scene,
        objects_by_pipe_id=objects_by_pipe,
    )


def _analysis_config(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise StereoAnalysisError("analysis must be an object")
    defaults: dict[str, Any] = {
        "minimum_focus_laplacian_variance": 0.0,
        "minimum_luminance_p05": 0.0,
        "maximum_luminance_p95": 255.0,
        "minimum_amodal_pixels": 25,
        "minimum_valid_depth_fraction": 0.25,
        "minimum_target_depth_fraction": 0.10,
        "minimum_color_support_fraction": 0.10,
        "minimum_width_assessable_fraction": 0.25,
        "maximum_width_relative_error": 0.60,
        "depth_tolerance_mm": 50.0,
        "occlusion_margin_mm": 30.0,
        "free_space_margin_mm": 40.0,
        "minimum_free_space_fraction": 0.65,
        "fully_occluded_fraction": 0.90,
        "color_delta_e76_tolerance": 70.0,
        "minimum_local_contrast_delta_e76": 8.0,
        "minimum_repeated_absence_captures": 2,
        "minimum_depth_mm": 100.0,
        "maximum_depth_mm": 5000.0,
        "left_right_consistency_px": 2.0,
        "stereo_matching": {
            "preprocessing": "none",
            "min_disparity": 0,
            "num_disparities": 128,
            "block_size": 5,
            "uniqueness_ratio": 5,
            "speckle_window_size": 50,
            "speckle_range": 2,
            "disp12_max_diff": 2,
        },
    }
    config = {**defaults, **payload}
    if isinstance(payload.get("stereo_matching"), dict):
        config["stereo_matching"] = {
            **defaults["stereo_matching"],
            **payload["stereo_matching"],
        }
    numeric_non_negative = (
        "minimum_focus_laplacian_variance",
        "minimum_luminance_p05",
        "maximum_luminance_p95",
        "minimum_valid_depth_fraction",
        "minimum_target_depth_fraction",
        "minimum_color_support_fraction",
        "minimum_width_assessable_fraction",
        "maximum_width_relative_error",
        "depth_tolerance_mm",
        "occlusion_margin_mm",
        "free_space_margin_mm",
        "minimum_free_space_fraction",
        "fully_occluded_fraction",
        "color_delta_e76_tolerance",
        "minimum_local_contrast_delta_e76",
        "minimum_depth_mm",
        "maximum_depth_mm",
        "left_right_consistency_px",
    )
    for key in numeric_non_negative:
        config[key] = _finite_number(config[key], f"analysis.{key}", non_negative=True)
    for key in (
        "minimum_valid_depth_fraction",
        "minimum_target_depth_fraction",
        "minimum_color_support_fraction",
        "minimum_width_assessable_fraction",
        "minimum_free_space_fraction",
        "fully_occluded_fraction",
    ):
        if config[key] > 1:
            raise StereoAnalysisError(f"analysis.{key} must not exceed 1")
    if config["left_right_consistency_px"] > _MAX_SGBM_DISP12_DIFF:
        raise StereoAnalysisError(
            "analysis.left_right_consistency_px exceeds the supported range"
        )
    if config["minimum_luminance_p05"] > config["maximum_luminance_p95"]:
        raise StereoAnalysisError("analysis luminance bounds are reversed")
    if config["minimum_depth_mm"] >= config["maximum_depth_mm"]:
        raise StereoAnalysisError("analysis depth bounds are reversed")
    config["minimum_amodal_pixels"] = _positive_integer(
        config["minimum_amodal_pixels"], "analysis.minimum_amodal_pixels"
    )
    config["minimum_repeated_absence_captures"] = _positive_integer(
        config["minimum_repeated_absence_captures"],
        "analysis.minimum_repeated_absence_captures",
    )
    if config["minimum_repeated_absence_captures"] < 2:
        raise StereoAnalysisError(
            "analysis.minimum_repeated_absence_captures must be at least 2"
        )
    matching = config["stereo_matching"]
    if not isinstance(matching, dict):
        raise StereoAnalysisError("analysis.stereo_matching must be an object")
    if matching.get("preprocessing") not in ("none", "low_light"):
        raise StereoAnalysisError("stereo preprocessing must be 'none' or 'low_light'")
    for key in (
        "min_disparity",
        "num_disparities",
        "block_size",
        "uniqueness_ratio",
        "speckle_window_size",
        "speckle_range",
        "disp12_max_diff",
    ):
        if type(matching.get(key)) is not int:
            raise StereoAnalysisError(f"analysis.stereo_matching.{key} must be an integer")
    if matching["min_disparity"] < 0:
        raise StereoAnalysisError(
            "stereo min_disparity must be non-negative for the horizontal rig"
        )
    if (
        matching["num_disparities"] <= 0
        or matching["num_disparities"] % 16
        or matching["num_disparities"] > _MAX_SGBM_DISPARITIES
    ):
        raise StereoAnalysisError(
            "stereo num_disparities must be a positive multiple of 16 "
            f"not exceeding {_MAX_SGBM_DISPARITIES}"
        )
    if (
        matching["block_size"] < 3
        or matching["block_size"] % 2 == 0
        or matching["block_size"] > _MAX_SGBM_BLOCK_SIZE
    ):
        raise StereoAnalysisError(
            "stereo block_size must be odd, at least 3, and no larger than "
            f"{_MAX_SGBM_BLOCK_SIZE}"
        )
    if not 0 <= matching["uniqueness_ratio"] <= 100:
        raise StereoAnalysisError("stereo uniqueness_ratio must be between 0 and 100")
    if not 0 <= matching["speckle_window_size"] <= _MAX_SGBM_SPECKLE_WINDOW:
        raise StereoAnalysisError(
            "stereo speckle_window_size is outside the supported range"
        )
    if not 0 <= matching["speckle_range"] <= 255:
        raise StereoAnalysisError("stereo speckle_range must be between 0 and 255")
    if not 0 <= matching["disp12_max_diff"] <= _MAX_SGBM_DISP12_DIFF:
        raise StereoAnalysisError("stereo disp12_max_diff is outside the supported range")
    return config


def _capture_groups_from_manifest(
    manifest_file: Path,
    payload: object,
    calibration: StereoCalibration,
) -> tuple[str, float, list[dict[str, Any]]]:
    if not isinstance(payload, dict):
        raise StereoAnalysisError("capture must be an object")
    if payload.get("kind") != "stereo_still_capture_set":
        raise StereoAnalysisError("capture.kind must be 'stereo_still_capture_set'")
    if payload.get("camera_layout") != "stereo":
        raise StereoAnalysisError("capture.camera_layout must be 'stereo'")
    run_id = _require_string(payload.get("capture_group_id"), "capture.capture_group_id")
    interval = _finite_number(
        payload.get("interval_minutes"), "capture.interval_minutes", positive=True
    )
    if not 5 <= interval <= 60:
        raise StereoAnalysisError("capture.interval_minutes must be between 5 and 60")
    groups = payload.get("capture_groups")
    if not isinstance(groups, list) or not groups:
        raise StereoAnalysisError("capture.capture_groups must be a non-empty list")
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    previous_time: datetime | None = None
    normalized: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        field = f"capture.capture_groups[{index}]"
        if not isinstance(group, dict):
            raise StereoAnalysisError(f"{field} must be an object")
        capture_id = _safe_identifier(group.get("capture_id"), f"{field}.capture_id")
        if capture_id in seen_ids:
            raise StereoAnalysisError(f"Duplicate capture_id: {capture_id!r}")
        seen_ids.add(capture_id)
        views = group.get("views")
        if not isinstance(views, dict) or set(views) != set(_VIEW_ROLES):
            raise StereoAnalysisError(f"{field}.views must contain exactly left and right")
        times: dict[str, datetime] = {}
        normalized_views: dict[str, dict[str, Any]] = {}
        for role in _VIEW_ROLES:
            view = views[role]
            if not isinstance(view, dict):
                raise StereoAnalysisError(f"{field}.views.{role} must be an object")
            expected_camera = getattr(calibration, role)
            if view.get("camera_id") != expected_camera.camera_id:
                raise StereoAnalysisError(
                    f"{field}.views.{role}.camera_id does not match calibration"
                )
            if view.get("expected_width") != expected_camera.width or view.get(
                "expected_height"
            ) != expected_camera.height:
                raise StereoAnalysisError(
                    f"{field}.views.{role} dimensions do not match calibration"
                )
            if view.get("orientation_policy") != _ORIENTATION_POLICY:
                raise StereoAnalysisError(
                    f"{field}.views.{role}.orientation_policy must be {_ORIENTATION_POLICY}"
                )
            _require_string(
                view.get("timestamp_source"),
                f"{field}.views.{role}.timestamp_source",
            )
            relative_path = _require_string(view.get("path"), f"{field}.views.{role}.path")
            resolved = _relative_asset_path(
                manifest_file, relative_path, f"{field}.views.{role}.path"
            )
            path_key = str(resolved).casefold()
            if path_key in seen_paths:
                raise StereoAnalysisError("Every left/right photo path must be unique")
            seen_paths.add(path_key)
            _sha256_text(view.get("sha256"), f"{field}.views.{role}.sha256")
            times[role] = _parse_timestamp(
                view.get("captured_at"), f"{field}.views.{role}.captured_at"
            )
            normalized_views[role] = dict(view)
        delta_ms = abs((times["right"] - times["left"]).total_seconds() * 1000.0)
        pair_time = max(times.values())
        if previous_time is not None and pair_time <= previous_time:
            raise StereoAnalysisError("capture_groups must be strictly chronological")
        previous_time = pair_time
        normalized.append(
            {
                "capture_id": capture_id,
                "views": normalized_views,
                "sync_delta_ms": delta_ms,
                "sync_valid": delta_ms <= calibration.max_sync_delta_ms,
                "captured_at": pair_time.isoformat(),
            }
        )
    return run_id, interval, normalized


def _quality(image: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    p05, p50, p95 = (float(value) for value in np.percentile(gray, (5, 50, 95)))
    focus = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    passed = bool(
        focus >= config["minimum_focus_laplacian_variance"]
        and p05 >= config["minimum_luminance_p05"]
        and p95 <= config["maximum_luminance_p95"]
    )
    reasons: list[str] = []
    if focus < config["minimum_focus_laplacian_variance"]:
        reasons.append("FOCUS_BELOW_THRESHOLD")
    if p05 < config["minimum_luminance_p05"]:
        reasons.append("DARK_EXPOSURE")
    if p95 > config["maximum_luminance_p95"]:
        reasons.append("BRIGHT_EXPOSURE")
    diagnostics = _image_signal_diagnostics(gray)
    return {
        "passed": passed,
        "reason_codes": reasons,
        "focus_laplacian_variance": focus,
        "luminance_p05": p05,
        "luminance_p50": p50,
        "luminance_p95": p95,
        **diagnostics,
    }


def _image_signal_diagnostics(gray: np.ndarray) -> dict[str, Any]:
    """Describe raw signal, without mistaking sensor noise for useful texture.

    Warnings are diagnostic, not relaxed or additional installation gates.
    Dark pipe paint can resemble underexposure, so keep the measured values.
    """
    smooth = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 1.0)
    mean = cv2.boxFilter(smooth, -1, (9, 9))
    variance = np.maximum(cv2.boxFilter(smooth * smooth, -1, (9, 9)) - mean * mean, 0)
    residual = gray.astype(np.float32) - smooth
    noise = float(np.median(np.abs(residual - np.median(residual))) / 0.67448975)
    dark = float(np.mean(gray <= 16))
    texture = float(np.mean(np.sqrt(variance) >= max(2.0, 2.0 * noise)))
    warnings = []
    if float(np.median(gray)) < 32:
        warnings.append("LOW_LIGHT")
    if dark > 0.5:
        warnings.append("DARK_REGION_DOMINANT")
    if texture < 0.1:
        warnings.append("LOW_TEXTURE")
    return {"dark_pixel_fraction": dark, "saturated_pixel_fraction": float(np.mean(gray >= 250)),
            "high_frequency_noise_estimate": noise, "textured_fraction": texture,
            "warning_codes": warnings}


def _stereo_gray(image: np.ndarray, preprocessing: str) -> np.ndarray:
    """Enhance only the matching copy; color evidence always uses original BGR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if preprocessing == "low_light":
        gray = cv2.fastNlMeansDenoising(gray, None, 5.0, 7, 21)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray


def _rectify(
    image: np.ndarray,
    camera: CameraCalibration,
    rectified: bool,
) -> np.ndarray:
    if rectified:
        return image.copy()
    assert camera.rectification_matrix is not None
    assert camera.projection_matrix is not None
    maps = cv2.initUndistortRectifyMap(
        camera.intrinsic,
        camera.distortion,
        camera.rectification_matrix,
        camera.projection_matrix,
        (camera.width, camera.height),
        cv2.CV_32FC1,
    )
    return cv2.remap(image, maps[0], maps[1], cv2.INTER_LINEAR)


def _compute_stereo_depth(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    calibration: StereoCalibration,
    config: dict[str, Any],
) -> StereoDepthResult:
    """Compute fail-closed SGBM depth for each rectified view."""

    height, width = left_bgr.shape[:2]
    invalid_depth = np.full((height, width), np.nan, dtype=np.float32)
    invalid_mask = np.zeros((height, width), dtype=bool)
    matching = config["stereo_matching"]
    audit_metadata = {
        "algorithm_revision": STEREO_ALGORITHM_REVISION,
        "opencv_version": cv2.__version__,
        "matcher": dict(matching),
        "validation_scope": "BIDIRECTIONAL_MATCHES_ONLY_NOT_METRIC_ACCURACY",
    }
    if matching["min_disparity"] + matching["num_disparities"] >= width:
        return StereoDepthResult(
            invalid_depth,
            invalid_depth.copy(),
            invalid_mask,
            invalid_mask.copy(),
            {
                **audit_metadata,
                "status": "INVALID",
                "reason_codes": ["DISPARITY_RANGE_EXCEEDS_IMAGE_WIDTH"],
                "valid_left_fraction": 0.0,
                "valid_right_fraction": 0.0,
            },
        )

    block = matching["block_size"]
    common = {
        "numDisparities": matching["num_disparities"],
        "blockSize": block,
        "P1": 8 * block * block,
        "P2": 32 * block * block,
        "disp12MaxDiff": matching["disp12_max_diff"],
        "preFilterCap": 31,
        "uniquenessRatio": matching["uniqueness_ratio"],
        "speckleWindowSize": matching["speckle_window_size"],
        "speckleRange": matching["speckle_range"],
        "mode": cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    }
    try:
        left_gray = _stereo_gray(left_bgr, matching["preprocessing"])
        right_gray = _stereo_gray(right_bgr, matching["preprocessing"])
        left_matcher = cv2.StereoSGBM_create(
            minDisparity=matching["min_disparity"], **common
        )
        right_matcher = cv2.StereoSGBM_create(
            minDisparity=-matching["min_disparity"] - matching["num_disparities"],
            **common,
        )
        left_disparity = left_matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        right_disparity = right_matcher.compute(right_gray, left_gray).astype(np.float32) / 16.0
    except cv2.error as error:
        return StereoDepthResult(
            invalid_depth,
            invalid_depth.copy(),
            invalid_mask,
            invalid_mask.copy(),
            {
                **audit_metadata,
                "status": "INVALID",
                "reason_codes": ["OPENCV_STEREO_MATCHING_FAILED"],
                "opencv_error": str(error),
                "valid_left_fraction": 0.0,
                "valid_right_fraction": 0.0,
            },
        )

    minimum_depth = config["minimum_depth_mm"]
    maximum_depth = config["maximum_depth_mm"]
    left_candidate = left_disparity > max(float(matching["min_disparity"]), 0.0)
    right_minimum = -matching["min_disparity"] - matching["num_disparities"]
    # OpenCV encodes an invalid match as minDisparity - 1.  For the
    # right matcher this is negative too, so a sign check is insufficient.
    right_candidate = (right_disparity >= right_minimum) & (right_disparity < 0.0)

    # A one-way disparity can be a false match, especially on smooth pipes.
    # Only mutually consistent left/right samples are allowed to become depth
    # or free-space evidence.
    rows, columns = np.indices(left_disparity.shape)
    right_columns = np.zeros(left_disparity.shape, dtype=np.int32)
    right_columns[left_candidate] = np.rint(
        columns[left_candidate] - left_disparity[left_candidate]
    ).astype(np.int32)
    left_in_bounds = left_candidate & (right_columns >= 0) & (right_columns < width)
    sampled_right = np.full(left_disparity.shape, np.nan, dtype=np.float32)
    sampled_right[left_in_bounds] = right_disparity[
        rows[left_in_bounds], right_columns[left_in_bounds]
    ]
    consistency_limit = config["left_right_consistency_px"]
    left_valid = (
        left_in_bounds
        & np.isfinite(sampled_right)
        & (sampled_right < 0)
        & (sampled_right >= right_minimum)
        & (np.abs(left_disparity + sampled_right) <= consistency_limit)
    )

    left_columns = np.zeros(right_disparity.shape, dtype=np.int32)
    left_columns[right_candidate] = np.rint(
        columns[right_candidate] - right_disparity[right_candidate]
    ).astype(np.int32)
    right_in_bounds = right_candidate & (left_columns >= 0) & (left_columns < width)
    sampled_left = np.full(right_disparity.shape, np.nan, dtype=np.float32)
    sampled_left[right_in_bounds] = left_disparity[
        rows[right_in_bounds], left_columns[right_in_bounds]
    ]
    right_valid = (
        right_in_bounds
        & np.isfinite(sampled_left)
        & (sampled_left > 0)
        & (np.abs(right_disparity + sampled_left) <= consistency_limit)
    )
    left_depth = np.full(left_disparity.shape, np.nan, dtype=np.float32)
    right_depth = np.full(right_disparity.shape, np.nan, dtype=np.float32)
    left_depth[left_valid] = (
        calibration.left.fx * calibration.baseline_mm / left_disparity[left_valid]
    )
    right_depth[right_valid] = (
        calibration.right.fx * calibration.baseline_mm / -right_disparity[right_valid]
    )
    left_valid &= np.isfinite(left_depth) & (left_depth >= minimum_depth) & (
        left_depth <= maximum_depth
    )
    right_valid &= np.isfinite(right_depth) & (right_depth >= minimum_depth) & (
        right_depth <= maximum_depth
    )
    left_depth[~left_valid] = np.nan
    right_depth[~right_valid] = np.nan
    return StereoDepthResult(
        left_depth,
        right_depth,
        left_valid,
        right_valid,
        {
            **audit_metadata,
            "status": "VALID" if np.any(left_valid) and np.any(right_valid) else "INVALID",
            "reason_codes": (
                []
                if np.any(left_valid) and np.any(right_valid)
                else ["NO_BIDIRECTIONAL_VALID_DISPARITY"]
            ),
            "valid_left_fraction": float(np.mean(left_valid)),
            "valid_right_fraction": float(np.mean(right_valid)),
            "searchable_depth_min_mm": float(calibration.left.fx * calibration.baseline_mm /
                                               (matching["min_disparity"] + matching["num_disparities"] - 1)),
            "warning_codes": (["LOW_STEREO_COVERAGE"] if min(float(np.mean(left_valid)), float(np.mean(right_valid))) < 0.25 else []),
            "left_right_consistency_px": consistency_limit,
            "depth_convention": "camera_z_mm",
        },
    )


def _project_points(
    points_world: np.ndarray,
    camera: CameraCalibration,
    rectified: bool,
) -> tuple[np.ndarray, np.ndarray]:
    camera_points = (
        camera.rotation_world_to_camera @ (points_world - camera.center_world_mm).T
    ).T
    depths = camera_points[:, 2]
    pixels = np.full((len(camera_points), 2), np.nan, dtype=np.float64)
    valid_depth = np.isfinite(depths) & (np.abs(depths) > 1.0e-9)
    if np.any(valid_depth):
        indices = np.flatnonzero(valid_depth)
        projected_points = camera_points[valid_depth]
        if rectified and camera.rectification_matrix is not None:
            projected_points = (camera.rectification_matrix @ projected_points.T).T
        projected_depths = projected_points[:, 2]
        # Visibility is defined in the rectified camera frame.  Points with
        # negative rectified Z are behind the sensor and must never yield a
        # nominal CAD projection even when their raw-camera Z was positive.
        valid_projected = np.isfinite(projected_depths) & (projected_depths > 1.0e-9)
        if np.any(valid_projected):
            matrix = (
                camera.projection_matrix
                if rectified and camera.projection_matrix is not None
                else camera.intrinsic
            )
            # A rectified image is produced by undistortPoints/remap.  Its
            # pixel projection uses P[:,:3] on each eye's rectified camera
            # coordinates; P's fourth column belongs to the stereo Q/P
            # convention and must not be re-applied to CAD points already
            # expressed relative to that eye's optical centre.
            if matrix.shape[1] == 4:
                matrix = matrix[:, :3]
            output_indices = indices[valid_projected]
            normalized = (
                projected_points[valid_projected, :2]
                / projected_depths[valid_projected, None]
            )
            pixels[output_indices] = np.column_stack(
                (
                    matrix[0, 0] * normalized[:, 0]
                    + matrix[0, 2],
                    matrix[1, 1] * normalized[:, 1]
                    + matrix[1, 2],
                )
            ).astype(np.float64)
            depths[output_indices] = projected_depths[valid_projected]
    if rectified:
        return pixels, depths
    assert camera.rectification_matrix is not None
    assert camera.projection_matrix is not None
    raw, _ = cv2.projectPoints(
        camera_points,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        camera.intrinsic,
        camera.distortion,
    )
    pixels = cv2.undistortPoints(
        raw,
        camera.intrinsic,
        camera.distortion,
        R=camera.rectification_matrix,
        P=camera.projection_matrix,
    ).reshape(-1, 2)
    rectified_points = (camera.rectification_matrix @ camera_points.T).T
    return pixels, rectified_points[:, 2]


def _mesh_depth_projection(
    cad_object: CadObject,
    camera: CameraCalibration,
    rectified: bool,
) -> np.ndarray:
    """Rasterize a CAD object's triangle mesh into a conservative depth image."""

    shape = (camera.height, camera.width)
    object_depth = np.full(shape, np.inf, dtype=np.float32)
    pixels, depths = _project_points(cad_object.vertices_world_mm, camera, rectified)
    height, width = shape
    for triangle in cad_object.triangles:
        triangle_depths = depths[triangle]
        triangle_pixels = pixels[triangle]
        if (
            np.any(triangle_depths <= 1e-6)
            or not np.all(np.isfinite(triangle_pixels))
            or not np.all(np.isfinite(triangle_depths))
            or np.any(np.abs(triangle_pixels) > 1.0e7)
        ):
            continue
        minimum = np.floor(np.min(triangle_pixels, axis=0)).astype(np.int64)
        maximum = np.ceil(np.max(triangle_pixels, axis=0)).astype(np.int64)
        x1 = max(0, int(minimum[0]))
        y1 = max(0, int(minimum[1]))
        x2 = min(width - 1, int(maximum[0]))
        y2 = min(height - 1, int(maximum[1]))
        if x1 > x2 or y1 > y2:
            continue
        local = np.rint(triangle_pixels - np.asarray([x1, y1])).astype(np.int32)
        triangle_mask = np.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=np.uint8)
        cv2.fillConvexPoly(triangle_mask, local, 1, cv2.LINE_8)
        if not np.any(triangle_mask):
            continue
        # Constant triangle depth is sufficient for the current straight-pipe
        # visibility gate and remains conservative at layer boundaries.  The
        # observed depth comparison retains a much wider calibrated tolerance.
        triangle_depth = float(np.mean(triangle_depths))
        region = object_depth[y1 : y2 + 1, x1 : x2 + 1]
        update = triangle_mask.astype(bool) & (triangle_depth < region)
        region[update] = triangle_depth
    return object_depth


def _nominal_projections(
    scene: CadScene,
    camera: CameraCalibration,
    rectified: bool,
    occlusion_margin_mm: float,
) -> dict[str, PipeProjection]:
    shape = (camera.height, camera.width)
    object_depths = {
        pipe.pipe_id: _mesh_depth_projection(
            scene.objects_by_pipe_id[pipe.pipe_id], camera, rectified
        )
        for pipe in scene.pipes
    }
    nearest = np.full(shape, np.inf, dtype=np.float32)
    for depth in object_depths.values():
        nearest = np.minimum(nearest, depth)

    result: dict[str, PipeProjection] = {}
    for pipe in scene.pipes:
        object_depth = object_depths[pipe.pipe_id]
        mask = np.isfinite(object_depth)
        amodal_pixels = int(np.count_nonzero(mask))
        centerline_pixels, centerline_depths = _project_points(
            pipe.centerline_world_mm, camera, rectified
        )
        if np.all(np.isfinite(centerline_pixels)) and np.all(centerline_depths > 1e-6):
            mean_depth = float(np.mean(centerline_depths))
            centerline = centerline_pixels.tolist()
            diameter_px = max(1.0, camera.fx * pipe.nominal_diameter_mm / mean_depth)
        else:
            mean_depth = None
            centerline = None
            diameter_px = None
        if amodal_pixels == 0:
            visible_pixels = 0
            state = "OUT_OF_FRUSTUM"
        else:
            nominal_visible = mask & (object_depth <= nearest + occlusion_margin_mm)
            visible_pixels = int(np.count_nonzero(nominal_visible))
            if visible_pixels == 0:
                state = "FULLY_OCCLUDED"
            elif visible_pixels < amodal_pixels:
                state = "PARTIALLY_OCCLUDED"
            else:
                state = "FULLY_VISIBLE"
        result[pipe.pipe_id] = PipeProjection(
            mask=mask,
            expected_depth_mm=mean_depth,
            centerline_px=centerline,
            predicted_diameter_px=diameter_px,
            amodal_pixels=amodal_pixels,
            nominal_visible_pixels=visible_pixels,
            nominal_occlusion_state=state,
        )
    return result


def _hex_to_lab(color: str) -> np.ndarray:
    red, green, blue = (int(color[index : index + 2], 16) for index in (1, 3, 5))
    encoded = cv2.cvtColor(
        np.asarray([[[blue, green, red]]], dtype=np.uint8), cv2.COLOR_BGR2LAB
    )[0, 0].astype(np.float64)
    return np.asarray([encoded[0] * 100.0 / 255.0, encoded[1] - 128.0, encoded[2] - 128.0])


def _image_lab(image: np.ndarray) -> np.ndarray:
    encoded = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float64)
    encoded[:, :, 0] *= 100.0 / 255.0
    encoded[:, :, 1:] -= 128.0
    return encoded


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    rows, columns = np.nonzero(mask)
    if len(rows) == 0:
        return None
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max() - columns.min() + 1),
        int(rows.max() - rows.min() + 1),
    ]


def _color_geometry_metrics(
    lab_image: np.ndarray,
    pipe: PipePrior,
    projection: PipeProjection,
    config: dict[str, Any],
    target_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure colour and width, optionally restricted to depth-matched pixels.

    ``target_mask`` is supplied by the stereo depth gate.  Keeping the
    restriction here (rather than merely comparing two independent scalar
    fractions) prevents a same-colour foreground object from satisfying the
    colour/width gates while a disjoint handful of pixels satisfies the depth
    gate.
    """
    if target_mask is None:
        target_mask = projection.mask
    else:
        target_mask = np.asarray(target_mask, dtype=bool) & projection.mask
    if projection.amodal_pixels == 0 or projection.predicted_diameter_px is None:
        return {
            "color_support_fraction": 0.0,
            "color_support_fraction_amodal": 0.0,
            "joint_target_color_support_fraction": 0.0,
            "joint_target_color_support_fraction_amodal": 0.0,
            "observed_diameter_px": None,
            "predicted_diameter_px": projection.predicted_diameter_px,
            "width_relative_error": None,
            "color_gate_passed": False,
            "joint_target_color_gate_passed": False,
            "width_gate_passed": False,
            "width_assessable": False,
            "width_gate_status": "NOT_ASSESSABLE",
        }
    nominal_delta = np.linalg.norm(lab_image - _hex_to_lab(pipe.color_srgb), axis=2)
    nominal_support = nominal_delta <= config["color_delta_e76_tolerance"]
    # A corrupt calibration/model scale can imply an enormous projected
    # diameter.  Bound the morphology kernel before handing it to OpenCV;
    # such a sample will normally fail the independent width gate rather than
    # exhausting process memory.
    try:
        raw_search_radius = int(math.ceil(projection.predicted_diameter_px))
    except (OverflowError, ValueError):
        raw_search_radius = _MAX_MORPH_RADIUS_PIXELS
    search_radius = min(max(2, raw_search_radius), _MAX_MORPH_RADIUS_PIXELS)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * search_radius + 1, 2 * search_radius + 1)
    )
    ring = cv2.dilate(projection.mask.astype(np.uint8), kernel).astype(bool)
    ring &= ~projection.mask
    if np.any(ring):
        local_background_lab = np.median(lab_image[ring], axis=0)
        contrast = np.linalg.norm(lab_image - local_background_lab, axis=2)
        contrast_support = contrast >= config["minimum_local_contrast_delta_e76"]
    else:
        local_background_lab = None
        contrast_support = np.ones(projection.mask.shape, dtype=bool)
    support = nominal_support & contrast_support & projection.mask
    support_pixels = int(np.count_nonzero(support))
    support_fraction_amodal = float(support_pixels / projection.amodal_pixels)
    support_denominator = (
        projection.nominal_visible_pixels
        if projection.nominal_visible_pixels > 0
        else projection.amodal_pixels
    )
    support_fraction = float(min(1.0, support_pixels / support_denominator))
    target_support = support & target_mask
    target_support_pixels = int(np.count_nonzero(target_support))
    target_support_fraction_amodal = float(
        target_support_pixels / projection.amodal_pixels
    )
    target_support_fraction = float(
        min(1.0, target_support_pixels / support_denominator)
    )

    # Measure width only from depth-matched, colour-supported pixels inside the
    # actual mesh silhouette.  This avoids a white/grey background component
    # or a same-colour foreground object swallowing the measurement.
    observed_width: float | None = None
    if projection.centerline_px is not None and target_support_pixels >= 3:
        axis_points = np.asarray(projection.centerline_px, dtype=np.float64)
        direction = axis_points[1] - axis_points[0]
        length = float(np.linalg.norm(direction))
        if length > 1e-9:
            direction /= length
            normal = np.asarray([-direction[1], direction[0]])
            rows, columns = np.nonzero(target_support)
            points = np.column_stack((columns, rows)).astype(np.float64)
            relative = points - axis_points[0]
            axial = relative @ direction
            transverse = relative @ normal
            central = (axial >= 0.10 * length) & (axial <= 0.90 * length)
            if np.count_nonzero(central) >= 3:
                bins = np.rint(axial[central]).astype(np.int32)
                values = transverse[central]
                widths: list[float] = []
                for bin_id in np.unique(bins):
                    section = values[bins == bin_id]
                    if len(section) >= 2:
                        widths.append(float(np.max(section) - np.min(section) + 1.0))
                if widths:
                    observed_width = float(np.median(widths))
    width_error = (
        abs(observed_width - projection.predicted_diameter_px)
        / max(projection.predicted_diameter_px, 1e-9)
        if observed_width is not None
        else None
    )
    nominal_visible_fraction = float(
        projection.nominal_visible_pixels / projection.amodal_pixels
    )
    width_assessable = bool(
        projection.nominal_occlusion_state
        not in {"FULLY_OCCLUDED", "OUT_OF_FRUSTUM"}
        and (
            projection.nominal_occlusion_state == "FULLY_VISIBLE"
            or target_support_fraction_amodal
            >= config["minimum_width_assessable_fraction"]
        )
    )
    width_passed = bool(
        width_error is not None
        and width_error <= config["maximum_width_relative_error"]
    )
    return {
        "color_support_fraction": support_fraction,
        "color_support_fraction_amodal": support_fraction_amodal,
        "joint_target_color_support_fraction": target_support_fraction,
        "joint_target_color_support_fraction_amodal": target_support_fraction_amodal,
        "local_background_lab": (
            local_background_lab.tolist() if local_background_lab is not None else None
        ),
        "observed_diameter_px": observed_width,
        "predicted_diameter_px": projection.predicted_diameter_px,
        "width_relative_error": width_error,
        "color_gate_passed": support_fraction
        >= config["minimum_color_support_fraction"],
        "joint_target_color_gate_passed": target_support_fraction
        >= config["minimum_color_support_fraction"],
        "nominal_visible_fraction": nominal_visible_fraction,
        "width_assessable": width_assessable,
        "width_gate_passed": width_passed,
        "width_gate_status": (
            "OCCLUSION_LIMITED"
            if not width_assessable
            else "PASSED"
            if width_passed
            else "FAILED"
        ),
    }


def _view_evidence(
    pipe: PipePrior,
    projection: PipeProjection,
    image: np.ndarray,
    depth: np.ndarray,
    valid_depth: np.ndarray,
    quality: dict[str, Any],
    pair_healthy: bool,
    calibration: StereoCalibration,
    config: dict[str, Any],
) -> dict[str, Any]:
    mask = projection.mask
    amodal = projection.amodal_pixels
    valid_pixels = int(np.count_nonzero(valid_depth & mask))
    valid_fraction = valid_pixels / amodal if amodal else 0.0
    target = np.zeros(mask.shape, dtype=bool)
    if amodal and projection.expected_depth_mm is not None:
        expected = projection.expected_depth_mm
        target = valid_depth & mask & (np.abs(depth - expected) <= config["depth_tolerance_mm"])
        foreground = valid_depth & mask & (
            depth < expected - config["occlusion_margin_mm"]
        )
        free_space = valid_depth & mask & (
            depth > expected + config["free_space_margin_mm"]
        )
        target_fraction = float(np.count_nonzero(target) / amodal)
        foreground_fraction = float(np.count_nonzero(foreground) / amodal)
        free_fraction = float(np.count_nonzero(free_space) / amodal)
    else:
        target_fraction = foreground_fraction = free_fraction = 0.0

    color = _color_geometry_metrics(
        _image_lab(image),
        pipe,
        projection,
        config,
        target_mask=target,
    )
    in_frame = amodal >= config["minimum_amodal_pixels"]
    sensor_healthy = bool(pair_healthy and quality["passed"])
    depth_gate = bool(
        valid_fraction >= config["minimum_valid_depth_fraction"]
        and target_fraction >= config["minimum_target_depth_fraction"]
    )
    direct = bool(
        calibration.validated
        and calibration.registration_validated
        and sensor_healthy
        and in_frame
        and depth_gate
        and color["joint_target_color_gate_passed"]
        and (color["width_gate_passed"] or not color["width_assessable"])
    )

    if not in_frame:
        occlusion_state = "OUT_OF_FRUSTUM"
    elif direct and projection.nominal_occlusion_state == "FULLY_OCCLUDED":
        occlusion_state = "PARTIALLY_OCCLUDED"
    elif foreground_fraction >= config["fully_occluded_fraction"] and not direct:
        occlusion_state = "FULLY_OCCLUDED"
    elif foreground_fraction > 0:
        occlusion_state = "PARTIALLY_OCCLUDED"
    else:
        occlusion_state = projection.nominal_occlusion_state

    expected_region_unoccluded = bool(
        projection.nominal_occlusion_state == "FULLY_VISIBLE"
        and foreground_fraction == 0.0
    )
    negative_candidate = bool(
        calibration.validated
        and calibration.registration_validated
        and sensor_healthy
        and in_frame
        and expected_region_unoccluded
        and valid_fraction >= config["minimum_valid_depth_fraction"]
        and free_fraction >= config["minimum_free_space_fraction"]
        and not direct
        and not color["color_gate_passed"]
    )
    if occlusion_state in {"FULLY_OCCLUDED", "OUT_OF_FRUSTUM"}:
        color = dict(color)
        color["width_assessable"] = False
        color["width_gate_status"] = (
            "OCCLUDED" if occlusion_state == "FULLY_OCCLUDED" else "OUT_OF_FRUSTUM"
        )
    reasons: list[str] = []
    if direct:
        evidence = "DIRECT_STEREO_CAD_EVIDENCE"
        reasons.extend(["DEPTH_MATCH", "TARGET_COLOR_MATCH"])
        reasons.append(
            "WIDTH_MATCH"
            if color["width_gate_passed"]
            else "WIDTH_NOT_ASSESSABLE_DUE_TO_OCCLUSION"
        )
    elif negative_candidate:
        evidence = "NEGATIVE_FREE_SPACE_CANDIDATE"
        reasons.append("EXPECTED_REGION_VALID_FREE_SPACE")
    else:
        evidence = "INCONCLUSIVE"
        if occlusion_state == "FULLY_OCCLUDED":
            reasons.append("FULLY_OCCLUDED")
        if not in_frame:
            reasons.append("OUT_OF_FRUSTUM_OR_TOO_FEW_EXPECTED_PIXELS")
        if not sensor_healthy:
            reasons.append("CAPTURE_HEALTH_FAILED")
        if valid_fraction < config["minimum_valid_depth_fraction"]:
            reasons.append("INSUFFICIENT_VALID_DEPTH")
        if not color["color_gate_passed"]:
            reasons.append("COLOR_GATE_FAILED")
        elif not color["joint_target_color_gate_passed"]:
            reasons.append("TARGET_COLOR_INTERSECTION_GATE_FAILED")
        if color["width_assessable"] and not color["width_gate_passed"]:
            reasons.append("WIDTH_GATE_FAILED")
        if not calibration.validated:
            reasons.append("CALIBRATION_NOT_VALIDATED")
        if not calibration.registration_validated:
            reasons.append("CAD_REGISTRATION_NOT_VALIDATED")
        if not reasons:
            reasons.append("NO_QUALIFIED_EVIDENCE")
    return {
        "projection_geometry_source": "cad_triangle_mesh",
        "occlusion_state": occlusion_state,
        "nominal_cad_occlusion_state": projection.nominal_occlusion_state,
        "assessable": bool(
            sensor_healthy
            and in_frame
            and valid_fraction >= config["minimum_valid_depth_fraction"]
            and occlusion_state not in {"FULLY_OCCLUDED", "OUT_OF_FRUSTUM"}
        ),
        "installation_evidence": evidence,
        "reason_codes": reasons,
        "amodal_pixels_in_frame": amodal,
        "amodal_bbox_xywh": _mask_bbox(mask),
        "visible_bbox_xywh": _mask_bbox(target),
        "nominal_visible_pixels": projection.nominal_visible_pixels,
        "centerline_px": projection.centerline_px,
        "expected_depth_mm": projection.expected_depth_mm,
        "valid_depth_fraction": valid_fraction,
        "target_depth_fraction": target_fraction,
        "foreground_occlusion_fraction": foreground_fraction,
        "free_space_fraction": free_fraction,
        "expected_region_in_frame": in_frame,
        "expected_region_unoccluded": expected_region_unoccluded,
        "direct_instance_evidence": direct,
        "negative_candidate": negative_candidate,
        **color,
    }


def _pipe_result(
    pipe: PipePrior,
    group_results: list[dict[str, Any]],
    calibration: StereoCalibration,
    config: dict[str, Any],
    interval_minutes: float,
) -> dict[str, Any]:
    positive_ids: list[str] = []
    negative_ids: list[str] = []
    visibility_by_capture: dict[str, Any] = {}
    for group in group_results:
        capture_id = group["capture_id"]
        views = group["pipe_evidence"][pipe.pipe_id]
        visibility_by_capture[capture_id] = views
        if any(views[role]["direct_instance_evidence"] for role in _VIEW_ROLES):
            positive_ids.append(capture_id)
        if all(views[role]["negative_candidate"] for role in _VIEW_ROLES):
            negative_ids.append(capture_id)

    current = group_results[-1]
    current_views = current["pipe_evidence"][pipe.pipe_id]
    current_fully_occluded = all(
        current_views[role]["occlusion_state"] == "FULLY_OCCLUDED"
        for role in _VIEW_ROLES
    )
    current_healthy = bool(
        current["pair_healthy"]
        and current["depth_audit"]["status"] == "VALID"
        and all(current["quality"][role]["passed"] for role in _VIEW_ROLES)
        and calibration.validated
        and calibration.registration_validated
    )
    direct_current = any(
        current_views[role]["direct_instance_evidence"] for role in _VIEW_ROLES
    )
    # A direct hit in one eye and a qualified-looking free-space candidate in
    # the other is contradictory even before the pair-level negative gate can
    # be assembled.  Do not let a single-eye false positive become INSTALLED.
    negative_current_any = any(
        current_views[role]["negative_candidate"] for role in _VIEW_ROLES
    )
    repeated_required = max(2, config["minimum_repeated_absence_captures"])
    negative_evidence: NegativeInstallationEvidence | None = None
    minimum_separation_seconds = interval_minutes * 60.0 * 0.8
    qualified_negative_ids: list[str] = []
    qualified_signatures: set[str] = set()
    last_selected_time: datetime | None = None
    # Only a consecutive absence sequence ending at the current capture can
    # describe the current state. Nearby retries and byte-identical frozen
    # pairs remain diagnostics but do not increase the repetition count.
    for group in reversed(group_results):
        views = group["pipe_evidence"][pipe.pipe_id]
        if not all(views[role]["negative_candidate"] for role in _VIEW_ROLES):
            break
        captured_time = datetime.fromisoformat(group["captured_at"])
        if (
            last_selected_time is not None
            and (last_selected_time - captured_time).total_seconds()
            < minimum_separation_seconds
        ):
            continue
        if group["pair_signature"] in qualified_signatures:
            continue
        qualified_negative_ids.append(group["capture_id"])
        qualified_signatures.add(group["pair_signature"])
        last_selected_time = captured_time
    qualified_negative_ids.reverse()
    independent_negative_count = len(qualified_negative_ids)
    current_is_negative = bool(negative_ids and negative_ids[-1] == current["capture_id"])
    if current_is_negative:
        negative_evidence = NegativeInstallationEvidence(
            calibration_validated=calibration.validated,
            registration_validated=calibration.registration_validated,
            expected_region_in_frame=all(
                current_views[role]["expected_region_in_frame"] for role in _VIEW_ROLES
            ),
            expected_region_unoccluded=all(
                current_views[role]["expected_region_unoccluded"] for role in _VIEW_ROLES
            ),
            sensor_health_validated=current_healthy,
            free_space_validated=independent_negative_count >= repeated_required,
            repeated_absence_observations=independent_negative_count,
            independent_evidence_sources=2,
        )
    conflict = bool(
        direct_current
        and (
            negative_current_any
            or (negative_evidence is not None and negative_evidence.is_qualified())
        )
    )
    if current_fully_occluded:
        state = "UNKNOWN"
        basis = "CURRENT_VIEW_FULLY_OCCLUDED"
        reasons = ["FULLY_OCCLUDED_IN_BOTH_VIEWS"]
    elif not current_healthy:
        state = "UNKNOWN"
        basis = "UNHEALTHY_CURRENT_EVIDENCE"
        reasons = ["CAPTURE_CALIBRATION_OR_DEPTH_HEALTH_FAILED"]
    else:
        state = classify_installation_state(
            direct_instance_evidence=direct_current,
            negative_evidence=negative_evidence,
            evidence_healthy=current_healthy,
            conflict=conflict,
        )
        if state == "INSTALLED":
            basis = "DIRECT_STEREO_CAD_EVIDENCE"
            reasons = ["CURRENT_CAPTURE_HAS_GEOMETRIC_IDENTITY_EVIDENCE"]
        elif state == "NOT_INSTALLED":
            basis = "QUALIFIED_REPEATED_STEREO_FREE_SPACE"
            reasons = ["ALL_NEGATIVE_EVIDENCE_GATES_PASSED"]
        elif conflict:
            basis = "CONFLICTING_EVIDENCE"
            reasons = ["POSITIVE_AND_QUALIFIED_NEGATIVE_EVIDENCE_CONFLICT"]
        elif negative_ids:
            basis = "INSUFFICIENT_REPEATED_NEGATIVE_EVIDENCE"
            reasons = [
                "NEGATIVE_EVIDENCE_REQUIRES_AT_LEAST_TWO_INDEPENDENT_CAPTURES"
            ]
        else:
            basis = "INSUFFICIENT_EVIDENCE"
            reasons = ["NO_QUALIFIED_POSITIVE_OR_NEGATIVE_EVIDENCE"]
    return {
        "instance_id": pipe.instance_id,
        "pipe_id": pipe.pipe_id,
        "cad_object_id": pipe.cad_object_id,
        "cad_uuid": pipe.cad_uuid,
        "layer_id": pipe.layer_id,
        "color_class": pipe.color_class,
        "color_srgb": pipe.color_srgb,
        "nominal_diameter_mm": pipe.nominal_diameter_mm,
        "installation_state": state,
        "installation_state_zh": installation_state_label_zh(state),
        "state_basis": basis,
        "reason_codes": reasons,
        "positive_evidence_capture_ids": positive_ids,
        "negative_evidence_capture_ids": negative_ids,
        "qualified_negative_evidence_capture_ids": qualified_negative_ids,
        "negative_evidence_unique_capture_count": independent_negative_count,
        "minimum_negative_capture_separation_seconds": minimum_separation_seconds,
        "visibility_by_view": current_views,
        "visibility_by_capture": visibility_by_capture,
    }


def _safe_output_paths(
    manifest_file: Path,
    scene: CadScene,
    groups: list[dict[str, Any]],
    report_output_path: str | Path | None,
    evidence_dir: str | Path | None,
) -> None:
    paths: dict[str, Path] = {"manifest": manifest_file, "model": scene.model_path}
    for group in groups:
        for role in _VIEW_ROLES:
            paths[f"{group['capture_id']}_{role}"] = _relative_asset_path(
                manifest_file,
                group["views"][role]["path"],
                f"{group['capture_id']}.{role}.path",
            )
    if report_output_path is not None:
        paths["report"] = Path(report_output_path).resolve()
    if evidence_dir is not None:
        evidence_root = Path(evidence_dir).resolve()
        for group in groups:
            for role in _VIEW_ROLES:
                paths[f"overlay_{group['capture_id']}_{role}"] = (
                    evidence_root / f"{group['capture_id']}_{role}_overlay.png"
                ).resolve()
    # Use the shared path guard rather than comparing only normalized strings.
    # On Windows (and on POSIX filesystems with hard links), two distinct names
    # can still designate the same input inode.  An atomic report replace at
    # such a path would otherwise overwrite the manifest/model/photo that is
    # being analyzed.
    try:
        ensure_paths_distinct(**paths)
    except ValueError as error:
        raise StereoAnalysisError(str(error)) from error


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_evidence_overlays(
    evidence_dir: Path,
    scene: CadScene,
    calibration: StereoCalibration,
    group_results: list[dict[str, Any]],
    pipe_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence_dir = evidence_dir.resolve()
    if evidence_dir.exists() and not evidence_dir.is_dir():
        raise StereoAnalysisError("evidence_dir exists and is not a directory")
    state_by_pipe = {item["pipe_id"]: item["installation_state"] for item in pipe_results}
    colors = {
        "INSTALLED": (40, 180, 40),
        "NOT_INSTALLED": (30, 30, 230),
        "UNKNOWN": (0, 190, 255),
    }
    records: list[dict[str, Any]] = []
    for group in group_results:
        for role in _VIEW_ROLES:
            overlay = group["rectified_images"][role].copy()
            camera = getattr(calibration, role)
            projections = _nominal_projections(
                scene,
                camera,
                calibration.rectified,
                0.0,
            )
            for pipe in scene.pipes:
                mask = projections[pipe.pipe_id].mask.astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, colors[state_by_pipe[pipe.pipe_id]], 2)
            success, encoded = cv2.imencode(".png", overlay)
            if not success:
                raise StereoAnalysisError("OpenCV could not encode an evidence overlay")
            path = evidence_dir / f"{group['capture_id']}_{role}_overlay.png"
            path = path.resolve()
            try:
                path.relative_to(evidence_dir)
            except ValueError as error:
                raise StereoAnalysisError("Evidence overlay path escaped evidence_dir") from error
            content = encoded.tobytes()
            _atomic_write(path, content)
            records.append(
                {
                    "capture_id": group["capture_id"],
                    "view_role": role,
                    "path": str(path),
                    "sha256": _hash_bytes(content),
                }
            )
    return records


def analyze_stereo_capture(
    manifest_path: str | Path,
    report_output_path: str | Path | None = None,
    evidence_dir: str | Path | None = None,
    *,
    measurement_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze explicitly paired stereo photographs and return per-pipe states.

    Structural or integrity failures raise :class:`StereoAnalysisError` and do
    not create a report.  Runtime evidence failures are represented by
    ``UNKNOWN`` states with reason codes.
    """

    manifest_file = Path(manifest_path).resolve()
    log_event(_LOGGER, "analysis_start", mode="stereo", manifest=manifest_file)
    manifest, manifest_sha256 = _read_json_snapshot(manifest_file)
    if manifest.get("schema_version") != "2.0":
        raise StereoAnalysisError("Stereo manifests require schema_version='2.0'")
    analysis_payload = manifest.get("analysis")
    if isinstance(analysis_payload, dict) and analysis_payload.get("mode") == "elevation_auto":
        from .elevation_auto import analyze_elevation_auto_manifest
        return analyze_elevation_auto_manifest(manifest_file, report_output_path=report_output_path, evidence_dir=evidence_dir)
    if isinstance(analysis_payload, dict) and analysis_payload.get("mode") == "elevation_depth":
        from .elevation_depth import analyze_elevation_depth_manifest

        return analyze_elevation_depth_manifest(
            manifest_file, report_output_path=report_output_path, evidence_dir=evidence_dir
        )
    dataset_id = _require_string(manifest.get("dataset_id"), "dataset_id")
    model_revision = _require_string(manifest.get("model_revision"), "model_revision")
    scene = _load_cad_scene(manifest_file, manifest.get("model"))
    calibration = _calibration_from_manifest(manifest.get("stereo_calibration"))
    projected_pixel_budget = calibration.left.width * calibration.left.height * len(scene.pipes)
    if projected_pixel_budget > _MAX_PROJECTED_PIPE_PIXELS:
        raise StereoAnalysisError(
            "The calibrated image and pipe count exceed the projection memory budget"
        )
    config = _analysis_config(manifest.get("analysis"))
    from .metrology import analyze_local_geometry, measurement_settings

    metric_config = measurement_settings(
        measurement_options if measurement_options is not None else manifest.get("measurement_settings")
    )
    run_id, interval_minutes, groups = _capture_groups_from_manifest(
        manifest_file, manifest.get("capture"), calibration
    )
    _safe_output_paths(
        manifest_file, scene, groups, report_output_path, evidence_dir
    )

    projections = {
        role: _nominal_projections(
            scene,
            getattr(calibration, role),
            calibration.rectified,
            config["occlusion_margin_mm"],
        )
        for role in _VIEW_ROLES
    }
    group_results: list[dict[str, Any]] = []
    photo_hashes: dict[str, str] = {}
    for group_index, group in enumerate(groups):
        images: dict[str, np.ndarray] = {}
        integrities: dict[str, dict[str, Any]] = {}
        qualities: dict[str, dict[str, Any]] = {}
        for role in _VIEW_ROLES:
            view = group["views"][role]
            try:
                raw_image, integrity = load_photo_snapshot(manifest_file, view)
            except (ValueError, FileNotFoundError) as error:
                raise StereoAnalysisError(str(error)) from error
            image = _rectify(raw_image, getattr(calibration, role), calibration.rectified)
            images[role] = image
            integrities[role] = integrity
            qualities[role] = _quality(image, config)
            photo_hashes[str(_relative_asset_path(manifest_file, view["path"], "view.path"))] = integrity[
                "actual_sha256"
            ]
        pair_healthy = bool(
            group["sync_valid"]
            and all(qualities[role]["passed"] for role in _VIEW_ROLES)
        )
        depth = _compute_stereo_depth(images["left"], images["right"], calibration, config)
        depth_by_role = {
            "left": (depth.left_depth_mm, depth.left_valid),
            "right": (depth.right_depth_mm, depth.right_valid),
        }
        pipe_evidence: dict[str, dict[str, Any]] = {}
        for pipe in scene.pipes:
            pipe_evidence[pipe.pipe_id] = {}
            for role in _VIEW_ROLES:
                depth_image, valid = depth_by_role[role]
                pipe_evidence[pipe.pipe_id][role] = _view_evidence(
                    pipe,
                    projections[role][pipe.pipe_id],
                    images[role],
                    depth_image,
                    valid,
                    qualities[role],
                    pair_healthy,
                    calibration,
                    config,
                )
        group_results.append(
            {
                "capture_id": group["capture_id"],
                "captured_at": group["captured_at"],
                "sync_delta_ms": group["sync_delta_ms"],
                "sync_valid": group["sync_valid"],
                "pair_healthy": pair_healthy,
                "quality": qualities,
                "photos": integrities,
                "pair_signature": (
                    integrities["left"]["actual_sha256"]
                    + ":"
                    + integrities["right"]["actual_sha256"]
                ),
                "depth_audit": depth.audit,
                "pipe_evidence": pipe_evidence,
                "rectified_images": images,
            }
        )
        if group_index == len(groups) - 1:
            local_geometry = analyze_local_geometry(
                scene.pipes, projections, images, depth, calibration, pair_healthy,
                metric_config, config["color_delta_e76_tolerance"],
            )
            local_geometry["capture_id"] = group["capture_id"]
            local_geometry["calibration_id"] = calibration.calibration_id

    pipe_results = [
        _pipe_result(pipe, group_results, calibration, config, interval_minutes)
        for pipe in scene.pipes
    ]
    ambiguous_ids = {row["pipe_id"] for row in local_geometry["pipes"]
                     if any("AMBIGUOUS_LOCAL_PIPE_IDENTITY" in code or "SHARED_OR_OVERLAPPING_OBSERVATION" in code
                            for code in row["reason_codes"])}
    for row in pipe_results:
        if row["pipe_id"] in ambiguous_ids:
            row.update(installation_state="UNKNOWN", installation_state_zh=installation_state_label_zh("UNKNOWN"),
                       state_basis="AMBIGUOUS_LOCAL_GEOMETRY", reason_codes=["LOCAL_OBSERVATION_IDENTITY_AMBIGUOUS"])
    counts_counter = Counter(item["installation_state"] for item in pipe_results)
    counts = {state: int(counts_counter[state]) for state in INSTALLATION_STATES}

    if _hash_file(manifest_file) != manifest_sha256:
        raise StereoAnalysisError("Manifest changed during analysis")
    if _hash_file(scene.model_path) != scene.actual_sha256:
        raise StereoAnalysisError("CAD model changed during analysis")
    for path_text, digest in photo_hashes.items():
        if _hash_file(Path(path_text)) != digest:
            raise StereoAnalysisError(f"Photo changed during analysis: {path_text}")

    public_groups = []
    for group in group_results:
        public_groups.append(
            {
                key: value
                for key, value in group.items()
                if key not in {"rectified_images", "pipe_evidence"}
            }
            | {"pipe_evidence": group["pipe_evidence"]}
        )
    report: dict[str, Any] = {
        "schema_version": "2.0",
        "report_type": "stereo-cad-pipe-installation-state",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "software_version": __version__,
        "dataset_id": dataset_id,
        "model_revision": model_revision,
        "capture_group_id": run_id,
        "production_authority": False,
        "field_installation_state_inferred": True,
        "field_acceptance_passed": None,
        "model": {
            "path": str(manifest["model"]["path"]),
            "format": scene.model_format,
            "sha256": scene.actual_sha256,
            "expected_sha256": scene.expected_sha256,
            "verified": True,
            "object_binding_validation": scene.object_binding_validation,
            "object_count": scene.mesh_scene.object_count,
            "objects": [
                {
                    "object_id": item.object_id,
                    "guid": item.guid,
                    "pipe_id": item.pipe_id,
                    "name": item.name,
                    "layer_path": item.layer_path,
                    "geometry_type": item.geometry_type,
                    "mesh_source": item.mesh_source,
                    "vertex_count": item.vertex_count,
                    "triangle_count": item.triangle_count,
                    "bbox_min_mm": list(item.bbox_min_mm),
                    "bbox_max_mm": list(item.bbox_max_mm),
                    "watertight": item.watertight,
                }
                for item in scene.mesh_scene.objects
            ],
        },
        "inputs": {
            "manifest": {"path": str(manifest_file), "sha256": manifest_sha256},
            "model": {
                "path": str(manifest["model"]["path"]),
                "actual_sha256": scene.actual_sha256,
                "expected_sha256": scene.expected_sha256,
                "verified": True,
            },
        },
        "calibration_audit": {
            "calibration_id": calibration.calibration_id,
            "validated": calibration.validated,
            "registration_validated": calibration.registration_validated,
            "rectified_input": calibration.rectified,
            "baseline_mm": calibration.baseline_mm,
            "left_camera_id": calibration.left.camera_id,
            "right_camera_id": calibration.right.camera_id,
            "image_size": [calibration.left.width, calibration.left.height],
        },
        "capture_audit": {
            "input_kind": "stereo_still_capture_set",
            "camera_layout": "stereo",
            "interval_minutes": interval_minutes,
            "capture_count": len(group_results),
            "continuous_video_used": False,
            "groups": public_groups,
        },
        "counts": counts,
        "pipes": pipe_results,
        "local_measurements": local_geometry,
        "state_semantics": {
            "INSTALLED": "Current paired images contain qualified CAD-registered color, width, and depth evidence.",
            "NOT_INSTALLED": "At least two independent paired captures show qualified, unoccluded free space at the designed pipe location.",
            "UNKNOWN": "Evidence is occluded, out of frame, unhealthy, invalid, ambiguous, conflicting, or insufficient.",
            "safety_rule": "FULLY_OCCLUDED and invalid disparity never imply NOT_INSTALLED.",
        },
        "limitations": [
            "Field thresholds and physical accuracy still require acceptance with the actual camera and manufactured pipes.",
            "Passive stereo on smooth or reflective pipe surfaces may have invalid disparity; those pixels remain inconclusive.",
            "The first implementation models each manifest pipe as one straight cylindrical centerline segment.",
        ],
    }

    if evidence_dir is not None:
        evidence_records = _write_evidence_overlays(
            Path(evidence_dir).resolve(), scene, calibration, group_results, pipe_results
        )
        report["evidence_files"] = evidence_records
    if report_output_path is not None:
        content = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        _atomic_write(Path(report_output_path).resolve(), content)
    log_event(
        _LOGGER,
        "analysis_finished",
        mode="stereo",
        dataset_id=report.get("dataset_id"),
        counts=report.get("counts"),
    )
    return report


__all__ = [
    "CadScene",
    "StereoAnalysisError",
    "StereoDepthResult",
    "analyze_stereo_capture",
]
