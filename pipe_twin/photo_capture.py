"""Validation and immutable decoding helpers for manifest-bound still photos."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import cv2
import numpy as np


_SHA256_PATTERN = re.compile(r"[0-9a-fA-F]{64}\Z")
_ORIENTATION_POLICY = "RAW_PIXELS_NO_EXIF_TRANSFORM"
_TIMESTAMP_SOURCES = {
    "CAMERA_HARDWARE_CLOCK",
    "CAMERA_SYSTEM_CLOCK",
    "EXIF_DATETIME_ORIGINAL",
    "HOST_SYSTEM_CLOCK",
    "MANIFEST_OPERATOR_CONFIRMED",
}
_MAX_PHOTO_BYTES = 100 * 1024 * 1024
_MAX_PHOTO_PIXELS = 50_000_000


def _require_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_positive_integer(value: object, field: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _validate_timestamp(value: object, field: str) -> None:
    timestamp = _require_nonempty_string(value, field)
    normalized = (
        timestamp[:-1] + "+00:00" if timestamp.endswith(("Z", "z")) else timestamp
    )
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 timestamp with a time zone") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a UTC offset or Z suffix")


def _validate_view(view: Mapping[str, Any], field: str) -> tuple[str, str]:
    _require_nonempty_string(view.get("camera_id"), f"{field}.camera_id")
    path = _require_nonempty_string(view.get("path"), f"{field}.path")
    if "\\" in path:
        raise ValueError(f"{field}.path must use portable forward slashes")
    portable_path = PurePosixPath(path.replace("\\", "/"))
    windows_path = PureWindowsPath(path)
    if portable_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ValueError(f"{field}.path must be relative to the manifest directory")
    if ".." in portable_path.parts:
        raise ValueError(f"{field}.path must not leave the manifest directory")
    sha256 = _require_nonempty_string(view.get("sha256"), f"{field}.sha256")
    if _SHA256_PATTERN.fullmatch(sha256) is None:
        raise ValueError(f"{field}.sha256 must contain exactly 64 hexadecimal characters")
    expected_width = _require_positive_integer(
        view.get("expected_width"), f"{field}.expected_width"
    )
    expected_height = _require_positive_integer(
        view.get("expected_height"), f"{field}.expected_height"
    )
    if expected_width * expected_height > _MAX_PHOTO_PIXELS:
        raise ValueError(
            f"{field} declares more than {_MAX_PHOTO_PIXELS} decoded pixels"
        )
    _validate_timestamp(view.get("captured_at"), f"{field}.captured_at")
    timestamp_source = _require_nonempty_string(
        view.get("timestamp_source"), f"{field}.timestamp_source"
    )
    if timestamp_source not in _TIMESTAMP_SOURCES:
        raise ValueError(
            f"{field}.timestamp_source must be one of {sorted(_TIMESTAMP_SOURCES)}"
        )
    if view.get("orientation_policy") != _ORIENTATION_POLICY:
        raise ValueError(
            f"{field}.orientation_policy must be {_ORIENTATION_POLICY!r}"
        )
    return path, sha256.lower()


def _model_pipe_ids(manifest: Mapping[str, Any]) -> set[str]:
    model = manifest.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Manifest model must be an object")
    pipes = model.get("pipes")
    if not isinstance(pipes, list) or not pipes:
        raise ValueError("Manifest model.pipes must be a non-empty list")

    pipe_ids: list[str] = []
    for index, pipe in enumerate(pipes):
        if not isinstance(pipe, Mapping):
            raise ValueError(f"model.pipes[{index}] must be an object")
        pipe_ids.append(
            _require_nonempty_string(pipe.get("pipe_id"), f"model.pipes[{index}].pipe_id")
        )
    if len(set(pipe_ids)) != len(pipe_ids):
        raise ValueError("model pipe_id values must be unique")
    return set(pipe_ids)


def _finite_number(value: object, field: str, *, allow_zero: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{field} must be a finite number")
    numeric = float(value)
    if numeric < 0 or (not allow_zero and numeric == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{field} must be {qualifier}")
    return numeric


def _validate_analysis_config(
    analysis: Mapping[str, Any], manifest: Mapping[str, Any]
) -> None:
    roi = analysis.get("analysis_roi_xyxy")
    if (
        not isinstance(roi, list)
        or len(roi) != 4
        or any(type(value) is not int for value in roi)
    ):
        raise ValueError("capture.analysis.analysis_roi_xyxy must contain four integers")
    if min(roi) < 0 or roi[0] >= roi[2] or roi[1] >= roi[3]:
        raise ValueError(
            "capture.analysis.analysis_roi_xyxy must use non-negative bounds "
            "with positive area"
        )

    _require_positive_integer(
        analysis.get("minimum_component_area_px"),
        "capture.analysis.minimum_component_area_px",
    )
    for key in ("minimum_long_side_px", "minimum_aspect_ratio", "delta_e76_tolerance"):
        _finite_number(analysis.get(key), f"capture.analysis.{key}")
    for key in ("diameter_absolute_tolerance_mm", "diameter_relative_tolerance"):
        _finite_number(
            analysis.get(key),
            f"capture.analysis.{key}",
            allow_zero=True,
        )

    color_rules = analysis.get("color_rules")
    if not isinstance(color_rules, Mapping):
        raise ValueError("capture.analysis.color_rules must be an object")
    model = manifest["model"]
    required_colors = {
        _require_nonempty_string(
            pipe.get("appearance_color"), "model pipe appearance_color"
        )
        for pipe in model["pipes"]
    }
    missing_colors = required_colors - set(color_rules)
    if missing_colors:
        raise ValueError(
            "capture.analysis.color_rules is missing model colors: "
            f"{sorted(missing_colors)}"
        )
    for color in required_colors:
        rules = color_rules[color]
        if not isinstance(rules, list) or not rules:
            raise ValueError(
                f"capture.analysis.color_rules.{color} must be a non-empty list"
            )
        for index, rule in enumerate(rules):
            field = f"capture.analysis.color_rules.{color}[{index}]"
            if (
                not isinstance(rule, list)
                or len(rule) != 6
                or any(type(value) is not int for value in rule)
            ):
                raise ValueError(f"{field} must contain six integers")
            limits = (179, 179, 255, 255, 255, 255)
            if any(value < 0 or value > limit for value, limit in zip(rule, limits)):
                raise ValueError(f"{field} contains an out-of-range HSV bound")
            if rule[0] > rule[1] or rule[2] > rule[3] or rule[4] > rule[5]:
                raise ValueError(f"{field} lower bounds must not exceed upper bounds")


def validate_still_capture_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate the current fail-closed, mono still-capture manifest contract."""

    if not isinstance(manifest, Mapping):
        raise ValueError("Manifest must be an object")
    if manifest.get("schema_version") != "1.1":
        raise ValueError("Still-capture manifests require schema_version='1.1'")
    capture = manifest.get("capture")
    if not isinstance(capture, Mapping):
        raise ValueError("Manifest capture must be an object")
    if capture.get("kind") != "still_capture_set":
        raise ValueError("capture.kind must be 'still_capture_set'")
    if capture.get("camera_layout") != "mono":
        raise ValueError(
            "The current still-capture implementation supports only "
            "camera_layout='mono'"
        )
    _require_nonempty_string(
        capture.get("capture_group_id"), "capture.capture_group_id"
    )
    calibration_id = capture.get("calibration_id")
    if calibration_id is not None:
        _require_nonempty_string(calibration_id, "capture.calibration_id")

    interval = capture.get("interval_minutes")
    try:
        interval_is_finite = (
            type(interval) in (int, float) and math.isfinite(interval)
        )
    except OverflowError:
        interval_is_finite = False
    if not interval_is_finite:
        raise ValueError("capture.interval_minutes must be a finite number")
    if not 5 <= interval <= 60:
        raise ValueError("capture.interval_minutes must be between 5 and 60 inclusive")

    analysis = capture.get("analysis")
    if not isinstance(analysis, Mapping):
        raise ValueError("capture.analysis must be an object")

    known_pipe_ids = _model_pipe_ids(manifest)
    _validate_analysis_config(analysis, manifest)
    captures = capture.get("captures")
    if not isinstance(captures, list) or not captures:
        raise ValueError("capture.captures must be a non-empty list")

    seen_capture_ids: set[str] = set()
    seen_paths: set[str] = set()
    for index, item in enumerate(captures):
        item_field = f"capture.captures[{index}]"
        if not isinstance(item, Mapping):
            raise ValueError(f"{item_field} must be an object")
        capture_id = _require_nonempty_string(
            item.get("capture_id"), f"{item_field}.capture_id"
        )
        if capture_id in seen_capture_ids:
            raise ValueError(f"Duplicate capture_id: {capture_id!r}")
        seen_capture_ids.add(capture_id)

        views = item.get("views")
        if not isinstance(views, Mapping) or set(views) != {"mono"}:
            raise ValueError(f"{item_field}.views must contain exactly the 'mono' view")
        mono = views["mono"]
        if not isinstance(mono, Mapping):
            raise ValueError(f"{item_field}.views.mono must be an object")
        path, _ = _validate_view(mono, f"{item_field}.views.mono")
        roi = analysis["analysis_roi_xyxy"]
        if roi[2] > mono["expected_width"] or roi[3] > mono["expected_height"]:
            raise ValueError(
                f"{item_field}.views.mono dimensions do not contain the analysis ROI"
            )

        path_key = str(PurePosixPath(path.replace("\\", "/"))).casefold()
        if path_key in seen_paths:
            raise ValueError(f"Still-photo paths must be unique: {path!r}")
        seen_paths.add(path_key)

        expected_ids = item.get("expected_visible_pipe_ids")
        if expected_ids is None:
            continue
        if not isinstance(expected_ids, list) or any(
            not isinstance(pipe_id, str) or not pipe_id for pipe_id in expected_ids
        ):
            raise ValueError(
                f"{item_field}.expected_visible_pipe_ids must be null or a list of pipe_id strings"
            )
        if len(set(expected_ids)) != len(expected_ids):
            raise ValueError(
                f"{item_field}.expected_visible_pipe_ids must not contain duplicates"
            )
        unknown_ids = set(expected_ids) - known_pipe_ids
        if unknown_ids:
            raise ValueError(
                f"{item_field}.expected_visible_pipe_ids contains unknown pipe_id values: "
                f"{sorted(unknown_ids)}"
            )


def analysis_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return a detached detector configuration for legacy video or still input."""

    has_capture_key = "capture" in manifest
    has_video_key = "video" in manifest
    if has_capture_key == has_video_key:
        raise ValueError("Manifest must contain exactly one legacy video or still capture object")
    source_key = "capture" if has_capture_key else "video"
    source = manifest.get(source_key)
    if not isinstance(source, Mapping):
        raise ValueError(f"Manifest {source_key} must be an object")
    if has_capture_key:
        validate_still_capture_manifest(manifest)
        analysis = source["analysis"]
        assert isinstance(analysis, Mapping)
        return dict(analysis)

    return dict(source)


def _encoded_image_dimensions(raw: bytes) -> tuple[str, int, int]:
    png_signature = b"\x89PNG\r\n\x1a\n"
    if raw.startswith(png_signature):
        if len(raw) < 24 or raw[12:16] != b"IHDR":
            raise ValueError("Malformed PNG header")
        width = int.from_bytes(raw[16:20], "big")
        height = int.from_bytes(raw[20:24], "big")
        return "png", width, height

    if not raw.startswith(b"\xff\xd8"):
        raise ValueError("Still photos must be encoded as PNG or JPEG")

    start_of_frame_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    offset = 2
    while offset < len(raw):
        if raw[offset] != 0xFF:
            raise ValueError("Malformed JPEG marker stream")
        while offset < len(raw) and raw[offset] == 0xFF:
            offset += 1
        if offset >= len(raw):
            break
        marker = raw[offset]
        offset += 1
        if marker in {0x01, *range(0xD0, 0xD9)}:
            continue
        if marker in {0xD9, 0xDA}:
            break
        if offset + 2 > len(raw):
            raise ValueError("Truncated JPEG segment")
        segment_length = int.from_bytes(raw[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(raw):
            raise ValueError("Invalid JPEG segment length")
        if marker in start_of_frame_markers:
            if segment_length < 7:
                raise ValueError("Truncated JPEG start-of-frame segment")
            height = int.from_bytes(raw[offset + 3 : offset + 5], "big")
            width = int.from_bytes(raw[offset + 5 : offset + 7], "big")
            return "jpeg", width, height
        offset += segment_length
    raise ValueError("JPEG does not contain a supported start-of-frame marker")


def resolve_photo_path(manifest_file: Path, relative_path: str) -> Path:
    """Resolve a validated photo path without allowing dataset-root escape."""

    source_root = Path(manifest_file).resolve().parent
    source_path = (source_root / relative_path).resolve()
    try:
        source_path.relative_to(source_root)
    except ValueError as error:
        raise ValueError("view.path must resolve inside the manifest directory") from error
    return source_path


def load_photo_snapshot(
    manifest_file: Path,
    view: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Hash and decode one immutable raw-byte snapshot without applying EXIF orientation."""

    if not isinstance(view, Mapping):
        raise ValueError("Photo view must be an object")
    relative_path, expected_sha256 = _validate_view(view, "view")
    source_path = resolve_photo_path(Path(manifest_file), relative_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    encoded_size = source_path.stat().st_size
    if encoded_size <= 0 or encoded_size > _MAX_PHOTO_BYTES:
        raise ValueError(
            f"Photo size for {relative_path!r} must be between 1 and "
            f"{_MAX_PHOTO_BYTES} bytes"
        )

    with source_path.open("rb") as stream:
        raw = stream.read(_MAX_PHOTO_BYTES + 1)
    if len(raw) != encoded_size or len(raw) > _MAX_PHOTO_BYTES:
        raise ValueError(f"Photo size changed while reading {relative_path!r}")
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"Photo SHA-256 mismatch for {relative_path!r}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )

    encoded_format, header_width, header_height = _encoded_image_dimensions(raw)
    expected_width = _require_positive_integer(
        view.get("expected_width"), "view.expected_width"
    )
    expected_height = _require_positive_integer(
        view.get("expected_height"), "view.expected_height"
    )
    if header_width != expected_width or header_height != expected_height:
        raise ValueError(
            f"Encoded photo dimensions for {relative_path!r} are "
            f"{header_width}x{header_height}, expected {expected_width}x{expected_height}"
        )
    if header_width * header_height > _MAX_PHOTO_PIXELS:
        raise ValueError(f"Photo {relative_path!r} exceeds the decoded-pixel safety limit")

    encoded = np.frombuffer(raw, dtype=np.uint8)
    decode_flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    frame = cv2.imdecode(encoded, decode_flags)
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"OpenCV could not decode a three-channel photo: {relative_path!r}")

    height, width = frame.shape[:2]
    if width != expected_width or height != expected_height:
        raise ValueError(
            f"Photo dimensions for {relative_path!r} are {width}x{height}, "
            f"expected {expected_width}x{expected_height} raw pixels"
        )

    integrity = {
        "path": relative_path,
        "expected_sha256": expected_sha256,
        "actual_sha256": actual_sha256,
        "verified": True,
        "size_bytes": len(raw),
        "encoded_format": encoded_format,
        "decoded_width": width,
        "decoded_height": height,
        "orientation_policy": _ORIENTATION_POLICY,
        "exif_orientation_transform_applied": False,
    }
    return frame, integrity
