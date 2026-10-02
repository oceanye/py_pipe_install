"""Unattended stereo image capture for a remote field computer.

The command deliberately stores raw camera pixels.  It does not calibrate or
rectify images; the resulting JSON is an auditable hand-off that can be
converted into a field capture manifest after calibration has been selected.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import cv2
import numpy as np

from .pipeline import atomic_write_text
from .calibration_wizard import detect_board_corners
from .stereo_camera import (
    LAYOUT_SIDE_BY_SIDE_LR,
    LAYOUT_SIDE_BY_SIDE_RL,
    StereoCameraSession,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")


def _chessboard_status(
    image: np.ndarray,
    *,
    columns: int,
    rows: int,
) -> dict[str, Any]:
    errors: list[dict[str, str]] = []
    observation = detect_board_corners(
        image,
        pattern=(columns, rows),
        sb_accuracy=False,
        on_cv_error=lambda stage, error: errors.append({"stage": stage, "error": str(error)}),
    )
    result: dict[str, Any] = {
        "found": observation is not None,
        "inner_corners": [int(columns), int(rows)],
        "detector": observation.detector if observation is not None else None,
        "errors": errors,
    }
    if observation is not None:
        corners = observation.corners_px
        # Keep only a compact quality signal; raw images remain the source of
        # truth and can be fed to the calibration wizard later.
        result["corner_count"] = int(len(corners))
        result["image_center_distance_px"] = round(
            float(
                np.linalg.norm(
                    np.mean(corners.reshape(-1, 2), axis=0)
                    - np.asarray([image.shape[1] / 2, image.shape[0] / 2])
                )
            ),
            3,
        )
    return result


def _encode_png(image: np.ndarray) -> bytes:
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("camera frame must be a three-channel BGR image")
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("OpenCV could not encode a camera frame as PNG")
    return bytes(encoded)


IMAGE_HEALTH_POLICY: dict[str, Any] = {
    "name": "LUMINANCE_EXPOSURE_DIAGNOSTIC_V1",
    "usable_rule": "p50>=3 AND p95>=12 AND max>=16 AND dark_fraction<0.995",
    "purpose": "flag exposure transitions without discarding raw evidence",
}


def _image_health(image: np.ndarray) -> dict[str, Any]:
    """Return a deterministic, non-destructive exposure health record.

    This is deliberately a quality signal rather than a capture gate.  A
    camera may recover from an exposure transition after a frame is read, and
    the raw frame is still useful when diagnosing that transition.
    """
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 3:
        gray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    elif array.ndim == 2:
        gray = array
    else:
        raise ValueError("camera frame must be a grayscale or three-channel image")
    gray = np.asarray(gray, dtype=np.uint8)
    p05, p50, p95 = np.percentile(gray, [5, 50, 95])
    dark_fraction = float(np.mean(gray <= 2))
    saturated_fraction = float(np.mean(gray >= 253))
    maximum = int(np.max(gray))
    usable = bool(p50 >= 3.0 and p95 >= 12.0 and maximum >= 16 and dark_fraction < 0.995)
    reasons: list[str] = []
    if p50 < 3.0:
        reasons.append("LOW_LUMINANCE_P50")
    if p95 < 12.0:
        reasons.append("LOW_LUMINANCE_P95")
    if maximum < 16:
        reasons.append("LOW_LUMINANCE_MAX")
    if dark_fraction >= 0.995:
        reasons.append("NEAR_BLACK_FRACTION")
    return {
        "width": int(gray.shape[1]),
        "height": int(gray.shape[0]),
        "mean": round(float(np.mean(gray)), 3),
        "p05": round(float(p05), 3),
        "p50": round(float(p50), 3),
        "p95": round(float(p95), 3),
        "max": maximum,
        "dark_fraction": round(dark_fraction, 6),
        "saturated_fraction": round(saturated_fraction, 6),
        "usable": usable,
        "reason_codes": reasons,
    }


def _pair_image_health(pair: Any) -> dict[str, Any]:
    left = _image_health(pair.left)
    right = _image_health(pair.right)
    identical = bool(np.array_equal(np.asarray(pair.left), np.asarray(pair.right)))
    reasons = sorted(set(left["reason_codes"] + right["reason_codes"]))
    if identical:
        reasons.append("LEFT_RIGHT_IDENTICAL")
    return {
        "left": left,
        "right": right,
        "left_right_identical": identical,
        "pair_usable": bool(left["usable"] and right["usable"] and not identical),
        "reason_codes": sorted(set(reasons)),
    }


def _startup_pair_health(pair: Any) -> tuple[str | None, dict[str, Any]]:
    """Reject deterministic UVC startup frames before they enter evidence."""

    left = np.asarray(pair.left)
    right = np.asarray(pair.right)
    stats = {
        "left_mean": round(float(np.mean(left)), 3),
        "right_mean": round(float(np.mean(right)), 3),
        "left_max": int(np.max(left)),
        "right_max": int(np.max(right)),
        "left_right_identical": bool(np.array_equal(left, right)),
    }
    if stats["left_max"] <= 2 and stats["right_max"] <= 2:
        return "BOTH_EYES_NEAR_BLACK", stats
    if stats["left_right_identical"]:
        return "LEFT_RIGHT_IDENTICAL", stats
    return None, stats


def capture_stereo_pairs(
    output_dir: str | Path,
    *,
    left_index: int = 0,
    right_index: int | None = None,
    layout: str = LAYOUT_SIDE_BY_SIDE_LR,
    eye_width: int = 640,
    eye_height: int = 480,
    count: int | None = None,
    interval_s: float = 0.0,
    duration_s: float | None = None,
    detect_chessboard: bool = False,
    board_columns: int = 11,
    board_rows: int = 7,
    backend: int | None = None,
    session_factory: Callable[..., Any] = StereoCameraSession,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    startup_max_reads: int = 20,
) -> Path:
    """Capture paired images and return the generated ``capture.json`` path.

    ``count`` limits the number of pairs.  ``duration_s`` is checked between
    reads and is not a hard timeout for blocking camera or detector calls;
    when both are supplied the first limit reached wins.  With neither set a
    single pair is captured.  Existing output directories are rejected to
    prevent accidental mixing of runs.
    """
    if count is None and duration_s is None:
        count = 1
    if count is not None and (type(count) is not int or count <= 0):
        raise ValueError("count must be a positive integer or None")
    if duration_s is not None and (
        isinstance(duration_s, bool) or not np.isfinite(float(duration_s)) or duration_s <= 0
    ):
        raise ValueError("duration_s must be a positive finite number")
    if not np.isfinite(float(interval_s)) or interval_s < 0:
        raise ValueError("interval_s must be a non-negative finite number")
    if type(board_columns) is not int or board_columns < 3 or type(board_rows) is not int or board_rows < 3:
        raise ValueError("board_columns and board_rows must be integers >= 3")
    if type(startup_max_reads) is not int or startup_max_reads <= 0:
        raise ValueError("startup_max_reads must be a positive integer")

    root = Path(output_dir).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"capture output directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    run_id = f"remote-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    records: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "capture_kind": "remote_stereo_raw",
        "run_id": run_id,
        "status": "RUNNING",
        "started_at": _timestamp(),
        "camera_opened_at": None,
        "first_frame_at": None,
        "finished_at": None,
        "pixel_policy": "RAW_CAMERA_BGR_PIXELS_NO_TRANSFORM",
        "rectified": False,
        "calibration_validated": False,
        "validation_scope": "RAW_CAPTURE_ONLY_NOT_CALIBRATION_VALIDATION",
        "camera_layout": layout,
        "camera": {
            "layout": layout,
            "left_index": int(left_index),
            "right_index": None if right_index is None else int(right_index),
            "backend": None if backend is None else int(backend),
        },
        "requested_eye_size_px": [int(eye_width), int(eye_height)],
        "requested_frame_size_px": [int(eye_width * 2), int(eye_height)]
        if layout in {LAYOUT_SIDE_BY_SIDE_LR, LAYOUT_SIDE_BY_SIDE_RL}
        else [int(eye_width), int(eye_height)],
        "pair_count": 0,
        "usable_pair_count": 0,
        "unusable_pair_count": 0,
        "quality_status": "PENDING",
        "quality_scope": "PER_FRAME_EXPOSURE_DIAGNOSTIC_RAW_FRAMES_RETAINED",
        "image_health_policy": IMAGE_HEALTH_POLICY,
        "requested_pair_count": count,
        "discarded_pair_count": 0,
        "startup_warmup": {
            "max_reads": startup_max_reads,
            "discarded": [],
            "accepted_read": None,
        },
        "interval_s": float(interval_s),
        "duration_limit_s": None if duration_s is None else float(duration_s),
        "duration_policy": "CHECK_BETWEEN_READS_NOT_HARD_TIMEOUT",
        "interval_buffer_policy": "CONTINUOUS_READ_AND_DISCARD",
        "chessboard_detection": {
            "enabled": bool(detect_chessboard),
            "inner_corners": [int(board_columns), int(board_rows)],
        },
        "captures": records,
    }
    manifest = root / "capture.json"

    def persist() -> None:
        payload["pair_count"] = len(records)
        payload["usable_pair_count"] = sum(
            1 for item in records if item.get("image_health", {}).get("pair_usable") is True
        )
        payload["unusable_pair_count"] = len(records) - payload["usable_pair_count"]
        if not records:
            payload["quality_status"] = "PENDING"
        elif payload["usable_pair_count"] == len(records):
            payload["quality_status"] = "USABLE"
        elif payload["usable_pair_count"]:
            payload["quality_status"] = "PARTIAL"
        else:
            payload["quality_status"] = "UNUSABLE"
        atomic_write_text(manifest, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    # This record exists even if constructing/opening a camera session fails.
    persist()
    session_kwargs = {
        "layout": layout,
        "left_index": left_index,
        "right_index": right_index,
        "eye_width": eye_width,
        "eye_height": eye_height,
        "backend": backend,
    }
    stage = "open_camera"
    try:
        with session_factory(**session_kwargs) as session:
            started = clock()
            payload["camera_opened_at"] = _timestamp()
            persist()
            pending_pair = None
            for startup_read in range(1, startup_max_reads + 1):
                stage = "startup_warmup"
                candidate = session.read_pair()
                issue, stats = _startup_pair_health(candidate)
                if issue is None:
                    payload["startup_warmup"]["accepted_read"] = startup_read
                    payload["startup_warmup"]["accepted_stats"] = stats
                    pending_pair = candidate
                    persist()
                    break
                payload["startup_warmup"]["discarded"].append(
                    {"read": startup_read, "reason": issue, **stats}
                )
                payload["discarded_pair_count"] += 1
                persist()
            if pending_pair is None:
                raise RuntimeError(
                    f"camera did not produce a non-black distinct stereo pair "
                    f"within {startup_max_reads} startup reads"
                )
            while (count is None or len(records) < count) and (
                duration_s is None or (clock() - started) < duration_s
            ):
                stage = "read_pair"
                if pending_pair is not None:
                    pair = pending_pair
                    pending_pair = None
                else:
                    pair = session.read_pair()
                stage = "encode_png"
                left_data = _encode_png(pair.left)
                right_data = _encode_png(pair.right)
                sequence = len(records) + 1
                left_name = f"pair_{sequence:04d}_left.png"
                right_name = f"pair_{sequence:04d}_right.png"
                stage = "write_png"
                (root / left_name).write_bytes(left_data)
                (root / right_name).write_bytes(right_data)
                stage = "image_health"
                image_health = _pair_image_health(pair)
                record: dict[str, Any] = {
                    "capture_id": f"{run_id}-{sequence:04d}",
                    "left": {
                        "path": left_name,
                        "sha256": hashlib.sha256(left_data).hexdigest(),
                        "width": int(pair.left.shape[1]),
                        "height": int(pair.left.shape[0]),
                        "captured_at": pair.left_captured_at,
                    },
                    "right": {
                        "path": right_name,
                        "sha256": hashlib.sha256(right_data).hexdigest(),
                        "width": int(pair.right.shape[1]),
                        "height": int(pair.right.shape[0]),
                        "captured_at": pair.right_captured_at,
                    },
                    "sync_delta_ms": float(pair.sync_delta_ms),
                    "timestamp_source": pair.timestamp_source,
                    "provenance": pair.provenance,
                    "image_health": image_health,
                }
                records.append(record)
                if len(records) == 1:
                    payload["first_frame_at"] = pair.left_captured_at
                # Make the completed image pair reviewable before potentially
                # slow chessboard detection or another camera read.
                persist()
                if detect_chessboard:
                    stage = "detect_chessboard"
                    record["chessboard"] = {
                        "left": _chessboard_status(pair.left, columns=board_columns, rows=board_rows),
                        "right": _chessboard_status(pair.right, columns=board_columns, rows=board_rows),
                    }
                    persist()
                if interval_s and (count is None or len(records) < count):
                    next_capture_at = clock() + interval_s
                    if duration_s is not None:
                        next_capture_at = min(next_capture_at, started + duration_s)
                    # CAP_PROP_BUFFERSIZE is advisory for many UVC drivers.
                    # Continue consuming frames instead of sleeping through a
                    # growing queue of stale frames between saved captures.
                    while clock() < next_capture_at:
                        stage = "drain_interval_frames"
                        session.read_pair()
                        payload["discarded_pair_count"] += 1
                        remaining = next_capture_at - clock()
                        if remaining > 0:
                            sleep(min(0.005, remaining))
            stage = "close_camera"
        if not records:
            stage = "check_pair_count"
            raise RuntimeError("no stereo pairs captured before the duration limit")
    except (Exception, KeyboardInterrupt) as error:
        payload["status"] = "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED"
        payload["error"] = {"stage": stage, "type": type(error).__name__, "message": str(error)}
        payload["finished_at"] = _timestamp()
        persist()
        raise
    payload["status"] = "COMPLETED"
    payload["stop_reason"] = "COUNT_LIMIT" if count is not None and len(records) >= count else "DURATION_LIMIT"
    payload["finished_at"] = _timestamp()
    persist()
    return manifest


__all__ = ["capture_stereo_pairs"]
