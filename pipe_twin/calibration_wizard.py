"""Built-in chessboard stereo calibration wizard.

Replaces the external calibration JSON with an in-app procedure: print a
chessboard target, grab >=10 pose-diverse stereo pairs from the connected
camera, then run ``cv2.stereoCalibrate`` + ``stereoRectify``.  The wizard
emits

* a calibration dict in the exact shape required by
  ``stereo_analyzer._calibration_from_manifest`` — it describes the
  *rectified* rig (identical rotations, baseline along +X, zero distortion,
  the common rectified K), and
* a *rectification recipe* (original K/D plus R1/R2/P1/P2) stored in the
  workbench profile, outside the calibration contract, so raw camera frames
  can be remapped into the declared rectified images before capture.

The calibration contract schema itself is never changed here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import struct
import threading
import time
import uuid
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import cv2
import numpy as np

from .workbench_profile import (
    calibration_ids_match,
    validate_rectification_recipe,
)


CHESS_ID_MARKER = "-chess-"
_OPERATOR_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")

# Sanity bounds for recovered parameters; far outside means the solve drifted.
_FOCAL_RANGE = (0.2, 4.0)
_BASELINE_RANGE_MM = (20.0, 2000.0)
_MIN_CENTROID_SPREAD_RATIO = 0.10
_MIN_TILT_SPAN_DEG = 8.0
_MIN_DEPTH_SPREAD_RATIO = 0.08
_MIN_PROJECTED_SCALE_SPAN_RATIO = 0.08
_MIN_PROJECTIVE_SHAPE_SPAN = 0.04
_MAX_DISTORTION_ABS = 5.0
_DIAGNOSTIC_ROOT = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "measurement_workbench"
    / "calibration_diagnostics"
)
_CALIBRATION_CAPTURE_ROOT = (
    Path(__file__).resolve().parents[1]
    / "outputs"
    / "measurement_workbench"
    / "calibration_captures"
)


class ChessboardCalibrationError(ValueError):
    """Raised when the wizard cannot produce a trustworthy calibration."""


def read_calibration_image(path: str | Path) -> np.ndarray | None:
    """Read a calibration image from a Unicode-safe filesystem path.

    ``cv2.imread`` still fails for many non-ASCII Windows paths. The vendor
    bundle lives below a Chinese directory, so decode bytes after Python has
    opened the path instead of passing the path string to OpenCV.
    """

    try:
        encoded = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    try:
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    except cv2.error:
        return None


# ---------------------------------------------------------------------------
# Printable target
# ---------------------------------------------------------------------------


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


# Printable area inside the A4 page: 10 mm side margins, header text above
# y=42 mm and the 100 mm check line below y=265 mm.
_A4_BOARD_WIDTH_MM = 190.0
_A4_BOARD_HEIGHT_MM = 220.0


def printable_chessboard_png(
    *,
    square_mm: float,
    columns: int,
    rows: int,
    dpi: int = 300,
) -> bytes:
    """Return an A4 PNG chessboard with an explicit DPI and metric labels."""
    if type(square_mm) not in (int, float) or not math.isfinite(square_mm) or square_mm <= 0:
        raise ChessboardCalibrationError("棋盘格格边长必须是正的有限数值")
    if type(columns) is not int or type(rows) is not int or columns < 4 or rows < 4:
        raise ChessboardCalibrationError("棋盘格列数和行数都必须是不小于 4 的整数")
    if type(dpi) is not int or not 150 <= dpi <= 1200:
        raise ChessboardCalibrationError("PNG 打印分辨率应为 150 到 1200 DPI 的整数")
    pixels_per_mm = dpi / 25.4
    page_width = round(210.0 * pixels_per_mm)
    page_height = round(297.0 * pixels_per_mm)
    square_px = square_mm * pixels_per_mm
    width_mm = columns * square_mm
    height_mm = rows * square_mm
    if width_mm > _A4_BOARD_WIDTH_MM or height_mm > _A4_BOARD_HEIGHT_MM:
        maximum_columns = int(_A4_BOARD_WIDTH_MM // square_mm)
        maximum_rows = int(_A4_BOARD_HEIGHT_MM // square_mm)
        raise ChessboardCalibrationError(
            f"棋盘格 {columns}×{rows} 格、格边长 {square_mm:g} mm 需要 "
            f"{width_mm:g}×{height_mm:g} mm，超出 A4 可打印区域 "
            f"{_A4_BOARD_WIDTH_MM:g}×{_A4_BOARD_HEIGHT_MM:g} mm。"
            f"当前格边长下最多 {maximum_columns} 列 × {maximum_rows} 行；"
            "请减小“打印后实测格边长”或减少列/行数。"
        )

    page = np.full((page_height, page_width), 255, dtype=np.uint8)
    board_width_px = round(columns * square_px)
    board_height_px = round(rows * square_px)
    board_left = (page_width - board_width_px) // 2
    board_top = round(42.0 * pixels_per_mm)
    for row in range(rows):
        y1 = board_top + round(row * square_px)
        y2 = board_top + round((row + 1) * square_px)
        for column in range(columns):
            if (row + column) % 2 != 0:
                continue
            x1 = board_left + round(column * square_px)
            x2 = board_left + round((column + 1) * square_px)
            page[y1:y2, x1:x2] = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(page, "PIPE TWIN CHESSBOARD CALIBRATION", (round(22 * pixels_per_mm), round(12 * pixels_per_mm)), font, 1.05, 0, 3, cv2.LINE_AA)
    cv2.putText(page, "A4 / PRINT 100% ACTUAL SIZE (NO FIT-TO-PAGE)", (round(24 * pixels_per_mm), round(21 * pixels_per_mm)), font, 0.72, 0, 2, cv2.LINE_AA)
    cv2.putText(
        page,
        f"SQUARE {float(square_mm):.3f} mm  GRID {columns} x {rows} SQUARES"
        f"  ({columns - 1} x {rows - 1} INNER CORNERS)",
        (round(14 * pixels_per_mm), round(30 * pixels_per_mm)),
        font,
        0.62,
        0,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        page,
        "MEASURE A PRINTED SQUARE EDGE AND ENTER IT IN THE WIZARD",
        (round(16 * pixels_per_mm), round(37 * pixels_per_mm)),
        font,
        0.62,
        0,
        2,
        cv2.LINE_AA,
    )

    scale_length = round(100.0 * pixels_per_mm)
    scale_left = (page_width - scale_length) // 2
    scale_y = page_height - round(25 * pixels_per_mm)
    tick = round(3 * pixels_per_mm)
    cv2.line(page, (scale_left, scale_y), (scale_left + scale_length, scale_y), 0, 3)
    cv2.line(page, (scale_left, scale_y - tick), (scale_left, scale_y + tick), 0, 3)
    cv2.line(page, (scale_left + scale_length, scale_y - tick), (scale_left + scale_length, scale_y + tick), 0, 3)
    cv2.putText(page, "CHECK LINE: 100.000 mm", (scale_left + round(19 * pixels_per_mm), scale_y - round(5 * pixels_per_mm)), font, 0.72, 0, 2, cv2.LINE_AA)

    ok, encoded = cv2.imencode(".png", page, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise ChessboardCalibrationError("OpenCV 无法编码棋盘格 PNG")
    png = bytes(encoded)
    pixels_per_metre = round(dpi / 0.0254)
    physical = _png_chunk(b"pHYs", struct.pack(">IIB", pixels_per_metre, pixels_per_metre, 1))
    metadata = _png_chunk(
        b"tEXt",
        (
            f"Description\x00pipe-twin-chessboard;square_mm={float(square_mm):.3f};"
            f"columns={columns};rows={rows};dpi={dpi};square_px={square_px:.6f};"
            f"board_px={board_width_px}x{board_height_px};page_px={page_width}x{page_height}"
        ).encode("latin-1"),
    )
    return png[:33] + physical + metadata + png[33:]


def write_printable_chessboard_png(
    path: str | Path,
    *,
    square_mm: float,
    columns: int,
    rows: int,
    dpi: int = 300,
) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != ".png":
        destination = destination.with_suffix(".png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(
        printable_chessboard_png(square_mm=square_mm, columns=columns, rows=rows, dpi=dpi)
    )
    return destination


# Compatibility facade for the non-interactive CLI.  The workbench uses the
# richer ``ChessboardWizardDialog`` below; scripts can still pass two sorted
# image folders and receive the same manifest calibration contract.
CalibrationWizardError = ChessboardCalibrationError


def calibrate_stereo_from_folders(
    left_dir: str | Path,
    right_dir: str | Path,
    *,
    board_columns: int = 9,
    board_rows: int = 6,
    square_size_mm: float = 25.0,
    calibration_id: str = "FIELD-AUTO-STEREO",
    min_pairs: int = 8,
    max_rms_px: float | None = 1.5,
    expected_baseline_mm: float | None = None,
    left_camera_pose: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the same guarded solve as the GUI over paired image folders."""
    if not isinstance(calibration_id, str) or not _OPERATOR_PATTERN.match(calibration_id):
        raise CalibrationWizardError(
            "calibration_id 只能包含字母、数字、点、下划线和连字符（≤32 字符）"
        )
    def images(folder: str | Path) -> list[Path]:
        path = Path(folder)
        if not path.is_dir():
            raise CalibrationWizardError(f"标定照片目录不存在：{path}")
        values = sorted(item for item in path.iterdir() if item.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
        if not values:
            raise CalibrationWizardError(f"目录没有 PNG/JPEG 标定照片：{path}")
        return values

    left, right = images(left_dir), images(right_dir)
    if len(left) != len(right):
        raise CalibrationWizardError(f"左右照片数量不同（左 {len(left)} 张，右 {len(right)} 张）；请按同一批次补齐。")
    pattern = (int(board_columns), int(board_rows))
    image_pairs: list[tuple[Path, Path, np.ndarray, np.ndarray]] = []
    image_size: tuple[int, int] | None = None
    rejected: list[dict[str, str]] = []
    for left_path, right_path in zip(left, right):
        left_image = read_calibration_image(left_path)
        right_image = read_calibration_image(right_path)
        if left_image is None or right_image is None:
            rejected.append({"left": str(left_path), "right": str(right_path), "reason": "无法读取照片"})
            continue
        size = (int(left_image.shape[1]), int(left_image.shape[0]))
        if (int(right_image.shape[1]), int(right_image.shape[0])) != size:
            rejected.append({"left": str(left_path), "right": str(right_path), "reason": "左右分辨率不同"})
            continue
        image_size = image_size or size
        if size != image_size:
            rejected.append({"left": str(left_path), "right": str(right_path), "reason": "照片分辨率不一致"})
            continue
        image_pairs.append((left_path, right_path, left_image, right_image))

    def detect_pairs(
        *, use_sb: bool
    ) -> tuple[list[tuple[BoardObservation, BoardObservation]], list[dict[str, str]]]:
        pairs: list[tuple[BoardObservation, BoardObservation]] = []
        detection_rejected: list[dict[str, str]] = []
        for left_path, right_path, left_image, right_image in image_pairs:
            left_observation = detect_board_corners(
                left_image,
                pattern=pattern,
                use_sb=use_sb,
                sb_accuracy=use_sb,
                fallback_to_classic=not use_sb,
            )
            right_observation = detect_board_corners(
                right_image,
                pattern=pattern,
                use_sb=use_sb,
                sb_accuracy=use_sb,
                fallback_to_classic=not use_sb,
            )
            if left_observation is None or right_observation is None:
                detection_rejected.append(
                    {
                        "left": str(left_path),
                        "right": str(right_path),
                        "reason": "未同时检测到完整棋盘",
                    }
                )
                continue
            pairs.append((left_observation, right_observation))
        return pairs, detection_rejected

    # Use one detector for the complete offline data set. Classic is the
    # deterministic first pass. Only when it cannot produce the configured
    # minimum number of complete stereo pairs do we retry the complete readable
    # set with strict SB detection. The SB pass is not allowed to fall back to
    # classic, otherwise a nominally-SB solve can still contain mixed detector
    # conventions and be rejected by the solver's own consistency gate.
    pairs, detector_rejected = detect_pairs(use_sb=False)
    detector_mode = "classic"
    if len(pairs) < int(min_pairs):
        pairs, detector_rejected = detect_pairs(use_sb=True)
        detector_mode = "sb"
    rejected.extend(detector_rejected)
    if image_size is None or len(pairs) < int(min_pairs):
        raise CalibrationWizardError(f"有效左右棋盘照片只有 {len(pairs)} 对，至少需要 {int(min_pairs)} 对；请补拍不同位置和角度。")
    result = solve_stereo_calibration(
        pairs=pairs,
        image_size=image_size,
        square_mm=float(square_size_mm),
        pattern=pattern,
        max_reprojection_rms_px=float(max_rms_px if max_rms_px is not None else 1.5),
        min_pairs=int(min_pairs),
        operator=calibration_id,
        max_sync_delta_ms=5.0,
        layout="paired_folders",
        expected_baseline_mm=expected_baseline_mm,
    )
    if not result.validated:
        raise CalibrationWizardError(
            "标定质量门禁未通过：" + "；".join(result.rejection_reasons)
        )
    calibration = copy.deepcopy(result.calibration)
    if left_camera_pose is not None:
        pose = dict(left_camera_pose)
        if set(pose) != {"rotation_world_to_camera", "center_world_mm"}:
            raise CalibrationWizardError(
                "left_camera_pose 必须且只能包含 rotation_world_to_camera 和 center_world_mm"
            )
        rotation = np.asarray(pose["rotation_world_to_camera"], dtype=float)
        center = np.asarray(pose["center_world_mm"], dtype=float)
        if (
            rotation.shape != (3, 3)
            or center.shape != (3,)
            or not np.all(np.isfinite(rotation))
            or not np.all(np.isfinite(center))
            or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
            or not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6)
        ):
            raise CalibrationWizardError("left_camera_pose 不是有效的相机刚体位姿")
        baseline_camera = np.asarray(
            [float(calibration["baseline_mm"]), 0.0, 0.0], dtype=float
        )
        right_center = center + rotation.T @ baseline_camera
        calibration["left_camera"]["rotation_world_to_camera"] = rotation.tolist()
        calibration["right_camera"]["rotation_world_to_camera"] = rotation.tolist()
        calibration["left_camera"]["center_world_mm"] = center.tolist()
        calibration["right_camera"]["center_world_mm"] = right_center.tolist()
        calibration["registration_validated"] = True
        calibration["calibration_id"] = (
            f"{calibration['calibration_id']}-pose-"
            f"{_audit_hash({'rotation': rotation.tolist(), 'center': center.tolist()})}"
        )
        from .stereo_analyzer import _calibration_from_manifest

        _calibration_from_manifest(calibration)
    calibration["source_audit"] = {"auto_calibration": {
        "method": "opencv_chessboard_stereo_calibrate_v2",
        "board_inner_corners": [pattern[0], pattern[1]],
        "square_size_mm": float(square_size_mm),
        "candidate_pairs": len(left),
        "detected_pairs": len(pairs),
        "accepted_pairs": result.pair_count,
        "corner_detector": detector_mode,
        "rejected_pairs": rejected,
        "solver_discarded_pair_indices": result.audit.get("discarded_pair_indices", []),
        "rms_left_px": result.left_rms_px, "rms_right_px": result.right_rms_px,
        "rms_stereo_px": result.stereo_rms_px,
        "right_frame_transform": result.audit["right_frame_transform"],
        "rectification_recipe": result.recipe,
        "solve_audit": result.audit,
        "registration_required": True,
    }}
    return calibration


def write_calibration(path: str | Path, calibration: Mapping[str, Any]) -> Path:
    """Write a portable calibration, including its raw-frame remap when present."""

    source_audit = calibration.get("source_audit", {})
    auto_audit = (
        source_audit.get("auto_calibration", {})
        if isinstance(source_audit, Mapping)
        else {}
    )
    recipe = (
        auto_audit.get("rectification_recipe")
        if isinstance(auto_audit, Mapping)
        else None
    )
    if recipe is not None:
        from .workbench_profile import save_camera_calibration_bundle

        return save_camera_calibration_bundle(
            path,
            calibration,
            rectification_recipe=recipe,
            qr_settings={},
        )
    from .pipeline import atomic_write_text

    destination = Path(path)
    return atomic_write_text(
        destination,
        json.dumps(
            dict(calibration), ensure_ascii=False, indent=2, allow_nan=False
        )
        + "\n",
    )


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoardObservation:
    """One accepted chessboard detection in canonical corner order."""

    corners_px: np.ndarray
    sharpness: float
    centroid_zone: tuple[int, int]
    detector: str = "unknown"

    @property
    def centroid_px(self) -> np.ndarray:
        return self.corners_px.mean(axis=0)


class CalibrationCaptureArchive:
    """Lossless raw left/right frames and an auditable session manifest."""

    def __init__(self, session_path: Path, manifest: dict[str, Any]) -> None:
        self.session_path = session_path
        self.manifest = manifest

    @classmethod
    def create(
        cls,
        *,
        session_metadata: Mapping[str, Any],
        root: str | Path = _CALIBRATION_CAPTURE_ROOT,
    ) -> "CalibrationCaptureArchive":
        destination = Path(root).resolve()
        destination.mkdir(parents=True, exist_ok=True)
        session_id = (
            f"{datetime.now():%Y%m%d_%H%M%S_%f}_"
            f"{uuid.uuid4().hex[:8]}"
        )
        session_path = destination / session_id
        (session_path / "left").mkdir(parents=True)
        (session_path / "right").mkdir(parents=True)
        metadata = json.loads(
            json.dumps(dict(session_metadata), ensure_ascii=False, allow_nan=False)
        )
        archive = cls(
            session_path,
            {
                "kind": "pipe-twin-calibration-capture-session",
                "schema_version": "1.0",
                "session_id": session_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "session": metadata,
                "pairs": [],
            },
        )
        archive._write_manifest()
        return archive

    @classmethod
    def open_existing(
        cls,
        session_path: str | Path,
        *,
        required_root: str | Path = _CALIBRATION_CAPTURE_ROOT,
    ) -> "CalibrationCaptureArchive":
        root = Path(required_root).resolve()
        resolved = Path(session_path).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise ChessboardCalibrationError(
                "原始照片会话不在标定归档目录内"
            ) from error
        manifest_path = resolved / "session.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ChessboardCalibrationError(
                f"无法读取原始照片会话：{error}"
            ) from error
        if (
            not isinstance(manifest, dict)
            or manifest.get("kind") != "pipe-twin-calibration-capture-session"
            or manifest.get("schema_version") != "1.0"
            or not isinstance(manifest.get("session"), dict)
            or not isinstance(manifest.get("pairs"), list)
        ):
            raise ChessboardCalibrationError("原始照片会话清单格式无效")
        return cls(resolved, manifest)

    @property
    def manifest_path(self) -> Path:
        return self.session_path / "session.json"

    def _write_manifest(self) -> None:
        from .pipeline import atomic_write_text

        self.manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_text(
            self.manifest_path,
            json.dumps(
                self.manifest,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
        )

    def update_session(self, values: Mapping[str, Any]) -> None:
        clean = json.loads(
            json.dumps(dict(values), ensure_ascii=False, allow_nan=False)
        )
        self.manifest["session"].update(clean)
        self._write_manifest()

    def add_pair(
        self,
        left_frame: np.ndarray,
        right_frame: np.ndarray,
        *,
        metadata: Mapping[str, Any],
    ) -> int:
        frames = (np.asarray(left_frame), np.asarray(right_frame))
        if any(frame.size == 0 or frame.ndim not in {2, 3} for frame in frames):
            raise ChessboardCalibrationError("标定原始照片数组为空或形状无效")
        capture_id = len(self.manifest["pairs"]) + 1
        filename = f"{capture_id:06d}.png"
        left_path = self.session_path / "left" / filename
        right_path = self.session_path / "right" / filename
        left_temporary = left_path.with_name(f"{left_path.stem}.tmp.png")
        right_temporary = right_path.with_name(f"{right_path.stem}.tmp.png")
        try:
            if not cv2.imwrite(str(left_temporary), frames[0]):
                raise OSError("左目 PNG 写入失败")
            if not cv2.imwrite(str(right_temporary), frames[1]):
                raise OSError("右目 PNG 写入失败")
            left_temporary.replace(left_path)
            right_temporary.replace(right_path)
        finally:
            left_temporary.unlink(missing_ok=True)
            right_temporary.unlink(missing_ok=True)
        record = json.loads(
            json.dumps(dict(metadata), ensure_ascii=False, allow_nan=False)
        )
        record.update(
            {
                "capture_id": capture_id,
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "left_file": left_path.relative_to(self.session_path).as_posix(),
                "right_file": right_path.relative_to(self.session_path).as_posix(),
                "status": "active",
            }
        )
        self.manifest["pairs"].append(record)
        self._write_manifest()
        return capture_id

    def mark_status(
        self,
        capture_ids: list[int],
        *,
        status: str,
        reason: str,
    ) -> None:
        selected = set(int(value) for value in capture_ids)
        if not selected:
            return
        for record in self.manifest["pairs"]:
            if int(record["capture_id"]) in selected:
                record["status"] = str(status)
                record["status_reason"] = str(reason)
                record["status_updated_at"] = datetime.now(timezone.utc).isoformat()
        self._write_manifest()


def _reconcile_restored_capture_status(
    *,
    previous_archive: CalibrationCaptureArchive | None,
    previous_capture_ids: list[int | None],
    restored_archive: CalibrationCaptureArchive | None,
    restored_capture_ids: list[int | None],
) -> None:
    """Keep archive status aligned with the pairs restored into the solver."""

    previous_ids = {
        int(capture_id)
        for capture_id in previous_capture_ids
        if capture_id is not None
    }
    restored_ids = {
        int(capture_id)
        for capture_id in restored_capture_ids
        if capture_id is not None
    }
    if previous_archive is not None:
        replaced_ids = set(previous_ids)
        if (
            restored_archive is not None
            and previous_archive.session_path == restored_archive.session_path
        ):
            replaced_ids -= restored_ids
        previous_archive.mark_status(
            sorted(replaced_ids),
            status="excluded_restore_replaced",
            reason="界面恢复了另一份最近照片组",
        )
    if restored_archive is not None:
        restored_archive.mark_status(
            sorted(restored_ids),
            status="active",
            reason="从最近诊断恢复并继续使用",
        )


def _retain_successful_calibration_pairs(
    pairs: list[tuple[BoardObservation, BoardObservation]],
    capture_ids: list[int | None],
    discarded_pair_indices: list[int],
) -> tuple[
    list[tuple[BoardObservation, BoardObservation]],
    list[int | None],
    list[int],
]:
    """Apply the solver's one-based rejection indices to the live pair set."""

    discarded = {
        int(index)
        for index in discarded_pair_indices
        if 1 <= int(index) <= len(pairs)
    }
    aligned_ids = list(capture_ids[: len(pairs)])
    aligned_ids.extend([None] * (len(pairs) - len(aligned_ids)))
    retained_pairs = [
        pair
        for index, pair in enumerate(pairs, start=1)
        if index not in discarded
    ]
    retained_capture_ids = [
        capture_id
        for index, capture_id in enumerate(aligned_ids, start=1)
        if index not in discarded
    ]
    discarded_capture_ids = [
        int(capture_id)
        for index, capture_id in enumerate(aligned_ids, start=1)
        if index in discarded and capture_id is not None
    ]
    return retained_pairs, retained_capture_ids, discarded_capture_ids


def _canonicalize_corner_order(corners: np.ndarray, pattern: tuple[int, int]) -> np.ndarray:
    """Flip detector output so corner[0] is the top-left-most inner corner.

    OpenCV may return the grid rotated by 180 degrees between images, which
    silently corrupts stereo correspondence; both eyes and all poses must use
    the same labelling.  ``pattern`` is (columns, rows) of inner corners.
    """
    flat = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    columns, rows = int(pattern[0]), int(pattern[1])
    if len(flat) != columns * rows:
        raise ChessboardCalibrationError("棋盘格内角点数量与网格不一致")
    start = flat[0] + flat[1]
    end = flat[-1] + flat[-2]
    if start.sum() > end.sum():
        return flat.reshape(rows, columns, 2)[::-1, ::-1].reshape(-1, 2).copy()
    return flat.copy()


def _laplacian_sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def detect_board_corners(
    image: np.ndarray,
    *,
    pattern: tuple[int, int],
    on_cv_error: Callable[[str, Exception], None] | None = None,
    use_sb: bool = True,
    sb_accuracy: bool = True,
    fallback_to_classic: bool = True,
) -> BoardObservation | None:
    """Detect inner corners; return ``None`` when the board is not found."""
    if image is None or not isinstance(image, np.ndarray) or image.ndim not in {2, 3}:
        return None
    if (
        type(pattern) is not tuple
        or len(pattern) != 2
        or any(type(value) is not int or value < 3 for value in pattern)
    ):
        raise ChessboardCalibrationError("棋盘格内角点网格必须是两个不小于 3 的整数")
    def _report(stage: str, error: Exception) -> None:
        if on_cv_error is not None:
            on_cv_error(stage, error)

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    except cv2.error as error:
        _report("cvt_color", error)
        return None
    height, width = gray.shape[:2]
    corners: np.ndarray | None = None
    detector = "unknown"
    if use_sb and hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        if sb_accuracy and hasattr(cv2, "CALIB_CB_ACCURACY"):
            flags |= cv2.CALIB_CB_ACCURACY
        # CALIB_CB_LARGER is intended for oversized/partial boards whose
        # corner identity is recovered from the optional metadata API.  The
        # wizard uses a fixed complete grid and the regular Python API; some
        # OpenCV builds throw an opaque native exception when LARGER sees a
        # partial board moving through a live frame.
        try:
            found, candidate = cv2.findChessboardCornersSB(gray, pattern, flags)
            if found:
                corners = candidate
                detector = "sb"
        except cv2.error as error:
            # The classic detector remains safe for calibration and keeps a
            # transient SB backend failure from taking down the live preview.
            _report("find_chessboard_corners_sb", error)
    if corners is None and (not use_sb or fallback_to_classic):
        try:
            found, candidate = cv2.findChessboardCorners(
                gray,
                pattern,
                cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_FILTER_QUADS
                | cv2.CALIB_CB_NORMALIZE_IMAGE,
            )
            if found:
                refined = cv2.cornerSubPix(
                    gray,
                    candidate,
                    (11, 11),
                    (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
                )
                corners = refined
                detector = "classic"
        except cv2.error as error:
            _report("find_chessboard_corners_classic", error)
            return None
    if corners is None:
        return None
    ordered = _canonicalize_corner_order(corners, pattern)
    centroid = ordered.mean(axis=0)
    zone = (
        min(2, max(0, int(centroid[0] / max(1, width) * 3))),
        min(2, max(0, int(centroid[1] / max(1, height) * 3))),
    )
    try:
        sharpness = _laplacian_sharpness(gray)
    except cv2.error as error:
        _report("laplacian_sharpness", error)
        return None
    return BoardObservation(
        corners_px=ordered,
        sharpness=sharpness,
        centroid_zone=zone,
        detector=detector,
    )


def align_pair_orientation(
    left: BoardObservation, right: BoardObservation, *, pattern: tuple[int, int]
) -> BoardObservation:
    """Match the right eye's grid labelling to the left eye's.

    Within one stereo pair the two views are nearly parallel, so the correct
    labelling puts the right eye's first corner at almost the same image
    position as the left eye's first corner; a 180-degree relabelling would be
    roughly a board-diagonal away.  This guards against the detector choosing
    different orientations for the two eyes of the same instant.
    """
    columns, rows = int(pattern[0]), int(pattern[1])
    flat = np.asarray(right.corners_px, dtype=np.float64).reshape(-1, 2)
    if len(flat) != columns * rows:
        raise ChessboardCalibrationError("棋盘格内角点数量与网格不一致")
    left_grid = np.asarray(left.corners_px, dtype=np.float64).reshape(rows, columns, 2)
    right_grid = flat.reshape(rows, columns, 2)
    flipped_grid = right_grid[::-1, ::-1]
    flipped = flipped_grid.reshape(-1, 2)

    def _basis(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        horizontal = np.median(grid[:, 1:] - grid[:, :-1], axis=(0, 1))
        vertical = np.median(grid[1:, :] - grid[:-1, :], axis=(0, 1))
        return horizontal / max(float(np.linalg.norm(horizontal)), 1e-12), vertical / max(
            float(np.linalg.norm(vertical)), 1e-12
        )

    left_horizontal, left_vertical = _basis(left_grid)

    def _score(candidate: np.ndarray) -> float:
        horizontal, vertical = _basis(candidate)
        return float(
            np.dot(left_horizontal, horizontal) + np.dot(left_vertical, vertical)
        )

    as_is_score = _score(right_grid)
    flipped_score = _score(flipped_grid)
    # Compare grid directions instead of absolute first-corner positions.
    # At close range, stereo disparity can exceed the projected board width;
    # the old nearest-first-corner rule then selected the 180-degree-wrong
    # correspondence and corrupted the whole stereo solve.
    if flipped_score > as_is_score:
        return BoardObservation(
            flipped.copy(), right.sharpness, right.centroid_zone, right.detector
        )
    return right


def pose_diversity_report(
    observations: list[BoardObservation], *, image_size: tuple[int, int]
) -> dict[str, Any]:
    """Summarise pose spread across accepted left-eye observations."""
    if not observations:
        return {"pair_count": 0, "unique_zones": 0, "centroid_spread_ratio": 0.0}
    centroids = np.asarray([item.centroid_px for item in observations], dtype=float)
    zones = {item.centroid_zone for item in observations}
    diagonal = float(math.hypot(image_size[0], image_size[1]))
    spread = float(np.std(centroids, axis=0).mean()) if len(observations) > 1 else 0.0
    return {
        "pair_count": len(observations),
        "unique_zones": len(zones),
        "centroid_spread_ratio": spread / diagonal if diagonal else 0.0,
        "zones": sorted(zones),
    }


def _board_shape_signature(
    observation: BoardObservation, pattern: tuple[int, int]
) -> np.ndarray:
    """Describe scale and projective shape without requiring camera intrinsics."""

    columns, rows = int(pattern[0]), int(pattern[1])
    grid = np.asarray(observation.corners_px, dtype=float).reshape(rows, columns, 2)
    horizontal_vectors = grid[:, 1:] - grid[:, :-1]
    vertical_vectors = grid[1:, :] - grid[:-1, :]
    horizontal_lengths = np.linalg.norm(horizontal_vectors, axis=2)
    vertical_lengths = np.linalg.norm(vertical_vectors, axis=2)
    horizontal = np.median(horizontal_vectors, axis=(0, 1))
    vertical = np.median(vertical_vectors, axis=(0, 1))
    horizontal_size = max(float(np.median(horizontal_lengths)), 1e-12)
    vertical_size = max(float(np.median(vertical_lengths)), 1e-12)
    cosine = float(
        np.dot(horizontal, vertical)
        / max(float(np.linalg.norm(horizontal) * np.linalg.norm(vertical)), 1e-12)
    )
    left_vertical = max(float(np.median(vertical_lengths[:, 0])), 1e-12)
    right_vertical = max(float(np.median(vertical_lengths[:, -1])), 1e-12)
    top_horizontal = max(float(np.median(horizontal_lengths[0, :])), 1e-12)
    bottom_horizontal = max(float(np.median(horizontal_lengths[-1, :])), 1e-12)
    return np.asarray(
        [
            math.sqrt(horizontal_size * vertical_size),
            math.log(horizontal_size / vertical_size),
            cosine,
            math.log(right_vertical / left_vertical),
            math.log(bottom_horizontal / top_horizontal),
        ],
        dtype=float,
    )


def projective_pose_report(
    observations: list[BoardObservation],
    *,
    image_size: tuple[int, int],
    pattern: tuple[int, int],
) -> dict[str, Any]:
    """Audit distance and tilt variation directly from detected image grids.

    This report intentionally does not use a recovered camera matrix.  A
    degenerate planar solve can invent large tilt/depth changes together with
    a huge focal length, which made the previous pose gate self-validating.
    """

    if not observations:
        return {
            "pose_count": 0,
            "projected_scale_span_ratio": 0.0,
            "projective_shape_span": 0.0,
        }
    signatures = np.stack(
        [_board_shape_signature(observation, pattern) for observation in observations]
    )
    scale = signatures[:, 0]
    median_scale = max(float(np.median(scale)), 1e-12)
    if len(observations) > 2:
        scale_low, scale_high = np.percentile(scale, [10, 90])
        shape_spans = np.percentile(signatures[:, 1:], 90, axis=0) - np.percentile(
            signatures[:, 1:], 10, axis=0
        )
    else:
        scale_low, scale_high = float(np.min(scale)), float(np.max(scale))
        shape_spans = np.ptp(signatures[:, 1:], axis=0)
    diagonal = max(float(math.hypot(*image_size)), 1e-12)
    return {
        "pose_count": len(observations),
        "projected_scale_min_px": float(np.min(scale)),
        "projected_scale_median_px": median_scale,
        "projected_scale_max_px": float(np.max(scale)),
        "projected_scale_span_ratio": float(scale_high - scale_low) / median_scale,
        "projected_scale_to_image_ratio": median_scale / diagonal,
        "projective_shape_span": float(np.max(np.abs(shape_spans))),
        "shape_component_spans": [float(value) for value in shape_spans],
    }


# ---------------------------------------------------------------------------
# Rectification recipe
# ---------------------------------------------------------------------------


def build_rectification_recipe(
    *,
    calibration_id: str,
    K1: np.ndarray,
    D1: np.ndarray,
    K2: np.ndarray,
    D2: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
    image_size: tuple[int, int],
    alpha: float = 0.0,
    right_frame_transform: str = "none",
    definition: str = "cv2.stereoRectify(alpha=0) raw->rectified remap",
) -> dict:
    recipe = {
        "calibration_id": calibration_id,
        "K1": np.asarray(K1, dtype=float).tolist(),
        "D1": np.asarray(D1, dtype=float).reshape(-1).tolist(),
        "K2": np.asarray(K2, dtype=float).tolist(),
        "D2": np.asarray(D2, dtype=float).reshape(-1).tolist(),
        "R1": np.asarray(R1, dtype=float).tolist(),
        "R2": np.asarray(R2, dtype=float).tolist(),
        "P1": np.asarray(P1, dtype=float).tolist(),
        "P2": np.asarray(P2, dtype=float).tolist(),
        "image_width_px": int(image_size[0]),
        "image_height_px": int(image_size[1]),
        "output_width_px": int(image_size[0]),
        "output_height_px": int(image_size[1]),
        "alpha": float(alpha),
        "right_frame_transform": right_frame_transform,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "definition": definition,
    }
    return validate_rectification_recipe(recipe)


class Rectifier:
    """Remap raw camera frames into the declared rectified images."""

    def __init__(self, recipe: Mapping[str, Any]) -> None:
        validated = validate_rectification_recipe(recipe)
        self.recipe = validated
        self.size = (validated["image_width_px"], validated["image_height_px"])
        self.right_frame_transform = validated["right_frame_transform"]
        self._maps: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._sources = {
            "left": (
                np.asarray(validated["K1"], dtype=np.float64),
                np.asarray(validated["D1"], dtype=np.float64),
                np.asarray(validated["R1"], dtype=np.float64),
                np.asarray(validated["P1"], dtype=np.float64),
            ),
            "right": (
                np.asarray(validated["K2"], dtype=np.float64),
                np.asarray(validated["D2"], dtype=np.float64),
                np.asarray(validated["R2"], dtype=np.float64),
                np.asarray(validated["P2"], dtype=np.float64),
            ),
        }

    def rectify(self, role: str, image: np.ndarray) -> np.ndarray:
        if role not in self._sources:
            raise ChessboardCalibrationError(f"未知的相机角色：{role!r}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ChessboardCalibrationError("待矫正图像必须是三通道 BGR 帧")
        if (image.shape[1], image.shape[0]) != self.size:
            raise ChessboardCalibrationError(
                f"{role} 原始帧尺寸 {image.shape[1]}×{image.shape[0]} 与"
                f"矫正配方要求 {self.size[0]}×{self.size[1]} 不一致"
            )
        if role == "right":
            from .stereo_camera import apply_frame_transform

            image = apply_frame_transform(image, self.right_frame_transform)
        if role not in self._maps:
            matrix, distortion, rotation, projection = self._sources[role]
            self._maps[role] = cv2.initUndistortRectifyMap(
                matrix,
                distortion,
                rotation,
                projection,
                self.size,
                cv2.CV_32FC1,
            )
        map_x, map_y = self._maps[role]
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)


def rectifier_for_calibration(
    calibration: Mapping[str, Any] | None,
    profile: Mapping[str, Any] | None,
) -> Rectifier | None:
    """Return the fail-closed raw→rectified remap for a wizard calibration.

    * External calibrations (no ``-chess-`` marker) keep today's behaviour:
      the input stream is assumed to be already rectified, so ``None``.
    * Wizard calibrations without a matching valid recipe raise, because
      silently using the rectified K on raw frames would corrupt every depth.
    """
    if not isinstance(calibration, dict):
        return None
    current_id = str(calibration.get("calibration_id", ""))
    recipe = (profile or {}).get("rectification_recipe")
    recipe_id = str((recipe or {}).get("calibration_id", "")) if isinstance(recipe, dict) else ""
    if recipe is not None and recipe_id and calibration_ids_match(current_id, recipe_id):
        from .stereo_analyzer import _calibration_from_manifest

        parsed = _calibration_from_manifest(calibration)
        validated = validate_rectification_recipe(recipe)
        if (validated["image_width_px"], validated["image_height_px"]) != (
            parsed.left.width,
            parsed.left.height,
        ):
            raise ChessboardCalibrationError(
                "极线矫正配方尺寸与当前标定不一致；请在工作台重新标定或重置配置。"
            )
        return Rectifier(validated)
    if CHESS_ID_MARKER in current_id:
        raise ChessboardCalibrationError(
            "当前标定来自棋盘格向导，但缺少匹配的极线矫正配方；"
            "相机输出的是原始帧，直接分析会得到错误深度。"
            "请在工作台重新标定，或重置工作台配置。"
        )
    return None


# ---------------------------------------------------------------------------
# Solve
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WizardResult:
    calibration: dict
    recipe: dict
    stereo_rms_px: float
    left_rms_px: float
    right_rms_px: float
    per_view_rms_px: list[float]
    pair_count: int
    validated: bool
    rejection_reasons: list[str]
    audit: dict


def _audit_hash(audit: Mapping[str, Any]) -> str:
    canonical = json.dumps(audit, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _object_points(pattern: tuple[int, int], square_mm: float) -> np.ndarray:
    columns, rows = pattern
    grid = np.zeros((columns * rows, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    return grid * square_mm


def _camera_dict(role: str, center_world_mm: list[float], size: tuple[int, int], intrinsic: np.ndarray) -> dict:
    return {
        "camera_id": f"chessboard-{role}",
        "width": int(size[0]),
        "height": int(size[1]),
        "fx": float(intrinsic[0, 0]),
        "fy": float(intrinsic[1, 1]),
        "cx": float(intrinsic[0, 2]),
        "cy": float(intrinsic[1, 2]),
        "distortion_coefficients": [0.0] * 5,
        "rotation_world_to_camera": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "center_world_mm": list(center_world_mm),
    }


def _validate_solve_inputs(
    pairs: list[tuple[BoardObservation, BoardObservation]],
    image_size: tuple[int, int],
    square_mm: float,
    pattern: tuple[int, int],
    max_reprojection_rms_px: float,
    min_pairs: int,
    max_sync_delta_ms: float,
) -> None:
    if (
        type(image_size) is not tuple
        or len(image_size) != 2
        or any(type(value) is not int or value <= 0 for value in image_size)
    ):
        raise ChessboardCalibrationError("图像尺寸必须是两个正整数")
    if (
        type(pattern) is not tuple
        or len(pattern) != 2
        or any(type(value) is not int or value < 3 for value in pattern)
    ):
        raise ChessboardCalibrationError("棋盘格内角点规格必须是两个不小于 3 的整数")
    if not math.isfinite(float(square_mm)) or not 0 < float(square_mm) <= 1000:
        raise ChessboardCalibrationError("棋盘格实测格边长必须在 0 到 1000 mm 之间")
    if (
        not isinstance(max_reprojection_rms_px, (int, float))
        or not math.isfinite(float(max_reprojection_rms_px))
        or float(max_reprojection_rms_px) <= 0
    ):
        raise ChessboardCalibrationError("允许 RMS 必须是正数")
    if type(min_pairs) is not int or min_pairs < 3:
        raise ChessboardCalibrationError("最少照片组数必须是不小于 3 的整数")
    if not math.isfinite(float(max_sync_delta_ms)) or max_sync_delta_ms < 0:
        raise ChessboardCalibrationError("max_sync_delta_ms 必须是非负有限数")
    columns, rows = pattern
    width, height = image_size
    detector_names: set[str] = set()
    for pair_index, pair in enumerate(pairs, start=1):
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ChessboardCalibrationError(f"第 {pair_index} 组不是左右角点对")
        for role, observation in zip(("左", "右"), pair):
            points = np.asarray(observation.corners_px, dtype=float)
            if points.shape != (columns * rows, 2) or not np.all(np.isfinite(points)):
                raise ChessboardCalibrationError(
                    f"第 {pair_index} 组{role}目角点数量、形状或有限性无效"
                )
            if (
                np.any(points[:, 0] < -1)
                or np.any(points[:, 0] > width)
                or np.any(points[:, 1] < -1)
                or np.any(points[:, 1] > height)
            ):
                raise ChessboardCalibrationError(f"第 {pair_index} 组{role}目角点超出图像")
            detector = str(getattr(observation, "detector", "unknown"))
            if detector != "unknown":
                detector_names.add(detector)
        left_detector = str(getattr(pair[0], "detector", "unknown"))
        right_detector = str(getattr(pair[1], "detector", "unknown"))
        if (
            "unknown" not in {left_detector, right_detector}
            and left_detector != right_detector
        ):
            raise ChessboardCalibrationError(
                f"第 {pair_index} 组左右目使用了不同角点检测器，必须重新采集"
            )
    if len(detector_names) > 1:
        raise ChessboardCalibrationError(
            "标定照片混用了 SB 与 classic 角点检测器；请使用同一检测器重新采集"
        )


def _intrinsic_problems(
    matrix: np.ndarray,
    distortion: np.ndarray,
    image_size: tuple[int, int],
    role: str,
) -> list[str]:
    """Return physical sanity failures for one raw camera solution."""

    problems: list[str] = []
    K = np.asarray(matrix, dtype=float)
    D = np.asarray(distortion, dtype=float).reshape(-1)
    width, height = image_size
    focal_min = _FOCAL_RANGE[0] * max(image_size)
    focal_max = _FOCAL_RANGE[1] * max(image_size)
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        return [f"{role}目原始内参矩阵无效"]
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    if not (focal_min <= fx <= focal_max and focal_min <= fy <= focal_max):
        problems.append(f"{role}目原始焦距超出合理范围：fx={fx:.1f}, fy={fy:.1f} px")
    if not (0 <= cx <= width and 0 <= cy <= height):
        problems.append(f"{role}目原始主点超出图像：cx={cx:.1f}, cy={cy:.1f} px")
    if not np.all(np.isfinite(D)) or np.any(np.abs(D) > _MAX_DISTORTION_ABS):
        largest = float(np.max(np.abs(D))) if D.size and np.all(np.isfinite(D)) else float("inf")
        problems.append(f"{role}目畸变系数异常：最大绝对值 {largest:.3g}")
    return problems


def _monocular_view_rms(
    object_points: np.ndarray,
    image_points: list[np.ndarray],
    matrix: np.ndarray,
    distortion: np.ndarray,
    rotation_vectors: Any,
    translation_vectors: Any,
) -> np.ndarray:
    """Return one geometric reprojection RMS for every monocular view."""

    errors: list[float] = []
    for points, rotation, translation in zip(
        image_points, rotation_vectors, translation_vectors
    ):
        projected = cv2.projectPoints(
            object_points, rotation, translation, matrix, distortion
        )[0].reshape(-1, 2)
        observed = np.asarray(points, dtype=float).reshape(-1, 2)
        residual = projected - observed
        errors.append(float(np.sqrt(np.mean(np.sum(residual * residual, axis=1)))))
    return np.asarray(errors, dtype=float)


def calibration_pose_report(
    object_points: np.ndarray,
    image_points: list[np.ndarray],
    matrix: np.ndarray,
    distortion: np.ndarray,
) -> dict[str, Any]:
    """Measure out-of-plane tilt and distance variation of accepted views."""

    tilt_x: list[float] = []
    tilt_y: list[float] = []
    depths: list[float] = []
    failed_pose_count = 0
    for points in image_points:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                points,
                matrix,
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                failed_pose_count += 1
                continue
            rotation = cv2.Rodrigues(rvec)[0]
            normal = np.asarray(rotation[:, 2], dtype=float)
            depth = float(np.asarray(tvec).reshape(3)[2])
            if not np.all(np.isfinite(normal)) or not math.isfinite(depth) or depth <= 0:
                failed_pose_count += 1
                continue
            if normal[2] < 0:
                normal = -normal
            tilt_x.append(math.degrees(math.atan2(float(normal[0]), float(normal[2]))))
            tilt_y.append(math.degrees(math.atan2(float(normal[1]), float(normal[2]))))
            depths.append(depth)
        except (cv2.error, ValueError, FloatingPointError):
            failed_pose_count += 1
    if not depths:
        return {
            "pose_count": 0,
            "failed_pose_count": failed_pose_count,
            "tilt_span_deg": 0.0,
            "depth_spread_ratio": 0.0,
        }
    x_span = float(np.percentile(tilt_x, 95) - np.percentile(tilt_x, 5))
    y_span = float(np.percentile(tilt_y, 95) - np.percentile(tilt_y, 5))
    depth_median = float(np.median(depths))
    depth_span = float(np.percentile(depths, 90) - np.percentile(depths, 10))
    return {
        "pose_count": len(depths),
        "failed_pose_count": failed_pose_count,
        "tilt_x_span_deg": x_span,
        "tilt_y_span_deg": y_span,
        "tilt_span_deg": max(x_span, y_span),
        "depth_min_mm": float(np.min(depths)),
        "depth_median_mm": depth_median,
        "depth_max_mm": float(np.max(depths)),
        "depth_spread_ratio": depth_span / max(abs(depth_median), 1e-9),
    }


def rectification_quality_report(
    left_points: list[np.ndarray],
    right_points: list[np.ndarray],
    K1: np.ndarray,
    D1: np.ndarray,
    K2: np.ndarray,
    D2: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    P1: np.ndarray,
    P2: np.ndarray,
) -> dict[str, Any]:
    """Audit the exact horizontal epipolar geometry consumed by metrology."""

    vertical_errors: list[np.ndarray] = []
    disparities: list[np.ndarray] = []
    per_pair_vertical_rms: list[float] = []
    for left, right in zip(left_points, right_points):
        left_rectified = cv2.undistortPoints(left, K1, D1, R=R1, P=P1).reshape(-1, 2)
        right_rectified = cv2.undistortPoints(right, K2, D2, R=R2, P=P2).reshape(-1, 2)
        vertical = left_rectified[:, 1] - right_rectified[:, 1]
        disparity = left_rectified[:, 0] - right_rectified[:, 0]
        vertical_errors.append(vertical)
        disparities.append(disparity)
        per_pair_vertical_rms.append(float(np.sqrt(np.mean(vertical * vertical))))
    vertical = np.concatenate(vertical_errors)
    disparity = np.concatenate(disparities)
    p1 = np.asarray(P1, dtype=float)
    p2 = np.asarray(P2, dtype=float)
    if (
        not np.all(np.isfinite(vertical))
        or not np.all(np.isfinite(disparity))
        or not np.all(np.isfinite(p1))
        or not np.all(np.isfinite(p2))
        or abs(float(p2[0, 0])) < 1e-12
        or abs(float(p2[1, 1])) < 1e-12
    ):
        raise ValueError("极线校正质量计算包含非有限数值或零焦距")
    horizontal_encoded_baseline = -float(p2[0, 3]) / float(p2[0, 0])
    vertical_projection_mm = -float(p2[1, 3]) / float(p2[1, 1])
    return {
        "vertical_rms_px": float(np.sqrt(np.mean(vertical * vertical))),
        "vertical_p95_abs_px": float(np.percentile(np.abs(vertical), 95)),
        "vertical_max_abs_px": float(np.max(np.abs(vertical))),
        "per_pair_vertical_rms_px": per_pair_vertical_rms,
        "median_disparity_px": float(np.median(disparity)),
        "positive_disparity_ratio": float(np.mean(disparity > 0)),
        "horizontal_encoded_baseline_mm": horizontal_encoded_baseline,
        "vertical_projection_mm": vertical_projection_mm,
        "p1_p2_k_identical": bool(np.allclose(p1[:, :3], p2[:, :3], atol=1e-6)),
    }


def transform_board_observation(
    observation: BoardObservation,
    *,
    transform: str,
    image_size: tuple[int, int],
    pattern: tuple[int, int],
) -> BoardObservation:
    """Map detected corners as if a sensor-orientation fix preceded detection.

    A reflected UVC sensor can still produce an excellent monocular result,
    while no proper rigid transform can join it to the other eye.  Coordinates
    and grid labels must both be transformed; changing only the labels would
    associate different physical squares between the two cameras.
    """

    from .stereo_camera import SUPPORTED_FRAME_TRANSFORMS

    if transform not in SUPPORTED_FRAME_TRANSFORMS:
        raise ChessboardCalibrationError(f"未知画面修正方式：{transform!r}")
    columns, rows = int(pattern[0]), int(pattern[1])
    grid = np.asarray(observation.corners_px, dtype=np.float64).reshape(
        rows, columns, 2
    ).copy()
    width, height = int(image_size[0]), int(image_size[1])
    if transform in {"flip_horizontal", "rotate_180"}:
        grid[..., 0] = width - 1 - grid[..., 0]
        grid = grid[:, ::-1]
    if transform in {"flip_vertical", "rotate_180"}:
        grid[..., 1] = height - 1 - grid[..., 1]
        grid = grid[::-1, :]
    flat = grid.reshape(-1, 2).copy()
    centroid = flat.mean(axis=0)
    zone = (
        min(2, max(0, int(centroid[0] / max(1, width) * 3))),
        min(2, max(0, int(centroid[1] / max(1, height) * 3))),
    )
    return BoardObservation(flat, observation.sharpness, zone, observation.detector)


def _baseline_anchored_pinhole_attempt(
    *,
    object_lists: list[np.ndarray],
    left_points: list[np.ndarray],
    right_points: list[np.ndarray],
    image_size: tuple[int, int],
    expected_baseline_mm: float,
) -> dict[str, Any]:
    """Resolve a weak planar data set with measured baseline and a stable K.

    Nearly front-facing chessboard sweeps have a focal-length/depth ambiguity.
    The unconstrained Brown model can reduce monocular RMS by inventing focal
    lengths, principal points and distortion.  A measured rig baseline removes
    that ambiguity: search a centred square-pixel pinhole model, keep all
    intrinsics fixed during each stereo solve, and select the focal length that
    reproduces the independently measured lens-centre distance.
    """

    width, height = image_size
    focal_min = max(1.0, _FOCAL_RANGE[0] * max(image_size))
    # The anchor path is only a fallback for ordinary UVC stereo modules.  A
    # very narrow-FOV root can reproduce the same measured baseline while
    # sending stereoRectify's principal point outside the image.  Keep this
    # fallback within roughly 37° horizontal FOV; specialised telephoto rigs
    # need supplied factory intrinsics instead of self-calibration.
    focal_max = min(
        _FOCAL_RANGE[1] * max(image_size), 1.5 * max(image_size)
    )
    mono_flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_FOCAL_LENGTH
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
    )
    mono_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        80,
        1e-7,
    )
    stereo_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        1e-7,
    )
    evaluated: dict[float, dict[str, Any]] = {}

    def evaluate(focal_px: float) -> dict[str, Any] | None:
        focal_px = float(np.clip(focal_px, focal_min, focal_max))
        key = round(focal_px, 6)
        if key in evaluated:
            return evaluated[key]
        intrinsic = np.asarray(
            [
                [focal_px, 0.0, (width - 1) / 2.0],
                [0.0, focal_px, (height - 1) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.zeros(5, dtype=np.float64)
        try:
            left_rms, K1, D1, _left_rv, _left_tv = cv2.calibrateCamera(
                object_lists,
                left_points,
                image_size,
                intrinsic.copy(),
                distortion.copy(),
                flags=mono_flags,
                criteria=mono_criteria,
            )
            right_rms, K2, D2, _right_rv, _right_tv = cv2.calibrateCamera(
                object_lists,
                right_points,
                image_size,
                intrinsic.copy(),
                distortion.copy(),
                flags=mono_flags,
                criteria=mono_criteria,
            )
            (
                stereo_rms,
                _stereo_k1,
                _stereo_d1,
                _stereo_k2,
                _stereo_d2,
                rotation,
                translation,
                _essential,
                _fundamental,
                per_view,
            ) = cv2.stereoCalibrate(
                object_lists,
                left_points,
                right_points,
                K1,
                D1,
                K2,
                D2,
                image_size,
                np.eye(3, dtype=np.float64),
                np.zeros((3, 1), dtype=np.float64),
                None,
                None,
                None,
                flags=cv2.CALIB_FIX_INTRINSIC,
                criteria=stereo_criteria,
            )
            R1, R2, P1, P2, _Q, _roi1, _roi2 = cv2.stereoRectify(
                K1,
                D1,
                K2,
                D2,
                image_size,
                rotation,
                translation,
                alpha=0.0,
                flags=cv2.CALIB_ZERO_DISPARITY,
            )
            geometry = rectification_quality_report(
                left_points,
                right_points,
                K1,
                D1,
                K2,
                D2,
                R1,
                R2,
                P1,
                P2,
            )
        except (cv2.error, ValueError, FloatingPointError):
            return None
        values = (K1, D1, K2, D2, rotation, translation, per_view)
        if not all(np.all(np.isfinite(value)) for value in values):
            return None
        baseline = float(np.linalg.norm(translation))
        rectified_intrinsic_problems = _intrinsic_problems(
            np.asarray(P1[:, :3], dtype=float),
            np.zeros(5, dtype=float),
            image_size,
            "矫正后",
        )
        attempt = {
            "model": "baseline_anchored_centered_pinhole",
            "focal_px": focal_px,
            "left_rms": float(left_rms),
            "right_rms": float(right_rms),
            "stereo_rms": float(stereo_rms),
            "K1": K1,
            "D1": D1,
            "K2": K2,
            "D2": D2,
            "R": rotation,
            "T": translation,
            "per_view": per_view,
            "baseline_mm": baseline,
            "baseline_relative_error": abs(baseline - expected_baseline_mm)
            / expected_baseline_mm,
            "rectified_intrinsic_problems": rectified_intrinsic_problems,
            "rectification_geometry": geometry,
        }
        evaluated[key] = attempt
        return attempt

    for focal in np.geomspace(focal_min, focal_max, num=9):
        evaluate(float(focal))
    if not evaluated:
        raise ChessboardCalibrationError("实测基线约束求解失败")
    for factor in (1.35, 1.12):
        closest = min(
            evaluated.values(), key=lambda item: item["baseline_relative_error"]
        )
        centre = float(closest["focal_px"])
        for focal in np.geomspace(centre / factor, centre * factor, num=5):
            evaluate(float(focal))
    return min(
        evaluated.values(),
        key=lambda item: (
            bool(item["rectified_intrinsic_problems"]),
            item["baseline_relative_error"],
            item["stereo_rms"],
            float(item["rectification_geometry"]["vertical_rms_px"]),
        ),
    )


def solve_stereo_calibration(
    *,
    pairs: list[tuple[BoardObservation, BoardObservation]],
    image_size: tuple[int, int],
    square_mm: float,
    pattern: tuple[int, int],
    max_reprojection_rms_px: float,
    min_pairs: int,
    operator: str,
    max_sync_delta_ms: float,
    layout: str,
    expected_baseline_mm: float | None = None,
    _allow_outlier_pruning: bool = True,
    _monocular_pruning_complete: bool = False,
    _input_pair_count: int | None = None,
    _discarded_pair_indices: list[int] | None = None,
    _pruning_audit: Mapping[str, Any] | None = None,
) -> WizardResult:
    """Calibrate from accepted pairs and build the rectified-rig contract."""
    if not isinstance(operator, str) or not _OPERATOR_PATTERN.match(operator):
        raise ChessboardCalibrationError("操作员代号只能包含字母、数字、点、下划线和连字符（≤32 字符）")
    if not pairs:
        raise ChessboardCalibrationError("没有可用的棋盘格照片对")
    input_pair_count = int(_input_pair_count or len(pairs))
    discarded_pair_indices = sorted(set(_discarded_pair_indices or []))
    current_original_indices = [
        index
        for index in range(1, input_pair_count + 1)
        if index not in set(discarded_pair_indices)
    ]
    if len(current_original_indices) != len(pairs):
        raise ChessboardCalibrationError("异常组索引与当前照片组数量不一致")

    def merge_pruning_audit(
        step: Mapping[str, Any], combined_indices: list[int]
    ) -> dict[str, Any]:
        previous = dict(_pruning_audit or {})
        if not previous:
            merged = dict(step)
        else:
            previous_steps = previous.get("steps")
            if isinstance(previous_steps, list):
                steps = [dict(item) for item in previous_steps]
            else:
                steps = [previous]
            steps.append(dict(step))
            merged = {"stage": "multi_stage", "steps": steps}
        merged["discarded_pair_indices"] = list(combined_indices)
        return merged

    _validate_solve_inputs(
        pairs,
        image_size,
        square_mm,
        pattern,
        max_reprojection_rms_px,
        min_pairs,
        max_sync_delta_ms,
    )
    if expected_baseline_mm is not None and (
        not math.isfinite(float(expected_baseline_mm))
        or not _BASELINE_RANGE_MM[0] <= float(expected_baseline_mm) <= _BASELINE_RANGE_MM[1]
    ):
        raise ChessboardCalibrationError(
            f"实测镜头中心距必须在 {_BASELINE_RANGE_MM[0]:g}–{_BASELINE_RANGE_MM[1]:g} mm 之间"
        )

    object_points = _object_points(pattern, float(square_mm))
    aligned = [
        (pair[0], align_pair_orientation(pair[0], pair[1], pattern=pattern))
        for pair in pairs
    ]
    left_points = [pair[0].corners_px.reshape(-1, 1, 2).astype(np.float32) for pair in aligned]
    object_lists = [object_points] * len(pairs)
    size = (int(image_size[0]), int(image_size[1]))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-6)

    left_rms, K1, D1, rvecs_l, tvecs_l = cv2.calibrateCamera(
        object_lists, left_points, size, None, None, flags=cv2.CALIB_FIX_K3, criteria=criteria
    )
    if not math.isfinite(float(left_rms)) or not all(
        np.all(np.isfinite(value)) for value in (K1, D1)
    ):
        raise ChessboardCalibrationError("左目单目标定返回了非有限数值")
    native_right_points = [
        pair[1].corners_px.reshape(-1, 1, 2).astype(np.float32)
        for pair in aligned
    ]
    (
        native_right_rms,
        native_K2,
        native_D2,
        native_rvecs_r,
        native_tvecs_r,
    ) = cv2.calibrateCamera(
        object_lists,
        native_right_points,
        size,
        None,
        None,
        flags=cv2.CALIB_FIX_K3,
        criteria=criteria,
    )
    if not math.isfinite(float(native_right_rms)) or not all(
        np.all(np.isfinite(value)) for value in (native_K2, native_D2)
    ):
        raise ChessboardCalibrationError("右目单目标定返回了非有限数值")
    left_mono_per_pair = _monocular_view_rms(
        object_points, left_points, K1, D1, rvecs_l, tvecs_l
    )
    right_mono_per_pair = _monocular_view_rms(
        object_points,
        native_right_points,
        native_K2,
        native_D2,
        native_rvecs_r,
        native_tvecs_r,
    )
    mono_pair_scores = np.maximum(left_mono_per_pair, right_mono_per_pair)
    quality_target = float(max_reprojection_rms_px)
    if (
        _allow_outlier_pruning
        and not _monocular_pruning_complete
        and len(pairs) > min_pairs
    ):
        median = float(np.median(mono_pair_scores))
        mad = float(np.median(np.abs(mono_pair_scores - median)))
        robust_limit = max(median + 3.5 * 1.4826 * mad, 0.5)
        candidates = [
            index
            for index, score in enumerate(mono_pair_scores)
            if score > robust_limit
        ]
        maximum_removals = min(
            len(pairs) - min_pairs,
            max(1, len(pairs) // 5),
            max(0, max(1, input_pair_count // 3) - len(discarded_pair_indices)),
        )
        if candidates and maximum_removals > 0:
            ranked = sorted(
                candidates, key=lambda index: mono_pair_scores[index], reverse=True
            )
            removed = sorted(ranked[:maximum_removals])
            retained = [pair for index, pair in enumerate(pairs) if index not in removed]
            new_original_indices = [current_original_indices[index] for index in removed]
            combined_indices = sorted(
                set(discarded_pair_indices + new_original_indices)
            )
            pruning_step = {
                "stage": "monocular_reprojection",
                "threshold_px": robust_limit,
                "left_per_pair_rms_px": left_mono_per_pair.tolist(),
                "right_per_pair_rms_px": right_mono_per_pair.tolist(),
                "discarded_pair_indices": new_original_indices,
            }
            return solve_stereo_calibration(
                pairs=retained,
                image_size=image_size,
                square_mm=square_mm,
                pattern=pattern,
                max_reprojection_rms_px=max_reprojection_rms_px,
                min_pairs=min_pairs,
                operator=operator,
                max_sync_delta_ms=max_sync_delta_ms,
                layout=layout,
                expected_baseline_mm=expected_baseline_mm,
                _allow_outlier_pruning=True,
                _monocular_pruning_complete=True,
                _input_pair_count=input_pair_count,
                _discarded_pair_indices=combined_indices,
                _pruning_audit=merge_pruning_audit(
                    pruning_step, combined_indices
                ),
            )
    transform_audit: list[dict[str, Any]] = []

    def _try_transform(transform: str) -> dict[str, Any]:
        transformed = [
            (
                pair[0],
                transform_board_observation(
                    pair[1], transform=transform, image_size=size, pattern=pattern
                ),
            )
            for pair in aligned
        ]
        candidate_right_points = [
            pair[1].corners_px.reshape(-1, 1, 2).astype(np.float32)
            for pair in transformed
        ]
        candidate_right_rms, candidate_k2, candidate_d2, _rv, _tv = cv2.calibrateCamera(
            object_lists,
            candidate_right_points,
            size,
            None,
            None,
            flags=cv2.CALIB_FIX_K3,
            criteria=criteria,
        )
        (
            candidate_stereo_rms,
            _candidate_k1,
            _candidate_d1,
            _candidate_k2,
            _candidate_d2,
            candidate_r,
            candidate_t,
            _candidate_e,
            _candidate_f,
            candidate_per_view,
        ) = cv2.stereoCalibrate(
            object_lists,
            left_points,
            candidate_right_points,
            K1,
            D1,
            candidate_k2,
            candidate_d2,
            size,
            np.eye(3, dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            None,
            None,
            None,
            flags=cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_FIX_K3,
            criteria=(
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                100,
                1e-7,
            ),
        )
        if (
            not math.isfinite(float(candidate_right_rms))
            or not math.isfinite(float(candidate_stereo_rms))
            or not all(
                np.all(np.isfinite(value))
                for value in (
                    candidate_k2,
                    candidate_d2,
                    candidate_r,
                    candidate_t,
                    candidate_per_view,
                )
            )
        ):
            raise ValueError("OpenCV 标定返回了非有限数值")
        candidate_baseline = float(np.linalg.norm(candidate_t))
        candidate_geometry: dict[str, Any]
        geometry_compatible = False
        try:
            c_r1, c_r2, c_p1, c_p2, _c_q, _c_roi1, _c_roi2 = cv2.stereoRectify(
                K1,
                D1,
                candidate_k2,
                candidate_d2,
                size,
                candidate_r,
                candidate_t,
                alpha=0.0,
                flags=cv2.CALIB_ZERO_DISPARITY,
            )
            candidate_geometry = rectification_quality_report(
                left_points,
                candidate_right_points,
                K1,
                D1,
                candidate_k2,
                candidate_d2,
                c_r1,
                c_r2,
                c_p1,
                c_p2,
            )
            encoded = float(candidate_geometry["horizontal_encoded_baseline_mm"])
            vertical = float(candidate_geometry["vertical_projection_mm"])
            geometry_compatible = bool(
                _BASELINE_RANGE_MM[0]
                <= candidate_baseline
                <= _BASELINE_RANGE_MM[1]
                and encoded > 0
                and math.isclose(
                    encoded, candidate_baseline, rel_tol=1e-4, abs_tol=1e-3
                )
                and abs(vertical) <= max(1e-3, candidate_baseline * 1e-4)
                and float(candidate_geometry["median_disparity_px"]) > 0
                and float(candidate_geometry["positive_disparity_ratio"]) >= 0.98
            )
        except (cv2.error, ValueError, FloatingPointError) as error:
            candidate_geometry = {
                "error": f"{type(error).__name__}: {error}"
            }
        return {
            "transform": transform,
            "right_rms": float(candidate_right_rms),
            "stereo_rms": float(candidate_stereo_rms),
            "K2": candidate_k2,
            "D2": candidate_d2,
            "R": candidate_r,
            "T": candidate_t,
            "baseline_mm": candidate_baseline,
            "per_view": candidate_per_view,
            "right_points": candidate_right_points,
            "geometry_compatible": geometry_compatible,
            "rectification_geometry": candidate_geometry,
            "per_pair_rms": np.max(
                np.asarray(candidate_per_view, dtype=float).reshape(-1, 2), axis=1
            ),
        }

    attempts: list[dict[str, Any]] = []

    def _evaluate(transform: str) -> None:
        try:
            attempt = _try_transform(transform)
            attempts.append(attempt)
            transform_audit.append(
                {
                    "transform": transform,
                    "right_rms_px": attempt["right_rms"],
                    "fixed_intrinsic_stereo_rms_px": attempt["stereo_rms"],
                    "baseline_mm": attempt["baseline_mm"],
                    "median_pair_rms_px": float(np.median(attempt["per_pair_rms"])),
                    "geometry_compatible": attempt["geometry_compatible"],
                    "rectification_geometry": attempt["rectification_geometry"],
                }
            )
        except (cv2.error, ValueError) as error:
            transform_audit.append(
                {"transform": transform, "error": f"{type(error).__name__}: {error}"}
            )

    _evaluate("none")
    if not attempts:
        raise ChessboardCalibrationError("双目标定初始求解失败")
    # Sensor reflection is only considered for the characteristic gross
    # stereo mismatch.  A healthy rig should keep its native orientation even
    # if another candidate happens to improve a noisy solve by a tiny amount.
    quality_target = float(max_reprojection_rms_px)
    gross_mismatch_px = max(2.0, quality_target * 4.0)
    if attempts[0]["stereo_rms"] > gross_mismatch_px:
        _evaluate("flip_horizontal")
        _evaluate("flip_vertical")
        _evaluate("rotate_180")
    best = min(
        attempts,
        key=lambda item: (
            not item["geometry_compatible"],
            float(np.median(item["per_pair_rms"])),
            item["stereo_rms"],
        ),
    )
    native = attempts[0]
    native_score = float(np.median(native["per_pair_rms"]))
    best_score = float(np.median(best["per_pair_rms"]))
    if (
        native["geometry_compatible"] == best["geometry_compatible"]
        and native_score <= best_score * 1.05 + 0.02
    ):
        best = native
    right_frame_transform = str(best["transform"])
    right_points = best["right_points"]
    right_rms = float(best["right_rms"])
    K2, D2 = best["K2"], best["D2"]
    stereo_rms = float(best["stereo_rms"])
    fixed_intrinsic_stereo_rms = stereo_rms
    R, T, per_view = best["R"], best["T"], best["per_view"]
    solve_mode = "fix_intrinsics"
    selected_per_pair_rms = np.asarray(best["per_pair_rms"], dtype=float)
    intrinsic_model_audit: list[dict[str, Any]] = [
        {
            "model": "free_brown_k1_k2_tangential",
            "selected_transform": right_frame_transform,
            "left_rms_px": float(left_rms),
            "right_rms_px": float(right_rms),
            "stereo_rms_px": float(stereo_rms),
            "baseline_mm": float(np.linalg.norm(T)),
            "intrinsic_problems": _intrinsic_problems(K1, D1, size, "左")
            + _intrinsic_problems(K2, D2, size, "右"),
            "rectification_geometry": best["rectification_geometry"],
        }
    ]
    full_model_problems = intrinsic_model_audit[0]["intrinsic_problems"]
    full_vertical_rms = float(
        best["rectification_geometry"].get("vertical_rms_px", float("inf"))
    )
    full_model_unstable = bool(
        full_model_problems
        or stereo_rms > max(2.0, quality_target * 4.0)
        or full_vertical_rms > max(2.0, quality_target * 4.0)
    )
    if expected_baseline_mm is not None and full_model_unstable:
        try:
            anchored = _baseline_anchored_pinhole_attempt(
                object_lists=object_lists,
                left_points=left_points,
                right_points=right_points,
                image_size=size,
                expected_baseline_mm=float(expected_baseline_mm),
            )
            intrinsic_model_audit.append(
                {
                    "model": anchored["model"],
                    "focal_px": anchored["focal_px"],
                    "left_rms_px": anchored["left_rms"],
                    "right_rms_px": anchored["right_rms"],
                    "stereo_rms_px": anchored["stereo_rms"],
                    "baseline_mm": anchored["baseline_mm"],
                    "baseline_relative_error": anchored[
                        "baseline_relative_error"
                    ],
                    "rectified_intrinsic_problems": anchored[
                        "rectified_intrinsic_problems"
                    ],
                    "rectification_geometry": anchored[
                        "rectification_geometry"
                    ],
                }
            )
            anchor_geometry = anchored["rectification_geometry"]
            anchor_is_usable = bool(
                anchored["baseline_relative_error"] <= 0.15
                and float(anchor_geometry["horizontal_encoded_baseline_mm"]) > 0
                and float(anchor_geometry["median_disparity_px"]) > 0
                and float(anchor_geometry["positive_disparity_ratio"]) >= 0.98
                and not _intrinsic_problems(
                    anchored["K1"], anchored["D1"], size, "左"
                )
                and not _intrinsic_problems(
                    anchored["K2"], anchored["D2"], size, "右"
                )
                and not anchored["rectified_intrinsic_problems"]
            )
            if anchor_is_usable:
                K1, D1 = anchored["K1"], anchored["D1"]
                K2, D2 = anchored["K2"], anchored["D2"]
                left_rms = float(anchored["left_rms"])
                right_rms = float(anchored["right_rms"])
                stereo_rms = float(anchored["stereo_rms"])
                fixed_intrinsic_stereo_rms = stereo_rms
                R, T, per_view = anchored["R"], anchored["T"], anchored["per_view"]
                selected_per_pair_rms = np.max(
                    np.asarray(per_view, dtype=float).reshape(-1, 2), axis=1
                )
                solve_mode = "baseline_anchored_centered_pinhole"
        except (ChessboardCalibrationError, cv2.error, ValueError) as error:
            intrinsic_model_audit.append(
                {
                    "model": "baseline_anchored_centered_pinhole",
                    "error": f"{type(error).__name__}: {error}",
                }
            )

    if _allow_outlier_pruning and len(pairs) > min_pairs:
        scores = selected_per_pair_rms
        median = float(np.median(scores))
        mad = float(np.median(np.abs(scores - median)))
        robust_limit = max(
            median + 3.5 * 1.4826 * mad,
            quality_target * 2.0,
        )
        candidates = [
            index
            for index, score in enumerate(scores)
            if score > robust_limit and score > quality_target * 2.0
        ]
        maximum_removals = min(
            len(pairs) - min_pairs,
            max(1, len(pairs) // 5),
            max(0, max(1, input_pair_count // 3) - len(discarded_pair_indices)),
        )
        if candidates and maximum_removals > 0:
            ranked = sorted(candidates, key=lambda index: scores[index], reverse=True)
            removed = sorted(ranked[:maximum_removals])
            retained = [pair for index, pair in enumerate(pairs) if index not in removed]
            new_original_indices = [current_original_indices[index] for index in removed]
            combined_indices = sorted(
                set(discarded_pair_indices + new_original_indices)
            )
            pruning_step = {
                "stage": "stereo_reprojection",
                "threshold_px": robust_limit,
                "per_pair_rms_px": scores.tolist(),
                "discarded_pair_indices": new_original_indices,
            }
            return solve_stereo_calibration(
                pairs=retained,
                image_size=image_size,
                square_mm=square_mm,
                pattern=pattern,
                max_reprojection_rms_px=max_reprojection_rms_px,
                min_pairs=min_pairs,
                operator=operator,
                max_sync_delta_ms=max_sync_delta_ms,
                layout=layout,
                expected_baseline_mm=expected_baseline_mm,
                _allow_outlier_pruning=True,
                _monocular_pruning_complete=True,
                _input_pair_count=input_pair_count,
                _discarded_pair_indices=combined_indices,
                _pruning_audit=merge_pruning_audit(
                    pruning_step, combined_indices
                ),
            )

    R1, R2, P1, P2, _Q, _roi1, _roi2 = cv2.stereoRectify(
        K1,
        D1,
        K2,
        D2,
        size,
        R,
        T,
        alpha=0.0,
        # CALIB_ZERO_DISPARITY forces identical principal points, hence an
        # identical rectified K in P1/P2 — required by the calibration contract.
        flags=cv2.CALIB_ZERO_DISPARITY,
    )
    if not all(np.all(np.isfinite(value)) for value in (R1, R2, P1, P2)):
        raise ChessboardCalibrationError("极线校正返回了非有限数值")
    rectified_intrinsic = np.asarray(P1[:, :3], dtype=np.float64)
    baseline = float(np.linalg.norm(np.asarray(T, dtype=np.float64)))
    pose_geometry = calibration_pose_report(object_points, left_points, K1, D1)
    image_pose_geometry = projective_pose_report(
        [pair[0] for pair in pairs], image_size=size, pattern=pattern
    )
    rectification_geometry = rectification_quality_report(
        left_points,
        right_points,
        K1,
        D1,
        K2,
        D2,
        R1,
        R2,
        P1,
        P2,
    )
    k_identical = bool(rectification_geometry["p1_p2_k_identical"])

    per_view_values = np.asarray(per_view, dtype=float).reshape(-1)
    audit = {
        "mode": "chessboard_stereo_wizard",
        "solve_mode": solve_mode,
        "pattern_inner_corners": list(pattern),
        "square_mm": float(square_mm),
        "pair_count": len(pairs),
        "input_pair_count": input_pair_count,
        "discarded_pair_indices": discarded_pair_indices,
        "image_size_px": [size[0], size[1]],
        "stereo_rms_px": float(stereo_rms),
        "fixed_intrinsic_stereo_rms_px": float(fixed_intrinsic_stereo_rms),
        "left_rms_px": float(left_rms),
        "right_rms_px": float(right_rms),
        "per_view_rms_px": [float(value) for value in per_view_values],
        "per_pair_rms_px": [
            [float(value) for value in row]
            for row in per_view_values.reshape(-1, 2)
        ],
        "right_frame_transform": right_frame_transform,
        "right_frame_transform_candidates": transform_audit,
        "intrinsic_model_candidates": intrinsic_model_audit,
        "outlier_pruning": dict(_pruning_audit or {}),
        "baseline_mm": baseline,
        "expected_baseline_mm": (
            None if expected_baseline_mm is None else float(expected_baseline_mm)
        ),
        "calibration_pose_geometry": pose_geometry,
        "image_pose_geometry": image_pose_geometry,
        "rectification_geometry": rectification_geometry,
        "p1_p2_k_identical": k_identical,
        "rectify_alpha": 0.0,
        "diversity": pose_diversity_report([pair[0] for pair in pairs], image_size=size),
        "layout": layout,
        "max_sync_delta_ms": float(max_sync_delta_ms),
        "max_reprojection_rms_px": float(max_reprojection_rms_px),
        "minimum_pairs": int(min_pairs),
        "opencv_version": cv2.__version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    calibration_id = f"{operator}{CHESS_ID_MARKER}{_audit_hash(audit)}"
    audit["calibration_id"] = calibration_id

    left_camera = _camera_dict("left", [0.0, 0.0, 0.0], size, rectified_intrinsic)
    right_camera = _camera_dict("right", [baseline, 0.0, 0.0], size, rectified_intrinsic)
    calibration = {
        "calibration_id": calibration_id,
        "validated": False,
        "registration_validated": False,
        "rectified": True,
        "baseline_mm": baseline,
        "max_sync_delta_ms": float(max_sync_delta_ms),
        "left_camera": left_camera,
        "right_camera": right_camera,
    }

    reasons: list[str] = []
    if len(pairs) < min_pairs:
        reasons.append(f"照片对不足：需要至少 {min_pairs} 对，当前 {len(pairs)} 对")
    diversity = audit["diversity"]
    if diversity["unique_zones"] < min(6, len(pairs)):
        reasons.append(
            f"棋盘格姿态太单一：至少需要 6 个不同画面区域，当前 {diversity['unique_zones']} 个"
        )
    if diversity["centroid_spread_ratio"] < _MIN_CENTROID_SPREAD_RATIO:
        reasons.append(
            f"棋盘格位置变化不足：质心散布 {diversity['centroid_spread_ratio']:.2f}"
            f"（要求 ≥ {_MIN_CENTROID_SPREAD_RATIO:.2f}），请覆盖画面四角和中心"
        )
    if (
        image_pose_geometry["projected_scale_span_ratio"]
        < _MIN_PROJECTED_SCALE_SPAN_RATIO
    ):
        reasons.append(
            "棋盘格远近变化不足：图像尺度跨度 "
            f"{image_pose_geometry['projected_scale_span_ratio']:.2f}"
            f"（要求 ≥ {_MIN_PROJECTED_SCALE_SPAN_RATIO:.2f}），请在近、中、远距离各拍几组"
        )
    if image_pose_geometry["projective_shape_span"] < _MIN_PROJECTIVE_SHAPE_SPAN:
        reasons.append(
            "棋盘格倾斜变化不足：投影形状跨度 "
            f"{image_pose_geometry['projective_shape_span']:.2f}"
            f"（要求 ≥ {_MIN_PROJECTIVE_SHAPE_SPAN:.2f}），请分别前后和左右倾斜硬质平板"
        )
    if (
        expected_baseline_mm is None
        and full_model_unstable
        and solve_mode == "fix_intrinsics"
    ):
        reasons.append(
            "自由内参求解不稳定且未填写实测镜头中心距；请量取两只镜头光学中心距离并填写，"
            "该值用于消除近似正对棋盘时的焦距/距离歧义"
        )
    if pose_geometry["pose_count"] != len(pairs):
        reasons.append(
            f"有 {pose_geometry['failed_pose_count']} 组棋盘格无法恢复有效正深度姿态"
        )
    if pose_geometry["tilt_span_deg"] < _MIN_TILT_SPAN_DEG:
        reasons.append(
            f"棋盘格前后/左右倾斜变化不足：当前 {pose_geometry['tilt_span_deg']:.1f}°，"
            f"要求至少 {_MIN_TILT_SPAN_DEG:g}°"
        )
    if pose_geometry["depth_spread_ratio"] < _MIN_DEPTH_SPREAD_RATIO:
        reasons.append(
            f"棋盘格远近变化不足：当前 {pose_geometry['depth_spread_ratio']:.2f}，"
            f"要求至少 {_MIN_DEPTH_SPREAD_RATIO:.2f}"
        )
    worst = max(float(left_rms), float(right_rms), float(stereo_rms))
    if worst > max_reprojection_rms_px:
        reasons.append(
            f"重投影误差过大：左右目 {float(left_rms):.3f}/{float(right_rms):.3f} px，"
            f"双目 {float(stereo_rms):.3f} px，上限 {max_reprojection_rms_px:g} px"
        )
    focal_bound = _FOCAL_RANGE[1] * max(size)
    fx, fy = float(rectified_intrinsic[0, 0]), float(rectified_intrinsic[1, 1])
    cx, cy = float(rectified_intrinsic[0, 2]), float(rectified_intrinsic[1, 2])
    if not (
        _FOCAL_RANGE[0] * max(size) <= fx <= focal_bound
        and _FOCAL_RANGE[0] * max(size) <= fy <= focal_bound
    ):
        reasons.append(f"矫正后焦距超出合理范围：fx={fx:.1f}, fy={fy:.1f} px")
    if not (0 <= cx <= size[0] and 0 <= cy <= size[1]):
        reasons.append(f"矫正后主点超出图像：cx={cx:.1f}, cy={cy:.1f} px")
    reasons.extend(_intrinsic_problems(K1, D1, size, "左"))
    reasons.extend(_intrinsic_problems(K2, D2, size, "右"))
    if not _BASELINE_RANGE_MM[0] <= baseline <= _BASELINE_RANGE_MM[1]:
        reasons.append(f"基线 {baseline:.2f} mm 超出合理范围 {_BASELINE_RANGE_MM[0]:g}–{_BASELINE_RANGE_MM[1]:g} mm")
    if expected_baseline_mm is not None:
        baseline_tolerance = max(3.0, float(expected_baseline_mm) * 0.15)
        if abs(baseline - float(expected_baseline_mm)) > baseline_tolerance:
            reasons.append(
                f"求得基线 {baseline:.2f} mm 与实测镜头中心距 {float(expected_baseline_mm):.2f} mm "
                f"相差超过 {baseline_tolerance:.2f} mm"
            )
    if not k_identical:
        reasons.append("stereoRectify 后左右投影矩阵 K 不一致，无法满足极线校正契约")
    vertical_rms = float(rectification_geometry["vertical_rms_px"])
    vertical_p95 = float(rectification_geometry["vertical_p95_abs_px"])
    if vertical_rms > float(max_reprojection_rms_px) or vertical_p95 > max(
        1.0, float(max_reprojection_rms_px) * 2.0
    ):
        reasons.append(
            f"极线校正后垂直残差过大：RMS {vertical_rms:.3f} px，"
            f"P95 {vertical_p95:.3f} px"
        )
    encoded_baseline = float(rectification_geometry["horizontal_encoded_baseline_mm"])
    vertical_projection = float(rectification_geometry["vertical_projection_mm"])
    if encoded_baseline <= 0 or not math.isclose(
        encoded_baseline, baseline, rel_tol=1e-4, abs_tol=1e-3
    ):
        reasons.append(
            f"矫正投影矩阵的水平基线 {encoded_baseline:.3f} mm 与求解基线 {baseline:.3f} mm 不一致"
        )
    if abs(vertical_projection) > max(1e-3, baseline * 1e-4):
        reasons.append(
            f"矫正结果不是水平双目：垂直投影基线为 {vertical_projection:.3f} mm"
        )
    if (
        float(rectification_geometry["median_disparity_px"]) <= 0
        or float(rectification_geometry["positive_disparity_ratio"]) < 0.98
    ):
        reasons.append(
            "矫正后视差方向与深度公式不一致："
            f"中位视差 {float(rectification_geometry['median_disparity_px']):.3f} px，"
            f"正视差比例 {float(rectification_geometry['positive_disparity_ratio']):.1%}"
        )

    from .capture_gui import field_calibration_problem
    from .stereo_analyzer import _calibration_from_manifest

    # ``validated`` is decided by these gates, so evaluate the contract and
    # the field gate against the tentatively-valid copy, not the False flag.
    calibration["validated"] = True
    try:
        _calibration_from_manifest(calibration)
    except ValueError as error:
        reasons.append(f"标定契约校验失败：{error}")
    field_problem = field_calibration_problem(calibration)
    if field_problem:
        reasons.append(field_problem)

    try:
        recipe = build_rectification_recipe(
            calibration_id=calibration_id,
            K1=K1,
            D1=D1,
            K2=K2,
            D2=D2,
            R1=R1,
            R2=R2,
            P1=P1,
            P2=P2,
            image_size=size,
            alpha=0.0,
            right_frame_transform=right_frame_transform,
        )
    except ValueError as error:
        recipe = {}
        reasons.append(f"极线矫正配方校验失败：{error}")
    validated = not reasons
    calibration["validated"] = validated
    return WizardResult(
        calibration=calibration,
        recipe=recipe,
        stereo_rms_px=float(stereo_rms),
        left_rms_px=float(left_rms),
        right_rms_px=float(right_rms),
        per_view_rms_px=audit["per_view_rms_px"],
        pair_count=len(pairs),
        validated=validated,
        rejection_reasons=reasons,
        audit=audit,
    )


def write_calibration_diagnostic(
    pairs: list[tuple[BoardObservation, BoardObservation]],
    result: WizardResult | None,
    *,
    image_size: tuple[int, int],
    pattern: tuple[int, int],
    square_mm: float,
    root: str | Path = "outputs/measurement_workbench/calibration_diagnostics",
    solve_error: str = "",
    filename: str | None = None,
    metadata_extra: Mapping[str, Any] | None = None,
    solve_options: Mapping[str, Any] | None = None,
) -> Path:
    """Persist all accepted corner coordinates so a failed solve is replayable."""

    if not pairs:
        raise ValueError("没有可写入诊断文件的角点对")
    destination = Path(root).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if filename is not None:
        if Path(filename).name != filename or not filename.endswith(".npz"):
            raise ValueError("标定诊断文件名必须是当前目录下的 .npz 文件")
        path = destination / filename
    else:
        path = destination / f"calibration_{datetime.now():%Y%m%d_%H%M%S_%f}.npz"
    metadata = {
        "image_size_px": [int(image_size[0]), int(image_size[1])],
        "pattern_inner_corners": [int(pattern[0]), int(pattern[1])],
        "square_mm": float(square_mm),
        "result_audit": result.audit if result is not None else {},
        "rejection_reasons": result.rejection_reasons if result is not None else [],
        "solve_error": str(solve_error),
    }
    if solve_options is not None:
        metadata["solve_options"] = json.loads(
            json.dumps(dict(solve_options), ensure_ascii=False, allow_nan=False)
        )
    if metadata_extra:
        metadata["capture_archive"] = json.loads(
            json.dumps(
                dict(metadata_extra), ensure_ascii=False, allow_nan=False
            )
        )
    np.savez_compressed(
        path,
        left_points=np.stack([pair[0].corners_px for pair in pairs]),
        right_points=np.stack([pair[1].corners_px for pair in pairs]),
        left_sharpness=np.asarray([pair[0].sharpness for pair in pairs]),
        right_sharpness=np.asarray([pair[1].sharpness for pair in pairs]),
        left_zones=np.asarray([pair[0].centroid_zone for pair in pairs], dtype=np.int8),
        right_zones=np.asarray([pair[1].centroid_zone for pair in pairs], dtype=np.int8),
        left_detectors=np.asarray([pair[0].detector for pair in pairs]),
        right_detectors=np.asarray([pair[1].detector for pair in pairs]),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    return path


def write_calibration_checkpoint(
    pairs: list[tuple[BoardObservation, BoardObservation]],
    *,
    image_size: tuple[int, int],
    pattern: tuple[int, int],
    square_mm: float,
    root: str | Path = "outputs/measurement_workbench/calibration_diagnostics",
    metadata_extra: Mapping[str, Any] | None = None,
    solve_options: Mapping[str, Any] | None = None,
) -> Path:
    """Overwrite the session checkpoint so accepted photos survive a restart."""

    return write_calibration_diagnostic(
        pairs,
        None,
        image_size=image_size,
        pattern=pattern,
        square_mm=square_mm,
        root=root,
        filename="calibration_autosave.npz",
        metadata_extra=metadata_extra,
        solve_options=solve_options,
    )


def read_calibration_diagnostic(
    path: str | Path,
) -> tuple[list[tuple[BoardObservation, BoardObservation]], dict[str, Any]]:
    """Load validated corner pairs and metadata from a diagnostic checkpoint."""

    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as payload:
            left_points = np.asarray(payload["left_points"], dtype=float)
            right_points = np.asarray(payload["right_points"], dtype=float)
            left_sharpness = np.asarray(payload["left_sharpness"], dtype=float)
            right_sharpness = np.asarray(payload["right_sharpness"], dtype=float)
            left_zones = np.asarray(payload["left_zones"], dtype=int)
            right_zones = np.asarray(payload["right_zones"], dtype=int)
            left_detectors = (
                np.asarray(payload["left_detectors"], dtype=str)
                if "left_detectors" in payload.files
                else np.full(left_sharpness.shape, "unknown")
            )
            right_detectors = (
                np.asarray(payload["right_detectors"], dtype=str)
                if "right_detectors" in payload.files
                else np.full(right_sharpness.shape, "unknown")
            )
            metadata = json.loads(str(payload["metadata_json"]))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as error:
        raise ChessboardCalibrationError(f"无法读取标定诊断文件：{error}") from error
    count = int(left_points.shape[0]) if left_points.ndim == 3 else 0
    expected_shapes = {
        "right_points": right_points.shape == left_points.shape,
        "left_sharpness": left_sharpness.shape == (count,),
        "right_sharpness": right_sharpness.shape == (count,),
        "left_zones": left_zones.shape == (count, 2),
        "right_zones": right_zones.shape == (count, 2),
        "left_detectors": left_detectors.shape == (count,),
        "right_detectors": right_detectors.shape == (count,),
    }
    if count == 0 or left_points.shape[-1:] != (2,) or not all(expected_shapes.values()):
        raise ChessboardCalibrationError(
            f"标定诊断数组形状不一致：left_points={left_points.shape}"
        )
    if not isinstance(metadata, dict):
        raise ChessboardCalibrationError("标定诊断 metadata_json 必须是对象")
    try:
        image_size = tuple(int(value) for value in metadata["image_size_px"])
        pattern = tuple(int(value) for value in metadata["pattern_inner_corners"])
        square_mm = float(metadata["square_mm"])
    except (KeyError, TypeError, ValueError) as error:
        raise ChessboardCalibrationError(f"标定诊断元数据不完整：{error}") from error
    if (
        len(image_size) != 2
        or min(image_size) <= 0
        or len(pattern) != 2
        or min(pattern) <= 0
        or left_points.shape[1] != pattern[0] * pattern[1]
        or not math.isfinite(square_mm)
        or square_mm <= 0
    ):
        raise ChessboardCalibrationError("标定诊断中的图像尺寸、棋盘规格或格边长无效")
    pairs = [
        (
            BoardObservation(
                left_points[index].copy(),
                float(left_sharpness[index]),
                tuple(int(value) for value in left_zones[index]),
                str(left_detectors[index]),
            ),
            BoardObservation(
                right_points[index].copy(),
                float(right_sharpness[index]),
                tuple(int(value) for value in right_zones[index]),
                str(right_detectors[index]),
            ),
        )
        for index in range(count)
    ]
    return pairs, metadata


def reusable_calibration_pairs(
    path: str | Path,
) -> tuple[
    list[tuple[BoardObservation, BoardObservation]],
    dict[str, Any],
    list[int],
]:
    """Load a checkpoint while omitting views already rejected by its solve."""

    pairs, metadata = read_calibration_diagnostic(path)
    audit = metadata.get("result_audit", {})
    if not isinstance(audit, Mapping):
        audit = {}
    discarded = sorted(
        {
            int(value)
            for value in audit.get("discarded_pair_indices", [])
            if type(value) in {int, float} and float(value).is_integer()
        }
    )
    if any(index < 1 or index > len(pairs) for index in discarded):
        raise ChessboardCalibrationError("标定诊断中的异常组号超出照片组范围")
    discarded_set = set(discarded)
    retained = [
        pair
        for index, pair in enumerate(pairs, start=1)
        if index not in discarded_set
    ]
    if not retained:
        raise ChessboardCalibrationError("标定诊断没有可继续使用的照片组")
    return retained, metadata, discarded


def _select_recent_recoverable_diagnostic(
    candidates: list[Path], *, minimum_pairs: int
) -> tuple[Path, list[dict[str, Any]]]:
    """Prefer the newest set that is large enough to run calibration."""

    inventory: list[dict[str, Any]] = []
    for source in candidates:
        try:
            retained, _metadata, discarded = reusable_calibration_pairs(source)
            inventory.append(
                {
                    "path": source,
                    "reusable_pair_count": len(retained),
                    "discarded_pair_indices": list(discarded),
                    "modified_time_ns": source.stat().st_mtime_ns,
                }
            )
        except (OSError, ValueError, ChessboardCalibrationError) as error:
            inventory.append(
                {
                    "path": source,
                    "error": f"{type(error).__name__}: {error}",
                    "modified_time_ns": (
                        source.stat().st_mtime_ns if source.exists() else 0
                    ),
                }
            )
    readable = [item for item in inventory if "reusable_pair_count" in item]
    if not readable:
        raise ChessboardCalibrationError("没有可读取的标定照片组")
    solvable = [
        item
        for item in readable
        if int(item["reusable_pair_count"]) >= int(minimum_pairs)
    ]
    chosen = max(
        solvable or readable,
        key=lambda item: int(item["modified_time_ns"]),
    )
    return Path(chosen["path"]), inventory


def _diagnostic_solve_options(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Prefer the completed solve audit, with checkpoint settings as fallback."""
    options = metadata.get("solve_options", {})
    audit = metadata.get("result_audit", {})
    return {
        **(dict(options) if isinstance(options, Mapping) else {}),
        **(dict(audit) if isinstance(audit, Mapping) else {}),
    }


def replay_calibration_diagnostic(
    path: str | Path,
    *,
    expected_baseline_mm: float | None = None,
    max_reprojection_rms_px: float | None = None,
) -> WizardResult:
    """Re-run a saved calibration failure without reopening the camera.

    ``expected_baseline_mm`` can supply a newly measured rig baseline when the
    original capture was saved before the operator entered that value.
    """

    pairs, metadata = read_calibration_diagnostic(path)
    audit = _diagnostic_solve_options(metadata)
    image_size = tuple(int(value) for value in metadata["image_size_px"])
    pattern = tuple(int(value) for value in metadata["pattern_inner_corners"])
    baseline = (
        audit.get("expected_baseline_mm")
        if expected_baseline_mm is None
        else float(expected_baseline_mm)
    )
    return solve_stereo_calibration(
        pairs=pairs,
        image_size=image_size,
        square_mm=float(metadata["square_mm"]),
        pattern=pattern,
        max_reprojection_rms_px=float(
            audit.get("max_reprojection_rms_px", 1.5)
            if max_reprojection_rms_px is None
            else max_reprojection_rms_px
        ),
        min_pairs=int(audit.get("minimum_pairs", 10)),
        operator="replay",
        max_sync_delta_ms=float(audit.get("max_sync_delta_ms", 1.0)),
        layout=str(audit.get("layout", "diagnostic_replay")),
        expected_baseline_mm=baseline,
    )


def save_wizard_result(
    result: WizardResult,
    *,
    profile_path: str | Path | None = None,
    extras: Mapping[str, Any] | None = None,
) -> Path:
    """Persist a validated wizard result to the owned calibration + profile."""
    # Function-level imports keep this patchable in tests.
    from .workbench_profile import (
        default_calibration_path,
        update_profile,
        write_standalone_calibration,
    )

    if not result.validated:
        raise ChessboardCalibrationError("标定未通过门禁校验，不能保存")
    calibration_path = write_standalone_calibration(
        result.calibration, path=default_calibration_path()
    )
    updates = {
        "calibration_path": str(calibration_path),
        "calibration_current": copy.deepcopy(result.calibration),
        "rectification_recipe": copy.deepcopy(result.recipe),
    }
    if extras:
        unknown = sorted(set(extras) - {"camera", "chessboard", "wizard"})
        if unknown:
            raise ChessboardCalibrationError(f"不支持的附加配置段落：{unknown}")
        updates.update(copy.deepcopy(dict(extras)))
    update_profile(updates, path=profile_path)
    return calibration_path


_MAX_AUTO_PAIRS = 30
_AUTO_CAPTURE_COOLDOWN_S = 1.0
_NOVEL_POSE_RATIO = 0.06
_MIN_COVERED_ZONES = 6
_DEFAULT_SHARPNESS_GATE = 60.0

_ZONE_NAMES: dict[tuple[int, int], str] = {
    (0, 0): "左上",
    (1, 0): "上中",
    (2, 0): "右上",
    (0, 1): "左中",
    (1, 1): "中心",
    (2, 1): "右中",
    (0, 2): "左下",
    (1, 2): "下中",
    (2, 2): "右下",
}


def preferred_stream_mode(
    sizes: list[tuple[int, int]], *, layout: str
) -> tuple[int, int]:
    """Choose the highest-resolution plausible stereo stream.

    A side-by-side frame contains two landscape eye images.  The old chooser
    hard-coded 2560×720 ahead of 3840×1080, forcing this camera to calibrate
    from rescaled 1280×720 eyes even when its native 1920×1080 pair was
    available.  Native resolution gives corner fitting and RMS gates the best
    evidence.
    """
    if not sizes:
        raise ChessboardCalibrationError("相机没有返回可选流分辨率")
    if layout == "separate_devices":
        return max(sizes, key=lambda size: size[0] * size[1])
    plausible = [
        size
        for size in sizes
        if size[0] % 2 == 0
        and 1.2 <= (size[0] / 2) / max(1, size[1]) <= 2.0
    ]
    return max(plausible or sizes, key=lambda size: size[0] * size[1])


def zone_coverage_report(zones: Any) -> dict[str, Any]:
    """Map the 3×3 zones actually swept to operator-facing coverage info."""
    covered = {zone for zone in zones if zone in _ZONE_NAMES}
    missing = [name for zone, name in _ZONE_NAMES.items() if zone not in covered]
    return {
        "covered": len(covered),
        "total": len(_ZONE_NAMES),
        "required": _MIN_COVERED_ZONES,
        "missing": missing,
    }


def pose_is_novel(
    previous_xy: Any,
    new_xy: Any,
    image_size: tuple[int, int],
    *,
    min_ratio: float = _NOVEL_POSE_RATIO,
) -> bool:
    """Require real pose change before an automatic capture is accepted.

    Without this, sweeping the board slowly would stack many near-identical
    pairs that look numerous but add no calibration information.
    """
    if previous_xy is None:
        return True
    diagonal = float(math.hypot(image_size[0], image_size[1]))
    if diagonal <= 0:
        return True
    delta = float(np.linalg.norm(np.asarray(new_xy, dtype=float) - np.asarray(previous_xy, dtype=float)))
    return delta >= min_ratio * diagonal


def observation_is_novel(
    previous: BoardObservation | None,
    current: BoardObservation,
    image_size: tuple[int, int],
    pattern: tuple[int, int],
) -> bool:
    """Accept translation, distance change, or projective tilt as a new pose."""

    if previous is None:
        return True
    if pose_is_novel(previous.centroid_px, current.centroid_px, image_size):
        return True
    old_signature = _board_shape_signature(previous, pattern)
    new_signature = _board_shape_signature(current, pattern)
    scale_change = abs(math.log(new_signature[0] / max(old_signature[0], 1e-12)))
    shape_change = float(np.linalg.norm(new_signature[1:] - old_signature[1:]))
    return scale_change >= 0.10 or shape_change >= 0.04


def draw_detection_overlay(
    frame: np.ndarray, observation: BoardObservation, pattern: tuple[int, int]
) -> np.ndarray:
    """Return ``frame`` with the detected corners drawn on a copy.

    The overlay is purely cosmetic and must never take the preview down:
    canonicalized corners are float64 while OpenCV requires CV_32FC2, and a
    drawing failure degrades to the plain frame.
    """
    display = frame.copy()
    try:
        cv2.drawChessboardCorners(
            display,
            pattern,
            observation.corners_px.astype(np.float32).reshape(-1, 1, 2),
            True,
        )
    except cv2.error:
        return frame
    return display


class ChessboardWizardDialog:
    """Print a chessboard, grab pose-diverse pairs, calibrate, and save.

    Uses only the host application's Tk wrappers (``app.tk``/``app.ttk``/...)
    so the module import stays tkinter-free.  On success the calibration, its
    rectification recipe, and the wizard settings are persisted into the
    workbench profile; CAD registration (QR/side-view) is intentionally NOT
    done here and must follow once.
    """

    def __init__(
        self,
        app: Any,
        *,
        owner: Any | None = None,
        profile_path: str | Path | None = None,
    ) -> None:
        from .workbench_profile import default_profile, load_profile

        self.app = app
        self.owner = owner
        self.profile_path = profile_path
        profile, problem = load_profile(profile_path)
        self.profile = profile if problem == "" and profile is not None else default_profile()
        chess = self.profile.get("chessboard", {})
        wizard = self.profile.get("wizard", {})
        camera = self.profile.get("camera", {})
        tk, ttk = app.tk, app.ttk
        self.window = tk.Toplevel(app.root)
        self.window.title("棋盘格双目标定向导")
        self.window.geometry("1080x780")
        self.window.minsize(960, 700)
        self.window.transient(app.root)
        self.window.protocol("WM_DELETE_WINDOW", self.close)

        self.columns = tk.StringVar(value=str(chess.get("columns", 9)))
        self.rows = tk.StringVar(value=str(chess.get("rows", 7)))
        self.square_mm = tk.StringVar(value=f"{chess.get('square_mm', 20.0):g}")
        self.measured_square_mm = tk.StringVar(value=f"{chess.get('square_mm', 20.0):g}")
        self.operator = tk.StringVar(value=str(wizard.get("operator", "field")))
        self.max_rms = tk.StringVar(value=str(wizard.get("max_reprojection_rms_px", 1.5)))
        self.min_pairs = tk.StringVar(value=str(wizard.get("min_pairs", 10)))
        self.expected_baseline_mm = tk.StringVar(
            value=f"{float(wizard.get('expected_baseline_mm', 0.0)):g}"
        )
        self.nominal_fov_deg = tk.StringVar(
            value=f"{float(wizard.get('nominal_fov_deg', 0.0)):g}"
        )
        self.nominal_focal_length_mm = tk.StringVar(
            value=f"{float(wizard.get('nominal_focal_length_mm', 0.0)):g}"
        )
        self.max_sync = tk.StringVar(value="1.0")
        self.sharpness_gate = tk.StringVar(value="60")
        self.eye_width = tk.StringVar(value="1920")
        self.eye_height = tk.StringVar(value="1080")
        self.mode = tk.StringVar(value=str(camera.get("layout", "side_by_side_left_right")))
        self.left_index = tk.StringVar(value=str(camera.get("left_index", 0)))
        self.right_index = tk.StringVar(value=str(camera.get("right_index", 1)))
        self.session: Any | None = None
        self.after_id: str | None = None
        self.open_after_id: str | None = None
        self._opening = False
        self._closing = False
        self._probing_modes = False
        self._probe_cancelled = threading.Event()
        self._start_after_probe = False
        self._detector_error_keys: set[tuple[str, str, str]] = set()
        self._sb_disabled_roles: set[str] = set()
        self._tick = 0
        self.preview_images: list[Any] = []
        self._last_auto_time = 0.0
        self._last_left_observation: BoardObservation | None = None
        self.pairs: list[tuple[BoardObservation, BoardObservation]] = []
        self._pair_capture_ids: list[int | None] = []
        self.capture_archive: CalibrationCaptureArchive | None = None
        self.continuous = tk.BooleanVar(value=True)
        self.message = tk.StringVar(
            value=(
                "步骤：生成并 100% 打印棋盘格 → 量取实测格边长 → 打开预览，"
                "把纸张固定在硬质平板上，覆盖各区域并改变远近、前后倾斜和左右倾斜，"
                "抓拍 ≥10 组 → 完成标定。"
                + ("基础模式标定后可直接抓拍评估。" if getattr(self.owner, "analysis_mode", None) in {"elevation_depth", "elevation_auto"}
                   else "标定后仍需一次二维码定位。")
            )
        )

        main = ttk.Frame(self.window, padding=10)
        main.pack(fill="both", expand=True)

        params = ttk.LabelFrame(main, text="棋盘格与求解参数", padding=8)
        params.pack(fill="x", pady=(0, 6))
        for column, (label, variable, width) in enumerate((
            ("横向方格", self.columns, 5),
            ("纵向方格", self.rows, 5),
            ("文件格边长/mm", self.square_mm, 7),
            ("打印后实测格边长/mm", self.measured_square_mm, 9),
            ("操作员代号", self.operator, 12),
            ("允许RMS/px", self.max_rms, 6),
            ("最少组数", self.min_pairs, 5),
            ("同步容差/ms", self.max_sync, 7),
            ("清晰度门限", self.sharpness_gate, 6),
        )):
            column *= 2
            ttk.Label(params, text=label).grid(row=0, column=column, padx=(8, 2), sticky="w")
            ttk.Entry(params, textvariable=variable, width=width).grid(row=0, column=column + 1, sticky="w")
        ttk.Button(params, text="生成 A4 棋盘格打印 PNG", command=self.export_target).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=(7, 0)
        )
        ttk.Label(
            params,
            text="打印选择 100%/实际大小；打印后用量具测量格边长并填入“打印后实测格边长”。",
            foreground="#4A6178",
        ).grid(row=1, column=4, columnspan=6, sticky="w", pady=(7, 0))
        self.fit_hint = tk.StringVar(value="")
        ttk.Label(params, textvariable=self.fit_hint, foreground="#8A4E00").grid(
            row=2, column=0, columnspan=12, sticky="w", pady=(4, 0)
        )
        for variable in (self.columns, self.rows, self.square_mm):
            variable.trace_add("write", lambda *_e: self._update_fit_hint())
        self._update_fit_hint()

        camera_row = ttk.LabelFrame(main, text="相机", padding=8)
        camera_row.pack(fill="x", pady=(0, 6))
        ttk.Label(camera_row, text="采集方式：").pack(side="left")
        self.layout_combo = ttk.Combobox(
            camera_row,
            textvariable=self.mode,
            values=("side_by_side_left_right", "side_by_side_right_left", "separate_devices"),
            state="readonly",
            width=24,
        )
        self.layout_combo.pack(side="left", padx=(3, 10))
        self.layout_combo.bind("<<ComboboxSelected>>", lambda _e: self._sync_default())
        ttk.Label(camera_row, text="每目宽×高：").pack(side="left")
        ttk.Entry(camera_row, textvariable=self.eye_width, width=6).pack(side="left", padx=(3, 2))
        ttk.Entry(camera_row, textvariable=self.eye_height, width=6).pack(side="left", padx=(0, 10))
        ttk.Label(camera_row, text="左索引：").pack(side="left")
        ttk.Entry(camera_row, textvariable=self.left_index, width=4).pack(side="left", padx=(3, 8))
        ttk.Label(camera_row, text="右索引：").pack(side="left")
        ttk.Entry(camera_row, textvariable=self.right_index, width=4).pack(side="left", padx=(3, 8))
        ttk.Button(camera_row, text="检测设备", command=self.detect).pack(side="left", padx=3)
        ttk.Button(camera_row, text="打开预览", command=self.start).pack(side="left", padx=3)
        ttk.Button(camera_row, text="停止", command=self.stop).pack(side="left", padx=3)
        mode_row = ttk.Frame(camera_row)
        mode_row.pack(fill="x", pady=(6, 0))
        ttk.Label(mode_row, text="流分辨率：").pack(side="left")
        from .stereo_camera import STEREO_MODE_CANDIDATES

        self._mode_sizes: list[tuple[int, int]] = list(STEREO_MODE_CANDIDATES)
        self._mode_labels: list[str] = [
            self._mode_label(width, height) for width, height in self._mode_sizes
        ]
        default_mode = preferred_stream_mode(self._mode_sizes, layout=self._layout())
        self.stream_mode = tk.StringVar(
            value=self._mode_labels[self._mode_sizes.index(default_mode)]
        )
        self.stream_combo = ttk.Combobox(
            mode_row,
            textvariable=self.stream_mode,
            values=tuple(self._mode_labels),
            state="readonly",
            width=30,
        )
        self.stream_combo.pack(side="left", padx=(3, 10))
        self.stream_combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_stream_mode())
        ttk.Label(mode_row, text="实测镜头中心距/mm（建议必填）：").pack(side="left")
        ttk.Entry(
            mode_row, textvariable=self.expected_baseline_mm, width=7
        ).pack(side="left", padx=(3, 10))
        ttk.Label(
            mode_row,
            text="并排双目只在特定分辨率输出；选错会只看到单目画面。",
            foreground="#8A4E00",
        ).pack(side="left")
        hardware_row = ttk.Frame(camera_row)
        hardware_row.pack(fill="x", pady=(4, 0))
        ttk.Label(hardware_row, text="硬件规格（随原始照片保存）：视场角/°").pack(
            side="left"
        )
        ttk.Entry(hardware_row, textvariable=self.nominal_fov_deg, width=7).pack(
            side="left", padx=(3, 10)
        )
        ttk.Label(hardware_row, text="镜头焦距/mm").pack(side="left")
        ttk.Entry(
            hardware_row,
            textvariable=self.nominal_focal_length_mm,
            width=7,
        ).pack(side="left", padx=(3, 10))
        ttk.Label(
            hardware_row,
            text="焦距毫米值仅作硬件记录；像素内参仍由棋盘照片求解。",
            foreground="#4A6178",
        ).pack(side="left")
        self.sync_gate_label = tk.StringVar(value="标定阶段为静态棋盘，同步容差只用于记录。")
        ttk.Label(camera_row, textvariable=self.sync_gate_label, foreground="#4A6178").pack(
            anchor="w", pady=(2, 0)
        )
        self.exposure_status = tk.StringVar(value="快门目标 ≤1/200 秒（5 ms）；待连接")
        ttk.Label(camera_row, textvariable=self.exposure_status, foreground="#8A4E00", wraplength=1040).pack(
            anchor="w", pady=(2, 0)
        )

        previews = ttk.Frame(main)
        previews.pack(fill="both", expand=True)
        previews.columnconfigure(0, weight=1)
        previews.columnconfigure(1, weight=1)
        self.left_status = tk.StringVar(value="左目：未打开预览")
        self.right_status = tk.StringVar(value="右目：未打开预览")
        ttk.Label(previews, text="左目原始帧").grid(row=0, column=0, pady=3)
        ttk.Label(previews, text="右目原始帧").grid(row=0, column=1, pady=3)
        ttk.Label(previews, textvariable=self.left_status, foreground="#355371").grid(
            row=2, column=0, pady=(2, 0)
        )
        ttk.Label(previews, textvariable=self.right_status, foreground="#355371").grid(
            row=2, column=1, pady=(2, 0)
        )
        self.left_preview = ttk.Label(previews, anchor="center")
        self.right_preview = ttk.Label(previews, anchor="center")
        self.left_preview.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        self.right_preview.grid(row=1, column=1, sticky="nsew", padx=(4, 0))
        previews.rowconfigure(1, weight=1)

        bottom = ttk.Frame(main)
        bottom.pack(fill="both", pady=(6, 0))
        columns = ("index", "left_zone", "right_zone")
        self.tree = ttk.Treeview(bottom, columns=columns, show="headings", height=5)
        for key, title, width in (
            ("index", "组号", 60),
            ("left_zone", "左目画面区域", 110),
            ("right_zone", "右目画面区域", 110),
        ):
            self.tree.heading(key, text=title)
            self.tree.column(key, width=width)
        self.tree.pack(side="left", fill="both", expand=True)
        actions = ttk.Frame(bottom)
        actions.pack(side="left", fill="y", padx=(8, 0))
        ttk.Checkbutton(
            actions,
            text="连续抓拍（移动棋盘格自动采集）",
            variable=self.continuous,
        ).pack(fill="x", pady=(0, 3))
        ttk.Button(actions, text="抓拍一组", command=self.capture_pair).pack(fill="x", pady=3)
        ttk.Button(
            actions,
            text="移除选中组",
            command=self.remove_selected,
        ).pack(fill="x", pady=3)
        ttk.Button(actions, text="移除上一组", command=self.remove_last).pack(fill="x", pady=3)
        ttk.Button(
            actions,
            text="恢复最近照片组",
            command=self.restore_recent_pairs,
        ).pack(fill="x", pady=3)
        ttk.Button(actions, text="完成标定并保存", command=self.finish).pack(fill="x", pady=(14, 3))
        ttk.Button(actions, text="查看完整日志", command=self.show_log).pack(fill="x", pady=3)
        ttk.Button(actions, text="关闭", command=self.close).pack(fill="x", pady=3)

        ttk.Label(main, textvariable=self.message, wraplength=1040, foreground="#355371").pack(
            fill="x", pady=(6, 0)
        )

    # -- helpers -----------------------------------------------------------

    def _update_fit_hint(self) -> None:
        try:
            columns = int(self.columns.get())
            rows = int(self.rows.get())
            square_mm = float(self.square_mm.get())
        except ValueError:
            self.fit_hint.set("棋盘格占用：列/行须为整数，格边长须为数值。")
            return
        width_mm = columns * square_mm
        height_mm = rows * square_mm
        fits = width_mm <= _A4_BOARD_WIDTH_MM and height_mm <= _A4_BOARD_HEIGHT_MM
        self.fit_hint.set(
            f"棋盘格占用 {width_mm:g}×{height_mm:g} mm（A4 可打印区域 "
            f"{_A4_BOARD_WIDTH_MM:g}×{_A4_BOARD_HEIGHT_MM:g} mm）—— "
            + ("可打印" if fits else "超出 A4，无法打印")
        )

    def _int_value(self, variable: Any, label: str, *, minimum: int = 0) -> int:
        try:
            value = int(variable.get())
        except ValueError as error:
            raise ChessboardCalibrationError(f"{label} 必须是整数") from error
        if value < minimum:
            raise ChessboardCalibrationError(f"{label} 不能小于 {minimum}")
        return value

    def _float_value(self, variable: Any, label: str, *, positive: bool = True) -> float:
        try:
            value = float(variable.get())
        except ValueError as error:
            raise ChessboardCalibrationError(f"{label} 必须是数值") from error
        if positive and value <= 0:
            raise ChessboardCalibrationError(f"{label} 必须为正数")
        return value

    def _pattern(self) -> tuple[int, int]:
        columns = self._int_value(self.columns, "横向方格", minimum=4)
        rows = self._int_value(self.rows, "纵向方格", minimum=4)
        return columns - 1, rows - 1

    def _layout(self) -> str:
        layout = self.mode.get()
        if layout not in {"side_by_side_left_right", "side_by_side_right_left", "separate_devices"}:
            raise ChessboardCalibrationError("请选择有效的双目采集方式")
        return layout

    def _sync_default(self) -> None:
        if self._layout() == "separate_devices":
            self.max_sync.set("50.0")
            self.sync_gate_label.set("独立设备：抓拍时将检查主机时间差。")
        else:
            self.max_sync.set("1.0")
            self.sync_gate_label.set("同一并排帧：时间差恒为 0。")
        self.stop()

    def _indices(self) -> tuple[int, int | None]:
        left = self._int_value(self.left_index, "左索引")
        right = None
        if self._layout() == "separate_devices":
            right = self._int_value(self.right_index, "右索引")
            if right == left:
                raise ChessboardCalibrationError("独立设备模式下左右索引不能相同")
        return left, right

    def _eye_size(self) -> tuple[int, int]:
        return (
            self._int_value(self.eye_width, "每目宽度", minimum=1),
            self._int_value(self.eye_height, "每目高度", minimum=1),
        )

    def _hardware_specs(self) -> dict[str, float]:
        baseline = self._float_value(
            self.expected_baseline_mm, "实测镜头中心距", positive=False
        )
        nominal_fov = self._float_value(
            self.nominal_fov_deg, "标称视场角", positive=False
        )
        nominal_focal = self._float_value(
            self.nominal_focal_length_mm, "标称镜头焦距", positive=False
        )
        if baseline < 0 or nominal_fov < 0 or nominal_focal < 0:
            raise ChessboardCalibrationError("硬件规格可填 0（未知）或正数")
        if nominal_fov and not 10 <= nominal_fov < 180:
            raise ChessboardCalibrationError("标称视场角必须为 0 或 10～180 度")
        if nominal_focal and not 0.1 <= nominal_focal <= 100:
            raise ChessboardCalibrationError("标称镜头焦距必须为 0 或 0.1～100 mm")
        return {
            "expected_baseline_mm": baseline,
            "nominal_fov_deg": nominal_fov,
            "nominal_focal_length_mm": nominal_focal,
        }

    def _capture_session_metadata(self) -> dict[str, Any]:
        return {
            "hardware": self._hardware_specs(),
            "camera": {
                "layout": self._layout(),
                "left_index": self._indices()[0],
                "right_index": self._indices()[1],
                "eye_size_px": list(self._eye_size()),
                "exposure": {
                    str(index): dict(record)
                    for index, record in getattr(getattr(self, "session", None), "exposure_settings", {}).items()
                },
            },
            "target": {
                "pattern_inner_corners": list(self._pattern()),
                "measured_square_mm": self._float_value(
                    self.measured_square_mm, "打印后实测格边长"
                ),
            },
        }

    def _current_solve_options(self) -> dict[str, Any]:
        return {
            "max_reprojection_rms_px": self._float_value(self.max_rms, "允许RMS"),
            "minimum_pairs": self._int_value(self.min_pairs, "最少组数", minimum=3),
            "max_sync_delta_ms": self._float_value(self.max_sync, "同步容差", positive=False),
            "layout": self._layout(),
            "expected_baseline_mm": self._hardware_specs()["expected_baseline_mm"] or None,
        }

    def _ensure_capture_archive(self) -> CalibrationCaptureArchive:
        if self.capture_archive is None:
            self.capture_archive = CalibrationCaptureArchive.create(
                session_metadata=self._capture_session_metadata()
            )
        else:
            self.capture_archive.update_session(self._capture_session_metadata())
        return self.capture_archive

    def _save_pair_checkpoint(self) -> Path | None:
        """Persist the current accepted set without interrupting capture."""

        checkpoint = _DIAGNOSTIC_ROOT / "calibration_autosave.npz"
        if not self.pairs:
            checkpoint.unlink(missing_ok=True)
            return None
        return write_calibration_checkpoint(
            list(self.pairs),
            image_size=self._eye_size(),
            pattern=self._pattern(),
            square_mm=self._float_value(
                self.measured_square_mm, "打印后实测格边长"
            ),
            root=_DIAGNOSTIC_ROOT,
            solve_options=self._current_solve_options(),
            metadata_extra={
                "session_path": (
                    str(self.capture_archive.session_path)
                    if self.capture_archive is not None
                    else ""
                ),
                "pair_capture_ids": list(self._pair_capture_ids),
            },
        )

    def restore_recent_pairs(self) -> None:
        """Restore the newest failed solve/checkpoint and remove known bad views."""

        from .logging_config import get_logger, log_event

        try:
            candidates = sorted(
                _DIAGNOSTIC_ROOT.glob("calibration_*.npz"),
                key=lambda path: path.stat().st_mtime_ns,
                reverse=True,
            )
            if not candidates:
                raise ChessboardCalibrationError("没有找到可恢复的标定照片组")
            restore_minimum_pairs = self._int_value(
                self.min_pairs, "最少组数", minimum=3
            )
            source, restore_inventory = _select_recent_recoverable_diagnostic(
                candidates,
                minimum_pairs=restore_minimum_pairs,
            )
            retained, metadata, discarded_indices = reusable_calibration_pairs(source)
            audit = _diagnostic_solve_options(metadata)
            discarded = set(discarded_indices)

            image_size = tuple(int(value) for value in metadata["image_size_px"])
            pattern = tuple(
                int(value) for value in metadata["pattern_inner_corners"]
            )
            self.eye_width.set(str(image_size[0]))
            self.eye_height.set(str(image_size[1]))
            self.columns.set(str(pattern[0] + 1))
            self.rows.set(str(pattern[1] + 1))
            self.measured_square_mm.set(f"{float(metadata['square_mm']):g}")
            saved_rms = audit.get("max_reprojection_rms_px")
            if isinstance(saved_rms, (int, float)) and float(saved_rms) > 0:
                self.max_rms.set(f"{float(saved_rms):g}")
            if type(audit.get("minimum_pairs")) is int:
                self.min_pairs.set(str(audit["minimum_pairs"]))
            saved_sync = audit.get("max_sync_delta_ms")
            if isinstance(saved_sync, (int, float)) and float(saved_sync) >= 0:
                self.max_sync.set(f"{float(saved_sync):g}")
            saved_layout = audit.get("layout")
            if saved_layout in {
                "side_by_side_left_right",
                "side_by_side_right_left",
                "separate_devices",
            }:
                self.mode.set(str(saved_layout))
            saved_baseline = audit.get("expected_baseline_mm")
            if isinstance(saved_baseline, (int, float)) and float(saved_baseline) > 0:
                self.expected_baseline_mm.set(f"{float(saved_baseline):g}")

            previous_archive = self.capture_archive
            previous_capture_ids = list(self._pair_capture_ids)
            restored_archive: CalibrationCaptureArchive | None = None
            restored_capture_ids: list[int | None] = [None] * len(retained)
            archive_note = ""
            archive_link = metadata.get("capture_archive", {})
            if isinstance(archive_link, Mapping) and archive_link.get("session_path"):
                try:
                    all_capture_ids = list(archive_link.get("pair_capture_ids", []))
                    original_count = len(retained) + len(discarded)
                    if len(all_capture_ids) != original_count:
                        raise ChessboardCalibrationError(
                            "原始照片编号数量与角点组不一致"
                        )
                    restored_capture_ids = [
                        (
                            int(capture_id)
                            if capture_id is not None
                            else None
                        )
                        for index, capture_id in enumerate(
                            all_capture_ids, start=1
                        )
                        if index not in discarded
                    ]
                    restored_archive = CalibrationCaptureArchive.open_existing(
                        str(archive_link["session_path"])
                    )
                    hardware = restored_archive.manifest["session"].get(
                        "hardware", {}
                    )
                    if isinstance(hardware, Mapping):
                        for key, variable in (
                            ("expected_baseline_mm", self.expected_baseline_mm),
                            ("nominal_fov_deg", self.nominal_fov_deg),
                            (
                                "nominal_focal_length_mm",
                                self.nominal_focal_length_mm,
                            ),
                        ):
                            value = hardware.get(key)
                            if isinstance(value, (int, float)) and float(value) > 0:
                                variable.set(f"{float(value):g}")
                    archive_note = "；原始左右 PNG 会话已重新关联"
                except (OSError, ValueError) as error:
                    archive_note = f"；原始照片会话未关联：{error}"
            _reconcile_restored_capture_status(
                previous_archive=previous_archive,
                previous_capture_ids=previous_capture_ids,
                restored_archive=restored_archive,
                restored_capture_ids=restored_capture_ids,
            )
            self.capture_archive = restored_archive
            self.pairs = retained
            self._pair_capture_ids = restored_capture_ids
            self._last_left_observation = retained[-1][0]
            self._refresh_pairs()
            self._save_pair_checkpoint()
            coverage = zone_coverage_report(
                left.centroid_zone for left, _right in retained
            )
            removal_note = (
                "；已跳过先前判定异常的第 "
                + "、".join(str(value) for value in sorted(discarded))
                + " 组"
                if discarded
                else ""
            )
            missing_note = (
                "；还需补拍区域：" + "、".join(coverage["missing"])
                if coverage["missing"]
                else "；区域覆盖已满足"
            )
            selection_note = (
                "；较新的自动保存不足最少组数，已恢复最近可求解的历史组"
                if source != candidates[0]
                else ""
            )
            self.message.set(
                f"已从 {source.name} 恢复 {len(retained)} 组{removal_note}"
                f"{missing_note}{archive_note}{selection_note}。现有照片可继续使用。"
            )
            log_event(
                get_logger("calibration_wizard"),
                "wizard_pairs_restored",
                source=source,
                restored_pair_count=len(retained),
                discarded_pair_indices=sorted(discarded),
                restored_capture_ids=[
                    capture_id
                    for capture_id in restored_capture_ids
                    if capture_id is not None
                ],
                capture_session=(
                    restored_archive.session_path
                    if restored_archive is not None
                    else None
                ),
                candidate_inventory=[
                    {
                        **item,
                        "path": str(item["path"]),
                    }
                    for item in restore_inventory
                ],
                missing_zones=coverage["missing"],
            )
        except Exception as error:
            log_event(
                get_logger("calibration_wizard"),
                "wizard_pairs_restore_failed",
                error_type=type(error).__name__,
                error=str(error),
            )
            self.app.messagebox.showerror(
                "恢复照片组失败", str(error), parent=self.window
            )

    # -- actions -----------------------------------------------------------

    def export_target(self) -> None:
        try:
            columns = self._int_value(self.columns, "横向方格", minimum=4)
            rows = self._int_value(self.rows, "纵向方格", minimum=4)
            square_mm = self._float_value(self.square_mm, "文件格边长")
            selected = self.app.filedialog.asksaveasfilename(
                parent=self.window,
                title="保存棋盘格打印文件",
                initialfile=f"棋盘格_{columns}x{rows}_{square_mm:g}mm_300dpi.png",
                defaultextension=".png",
                filetypes=(("PNG", "*.png"),),
            )
            if not selected:
                return
            protect = getattr(getattr(self.app, "measurement_panel", None), "_protect_output", None)
            if protect is not None:
                protect(Path(selected))
            path = write_printable_chessboard_png(
                selected, square_mm=square_mm, columns=columns, rows=rows, dpi=300
            )
            self.measured_square_mm.set(f"{square_mm:g}")
            self.message.set(
                f"棋盘格打印文件已保存：{path}。请按 100%/实际大小打印，"
                "然后用量具测量格边长并更新“打印后实测格边长”。"
            )
        except (OSError, ValueError) as error:
            self.app.messagebox.showerror("棋盘格生成失败", str(error), parent=self.window)

    def detect(self) -> None:
        from .logging_config import get_logger, log_event
        from .stereo_camera import probe_video_devices, run_in_background

        self.stop()
        if self.open_after_id is not None or self._opening:
            self.message.set("正在处理上一个相机任务，请稍候…")
            return
        self._probe_cancelled = threading.Event()
        self._start_after_probe = False
        layout = self._layout()
        required_devices = 2 if layout == "separate_devices" else 1
        self.message.set("正在检测视频设备索引 0 / 5…（打开超时的设备会自动跳过；点“停止”可中断）")
        self.left_status.set("左目：检测设备中…")
        self.right_status.set("右目：检测设备中…")
        log_event(get_logger("calibration_wizard"), "wizard_device_probe_start")
        outcome: dict[str, Any] = {}
        progress_state: dict[str, Any] = {"index": -1, "status": ""}
        run_in_background(
            lambda: probe_video_devices(
                maximum_index=5,
                maximum_devices=required_devices,
                per_index_timeout_s=5.0,
                progress=lambda index, status: progress_state.update(index=index, status=status),
                cancelled=self._probe_cancelled.is_set,
            ),
            lambda result, error: outcome.setdefault("done", (result, error)),
        )
        self._poll_detect(outcome, progress_state)

    def _poll_detect(self, outcome: dict[str, Any], progress_state: dict[str, Any]) -> None:
        if self._closing:
            return
        if "done" not in outcome:
            index, status = progress_state["index"], progress_state["status"]
            if index >= 0 and status == "timeout":
                self.message.set(
                    f"索引 {index} 打开超时（多为虚拟摄像头驱动），已跳过；继续检测后续索引…"
                )
            elif index >= 0:
                self.message.set(f"正在检测视频设备索引 {index} / 5…")
            self.open_after_id = self.window.after(
                200, lambda: self._poll_detect(outcome, progress_state)
            )
            return
        self.open_after_id = None
        result, error = outcome["done"]
        cancelled = self._probe_cancelled.is_set()
        queued_start = self._start_after_probe
        self._start_after_probe = False
        try:
            if error is not None and not queued_start:
                raise error
            devices = result or []
            if not devices and not queued_start:
                detail = "检测已停止，未得到完整结果。" if cancelled else ""
                raise ChessboardCalibrationError(
                    f"未检测到 OpenCV 可打开的视频设备。{detail}"
                )
            if devices:
                self.left_index.set(str(devices[0]["index"]))
                if len(devices) > 1:
                    self.right_index.set(str(devices[1]["index"]))
            summary = "；".join(
                f"索引 {item['index']}：{item['width']}×{item['height']}" for item in devices
            )
            prefix = "检测已停止；" if cancelled else ""
            self.message.set(
                f"{prefix}检测到 {len(devices)} 个可用设备：{summary}；正在探测支持的分辨率…"
            )
            self.left_status.set("左目：未打开预览")
            self.right_status.set("右目：未打开预览")
            if queued_start:
                from .logging_config import get_logger, log_event

                log_event(
                    get_logger("calibration_wizard"),
                    "wizard_camera_open_resumed_after_probe",
                    detected_devices=len(devices),
                )
                self.message.set("设备探测已停止，正在按当前索引和分辨率打开预览…")
                self.window.after(0, self.start)
            elif not cancelled:
                self._probe_modes(int(devices[0]["index"]))
        except Exception as error:  # surfaced to the operator, never silent
            self.app.messagebox.showerror("相机检测失败", str(error), parent=self.window)
            self.message.set(f"相机检测失败：{error}")

    def _mode_label(self, width: int, height: int) -> str:
        if self._layout() != "separate_devices" and width % 2 == 0:
            return f"{width}×{height}（每目 {width // 2}×{height}）"
        return f"{width}×{height}"

    def _probe_modes(self, index: int) -> None:
        from .stereo_camera import probe_video_modes, run_in_background

        self._probing_modes = True
        self._mode_before_probe = self.stream_mode.get()
        self.stream_mode.set("正在探测相机支持的分辨率（数秒）…")
        outcome: dict[str, Any] = {}
        run_in_background(
            lambda: probe_video_modes(
                index=index,
                cancelled=self._probe_cancelled.is_set,
            ),
            lambda result, error: outcome.setdefault("done", (result, error)),
        )
        self._poll_modes(outcome)

    def _poll_modes(self, outcome: dict[str, Any]) -> None:
        if self._closing:
            return
        if "done" not in outcome:
            self.open_after_id = self.window.after(200, lambda: self._poll_modes(outcome))
            return
        self.open_after_id = None
        self._probing_modes = False
        sizes, error = outcome["done"]
        queued_start = self._start_after_probe
        self._start_after_probe = False
        if queued_start:
            from .logging_config import get_logger, log_event

            self.stream_mode.set(self._mode_before_probe)
            log_event(
                get_logger("calibration_wizard"),
                "wizard_camera_open_resumed_after_mode_probe",
                detected_modes=len(sizes or []),
            )
            self.message.set("分辨率探测已停止，正在按当前选择打开预览…")
            self.window.after(0, self.start)
            return
        if error is not None or not sizes:
            # Keep the static candidate list selectable; some drivers refuse
            # to reopen after many rapid cycles, which is not an error the
            # operator can act on.
            self.stream_mode.set(self._mode_before_probe)
            self.message.set(
                (
                    "分辨率探测已停止；已保留原选择，可直接打开预览。"
                    if self._probe_cancelled.is_set()
                    else "自动分辨率探测未成功（部分驱动多次开关后会暂时拒绝打开）；"
                    "已预置常见并排分辨率，可直接从下拉框选择，或手动输入每目宽×高。"
                )
            )
            return
        self._mode_sizes = list(sizes)
        self._mode_labels = [
            self._mode_label(width, height) for width, height in self._mode_sizes
        ]

        preferred = preferred_stream_mode(self._mode_sizes, layout=self._layout())
        best = self._mode_sizes.index(preferred)
        self.stream_combo.configure(values=tuple(self._mode_labels))
        self.stream_mode.set(self._mode_labels[best])
        self._apply_stream_mode()
        self.message.set(
            f"相机实测支持 {len(sizes)} 种流分辨率，已选 {self.stream_mode.get()}。"
        )

    def _apply_stream_mode(self) -> None:
        label = self.stream_mode.get()
        if label not in getattr(self, "_mode_labels", []):
            return
        width, height = self._mode_sizes[self._mode_labels.index(label)]
        if self._layout() != "separate_devices" and width % 2 == 0:
            self.eye_width.set(str(width // 2))
            self.eye_height.set(str(height))
        else:
            self.eye_width.set(str(width))
            self.eye_height.set(str(height))
        self.message.set(
            f"已选流分辨率 {width}×{height}；每目按 "
            f"{self.eye_width.get()}×{self.eye_height.get()} 打开。"
        )

    def start(self) -> bool:
        from .logging_config import get_logger, log_event
        from .stereo_camera import StereoCameraSession, run_in_background

        if self._opening:
            self.message.set("相机正在打开，请稍候…")
            return False
        if self._probing_modes or self.open_after_id is not None:
            self._start_after_probe = True
            self._probe_cancelled.set()
            log_event(
                get_logger("calibration_wizard"),
                "wizard_camera_open_queued",
                probing_modes=self._probing_modes,
            )
            self.message.set("正在停止设备探测；当前驱动调用结束后将自动打开预览…")
            return True
        try:
            layout = self._layout()
            left, right = self._indices()
            eye_width, eye_height = self._eye_size()
        except ValueError as error:
            self.app.messagebox.showerror("相机打开失败", str(error), parent=self.window)
            return False
        self.stop()
        self._probe_cancelled = threading.Event()
        self._detector_error_keys.clear()
        self._sb_disabled_roles.clear()
        self._opening = True
        self.message.set("正在打开相机；部分设备在 DirectShow 下需要数秒，请稍候…")
        self.exposure_status.set("快门目标 ≤1/200 秒（5 ms）；正在连接并设置")
        self.left_status.set("左目：正在打开相机…")
        self.right_status.set("右目：正在打开相机…")
        logger = get_logger("calibration_wizard")
        log_event(
            logger,
            "wizard_camera_open_start",
            layout=layout,
            left_index=left,
            right_index=right,
            eye_width=eye_width,
            eye_height=eye_height,
        )
        outcome: dict[str, Any] = {}

        def _task() -> StereoCameraSession:
            session = StereoCameraSession(
                layout=layout,
                left_index=left,
                right_index=right,
                eye_width=eye_width,
                eye_height=eye_height,
            )
            session.open()
            return session

        def _on_result(result: Any, error: Exception | None) -> None:
            if error is not None and result is None:
                outcome["done"] = (None, error)
                return
            if self._closing:
                result.close()
                outcome["done"] = (None, ChessboardCalibrationError("窗口已关闭"))
                return
            outcome["done"] = (result, None)

        run_in_background(_task, _on_result)
        self._poll_open(outcome)
        return True

    def _poll_open(self, outcome: dict[str, Any]) -> None:
        if self._closing:
            return
        if "done" not in outcome:
            self.open_after_id = self.window.after(200, lambda: self._poll_open(outcome))
            return
        self.open_after_id = None
        self._opening = False
        session, error = outcome["done"]
        from .logging_config import get_logger, log_event

        logger = get_logger("calibration_wizard")
        if error is not None:
            self.exposure_status.set("快门目标 ≤1/200 秒（5 ms）；相机未连接")
            log_event(logger, "wizard_camera_open_failed", error=str(error))
            self.message.set(f"相机打开失败：{error}")
            self.left_status.set("左目：未打开")
            self.right_status.set("右目：未打开")
            self.app.messagebox.showerror("相机打开失败", str(error), parent=self.window)
            return
        self.session = session
        self.exposure_status.set(session.exposure_summary)
        log_event(logger, "wizard_camera_open_finished")
        self.message.set("相机已打开；持棋盘格覆盖画面各区域，点击“抓拍一组”。")
        # Show raw frames before running a full-resolution detector so opening
        # a 3840x1080 stream cannot look like a frozen/failed preview.
        self._tick = 1
        self._update_preview()

    def _photo(self, frame: np.ndarray) -> Any:
        import base64

        height, width = frame.shape[:2]
        scale = min(470 / width, 420 / height, 1.0)
        shown = (
            cv2.resize(
                frame,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            if scale < 1
            else frame
        )
        ok, encoded = cv2.imencode(".png", shown)
        if not ok:
            raise ChessboardCalibrationError("无法生成相机预览")
        return self.app.tk.PhotoImage(
            data=base64.b64encode(encoded).decode("ascii"), format="png"
        )

    def _update_preview(self) -> None:
        if self.session is None:
            return
        stage = "read_pair"
        try:
            pair = self.session.read_pair()
            stage = "parse_pattern"
            pattern = self._pattern()
            status = {role: "未检出" for role, _frame in (("left", pair.left), ("right", pair.right))}
            displays = {"left": pair.left, "right": pair.right}
            observations: dict[str, BoardObservation] = {}
            if self._tick % 3 == 0:
                for role, frame in (("left", pair.left), ("right", pair.right)):
                    stage = f"detect_{role}"
                    observation = detect_board_corners(
                        frame,
                        pattern=pattern,
                        use_sb=role not in self._sb_disabled_roles,
                        sb_accuracy=False,
                        on_cv_error=lambda detector_stage, error, current_role=role, current_frame=frame: self._log_detector_cv_error(
                            current_role, current_frame, detector_stage, error
                        ),
                    )
                    if observation is not None:
                        observations[role] = observation
                        displays[role] = draw_detection_overlay(frame, observation, pattern)
                        status[role] = (
                            f"已检出，清晰度 {observation.sharpness:.0f}，"
                            f"区域 {observation.centroid_zone}"
                        )
                    else:
                        status[role] = "未检出，请让整个棋盘格进入画面"
                self._maybe_auto_capture(pair, observations)
            stage = "encode_preview"
            self.preview_images = [self._photo(displays["left"]), self._photo(displays["right"])]
            stage = "update_widgets"
            self.left_preview.configure(image=self.preview_images[0])
            self.right_preview.configure(image=self.preview_images[1])
            self.left_status.set(f"左目：{status['left']}")
            self.right_status.set(f"右目：{status['right']}")
            self._tick += 1
            self.after_id = self.window.after(120, self._update_preview)
        except Exception as error:  # native cv2 errors included, never silent
            from .logging_config import get_logger

            get_logger("calibration_wizard").exception(
                "wizard_preview_failed",
                extra={
                    "event": "wizard_preview_failed",
                    "fields": {
                        "stage": stage,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "eye_width": self.eye_width.get(),
                        "eye_height": self.eye_height.get(),
                        "tick": self._tick,
                    },
                },
            )
            self.stop()
            self.app.messagebox.showerror("相机读取失败", str(error), parent=self.window)

    def _log_detector_cv_error(
        self,
        role: str,
        frame: np.ndarray,
        stage: str,
        error: Exception,
    ) -> None:
        """Record each distinct native detector failure once per preview."""

        key = (role, stage, str(error))
        if key in self._detector_error_keys:
            return
        self._detector_error_keys.add(key)
        if stage == "find_chessboard_corners_sb":
            # Both eyes must use the same detector within a stereo pair.
            # If one SB backend fails, switch the complete session to the
            # classic detector; the current mixed-backend frame is skipped.
            self._sb_disabled_roles.update(("left", "right"))
        from .logging_config import get_logger

        get_logger("calibration_wizard").warning(
            "wizard_detector_cv_error",
            exc_info=True,
            extra={
                "event": "wizard_detector_cv_error",
                "fields": {
                    "role": role,
                    "stage": stage,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "frame_shape": tuple(int(value) for value in frame.shape),
                    "fallback": "classic_or_no_detection",
                },
            },
        )

    def _maybe_auto_capture(
        self, pair: Any, observations: dict[str, BoardObservation]
    ) -> None:
        """Accept a pair automatically when the pose really changed.

        Continuously sweeping the board replaces per-pose button clicks; the
        novelty gate keeps near-identical frames from padding the count.
        """
        if not self.continuous.get() or len(observations) < 2:
            return
        if len(self.pairs) >= _MAX_AUTO_PAIRS:
            self.continuous.set(False)
            self.message.set(f"已达自动采集上限 {_MAX_AUTO_PAIRS} 组，请点击“完成标定并保存”。")
            return
        now = time.monotonic()
        if now - self._last_auto_time < _AUTO_CAPTURE_COOLDOWN_S:
            return
        left_obs = observations["left"]
        right_obs = observations["right"]
        if left_obs.detector != right_obs.detector:
            self.message.set("左右目检测器刚完成同步切换；本帧已跳过，请保持棋盘格稳定。")
            return
        # Motion blur (long exposure while the board moves) inflates the
        # reprojection RMS without being visible in the pair count, so blurred
        # frames are skipped before they can enter the solve.
        try:
            threshold = float(self.sharpness_gate.get())
        except ValueError:
            threshold = _DEFAULT_SHARPNESS_GATE
        if not math.isfinite(threshold) or threshold < 0:
            threshold = _DEFAULT_SHARPNESS_GATE
        if threshold > 0 and min(left_obs.sharpness, right_obs.sharpness) < threshold:
            self.message.set(
                f"清晰度不足（左 {left_obs.sharpness:.0f} / 右 {right_obs.sharpness:.0f}"
                f" < {threshold:g}），已跳过自动抓拍；请放慢移动或增加光照。"
            )
            return
        if not observation_is_novel(
            self._last_left_observation,
            left_obs,
            (pair.left.shape[1], pair.left.shape[0]),
            self._pattern(),
        ):
            return
        self._accept_pair(
            left_obs,
            observations["right"],
            automatic=True,
            raw_left=pair.left,
            raw_right=pair.right,
        )
        self._last_auto_time = now
        self._last_left_observation = left_obs

    def _accept_pair(
        self,
        left_obs: BoardObservation,
        right_obs: BoardObservation,
        *,
        automatic: bool,
        raw_left: np.ndarray | None = None,
        raw_right: np.ndarray | None = None,
    ) -> None:
        from .logging_config import get_logger, log_event

        if (
            left_obs.detector != "unknown"
            and right_obs.detector != "unknown"
            and left_obs.detector != right_obs.detector
        ):
            raise ChessboardCalibrationError("左右目必须使用同一个角点检测器")
        existing_detectors = {
            observation.detector
            for pair in self.pairs
            for observation in pair
            if observation.detector != "unknown"
        }
        incoming_detectors = {
            detector
            for detector in (left_obs.detector, right_obs.detector)
            if detector != "unknown"
        }
        reset_count = 0
        if existing_detectors and incoming_detectors != existing_detectors:
            reset_count = len(self.pairs)
            reset_capture_ids = [
                value for value in self._pair_capture_ids if value is not None
            ]
            if self.capture_archive is not None:
                self.capture_archive.mark_status(
                    reset_capture_ids,
                    status="excluded_detector_reset",
                    reason="角点检测器在同一数据集中发生切换",
                )
            self.pairs.clear()
            self._pair_capture_ids.clear()
            self._last_left_observation = None
            log_event(
                get_logger("calibration_wizard"),
                "wizard_detector_dataset_reset",
                discarded_pair_count=reset_count,
                previous_detectors=sorted(existing_detectors),
                new_detectors=sorted(incoming_detectors),
            )
        aligned_right = align_pair_orientation(
            left_obs, right_obs, pattern=self._pattern()
        )
        orientation_flipped = not np.array_equal(
            aligned_right.corners_px, right_obs.corners_px
        )
        if (raw_left is None) != (raw_right is None):
            raise ChessboardCalibrationError("左右标定原始照片必须成对保存")
        capture_id: int | None = None
        if raw_left is not None and raw_right is not None:
            capture_id = self._ensure_capture_archive().add_pair(
                raw_left,
                raw_right,
                metadata={
                    "automatic": bool(automatic),
                    "left_zone": list(left_obs.centroid_zone),
                    "right_zone": list(aligned_right.centroid_zone),
                    "left_sharpness": float(left_obs.sharpness),
                    "right_sharpness": float(aligned_right.sharpness),
                    "detector": left_obs.detector,
                    "right_corner_order_rotated_180": orientation_flipped,
                    "median_raw_disparity_px": float(
                        np.median(
                            left_obs.corners_px[:, 0]
                            - aligned_right.corners_px[:, 0]
                        )
                    ),
                },
            )
        self.pairs.append((left_obs, aligned_right))
        self._pair_capture_ids.append(capture_id)
        self._last_left_observation = left_obs
        minimum = self._int_value(self.min_pairs, "最少组数", minimum=3)
        count = len(self.pairs)
        self._refresh_pairs()
        coverage = zone_coverage_report(
            left.centroid_zone for left, _right in self.pairs
        )
        diversity = pose_diversity_report(
            [left for left, _right in self.pairs],
            image_size=self._eye_size(),
        )
        image_geometry = projective_pose_report(
            [left for left, _right in self.pairs],
            image_size=self._eye_size(),
            pattern=self._pattern(),
        )
        summary = f"{'自动' if automatic else '手动'}接受第 {count} 组（建议 ≥ {minimum} 组）"
        if reset_count:
            summary = (
                f"角点检测器已切换，旧的 {reset_count} 组已清除；" + summary
            )
        if coverage["missing"]:
            summary += (
                f"；区域覆盖 {coverage['covered']}/9（要求 ≥{coverage['required']}），"
                f"尚缺：{'、'.join(coverage['missing'])}"
            )
        else:
            summary += "；区域覆盖 9/9。"
        summary += (
            f"；质心散布 {diversity['centroid_spread_ratio']:.2f}/"
            f"{_MIN_CENTROID_SPREAD_RATIO:.2f}；远近跨度 "
            f"{image_geometry['projected_scale_span_ratio']:.2f}/"
            f"{_MIN_PROJECTED_SCALE_SPAN_RATIO:.2f}；倾斜跨度 "
            f"{image_geometry['projective_shape_span']:.2f}/"
            f"{_MIN_PROJECTIVE_SHAPE_SPAN:.2f}。"
        )
        if (
            count >= minimum
            and coverage["covered"] >= coverage["required"]
            and diversity["centroid_spread_ratio"] >= _MIN_CENTROID_SPREAD_RATIO
            and image_geometry["projected_scale_span_ratio"]
            >= _MIN_PROJECTED_SCALE_SPAN_RATIO
            and image_geometry["projective_shape_span"]
            >= _MIN_PROJECTIVE_SHAPE_SPAN
        ):
            summary += " 已满足组数、区域、远近与倾斜要求，可点击“完成标定并保存”。"
        if count >= _MAX_AUTO_PAIRS:
            self.continuous.set(False)
            summary += f" 已达自动采集上限 {_MAX_AUTO_PAIRS} 组。"
        try:
            checkpoint = self._save_pair_checkpoint()
        except (OSError, ValueError) as error:
            checkpoint = None
            summary += f" 照片组自动保存失败：{error}"
        self.message.set(summary)
        log_event(
            get_logger("calibration_wizard"),
            "wizard_pair_accepted",
            count=count,
            automatic=automatic,
            left_zone=left_obs.centroid_zone,
            right_zone=aligned_right.centroid_zone,
            right_orientation_flipped=orientation_flipped,
            median_disparity_px=float(
                np.median(left_obs.corners_px[:, 0] - aligned_right.corners_px[:, 0])
            ),
            projected_scale_px=float(
                _board_shape_signature(left_obs, self._pattern())[0]
            ),
            image_pose_geometry=image_geometry,
            detector=left_obs.detector,
            checkpoint_path=checkpoint,
            capture_session=(
                self.capture_archive.session_path
                if self.capture_archive is not None
                else None
            ),
            capture_id=capture_id,
        )

    def stop(self) -> None:
        # Also interrupts a running device sweep between indices.
        self._probe_cancelled.set()
        self._start_after_probe = False
        if self.after_id is not None:
            try:
                self.window.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
        if self.session is not None:
            self.session.close()
            self.session = None

    def capture_pair(self) -> None:
        try:
            if self._opening:
                self.message.set("相机仍在后台打开，请稍候…")
                return
            if self.session is None:
                self.start()
                if self.session is None:
                    self.message.set("相机正在后台打开；打开后请再次点击“抓拍一组”。")
                return
            pair = self.session.read_pair()
            if np.array_equal(pair.left, pair.right):
                raise ChessboardCalibrationError(
                    "左右画面完全相同，可能仍在预热或当前不是双目输出。"
                )
            max_sync = self._float_value(self.max_sync, "同步容差", positive=False)
            if pair.sync_delta_ms > max_sync:
                raise ChessboardCalibrationError(
                    f"左右抓拍时间差 {pair.sync_delta_ms:.3f} ms 超过同步容差 {max_sync:g} ms。"
                )
            pattern = self._pattern()
            observations = []
            for role, frame in (("left", pair.left), ("right", pair.right)):
                observation = detect_board_corners(
                    frame,
                    pattern=pattern,
                    use_sb=role not in self._sb_disabled_roles,
                    on_cv_error=lambda detector_stage, error, current_role=role, current_frame=frame: self._log_detector_cv_error(
                        current_role, current_frame, detector_stage, error
                    ),
                )
                if observation is None:
                    raise ChessboardCalibrationError(
                        f"{role} 目未检出完整棋盘格；请让整个棋盘格平整进入画面后重试。"
                    )
                observations.append(observation)
            if observations[0].detector != observations[1].detector:
                raise ChessboardCalibrationError(
                    "左右目角点检测器刚完成同步切换；请保持棋盘格稳定后重新抓拍"
                )
            try:
                threshold = float(self.sharpness_gate.get())
            except ValueError:
                threshold = _DEFAULT_SHARPNESS_GATE
            if threshold > 0 and min(
                observations[0].sharpness, observations[1].sharpness
            ) < threshold:
                raise ChessboardCalibrationError(
                    f"清晰度不足（左 {observations[0].sharpness:.0f} / "
                    f"右 {observations[1].sharpness:.0f} < {threshold:g}）；"
                    "请增加光照或放慢移动；确要忽略请把“清晰度门限”改为 0。"
                )
            self._accept_pair(
                observations[0],
                observations[1],
                automatic=False,
                raw_left=pair.left,
                raw_right=pair.right,
            )
        except Exception as error:  # native cv2 errors included, never silent
            from .logging_config import get_logger, log_event

            log_event(
                get_logger("calibration_wizard"),
                "wizard_pair_rejected",
                pairs=len(self.pairs),
                error=str(error),
            )
            self.app.messagebox.showerror("抓拍失败", str(error), parent=self.window)

    def _refresh_pairs(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for index, (left, right) in enumerate(self.pairs, start=1):
            self.tree.insert(
                "",
                "end",
                iid=str(index - 1),
                values=(index, f"区域 {left.centroid_zone}", f"区域 {right.centroid_zone}"),
            )

    def remove_last(self) -> None:
        if self.pairs:
            self.pairs.pop()
            capture_id = (
                self._pair_capture_ids.pop() if self._pair_capture_ids else None
            )
            if capture_id is not None and self.capture_archive is not None:
                self.capture_archive.mark_status(
                    [capture_id],
                    status="excluded_manual",
                    reason="操作员移除上一组",
                )
            self._last_left_observation = self.pairs[-1][0] if self.pairs else None
            self._refresh_pairs()
            try:
                self._save_pair_checkpoint()
                checkpoint_note = "；自动保存已更新"
            except (OSError, ValueError) as error:
                checkpoint_note = f"；自动保存失败：{error}"
            self.message.set(
                f"已移除上一组，剩余 {len(self.pairs)} 组{checkpoint_note}。"
            )

    def remove_selected(self) -> None:
        selected = sorted(
            {int(item) for item in self.tree.selection()}, reverse=True
        )
        if not selected:
            self.message.set("请先在照片组表格中选择需要删除的组。")
            return
        display_indices = [index + 1 for index in reversed(selected)]
        capture_ids: list[int] = []
        for index in selected:
            if 0 <= index < len(self.pairs):
                self.pairs.pop(index)
                if index < len(self._pair_capture_ids):
                    capture_id = self._pair_capture_ids.pop(index)
                    if capture_id is not None:
                        capture_ids.append(capture_id)
        if capture_ids and self.capture_archive is not None:
            self.capture_archive.mark_status(
                capture_ids,
                status="excluded_manual",
                reason="操作员移除选中组",
            )
        self._last_left_observation = self.pairs[-1][0] if self.pairs else None
        self._refresh_pairs()
        try:
            self._save_pair_checkpoint()
            checkpoint_note = "；自动保存已更新"
        except (OSError, ValueError) as error:
            checkpoint_note = f"；自动保存失败：{error}"
        self.message.set(
            "已移除选中组 "
            + "、".join(str(value) for value in display_indices)
            + f"，剩余 {len(self.pairs)} 组{checkpoint_note}。"
        )

    def show_log(self) -> None:
        from .log_viewer import LogViewerDialog

        LogViewerDialog(self.app, parent=self.window)

    def finish(self) -> None:
        eye_size: tuple[int, int] | None = None
        square_mm: float | None = None
        solve_options: dict[str, Any] = {}
        try:
            operator = str(self.operator.get()).strip()
            maximum_rms = self._float_value(self.max_rms, "允许RMS")
            minimum_pairs = self._int_value(self.min_pairs, "最少组数", minimum=3)
            square_mm = self._float_value(self.measured_square_mm, "打印后实测格边长")
            maximum_sync = self._float_value(self.max_sync, "同步容差", positive=False)
            eye_size = self._eye_size()
            layout = self._layout()
            hardware_specs = self._hardware_specs()
            expected_baseline = hardware_specs["expected_baseline_mm"]
            solve_options = self._current_solve_options()
            if self.capture_archive is not None:
                self.capture_archive.update_session(
                    self._capture_session_metadata()
                )
            result = solve_stereo_calibration(
                pairs=list(self.pairs),
                image_size=eye_size,
                square_mm=square_mm,
                pattern=self._pattern(),
                max_reprojection_rms_px=maximum_rms,
                min_pairs=minimum_pairs,
                operator=operator,
                max_sync_delta_ms=maximum_sync,
                layout=layout,
                expected_baseline_mm=(expected_baseline or None),
            )
            if not result.validated:
                from .logging_config import get_logger, log_event

                diagnostic_path: Path | None = None
                diagnostic_error = ""
                try:
                    diagnostic_path = write_calibration_diagnostic(
                        list(self.pairs),
                        result,
                        image_size=eye_size,
                        pattern=self._pattern(),
                        square_mm=square_mm,
                        metadata_extra={
                            "session_path": (
                                str(self.capture_archive.session_path)
                                if self.capture_archive is not None
                                else ""
                            ),
                            "pair_capture_ids": list(self._pair_capture_ids),
                        },
                    )
                except (OSError, ValueError) as error:
                    diagnostic_error = f"{type(error).__name__}: {error}"
                log_event(
                    get_logger("calibration_wizard"),
                    "wizard_solve_rejected",
                    reasons=result.rejection_reasons,
                    pair_count=result.pair_count,
                    stereo_rms_px=result.stereo_rms_px,
                    left_rms_px=result.left_rms_px,
                    right_rms_px=result.right_rms_px,
                    zones=sorted(zone.centroid_zone for zone, _right in self.pairs),
                    solve_audit=result.audit,
                    image_size_px=list(eye_size),
                    pattern_inner_corners=list(self._pattern()),
                    measured_square_mm=square_mm,
                    maximum_rms_px=maximum_rms,
                    diagnostic_path=diagnostic_path,
                    diagnostic_error=diagnostic_error,
                )
                diagnostic_note = (
                    f"\n诊断数据：{diagnostic_path}" if diagnostic_path is not None else ""
                )
                discarded_indices = [
                    int(value)
                    for value in result.audit.get("discarded_pair_indices", [])
                ]
                if discarded_indices:
                    discarded = set(discarded_indices)
                    discarded_capture_ids = [
                        capture_id
                        for index, capture_id in enumerate(
                            self._pair_capture_ids, start=1
                        )
                        if index in discarded and capture_id is not None
                    ]
                    if self.capture_archive is not None:
                        self.capture_archive.mark_status(
                            discarded_capture_ids,
                            status="excluded_solver",
                            reason="标定求解判定为异常组",
                        )
                    self.pairs = [
                        pair
                        for index, pair in enumerate(self.pairs, start=1)
                        if index not in discarded
                    ]
                    self._pair_capture_ids = [
                        capture_id
                        for index, capture_id in enumerate(
                            self._pair_capture_ids, start=1
                        )
                        if index not in discarded
                    ]
                    self._last_left_observation = (
                        self.pairs[-1][0] if self.pairs else None
                    )
                    self._refresh_pairs()
                    try:
                        self._save_pair_checkpoint()
                    except (OSError, ValueError) as error:
                        diagnostic_note += f"\n保留组自动保存失败：{error}"
                    diagnostic_note += (
                        "\n已自动移除不合格照片组："
                        + "、".join(str(value) for value in discarded_indices)
                        + "；可保持其余组并补拍新的近/中/远和倾斜姿态。"
                    )
                self.message.set(
                    "标定未通过门禁，未保存：\n· "
                    + "\n· ".join(result.rejection_reasons)
                    + diagnostic_note
                )
                self.app.messagebox.showwarning(
                    "标定未通过",
                    "请根据提示补充照片组后重试：\n· "
                    + "\n· ".join(result.rejection_reasons)
                    + diagnostic_note,
                    parent=self.window,
                )
                return
            chessboard = {
                "square_mm": square_mm,
                "columns": int(self.columns.get()),
                "rows": int(self.rows.get()),
                "dpi": int(self.profile.get("chessboard", {}).get("dpi", 300)),
            }
            wizard_section = {
                "operator": operator,
                "max_reprojection_rms_px": maximum_rms,
                "min_pairs": minimum_pairs,
                "expected_baseline_mm": expected_baseline,
                "nominal_fov_deg": hardware_specs["nominal_fov_deg"],
                "nominal_focal_length_mm": hardware_specs[
                    "nominal_focal_length_mm"
                ],
            }
            path = save_wizard_result(
                result,
                profile_path=self.profile_path,
                extras={
                    "camera": {
                        "layout": layout,
                        "left_index": self._indices()[0],
                        "right_index": self._indices()[1],
                    },
                    "chessboard": chessboard,
                    "wizard": wizard_section,
                },
            )
            discarded_indices = [
                int(value)
                for value in result.audit.get("discarded_pair_indices", [])
            ]
            (
                retained_pairs,
                retained_capture_ids,
                discarded_capture_ids,
            ) = _retain_successful_calibration_pairs(
                list(self.pairs),
                list(self._pair_capture_ids),
                discarded_indices,
            )
            self.pairs = retained_pairs
            self._pair_capture_ids = retained_capture_ids
            self._last_left_observation = self.pairs[-1][0] if self.pairs else None
            self._refresh_pairs()
            checkpoint_path: Path | None = None
            checkpoint_error = ""
            try:
                checkpoint_path = self._save_pair_checkpoint()
            except (OSError, ValueError) as error:
                checkpoint_error = f"{type(error).__name__}: {error}"
            if self.capture_archive is not None:
                self.capture_archive.update_session(
                    {
                        "result": {
                            "calibration_id": result.calibration["calibration_id"],
                            "calibration_path": str(path),
                            "stereo_rms_px": result.stereo_rms_px,
                            "baseline_mm": result.calibration["baseline_mm"],
                            "used_capture_ids": [
                                value
                                for value in retained_capture_ids
                                if value is not None
                            ],
                            "discarded_capture_ids": discarded_capture_ids,
                        }
                    }
                )
                self.capture_archive.mark_status(
                    discarded_capture_ids,
                    status="excluded_solver",
                    reason="标定求解判定为异常组",
                )
                self.capture_archive.mark_status(
                    [
                        value
                        for value in retained_capture_ids
                        if value is not None
                    ],
                    status="used_in_saved_calibration",
                    reason=result.calibration["calibration_id"],
                )
            checkpoint_note = (
                f"\n可恢复照片组已保存：{checkpoint_path}。"
                if checkpoint_path is not None
                else f"\n可恢复照片组保存失败：{checkpoint_error}"
            )
            self.message.set(
                f"标定完成并已保存：{path}\n"
                f"双目 RMS {result.stereo_rms_px:.3f} px，左右目 "
                f"{result.left_rms_px:.3f}/{result.right_rms_px:.3f} px，"
                f"基线 {result.calibration['baseline_mm']:.2f} mm，共 {result.pair_count} 组。\n"
                f"右目画面自动修正：{result.audit.get('right_frame_transform', 'none')}。\n"
                + ("基础立面模式无需二维码；请重新抓拍并评估。"
                   if getattr(self.owner, "analysis_mode", None) in {"elevation_depth", "elevation_auto"}
                   else "注意：尚未配准 CAD——请点击“二维码定位”（或设置相机方向），然后保存工作台配置。")
                + checkpoint_note
            )
            if self.owner is not None and not getattr(self.owner, "closed", False):
                self.owner.fields["calibration"].set(str(path))
                self.owner.calibration_override = None
                self.owner.pose_adjustment = {"mode": "keep"}
                self.owner.message.set(
                    "棋盘格标定已保存并选中；请双目抓拍并评估。"
                    if getattr(self.owner, "analysis_mode", None) in {"elevation_depth", "elevation_auto"}
                    else "棋盘格标定已保存并选中；请继续“二维码定位”完成 CAD 配准。"
                )
                if hasattr(self.owner, "refresh_calibration_status"):
                    from .workbench_profile import load_profile

                    self.owner.profile, self.owner.profile_problem = load_profile()
                    self.owner.refresh_calibration_status()
            from .logging_config import get_logger, log_event

            log_event(
                get_logger("calibration_wizard"),
                "wizard_calibration_saved",
                calibration_id=result.calibration["calibration_id"],
                stereo_rms_px=result.stereo_rms_px,
                baseline_mm=result.calibration["baseline_mm"],
                pair_count=result.pair_count,
                discarded_pair_indices=discarded_indices,
                discarded_capture_ids=discarded_capture_ids,
                retained_capture_ids=[
                    value for value in retained_capture_ids if value is not None
                ],
                checkpoint_path=checkpoint_path,
                checkpoint_error=checkpoint_error,
                right_frame_transform=result.audit.get(
                    "right_frame_transform", "none"
                ),
                capture_session=(
                    self.capture_archive.session_path
                    if self.capture_archive is not None
                    else None
                ),
            )
        except Exception as error:  # native cv2 errors included, never silent
            from .logging_config import get_logger, log_event

            diagnostic_path: Path | None = None
            diagnostic_error = ""
            if self.pairs:
                try:
                    diagnostic_path = write_calibration_diagnostic(
                        list(self.pairs),
                        None,
                        image_size=eye_size or self._eye_size(),
                        pattern=self._pattern(),
                        square_mm=(
                            square_mm
                            if square_mm is not None
                            else self._float_value(
                                self.measured_square_mm, "打印后实测格边长"
                            )
                        ),
                        solve_error=f"{type(error).__name__}: {error}",
                        solve_options=solve_options,
                        metadata_extra={
                            "session_path": (
                                str(self.capture_archive.session_path)
                                if self.capture_archive is not None
                                else ""
                            ),
                            "pair_capture_ids": list(self._pair_capture_ids),
                        },
                    )
                except Exception as diagnostic_exception:
                    diagnostic_error = (
                        f"{type(diagnostic_exception).__name__}: {diagnostic_exception}"
                    )
            log_event(
                get_logger("calibration_wizard"),
                "wizard_solve_failed",
                error_type=type(error).__name__,
                error=str(error),
                pair_count=len(self.pairs),
                diagnostic_path=diagnostic_path,
                diagnostic_error=diagnostic_error,
            )
            diagnostic_note = (
                f"\n诊断数据：{diagnostic_path}" if diagnostic_path is not None else ""
            )
            self.app.messagebox.showerror(
                "标定失败", str(error) + diagnostic_note, parent=self.window
            )

    def close(self) -> None:
        self._closing = True
        self.stop()
        if self.open_after_id is not None:
            try:
                self.window.after_cancel(self.open_after_id)
            except Exception:
                pass
            self.open_after_id = None
        self.window.destroy()


__all__ = [
    "BoardObservation",
    "CalibrationCaptureArchive",
    "CHESS_ID_MARKER",
    "ChessboardCalibrationError",
    "CalibrationWizardError",
    "ChessboardWizardDialog",
    "Rectifier",
    "WizardResult",
    "align_pair_orientation",
    "build_rectification_recipe",
    "calibration_pose_report",
    "calibrate_stereo_from_folders",
    "detect_board_corners",
    "observation_is_novel",
    "pose_diversity_report",
    "pose_is_novel",
    "preferred_stream_mode",
    "printable_chessboard_png",
    "projective_pose_report",
    "rectifier_for_calibration",
    "rectification_quality_report",
    "read_calibration_diagnostic",
    "reusable_calibration_pairs",
    "replay_calibration_diagnostic",
    "save_wizard_result",
    "solve_stereo_calibration",
    "transform_board_observation",
    "write_calibration_diagnostic",
    "write_calibration_checkpoint",
    "write_printable_chessboard_png",
    "write_calibration",
]
