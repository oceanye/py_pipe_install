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
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

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


class ChessboardCalibrationError(ValueError):
    """Raised when the wizard cannot produce a trustworthy calibration."""


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


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoardObservation:
    """One accepted chessboard detection in canonical corner order."""

    corners_px: np.ndarray
    sharpness: float
    centroid_zone: tuple[int, int]

    @property
    def centroid_px(self) -> np.ndarray:
        return self.corners_px.mean(axis=0)


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
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    height, width = gray.shape[:2]
    corners: np.ndarray | None = None
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_NORMALIZE_IMAGE
        for extra in ("CALIB_CB_ACCURACY", "CALIB_CB_LARGER"):
            if hasattr(cv2, extra):
                flags |= getattr(cv2, extra)
        found, candidate = cv2.findChessboardCornersSB(gray, pattern, flags)
        if found:
            corners = candidate
    if corners is None:
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
    if corners is None:
        return None
    ordered = _canonicalize_corner_order(corners, pattern)
    centroid = ordered.mean(axis=0)
    zone = (
        min(2, max(0, int(centroid[0] / max(1, width) * 3))),
        min(2, max(0, int(centroid[1] / max(1, height) * 3))),
    )
    return BoardObservation(
        corners_px=ordered,
        sharpness=_laplacian_sharpness(gray),
        centroid_zone=zone,
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
    flipped = flat.reshape(rows, columns, 2)[::-1, ::-1].reshape(-1, 2)
    as_is = float(np.linalg.norm(flat[0] - left.corners_px[0]))
    rotated = float(np.linalg.norm(flipped[0] - left.corners_px[0]))
    if rotated < as_is:
        return BoardObservation(flipped.copy(), right.sharpness, right.centroid_zone)
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
) -> WizardResult:
    """Calibrate from accepted pairs and build the rectified-rig contract."""
    if not isinstance(operator, str) or not _OPERATOR_PATTERN.match(operator):
        raise ChessboardCalibrationError("操作员代号只能包含字母、数字、点、下划线和连字符（≤32 字符）")
    if max_sync_delta_ms < 0:
        raise ChessboardCalibrationError("max_sync_delta_ms 不能为负")
    if not pairs:
        raise ChessboardCalibrationError("没有可用的棋盘格照片对")

    object_points = _object_points(pattern, float(square_mm))
    aligned = [
        (pair[0], align_pair_orientation(pair[0], pair[1], pattern=pattern))
        for pair in pairs
    ]
    left_points = [pair[0].corners_px.reshape(-1, 1, 2).astype(np.float32) for pair in aligned]
    right_points = [pair[1].corners_px.reshape(-1, 1, 2).astype(np.float32) for pair in aligned]
    object_lists = [object_points] * len(pairs)
    size = (int(image_size[0]), int(image_size[1]))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-6)

    left_rms, K1, D1, _rvecs_l, _tvecs_l = cv2.calibrateCamera(
        object_lists, left_points, size, None, None, flags=cv2.CALIB_FIX_K3, criteria=criteria
    )
    right_rms, K2, D2, _rvecs_r, _tvecs_r = cv2.calibrateCamera(
        object_lists, right_points, size, None, None, flags=cv2.CALIB_FIX_K3, criteria=criteria
    )
    stereo_rms, _K1, _D1, _K2, _D2, R, T, _E, _F, per_view = cv2.stereoCalibrate(
        object_lists,
        left_points,
        right_points,
        K1,
        D1,
        K2,
        D2,
        size,
        np.eye(3, dtype=np.float64),
        np.zeros((3, 1), dtype=np.float64),
        None,
        None,
        None,
        flags=cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_FIX_K3,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-9),
    )
    solve_mode = "fix_intrinsics"
    if stereo_rms > max_reprojection_rms_px:
        joint_rms, _K1j, _D1j, _K2j, _D2j, Rj, Tj, _Ej, _Fj, per_view_j = cv2.stereoCalibrate(
            object_lists,
            left_points,
            right_points,
            K1,
            D1,
            K2,
            D2,
            size,
            np.eye(3, dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            None,
            None,
            None,
            flags=cv2.CALIB_FIX_K3,
            criteria=criteria,
        )
        if joint_rms < stereo_rms:
            stereo_rms, R, T, per_view, solve_mode = joint_rms, Rj, Tj, per_view_j, "joint_intrinsics"

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
    rectified_intrinsic = np.asarray(P1[:, :3], dtype=np.float64)
    k_identical = bool(np.allclose(P1[:, :3], P2[:, :3], atol=1e-6))
    baseline = float(np.linalg.norm(np.asarray(T, dtype=np.float64)))

    audit = {
        "mode": "chessboard_stereo_wizard",
        "solve_mode": solve_mode,
        "pattern_inner_corners": list(pattern),
        "square_mm": float(square_mm),
        "pair_count": len(pairs),
        "image_size_px": [size[0], size[1]],
        "stereo_rms_px": float(stereo_rms),
        "left_rms_px": float(left_rms),
        "right_rms_px": float(right_rms),
        "per_view_rms_px": [float(value) for value in np.asarray(per_view).reshape(-1)],
        "baseline_mm": baseline,
        "p1_p2_k_identical": k_identical,
        "rectify_alpha": 0.0,
        "diversity": pose_diversity_report([pair[0] for pair in pairs], image_size=size),
        "layout": layout,
        "max_sync_delta_ms": float(max_sync_delta_ms),
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
    if diversity["centroid_spread_ratio"] < 0.15:
        reasons.append(
            f"棋盘格位置变化不足：质心散布 {diversity['centroid_spread_ratio']:.2f}（要求 ≥ 0.15），请覆盖画面四角和中心"
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
    if not _BASELINE_RANGE_MM[0] <= baseline <= _BASELINE_RANGE_MM[1]:
        reasons.append(f"基线 {baseline:.2f} mm 超出合理范围 {_BASELINE_RANGE_MM[0]:g}–{_BASELINE_RANGE_MM[1]:g} mm")
    if not k_identical:
        reasons.append("stereoRectify 后左右投影矩阵 K 不一致，无法满足极线校正契约")

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

    validated = not reasons
    calibration["validated"] = validated
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
    )
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
        unknown = sorted(set(extras) - {"chessboard", "wizard"})
        if unknown:
            raise ChessboardCalibrationError(f"不支持的附加配置段落：{unknown}")
        updates.update(copy.deepcopy(dict(extras)))
    update_profile(updates, path=profile_path)
    return calibration_path


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
        self.max_rms = tk.StringVar(value=str(wizard.get("max_reprojection_rms_px", 0.5)))
        self.min_pairs = tk.StringVar(value=str(wizard.get("min_pairs", 10)))
        self.max_sync = tk.StringVar(value="1.0")
        self.eye_width = tk.StringVar(value="1280")
        self.eye_height = tk.StringVar(value="720")
        self.mode = tk.StringVar(value=str(camera.get("layout", "side_by_side_left_right")))
        self.left_index = tk.StringVar(value=str(camera.get("left_index", 0)))
        self.right_index = tk.StringVar(value=str(camera.get("right_index", 1)))
        self.session: Any | None = None
        self.after_id: str | None = None
        self.open_after_id: str | None = None
        self._opening = False
        self._closing = False
        self._probe_cancelled = threading.Event()
        self._tick = 0
        self.preview_images: list[Any] = []
        self.pairs: list[tuple[BoardObservation, BoardObservation]] = []
        self.message = tk.StringVar(
            value=(
                "步骤：生成并 100% 打印棋盘格 → 量取实测格边长 → 打开预览，"
                "持棋盘格覆盖画面各区域抓拍 ≥10 组 → 完成标定。标定后仍需一次二维码定位。"
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
        self.sync_gate_label = tk.StringVar(value="标定阶段为静态棋盘，同步容差只用于记录。")
        ttk.Label(camera_row, textvariable=self.sync_gate_label, foreground="#4A6178").pack(
            side="left", padx=(10, 0)
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
        ttk.Button(actions, text="抓拍一组", command=self.capture_pair).pack(fill="x", pady=3)
        ttk.Button(actions, text="移除上一组", command=self.remove_last).pack(fill="x", pady=3)
        ttk.Button(actions, text="完成标定并保存", command=self.finish).pack(fill="x", pady=(14, 3))
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
        self._probe_cancelled.clear()
        self.message.set("正在检测视频设备索引 0 / 5…（打开超时的设备会自动跳过；点“停止”可中断）")
        self.left_status.set("左目：检测设备中…")
        self.right_status.set("右目：检测设备中…")
        log_event(get_logger("calibration_wizard"), "wizard_device_probe_start")
        outcome: dict[str, Any] = {}
        progress_state: dict[str, Any] = {"index": -1, "status": ""}
        run_in_background(
            lambda: probe_video_devices(
                maximum_index=5,
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
        try:
            if error is not None:
                raise error
            devices = result or []
            if not devices:
                detail = "检测已停止，未得到完整结果。" if cancelled else ""
                raise ChessboardCalibrationError(
                    f"未检测到 OpenCV 可打开的视频设备。{detail}"
                )
            self.left_index.set(str(devices[0]["index"]))
            if len(devices) > 1:
                self.right_index.set(str(devices[1]["index"]))
            width = int(devices[0]["width"])
            height = int(devices[0]["height"])
            if self._layout() in {"side_by_side_left_right", "side_by_side_right_left"}:
                width = max(1, width // 2)
            self.eye_width.set(str(width))
            self.eye_height.set(str(height))
            summary = "；".join(
                f"索引 {item['index']}：{item['width']}×{item['height']}" for item in devices
            )
            prefix = "检测已停止；" if cancelled else ""
            self.message.set(f"{prefix}检测到 {len(devices)} 个可用设备：{summary}")
            self.left_status.set("左目：未打开预览")
            self.right_status.set("右目：未打开预览")
        except Exception as error:  # surfaced to the operator, never silent
            self.app.messagebox.showerror("相机检测失败", str(error), parent=self.window)
            self.message.set(f"相机检测失败：{error}")

    def start(self) -> bool:
        from .logging_config import get_logger, log_event
        from .stereo_camera import StereoCameraSession, run_in_background

        if self._opening or self.open_after_id is not None:
            self.message.set("正在处理上一个相机任务，请稍候…")
            return False
        try:
            layout = self._layout()
            left, right = self._indices()
            eye_width, eye_height = self._eye_size()
        except ValueError as error:
            self.app.messagebox.showerror("相机打开失败", str(error), parent=self.window)
            return False
        self.stop()
        self._probe_cancelled.clear()
        self._opening = True
        self.message.set("正在打开相机；部分设备在 DirectShow 下需要数秒，请稍候…")
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
            log_event(logger, "wizard_camera_open_failed", error=str(error))
            self.message.set(f"相机打开失败：{error}")
            self.left_status.set("左目：未打开")
            self.right_status.set("右目：未打开")
            self.app.messagebox.showerror("相机打开失败", str(error), parent=self.window)
            return
        self.session = session
        log_event(logger, "wizard_camera_open_finished")
        self.message.set("相机已打开；持棋盘格覆盖画面各区域，点击“抓拍一组”。")
        self._tick = 0
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
        try:
            pair = self.session.read_pair()
            pattern = self._pattern()
            status = {role: "未检出" for role, _frame in (("left", pair.left), ("right", pair.right))}
            displays = {"left": pair.left, "right": pair.right}
            if self._tick % 3 == 0:
                for role, frame in (("left", pair.left), ("right", pair.right)):
                    observation = detect_board_corners(frame, pattern=pattern)
                    if observation is not None:
                        display = frame.copy()
                        cv2.drawChessboardCorners(
                            display, pattern, observation.corners_px.reshape(-1, 1, 2), True
                        )
                        displays[role] = display
                        status[role] = (
                            f"已检出，清晰度 {observation.sharpness:.0f}，"
                            f"区域 {observation.centroid_zone}"
                        )
                    else:
                        status[role] = "未检出，请让整个棋盘格进入画面"
            self.preview_images = [self._photo(displays["left"]), self._photo(displays["right"])]
            self.left_preview.configure(image=self.preview_images[0])
            self.right_preview.configure(image=self.preview_images[1])
            self.left_status.set(f"左目：{status['left']}")
            self.right_status.set(f"右目：{status['right']}")
            self._tick += 1
            self.after_id = self.window.after(120, self._update_preview)
        except Exception as error:  # native cv2 errors included, never silent
            from .logging_config import get_logger, log_event

            log_event(get_logger("calibration_wizard"), "wizard_preview_failed", error=str(error))
            self.stop()
            self.app.messagebox.showerror("相机读取失败", str(error), parent=self.window)

    def stop(self) -> None:
        # Also interrupts a running device sweep between indices.
        self._probe_cancelled.set()
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
                observation = detect_board_corners(frame, pattern=pattern)
                if observation is None:
                    raise ChessboardCalibrationError(
                        f"{role} 目未检出完整棋盘格；请让整个棋盘格平整进入画面后重试。"
                    )
                observations.append(observation)
            self.pairs.append((observations[0], observations[1]))
            minimum = self._int_value(self.min_pairs, "最少组数", minimum=3)
            self.message.set(
                f"已接受 {len(self.pairs)} 组（建议 ≥ {minimum} 组）；"
                "请继续更换棋盘格位置、角度和距离，覆盖画面四角与中心。"
            )
            self._refresh_pairs()
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
            self._refresh_pairs()
            self.message.set(f"已移除上一组，剩余 {len(self.pairs)} 组。")

    def finish(self) -> None:
        try:
            operator = str(self.operator.get()).strip()
            maximum_rms = self._float_value(self.max_rms, "允许RMS")
            minimum_pairs = self._int_value(self.min_pairs, "最少组数", minimum=3)
            square_mm = self._float_value(self.measured_square_mm, "打印后实测格边长")
            maximum_sync = self._float_value(self.max_sync, "同步容差", positive=False)
            eye_size = self._eye_size()
            layout = self._layout()
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
            )
            if not result.validated:
                self.message.set(
                    "标定未通过门禁，未保存：\n· " + "\n· ".join(result.rejection_reasons)
                )
                self.app.messagebox.showwarning(
                    "标定未通过",
                    "请根据提示补充照片组后重试：\n· "
                    + "\n· ".join(result.rejection_reasons),
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
            }
            path = save_wizard_result(
                result,
                profile_path=self.profile_path,
                extras={"chessboard": chessboard, "wizard": wizard_section},
            )
            self.message.set(
                f"标定完成并已保存：{path}\n"
                f"双目 RMS {result.stereo_rms_px:.3f} px，左右目 "
                f"{result.left_rms_px:.3f}/{result.right_rms_px:.3f} px，"
                f"基线 {result.calibration['baseline_mm']:.2f} mm，共 {result.pair_count} 组。\n"
                "注意：尚未配准 CAD——请点击“二维码定位”（或设置相机方向），然后保存工作台配置。"
            )
            if self.owner is not None:
                self.owner.fields["calibration"].set(str(path))
                self.owner.calibration_override = None
                self.owner.pose_adjustment = {"mode": "keep"}
                self.owner.message.set(
                    "棋盘格标定已保存并选中；请继续“二维码定位”完成 CAD 配准。"
                )
            from .logging_config import get_logger, log_event

            log_event(
                get_logger("calibration_wizard"),
                "wizard_calibration_saved",
                calibration_id=result.calibration["calibration_id"],
                stereo_rms_px=result.stereo_rms_px,
                baseline_mm=result.calibration["baseline_mm"],
                pair_count=result.pair_count,
            )
        except Exception as error:  # native cv2 errors included, never silent
            from .logging_config import get_logger, log_event

            log_event(get_logger("calibration_wizard"), "wizard_solve_failed", error=str(error))
            self.app.messagebox.showerror("标定失败", str(error), parent=self.window)

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
    "CHESS_ID_MARKER",
    "ChessboardCalibrationError",
    "ChessboardWizardDialog",
    "Rectifier",
    "WizardResult",
    "align_pair_orientation",
    "build_rectification_recipe",
    "detect_board_corners",
    "pose_diversity_report",
    "printable_chessboard_png",
    "rectifier_for_calibration",
    "save_wizard_result",
    "solve_stereo_calibration",
    "write_printable_chessboard_png",
]
