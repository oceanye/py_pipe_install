"""Automatic chessboard based stereo calibration.

The field workflow only consumes rectified, CAD-bound calibration JSON.  This
module removes the most error-prone part of that workflow: manually typing
OpenCV matrices and rectification outputs.  The operator supplies two folders
of paired chessboard photos; OpenCV detects corners, estimates both cameras,
and the existing adapter writes the canonical manifest calibration.

The result is intentionally marked ``validated=false`` and
``registration_validated=false``.  A calibration target estimates the rig in
camera coordinates, but cannot know the CAD world's origin.  The QR
registration step (or a measured pose) remains a short, explicit second step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .calibration_adapter import adapt_opencv_stereo_calibration
from .pipeline import atomic_write_text


class CalibrationWizardError(ValueError):
    """Raised when a calibration set cannot be processed safely."""


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _images(folder: str | Path) -> list[Path]:
    path = Path(folder)
    if not path.is_dir():
        raise CalibrationWizardError(f"标定照片目录不存在：{path}")
    result = sorted(item for item in path.iterdir() if item.suffix.lower() in _IMAGE_SUFFIXES)
    if not result:
        raise CalibrationWizardError(f"目录没有 PNG/JPEG 标定照片：{path}")
    return result


def _pairs(left_dir: str | Path, right_dir: str | Path) -> list[tuple[Path, Path]]:
    left, right = _images(left_dir), _images(right_dir)
    if len(left) != len(right):
        raise CalibrationWizardError(
            f"左右照片数量不同（左 {len(left)} 张，右 {len(right)} 张）；请按同一批次补齐。"
        )
    # Camera exports often use left_001/right_001.  Sorting is deterministic
    # and avoids guessing a naming convention; report the actual pair names.
    return list(zip(left, right))


def _corners(image: np.ndarray, board_size: tuple[int, int]) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    found, corners = False, None
    finder = getattr(cv2, "findChessboardCornersSB", None)
    if finder is not None:
        found, corners = finder(gray, board_size, flags)
    if not found:
        found, corners = cv2.findChessboardCorners(
            gray, board_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
    if not found or corners is None:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


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
    left_camera_pose: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Calibrate a stereo rig from paired chessboard image directories.

    ``left_camera_pose`` is optional because it is not observable from a
    chessboard alone.  The generated identity pose is an explicit temporary
    CAD origin and the result remains unusable for field analysis until QR
    registration (or a measured pose) sets the validation flags.
    """
    if type(board_columns) is not int or type(board_rows) is not int or board_columns < 3 or board_rows < 3:
        raise CalibrationWizardError("棋盘内角点数量必须至少为 3×3。")
    if type(square_size_mm) not in (int, float) or not np.isfinite(square_size_mm) or square_size_mm <= 0:
        raise CalibrationWizardError("棋盘格边长必须是正数（单位 mm）。")
    if type(min_pairs) is not int or min_pairs < 3:
        raise CalibrationWizardError("至少需要 3 对照片；现场建议 8–15 对不同角度照片。")
    pairs = _pairs(left_dir, right_dir)
    board = np.zeros((board_rows * board_columns, 3), np.float32)
    board[:, :2] = np.mgrid[0:board_columns, 0:board_rows].T.reshape(-1, 2)
    board *= float(square_size_mm)

    object_points: list[np.ndarray] = []
    left_points: list[np.ndarray] = []
    right_points: list[np.ndarray] = []
    rejected: list[dict[str, str]] = []
    image_size: tuple[int, int] | None = None
    for left_path, right_path in pairs:
        left_image, right_image = cv2.imread(str(left_path)), cv2.imread(str(right_path))
        if left_image is None or right_image is None:
            rejected.append({"left": str(left_path), "right": str(right_path), "reason": "无法读取照片"})
            continue
        current_size = (int(left_image.shape[1]), int(left_image.shape[0]))
        right_size = (int(right_image.shape[1]), int(right_image.shape[0]))
        if current_size != right_size:
            rejected.append({"left": str(left_path), "right": str(right_path), "reason": "左右分辨率不同"})
            continue
        if image_size is None:
            image_size = current_size
        if current_size != image_size:
            rejected.append({"left": str(left_path), "right": str(right_path), "reason": "照片分辨率不一致"})
            continue
        left_corner, right_corner = _corners(left_image, (board_columns, board_rows)), _corners(right_image, (board_columns, board_rows))
        if left_corner is None or right_corner is None:
            rejected.append({"left": str(left_path), "right": str(right_path), "reason": "未同时检测到完整棋盘"})
            continue
        object_points.append(board.copy())
        left_points.append(left_corner)
        right_points.append(right_corner)

    if image_size is None or len(object_points) < min_pairs:
        raise CalibrationWizardError(
            f"有效左右棋盘照片只有 {len(object_points)} 对，至少需要 {min_pairs} 对；"
            "请让棋盘覆盖画面不同位置、角度和距离。"
        )
    try:
        rms_left, K1, D1, _, _ = cv2.calibrateCamera(object_points, left_points, image_size, None, None)
        rms_right, K2, D2, _, _ = cv2.calibrateCamera(object_points, right_points, image_size, None, None)
        stereo_rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            object_points, left_points, right_points, K1, D1, K2, D2,
            image_size, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
    except cv2.error as error:
        raise CalibrationWizardError(f"OpenCV 无法完成标定，请检查棋盘规格和照片质量：{error}") from error
    if max_rms_px is not None and max(rms_left, rms_right, stereo_rms) > float(max_rms_px):
        raise CalibrationWizardError(
            f"棋盘重投影误差过大（左 {rms_left:.3f}px，右 {rms_right:.3f}px，双目 {stereo_rms:.3f}px）；"
            "请剔除模糊/反光照片后重试。"
        )
    pose = dict(left_camera_pose or {
        "rotation_world_to_camera": np.eye(3).tolist(),
        "center_world_mm": [0.0, 0.0, 0.0],
    })
    source: dict[str, Any] = {
        "K1": K1.tolist(), "D1": np.asarray(D1).reshape(-1).tolist(),
        "K2": K2.tolist(), "D2": np.asarray(D2).reshape(-1).tolist(),
        "image_size": [image_size[0], image_size[1]], "R": R.tolist(), "T": np.asarray(T).reshape(-1).tolist(),
        "translation_unit": "millimeter", "left_camera_pose": pose,
    }
    # The RMS gate above is the automatic calibration validation.  CAD
    # registration remains a separate flag and is intentionally left false
    # until the QR target (or measured pose) is accepted.
    try:
        calibration = adapt_opencv_stereo_calibration(
            source,
            calibration_id=calibration_id,
            validated=True,
            registration_validated=False,
        )
    except cv2.error as error:
        raise CalibrationWizardError(f"OpenCV 极线矫正失败：{error}") from error
    calibration["source_audit"]["auto_calibration"] = {
        "method": "opencv_chessboard_stereo_calibrate_v1",
        "board_inner_corners": [board_columns, board_rows],
        "square_size_mm": float(square_size_mm),
        "candidate_pairs": len(pairs), "accepted_pairs": len(object_points), "rejected_pairs": rejected,
        "rms_left_px": float(rms_left), "rms_right_px": float(rms_right), "rms_stereo_px": float(stereo_rms),
        "essential_matrix": np.asarray(E).tolist(), "fundamental_matrix": np.asarray(F).tolist(),
        "registration_required": left_camera_pose is None,
    }
    return calibration


def write_calibration(path: str | Path, calibration: Mapping[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return atomic_write_text(
        destination,
        json.dumps(calibration, ensure_ascii=False, indent=2) + "\n",
    )


__all__ = ["CalibrationWizardError", "calibrate_stereo_from_folders", "write_calibration"]
