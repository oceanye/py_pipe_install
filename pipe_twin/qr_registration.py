"""Printable QR control targets and camera-to-CAD pose registration."""

from __future__ import annotations

import copy
import hashlib
import html
import json
import math
import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np


WORLD_DIRECTIONS = {
    "+X": np.asarray([1.0, 0.0, 0.0]),
    "-X": np.asarray([-1.0, 0.0, 0.0]),
    "+Y": np.asarray([0.0, 1.0, 0.0]),
    "-Y": np.asarray([0.0, -1.0, 0.0]),
    "+Z": np.asarray([0.0, 0.0, 1.0]),
    "-Z": np.asarray([0.0, 0.0, -1.0]),
}


class QrRegistrationError(ValueError):
    """Raised when a QR target cannot produce a trustworthy registration."""


@dataclass(frozen=True)
class QrPoseEstimate:
    decoded_payload: str
    marker_edge_mm: float
    corners_px: np.ndarray
    rotation_marker_to_camera: np.ndarray
    translation_marker_to_camera_mm: np.ndarray
    camera_center_marker_mm: np.ndarray
    camera_distance_mm: float
    reprojection_rms_px: float
    source_image_sha256: str


def qr_payload(marker_id: str, marker_edge_mm: float) -> str:
    marker_id = str(marker_id).strip()
    if re.fullmatch(r"[A-Za-z0-9._-]{1,64}", marker_id) is None:
        raise QrRegistrationError("二维码编号只能使用 1 到 64 个英文字母、数字、点、下划线或短横线")
    edge = _positive_finite(marker_edge_mm, "二维码编码区边长")
    if not 30.0 <= edge <= 160.0:
        raise QrRegistrationError("二维码编码区边长应在 30 到 160 mm 之间")
    return f"PIPE_TWIN_QR_V1|id={marker_id}|edge_mm={edge:.3f}"


def _positive_finite(value: Any, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise QrRegistrationError(f"{field}必须是正数")
    return float(value)


def _qr_data_matrix(payload: str) -> np.ndarray:
    try:
        encoded = cv2.QRCodeEncoder_create().encode(payload)
    except cv2.error as error:
        raise QrRegistrationError(f"无法生成二维码：{error}") from error
    if not isinstance(encoded, np.ndarray) or encoded.ndim != 2:
        raise QrRegistrationError("OpenCV 未返回有效二维码矩阵")
    data = encoded
    while (
        min(data.shape) > 1
        and np.all(data[0] == 255)
        and np.all(data[-1] == 255)
        and np.all(data[:, 0] == 255)
        and np.all(data[:, -1] == 255)
    ):
        data = data[1:-1, 1:-1]
    if data.shape[0] != data.shape[1] or data.shape[0] < 21:
        raise QrRegistrationError("二维码编码矩阵尺寸无效")
    return data


def printable_qr_svg(
    *,
    marker_id: str,
    marker_edge_mm: float,
) -> str:
    """Return an A4 SVG whose QR data square has an exact physical edge."""
    payload = qr_payload(marker_id, marker_edge_mm)
    matrix = _qr_data_matrix(payload)
    edge = float(marker_edge_mm)
    modules = int(matrix.shape[0])
    module_mm = edge / modules
    quiet_modules = 4
    total = edge + 2 * quiet_modules * module_mm
    if total > 165.0:
        raise QrRegistrationError("当前二维码内容与边长超出 A4 定位板可打印范围")
    left = (210.0 - total) / 2.0
    top = 42.0
    data_left = left + quiet_modules * module_mm
    data_top = top + quiet_modules * module_mm
    rectangles: list[str] = []
    for row in range(modules):
        for column in range(modules):
            if int(matrix[row, column]) == 0:
                rectangles.append(
                    f'<rect x="{data_left + column * module_mm:.6f}" '
                    f'y="{data_top + row * module_mm:.6f}" '
                    f'width="{module_mm:.6f}" height="{module_mm:.6f}"/>'
                )
    safe_id = html.escape(str(marker_id).strip())
    safe_payload = html.escape(payload)
    edge_text = f"{edge:.3f}"
    qr_bottom = top + total
    dimension_y = qr_bottom + 7.0
    arrow_x = 62.0
    arrow_y = qr_bottom + 32.0
    return "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<svg xmlns="http://www.w3.org/2000/svg" width="210mm" height="297mm" '
            'viewBox="0 0 210 297">',
            f"<metadata>{safe_payload}</metadata>",
            '<rect width="210" height="297" fill="white"/>',
            '<g font-family="Arial, Microsoft YaHei, sans-serif" fill="black">',
            '<text x="105" y="13" text-anchor="middle" font-size="6" font-weight="bold">'
            'PIPE TWIN 二维码定位板</text>',
            '<text x="105" y="22" text-anchor="middle" font-size="4">'
            'A4 纸按 100% / 实际大小打印，禁止“适合页面”</text>',
            f'<text x="105" y="29" text-anchor="middle" font-size="3.8">'
            f'编号 {safe_id}　二维码编码区边长 {edge_text} mm</text>',
            '</g>',
            f'<rect x="{left:.6f}" y="{top:.6f}" width="{total:.6f}" '
            f'height="{total:.6f}" fill="white"/>',
            '<g fill="black" shape-rendering="crispEdges">',
            *rectangles,
            '</g>',
            '<g stroke="#555" fill="none" stroke-width="0.45">',
            f'<line x1="{data_left:.3f}" y1="{dimension_y:.3f}" '
            f'x2="{data_left + edge:.3f}" y2="{dimension_y:.3f}"/>',
            f'<line x1="{data_left:.3f}" y1="{dimension_y - 3:.3f}" '
            f'x2="{data_left:.3f}" y2="{dimension_y + 3:.3f}"/>',
            f'<line x1="{data_left + edge:.3f}" y1="{dimension_y - 3:.3f}" '
            f'x2="{data_left + edge:.3f}" y2="{dimension_y + 3:.3f}"/>',
            f'<line x1="{arrow_x:.3f}" y1="{arrow_y:.3f}" '
            f'x2="{arrow_x + 45:.3f}" y2="{arrow_y:.3f}"/>',
            f'<path d="M {arrow_x + 45:.3f} {arrow_y:.3f} l -4 -2 l 0 4 z" fill="#555"/>',
            f'<line x1="{arrow_x:.3f}" y1="{arrow_y:.3f}" '
            f'x2="{arrow_x:.3f}" y2="{arrow_y - 18:.3f}"/>',
            f'<path d="M {arrow_x:.3f} {arrow_y - 18:.3f} l -2 4 l 4 0 z" fill="#555"/>',
            '</g>',
            '<g font-family="Arial, Microsoft YaHei, sans-serif" fill="#333" font-size="4">',
            f'<text x="105" y="{dimension_y - 2:.3f}" text-anchor="middle">'
            f'二维码编码区边长 {edge_text} mm</text>',
            f'<text x="{arrow_x + 48:.3f}" y="{arrow_y + 1.5:.3f}">纸面向右 RIGHT</text>',
            f'<text x="{arrow_x + 3:.3f}" y="{arrow_y - 16:.3f}">纸面向上 UP</text>',
            f'<text x="105" y="{arrow_y + 12:.3f}" text-anchor="middle">'
            '安装后登记二维码中心的 CAD 坐标，并登记纸面 RIGHT / UP 对应的 CAD 方向</text>',
            '</g>',
            '<g stroke="black" fill="none" stroke-width="0.4">',
            '<line x1="55" y1="270" x2="155" y2="270"/>',
            '<line x1="55" y1="266" x2="55" y2="274"/>',
            '<line x1="155" y1="266" x2="155" y2="274"/>',
            '</g>',
            '<text x="105" y="280" text-anchor="middle" '
            'font-family="Arial, Microsoft YaHei, sans-serif" font-size="4">'
            '打印后用量具检查上方校验线应为 100.000 mm，并实测二维码编码区边长</text>',
            '</svg>',
            "",
        ]
    )


def write_printable_qr_svg(
    path: str | Path,
    *,
    marker_id: str,
    marker_edge_mm: float,
) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != ".svg":
        destination = destination.with_suffix(".svg")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        printable_qr_svg(marker_id=marker_id, marker_edge_mm=marker_edge_mm),
        encoding="utf-8",
    )
    return destination


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def printable_qr_png(
    *,
    marker_id: str,
    marker_edge_mm: float,
    dpi: int = 300,
) -> bytes:
    """Return an A4 PNG with an explicit DPI and a metric QR data square."""
    payload = qr_payload(marker_id, marker_edge_mm)
    if type(dpi) is not int or not 150 <= dpi <= 1200:
        raise QrRegistrationError("PNG 打印分辨率应为 150 到 1200 DPI 的整数")
    matrix = _qr_data_matrix(payload)
    pixels_per_mm = dpi / 25.4
    page_width = round(210.0 * pixels_per_mm)
    page_height = round(297.0 * pixels_per_mm)
    data_edge = round(float(marker_edge_mm) * pixels_per_mm)
    modules = int(matrix.shape[0])
    quiet = round(4 * data_edge / modules)
    if data_edge + 2 * quiet > page_width - round(20 * pixels_per_mm):
        raise QrRegistrationError("当前二维码内容与边长超出 A4 PNG 可打印范围")

    page = np.full((page_height, page_width), 255, dtype=np.uint8)
    data_left = (page_width - data_edge) // 2
    data_top = round(48.0 * pixels_per_mm)
    for row in range(modules):
        y1 = data_top + round(row * data_edge / modules)
        y2 = data_top + round((row + 1) * data_edge / modules)
        for column in range(modules):
            if int(matrix[row, column]) != 0:
                continue
            x1 = data_left + round(column * data_edge / modules)
            x2 = data_left + round((column + 1) * data_edge / modules)
            page[y1:y2, x1:x2] = 0

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(page, "PIPE TWIN QR REGISTRATION", (round(29 * pixels_per_mm), round(12 * pixels_per_mm)), font, 1.25, 0, 3, cv2.LINE_AA)
    cv2.putText(page, "A4 / 300 DPI / PRINT 100% ACTUAL SIZE", (round(25 * pixels_per_mm), round(22 * pixels_per_mm)), font, 0.78, 0, 2, cv2.LINE_AA)
    cv2.putText(page, f"ID: {marker_id}", (round(25 * pixels_per_mm), round(30 * pixels_per_mm)), font, 0.68, 0, 2, cv2.LINE_AA)
    cv2.putText(page, f"QR DATA SQUARE: {float(marker_edge_mm):.3f} mm", (round(25 * pixels_per_mm), round(37 * pixels_per_mm)), font, 0.68, 0, 2, cv2.LINE_AA)

    dimension_y = data_top + data_edge + round(10 * pixels_per_mm)
    cv2.line(page, (data_left, dimension_y), (data_left + data_edge, dimension_y), 70, 2)
    tick = round(3 * pixels_per_mm)
    cv2.line(page, (data_left, dimension_y - tick), (data_left, dimension_y + tick), 70, 2)
    cv2.line(page, (data_left + data_edge, dimension_y - tick), (data_left + data_edge, dimension_y + tick), 70, 2)

    arrow_origin = (round(65 * pixels_per_mm), dimension_y + round(23 * pixels_per_mm))
    cv2.arrowedLine(page, arrow_origin, (arrow_origin[0] + round(42 * pixels_per_mm), arrow_origin[1]), 70, 3, tipLength=0.08)
    cv2.arrowedLine(page, arrow_origin, (arrow_origin[0], arrow_origin[1] - round(17 * pixels_per_mm)), 70, 3, tipLength=0.13)
    cv2.putText(page, "RIGHT", (arrow_origin[0] + round(45 * pixels_per_mm), arrow_origin[1] + 8), font, 0.72, 40, 2, cv2.LINE_AA)
    cv2.putText(page, "UP", (arrow_origin[0] + 12, arrow_origin[1] - round(17 * pixels_per_mm)), font, 0.72, 40, 2, cv2.LINE_AA)

    scale_length = round(100.0 * pixels_per_mm)
    scale_left = (page_width - scale_length) // 2
    scale_y = page_height - round(25 * pixels_per_mm)
    cv2.line(page, (scale_left, scale_y), (scale_left + scale_length, scale_y), 0, 3)
    cv2.line(page, (scale_left, scale_y - tick), (scale_left, scale_y + tick), 0, 3)
    cv2.line(page, (scale_left + scale_length, scale_y - tick), (scale_left + scale_length, scale_y + tick), 0, 3)
    cv2.putText(page, "CHECK LINE: 100.000 mm", (scale_left + round(19 * pixels_per_mm), scale_y - round(5 * pixels_per_mm)), font, 0.72, 0, 2, cv2.LINE_AA)

    ok, encoded = cv2.imencode(".png", page, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise QrRegistrationError("OpenCV 无法编码二维码 PNG")
    png = bytes(encoded)
    # PNG pHYs stores pixels per metre. Insert it immediately after IHDR so
    # print software can map pixels to physical A4 dimensions at 100%.
    pixels_per_metre = round(dpi / 0.0254)
    physical = _png_chunk(
        b"pHYs",
        struct.pack(">IIB", pixels_per_metre, pixels_per_metre, 1),
    )
    metadata = _png_chunk(
        b"tEXt",
        (
            f"Description\x00{payload};dpi={dpi};data_edge_px={data_edge};"
            f"page_px={page_width}x{page_height}"
        ).encode("latin-1"),
    )
    return png[:33] + physical + metadata + png[33:]


def write_printable_qr_png(
    path: str | Path,
    *,
    marker_id: str,
    marker_edge_mm: float,
    dpi: int = 300,
) -> Path:
    destination = Path(path)
    if destination.suffix.lower() != ".png":
        destination = destination.with_suffix(".png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(
        printable_qr_png(
            marker_id=marker_id,
            marker_edge_mm=marker_edge_mm,
            dpi=dpi,
        )
    )
    return destination


def _square_object_points(edge_mm: float) -> np.ndarray:
    half = edge_mm / 2.0
    # Required order for SOLVEPNP_IPPE_SQUARE: top-left, top-right,
    # bottom-right, bottom-left. Marker +X is print-right, +Y is print-up,
    # and +Z points out of the printed front face.
    return np.asarray(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float64,
    )


def estimate_square_pose(
    *,
    corners_px: Any,
    marker_edge_mm: float,
    intrinsic: Any,
    decoded_payload: str,
    source_image_sha256: str,
) -> QrPoseEstimate:
    """Estimate one planar square pose and select the positive-depth solution."""
    edge = _positive_finite(marker_edge_mm, "二维码编码区边长")
    corners = np.asarray(corners_px, dtype=np.float64).reshape(-1, 2)
    camera_matrix = np.asarray(intrinsic, dtype=np.float64)
    if corners.shape != (4, 2) or not np.all(np.isfinite(corners)):
        raise QrRegistrationError("二维码角点必须是四个有限像素坐标")
    if camera_matrix.shape != (3, 3) or not np.all(np.isfinite(camera_matrix)):
        raise QrRegistrationError("相机内参矩阵必须为有限的 3×3 矩阵")
    object_points = _square_object_points(edge)
    distortion = np.zeros(5, dtype=np.float64)
    try:
        count, rotations, translations, _errors = cv2.solvePnPGeneric(
            object_points,
            corners,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        iterative_ok, iterative_rotation, iterative_translation = cv2.solvePnP(
            object_points,
            corners,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
    except cv2.error as error:
        raise QrRegistrationError(f"二维码位姿计算失败：{error}") from error
    pose_vectors = list(zip(rotations[:count], translations[:count]))
    if iterative_ok:
        pose_vectors.append((iterative_rotation, iterative_translation))
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for rvec, tvec in pose_vectors:
        rotation, _ = cv2.Rodrigues(rvec)
        translation = np.asarray(tvec, dtype=np.float64).reshape(3)
        camera_points = (rotation @ object_points.T).T + translation
        if np.min(camera_points[:, 2]) <= 0:
            continue
        projected, _ = cv2.projectPoints(
            object_points, rvec, tvec, camera_matrix, distortion
        )
        residual = projected.reshape(4, 2) - corners
        rms = float(np.sqrt(np.mean(np.sum(residual * residual, axis=1))))
        candidates.append((rms, rotation, translation))
    if not candidates:
        raise QrRegistrationError("二维码位姿没有得到位于相机前方的有效解")
    rms, rotation, translation = min(candidates, key=lambda item: item[0])
    camera_center = -(rotation.T @ translation)
    return QrPoseEstimate(
        decoded_payload=decoded_payload,
        marker_edge_mm=edge,
        corners_px=corners.copy(),
        rotation_marker_to_camera=rotation,
        translation_marker_to_camera_mm=translation,
        camera_center_marker_mm=camera_center,
        camera_distance_mm=float(np.linalg.norm(translation)),
        reprojection_rms_px=rms,
        source_image_sha256=source_image_sha256,
    )


def detect_qr_pose(
    image_path: str | Path,
    *,
    expected_payload: str,
    marker_edge_mm: float,
    intrinsic: Any,
    expected_size: tuple[int, int] | None = None,
) -> QrPoseEstimate:
    path = Path(image_path)
    if not path.is_file():
        raise QrRegistrationError(f"左目二维码照片不存在：{path}")
    image_bytes = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise QrRegistrationError("左目二维码照片无法解码")
    if expected_size is not None and (image.shape[1], image.shape[0]) != expected_size:
        raise QrRegistrationError(
            f"左目照片实际为 {image.shape[1]}×{image.shape[0]}；"
            f"标定要求 {expected_size[0]}×{expected_size[1]}"
        )
    detector = cv2.QRCodeDetector()
    try:
        decoded, corners, _straight = detector.detectAndDecode(image)
    except cv2.error as error:
        raise QrRegistrationError(f"二维码检测失败：{error}") from error
    if corners is None:
        raise QrRegistrationError("左目画面中未检测到二维码；请改善照明并增大二维码成像尺寸")
    if not decoded:
        raise QrRegistrationError("检测到二维码轮廓但无法解码；请避免反光、模糊和遮挡")
    if decoded != expected_payload:
        raise QrRegistrationError(
            f"二维码内容不匹配；检测到 {decoded!r}，预期 {expected_payload!r}"
        )
    return estimate_square_pose(
        corners_px=corners,
        marker_edge_mm=marker_edge_mm,
        intrinsic=intrinsic,
        decoded_payload=decoded,
        source_image_sha256=hashlib.sha256(image_bytes).hexdigest(),
    )


def register_calibration_from_qr(
    calibration: Mapping[str, Any],
    estimate: QrPoseEstimate,
    *,
    marker_center_world_mm: Any,
    print_right_world: str,
    print_up_world: str,
    registration_validated: bool,
    max_reprojection_rms_px: float = 2.0,
) -> dict[str, Any]:
    """Place a rectified stereo rig in CAD coordinates from one QR target."""
    from .stereo_analyzer import _calibration_from_manifest

    parsed = _calibration_from_manifest(calibration)
    if not parsed.rectified:
        raise QrRegistrationError("二维码定位要求使用已极线矫正的相机输出")
    if type(registration_validated) is not bool:
        raise QrRegistrationError("registration_validated 必须是布尔值")
    max_rms = _positive_finite(max_reprojection_rms_px, "最大重投影误差")
    if estimate.reprojection_rms_px > max_rms:
        raise QrRegistrationError(
            f"二维码重投影误差 {estimate.reprojection_rms_px:.3f} px 超过 "
            f"{max_rms:.3f} px，不能应用定位"
        )
    try:
        right_world = WORLD_DIRECTIONS[print_right_world]
        up_world = WORLD_DIRECTIONS[print_up_world]
    except KeyError as error:
        raise QrRegistrationError("纸面 RIGHT / UP 必须选择有效 CAD 方向") from error
    if abs(float(np.dot(right_world, up_world))) > 1e-12:
        raise QrRegistrationError("纸面 RIGHT 与 UP 的 CAD 方向必须互相垂直")
    center = np.asarray(marker_center_world_mm, dtype=np.float64).reshape(-1)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise QrRegistrationError("二维码中心 CAD 坐标必须包含三个有限毫米值")
    front_world = np.cross(right_world, up_world)
    world_from_marker = np.column_stack((right_world, up_world, front_world))
    rotation_world_to_camera = (
        estimate.rotation_marker_to_camera @ world_from_marker.T
    )
    left_center = center - (
        rotation_world_to_camera.T @ estimate.translation_marker_to_camera_mm
    )
    baseline_world = rotation_world_to_camera.T @ np.asarray(
        [parsed.baseline_mm, 0.0, 0.0]
    )
    right_center = left_center + baseline_world

    result = copy.deepcopy(dict(calibration))
    for role, camera_center in (("left", left_center), ("right", right_center)):
        camera = result[f"{role}_camera"]
        camera["rotation_world_to_camera"] = rotation_world_to_camera.tolist()
        camera["center_world_mm"] = camera_center.tolist()
    result["registration_validated"] = registration_validated
    audit = {
        "mode": "qr_single_planar_control",
        "decoded_payload": estimate.decoded_payload,
        "marker_edge_mm": estimate.marker_edge_mm,
        "marker_center_world_mm": center.tolist(),
        "print_right_world": print_right_world,
        "print_up_world": print_up_world,
        "marker_front_world": front_world.tolist(),
        "left_corners_px": estimate.corners_px.tolist(),
        "left_camera_distance_to_marker_mm": estimate.camera_distance_mm,
        "reprojection_rms_px": estimate.reprojection_rms_px,
        "max_reprojection_rms_px": max_rms,
        "source_image_sha256": estimate.source_image_sha256,
        "registration_validated": registration_validated,
        "definition": "Single measured planar QR control target; +X print-right, +Y print-up, +Z printed-front",
    }
    signature = hashlib.sha256(
        json.dumps(audit, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    result["calibration_id"] = f"{parsed.calibration_id}-qr-{signature}"
    result["registration_adjustment"] = audit
    _calibration_from_manifest(result)
    return result


__all__ = [
    "QrPoseEstimate",
    "QrRegistrationError",
    "WORLD_DIRECTIONS",
    "detect_qr_pose",
    "estimate_square_pose",
    "printable_qr_png",
    "printable_qr_svg",
    "qr_payload",
    "register_calibration_from_qr",
    "write_printable_qr_png",
    "write_printable_qr_svg",
]
