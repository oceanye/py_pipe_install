"""Create a self-contained, explicitly paired field capture from GUI inputs."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from .cad_model import load_cad_scene
from .camera_pose import (
    POSE_MODE_LABELS,
    SIDE_ELEVATION_VIEWS,
    apply_camera_pose,
    calibration_pose,
    estimate_pipe_roll_correction,
    model_center_from_pipes,
    suggested_side_view,
)
from .pipeline import atomic_write_text, ensure_paths_distinct
from .qr_registration import (
    QrRegistrationError,
    WORLD_DIRECTIONS,
    detect_qr_pose,
    qr_payload,
    register_calibration_from_qr,
    write_printable_qr_png,
)
from .stereo_camera import (
    LAYOUT_SEPARATE,
    LAYOUT_SIDE_BY_SIDE_LR,
    LAYOUT_SIDE_BY_SIDE_RL,
    CapturedStereoPair,
    StereoCameraError,
    StereoCameraSession,
    probe_video_devices,
)


_TIMESTAMP_SOURCES = {
    "CAMERA_HARDWARE_CLOCK",
    "HOST_SYSTEM_CLOCK",
    "MANIFEST_OPERATOR_CONFIRMED",
}
_CAPTURE_PROVENANCE_KEYS = {
    "capture_backend",
    "capture_layout",
    "capture_device_index",
    "side_by_side_order",
    "capture_sync_method",
}


def normalize_capture_time(value: str, *, field: str = "拍摄时间") -> tuple[str, bool]:
    """Accept common operator input and return millisecond ISO-8601 with an offset.

    The boolean indicates that the workstation's local time zone was added.
    A date without a clock time is never sufficient for stereo synchronization.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}不能为空；请选择照片后核对自动填写的时间。")
    text = value.strip()
    text = (
        text.replace("年", "-")
        .replace("月", "-")
        .replace("日", " ")
        .replace("时", ":")
        .replace("分", ":")
        .replace("秒", "")
        .replace("/", "-")
        .strip()
        .rstrip(":")
    )
    text = re.sub(r"\s+", " ", text)
    if not re.search(r"[T ]\d{1,2}:\d{1,2}", text):
        raise ValueError(f"{field}必须包含时、分，例如 2026-09-07 15:30:20。")
    # ``fromisoformat`` requires zero-padded calendar and clock fields.  Pad
    # the common hand-entered variant while leaving fractions and offsets intact.
    match = re.fullmatch(
        r"(\d{4})-(\d{1,2})-(\d{1,2})([T ])(\d{1,2}):(\d{1,2})"
        r"(?::(\d{1,2})(\.\d{1,6})?)?([+-]\d{2}:?\d{2}|[Zz])?",
        text,
    )
    if match:
        year, month, day, separator, hour, minute, second, fraction, offset = match.groups()
        second = second or "00"
        if offset and offset.lower() == "z":
            offset = "+00:00"
        elif offset and ":" not in offset:
            offset = offset[:3] + ":" + offset[3:]
        text = (
            f"{int(year):04d}-{int(month):02d}-{int(day):02d}{separator}"
            f"{int(hour):02d}:{int(minute):02d}:{int(second):02d}"
            f"{fraction or ''}{offset or ''}"
        )
    elif text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise ValueError(
            f"{field}格式无法识别；可输入 2026-09-07 15:30:20 或 "
            "2026-09-07T15:30:20.000+08:00。"
        ) from error
    assumed_local_zone = parsed.tzinfo is None or parsed.utcoffset() is None
    if assumed_local_zone:
        parsed = parsed.astimezone()
    return parsed.isoformat(timespec="milliseconds"), assumed_local_zone


def photo_file_time(path: str | Path) -> str:
    """Return a photo file's modification time in the workstation time zone."""
    timestamp = Path(path).stat().st_mtime
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="milliseconds")


def load_calibration_json(path_value: str | Path) -> dict[str, Any]:
    """Load a calibration object and reject blank/directory paths clearly."""
    text = str(path_value).strip()
    if not text:
        raise ValueError("请先选择真实双目标定 JSON，再连接相机。")
    path = Path(text)
    if not path.is_file():
        raise ValueError(f"双目标定 JSON 不存在或不是文件：{path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"双目标定 JSON 格式无效：{error}") from error
    if not isinstance(payload, dict):
        raise ValueError("双目标定文件应为 JSON 对象。")
    calibration = payload.get("stereo_calibration", payload)
    if not isinstance(calibration, dict):
        raise ValueError("双目标定 JSON 缺少 stereo_calibration 对象。")
    return calibration


_NON_FIELD_CALIBRATION_MARKERS = (
    "SYNTHETIC",
    "DEMO",
    "EXAMPLE",
    "REPLACE_WITH_REAL",
)


def field_calibration_problem(calibration: Any) -> str | None:
    """Explain why a calibration must not be used for a live field capture."""
    if not isinstance(calibration, dict):
        return "双目标定内容不是有效对象。"
    calibration_id = str(calibration.get("calibration_id", "")).strip()
    upper_id = calibration_id.upper()
    if not calibration_id:
        return "双目标定缺少 calibration_id，无法确认其来源。"
    if any(marker in upper_id for marker in _NON_FIELD_CALIBRATION_MARKERS):
        return (
            f"当前标定 ID“{calibration_id}”属于合成演示或格式示例，不能用于现场 USB 双目相机。"
            "请选择这台相机的真实标定 JSON；每目应为实际分辨率，并包含实测内参、畸变、毫米基线和极线校正状态。"
        )
    if calibration.get("validated") is not True:
        return "当前双目标定尚未标记为已验证（validated=true），不能用于现场测量。"
    if calibration.get("rectified") is not True:
        return "当前输入未声明为已极线校正（rectified=true），不能直接进行现场双目匹配。"
    return None


def _capture_provenance(
    payload: dict[str, dict[str, Any]] | None,
    role: str,
) -> dict[str, Any]:
    if payload is None:
        return {}
    if not isinstance(payload, dict) or set(payload) - {"left", "right"}:
        raise ValueError("camera_capture_provenance must contain only left/right records")
    record = payload.get(role, {})
    if not isinstance(record, dict) or set(record) - _CAPTURE_PROVENANCE_KEYS:
        raise ValueError(f"camera_capture_provenance.{role} contains unsupported fields")
    if "capture_device_index" in record and (
        type(record["capture_device_index"]) is not int
        or record["capture_device_index"] < 0
    ):
        raise ValueError(f"camera_capture_provenance.{role}.capture_device_index is invalid")
    for key, value in record.items():
        if key != "capture_device_index" and (
            not isinstance(value, str) or not value.strip() or len(value) > 128
        ):
            raise ValueError(f"camera_capture_provenance.{role}.{key} is invalid")
    return copy.deepcopy(record)


def catalog_from_model(
    path: str | Path,
    *,
    stl_unit: str | None = None,
) -> tuple[list[dict], list[str]]:
    """Suggest straight cylindrical CAD objects; these are DESIGN values only."""
    scene = load_cad_scene(path, stl_unit=stl_unit)
    pipes, skipped = [], []
    for item in scene.objects:
        vertices = np.unique(item.vertices_world_mm, axis=0)
        origin = np.mean(vertices, axis=0)
        _, _, vectors = np.linalg.svd(vertices - origin, full_matrices=False)
        axis = vectors[0]
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        axial = (vertices - origin) @ axis
        radial = vertices - origin - np.outer(axial, axis)
        radii = np.linalg.norm(radial, axis=1)
        radius = float(np.quantile(radii, 0.75))
        length = float(np.ptp(axial))
        # End-cap centre vertices may be present.  Judge the side-wall rings.
        outer = radii[radii > radius * 0.5]
        if radius <= 0 or length < 4 * radius or not len(outer) or np.max(np.abs(outer - radius)) > 0.05 * radius:
            skipped.append(item.object_id)
            continue
        index = len(pipes) + 1
        pipes.append({"instance_id": index, "pipe_id": f"P{index:03d}", "cad_object_id": item.object_id,
            "cad_uuid": item.guid, "layer_id": item.layer_path or "现场",
            "color_class": "cad_color", "color_srgb": item.color_srgb,
            "nominal_diameter_mm": round(2 * radius, 6),
            "centerline_world_mm": [(origin + axis * axial.min()).tolist(), (origin + axis * axial.max()).tolist()]})
    return pipes, skipped


def create_capture_dataset(*, output_root: Path, model_path: Path, pipes: list[dict], calibration: dict,
                           left_path: Path, right_path: Path, left_time: str, right_time: str,
                           pair_confirmed: bool, previous_manifest: Path | None = None,
                           stl_unit: str | None = None,
                           timestamp_sources: dict[str, str] | None = None,
                           camera_capture_provenance: dict[str, dict[str, Any]] | None = None) -> Path:
    from .stereo_analyzer import _calibration_from_manifest, _capture_groups_from_manifest, _load_cad_scene, _pipe_from_manifest

    if not pair_confirmed:
        raise ValueError("请确认左右图来自同一次同步拍摄；程序不会根据文件名自动配对。")
    calib = _calibration_from_manifest(calibration)
    if camera_capture_provenance:
        calibration_problem = field_calibration_problem(calibration)
        if calibration_problem:
            raise ValueError(calibration_problem)
    if not pipes:
        raise ValueError("管件目录为空，请读取模型或载入管件目录。")
    for index, pipe in enumerate(pipes):
        _pipe_from_manifest(pipe, index)
    model_path = Path(model_path).resolve()
    if model_path.suffix.lower() not in {".3mf", ".3dm", ".stl"}:
        raise ValueError("模型需要 3DM、3MF 或 STL 文件。")
    source_scene = load_cad_scene(
        model_path,
        required_object_ids={pipe["cad_object_id"] for pipe in pipes},
        stl_unit=stl_unit,
    )
    model_bytes = model_path.read_bytes()
    model_hash = hashlib.sha256(model_bytes).hexdigest()
    if source_scene.source_sha256 != model_hash:
        raise ValueError("模型在读取过程中发生变化，请重新选择。")
    if timestamp_sources is not None and (
        not isinstance(timestamp_sources, dict)
        or set(timestamp_sources) - {"left", "right"}
    ):
        raise ValueError("timestamp_sources must contain only left/right values")
    photos = {}
    for role, path, timestamp in (("left", left_path, left_time), ("right", right_path, right_time)):
        timestamp, assumed_local_zone = normalize_capture_time(
            timestamp,
            field="左目拍摄时间" if role == "left" else "右目拍摄时间",
        )
        data = Path(path).read_bytes()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        camera = getattr(calib, role)
        if image is None or image.shape[:2] != (camera.height, camera.width):
            raise ValueError(f"{role} 照片尺寸与标定尺寸不一致，不能缩放后直接使用原标定。")
        timestamp_source = (timestamp_sources or {}).get(
            role, "MANIFEST_OPERATOR_CONFIRMED"
        )
        if timestamp_source not in _TIMESTAMP_SOURCES:
            raise ValueError(f"{role} timestamp_source is unsupported")
        view_record = {"camera_id": camera.camera_id, "path": role + Path(path).suffix.lower(),
            "sha256": hashlib.sha256(data).hexdigest(), "expected_width": camera.width, "expected_height": camera.height,
            "captured_at": timestamp, "timestamp_source": timestamp_source,
            "timestamp_timezone_assumed": assumed_local_zone,
            "orientation_policy": "RAW_PIXELS_NO_EXIF_TRANSFORM"}
        view_record.update(_capture_provenance(camera_capture_provenance, role))
        photos[role] = (data, view_record)
    if photos["left"][1]["sha256"] == photos["right"][1]["sha256"]:
        raise ValueError("左右照片内容相同，请选择各自相机的原始照片。")
    run_id = f"field-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
    model_record = {"path": "model" + model_path.suffix.lower(), "sha256": model_hash,
                    "unit": "millimeter", "pipes": copy.deepcopy(pipes)}
    if model_path.suffix.lower() == ".stl":
        model_record["source_unit"] = source_scene.source_unit
    manifest = {"schema_version": "2.0", "dataset_id": run_id, "model_revision": model_hash[:16],
        "validation_scope": "FIELD_CAPTURE_PENDING_ACCEPTANCE",
        "model": model_record,
        "stereo_calibration": copy.deepcopy(calibration),
        "capture": {"kind": "stereo_still_capture_set", "camera_layout": "stereo", "capture_group_id": run_id,
                    "interval_minutes": 30, "capture_groups": []}, "analysis": {}}
    # Optional history is accepted only for the exact same model, identities,
    # and camera configuration.  Every copied byte is checked before writing.
    history_assets = {}
    if previous_manifest:
        from .gui import _safe_manifest_asset_path

        previous_manifest = Path(previous_manifest).resolve()
        old = json.loads(previous_manifest.read_text(encoding="utf-8"))
        if (old.get("model", {}).get("sha256") != model_hash or old.get("model", {}).get("pipes") != pipes
                or old.get("model", {}).get("source_unit") != model_record.get("source_unit")
                or old.get("stereo_calibration") != calibration):
            raise ValueError("保留历史需要同一模型、管件目录和标定。更换现场配置时请取消保留历史。")
        manifest["analysis"] = copy.deepcopy(old.get("analysis", {}))
        manifest["capture"]["interval_minutes"] = old.get("capture", {}).get("interval_minutes", 30)
        for index, group in enumerate(old.get("capture", {}).get("capture_groups", [])):
            record = copy.deepcopy(group)
            for role in ("left", "right"):
                view = record["views"][role]
                path = _safe_manifest_asset_path(previous_manifest, view["path"])
                if path is None:
                    raise ValueError("历史照片路径超出数据目录。")
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != view["sha256"]:
                    raise ValueError("历史照片内容已改变，不能继续沿用。")
                name = f"history_{index}_{role}{path.suffix}"
                history_assets[name] = data
                view["path"] = name
            manifest["capture"]["capture_groups"].append(record)
    if not manifest["analysis"]:
        # Search range must cover the registered model at its expected depth.
        all_points = np.array([p for pipe in pipes for p in pipe["centerline_world_mm"]])
        camera_points = (all_points - calib.left.center_world_mm) @ calib.left.rotation_world_to_camera.T
        positive_z = camera_points[:, 2][camera_points[:, 2] > 0]
        if not len(positive_z):
            raise ValueError("模型在相机后方，请检查 CAD 与相机的配准。")
        nearest_depth_mm = float(np.min(positive_z))
        expected_max_disparity_px = (
            calib.left.fx * calib.baseline_mm / nearest_depth_mm
        )
        count = int(np.ceil((expected_max_disparity_px + 48) / 16) * 16)
        if count >= calib.left.width:
            raise ValueError(
                "当前标定与 CAD 位姿需要的视差搜索范围已达到整幅图宽，"
                "左右目没有足够重叠区域。"
                f"最近 CAD 深度={nearest_depth_mm:.1f} mm，"
                f"焦距={calib.left.fx:.1f} px，基线={calib.baseline_mm:.1f} mm，"
                f"预计最大视差={expected_max_disparity_px:.1f} px，"
                f"搜索宽度={count} px，图像宽度={calib.left.width} px。"
                "请检查 STL 单位、二维码 CAD 坐标和纸面方向，或适当增加相机距离。"
            )
        manifest["analysis"] = {"stereo_matching": {"num_disparities": max(16, count)},
                                "minimum_focus_laplacian_variance": 5.0,
                                "minimum_luminance_p05": 1.0, "maximum_luminance_p95": 254.0,
                                "intake_disparity_estimate": {
                                    "nearest_registered_cad_depth_mm": nearest_depth_mm,
                                    "expected_max_disparity_px": expected_max_disparity_px,
                                    "selected_num_disparities": max(16, count),
                                    "image_width_px": calib.left.width,
                                }}
    manifest["capture"]["capture_groups"].append({"capture_id": run_id,
        "views": {role: record for role, (_, record) in photos.items()}})
    root = Path(output_root).resolve() / run_id
    manifest_path = root / "manifest.json"
    # Validate the pair contract before creating an on-disk capture package.
    _capture_groups_from_manifest(manifest_path, manifest["capture"], calib)
    root.mkdir(parents=True, exist_ok=False)
    (root / manifest["model"]["path"]).write_bytes(model_bytes)
    for name, data in history_assets.items():
        (root / name).write_bytes(data)
    for data, record in photos.values():
        (root / record["path"]).write_bytes(data)
    _load_cad_scene(manifest_path, manifest["model"])
    atomic_write_text(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest_path


_CAMERA_LAYOUT_LABELS = {
    LAYOUT_SIDE_BY_SIDE_LR: "单个并排双目流：左 | 右（推荐）",
    LAYOUT_SIDE_BY_SIDE_RL: "单个并排双目流：右 | 左",
    LAYOUT_SEPARATE: "两个独立相机设备",
}


class StereoCameraDialog:
    """Preview a connected UVC stereo stream and fill the intake form."""

    def __init__(self, owner: Any, calibration: dict) -> None:
        from .stereo_analyzer import _calibration_from_manifest

        self.owner = owner
        self.app = owner.app
        self.calibration = _calibration_from_manifest(calibration)
        tk, ttk = self.app.tk, self.app.ttk
        self.window = tk.Toplevel(owner.window)
        self.window.title("连接双目相机并同步抓拍")
        self.window.geometry("1020x700")
        self.window.minsize(850, 600)
        self.window.transient(owner.window)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.mode = tk.StringVar(value=_CAMERA_LAYOUT_LABELS[LAYOUT_SIDE_BY_SIDE_LR])
        self.left_index = tk.StringVar(value="0")
        self.right_index = tk.StringVar(value="1")
        self.message = tk.StringVar(
            value=(
                f"当前标定要求每目 {self.calibration.left.width}×"
                f"{self.calibration.left.height}；并排流应为 "
                f"{self.calibration.left.width * 2}×{self.calibration.left.height}。"
            )
        )
        self.session: StereoCameraSession | None = None
        self.after_id: str | None = None
        self.last_pair: CapturedStereoPair | None = None
        self.preview_images: list[Any] = []

        main = ttk.Frame(self.window, padding=12)
        main.pack(fill="both", expand=True)
        controls = ttk.Frame(main)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="采集方式：").pack(side="left")
        ttk.Combobox(
            controls,
            textvariable=self.mode,
            values=tuple(_CAMERA_LAYOUT_LABELS.values()),
            state="readonly",
            width=31,
        ).pack(side="left", padx=(3, 12))
        ttk.Label(controls, text="设备/左目索引：").pack(side="left")
        ttk.Entry(controls, textvariable=self.left_index, width=5).pack(
            side="left", padx=(3, 10)
        )
        ttk.Label(controls, text="右目索引：").pack(side="left")
        ttk.Entry(controls, textvariable=self.right_index, width=5).pack(
            side="left", padx=(3, 10)
        )
        ttk.Button(controls, text="检测设备", command=self.detect).pack(
            side="left", padx=3
        )
        ttk.Button(controls, text="打开预览", command=self.start).pack(
            side="left", padx=3
        )
        ttk.Button(controls, text="停止", command=self.stop).pack(side="left", padx=3)

        ttk.Label(main, textvariable=self.message, foreground="#355371").pack(
            fill="x", pady=(0, 8)
        )
        previews = ttk.Frame(main)
        previews.pack(fill="both", expand=True)
        previews.columnconfigure(0, weight=1)
        previews.columnconfigure(1, weight=1)
        previews.rowconfigure(1, weight=1)
        ttk.Label(previews, text="左目相机输出").grid(row=0, column=0, pady=4)
        ttk.Label(previews, text="右目相机输出").grid(row=0, column=1, pady=4)
        self.left_preview = ttk.Label(previews, anchor="center")
        self.right_preview = ttk.Label(previews, anchor="center")
        self.left_preview.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        self.right_preview.grid(row=1, column=1, sticky="nsew", padx=(4, 0))
        bottom = ttk.Frame(main)
        bottom.pack(fill="x", pady=(10, 0))
        ttk.Label(
            bottom,
            text="抓拍保存相机输出分辨率；预览缩小不改变分析图像。设备输出须已完成极线矫正。",
            foreground="#4A6178",
        ).pack(side="left")
        ttk.Button(bottom, text="取消", command=self.close).pack(side="right", padx=4)
        ttk.Button(
            bottom,
            text="同步抓拍并使用",
            command=self.capture,
        ).pack(side="right", padx=4)

    def _layout(self) -> str:
        inverse = {label: layout for layout, label in _CAMERA_LAYOUT_LABELS.items()}
        try:
            return inverse[self.mode.get()]
        except KeyError as error:
            raise StereoCameraError("请选择有效的双目采集方式") from error

    def _indices(self) -> tuple[int, int | None]:
        try:
            left = int(self.left_index.get())
            right = int(self.right_index.get()) if self._layout() == LAYOUT_SEPARATE else None
        except ValueError as error:
            raise StereoCameraError("相机索引必须是非负整数") from error
        if left < 0 or (right is not None and right < 0):
            raise StereoCameraError("相机索引必须是非负整数")
        return left, right

    def detect(self) -> None:
        self.stop()
        try:
            devices = probe_video_devices(maximum_index=5)
            if not devices:
                raise StereoCameraError("未检测到 OpenCV 可打开的视频设备")
            self.left_index.set(str(devices[0]["index"]))
            if len(devices) > 1:
                self.right_index.set(str(devices[1]["index"]))
            summary = "；".join(
                f"索引 {item['index']}：{item['width']}×{item['height']}"
                for item in devices
            )
            self.message.set(f"检测到 {len(devices)} 个视频设备：{summary}")
        except (OSError, StereoCameraError) as error:
            self.app.messagebox.showerror("相机检测失败", str(error), parent=self.window)

    def start(self) -> bool:
        try:
            self.stop()
            left, right = self._indices()
            self.session = StereoCameraSession(
                layout=self._layout(),
                left_index=left,
                right_index=right,
                eye_width=self.calibration.left.width,
                eye_height=self.calibration.left.height,
            )
            self.session.open()
            self.message.set("相机已打开，正在显示左右目实时预览。")
            self._update_preview()
            return True
        except (OSError, StereoCameraError) as error:
            self.stop()
            self.app.messagebox.showerror("相机打开失败", str(error), parent=self.window)
            return False

    def _photo(self, frame: np.ndarray) -> Any:
        height, width = frame.shape[:2]
        scale = min(460 / width, 500 / height, 1.0)
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
            raise StereoCameraError("无法生成相机预览")
        return self.app.tk.PhotoImage(
            data=base64.b64encode(encoded).decode("ascii"), format="png"
        )

    def _show_pair(self, pair: CapturedStereoPair) -> None:
        self.preview_images = [self._photo(pair.left), self._photo(pair.right)]
        self.left_preview.configure(image=self.preview_images[0])
        self.right_preview.configure(image=self.preview_images[1])

    def _update_preview(self) -> None:
        if self.session is None:
            return
        try:
            pair = self.session.read_pair()
            self.last_pair = pair
            self._show_pair(pair)
            if np.array_equal(pair.left, pair.right):
                self.message.set(
                    "相机仍在预热，或左右画面完全相同；请等待出现两个不同视角后再抓拍。"
                )
            else:
                self.message.set(
                    f"实时预览：左右主机时间差 {pair.sync_delta_ms:.3f} ms；"
                    "确认左右顺序正确后抓拍。"
                )
            self.after_id = self.window.after(80, self._update_preview)
        except (OSError, StereoCameraError) as error:
            self.stop()
            self.app.messagebox.showerror("相机读取失败", str(error), parent=self.window)

    def stop(self) -> None:
        if self.after_id is not None:
            try:
                self.window.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
        if self.session is not None:
            self.session.close()
            self.session = None

    def capture(self) -> None:
        try:
            if self.session is None and not self.start():
                return
            if self.session is None:
                return
            pair = self.session.read_pair()
            if np.array_equal(pair.left, pair.right):
                raise StereoCameraError(
                    "左右画面完全相同，可能仍在预热或当前不是双目输出；"
                    "请等待实时预览出现两个不同视角后重试。"
                )
            if pair.sync_delta_ms > self.calibration.max_sync_delta_ms:
                raise StereoCameraError(
                    f"本次左右抓拍时间差 {pair.sync_delta_ms:.3f} ms 超过标定允许的 "
                    f"{self.calibration.max_sync_delta_ms:g} ms"
                )
            encoded: dict[str, bytes] = {}
            for role, frame in (("left", pair.left), ("right", pair.right)):
                ok, payload = cv2.imencode(".png", frame)
                if not ok:
                    raise StereoCameraError(f"无法编码{role}相机原始图像")
                encoded[role] = payload.tobytes()
            from .measurement_gui import OUTPUT_ROOT

            capture_id = f"camera-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
            directory = OUTPUT_ROOT / "camera_intake" / capture_id
            directory.mkdir(parents=True, exist_ok=False)
            paths = {
                role: directory / f"{role}.png" for role in ("left", "right")
            }
            for role in ("left", "right"):
                paths[role].write_bytes(encoded[role])
            for role in ("left", "right"):
                self.owner.fields[role].set(str(paths[role]))
            self.owner.fields["left_time"].set(pair.left_captured_at)
            self.owner.fields["right_time"].set(pair.right_captured_at)
            self.owner.timestamp_sources = {
                "left": pair.timestamp_source,
                "right": pair.timestamp_source,
            }
            self.owner.camera_capture_provenance = pair.provenance
            if self._layout() in {LAYOUT_SIDE_BY_SIDE_LR, LAYOUT_SIDE_BY_SIDE_RL}:
                self.owner.confirmed.set(True)
                confirmation = "同一并排视频帧，已自动确认配对"
            else:
                self.owner.confirmed.set(False)
                confirmation = "两个独立设备，请核对硬件同步后勾选配对确认"
            self.owner.message.set(
                f"已从相机直接抓拍：{pair.left.shape[1]}×{pair.left.shape[0]} 每目，"
                f"时间差 {pair.sync_delta_ms:.3f} ms；{confirmation}。"
            )
            self.close()
        except (OSError, StereoCameraError) as error:
            self.app.messagebox.showerror("同步抓拍失败", str(error), parent=self.window)

    def close(self) -> None:
        self.stop()
        self.window.destroy()


class QrRegistrationDialog:
    """Generate a physical QR target and register the camera rig from it."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner
        app = owner.app
        tk, ttk = app.tk, app.ttk
        stored = owner.qr_settings
        self.window = tk.Toplevel(owner.window)
        self.window.title("1:1 二维码距离、方向与倾斜校正")
        self.window.geometry("760x670")
        self.window.resizable(False, False)
        self.window.transient(owner.window)
        self.marker_id = tk.StringVar(value=str(stored.get("marker_id", "PIPE-TWIN-QR-001")))
        self.marker_edge = tk.StringVar(value=str(stored.get("marker_edge_mm", 120.0)))
        self.measured_edge = tk.StringVar(
            value=str(stored.get("measured_marker_edge_mm", self.marker_edge.get()))
        )
        self.center = [
            tk.StringVar(value=str(value))
            for value in stored.get("marker_center_world_mm", [0.0, 0.0, 0.0])
        ]
        self.print_right = tk.StringVar(value=str(stored.get("print_right_world", "+X")))
        self.print_up = tk.StringVar(value=str(stored.get("print_up_world", "+Y")))
        self.max_rms = tk.StringVar(value=str(stored.get("max_reprojection_rms_px", 2.0)))
        self.print_measured = tk.BooleanVar(value=False)
        self.cad_confirmed = tk.BooleanVar(value=False)
        self.message = tk.StringVar(
            value="先生成并按 100% 打印；量具复核尺寸后，将二维码固定在已知 CAD 位置。"
        )

        frame = ttk.Frame(self.window, padding=14)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)
        ttk.Label(
            frame,
            text=(
                "二维码用于恢复相机到 CAD 的距离、方向和画面倾斜。"
                "检测使用左目已极线矫正图，双目基线作为刚体同步更新。"
            ),
            wraplength=710,
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 12))
        ttk.Label(frame, text="二维码编号").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.marker_id, width=34).grid(
            row=1, column=1, sticky="w", pady=5
        )
        ttk.Label(frame, text="文件标称边长 / mm").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(frame, textvariable=self.marker_edge, width=16).grid(
            row=2, column=1, sticky="w", pady=5
        )
        ttk.Label(frame, text="打印后实测边长 / mm").grid(
            row=2, column=2, sticky="e", pady=5
        )
        ttk.Entry(frame, textvariable=self.measured_edge, width=12).grid(
            row=2, column=3, sticky="w", padx=(8, 0), pady=5
        )
        ttk.Button(
            frame,
            text="生成 A4 300DPI 打印 PNG",
            command=self.export_marker,
        ).grid(row=1, column=2, columnspan=2, padx=8, pady=5)

        ttk.Separator(frame).grid(row=3, column=0, columnspan=4, sticky="ew", pady=12)
        ttk.Label(frame, text="二维码中心 CAD 坐标 / mm").grid(
            row=4, column=0, columnspan=4, sticky="w", pady=(0, 5)
        )
        center_frame = ttk.Frame(frame)
        center_frame.grid(row=5, column=0, columnspan=4, sticky="w")
        for axis, variable in zip("XYZ", self.center):
            ttk.Label(center_frame, text=axis).pack(side="left", padx=(0, 3))
            ttk.Entry(center_frame, textvariable=variable, width=15).pack(
                side="left", padx=(0, 12)
            )
        ttk.Label(frame, text="纸面 RIGHT 对应 CAD").grid(row=6, column=0, sticky="w", pady=8)
        ttk.Combobox(
            frame,
            textvariable=self.print_right,
            values=tuple(WORLD_DIRECTIONS),
            state="readonly",
            width=8,
        ).grid(row=6, column=1, sticky="w", pady=8)
        ttk.Label(frame, text="纸面 UP 对应 CAD").grid(row=6, column=2, sticky="e", pady=8)
        ttk.Combobox(
            frame,
            textvariable=self.print_up,
            values=tuple(WORLD_DIRECTIONS),
            state="readonly",
            width=8,
        ).grid(row=6, column=3, sticky="w", padx=(8, 0), pady=8)
        ttk.Label(frame, text="允许的最大重投影 RMS / px").grid(
            row=7, column=0, sticky="w", pady=5
        )
        ttk.Entry(frame, textvariable=self.max_rms, width=12).grid(
            row=7, column=1, sticky="w", pady=5
        )
        ttk.Label(
            frame,
            text="建议 ≤ 2 px；数值越小，角点与位姿拟合越一致。",
            foreground="#4A6178",
        ).grid(row=7, column=2, columnspan=2, sticky="w", pady=5)
        ttk.Checkbutton(
            frame,
            text="已按 100% 打印，并用量具复核 100 mm 校验线及二维码编码区实测边长",
            variable=self.print_measured,
        ).grid(row=8, column=0, columnspan=4, sticky="w", pady=(14, 5))
        ttk.Checkbutton(
            frame,
            text="已确认二维码中心 CAD 坐标以及纸面 RIGHT / UP 的实际安装方向",
            variable=self.cad_confirmed,
        ).grid(row=9, column=0, columnspan=4, sticky="w", pady=5)
        ttk.Label(
            frame,
            textvariable=self.message,
            wraplength=710,
            foreground="#355371",
        ).grid(row=10, column=0, columnspan=4, sticky="w", pady=(16, 8))
        ttk.Label(
            frame,
            text=(
                "实测建议：二维码尽量占左目画面 250 像素以上，保持平整、无反光。"
                "完成定位后固定相机；可移走二维码并重新同步抓拍管件。"
            ),
            wraplength=710,
            foreground="#8A4E00",
        ).grid(row=11, column=0, columnspan=4, sticky="w", pady=8)
        buttons = ttk.Frame(frame)
        buttons.grid(row=12, column=0, columnspan=4, sticky="e", pady=12)
        ttk.Button(buttons, text="关闭", command=self.window.destroy).pack(
            side="left", padx=4
        )
        ttk.Button(
            buttons,
            text="识别当前左目图并应用定位",
            command=self.apply,
        ).pack(side="left", padx=4)

    def _values(self) -> tuple[str, float, float, list[float], float]:
        marker_id = self.marker_id.get().strip()
        nominal_edge = float(self.marker_edge.get())
        measured_edge = float(self.measured_edge.get())
        payload = qr_payload(marker_id, nominal_edge)
        if not np.isfinite(measured_edge) or not 30.0 <= measured_edge <= 160.0:
            raise QrRegistrationError("打印后实测边长应在 30 到 160 mm 之间")
        center = [float(value.get()) for value in self.center]
        max_rms = float(self.max_rms.get())
        return payload, nominal_edge, measured_edge, center, max_rms

    def _remember(
        self,
        nominal_edge: float,
        measured_edge: float,
        center: list[float],
        max_rms: float,
    ) -> None:
        self.owner.qr_settings = {
            "marker_id": self.marker_id.get().strip(),
            "marker_edge_mm": nominal_edge,
            "measured_marker_edge_mm": measured_edge,
            "marker_center_world_mm": center,
            "print_right_world": self.print_right.get(),
            "print_up_world": self.print_up.get(),
            "max_reprojection_rms_px": max_rms,
        }

    def export_marker(self) -> None:
        try:
            _payload, nominal_edge, _measured_edge, center, max_rms = self._values()
            initial = f"{self.marker_id.get().strip()}_{nominal_edge:g}mm_300dpi_1to1.png"
            selected = self.owner.app.filedialog.asksaveasfilename(
                parent=self.window,
                title="保存 1:1 二维码定位板",
                initialfile=initial,
                defaultextension=".png",
                filetypes=(("PNG", "*.png"),),
            )
            if not selected:
                return
            self.owner.app.measurement_panel._protect_output(Path(selected))
            path = write_printable_qr_png(
                selected,
                marker_id=self.marker_id.get(),
                marker_edge_mm=nominal_edge,
            )
            self.measured_edge.set(f"{nominal_edge:g}")
            self._remember(nominal_edge, nominal_edge, center, max_rms)
            self.message.set(
                f"打印文件已保存：{path}。打印选择 100%/实际大小，之后用量具复核。"
            )
        except (OSError, ValueError) as error:
            self.owner.app.messagebox.showerror(
                "二维码生成失败", str(error), parent=self.window
            )

    def apply(self) -> None:
        try:
            if not self.print_measured.get():
                raise QrRegistrationError("请先确认打印比例并用量具复核打印尺寸")
            if not self.cad_confirmed.get():
                raise QrRegistrationError("请先确认二维码的 CAD 坐标和纸面方向")
            payload, nominal_edge, measured_edge, center, max_rms = self._values()
            calibration = load_calibration_json(
                self.owner.fields["calibration"].get()
            )
            from .stereo_analyzer import _calibration_from_manifest

            parsed = _calibration_from_manifest(calibration)
            if not parsed.validated:
                raise QrRegistrationError("双目标定尚未标记 validated=true，不能进行毫米定位")
            estimate = detect_qr_pose(
                self.owner.fields["left"].get(),
                expected_payload=payload,
                marker_edge_mm=measured_edge,
                intrinsic=parsed.left.intrinsic,
                expected_size=(parsed.left.width, parsed.left.height),
            )
            adjusted = register_calibration_from_qr(
                calibration,
                estimate,
                marker_center_world_mm=center,
                print_right_world=self.print_right.get(),
                print_up_world=self.print_up.get(),
                registration_validated=True,
                max_reprojection_rms_px=max_rms,
            )
            pose = calibration_pose(adjusted)
            self.owner.calibration_override = adjusted
            self.owner.pose_adjustment = {"mode": "keep"}
            self._remember(nominal_edge, measured_edge, center, max_rms)
            rig_center = ", ".join(f"{value:.2f}" for value in pose["center_world_mm"])
            forward = ", ".join(f"{value:.4f}" for value in pose["forward_world"])
            result = (
                f"定位已应用：二维码实测边长 {measured_edge:.2f} mm；"
                f"左目到二维码中心 {estimate.camera_distance_mm:.2f} mm；"
                f"重投影 RMS {estimate.reprojection_rms_px:.3f} px；"
                f"双目中心 CAD [{rig_center}] mm；观察方向 [{forward}]。"
            )
            self.message.set(result)
            self.owner.message.set(result + " 固定相机后可移走二维码并重新抓拍管件。")
        except (OSError, ValueError) as error:
            self.owner.app.messagebox.showerror(
                "二维码定位失败", str(error), parent=self.window
            )


class CameraPoseDialog:
    """Collect one rigid camera-to-CAD pose adjustment without editing source calibration."""

    def __init__(self, owner: Any, calibration: dict) -> None:
        self.owner = owner
        app = owner.app
        tk, ttk = app.tk, app.ttk
        current = calibration_pose(calibration)
        stored = owner.pose_adjustment
        self.calibration = calibration
        self.base_registration_validated = bool(current["registration_validated"])
        self.window = tk.Toplevel(owner.window)
        self.window.title("双目相机方向与倾斜校正")
        self.window.geometry("720x480")
        self.window.resizable(False, False)
        self.window.transient(owner.window)
        self.mode = tk.StringVar(value=POSE_MODE_LABELS.get(stored.get("mode", "keep"), POSE_MODE_LABELS["keep"]))
        center = stored.get("center_world_mm", current["center_world_mm"])
        self.center = [tk.StringVar(value=f"{value:.6g}") for value in center]
        self.yaw = tk.StringVar(value=str(stored.get("yaw_deg", 0.0)))
        self.pitch = tk.StringVar(value=str(stored.get("pitch_deg", 0.0)))
        self.roll = tk.StringVar(value=str(stored.get("roll_deg", 0.0)))
        self.validated = tk.BooleanVar(
            value=bool(stored.get("registration_validated", False))
        )
        self.distance = tk.StringVar(value=str(owner.side_view_distance_mm))
        self.auto_message = tk.StringVar(value="")
        try:
            target = model_center_from_pipes(owner.pipes)
            target_text = ", ".join(f"{value:.2f}" for value in target)
            self.target_text = f"管件组 CAD 中心 [{target_text}] mm"
        except ValueError as error:
            self.target_text = str(error)
        frame = ttk.Frame(self.window, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text="相机已经大致摆正时，抓拍管件后直接运行一键自动微调。程序只修正画面中的小角度倾斜。",
            wraplength=660,
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 12))
        quick = ttk.LabelFrame(frame, text="推荐操作", padding=10)
        quick.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(0, 10))
        ttk.Label(
            quick,
            text="先在现场数据录入窗口完成双目抓拍，然后点击：",
            foreground="#355371",
        ).grid(row=0, column=0, sticky="w", padx=(0, 12))
        ttk.Button(
            quick,
            text="一键自动微调角度",
            command=self.auto_tilt,
        ).grid(row=0, column=1, padx=5)
        ttk.Button(
            quick,
            text="二维码自动完整定位",
            command=self.open_qr,
        ).grid(row=0, column=2, padx=5)
        quick.columnconfigure(0, weight=1)

        side = ttk.LabelFrame(frame, text="可选：CAD 侧立面粗略方向", padding=9)
        side.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(0, 10))
        ttk.Label(side, text=self.target_text, foreground="#355371").grid(
            row=0, column=0, columnspan=4, sticky="w", pady=(0, 7)
        )
        ttk.Label(side, text="相机到模型中心距离 / mm").grid(
            row=1, column=0, sticky="w", pady=4
        )
        ttk.Entry(side, textvariable=self.distance, width=14).grid(
            row=1, column=1, sticky="w", padx=(6, 12), pady=4
        )
        for column, (view_key, definition) in enumerate(SIDE_ELEVATION_VIEWS.items()):
            ttk.Button(
                side,
                text=definition["label"],
                command=lambda key=view_key: self.select_side_view(key),
            ).grid(row=2, column=column, padx=4, pady=(7, 2), sticky="ew")
            side.columnconfigure(column, weight=1)
        ttk.Label(frame, text="当前/高级方向").grid(row=6, column=0, sticky="w", pady=5)
        ttk.Combobox(
            frame,
            textvariable=self.mode,
            values=tuple(POSE_MODE_LABELS.values()),
            state="readonly",
            width=39,
        ).grid(row=6, column=1, columnspan=3, sticky="ew", pady=5)
        ttk.Label(frame, text="双目基线中点（CAD 世界坐标 / mm）").grid(
            row=7, column=0, columnspan=4, sticky="w", pady=(12, 3)
        )
        center_frame = ttk.Frame(frame)
        center_frame.grid(row=8, column=0, columnspan=4, sticky="w")
        for axis, variable in zip("XYZ", self.center):
            ttk.Label(center_frame, text=axis).pack(side="left", padx=(0, 3))
            ttk.Entry(center_frame, textvariable=variable, width=14).pack(
                side="left", padx=(0, 12)
            )
        ttk.Label(frame, text="相对所选方向的角度修正（度）").grid(
            row=9, column=0, columnspan=4, sticky="w", pady=(14, 3)
        )
        angle_fields = (
            ("水平偏航 yaw", self.yaw, "让投影管轴左右转动"),
            ("上下俯仰 pitch", self.pitch, "修正相机抬头/低头"),
            ("画面滚转 roll", self.roll, "修正相机画面倾斜"),
        )
        for row, (label, variable, hint) in enumerate(angle_fields, start=10):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=5)
            ttk.Entry(frame, textvariable=variable, width=14).grid(row=row, column=1, sticky="w", pady=5)
            ttk.Label(frame, text=hint, foreground="#4A6178").grid(row=row, column=2, columnspan=2, sticky="w", pady=5)
        ttk.Label(
            frame,
            textvariable=self.auto_message,
            foreground="#355371",
            wraplength=700,
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(7, 0))
        ttk.Checkbutton(
            frame,
            text="已使用固定控制点验证调整后的相机—CAD 配准",
            variable=self.validated,
        ).grid(row=13, column=0, columnspan=4, sticky="w", pady=(10, 5))
        ttk.Label(
            frame,
            text=(
                "如果选择“沿用标定文件位姿”，下方位置和角度不生效。选择其他模式后，未勾选控制点验证时仍可保存数据，"
                "但自动安装与尺寸状态会保持待确认。管件不需要正对相机；完整三维旋转会用于 CAD 投影、点云和前后关系。"
            ),
            wraplength=700,
            foreground="#8A4E00",
        ).grid(row=14, column=0, columnspan=4, sticky="w", pady=8)
        self.advanced_visible = False
        self.advanced_widgets = [
            widget
            for widget in frame.grid_slaves()
            if int(widget.grid_info()["row"]) >= 5
        ]
        for widget in self.advanced_widgets:
            widget.grid_remove()
        self.advanced_toggle = ttk.Button(
            frame,
            text="显示可选方向与高级参数",
            command=self.toggle_advanced,
        )
        self.advanced_toggle.grid(row=3, column=0, columnspan=4, sticky="w", pady=7)
        buttons = ttk.Frame(frame)
        buttons.grid(row=4, column=0, columnspan=4, sticky="e", pady=10)
        ttk.Button(buttons, text="取消", command=self.window.destroy).pack(side="left", padx=4)
        ttk.Button(buttons, text="应用微调", command=self.save).pack(side="left", padx=4)

    def toggle_advanced(self) -> None:
        self.advanced_visible = not self.advanced_visible
        for widget in self.advanced_widgets:
            if self.advanced_visible:
                widget.grid()
            else:
                widget.grid_remove()
        self.advanced_toggle.configure(
            text="收起可选方向与高级参数"
            if self.advanced_visible
            else "显示可选方向与高级参数"
        )
        self.window.geometry("760x760" if self.advanced_visible else "720x480")

    def select_side_view(self, view_key: str) -> None:
        try:
            suggestion = suggested_side_view(
                self.owner.pipes,
                view_key,
                float(self.distance.get()),
            )
            self.mode.set(POSE_MODE_LABELS[suggestion["mode"]])
            for variable, value in zip(self.center, suggestion["center_world_mm"]):
                variable.set(f"{value:.6g}")
            self.yaw.set("0")
            self.pitch.set("0")
            self.roll.set("0")
            self.validated.set(False)
            self.owner.side_view_distance_mm = suggestion["distance_mm"]
            self.auto_message.set(
                f"已选择“{suggestion['view_label']}”，相机自动对准管件组 CAD 中心；"
                "可抓拍后运行自动轻微倾斜。"
            )
        except ValueError as error:
            self.owner.app.messagebox.showerror(
                "侧立面方向无效", str(error), parent=self.window
            )

    def auto_tilt(self) -> None:
        try:
            inverse_labels = {label: mode for mode, label in POSE_MODE_LABELS.items()}
            mode = inverse_labels[self.mode.get()]
            if mode == "keep":
                current = calibration_pose(self.calibration)
                mode = "adjust_current"
                self.mode.set(POSE_MODE_LABELS[mode])
                for variable, value in zip(self.center, current["center_world_mm"]):
                    variable.set(f"{value:.6g}")
                self.yaw.set("0")
                self.pitch.set("0")
                self.roll.set("0")
            candidate = apply_camera_pose(
                self.calibration,
                {
                    "mode": mode,
                    "center_world_mm": [float(value.get()) for value in self.center],
                    "yaw_deg": float(self.yaw.get()),
                    "pitch_deg": float(self.pitch.get()),
                    "roll_deg": 0.0,
                    "registration_validated": False,
                },
            )
            estimate = estimate_pipe_roll_correction(
                self.owner.fields["left"].get(),
                self.owner.pipes,
                candidate,
            )
            self.roll.set(f"{estimate['roll_correction_deg']:.6g}")
            self.validated.set(self.base_registration_validated)
            self.auto_message.set(
                f"自动微调完成：画面管轴 {estimate['observed_pipe_angle_deg']:.2f}°，"
                f"CAD 投影 {estimate['expected_cad_angle_deg']:.2f}°，"
                f"已填写 roll={estimate['roll_correction_deg']:.2f}°；"
                f"使用 {estimate['line_count']} 条线段。"
            )
        except (KeyError, ValueError) as error:
            self.owner.app.messagebox.showerror(
                "自动倾斜失败", str(error), parent=self.window
            )

    def open_qr(self) -> None:
        self.window.destroy()
        self.owner.qr_registration()

    def save(self) -> None:
        try:
            inverse_labels = {label: mode for mode, label in POSE_MODE_LABELS.items()}
            adjustment = {
                "mode": inverse_labels[self.mode.get()],
                "center_world_mm": [float(value.get()) for value in self.center],
                "yaw_deg": float(self.yaw.get()),
                "pitch_deg": float(self.pitch.get()),
                "roll_deg": float(self.roll.get()),
                "registration_validated": self.validated.get(),
            }
            adjusted = apply_camera_pose(self.calibration, adjustment)
            pose = calibration_pose(adjusted)
            self.owner.pose_adjustment = adjustment
            try:
                self.owner.side_view_distance_mm = float(self.distance.get())
            except ValueError:
                pass
            if adjustment["mode"] == "keep":
                self.owner.message.set("将沿用标定文件中的相机位置和方向。")
            else:
                forward = ", ".join(f"{value:.4f}" for value in pose["forward_world"])
                state = "已确认控制点配准" if pose["registration_validated"] else "配准待控制点确认"
                self.owner.message.set(
                    f"相机方向已设置：{POSE_MODE_LABELS[adjustment['mode']]}；世界前向 [{forward}]；"
                    f"yaw={adjustment['yaw_deg']:g}°，pitch={adjustment['pitch_deg']:g}°，"
                    f"roll={adjustment['roll_deg']:g}°；{state}。"
                )
            self.window.destroy()
        except (KeyError, ValueError) as error:
            self.owner.app.messagebox.showerror("相机位姿无效", str(error), parent=self.window)


class CaptureInputDialog:
    def __init__(self, app: Any) -> None:
        self.app = app
        tk, ttk = app.tk, app.ttk
        self.window = tk.Toplevel(app.root)
        self.window.title("现场数据录入 · 模型、双目照片与标定")
        self.window.geometry("990x740")
        self.window.minsize(900, 690)
        self.window.transient(app.root)
        self.pipes = copy.deepcopy(app.manifest.get("model", {}).get("pipes", []))
        self.fields = {key: tk.StringVar() for key in ("model", "calibration", "left", "right", "left_time", "right_time")}
        self.stl_unit = tk.StringVar(value="millimeter")
        self.pose_adjustment: dict[str, Any] = {"mode": "keep"}
        self.calibration_override: dict[str, Any] | None = None
        self.qr_settings: dict[str, Any] = {}
        self.side_view_distance_mm = 1000.0
        self.timestamp_sources = {
            "left": "MANIFEST_OPERATOR_CONFIRMED",
            "right": "MANIFEST_OPERATOR_CONFIRMED",
        }
        self.camera_capture_provenance: dict[str, dict[str, Any]] = {}
        self.confirmed, self.history = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
        self.message = tk.StringVar(value="相机已连接时可直接同步抓拍；也可导入已有双目照片。标定须含 CAD 世界坐标配准。")
        main = ttk.Frame(self.window, padding=12)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        for row, (key, label) in enumerate((("model", "CAD 模型"), ("calibration", "双目标定 JSON"), ("left", "左目照片（自动抓拍/文件）"), ("right", "右目照片（自动抓拍/文件）"))):
            ttk.Label(main, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
            ttk.Entry(main, textvariable=self.fields[key]).grid(row=row, column=1, sticky="ew", pady=4)
            ttk.Button(main, text="选择", command=lambda k=key: self.browse(k)).grid(row=row, column=2, padx=5)
        for row, key in ((4, "left_time"), (5, "right_time")):
            ttk.Label(main, text="左目拍摄时间" if key == "left_time" else "右目拍摄时间").grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(main, textvariable=self.fields[key]).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Label(main, text="选择照片后自动填写文件时间，请核对实际拍摄时间；支持 2026-09-07 10:00:00，未写时区时使用本机时区。").grid(row=6, column=0, columnspan=3, sticky="w", pady=4)
        model_toolbar = ttk.Frame(main)
        self.model_toolbar = model_toolbar
        model_toolbar.grid(row=7, column=0, columnspan=3, sticky="ew", pady=(7, 2))
        ttk.Label(model_toolbar, text="STL 坐标单位：").pack(side="left", padx=(0, 3))
        ttk.Combobox(
            model_toolbar,
            textvariable=self.stl_unit,
            values=("millimeter", "centimeter", "meter", "inch"),
            state="readonly",
            width=12,
        ).pack(side="left", padx=(0, 8))
        ttk.Button(model_toolbar, text="从模型读取直管目录", command=self.scan).pack(side="left", padx=3)
        ttk.Button(model_toolbar, text="导入管件目录", command=self.import_catalog).pack(side="left", padx=3)
        action_toolbar = ttk.Frame(main)
        self.action_toolbar = action_toolbar
        action_toolbar.grid(row=8, column=0, columnspan=3, sticky="ew", pady=(2, 7))
        ttk.Button(action_toolbar, text="连接双目相机抓拍", command=self.camera_capture).pack(side="left", padx=3)
        ttk.Button(action_toolbar, text="二维码定位", command=self.qr_registration).pack(side="left", padx=3)
        ttk.Button(action_toolbar, text="相机方向/倾斜校正", command=self.camera_pose).pack(side="left", padx=3)
        ttk.Button(action_toolbar, text="编辑所选管件", command=self.edit_pipe).pack(side="left", padx=3)
        ttk.Button(action_toolbar, text="移除所选管件", command=self.remove_pipe).pack(side="left", padx=3)
        ttk.Button(action_toolbar, text="导出标定格式示例", command=self.calibration_example).pack(side="left", padx=3)
        self.tree = ttk.Treeview(main, columns=("id", "object", "diameter", "color", "layer"), show="headings", height=8)
        for key, title in (("id", "管件 ID"), ("object", "CAD 对象 ID"), ("diameter", "设计外径/mm"), ("color", "物体颜色"), ("layer", "设计层")):
            self.tree.heading(key, text=title)
            self.tree.column(key, width=140 if key != "object" else 250)
        self.tree.grid(row=9, column=0, columnspan=3, sticky="nsew")
        main.rowconfigure(9, weight=1)
        self.tree.bind("<Double-1>", lambda _e: self.edit_pipe())
        ttk.Checkbutton(main, text="确认左右图来自同一次同步拍摄（不能用同一张照片代替双目）", variable=self.confirmed).grid(row=10, column=0, columnspan=3, sticky="w", pady=6)
        ttk.Checkbutton(main, text="保留当前清单的历史拍摄组，用于连续缺失证据判断（要求配置完全一致）", variable=self.history).grid(row=11, column=0, columnspan=3, sticky="w", pady=3)
        ttk.Label(main, textvariable=self.message, wraplength=910, foreground="#355371").grid(row=12, column=0, columnspan=3, sticky="w", pady=8)
        ttk.Button(main, text="创建现场数据并载入", command=self.create).grid(row=13, column=0, columnspan=3, sticky="e", pady=7)
        if app.manifest_path:
            from .gui import _safe_manifest_asset_path

            model = _safe_manifest_asset_path(app.manifest_path, app.manifest.get("model", {}).get("path"))
            self.fields["model"].set(str(model or ""))
            current_calibration = app.manifest.get("stereo_calibration")
            calibration_problem = field_calibration_problem(current_calibration)
            if calibration_problem:
                self.fields["calibration"].set("")
                self.message.set(
                    "当前已载入清单使用的是演示或未完成的标定，已取消自动沿用。"
                    "请在“双目标定 JSON”选择这台 USB 双目相机的真实标定文件。"
                )
            else:
                self.fields["calibration"].set(str(app.manifest_path))
            if app.manifest.get("model", {}).get("source_unit"):
                self.stl_unit.set(app.manifest["model"]["source_unit"])
        self.refresh()

    def browse(self, key: str) -> None:
        types = (("CAD", "*.3dm *.3mf *.stl"),) if key == "model" else (("JSON", "*.json"),) if key == "calibration" else (("照片", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),)
        selected = self.app.filedialog.askopenfilename(parent=self.window, filetypes=types)
        if selected:
            self.fields[key].set(selected)
            if key == "calibration":
                self.pose_adjustment = {"mode": "keep"}
                self.calibration_override = None
            elif key in {"left", "right"}:
                time_key = f"{key}_time"
                self.timestamp_sources[key] = "MANIFEST_OPERATOR_CONFIRMED"
                self.camera_capture_provenance.pop(key, None)
                self.confirmed.set(False)
                try:
                    self.fields[time_key].set(photo_file_time(selected))
                    self.message.set(
                        "已按本机时区带入照片文件修改时间；请核对为实际曝光时间，"
                        "左右目仍需满足标定文件的同步时间差。"
                    )
                except OSError as error:
                    self.app.messagebox.showerror(
                        "照片时间读取失败", str(error), parent=self.window
                    )

    def refresh(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for index, pipe in enumerate(self.pipes):
            self.tree.insert("", "end", iid=str(index), values=(pipe["pipe_id"], pipe["cad_object_id"], f"{pipe['nominal_diameter_mm']:.3f}", pipe["color_srgb"], pipe["layer_id"]))

    def scan(self) -> None:
        try:
            model_path = Path(self.fields["model"].get())
            self.pipes, skipped = catalog_from_model(
                model_path,
                stl_unit=self.stl_unit.get() if model_path.suffix.lower() == ".stl" else None,
            )
            self.refresh()
            stl_note = " STL 不含颜色和业务 ID，灰色为占位，必须逐管核对/编辑。" if model_path.suffix.lower() == ".stl" else ""
            self.message.set(f"已读取 {len(self.pipes)} 个直管候选（设计尺寸）。未纳入 {len(skipped)} 个对象：{', '.join(skipped) or '无'}。请核对全部目标、颜色与业务 ID；双击可修改。{stl_note}")
        except (ValueError, OSError) as error:
            self.app.messagebox.showerror("模型读取失败", str(error), parent=self.window)

    def import_catalog(self) -> None:
        selected = self.app.filedialog.askopenfilename(parent=self.window, filetypes=(("JSON", "*.json"),))
        if selected:
            try:
                from .stereo_analyzer import _pipe_from_manifest

                payload = json.loads(Path(selected).read_text(encoding="utf-8-sig"))
                if not isinstance(payload, (list, dict)):
                    raise ValueError("管件目录应为 JSON 列表或对象。")
                pipes = payload if isinstance(payload, list) else payload.get("pipes", payload.get("model", {}).get("pipes"))
                if not isinstance(pipes, list) or not pipes:
                    raise ValueError("目录需要非空 pipes 列表；也可选择包含 model.pipes 的完整清单。")
                for index, pipe in enumerate(pipes):
                    _pipe_from_manifest(pipe, index)
                self.pipes = pipes
                self.refresh()
            except (ValueError, OSError) as error:
                self.app.messagebox.showerror("目录载入失败", str(error), parent=self.window)

    def camera_pose(self) -> None:
        try:
            calibration = self.calibration_override or load_calibration_json(
                self.fields["calibration"].get()
            )
            CameraPoseDialog(self, calibration)
        except (OSError, ValueError) as error:
            self.app.messagebox.showerror("标定载入失败", str(error), parent=self.window)

    def camera_capture(self) -> None:
        try:
            calibration = load_calibration_json(self.fields["calibration"].get())
            calibration_problem = field_calibration_problem(calibration)
            if calibration_problem:
                raise ValueError(calibration_problem)
            StereoCameraDialog(self, calibration)
        except (OSError, ValueError) as error:
            self.app.messagebox.showerror("标定载入失败", str(error), parent=self.window)

    def qr_registration(self) -> None:
        QrRegistrationDialog(self)

    def edit_pipe(self) -> None:
        selected = self.tree.selection()
        if not selected:
            return
        pipe = self.pipes[int(selected[0])]
        app, window = self.app, self.app.tk.Toplevel(self.window)
        window.title("编辑管件设计属性")
        window.transient(self.window)
        fields = {}
        for row, (key, label) in enumerate((("pipe_id", "业务 ID"), ("nominal_diameter_mm", "设计外径/mm"), ("color_srgb", "实物颜色 #RRGGBB"), ("layer_id", "设计层"))):
            fields[key] = app.tk.StringVar(value=str(pipe[key]))
            app.ttk.Label(window, text=label).grid(row=row, column=0, padx=12, pady=7, sticky="w")
            app.ttk.Entry(window, textvariable=fields[key], width=35).grid(row=row, column=1, padx=12, pady=7)
        def save() -> None:
            try:
                from .stereo_analyzer import _pipe_from_manifest

                updated = pipe | {key: variable.get() for key, variable in fields.items()}
                updated["nominal_diameter_mm"] = float(updated["nominal_diameter_mm"])
                _pipe_from_manifest(updated, 0)
                if any(p is not pipe and p["pipe_id"] == updated["pipe_id"] for p in self.pipes):
                    raise ValueError("业务 ID 不能重复。")
                pipe.update(updated)
                self.refresh()
                window.destroy()
            except ValueError as error:
                app.messagebox.showerror("输入无效", str(error), parent=window)
        app.ttk.Button(window, text="保存属性", command=save).grid(row=4, column=1, padx=12, pady=12, sticky="e")

    def remove_pipe(self) -> None:
        selected = self.tree.selection()
        if selected:
            self.pipes.pop(int(selected[0]))
            self.refresh()

    def calibration_example(self) -> None:
        path = self.app.filedialog.asksaveasfilename(parent=self.window, initialfile="标定格式示例_需替换实机参数.json", defaultextension=".json")
        if path:
            try:
                self.app.measurement_panel._protect_output(Path(path))
                inputs = {str(Path(self.fields[key].get()).resolve()) for key in ("model", "calibration", "left", "right") if self.fields[key].get()}
                ensure_paths_distinct(output=path, **{f"input_{i}": p for i, p in enumerate(inputs)})
                demo = Path(__file__).resolve().parents[1] / "test_model" / "field_stereo_demo_manifest.json"
                calibration = json.loads(demo.read_text(encoding="utf-8"))["stereo_calibration"]
                calibration.update(calibration_id="REPLACE_WITH_REAL_CALIBRATION", validated=False, registration_validated=False)
                atomic_write_text(path, json.dumps({"note": "仅为格式示例；必须替换为实际校正后相机参数和 CAD 配准。", "stereo_calibration": calibration}, ensure_ascii=False, indent=2) + "\n")
                self.message.set(f"标定格式示例已保存：{path}")
            except (OSError, ValueError) as error:
                self.app.messagebox.showerror("保存失败", str(error), parent=self.window)

    def create(self) -> None:
        try:
            from .measurement_gui import OUTPUT_ROOT

            calibration = apply_camera_pose(
                self.calibration_override
                or load_calibration_json(self.fields["calibration"].get()),
                self.pose_adjustment,
            )
            left_time, _ = normalize_capture_time(
                self.fields["left_time"].get(), field="左目拍摄时间"
            )
            right_time, _ = normalize_capture_time(
                self.fields["right_time"].get(), field="右目拍摄时间"
            )
            self.fields["left_time"].set(left_time)
            self.fields["right_time"].set(right_time)
            path = create_capture_dataset(output_root=OUTPUT_ROOT / "captures",
                model_path=Path(self.fields["model"].get()), pipes=self.pipes, calibration=calibration,
                left_path=Path(self.fields["left"].get()), right_path=Path(self.fields["right"].get()),
                left_time=left_time, right_time=right_time,
                pair_confirmed=self.confirmed.get(), previous_manifest=self.app.manifest_path if self.history.get() else None,
                stl_unit=self.stl_unit.get() if Path(self.fields["model"].get()).suffix.lower() == ".stl" else None,
                timestamp_sources=self.timestamp_sources,
                camera_capture_provenance=self.camera_capture_provenance)
            self.app._load_sources(path, None)
            self.app.main_tabs.select(0)
            self.window.destroy()
        except (OSError, ValueError) as error:
            self.app.messagebox.showerror("现场数据创建失败", str(error), parent=self.window)
