"""Create a self-contained, explicitly paired field capture from GUI inputs."""

from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from .cad_model import load_cad_scene
from .camera_pose import POSE_MODE_LABELS, apply_camera_pose, calibration_pose
from .pipeline import atomic_write_text, ensure_paths_distinct


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
                           stl_unit: str | None = None) -> Path:
    from .stereo_analyzer import _calibration_from_manifest, _capture_groups_from_manifest, _load_cad_scene, _pipe_from_manifest

    if not pair_confirmed:
        raise ValueError("请确认左右图来自同一次同步拍摄；程序不会根据文件名自动配对。")
    calib = _calibration_from_manifest(calibration)
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
    photos = {}
    for role, path, timestamp in (("left", left_path, left_time), ("right", right_path, right_time)):
        try:
            parsed = datetime.fromisoformat(timestamp)
        except ValueError as error:
            raise ValueError("拍摄时间需要 ISO 8601 格式，例如 2026-09-07T10:00:00.000+08:00") from error
        if parsed.tzinfo is None:
            raise ValueError("拍摄时间必须包含时区，例如 +08:00。")
        data = Path(path).read_bytes()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        camera = getattr(calib, role)
        if image is None or image.shape[:2] != (camera.height, camera.width):
            raise ValueError(f"{role} 照片尺寸与标定尺寸不一致，不能缩放后直接使用原标定。")
        photos[role] = (data, {"camera_id": camera.camera_id, "path": role + Path(path).suffix.lower(),
            "sha256": hashlib.sha256(data).hexdigest(), "expected_width": camera.width, "expected_height": camera.height,
            "captured_at": timestamp, "timestamp_source": "MANIFEST_OPERATOR_CONFIRMED", "orientation_policy": "RAW_PIXELS_NO_EXIF_TRANSFORM"})
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
        count = int(np.ceil((calib.left.fx * calib.baseline_mm / np.min(positive_z) + 48) / 16) * 16)
        if count >= calib.left.width or count > 512:
            raise ValueError("当前相机距离/基线所需视差范围过大，请检查标定和配准或用已有清单配置。")
        manifest["analysis"] = {"stereo_matching": {"num_disparities": max(16, count)},
                                "minimum_focus_laplacian_variance": 5.0,
                                "minimum_luminance_p05": 1.0, "maximum_luminance_p95": 254.0}
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


class CameraPoseDialog:
    """Collect one rigid camera-to-CAD pose adjustment without editing source calibration."""

    def __init__(self, owner: Any, calibration: dict) -> None:
        self.owner = owner
        app = owner.app
        tk, ttk = app.tk, app.ttk
        current = calibration_pose(calibration)
        stored = owner.pose_adjustment
        self.calibration = calibration
        self.window = tk.Toplevel(owner.window)
        self.window.title("双目相机方向与倾斜校正")
        self.window.geometry("650x520")
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
        frame = ttk.Frame(self.window, padding=14)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame,
            text="调整的是双目相机整体相对 CAD 的位置和方向。左右相机始终保持刚性基线，原图不会被二次旋转。",
            wraplength=600,
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 12))
        ttk.Label(frame, text="相机所在方向").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Combobox(
            frame,
            textvariable=self.mode,
            values=tuple(POSE_MODE_LABELS.values()),
            state="readonly",
            width=39,
        ).grid(row=1, column=1, columnspan=3, sticky="ew", pady=5)
        ttk.Label(frame, text="双目基线中点（CAD 世界坐标 / mm）").grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(12, 3)
        )
        center_frame = ttk.Frame(frame)
        center_frame.grid(row=3, column=0, columnspan=4, sticky="w")
        for axis, variable in zip("XYZ", self.center):
            ttk.Label(center_frame, text=axis).pack(side="left", padx=(0, 3))
            ttk.Entry(center_frame, textvariable=variable, width=14).pack(
                side="left", padx=(0, 12)
            )
        ttk.Label(frame, text="相对所选方向的角度修正（度）").grid(
            row=4, column=0, columnspan=4, sticky="w", pady=(14, 3)
        )
        angle_fields = (
            ("水平偏航 yaw", self.yaw, "让投影管轴左右转动"),
            ("上下俯仰 pitch", self.pitch, "修正相机抬头/低头"),
            ("画面滚转 roll", self.roll, "修正相机画面倾斜"),
        )
        for row, (label, variable, hint) in enumerate(angle_fields, start=5):
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=5)
            ttk.Entry(frame, textvariable=variable, width=14).grid(row=row, column=1, sticky="w", pady=5)
            ttk.Label(frame, text=hint, foreground="#4A6178").grid(row=row, column=2, columnspan=2, sticky="w", pady=5)
        ttk.Checkbutton(
            frame,
            text="已使用固定控制点验证调整后的相机—CAD 配准",
            variable=self.validated,
        ).grid(row=8, column=0, columnspan=4, sticky="w", pady=(14, 5))
        ttk.Label(
            frame,
            text=(
                "如果选择“沿用标定文件位姿”，下方位置和角度不生效。选择其他模式后，未勾选控制点验证时仍可保存数据，"
                "但自动安装与尺寸状态会保持待确认。管件不需要正对相机；完整三维旋转会用于 CAD 投影、点云和前后关系。"
            ),
            wraplength=600,
            foreground="#8A4E00",
        ).grid(row=9, column=0, columnspan=4, sticky="w", pady=8)
        buttons = ttk.Frame(frame)
        buttons.grid(row=10, column=0, columnspan=4, sticky="e", pady=10)
        ttk.Button(buttons, text="取消", command=self.window.destroy).pack(side="left", padx=4)
        ttk.Button(buttons, text="应用方向与倾斜", command=self.save).pack(side="left", padx=4)

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
        self.confirmed, self.history = tk.BooleanVar(value=False), tk.BooleanVar(value=False)
        self.message = tk.StringVar(value="导入已校正的双目照片与标定 JSON；标定须含 CAD 世界坐标配准。")
        main = ttk.Frame(self.window, padding=12)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        for row, (key, label) in enumerate((("model", "CAD 模型"), ("calibration", "双目标定 JSON"), ("left", "左目原始照片"), ("right", "右目原始照片"))):
            ttk.Label(main, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
            ttk.Entry(main, textvariable=self.fields[key]).grid(row=row, column=1, sticky="ew", pady=4)
            ttk.Button(main, text="选择", command=lambda k=key: self.browse(k)).grid(row=row, column=2, padx=5)
        for row, key in ((4, "left_time"), (5, "right_time")):
            ttk.Label(main, text="左目拍摄时间" if key == "left_time" else "右目拍摄时间").grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(main, textvariable=self.fields[key]).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Label(main, text="时间示例：2026-09-07T10:00:00.000+08:00；填写实际拍摄时间，左右时间差会校验。").grid(row=6, column=0, columnspan=3, sticky="w", pady=4)
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
            payload = json.loads(
                Path(self.fields["calibration"].get()).read_text(encoding="utf-8-sig")
            )
            if not isinstance(payload, dict):
                raise ValueError("标定文件应为 JSON 对象。")
            calibration = payload.get("stereo_calibration", payload)
            CameraPoseDialog(self, calibration)
        except (OSError, ValueError) as error:
            self.app.messagebox.showerror("标定载入失败", str(error), parent=self.window)

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

            payload = json.loads(Path(self.fields["calibration"].get()).read_text(encoding="utf-8-sig"))
            if not isinstance(payload, dict):
                raise ValueError("标定文件应为 JSON 对象。")
            calibration = apply_camera_pose(
                payload.get("stereo_calibration", payload),
                self.pose_adjustment,
            )
            path = create_capture_dataset(output_root=OUTPUT_ROOT / "captures",
                model_path=Path(self.fields["model"].get()), pipes=self.pipes, calibration=calibration,
                left_path=Path(self.fields["left"].get()), right_path=Path(self.fields["right"].get()),
                left_time=self.fields["left_time"].get().strip(), right_time=self.fields["right_time"].get().strip(),
                pair_confirmed=self.confirmed.get(), previous_manifest=self.app.manifest_path if self.history.get() else None,
                stl_unit=self.stl_unit.get() if Path(self.fields["model"].get()).suffix.lower() == ".stl" else None)
            self.app._load_sources(path, None)
            self.app.main_tabs.select(0)
            self.window.destroy()
        except (OSError, ValueError) as error:
            self.app.messagebox.showerror("现场数据创建失败", str(error), parent=self.window)
