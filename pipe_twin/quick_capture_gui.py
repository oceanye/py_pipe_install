"""Small one-screen workflow for STL + stereo pipe status evaluation."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any


_PALETTE = ("#E74C3C", "#3498DB", "#2ECC71", "#F1C40F", "#9B59B6", "#E67E22")


class QuickCaptureDialog:
    """Keep the default field workflow to model, colours, calibration and pair."""

    def __init__(self, app: Any) -> None:
        from .capture_gui import (
            field_calibration_problem,
            load_calibration_json,
            normalize_capture_time,
            photo_file_time,
        )

        self.app = app
        self._field_calibration_problem = field_calibration_problem
        self._load_calibration_json = load_calibration_json
        self._normalize_capture_time = normalize_capture_time
        self._photo_file_time = photo_file_time
        tk, ttk = app.tk, app.ttk
        self.window = tk.Toplevel(app.root)
        self.window.title("快速双目评估")
        self.window.geometry("760x540")
        self.window.minsize(700, 500)
        self.window.transient(app.root)
        self.fields = {
            key: tk.StringVar()
            for key in ("model", "calibration", "left", "right", "left_time", "right_time")
        }
        self.stl_unit = tk.StringVar(value="millimeter")
        self.color_rules = tk.StringVar(value="扫描 STL 后自动生成")
        self.message = tk.StringVar(value="1. 选择 STL；2. 扫描并确认直径颜色；3. 选择标定和左右照片，或直接抓拍。")
        self.summary = tk.StringVar(value="尚未读取模型")
        self.pipes: list[dict[str, Any]] = []
        self.confirmed = tk.BooleanVar(value=True)
        self.timestamp_sources = {"left": "MANIFEST_OPERATOR_CONFIRMED", "right": "MANIFEST_OPERATOR_CONFIRMED"}
        self.camera_capture_provenance: dict[str, dict[str, Any]] = {}
        self.calibration_override: dict[str, Any] | None = None
        self.pose_adjustment = {"mode": "keep"}

        main = ttk.Frame(self.window, padding=18)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        ttk.Label(main, text="快速双目评估", font=("Segoe UI", 16, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", pady=(0, 3)
        )
        ttk.Label(
            main,
            text="输入 STL 和一组双目照片，系统按管道直径/颜色与 CAD 投影输出安装、未安装或遮蔽不确定。",
            foreground="#4A6178",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 14))

        self._path_row(main, 2, "STL 管道模型", "model", "选择 STL", self.browse_model)
        ttk.Label(main, text="STL 原坐标单位").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Combobox(
            main, textvariable=self.stl_unit,
            values=("millimeter", "centimeter", "meter", "inch"),
            state="readonly", width=16,
        ).grid(row=3, column=1, sticky="w", pady=5)
        ttk.Button(main, text="扫描模型", command=self.scan).grid(row=3, column=2, sticky="e", pady=5)
        ttk.Label(main, text="直径 → 颜色").grid(row=4, column=0, sticky="w", pady=5)
        ttk.Entry(main, textvariable=self.color_rules).grid(row=4, column=1, columnspan=2, sticky="ew", pady=5)
        ttk.Label(main, text="格式：22=#E74C3C;50=#3498DB（扫描后可直接修改）", foreground="#6B7785").grid(
            row=5, column=1, columnspan=2, sticky="w"
        )
        ttk.Label(main, textvariable=self.summary, foreground="#175D86").grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(3, 10)
        )

        self._calibration_row(main, 7)
        self._path_row(main, 8, "左目已矫正照片", "left", "选择照片", lambda: self.browse_photo("left"))
        self._path_row(main, 9, "右目已矫正照片", "right", "选择照片", lambda: self.browse_photo("right"))
        ttk.Label(main, text="拍摄时间").grid(row=10, column=0, sticky="w", pady=5)
        ttk.Entry(main, textvariable=self.fields["left_time"], width=27).grid(row=10, column=1, sticky="w", pady=5)
        ttk.Entry(main, textvariable=self.fields["right_time"], width=27).grid(row=10, column=2, sticky="w", pady=5)

        actions = ttk.Frame(main)
        actions.grid(row=11, column=0, columnspan=3, sticky="ew", pady=(14, 8))
        ttk.Button(actions, text="连接双目并抓拍", command=self.capture_camera).pack(side="left")
        ttk.Button(actions, text="创建并开始评估", command=self.create).pack(side="right")
        ttk.Button(actions, text="取消", command=self.window.destroy).pack(side="right", padx=6)
        ttk.Label(main, textvariable=self.message, wraplength=700, foreground="#355371").grid(
            row=12, column=0, columnspan=3, sticky="w", pady=8
        )

        if app.manifest_path:
            from .gui import _safe_manifest_asset_path

            model = _safe_manifest_asset_path(app.manifest_path, app.manifest.get("model", {}).get("path"))
            if model:
                self.fields["model"].set(str(model))

    def _path_row(self, parent: Any, row: int, label: str, key: str, button: str, command: Any) -> None:
        ttk = self.app.ttk
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=self.fields[key]).grid(row=row, column=1, sticky="ew", pady=5)
        ttk.Button(parent, text=button, command=command).grid(row=row, column=2, sticky="e", pady=5)

    def _calibration_row(self, parent: Any, row: int) -> None:
        """Show the existing file picker plus a one-click chessboard wizard."""
        ttk = self.app.ttk
        ttk.Label(parent, text="双目标定").grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=self.fields["calibration"]).grid(row=row, column=1, sticky="ew", pady=5)
        buttons = ttk.Frame(parent)
        buttons.grid(row=row, column=2, sticky="e", pady=5)
        ttk.Button(buttons, text="选择 JSON", command=self.browse_calibration).pack(side="left")
        ttk.Button(buttons, text="自动标定向导", command=self.auto_calibrate).pack(side="left", padx=(4, 0))

    def browse_model(self) -> None:
        selected = self.app.filedialog.askopenfilename(
            parent=self.window, filetypes=(("STL", "*.stl"), ("CAD", "*.3dm *.3mf"))
        )
        if selected:
            self.fields["model"].set(selected)
            self.scan()

    def browse_calibration(self) -> None:
        selected = self.app.filedialog.askopenfilename(parent=self.window, filetypes=(("JSON", "*.json"),))
        if selected:
            self.fields["calibration"].set(selected)

    def auto_calibrate(self) -> None:
        """Collect two folders and run the default checkerboard calibration."""
        try:
            left_dir = self.app.filedialog.askdirectory(parent=self.window, title="选择左目棋盘照片目录")
            if not left_dir:
                return
            right_dir = self.app.filedialog.askdirectory(parent=self.window, title="选择右目棋盘照片目录")
            if not right_dir:
                return
            from tkinter import simpledialog

            columns = simpledialog.askinteger("棋盘规格", "横向内角点数（默认 9）", initialvalue=9, minvalue=3, parent=self.window)
            rows = simpledialog.askinteger("棋盘规格", "纵向内角点数（默认 6）", initialvalue=6, minvalue=3, parent=self.window)
            square = simpledialog.askfloat("棋盘规格", "棋盘格边长（mm，默认 25）", initialvalue=25.0, minvalue=0.001, parent=self.window)
            if columns is None or rows is None or square is None:
                return
            from .calibration_wizard import calibrate_stereo_from_folders, write_calibration

            calibration = calibrate_stereo_from_folders(
                left_dir, right_dir, board_columns=columns, board_rows=rows,
                square_size_mm=square, calibration_id="FIELD-AUTO-STEREO",
            )
            destination = self.app.filedialog.asksaveasfilename(
                parent=self.window, title="保存自动生成的标定 JSON", defaultextension=".json",
                filetypes=(("JSON", "*.json"),), initialfile="stereo_calibration_auto.json",
            )
            if not destination:
                return
            write_calibration(destination, calibration)
            self.fields["calibration"].set(destination)
            quality = calibration["source_audit"]["auto_calibration"]
            self.message.set(
                f"自动标定完成：有效 {quality['accepted_pairs']}/{quality['candidate_pairs']} 对，"
                f"双目 RMS {quality['rms_stereo_px']:.3f}px。请继续用 QR 完成 CAD 配准。"
            )
        except (OSError, ValueError) as error:
            self.app.messagebox.showerror("自动标定失败", str(error), parent=self.window)

    def browse_photo(self, role: str) -> None:
        selected = self.app.filedialog.askopenfilename(
            parent=self.window, filetypes=(("照片", "*.png *.jpg *.jpeg"),)
        )
        if selected:
            self.fields[role].set(selected)
            self.fields[f"{role}_time"].set(self._photo_file_time(selected))

    def scan(self) -> None:
        try:
            from .capture_gui import catalog_from_model

            path = Path(self.fields["model"].get())
            self.pipes, skipped = catalog_from_model(
                path, stl_unit=self.stl_unit.get() if path.suffix.lower() == ".stl" else None
            )
            diameters = sorted({round(float(pipe["nominal_diameter_mm"]), 1) for pipe in self.pipes})
            self.color_rules.set(";".join(f"{diameter:g}={_PALETTE[i % len(_PALETTE)]}" for i, diameter in enumerate(diameters)))
            self.summary.set(f"已识别 {len(self.pipes)} 根直管；未纳入 {len(skipped)} 个组件。请确认直径颜色映射。")
            self.message.set("模型已读取。颜色按直径应用到所有同径管道；如需区分，请修改上面的映射。")
        except (OSError, ValueError) as error:
            self.app.messagebox.showerror("模型读取失败", str(error), parent=self.window)

    def _apply_colors(self) -> None:
        rules: dict[float, str] = {}
        for item in self.color_rules.get().split(";"):
            if not item.strip():
                continue
            try:
                diameter_text, color = item.split("=", 1)
                diameter = float(diameter_text.strip())
            except ValueError as error:
                raise ValueError("直径颜色格式应为 22=#E74C3C;50=#3498DB") from error
            if diameter <= 0 or re.fullmatch(r"#[0-9A-Fa-f]{6}", color.strip()) is None:
                raise ValueError("颜色必须是 #RRGGBB，直径必须为正数")
            rules[diameter] = color.strip().upper()
        if not rules:
            raise ValueError("请填写至少一组直径颜色映射")
        for pipe in self.pipes:
            diameter = float(pipe["nominal_diameter_mm"])
            matches = [(abs(diameter - key), value) for key, value in rules.items() if abs(diameter - key) <= 0.2]
            if not matches:
                raise ValueError(f"管道 {pipe['pipe_id']} 的直径 {diameter:g} mm 没有颜色映射")
            pipe["color_srgb"] = min(matches)[1]

    def capture_camera(self) -> None:
        try:
            calibration = self._load_calibration_json(self.fields["calibration"].get())
            problem = self._field_calibration_problem(calibration)
            if problem:
                raise ValueError(problem)
            from .capture_gui import StereoCameraDialog

            StereoCameraDialog(self, calibration)
        except (OSError, ValueError) as error:
            self.app.messagebox.showerror("无法连接双目", str(error), parent=self.window)

    def create(self) -> None:
        try:
            if not self.pipes:
                self.scan()
            self._apply_colors()
            calibration = self._load_calibration_json(self.fields["calibration"].get())
            problem = self._field_calibration_problem(calibration)
            if problem:
                raise ValueError(problem)
            left_time, _ = self._normalize_capture_time(self.fields["left_time"].get(), field="左目拍摄时间")
            right_time, _ = self._normalize_capture_time(self.fields["right_time"].get(), field="右目拍摄时间")
            from .capture_gui import create_capture_dataset
            from .measurement_gui import OUTPUT_ROOT

            manifest_path = create_capture_dataset(
                output_root=OUTPUT_ROOT / "captures",
                model_path=Path(self.fields["model"].get()),
                pipes=copy.deepcopy(self.pipes), calibration=calibration,
                left_path=Path(self.fields["left"].get()), right_path=Path(self.fields["right"].get()),
                left_time=left_time, right_time=right_time, pair_confirmed=True,
                previous_manifest=(self.app.manifest_path if self.app.manifest.get("validation_scope") == "FIELD_CAPTURE_PENDING_ACCEPTANCE" else None),
                stl_unit=self.stl_unit.get() if Path(self.fields["model"].get()).suffix.lower() == ".stl" else None,
                timestamp_sources=self.timestamp_sources,
                camera_capture_provenance=self.camera_capture_provenance,
            )
            self.app._load_sources(manifest_path, None)
            self.window.destroy()
            self.app._run_analysis()
        except (OSError, ValueError) as error:
            self.app.messagebox.showerror("评估失败", str(error), parent=self.window)


__all__ = ["QuickCaptureDialog"]
