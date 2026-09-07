"""Tk measurement workbench; importing this module does not create a GUI."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from .measurement_book import (
    KINDS, USES, build_measurement_summary, capture_signature, csv_text, empty_book,
    load_book, new_sample, save_book, scope_id,
)
from .metrology import measurement_settings
from .pipeline import atomic_write_text, ensure_paths_distinct


OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "measurement_workbench"


def fmt(value: Any, places: int = 2) -> str:
    return "—" if value is None else f"{value:.{places}f}" if isinstance(value, (int, float)) else str(value)


def image_rectangle(start: tuple, end: tuple, transform: tuple, size: tuple) -> list[int] | None:
    """Convert a dragged canvas rectangle to clipped original-image pixels."""
    scale, tx, ty = transform
    if scale <= 0:
        return None
    w, h = size
    xa, xb = sorted(((start[0] - tx) / scale, (end[0] - tx) / scale))
    ya, yb = sorted(((start[1] - ty) / scale, (end[1] - ty) / scale))
    x0, x1 = max(0, math.floor(xa)), min(w, math.ceil(xb))
    y0, y1 = max(0, math.floor(ya)), min(h, math.ceil(yb))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return [x0, y0, x1 - x0, y1 - y0]


REASONS = {
    "NO_BOUND_METRIC_REPORT": "尚无与当前输入匹配的自动测量结果",
    "INVALID_LOCAL_REPORT": "测量报告字段无效或与当前照片/标定不符，请重新分析",
    "INSUFFICIENT_VALID_3D_POINTS": "有效三维点不足",
    "INSUFFICIENT_LOCAL_COLOR_AND_DEPTH_SUPPORT": "局部颜色或有效深度不足",
    "CAPTURE_OR_CALIBRATION_NOT_READY": "照片质量、同步、标定或 CAD 配准未通过",
    "SURFACE_CURVATURE_NOT_RESOLVED": "表面曲率不足，无法确定真实外径",
    "CYLINDER_FIT_RESIDUAL_TOO_LARGE": "点云与圆柱拟合偏差过大",
    "LOCAL_AXIS_SUPPORT_TOO_SHORT": "局部轴向范围不足，请扩大选区",
    "INSUFFICIENT_VISIBLE_CYLINDER_ARC": "可见圆弧不足，外径待确认",
    "DIAMETER_VARIES_ACROSS_LOCAL_SECTIONS": "局部各截面外径不一致",
    "LEFT_RIGHT_METRIC_DISAGREEMENT": "左右目管径或位置不一致",
    "LEFT_RIGHT_AXIS_DISAGREEMENT": "左右目管轴方向不一致",
    "SHARED_OR_OVERLAPPING_OBSERVATION": "多根管件共享或重叠观测，身份待确认",
    "AMBIGUOUS_LOCAL_PIPE_IDENTITY": "存在多个相近候选，身份待确认",
    "OBSERVED_AXIS_OUTSIDE_ASSOCIATION_RANGE": "观测位置超出管件搜索范围",
    "OBSERVED_AXIS_DIRECTION_MISMATCH": "观测管轴方向与目标不符",
    "VIEWS_HAVE_NO_COMMON_SECTION": "左右目选区缺少共同截面",
    "NO_COMMON_OBSERVED_SECTION": "两根管件的可测范围缺少共同截面",
    "PAIR_REQUIRES_TWO_MEASURED_PIPES": "需要两根管件都有有效测量",
    "CLOSEST_APPROACH_OUTSIDE_OBSERVED_REGION": "轴线最近点位于可测局部之外",
    "NEGATIVE_GAP_CHECK_GEOMETRY": "净距为负，请检查碰撞或拟合结果",
}


def reason_text(code: str) -> str:
    if ":" in code:
        role, reason = code.split(":", 1)
        return {"LEFT": "左目", "RIGHT": "右目"}.get(role, role) + "：" + REASONS.get(reason, reason)
    return REASONS.get(code, code)


class ImageViewport:
    def __init__(self, parent: Any, panel: Any) -> None:
        self.panel = panel
        app = panel.app
        self.canvas = app.tk.Canvas(parent, background="#152235", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.source = None
        self.key = None
        self.photo = None
        self.zoom = 1.0
        self.pan = [0.0, 0.0]
        self.transform = (1.0, 0.0, 0.0)
        self.drag = None
        self.pan_start = None
        self.canvas.bind("<Configure>", lambda _e: self.draw())
        self.canvas.bind("<MouseWheel>", self._wheel)
        self.canvas.bind("<Button-4>", lambda e: self._wheel(e, 1.2))
        self.canvas.bind("<Button-5>", lambda e: self._wheel(e, 1 / 1.2))
        self.canvas.bind("<ButtonPress-1>", self._start)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._finish)
        self.canvas.bind("<ButtonPress-3>", self._pan_start)
        self.canvas.bind("<B3-Motion>", self._pan_move)

    def load(self, path: Path | None) -> None:
        try:
            key = (str(path), path.stat().st_mtime_ns, path.stat().st_size) if path else None
            if key != self.key:
                self.source = None
                self.key = key
                self.reset()
                if path:
                    import cv2
                    import numpy as np

                    self.source = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
                    if self.source is None:
                        raise ValueError("无法解码照片")
            self.draw()
        except (OSError, ValueError) as error:
            self.key = None
            self.source = None
            self.draw(str(error))

    def reset(self) -> None:
        self.zoom, self.pan = 1.0, [0.0, 0.0]
        self.draw()

    def draw(self, error: str = "") -> None:
        canvas = self.canvas
        canvas.delete("all")
        w, h = max(canvas.winfo_width(), 1), max(canvas.winfo_height(), 1)
        if self.source is None:
            canvas.create_text(w / 2, h / 2, text=error or "载入现场清单或打开合成示例\n选择管件后，观察局部测量范围",
                               fill="#C8D7E8", font=("Microsoft YaHei UI", 13), justify="center")
            return
        import cv2
        import numpy as np

        sh, sw = self.source.shape[:2]
        fit = min(w / sw, h / sh)
        scale = fit * self.zoom
        tx, ty = (w - sw * scale) / 2 + self.pan[0], (h - sh * scale) / 2 + self.pan[1]
        self.transform = (scale, tx, ty)
        shown = cv2.warpAffine(self.source, np.array([[scale, 0, tx], [0, scale, ty]]), (w, h),
                               flags=cv2.INTER_LINEAR, borderValue=(53, 34, 21))
        ok, encoded = cv2.imencode(".png", shown)
        if not ok:
            return
        self.photo = self.panel.app.tk.PhotoImage(data=base64.b64encode(encoded).decode("ascii"), format="png")
        canvas.create_image(0, 0, image=self.photo, anchor="nw")
        role, identity = self.panel.role.get(), self.panel.pipe_id.get()
        manual = self.panel.active_rois().get(identity, {}).get(role)
        selected = next((r for r in self.panel.summary.get("pipes", []) if r["pipe_id"] == identity), {})
        observed = selected.get("views", {}).get(role, {}).get("support_bbox_xywh")
        observed_color = "#62EDBB" if selected.get("measurement_status") == "MEASURED" else "#F3BE62"
        for box, color, label in ((manual, "#68B5FF", "手选局部范围"), (observed, observed_color, "局部点云范围")):
            if box:
                x, y, rw, rh = box
                canvas.create_rectangle(tx + x * scale, ty + y * scale, tx + (x + rw) * scale,
                                         ty + (y + rh) * scale, outline=color, width=2)
                canvas.create_text(tx + x * scale + 3, ty + y * scale + 3, anchor="nw", text=label,
                                   fill=color, font=("Microsoft YaHei UI", 10, "bold"))
        text = f"{'左' if role == 'left' else '右'}目 · {sw} × {sh} · 缩放 {self.zoom:.1f}×"
        if selected.get("raw_diameter_mm") is not None:
            text += f"   外径 {selected['raw_diameter_mm']:.2f} mm"
        references = [s for s in self.panel.book["samples"] if s["pipe_id"] == identity and s["kind"] == "diameter"
                      and s["capture_signature"] == capture_signature(self.panel.app.manifest)
                      and s["scope_id"] == scope_id(self.panel.app.manifest) and s["section_id"] == self.panel.section.get()]
        if references:
            text += f"   登记实测({references[-1]['section_id']}) {references[-1]['reference_mm']:.2f} mm"
        canvas.create_rectangle(0, 0, w, 29, fill="#152235", outline="")
        canvas.create_text(10, 14, text=text, anchor="w", fill="#F1F6FC", font=("Microsoft YaHei UI", 10))

    def _wheel(self, event: Any, factor: float | None = None) -> None:
        if self.source is None:
            return
        factor = factor or (1.2 if event.delta > 0 else 1 / 1.2)
        old = self.zoom
        self.zoom = min(12.0, max(0.5, old * factor))
        ratio = self.zoom / old
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.pan = [(self.pan[0] - event.x + w / 2) * ratio + event.x - w / 2,
                    (self.pan[1] - event.y + h / 2) * ratio + event.y - h / 2]
        self.draw()

    def _start(self, event: Any) -> None:
        if self.source is not None and self.panel.mark_region.get() and self.panel.pipe_id.get():
            self.drag = (event.x, event.y)

    def _drag(self, event: Any) -> None:
        if self.drag:
            self.canvas.delete("drag_roi")
            self.canvas.create_rectangle(*self.drag, event.x, event.y, outline="#68B5FF", width=2, tags="drag_roi")

    def _finish(self, event: Any) -> None:
        if self.drag and self.source is not None:
            box = image_rectangle(self.drag, (event.x, event.y), self.transform, self.source.shape[1::-1])
            self.drag = None
            if box:
                self.panel.set_roi(box)
            self.draw()

    def _pan_start(self, event: Any) -> None:
        self.pan_start = (event.x, event.y, *self.pan)

    def _pan_move(self, event: Any) -> None:
        if self.pan_start:
            x, y, px, py = self.pan_start
            self.pan = [px + event.x - x, py + event.y - y]
            self.draw()


class MeasurementPanel:
    def __init__(self, parent: Any, app: Any) -> None:
        self.app = app
        tk, ttk = app.tk, app.ttk
        self.book = empty_book()
        self.book_path = None
        self.active_scope = None
        self.explicit_book = False
        self.summary = {"pipes": [], "pairs": []}
        self.pipe_id, self.neighbor = tk.StringVar(), tk.StringVar()
        self.kind, self.use = tk.StringVar(value="外径"), tk.StringVar(value="登记对照")
        self.reference, self.section, self.notes = tk.StringVar(), tk.StringVar(value="S01"), tk.StringVar()
        self.tolerance = tk.StringVar(value="1.0")
        self.role = tk.StringVar(value="left")
        self.mark_region, self.only_related = tk.BooleanVar(value=False), tk.BooleanVar(value=True)
        self.overview, self.raw_label, self.correction_label, self.notice = [tk.StringVar() for _ in range(4)]
        self.notice.set("先载入数据，再运行分析；所有长度单位为 mm。")
        top = ttk.Frame(parent, padding=(10, 8))
        top.pack(fill="x")
        ttk.Label(top, textvariable=self.overview, font=("Microsoft YaHei UI", 11, "bold")).pack(side="left")
        ttk.Button(top, text="导出逐管状态", command=self.export).pack(side="right", padx=3)
        ttk.Button(top, text="工作簿另存", command=self.save_as).pack(side="right", padx=3)
        ttk.Button(top, text="打开工作簿", command=self.open_book).pack(side="right", padx=3)
        panes = ttk.Panedwindow(parent, orient="vertical")
        panes.pack(fill="both", expand=True, padx=10)
        upper = ttk.Frame(panes)
        lower = ttk.Frame(panes)
        panes.add(upper, weight=3)
        panes.add(lower, weight=2)
        visual = ttk.Frame(upper)
        visual.pack(side="left", fill="both", expand=True, padx=(0, 10))
        controls = ttk.Frame(visual)
        controls.pack(fill="x", pady=(0, 5))
        view = ttk.Combobox(controls, textvariable=self.role, values=("left", "right"), state="readonly", width=7)
        view.pack(side="left")
        view.bind("<<ComboboxSelected>>", lambda _e: self.update_photo())
        ttk.Checkbutton(controls, text="拖框选局部", variable=self.mark_region).pack(side="left", padx=8)
        ttk.Button(controls, text="清除本管选区", command=self.clear_roi).pack(side="left", padx=3)
        ttk.Button(controls, text="适应窗口", command=lambda: self.viewport.reset()).pack(side="left", padx=3)
        ttk.Label(controls, text="滚轮缩放 · 右键拖动").pack(side="right", padx=5)
        self.viewport = ImageViewport(visual, self)
        form = ttk.LabelFrame(upper, text="实测登记与二次校正", padding=10)
        form.pack(side="right", fill="y")
        form.columnconfigure(1, weight=1)

        def combo(label: str, variable: Any, values: tuple, row: int, width: int = 30) -> Any:
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 8))
            widget = ttk.Combobox(form, textvariable=variable, values=values, state="readonly", width=width)
            widget.grid(row=row, column=1, sticky="ew", pady=3)
            widget.bind("<<ComboboxSelected>>", lambda _e: self.selection_changed())
            return widget

        self.pipe_combo = combo("当前管件", self.pipe_id, (), 0)
        self.pipe_combo.bind("<<ComboboxSelected>>", self._choose_pipe)
        self.neighbor_combo = combo("另一根管件", self.neighbor, (), 1)
        combo("登记项目", self.kind, tuple(KINDS.values()), 2)
        for label, variable, row in (("实测值 / mm", self.reference, 3), ("局部截面编号", self.section, 4)):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(form, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=3)
        combo("样本用途", self.use, tuple(USES.values()), 5)
        ttk.Label(form, text="备注").grid(row=6, column=0, sticky="w", pady=3)
        ttk.Entry(form, textvariable=self.notes).grid(row=6, column=1, sticky="ew", pady=3)
        ttk.Label(form, textvariable=self.raw_label, foreground="#175D86").grid(row=7, column=0, columnspan=2, sticky="w", pady=5)
        actions = ttk.Frame(form)
        actions.grid(row=8, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(actions, text="添加实测样本", command=self.add_sample).pack(side="left")
        ttk.Button(actions, text="更新所选样本", command=self.update_sample).pack(side="left", padx=5)
        tolerance_row = ttk.Frame(form)
        tolerance_row.grid(row=9, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(tolerance_row, text="尺寸/位置对照阈值 ±").pack(side="left")
        ttk.Entry(tolerance_row, textvariable=self.tolerance, width=6).pack(side="left", padx=4)
        ttk.Label(tolerance_row, text="mm").pack(side="left")
        ttk.Button(tolerance_row, text="应用", command=self.apply_tolerance).pack(side="left", padx=6)
        ttk.Label(form, textvariable=self.correction_label, wraplength=360, foreground="#505B6B").grid(row=10, column=0, columnspan=2, sticky="w", pady=4)

        self.tabs = ttk.Notebook(lower)
        self.tabs.pack(fill="both", expand=True, pady=(9, 0))
        pipe_tab, pair_tab, sample_tab, detail_tab = [ttk.Frame(self.tabs) for _ in range(4)]
        for frame, title in ((pipe_tab, "逐管当前状态"), (pair_tab, "中心距 · 净距 · 前后"),
                             (sample_tab, "实测样本"), (detail_tab, "所选管件分析")):
            self.tabs.add(frame, text=title)
        self.pipe_tree = self._table(pipe_tab, [
            ("pipe_id", "管件编号", 220), ("installation_label", "安装状态", 85),
            ("nominal_diameter_mm", "设计外径", 83), ("raw_diameter_mm", "自动外径", 83),
            ("corrected_diameter_mm", "校正外径", 83), ("diameter_error_mm", "外径偏差", 83),
            ("position_error_mm", "轴线位置偏差", 100), ("camera_depth_mm", "相机深度", 95),
            ("comparison_label", "尺寸/位置对照", 180)])
        self.pipe_tree.bind("<<TreeviewSelect>>", self._tree_pipe)
        self.pipe_tree.bind("<Double-1>", lambda _e: self.tabs.select(detail_tab))
        pair_controls = ttk.Frame(pair_tab)
        pair_controls.pack(fill="x")
        ttk.Checkbutton(pair_controls, text="仅显示与所选管件有关的组合", variable=self.only_related, command=self._fill_pairs).pack(side="left")
        ttk.Label(pair_controls, text="中心距为三维局部轴距；净距扣除两管半径；前后以左相机为准。").pack(side="left", padx=16)
        self.pair_tree = self._table(pair_tab, [(k, label, width) for k, label, width in (
            ("pipe_id_a", "管件 A", 220), ("pipe_id_b", "管件 B", 220), ("center_distance_mm", "中心距/mm", 100),
            ("clear_gap_mm", "原始净距/mm", 110), ("corrected_clear_gap_mm", "校正净距/mm", 110),
            ("depth_delta_b_minus_a_mm", "深度差 B−A/mm", 125), ("order_label", "前后判定", 250))])
        ttk.Button(sample_tab, text="删除所选样本", command=self.delete_sample).pack(anchor="w", pady=3)
        self.sample_tree = self._table(sample_tab, [(k, label, width) for k, label, width in (
            ("sample_id", "样本", 100), ("pipe_id", "管件 A", 195), ("pipe_id_b", "管件 B", 195),
            ("kind_label", "项目", 65), ("raw_mm", "自动原始/mm", 95), ("reference_mm", "实测/mm", 90), ("raw_error_mm", "原始误差/mm", 100),
            ("use_label", "用途", 90), ("section_id", "截面", 60), ("capture_id", "照片组", 130), ("notes", "备注", 150))])
        self.sample_tree.bind("<<TreeviewSelect>>", self.select_sample)
        self.detail = tk.Text(detail_tab, wrap="word", font=("Microsoft YaHei UI", 10), padx=12, pady=8, height=10, state="disabled")
        self.detail.pack(fill="both", expand=True)
        ttk.Label(parent, textvariable=self.notice, padding=(10, 6), foreground="#344A64").pack(fill="x")

    def _table(self, parent: Any, fields: list) -> Any:
        ttk = self.app.ttk
        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        table = ttk.Treeview(frame, columns=[f[0] for f in fields], show="headings", selectmode="browse", height=8)
        for key, label, width in fields:
            table.heading(key, text=label)
            table.column(key, width=width, minwidth=60, anchor="w" if width > 120 else "center", stretch=width > 120)
        sy, sx = ttk.Scrollbar(frame, orient="vertical", command=table.yview), ttk.Scrollbar(frame, orient="horizontal", command=table.xview)
        table.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        table.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        table.tag_configure("warning", foreground="#9A5500")
        table.tag_configure("good", foreground="#187353")
        return table

    def _fill(self, table: Any, rows: list, id_key: str | None = None) -> None:
        table.delete(*table.get_children())
        for index, row in enumerate(rows):
            tag = "good" if row.get("measurement_status") == "MEASURED" or row.get("status") == "MEASURED" else "warning"
            table.insert("", "end", iid=row[id_key] if id_key else str(index),
                         values=[fmt(row.get(key)) for key in table["columns"]], tags=(tag,))

    def active_rois(self) -> dict:
        return self.book["settings"]["rois"] if self.book.get("roi_context") == capture_signature(self.app.manifest) else {}

    def analysis_options(self) -> dict:
        config = copy.deepcopy(self.book["settings"])
        config["tolerance_mm"] = float(self.tolerance.get())
        config["rois"] = copy.deepcopy(self.active_rois())
        return measurement_settings(config)

    def refresh(self) -> None:
        scope = scope_id(self.app.manifest)
        if scope != self.active_scope:
            self.active_scope = scope
            if not self.explicit_book:
                path = OUTPUT_ROOT / f"samples_{scope[:16]}.json"
                try:
                    self.book = load_book(path) if path.exists() else empty_book()
                    self.book_path = path
                    self.tolerance.set(str(self.book["settings"]["tolerance_mm"]))
                except (OSError, ValueError) as error:
                    self.book = empty_book()
                    # Preserve the unreadable file for recovery.
                    self.book_path = OUTPUT_ROOT / f"samples_recovery_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
                    self.notice.set(f"工作簿未载入：{error}；新登记将保存到恢复文件。")
        try:
            tolerance = self.analysis_options()["tolerance_mm"]
        except ValueError:
            tolerance = self.book["settings"]["tolerance_mm"]
        report = self.app.report
        metric = report.get("local_measurements") if report else None
        if (isinstance(metric, dict) and isinstance(metric.get("settings"), dict)
                and metric["settings"].get("rois", {}) != self.active_rois()):
            report = {key: value for key, value in report.items() if key != "local_measurements"}
        self.summary = build_measurement_summary(self.app.manifest, report, self.app.dashboard, self.book, tolerance)
        ids = [r["pipe_id"] for r in self.summary["pipes"]]
        self.pipe_combo.configure(values=ids)
        self.neighbor_combo.configure(values=ids)
        chosen = self.app.selected_pipe_id
        self.pipe_id.set(chosen if chosen in ids else ids[0] if ids else "")
        if self.neighbor.get() not in ids or self.neighbor.get() == self.pipe_id.get():
            self.neighbor.set(next((i for i in ids if i != self.pipe_id.get()), ""))
        measured = sum(r["measurement_status"] == "MEASURED" for r in self.summary["pipes"])
        scope_label = "合成示例" if "SYNTHETIC" in self.app.manifest.get("validation_scope", "") else "现场数据"
        self.overview.set(f"{scope_label if ids else '测量工作台'}   管件 {len(ids)}   有效局部测量 {measured}   对照阈值 ±{tolerance:g} mm")
        self._fill(self.pipe_tree, self.summary["pipes"], "pipe_id")
        sample_rows = [s | {"kind_label": KINDS[s["kind"]], "use_label": USES[s["use_for"]],
                          "raw_error_mm": s["raw_mm"] - s["reference_mm"] if s["raw_mm"] is not None else None}
                       for s in self.book["samples"]]
        self._fill(self.sample_tree, sample_rows, "sample_id")
        correction = self.summary["correction"]
        self.correction_label.set(correction["label"] + (f"；外径修正 {correction['offset_mm']:+.3f} mm" if correction["offset_mm"] is not None else "")
                                  + f"；独立验证 {correction['validation_count']} 条。")
        self.selection_changed()

    def selection_changed(self) -> None:
        identity = self.pipe_id.get()
        if identity and self.pipe_tree.exists(identity) and self.pipe_tree.selection() != (identity,):
            self.pipe_tree.selection_set(identity)
            self.pipe_tree.see(identity)
        value = self._raw_value()
        self.raw_label.set(f"当前自动值：{fmt(value)} mm" + ("（尚无有效测量）" if value is None else ""))
        self._fill_pairs()
        row = next((r for r in self.summary["pipes"] if r["pipe_id"] == identity), None)
        lines = ["选择管件后查看逐管分析。"]
        if row:
            lines = [f"{identity}  ·  {row['current_status']}",
                f"设计外径 {fmt(row['nominal_diameter_mm'])} mm；自动外径 {fmt(row['raw_diameter_mm'])} mm；校正外径 {fmt(row['corrected_diameter_mm'])} mm。",
                f"局部轴线相对设计位置偏差：{fmt(row['position_error_mm'])} mm。",
                "局部中心 XYZ（CAD 世界坐标，mm）：" + (", ".join(fmt(v) for v in row["center_world_mm"]) if row["center_world_mm"] else "待测量"),
                "测量说明：" + ("；".join(reason_text(c) for c in row["measurement_reasons"]) or "左右局部拟合通过；现场精度仍需实测验证。"),
                "安装依据：" + ", ".join(row["installation_reasons"]), "", "与其他管件的关系："]
            lines += [f"• {r['other_pipe_id']}：中心距 {r['center_distance_mm']:.2f} mm，净距 {r['clear_gap_mm']:.2f} mm，当前管件{r['order']}。" for r in row["relations"]]
            if not row["relations"]:
                lines.append("暂无两管均可测且局部范围相交的关系，不能自动确认间距或前后。")
        self.detail.configure(state="normal")
        self.detail.delete("1.0", "end")
        self.detail.insert("1.0", "\n".join(lines))
        self.detail.configure(state="disabled")
        self.update_photo()

    def _fill_pairs(self) -> None:
        rows = self.summary["pairs"]
        if self.only_related.get():
            rows = [r for r in rows if self.pipe_id.get() in (r["pipe_id_a"], r["pipe_id_b"])]
        self._fill(self.pair_tree, rows)

    def _choose_pipe(self, _event: Any = None) -> None:
        if self.pipe_id.get() and self.pipe_id.get() != self.app.selected_pipe_id:
            self.app._select_pipe(self.pipe_id.get())
        else:
            self.selection_changed()

    def _tree_pipe(self, _event: Any) -> None:
        selection = self.pipe_tree.selection()
        if selection and selection[0] != self.pipe_id.get():
            self.pipe_id.set(selection[0])
            self._choose_pipe()

    def update_photo(self) -> None:
        from .gui import _first_manifest_stereo_paths

        paths = _first_manifest_stereo_paths(self.app.manifest_path, self.app.manifest) if self.app.manifest_path else {}
        self.viewport.load(paths.get(self.role.get()))

    def _raw_value(self) -> float | None:
        kind = next(k for k, label in KINDS.items() if label == self.kind.get())
        if kind == "diameter":
            return next((r["raw_diameter_mm"] for r in self.summary["pipes"] if r["pipe_id"] == self.pipe_id.get()), None)
        pair = next((r for r in self.summary["pairs"] if {r["pipe_id_a"], r["pipe_id_b"]} == {self.pipe_id.get(), self.neighbor.get()}), {})
        return pair.get(kind + "_mm") if pair.get("status") == "MEASURED" else None

    def _protect_output(self, path: Path) -> None:
        from .gui import _first_manifest_stereo_paths, _safe_manifest_asset_path

        sources = [self.app.manifest_path, self.app.report_path]
        if self.app.manifest_path:
            sources.extend(_first_manifest_stereo_paths(self.app.manifest_path, self.app.manifest).values())
            sources.append(_safe_manifest_asset_path(self.app.manifest_path, self.app.manifest.get("model", {}).get("path")))
        unique = {str(Path(p).resolve()): p for p in sources if p}
        ensure_paths_distinct(output=path, **{f"source_{i}": p for i, p in enumerate(unique.values())})

    def _check_inputs(self) -> None:
        if not self.app.manifest_path:
            return
        from .gui import _loaded_asset_hashes

        try:
            manifest_hash, model_hash = _loaded_asset_hashes(self.app.manifest_path, self.app.manifest)
            changed = (manifest_hash != self.app._manifest_sha256 or model_hash != self.app._model_actual_sha256
                       or self.app._current_photo_hashes() != self.app._photo_actual_sha256)
        except (OSError, ValueError):
            changed = True
        if changed:
            self.app._clear_loaded_state(manifest=self.app.manifest, manifest_path=self.app.manifest_path,
                reason_code="INPUT_CHANGED", message="输入文件已改变，请重新载入并分析。", preserve_manifest=True)
            raise ValueError("模型、清单或照片已改变。旧结果已清除，请重新载入并分析。")

    def _store(self, book: dict) -> None:
        if self.book_path is None:
            self.book_path = OUTPUT_ROOT / f"samples_{scope_id(self.app.manifest)[:16]}.json"
        self._protect_output(self.book_path)
        save_book(self.book_path, book)
        self.book = book
        self.notice.set(f"已保存：{self.book_path}")
        self.refresh()

    def set_roi(self, box: list) -> None:
        try:
            book = copy.deepcopy(self.book)
            if book.get("roi_context") != capture_signature(self.app.manifest):
                book["settings"]["rois"] = {}
            book["roi_context"] = capture_signature(self.app.manifest)
            book["settings"]["rois"].setdefault(self.pipe_id.get(), {})[self.role.get()] = box
            self._store(book)
            self.notice.set("局部选区已保存；点击“运行双目识别与测量”更新自动结果。选区需包含管件两侧及一小段管轴。")
        except (OSError, ValueError) as error:
            self.app.messagebox.showerror("选区保存失败", str(error))

    def clear_roi(self) -> None:
        try:
            book = copy.deepcopy(self.book)
            book["settings"]["rois"].pop(self.pipe_id.get(), None)
            self._store(book)
            self.notice.set("已清除本管左右选区；重新分析时自动搜索局部范围。")
        except (OSError, ValueError) as error:
            self.app.messagebox.showerror("保存失败", str(error))

    def apply_tolerance(self) -> None:
        try:
            config = self.analysis_options()
            book = copy.deepcopy(self.book)
            book["settings"]["tolerance_mm"] = config["tolerance_mm"]
            self._store(book)
        except (ValueError, OSError) as error:
            self.app.messagebox.showerror("阈值无效", str(error))

    def add_sample(self) -> None:
        try:
            self._check_inputs()
            kind = next(k for k, label in KINDS.items() if label == self.kind.get())
            use = next(k for k, label in USES.items() if label == self.use.get())
            if use == "calibration" and kind != "diameter":
                raise ValueError("当前二次校正用于外径；中心距和净距请使用“登记对照”或“独立验证”。")
            sample = new_sample(self.app.manifest, pipe_id=self.pipe_id.get(), kind=kind,
                reference_mm=float(self.reference.get()), section_id=self.section.get().strip(), use_for=use,
                pipe_id_b=self.neighbor.get() if kind != "diameter" else "", raw_mm=self._raw_value(), notes=self.notes.get(),
                measurement_settings_id=hashlib.sha256(json.dumps(self.summary.get("measurement_settings"), sort_keys=True).encode()).hexdigest())
            book = copy.deepcopy(self.book)
            book["samples"].append(sample)
            self._store(book)
            self.reference.set("")
            if sample["raw_mm"] is None:
                self.notice.set("实测值已登记。该照片尚无有效自动值，此条不参与校正；分析成功后可重新添加带自动值的样本。")
        except (OSError, ValueError) as error:
            self.app.messagebox.showerror("无法添加样本", str(error))

    def select_sample(self, _event: Any = None) -> None:
        selected = self.sample_tree.selection()
        if selected:
            sample = next(s for s in self.book["samples"] if s["sample_id"] == selected[0])
            self.reference.set(str(sample["reference_mm"]))
            self.use.set(USES[sample["use_for"]])
            self.notes.set(sample["notes"])
            self.notice.set(f"已选样本 {sample['sample_id']}（{sample['pipe_id']} / {KINDS[sample['kind']]}）。更新仅修改实测值、用途和备注，保留原照片来源。")

    def update_sample(self) -> None:
        try:
            selected = self.sample_tree.selection()
            if not selected:
                raise ValueError("请先在“实测样本”页选择要修改的样本。")
            book = copy.deepcopy(self.book)
            sample = next(s for s in book["samples"] if s["sample_id"] == selected[0])
            use = next(k for k, label in USES.items() if label == self.use.get())
            if use == "calibration" and sample["kind"] != "diameter":
                raise ValueError("二次校正仅用于外径。")
            sample.update(reference_mm=float(self.reference.get()), use_for=use, notes=self.notes.get())
            self._store(book)
        except (OSError, ValueError) as error:
            self.app.messagebox.showerror("更新失败", str(error))

    def delete_sample(self) -> None:
        selected = self.sample_tree.selection()
        if selected:
            try:
                book = copy.deepcopy(self.book)
                book["samples"] = [s for s in book["samples"] if s["sample_id"] != selected[0]]
                self._store(book)
            except (OSError, ValueError) as error:
                self.app.messagebox.showerror("删除失败", str(error))

    def open_book(self) -> None:
        selected = self.app.filedialog.askopenfilename(title="打开实测工作簿", filetypes=(("JSON", "*.json"),))
        if selected:
            try:
                book = load_book(selected)
                self._protect_output(Path(selected))
                self.book, self.book_path, self.explicit_book = book, Path(selected), True
                self.tolerance.set(str(book["settings"]["tolerance_mm"]))
                self.refresh()
                self.notice.set(f"已打开 {selected}；仅相同标定配置的样本参与校正。")
            except (ValueError, OSError) as error:
                self.app.messagebox.showerror("工作簿载入失败", str(error))

    def save_as(self) -> None:
        selected = self.app.filedialog.asksaveasfilename(title="保存实测工作簿", defaultextension=".json", initialfile="实测工作簿.json", filetypes=(("JSON", "*.json"),))
        if selected:
            try:
                path = Path(selected)
                self._protect_output(path)
                save_book(path, self.book)
                self.book_path, self.explicit_book = path, True
                self.notice.set(f"已保存：{path}")
            except (OSError, ValueError) as error:
                self.app.messagebox.showerror("工作簿保存失败", str(error))

    def export_to(self, directory: Path) -> Path:
        self._check_inputs()
        self.refresh()
        if not self.summary["pipes"]:
            raise ValueError("请先载入包含管件的清单。")
        folder = directory / f"管件状态_{datetime.now():%Y%m%d_%H%M%S_%f}"
        folder.mkdir(parents=True, exist_ok=False)
        atomic_write_text(folder / "逐管状态.json", json.dumps(self.summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        atomic_write_text(folder / "逐管状态.csv", csv_text(self.summary["pipes"], [
            "pipe_id", "installation_label", "current_status", "nominal_diameter_mm", "raw_diameter_mm",
            "corrected_diameter_mm", "diameter_error_mm", "position_error_mm", "center_world_mm",
            "camera_depth_mm", "measurement_reasons", "relations"]))
        atomic_write_text(folder / "管件间距与前后.csv", csv_text(self.summary["pairs"], [
            "pipe_id_a", "pipe_id_b", "status", "center_distance_mm", "clear_gap_mm", "corrected_clear_gap_mm",
            "depth_delta_b_minus_a_mm", "order_label", "distance_definition", "reason_codes"]))
        save_book(folder / "实测工作簿.json", self.book)
        if self.app.report and self.summary["binding_valid"]:
            atomic_write_text(folder / "双目原始报告.json", json.dumps(self.app.report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        lines = ["# 管件当前状态分析", "", f"数据集：{self.summary['dataset_id']}。单位 mm；尺寸/位置对照阈值 ±{self.summary['tolerance_mm']:g} mm。", "",
                 self.summary["notes"], "", "| 管件 | 当前状态 | 设计外径 | 自动外径 | 校正外径 | 位置偏差 |", "|---|---|---:|---:|---:|---:|"]
        for row in self.summary["pipes"]:
            lines.append("| " + " | ".join(fmt(row.get(k)).replace("|", "/").replace("\n", " ") for k in (
                "pipe_id", "current_status", "nominal_diameter_mm", "raw_diameter_mm", "corrected_diameter_mm", "position_error_mm")) + " |")
        for row in self.summary["pipes"]:
            lines.extend(["", f"## {row['pipe_id']}", "", row["current_status"], "",
                "测量说明：" + ("；".join(reason_text(c) for c in row["measurement_reasons"]) or "局部拟合通过，现场精度待实测验证。")])
            if row["relations"]:
                lines.append("")
            lines += [f"- 与 {r['other_pipe_id']}：中心距 {r['center_distance_mm']:.2f} mm，净距 {r['clear_gap_mm']:.2f} mm，{r['order']}。" for r in row["relations"]]
        atomic_write_text(folder / "逐管分析.md", "\n".join(lines) + "\n")
        return folder

    def export(self) -> None:
        selected = self.app.filedialog.askdirectory(title="选择状态报告的保存目录")
        if selected:
            try:
                folder = self.export_to(Path(selected))
                self.notice.set(f"逐管状态、间距、前后关系和实测样本已导出：{folder}")
            except (ValueError, OSError) as error:
                self.app.messagebox.showerror("导出失败", str(error))
