"""Interactive independent pipe patches for color and highlight diagnostics."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from .capture_quality import (
    LEVEL_TEXT, REVISION, diagnostic_record, measure_patch, patch_mask,
    summarize_pair, validate_color,
)
from .elevation_gui import RegionCanvas, image_region
from .pipeline import atomic_write_text
from .logging_config import get_logger, log_event

COLORS = {"GREEN": "#197344", "AMBER": "#956000", "RED": "#B3261E", "PENDING": "#526477"}
LOGGER = get_logger("capture_quality_gui")


class PatchCanvas(RegionCanvas):
    """Manual diagnostic selection, independent of automatic recognition ROIs."""

    def __init__(self, parent, owner, role):
        super().__init__(parent, owner, role)
        self.drag = None
        self.canvas.bind("<ButtonPress-1>", self.start)
        self.canvas.bind("<B1-Motion>", self.move)
        self.canvas.bind("<ButtonRelease-1>", self.finish)

    def start(self, event):
        if self.image is not None and self.owner.selected_id():
            self.drag = (event.x, event.y)

    def move(self, event):
        if self.drag is not None:
            self.canvas.delete("drag")
            self.canvas.create_rectangle(*self.drag, event.x, event.y, outline="white", dash=(4, 2), tags="drag")

    def finish(self, event):
        start, self.drag = self.drag, None
        self.canvas.delete("drag")
        if start is not None and self.image is not None:
            region = image_region(start, (event.x, event.y), self.transform, (self.image.shape[1], self.image.shape[0]))
            if region is not None:
                self.owner.set_region(self.role, region)


def metric_text(result: dict | None, kind: str) -> str:
    field = "color_coverage_percent" if kind == "color" else "highlight_risk_percent"
    if not result or result.get(field) is None:
        return "待框选"
    return f"{result[field]:.1f}% · {LEVEL_TEXT[result[kind + '_level']]}"


class CaptureQualityDialog:
    """Regions belong to physical samples, never to authoritative CAD identities."""

    def __init__(self, owner):
        self.owner, self.app = owner, owner.app
        tk, ttk = self.app.tk, self.app.ttk
        self.pipes = []  # RegionCanvas adapter: diagnostic samples, not model pipes.
        self.results, self.report = {}, None
        self.busy, self.closed = False, False
        self.photo_hashes = {}
        self.delta_lab = 45.0
        self.policy_valid = True
        self.policy_traces = []
        self.policy_status = tk.StringVar()
        self.window = tk.Toplevel(owner.window)
        self.window.title("颜色与反光检查 · 调整灯光、曝光和参考色")
        self.window.geometry("1120x800")
        self.window.minsize(980, 720)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.message = tk.StringVar()
        self.advice = tk.StringVar()
        self.name = tk.StringVar()
        self.reference = tk.StringVar()
        self.choices = {}
        frame = ttk.Frame(self.window, padding=12)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="① 添加管面取样 → ② 左右各拖框选一小块同一管段 → ③ 看 C / H 并调整 → ④ 重新抓拍比较。",
                  foreground="#355371").pack(anchor="w")
        ttk.Label(frame, text="框内只保留可见管面，排除背景和支架；不必先识别成功。局部取样不代表整根管道，参考外径不确认模型管号。",
                  foreground="#355371").pack(anchor="w", pady=(2, 6))
        bar = ttk.Frame(frame)
        bar.pack(fill="x")
        ttk.Button(bar, text="添加管面取样", command=self.add_sample).pack(side="left")
        ttk.Button(bar, text="删除取样", command=self.remove_sample).pack(side="left", padx=4)
        ttk.Label(bar, text="名称").pack(side="left")
        name_entry = ttk.Entry(bar, textvariable=self.name, width=16)
        name_entry.pack(side="left", padx=4)
        name_entry.bind("<Return>", self.rename_sample)
        name_entry.bind("<FocusOut>", self.rename_sample)
        ttk.Label(bar, text="模型参考色").pack(side="left", padx=(10, 3))
        self.reference_box = ttk.Combobox(bar, textvariable=self.reference, values=(), state="readonly", width=23)
        self.reference_box.pack(side="left")
        self.reference_box.bind("<<ComboboxSelected>>", self.choose_reference)
        ttk.Button(bar, text="自选参考色", command=self.choose_color).pack(side="left", padx=4)
        panes = ttk.Panedwindow(frame, orient="horizontal")
        panes.pack(fill="both", expand=True, pady=8)
        self.views = {}
        for role, title in (("left", "左目管面取样"), ("right", "右目管面取样")):
            box = ttk.LabelFrame(panes, text=title, padding=3)
            panes.add(box, weight=1)
            self.views[role] = PatchCanvas(box, self, role)
        ttk.Label(frame, text="滚轮放大，右键拖动。框选后自动计算；换照片会清除旧框。颜色与高光均按左右较差值提示。",
                  foreground="#355371").pack(anchor="w")
        columns = ("color", "cl", "cr", "c", "hl", "hr", "h")
        self.tree = ttk.Treeview(frame, columns=columns, height=4, selectmode="browse")
        self.tree.heading("#0", text="管面取样")
        self.tree.column("#0", width=145, stretch=True)
        for col, text in zip(columns, ("参考色", "C 左", "C 右", "C 较差 ↑", "H 左", "H 右", "H 较差 ↓")):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=118 if col in {"c", "h"} else 102, stretch=False)
        self.tree.pack(fill="x", pady=6)
        self.tree.bind("<<TreeviewSelect>>", self.selected_changed)
        row = ttk.Frame(frame)
        row.pack(fill="x")
        self.color_label = ttk.Label(row, text="C 待框选", font=("Microsoft YaHei UI", 11, "bold"))
        self.color_label.pack(side="left", padx=(0, 18))
        self.highlight_label = ttk.Label(row, text="H 待框选", font=("Microsoft YaHei UI", 11, "bold"))
        self.highlight_label.pack(side="left")
        ttk.Label(frame, textvariable=self.advice, wraplength=1050, foreground="#355371").pack(fill="x", pady=5)
        ttk.Label(frame, textvariable=self.policy_status, wraplength=1050, foreground="#526477").pack(fill="x")
        actions = ttk.Frame(frame)
        actions.pack(fill="x")
        ttk.Button(actions, text="取左框非高光中位色", command=self.sample_color).pack(side="left")
        ttk.Button(actions, text="应用参考色到同径模型管", command=self.apply_color).pack(side="left", padx=4)
        ttk.Button(actions, text="清除左右检查框", command=self.clear_regions).pack(side="left", padx=4)
        ttk.Button(actions, text="重新抓拍", command=owner.capture_camera).pack(side="left", padx=4)
        ttk.Button(actions, text="保存检查记录", command=self.save).pack(side="right")
        ttk.Label(frame, text="C：≥60% 绿，20～60% 黄，<20% 红；H：≤1% 绿，1～5% 黄，>5% 红。初始建议阈值，仅供调整，不等于量测通过。",
                  foreground="#526477", wraplength=1050).pack(anchor="w", pady=(7, 0))
        ttk.Label(frame, text="H 为多通道接近饱和的风险，不是反光面积；白管漫反射也可能偏高。取样偏暗时先处理欠曝光。",
                  foreground="#526477", wraplength=1050).pack(anchor="w")
        ttk.Label(frame, textvariable=self.message, foreground="#355371", wraplength=1050).pack(fill="x", pady=4)
        self.refresh_inputs()
        for variable in (owner.color_delta_lab, owner.color_filter_enabled, owner.local_observation_label):
            self.policy_traces.append((variable, variable.trace_add("write", lambda *_: self.recalculate())))

    def selected_id(self):
        selected = self.tree.selection()
        return str(selected[0]) if selected else None

    def selected_sample(self):
        return next((p for p in self.pipes if p["pipe_id"] == self.selected_id()), None)

    def clear_inputs(self):
        self.photo_hashes = {}
        for sample in self.pipes:
            for role in ("left", "right"):
                sample.pop(f"{role}_region_px", None)
            sample.pop("metrics", None)
        for view in self.views.values():
            view.set_image(None)
        self.refresh_table()
        self.message.set("照片输入已变化，请载入或重新抓拍左右照片后框选。")

    def refresh_inputs(self):
        self.choices = {f"{p['nominal_diameter_mm']:g} mm · {p['color_srgb']}":
                        (p["nominal_diameter_mm"], p["color_srgb"]) for p in self.owner.pipes}
        self.reference_box.configure(values=tuple(self.choices))
        selected = self.owner.selected_pipe()
        if selected:
            self.reference.set(f"{selected['nominal_diameter_mm']:g} mm · {selected['color_srgb']}")
        if any(self.owner.views[r].image is None or r not in self.owner.image_hashes for r in ("left", "right")):
            self.clear_inputs()
            return
        hashes = dict(self.owner.image_hashes)
        changed = hashes != self.photo_hashes
        self.photo_hashes = hashes
        if changed:
            for sample in self.pipes:
                for role in ("left", "right"):
                    sample.pop(f"{role}_region_px", None)
                sample.pop("metrics", None)
        for role in ("left", "right"):
            self.views[role].set_image(self.owner.views[role].image.copy())
        if changed:
            self.message.set("新照片：请重新框选左右同一管段；检查记录按照片分别保存。")
            self.restore()
        if not self.pipes:
            self.add_sample()
        self.recalculate()

    def record_path(self):
        if set(self.photo_hashes) != {"left", "right"}:
            return None
        digest = hashlib.sha256(json.dumps(self.photo_hashes, sort_keys=True).encode()).hexdigest()
        return self.owner.output_root / "quality_checks" / f"{digest}.json"

    def restore(self):
        path = self.record_path()
        if path is None or not path.exists():
            return
        try:
            if path.stat().st_size > 1024 * 1024:
                raise ValueError("检查记录过大")
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("revision") != REVISION or record.get("photo_sha256") != self.photo_hashes:
                raise ValueError("检查记录与当前照片不一致")
            rows = record["samples"]
            if not isinstance(rows, list) or len(rows) > 32:
                raise ValueError("取样数量无效")
            restored = []
            for row in rows:
                sample = {"pipe_id": f"S{len(restored)+1:03d}", "name": str(row["name"])[:64],
                          "color_srgb": validate_color(row["color_srgb"])}
                diameter = row.get("reference_diameter_mm")
                if diameter is not None and any(diameter == p["nominal_diameter_mm"] for p in self.owner.pipes):
                    sample["reference_diameter_mm"] = diameter
                for role in ("left", "right"):
                    region = row.get(f"{role}_region_px")
                    if region is not None:
                        patch_mask(self.views[role].image, region)
                        sample[f"{role}_region_px"] = region
                restored.append(sample)
            self.pipes = restored
            self.message.set("已恢复这组照片的管面检查框；指标按当前规则重新计算。")
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.message.set(f"未恢复检查记录：{error}；请重新框选。")

    def add_sample(self):
        if len(self.pipes) >= 32:
            self.message.set("最多保留 32 个管面取样。")
            return
        number = max((int(p["pipe_id"][1:]) for p in self.pipes), default=0) + 1
        previous = self.selected_sample() or {}
        fallback = (previous.get("reference_diameter_mm"), previous.get("color_srgb", "#0000FF"))
        diameter, color = self.choices.get(self.reference.get(), fallback)
        sample = {"pipe_id": f"S{number:03d}", "name": f"管面 {number}", "color_srgb": color}
        if diameter is not None:
            sample["reference_diameter_mm"] = diameter
        self.pipes.append(sample)
        self.recalculate(select=sample["pipe_id"])

    def remove_sample(self):
        self.pipes = [p for p in self.pipes if p["pipe_id"] != self.selected_id()]
        self.recalculate()
        self.save(quiet=True)

    def rename_sample(self, _event=None):
        sample = self.selected_sample()
        if sample and self.name.get().strip():
            sample["name"] = self.name.get().strip()[:64]
            self.tree.item(sample["pipe_id"], text=sample["name"])
            self.save(quiet=True)

    def choose_reference(self, _event=None):
        sample = self.selected_sample()
        if sample and self.reference.get() in self.choices:
            sample["reference_diameter_mm"], sample["color_srgb"] = self.choices[self.reference.get()]
            self.recalculate()
            self.save(quiet=True)

    def set_color(self, color):
        sample = self.selected_sample()
        if sample:
            sample["color_srgb"] = validate_color(color)
            self.recalculate()
            self.save(quiet=True)

    def choose_color(self):
        from tkinter import colorchooser
        sample = self.selected_sample()
        if sample:
            color = colorchooser.askcolor(sample["color_srgb"], parent=self.window)[1]
            if color:
                self.set_color(color)

    def sample_color(self):
        sample = self.selected_sample()
        left = ((sample or {}).get("metrics", {}).get("eyes", {}) or {}).get("left")
        color = (left or {}).get("sampled_color_srgb")
        if color:
            self.set_color(color)
            self.message.set(f"检查参考色已设为 {color}；需要用于识别时点击“应用参考色到同径模型管”。")
        else:
            self.message.set("请在左图框选普通受光管面；需至少 64 个非高光、非近黑像素供取色。")

    def apply_color(self):
        sample = self.selected_sample()
        if not sample or self.owner.busy:
            self.message.set("请先选择管面取样，并等待当前评估完成。")
            return
        diameter = sample.get("reference_diameter_mm")
        targets = [p for p in self.owner.pipes if diameter is not None and abs(p["nominal_diameter_mm"] - diameter) <= 0.2]
        if not targets or not self.photo_hashes or self.photo_hashes != self.owner.image_hashes:
            self.message.set("请在当前照片中选取模型参考色和管面后重试。")
            return
        for target in targets:
            target["color_srgb"] = sample["color_srgb"]
        self.owner._clear_registration_anchors()
        self.owner.invalidate()
        self.owner.refresh_table()
        self.choices = {f"{p['nominal_diameter_mm']:g} mm · {p['color_srgb']}":
                        (p["nominal_diameter_mm"], p["color_srgb"]) for p in self.owner.pipes}
        self.reference_box.configure(values=tuple(self.choices))
        self.message.set(f"已更新 {len(targets)} 个 {diameter:g} mm 模型候选的参考色；返回主窗口“保存并评估”生效。")
        self.save(quiet=True)

    def set_region(self, role, box):
        sample = self.selected_sample()
        if sample is None or self.views[role].image is None:
            return
        patch_mask(self.views[role].image, box)
        sample[f"{role}_region_px"] = list(box)
        self.recalculate()
        self.save(quiet=True)

    def clear_regions(self):
        sample = self.selected_sample()
        if sample:
            for role in ("left", "right"):
                sample.pop(f"{role}_region_px", None)
            self.recalculate()
            self.save(quiet=True)

    def recalculate(self, select=None):
        from .matching_config import normalize_matching_settings
        try:
            self.delta_lab = normalize_matching_settings({"color_delta_lab": float(self.owner.color_delta_lab.get())})["color_delta_lab"]
            enabled = self.owner.color_filter_enabled.get()
            geometry_only = self.owner.local_observation_label.get() == "仅双目深度几何"
            context = ("当前仅使用深度几何，C 供调色参考。" if geometry_only else
                       "主界面颜色筛选开启；C 为 Lab 参考覆盖率。" if enabled else
                       "主界面颜色筛选关闭，C 仅作调色参考；应用参考色不会自动开启筛选。")
            self.policy_status.set(f"C 使用 Lab 参考色距离 ≤{self.delta_lab:g}。" + context)
            self.policy_valid = True
        except (ValueError, TypeError):
            self.policy_valid = False
            self.policy_status.set("主界面颜色偏差需为 0～150 之间的正数，请修正后计算。")
        for sample in self.pipes:
            eyes = {}
            for role in ("left", "right"):
                image = self.views[role].image
                box = sample.get(f"{role}_region_px")
                eyes[role] = measure_patch(image, sample["color_srgb"], patch_mask(image, box), delta_lab=self.delta_lab) if self.policy_valid and image is not None and box else None
            sample["metrics"] = {"eyes": eyes, "summary": summarize_pair(eyes)}
        self.refresh_table(select)

    def refresh_table(self, select=None):
        selected = select or self.selected_id()
        self.tree.delete(*self.tree.get_children())
        for sample in self.pipes:
            metrics = sample.get("metrics", {})
            eyes = metrics.get("eyes", {})
            summary = metrics.get("summary", {})
            values = [sample["color_srgb"]]
            for kind in ("color", "highlight"):
                values.extend(metric_text(eyes.get(role), kind) for role in ("left", "right"))
                values.append(metric_text(summary, kind))
            self.tree.insert("", "end", iid=sample["pipe_id"], text=sample["name"], values=values)
        if self.pipes:
            self.tree.selection_set(selected if selected in self.tree.get_children() else self.pipes[0]["pipe_id"])
        self.selected_changed()

    def selected_changed(self, _event=None):
        sample = self.selected_sample()
        summary = (sample or {}).get("metrics", {}).get("summary", {})
        self.name.set((sample or {}).get("name", ""))
        if sample:
            diameter = sample.get("reference_diameter_mm")
            self.reference.set(f"{diameter:g} mm · {sample['color_srgb']}" if diameter is not None else sample["color_srgb"])
        for kind, label, prefix in (("color", self.color_label, "C 颜色覆盖率 ↑"),
                                    ("highlight", self.highlight_label, "H 高光风险 ↓")):
            label.configure(text=f"{prefix}  {metric_text(summary, kind)}", foreground=COLORS[summary.get(kind + "_level", "PENDING")])
        self.advice.set(" ".join(summary.get("advice", ["添加管面取样，在左右图中分别框选。"])))
        for view in self.views.values():
            view.draw()

    def save(self, quiet=False):
        path = self.record_path()
        if path is None:
            if not quiet:
                self.message.set("请先载入左右照片。")
            return
        try:
            record = diagnostic_record(self.photo_hashes, self.pipes)
            record["matching_context"] = {"color_filter_enabled": bool(self.owner.color_filter_enabled.get()),
                                          "color_delta_lab_used": self.delta_lab if self.policy_valid else None,
                                          "local_observation_label": self.owner.local_observation_label.get(),
                                          "color_metric_scope": "LAB_REFERENCE_COVERAGE_NOT_ALL_PARALLEL_STRIP_HINTS"}
            record["saved_at"] = datetime.now(timezone.utc).isoformat()
            atomic_write_text(path, json.dumps(record, ensure_ascii=False, indent=2) + "\n")
            log_event(LOGGER, "capture_quality_saved", path=str(path), photo_sha256=self.photo_hashes,
                      summaries=[{"name": p["name"], **p.get("metrics", {}).get("summary", {})} for p in self.pipes])
            if not quiet:
                self.message.set(f"检查记录已保存：{path}")
        except OSError as error:
            self.message.set(f"检查记录保存失败：{error}")

    def close(self):
        self.save(quiet=True)
        for variable, trace in self.policy_traces:
            variable.trace_remove("write", trace)
        self.closed = True
        self.window.destroy()
