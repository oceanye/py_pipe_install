"""Independent, single-frame monitoring window with explicit cycle freshness."""
from __future__ import annotations

import base64
import copy
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import queue
import time
from types import SimpleNamespace

import cv2
import numpy as np

from .assist_display import assist_data, assist_preview_states, hypothesis_assignments, STATUS_LABELS
from .camera_view import camera_side_label
from .elevation_viewer import ElevationModelViewer, model_section_basis
from .relative_scene import draw_scene, model_states, scene_sections
from .section_status import solver_status_rows
from .photo_capture import load_photo_snapshot, resolve_photo_path


INTERVAL_MINUTES = (1, 2, 3, 5, 10)
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_ASSET_BYTES = 256 * 1024 * 1024
PHASE_LABELS = {"IDLE": "尚未开始", "OPENING": "正在连接相机", "WARMING_UP": "相机预热中",
                "CAPTURING": "① 单帧获取中", "ANALYZING": "② 分析中", "RESULT": "③ 结果已显示",
                "WAITING": "等待下一轮", "STOPPING": "正在停止并释放相机", "STOPPED": "监控已停止"}


def _manifest_bytes(path):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_MANIFEST_BYTES + 1)
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("本轮现场清单超出大小限制。")
    return raw


def _asset_hash(path):
    """Bounded streaming verification, without allocating another image copy."""
    digest, size = hashlib.sha256(), 0
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_ASSET_BYTES:
                raise ValueError("本轮模型或照片超出大小限制。")
            digest.update(chunk)
    return digest.hexdigest()


def _time_text(value):
    if not value:
        return "未记录"
    try:
        stamp = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return stamp.astimezone().strftime("%m-%d %H:%M:%S")
    except ValueError:
        return str(value)


class _PhotoPane:
    def __init__(self, parent, tk, ttk, title):
        self.tk, self.title = tk, title
        self.frame = ttk.Frame(parent)
        self.caption = tk.StringVar(value=f"{title} · 尚无本轮抓拍")
        ttk.Label(self.frame, textvariable=self.caption, style="Monitor.TLabel").pack(anchor="w", pady=(0, 4))
        self.canvas = tk.Canvas(self.frame, bg="#1F2B38", highlightthickness=0, width=400, height=180)
        self.canvas.pack(fill="both", expand=True)
        self.image = self.photo = None
        self.empty_text = "点击开始监控后获取新照片"
        self.canvas.bind("<Configure>", lambda _event: self.draw())

    def clear(self, caption, message):
        self.image = self.photo = None
        self.caption.set(f"{self.title} · {caption}")
        self.empty_text = message
        self.draw()

    def set_image(self, image, cycle, captured_at):
        self.image = image
        self.caption.set(f"{self.title} · 第 {cycle} 轮 · 拍摄 {_time_text(captured_at)}")
        self.draw()

    def draw(self):
        canvas = self.canvas
        canvas.delete("all")
        width, height = max(canvas.winfo_width(), 1), max(canvas.winfo_height(), 1)
        if self.image is None:
            canvas.create_text(width / 2, height / 2, text=self.empty_text, fill="#FFFFFF",
                               font=("Microsoft YaHei UI", 10), width=max(width - 30, 40))
            return
        rows, columns = self.image.shape[:2]
        scale = min(width / columns, height / rows)
        resized = cv2.resize(self.image, (max(1, round(columns * scale)), max(1, round(rows * scale))))
        ok, png = cv2.imencode(".png", resized)
        if ok:
            self.photo = self.tk.PhotoImage(data=base64.b64encode(png).decode("ascii"), format="png")
            canvas.create_image(width / 2, height / 2, image=self.photo)


class _CADSection:
    """Embed the established CAD-section renderer without a second window."""
    def __init__(self, owner, panel, canvas):
        self.owner = SimpleNamespace(registration_settings={}, assist_hypothesis_index=0)
        self.section_panel, self.section_canvas = panel, canvas
        self.section_result_mode = owner.app.tk.StringVar(value="LOCAL")
        self.view_mode = owner.app.tk.StringVar(value="监控方向")
        self.view_choices = {"监控方向": None}
        self.pipes, self.report, self.installation_states = [], {}, {}
        self.source_lines, self.section_basis = np.empty((0, 2, 3)), np.eye(3)

    def _is_current(self):
        return True  # The dialog supplies only the accepted cycle's snapshot.

    def camera_side_status(self):
        return "观察侧：" + camera_side_label(self.view_choices["监控方向"])

    def update(self, pipes, registration, report, selected=None):
        self.pipes, self.report = pipes, report
        self.owner.registration_settings = registration
        self.view_choices["监控方向"] = registration.get("camera_side_world")
        self.source_lines = np.asarray([pipe["centerline_world_mm"] for pipe in pipes], float) if pipes else np.empty((0, 2, 3))
        axis = registration.get("axis_world")
        self.section_basis = model_section_basis(self.source_lines, np.asarray(axis, float) if axis is not None else None) if pipes else np.eye(3)
        self.installation_states = assist_preview_states(report, model_states(report))
        ElevationModelViewer.draw_section(self, selected)


class AutoMonitorDialog:
    def __init__(self, owner, *, monitor_factory=None, clock=time.monotonic):
        self.owner, self.app, self.clock = owner, owner.app, clock
        self.monitor_factory = monitor_factory
        self.monitor = None
        self.closed = self.closing = self.locked = False
        self.stopping = False
        self.state = "IDLE"
        self.cycle_id = self.result_cycle = 0
        self.failed_cycle = None
        self.manifest_path = None
        self.manifest_hash = None
        self.input_hashes = {}
        self.report = None
        self.captured_at = None
        self.snapshot = {}
        self.pipes, self.registration, self.rows = [], {}, {}
        self.selected_pipe = None
        tk, ttk = self.app.tk, self.app.ttk
        self.window = tk.Toplevel(owner.window)
        self.window.title("自动监控 · 单帧分析")
        self.window.geometry("1280x860")
        self.window.minsize(960, 660)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        style = ttk.Style(self.window)
        style.configure("Monitor.TLabel", font=("Microsoft YaHei UI", 10))
        style.configure("Monitor.Title.TLabel", font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Monitor.TButton", padding=(10, 5))
        style.configure("Monitor.Treeview", rowheight=25)
        self.interval = tk.StringVar(value="1 分钟")
        self.phase = tk.StringVar(value=PHASE_LABELS["IDLE"])
        self.countdown = tk.StringVar(value="下一轮：开始后计时")
        self.progress = tk.StringVar(value="每轮只获取一对新照片；完成后再进入下一轮。")
        self.result_caption = tk.StringVar(value="断面结果 · 尚未开始")
        self.error = tk.StringVar()
        frame = ttk.Frame(self.window, padding=12)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1, minsize=175)
        frame.rowconfigure(4, weight=2, minsize=280)
        toolbar = ttk.Frame(frame)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(toolbar, text="采集间隔", style="Monitor.TLabel").pack(side="left")
        self.interval_box = ttk.Combobox(toolbar, textvariable=self.interval, state="readonly", width=9,
                                        values=[f"{value} 分钟" for value in INTERVAL_MINUTES])
        self.interval_box.pack(side="left", padx=(8, 16))
        self.start_button = ttk.Button(toolbar, text="开始监控", command=self.start, style="Monitor.TButton")
        self.start_button.pack(side="left", padx=3)
        self.stop_button = ttk.Button(toolbar, text="停止监控", command=self.stop, state="disabled", style="Monitor.TButton")
        self.stop_button.pack(side="left", padx=3)
        ttk.Label(toolbar, textvariable=self.countdown, style="Monitor.Title.TLabel").pack(side="right")
        status = ttk.Frame(frame)
        status.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(status, text="单帧获取 → 分析 → 结果显示", style="Monitor.TLabel").pack(side="left")
        ttk.Label(status, textvariable=self.phase, style="Monitor.Title.TLabel").pack(side="left", padx=20)
        photos = ttk.Frame(frame)
        photos.grid(row=2, column=0, sticky="nsew")
        photos.columnconfigure((0, 1), weight=1, uniform="photos")
        photos.rowconfigure(0, weight=1)
        self.photos = {role: _PhotoPane(photos, tk, ttk, label) for role, label in (("left", "左目"), ("right", "右目"))}
        for column, pane in enumerate(self.photos.values()):
            pane.frame.grid(row=0, column=column, sticky="nsew", padx=(0, 6) if column == 0 else (6, 0))
        ttk.Label(frame, textvariable=self.result_caption, style="Monitor.Title.TLabel").grid(row=3, column=0, sticky="w", pady=(10, 4))
        results = ttk.Frame(frame)
        results.grid(row=4, column=0, sticky="nsew")
        results.columnconfigure((0, 1), weight=1, uniform="sections")
        results.columnconfigure(2, weight=0, minsize=285)
        results.rowconfigure(0, weight=1)
        self.cad_panel = ttk.LabelFrame(results, text="CAD 对应断面")
        self.cad_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        measured_panel = ttk.LabelFrame(results, text="本轮测得断面 · 条件估计")
        measured_panel.grid(row=0, column=1, sticky="nsew", padx=5)
        self.cad_canvas = tk.Canvas(self.cad_panel, bg="#F2F5F8", width=200, height=250, highlightthickness=0)
        self.cad_canvas.pack(fill="both", expand=True)
        self.section_canvas = tk.Canvas(measured_panel, bg="#F2F5F8", width=200, height=250, highlightthickness=0)
        self.section_canvas.pack(fill="both", expand=True)
        self.cad = _CADSection(self, self.cad_panel, self.cad_canvas)
        table_panel = ttk.LabelFrame(results, text="管径 / 管号 / 辅助状态")
        table_panel.grid(row=0, column=2, sticky="nsew", padx=(5, 0))
        self.table = ttk.Treeview(table_panel, columns=("section", "diameter", "pipe", "status"),
                                 show="headings", height=6, style="Monitor.Treeview", selectmode="browse")
        for key, title, width in (("section", "断面", 42), ("diameter", "测径 mm", 65), ("pipe", "管号", 63), ("status", "辅助状态", 101)):
            self.table.heading(key, text=title)
            self.table.column(key, width=width, minwidth=35, stretch=True)
        scroll = ttk.Scrollbar(table_panel, command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.table.pack(fill="both", expand=True)
        self.table.bind("<<TreeviewSelect>>", self._selected)
        self.table.bind("<Return>", self._selected)
        self.cad_canvas.bind("<Configure>", lambda _event: self._draw_results())
        self.section_canvas.bind("<Configure>", lambda _event: self._draw_results())
        footer = ttk.Frame(frame)
        footer.grid(row=5, column=0, sticky="ew", pady=(7, 0))
        self.progress_label = ttk.Label(footer, textvariable=self.progress, style="Monitor.TLabel", wraplength=950)
        self.progress_label.pack(anchor="w", fill="x")
        self.error_label = ttk.Label(footer, textvariable=self.error, foreground="#A13124", style="Monitor.TLabel", wraplength=950)
        self.error_label.pack(anchor="w", fill="x")
        footer.bind("<Configure>", lambda event: [label.configure(wraplength=max(300, event.width - 8)) for label in (self.progress_label, self.error_label)])
        self.window.bind("<Control-Return>", lambda _event: self.start())
        self.window.bind("<Escape>", lambda _event: self.stop())
        self._draw_results()
        self.poll_id = self.window.after(100, self.poll)

    @property
    def active(self):
        return self.monitor is not None and self.monitor.active

    @property
    def camera_running(self):
        return self.active

    def start(self):
        if self.active or self.closing or self.closed:
            return
        try:
            self._release_if_idle()
            minutes = int(self.interval.get().split()[0])
            if minutes not in INTERVAL_MINUTES:
                raise ValueError("请选择 1、2、3、5 或 10 分钟。")
            configuration = copy.deepcopy(self.owner.monitor_configuration())
            if configuration.get("mode", "local") == "server":
                raise ValueError("服务器端不使用相机，请切换到前端或本地运算。")
            if self.owner.monitor_started() is False:
                raise ValueError("相机正在使用，请等待释放后开始监控。")
            self.locked = True
            self.snapshot = configuration
            self.pipes = configuration["dataset_arguments"]["pipe_specs"]
            self.registration = configuration["dataset_arguments"].get("registration_settings") or {}
            factory = self.monitor_factory
            if factory is None:
                from .auto_monitor import SingleFrameMonitor
                factory = SingleFrameMonitor
            self.monitor = factory(**configuration, interval_seconds=minutes * 60)
            self.cycle_id = self.result_cycle = 0
            self.failed_cycle = None
            self.manifest_path = self.manifest_hash = None
            self.input_hashes = {}
            self.error.set("")
            self.stopping = False
            self._clear_results("等待第一轮单帧获取")
            for pane in self.photos.values():
                pane.clear("等待新抓拍", "正在连接相机")
            self.monitor.start()
            self._phase("OPENING")
        except Exception as error:
            self.error.set(str(error))
            if self.active:
                self.monitor.stop()
                self.stopping = True
            else:
                self._release_if_idle()
        self._controls()

    def stop(self):
        if self.active:
            self.stopping = True
            self.monitor.stop()
            self._phase("STOPPING")
        self._controls()

    def _phase(self, state):
        self.state = state
        prefix = f"第 {self.cycle_id} 轮 · " if self.cycle_id else ""
        self.phase.set(prefix + PHASE_LABELS.get(state, state))

    def _controls(self):
        active = self.active
        self.start_button.configure(state="disabled" if active or self.closing else "normal")
        self.stop_button.configure(state="normal" if active and not self.stopping else "disabled")
        self.interval_box.configure(state="disabled" if active or self.closing else "readonly")

    def _release_if_idle(self):
        if self.locked and not self.active:
            self.locked = False
            if not getattr(self.owner, "closed", False):
                self.owner.monitor_stopped()

    def _clear_results(self, message):
        self.report = None
        self.selected_pipe = None
        self.result_cycle = 0
        self.rows.clear()
        self.table.delete(*self.table.get_children())
        self.result_caption.set(f"第 {self.cycle_id} 轮 · {message}" if self.cycle_id else message)
        self._draw_results()

    @staticmethod
    def _placeholder(canvas, message):
        canvas.delete("all")
        canvas.create_text(max(canvas.winfo_width(), 1) / 2, max(canvas.winfo_height(), 1) / 2,
                           text=message, width=max(canvas.winfo_width() - 24, 80), justify="center",
                           fill="#355371", font=("Microsoft YaHei UI", 10), tags="empty")

    def _draw_results(self):
        if self.report is None:
            message = "本轮尚无结果\n获取或分析期间不显示断面"
            self._placeholder(self.cad_canvas, message)
            self._placeholder(self.section_canvas, message)
            return
        self.cad.update(self.pipes, self.registration, self.report, self.selected_pipe)
        scenes = scene_sections(self.report)
        if scenes and scenes[0]["items"]:
            draw_scene(self.section_canvas, scenes[0], selected_pipe=self.selected_pipe,
                       advisory_assignments=hypothesis_assignments(self.report))
        else:
            self._placeholder(self.section_canvas, "本轮未解出可用断面\n详见辅助状态与进度说明")

    def _selected(self, _event=None):
        selected = self.table.selection()
        self.selected_pipe = self.rows.get(selected[0], {}).get("display_pipe_id") if selected else None
        self._draw_results()

    def _begin_cycle(self, cycle):
        if cycle <= self.cycle_id:
            return
        self.cycle_id = cycle
        self.failed_cycle = None
        self.manifest_path = self.manifest_hash = None
        self.input_hashes = {}
        self.captured_at = None
        self.error.set("")
        self._clear_results("单帧获取中，等待本轮结果")
        for pane in self.photos.values():
            pane.clear(f"第 {cycle} 轮 · 获取中", "正在获取本轮新照片")

    def handle_event(self, event):
        kind, cycle = event.get("kind"), int(event.get("cycle_id") or 0)
        if self.closed or cycle < self.cycle_id:
            return
        if self.stopping and kind not in ("error", "stopped") and event.get("state") != "STOPPING":
            return
        if kind == "state":
            state = event.get("state", "IDLE")
            if state in ("CAPTURING", "ANALYZING"):
                self._begin_cycle(cycle)
                if self.report is not None:
                    self._clear_results("等待本轮结果")
            self._phase(state)
            if state == "ANALYZING":
                self.result_caption.set(f"第 {cycle} 轮 · 拍摄 {_time_text(self.captured_at)} · 分析中，断面暂不显示")
            return
        if kind == "capture":
            self._begin_cycle(cycle)
            self.manifest_path = Path(event["manifest_path"]).resolve()
            raw = _manifest_bytes(self.manifest_path)
            self.manifest_hash = hashlib.sha256(raw).hexdigest()
            manifest = json.loads(raw)
            groups = manifest["capture"]["capture_groups"]
            if len(groups) != 1 or groups[0].get("temporal_frames"):
                raise ValueError("监控本轮必须仅包含新抓拍的一对照片。")
            views = groups[0]["views"]
            model = manifest["model"]
            model_path = resolve_photo_path(self.manifest_path, model["path"])
            if _asset_hash(model_path) != model["sha256"]:
                raise ValueError("本轮模型文件已改变，照片与结果未显示。")
            verified = {model_path: model["sha256"]}
            self.captured_at = event.get("captured_at") or views["left"].get("captured_at")
            images = {}
            for role in ("left", "right"):
                images[role], integrity = load_photo_snapshot(self.manifest_path, views[role])
                verified[resolve_photo_path(self.manifest_path, views[role]["path"])] = integrity["actual_sha256"]
            if hashlib.sha256(_manifest_bytes(self.manifest_path)).hexdigest() != self.manifest_hash:
                raise ValueError("本轮现场清单在加载时改变，照片与结果未显示。")
            self.input_hashes = verified
            for role, image in images.items():
                self.photos[role].set_image(image, cycle, self.captured_at)
            self._clear_results(f"拍摄 {_time_text(self.captured_at)} · 等待分析结果")
        elif kind == "progress":
            self.progress.set(str(event.get("message", "")))
        elif kind == "error":
            self.failed_cycle = cycle
            self._clear_results("本轮未完成，断面暂不显示")
            self.error.set(f"第 {cycle} 轮 · {event.get('message', '监控失败')}")
        elif kind == "result":
            if cycle != self.cycle_id or cycle == self.failed_cycle or self.manifest_path is None or Path(event["manifest_path"]).resolve() != self.manifest_path:
                return
            if hashlib.sha256(_manifest_bytes(self.manifest_path)).hexdigest() != self.manifest_hash:
                raise ValueError("本轮现场文件已改变，结果未显示。")
            report = event.get("report")
            if not isinstance(report, dict):
                raise ValueError("本轮报告无效。")
            from .ai_gui import report_matches_scene
            if not report_matches_scene(self.manifest_path, report):
                raise ValueError("报告缺少本轮来源绑定或输入哈希不一致，结果未显示。")
            if len(self.input_hashes) != 3:
                raise ValueError("本轮模型和双目照片尚未通过完整核验。")
            for path, expected in self.input_hashes.items():
                if _asset_hash(path) != expected:
                    raise ValueError("本轮模型或照片文件在分析期间改变，结果未显示。")
            self.report = report
            self.result_cycle = cycle
            self.error.set("")
            self._phase("RESULT")
            result_time = event.get("finished_at") or report.get("generated_at") or datetime.now().astimezone()
            self.result_caption.set(f"第 {cycle} 轮 · 拍摄 {_time_text(self.captured_at)} · 结果 {_time_text(result_time)}")
            self._fill_table()
            self._draw_results()
            self.progress.set("本轮结果已显示。测径为条件量测；CAD 圆径为设计值；辅助管号仍待确认。")
        elif kind == "stopped":
            if not self.active:
                self._phase("STOPPED")

    def _fill_table(self):
        self.table.delete(*self.table.get_children())
        self.rows.clear()
        mapping = hypothesis_assignments(self.report)
        assistance = {item["observation_id"]: item for item in assist_data(self.report).get("observations", [])}
        rows = list(self.report.get("solver_status") or solver_status_rows(self.report))
        represented = {row.get("candidate_id") for row in rows}
        for identity, observation in assistance.items():
            if identity not in represented:
                rows.append({"candidate_id": identity, "label": observation.get("label", identity),
                             "diameter_mm": None, "solved": False})
        for index, row in enumerate(rows):
            identity = row.get("candidate_id")
            observation = assistance.get(identity, {})
            pipe_id = row.get("pipe_id") or mapping.get(identity)
            diameter = row.get("diameter_mm")
            status = STATUS_LABELS.get(observation.get("status"), "待对应" if not pipe_id else "已对应" if row.get("pipe_id") else "辅助候选")
            if not row.get("solved"):
                status = "断面未解 · " + status
            self.rows[str(index)] = dict(row, display_pipe_id=pipe_id)
            self.table.insert("", "end", iid=str(index), values=(row.get("label", identity),
                              f"{diameter:.2f}" if isinstance(diameter, (int, float)) else "—", pipe_id or "待对应", status))

    def _countdown(self):
        if not self.active:
            self.countdown.set("下一轮：已停止" if self.monitor is not None else "下一轮：开始后计时")
            return
        if self.stopping:
            self.countdown.set("正在释放相机…")
            return
        due = self.monitor.next_due
        if due is None:
            self.countdown.set("下一轮：等待相机就绪")
            return
        seconds = max(0, math.ceil(due - self.clock()))
        if seconds == 0 and self.state in ("CAPTURING", "ANALYZING"):
            self.countdown.set("已到期 · 等待本轮完成，不累积任务")
        elif self.state == "WARMING_UP":
            self.countdown.set(f"首次采集：预热剩余 {seconds} 秒")
        else:
            self.countdown.set(f"下一轮：{seconds} 秒")

    def poll(self):
        if self.closed:
            return
        if getattr(self.owner, "closed", False):
            self.close()
            return
        if self.monitor is not None:
            for _ in range(128):
                try:
                    event = self.monitor.events.get_nowait()
                except queue.Empty:
                    break
                try:
                    self.handle_event(event)
                except Exception as error:
                    self.failed_cycle = self.cycle_id
                    self._clear_results("本轮结果未显示")
                    self.error.set(str(error))
        self._release_if_idle()
        self._controls()
        self._countdown()
        if self.closing and not self.active:
            self._destroy()
            return
        self.poll_id = self.window.after(100, self.poll)

    def close(self):
        if self.closed:
            return
        self.closing = True
        self.stop()
        if getattr(self.owner, "closed", False) or not self.active:
            self._release_if_idle()
            self._destroy()

    def _destroy(self):
        if self.closed:
            return
        self.closed = True
        try:
            self.window.after_cancel(self.poll_id)
            self.window.destroy()
        except self.app.tk.TclError:
            pass
