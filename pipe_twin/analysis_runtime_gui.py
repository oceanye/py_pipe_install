"""Execution-mode controls and worker lifecycles for the shared field GUI.

Workers publish plain queue messages. Only the Tk thread changes widgets or
accepts a report; every remote completion is bound to its task and input files.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import queue
import threading
import time

from .analysis_runtime_settings import (
    AnalysisRuntimeSettings, MODE_LABELS, load_runtime_settings, save_runtime_settings,
)


def save_scene_snapshot(arguments):
    from .elevation_dataset import create_elevation_dataset, elevation_history_compatible, load_elevation_dataset
    arguments = dict(arguments)
    expected = arguments.pop("_preview_hashes")
    previous = arguments["previous_manifest"]
    names = ("calibration", "pipe_specs", "model_path", "stl_unit", "rectification_recipe",
             "analysis_settings", "scope_settings", "mode", "registration_settings")
    reset = bool(previous and not elevation_history_compatible(previous, **{name: arguments.get(name) for name in names}))
    if reset:
        arguments["previous_manifest"] = None
    path = create_elevation_dataset(**arguments)
    loaded = load_elevation_dataset(path)
    latest = loaded["manifest"]["capture"]["capture_groups"][-1]["views"]
    if any(latest[role]["sha256"] != expected[role] for role in ("left", "right")):
        raise ValueError(f"保存期间照片发生变化，请重新载入。现场包：{path}")
    return path, reset


def input_fingerprints(arguments):
    paths = [arguments["model_path"], arguments["left_path"], arguments["right_path"]]
    for frame in arguments.get("temporal_frames", []):
        paths.extend(frame[f"{role}_path"] for role in ("left", "right"))
    return {str(Path(path).resolve()): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in paths if path is not None}


class AnalysisRuntimeController:
    def __init__(self, owner, *, settings_path=None):
        self.owner = owner
        self.settings_path = settings_path
        self.problem = ""
        try:
            self.settings = load_runtime_settings(path=settings_path)
        except (OSError, ValueError) as error:
            self.settings = AnalysisRuntimeSettings()
            self.problem = str(error)
        self.server = None
        self.active = None
        self.changing = False
        self.checking = False
        self.closed = threading.Event()
        self.lock = threading.Lock()
        self.addresses = []
        self.server_health = {}
        self.revision = 0
        self.next_health = 0.0
        self.row = None
        self.events = queue.Queue()

    @property
    def mode(self):
        return self.settings.mode

    def build(self, controls, notebook):
        owner = self.owner
        tk, ttk = owner.app.tk, owner.app.ttk
        self.status = tk.StringVar(value=self.problem or f"{MODE_LABELS[self.mode]} · {'未启动服务' if self.mode == 'server' else '待连接' if self.mode == 'client' else '就绪'}")
        self.mode_label = tk.StringVar(value=MODE_LABELS[self.mode])
        self.address = tk.StringVar(value=self.settings.address)
        self.port = tk.StringVar(value=str(self.settings.port))
        self.endpoint = tk.StringVar()
        self.capabilities = tk.StringVar(value="正在检测本机计算能力…")
        self.server_capabilities = tk.StringVar(value="服务器能力：尚未连接")
        self.address_hint = tk.StringVar(value="正在检测本机 IP…")
        page = ttk.Frame(notebook, padding=12)
        owner.settings_pages["runtime"] = page
        notebook.add(page, text="运行模式")
        mode_row = ttk.Frame(page); mode_row.pack(fill="x", pady=8)
        ttk.Label(mode_row, text="运行方式").pack(side="left")
        self.mode_box = ttk.Combobox(mode_row, textvariable=self.mode_label,
                                    values=[MODE_LABELS[key] for key in ("server", "client", "local")],
                                    width=18, state="readonly")
        self.mode_box.pack(side="left", padx=12)
        self.mode_box.bind("<<ComboboxSelected>>", self._mode_selected)
        address_row = ttk.Frame(page); address_row.pack(fill="x", pady=8)
        ttk.Label(address_row, text="服务 IP / 主机名").pack(side="left")
        self.address_box = ttk.Combobox(address_row, textvariable=self.address, width=29)
        self.address_box.pack(side="left", padx=8)
        ttk.Label(address_row, text="计算端口").pack(side="left")
        self.port_box = ttk.Entry(address_row, textvariable=self.port, width=8)
        self.port_box.pack(side="left", padx=8)
        ttk.Label(page, text="服务器端：填写本机要监听的 IP，启动后供前端连接。\n前端：填写服务器 IP 与计算端口；每次评估只提交当前批照片，历史现场保留在原目录。\n本地运算：使用本机原有分析流程。", wraplength=760).pack(anchor="w", pady=8)
        ttk.Label(page, textvariable=self.address_hint, wraplength=760).pack(anchor="w", pady=5)
        ttk.Label(page, textvariable=self.capabilities, wraplength=760).pack(anchor="w", pady=5)
        ttk.Label(page, textvariable=self.server_capabilities, wraplength=760).pack(anchor="w", pady=5)
        buttons = ttk.Frame(page); buttons.pack(fill="x", pady=12)
        self.apply_button = ttk.Button(buttons, text="保存运行设置", command=self.apply)
        self.apply_button.pack(side="left", padx=4)
        self.server_button = ttk.Button(buttons, text="启动服务", command=self.toggle_server)
        self.server_button.pack(side="left", padx=4)
        self.check_button = ttk.Button(buttons, text="检测连接", command=self.check_connection)
        self.check_button.pack(side="left", padx=4)
        ttk.Label(page, textvariable=self.status, wraplength=760, foreground="#355371").pack(anchor="w", pady=8)
        self.row = ttk.Frame(controls)
        self.row.pack(fill="x", pady=(2, 3))
        self.status_label = ttk.Label(self.row, textvariable=self.status, width=1, foreground="#355371")
        self.status_label.pack(side="left", fill="x", expand=True)
        self.endpoint_entry = ttk.Entry(self.row, textvariable=self.endpoint, width=31, state="readonly")
        self.endpoint_entry.pack(side="left", padx=5)
        self.copy_button = ttk.Button(self.row, text="复制地址", command=self.copy_address)
        self.copy_button.pack(side="left")
        self.cancel_button = ttk.Button(self.row, text="取消远程", command=self.cancel_remote, state="disabled")
        self.cancel_button.pack(side="left", padx=4)
        if self.mode == "client":
            owner.history.set(False)
        self.refresh_controls()
        threading.Thread(target=self._probe, daemon=True).start()

    def _probe(self):
        try:
            from .analysis_capabilities import probe_capabilities, listening_addresses
            payload = (probe_capabilities(), listening_addresses())
            self.events.put((0, "runtime_probe", payload))
        except Exception as error:
            self.events.put((0, "runtime_probe_error", str(error)))

    def _mode_selected(self, _event=None):
        if self.mode_label.get() == MODE_LABELS["server"] and self.address.get() in ("127.0.0.1", "localhost"):
            self.address.set(next((item for item in self.addresses if not item.startswith("127.")), "127.0.0.1"))

    def _candidate(self):
        modes = {label: key for key, label in MODE_LABELS.items()}
        try:
            port = int(self.port.get())
        except ValueError:
            raise ValueError("计算端口须为 1–65535 的整数。") from None
        return AnalysisRuntimeSettings(mode=modes.get(self.mode_label.get(), ""), address=self.address.get(), port=port)

    def _server_busy(self):
        if self.server is None:
            return False
        return self.server.is_busy

    def apply(self):
        owner = self.owner
        if owner.busy or self.changing or self.checking or self._server_busy():
            self.status.set("当前任务或连接操作尚未结束，请等待完成后切换。")
            return False
        # An open camera dialog can still be opening a device in its worker.
        for name in ("camera_dialog", "calibration_dialog", "ground_monitor"):
            dialog = getattr(owner, name, None)
            if dialog is not None and dialog.window.winfo_exists():
                self.status.set("请先关闭抓拍、标定或实时调姿窗口，释放相机后再切换。")
                return False
        try:
            candidate = self._candidate()
            if self.server is not None and candidate != self.settings:
                self.status.set("请先停止计算服务，再修改运行模式或监听地址。")
                return False
            save_runtime_settings(candidate, path=self.settings_path)
        except (OSError, ValueError) as error:
            self.status.set(str(error))
            return False
        previous = self.mode
        changed = candidate != self.settings
        self.settings = candidate
        self.revision += 1
        if previous != self.mode:
            owner.invalidate()
        if self.mode == "client":
            owner.history.set(False)
        if changed:
            self.server_capabilities.set("服务器能力：尚未连接" if self.mode == "client" else "服务器能力：启动服务后显示")
        self.status.set(f"{MODE_LABELS[self.mode]} · 设置已保存" + ("，服务未启动" if self.mode == "server" and self.server is None else ""))
        owner._set_controls(True)
        return True

    def refresh_controls(self):
        if self.row is None or self.owner.closed:
            return
        busy = self.owner.busy or self.changing
        if self.mode == "server":
            def disable(widget):
                for child in widget.winfo_children():
                    if hasattr(child, "state"):
                        child.state(["disabled"])
                    disable(child)
            disable(self.owner.controls)
            # Settings and service controls remain available in server mode.
            self.owner.settings_button.state(["!disabled"])
        for widget in (self.mode_box, self.address_box, self.port_box, self.apply_button):
            widget.state(["disabled"] if busy else ["!disabled"])
        self.server_button.configure(text="停止服务" if self.server is not None else "启动服务",
                                     state="normal" if self.mode == "server" and not busy else "disabled")
        self.check_button.configure(state="normal" if self.mode == "client" and not busy and not self.checking else "disabled")
        self.endpoint_entry.state(["!disabled", "readonly"])
        self.endpoint.set(self.settings.base_url if self.mode == "client" else self._server_endpoint())
        self.copy_button.configure(state="normal" if self.endpoint.get() else "disabled")
        self.cancel_button.configure(state="normal" if self.active else "disabled")
        if self.mode == "server":
            self.owner.ai_run_button.configure(state="disabled")
        self.owner.history_button.configure(
            text="前端仅提交当前批（历史现场单独保留）" if self.mode == "client" else "沿用上次场景历史（相机未移动）",
            state="normal" if self.mode == "local" and not busy else "disabled")
        self.owner.run_button.configure(text="⑤ 保存并提交服务器" if self.mode == "client" else "⑤ 保存并评估")

    def _server_endpoint(self):
        if self.server is None:
            return ""
        address = self.settings.address
        if address == "0.0.0.0":
            address = next((item for item in self.addresses if not item.startswith("127.")), "127.0.0.1")
        return AnalysisRuntimeSettings(address=address, port=self.server.port).base_url

    def copy_address(self):
        if self.endpoint.get():
            self.owner.window.clipboard_clear()
            self.owner.window.clipboard_append(self.endpoint.get())
            self.owner.message.set(f"已复制计算服务地址：{self.endpoint.get()}")

    def toggle_server(self):
        if self.changing or self.owner.busy or self.mode != "server":
            return
        if self.server is not None:
            if self._server_busy():
                self.status.set("计算服务有未完成任务，请等待任务结束后停止服务。")
                return
            self.changing = True
            self.revision += 1
            self.checking = False
            self.status.set("正在停止计算服务…")
            server = self.server
            threading.Thread(target=self._stop_server, args=(server, self.revision), daemon=True).start()
        else:
            if not self.apply():
                return
            self.changing = True
            self.status.set("正在启动计算服务…")
            threading.Thread(target=self._start_server, args=(self.settings, self.revision), daemon=True).start()
        self.refresh_controls()

    def _start_server(self, settings, revision):
        server = None
        try:
            from .analysis_server import AnalysisServer
            server = AnalysisServer(bind=settings.address, port=settings.port,
                                    work_root=self.owner.output_root / "remote_jobs")
            server.start()
            with self.lock:
                if not self.closed.is_set():
                    self.server = server
            if self.closed.is_set():
                server.stop()
                return
            self.events.put((revision, "runtime_server_started", server.health()))
        except Exception as error:
            if server is not None:
                server.stop()
            with self.lock:
                if self.server is server:
                    self.server = None
            self.events.put((revision, "runtime_error", str(error)))

    def _stop_server(self, server, revision):
        try:
            server.stop()
            self.events.put((revision, "runtime_server_stopped", None))
        except Exception as error:
            self.events.put((revision, "runtime_error", str(error)))

    def check_connection(self):
        if self.mode != "client" or self.checking or not self.apply():
            return
        self.checking = True
        self.status.set(f"正在连接 {self.settings.base_url}…")
        self.refresh_controls()
        threading.Thread(target=self._health, args=(self.revision, self.settings.base_url, None), daemon=True).start()

    def _health(self, revision, base_url, server):
        try:
            if server is not None:
                health = server.health()
            else:
                from .analysis_transport import RemoteAnalysisClient
                health = RemoteAnalysisClient(base_url, timeout=5).health()
            self.events.put((revision, "runtime_health", health))
        except Exception as error:
            self.events.put((revision, "runtime_health_error", str(error)))

    def tick(self):
        while True:
            try:
                self.handle(*self.events.get_nowait())
            except queue.Empty:
                break
        if self.server is not None and not self.changing and not self.checking and time.monotonic() >= self.next_health:
            self.next_health = time.monotonic() + 1
            self.checking = True
            threading.Thread(target=self._health, args=(self.revision, None, self.server), daemon=True).start()

    def handle(self, revision, kind, payload):
        if self.closed.is_set():
            return
        if kind == "runtime_probe":
            capabilities, self.addresses = payload
            self.address_box.configure(values=self.addresses)
            self.address_hint.set("本机 IP 候选：" + "、".join(self.addresses) + "；前端需能访问所选 IP 与端口。")
            self.capabilities.set(self._capability_text("本机", capabilities))
            return
        if kind == "runtime_probe_error":
            self.capabilities.set("计算能力检测失败：" + payload)
            return
        if revision != self.revision:
            return
        if kind in ("runtime_health", "runtime_health_error"):
            self.checking = False
        else:
            self.changing = False
        if kind in ("runtime_health", "runtime_server_started"):
            self.server_health = payload
            self.server_capabilities.set(self._capability_text("计算服务（本机）" if self.mode == "server" else "服务器", payload.get("capabilities") or {}))
            jobs = payload.get("jobs", [])
            job = payload.get("active_job")
            if not isinstance(job, dict):
                job = next((item for item in jobs if item.get("status", "").lower() in ("running", "queued")), None)
            detail = "空闲"
            if job:
                detail = str(job.get("progress") or job.get("stage") or job.get("status") or "处理中")
            elif jobs:
                latest = jobs[0]
                detail = f"最近任务：{latest.get('status', '')} {latest.get('progress') or ''}"
            self.status.set(f"{'计算服务运行中' if self.mode == 'server' else '服务器已连接'} · {detail}")
        elif kind == "runtime_server_stopped":
            self.server = None
            self.server_health = {}
            self.status.set("服务器端 · 服务已停止")
        else:
            self.status.set(("连接失败：" if kind == "runtime_health_error" else "服务操作失败：") + str(payload))
        self.refresh_controls()

    @staticmethod
    def _capability_text(label, capabilities):
        cpu, gpu = capabilities.get("cpu", {}), capabilities.get("gpu", {})
        backend = capabilities.get("execution_backend") or capabilities.get("backend") or "未报告"
        preprocessing = capabilities.get('preprocessing', {})
        denoising_note = ''
        if preprocessing.get('clahe_available') is False:
            denoising_note = ' 缺少弱光处理接口，请补齐 OpenCV 或使用原始灰度。'
        elif preprocessing.get('fast_nl_means_available') is False:
            denoising_note = (' 弱光去噪使用双边兼容处理（缺少 photo 接口）。'
                if preprocessing.get('bilateral_available') and preprocessing.get('clahe_available')
                else ' 缺少弱光处理接口，请补齐 OpenCV 或使用原始灰度。')
        return (f"{label}计算：{backend} · CPU {cpu.get('logical_cores', '未报告')} 核；"
                f"CUDA 设备 {gpu.get('cuda_device_count', '未报告')} 个，"
                + ("GPU 执行可用。" if gpu.get("execution_supported") else "GPU 执行暂未启用。") + denoising_note)

    def permits(self, *, camera=False):
        if self.mode == "server" or self.changing:
            self.owner.message.set("服务器端仅接收计算任务；请在运行设置中切换到前端或本地运算。")
            return False
        return True

    def start_remote(self, arguments, generation):
        event = threading.Event()
        self.active = (generation, event)
        self.status.set("正在保存当前批并提交服务器…")
        self.refresh_controls()
        threading.Thread(target=self._remote_worker,
                         args=(arguments, generation, event, self.settings.base_url), daemon=True).start()

    def _remote_worker(self, arguments, generation, event, base_url):
        path = None
        try:
            from .analysis_transport import RemoteAnalysisClient
            fingerprints = input_fingerprints(arguments)
            arguments = dict(arguments, previous_manifest=None)
            path, reset = save_scene_snapshot(arguments)
            if event.is_set():
                return
            report = RemoteAnalysisClient(base_url).analyze(
                path, report_output_path=path.parent / "report.json", evidence_dir=path.parent / "evidence",
                progress=lambda text: self.owner.messages.put((generation, "remote_progress", (event, str(text)))),
                cancel_event=event)
            self.owner.messages.put((generation, "remote_ok", (event, path, report, reset, fingerprints)))
        except Exception as error:
            self.owner.messages.put((generation, "remote_error", (event, path, error)))

    def accept_remote(self, generation, kind, payload):
        """Return an accepted ordinary GUI event, or consume an obsolete event."""
        owner = self.owner
        event = payload[0]
        if self.active != (generation, event) or event.is_set() or owner.closed:
            return None
        if generation != owner.generation:
            self.cancel_remote(notify=False)
            self.status.set("输入已改变；旧远程任务不再载入，已保存的现场保留在历史目录。")
            return None
        if kind == "remote_progress":
            self.status.set(payload[1])
            owner.message.set(payload[1])
            return None
        self.active = None
        if kind == "remote_error":
            _, path, error = payload
            self.status.set(f"远程分析失败：{error}")
            suffix = f"\n当前批现场包已保存：{path}" if path else ""
            return "error", ValueError(str(error) + suffix)
        _, path, report, reset, fingerprints = payload
        try:
            from .ai_gui import report_matches_scene
            if not report_matches_scene(path, report):
                raise ValueError("远程报告与当前批的输入哈希不一致。")
            for name, digest in fingerprints.items():
                if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                    raise ValueError("现场文件在计算时发生改变，旧远程结果未载入。")
            for role in ("left", "right"):
                if str(Path(owner.fields[role].get()).resolve()) != owner.image_paths.get(role):
                    raise ValueError("当前照片已改变，旧远程结果未载入。")
        except (OSError, ValueError, KeyError, TypeError) as error:
            owner.busy = False
            owner._set_controls(True)
            self.status.set(f"{error} 历史包：{path.parent}")
            owner.message.set(self.status.get())
            return None
        self.status.set("服务器分析完成 · 当前批报告已接收")
        return "ok", (path, report, reset)

    def cancel_remote(self, *, notify=True):
        if self.active is None:
            return
        self.active[1].set()
        self.active = None
        self.owner.busy = False
        self.owner._set_controls(True)
        if notify:
            self.status.set("已取消远程任务；已保存的现场与历史结果保留。")
            self.owner.message.set(self.status.get())

    def close(self):
        self.closed.set()
        if self.active is not None:
            self.active[1].set()
            self.active = None
        with self.lock:
            server = self.server
            self.server = None
        if server is not None:
            threading.Thread(target=server.stop, daemon=True).start()
