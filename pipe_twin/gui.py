"""Fail-closed local dashboard for CAD-bound pipe installation results.

The data binding and projection helpers in this module deliberately have no
Tk dependency.  ``tkinter`` is imported only while launching the desktop
application so the existing command-line tools and headless CI remain usable.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import os
import queue
import re
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .logging_config import get_logger, log_event
from .dxf_elevation import DxfElevation, arc_points, catalog_from_dxf, read_dxf_elevation
from .gui_dashboard import (
    INSTALLATION_STATES,
    STATE_PRESENTATION,
    _FIELD_CALIBRATION_ID_MARKERS,
    _STEREO_EVIDENCE_TYPES,
    _STEREO_OCCLUSION_STATES,
    _STEREO_VIEW_ROLES,
    _STEREO_VISIBILITY_VALUES,
    _UNASSESSABLE_VIEW_STATES,
    _WORKBENCH_ROOT,
    _coerce_centerline,
    _coerce_positive_number,
    _empty_dashboard,
    _explicitly_unassessable_in_both_stereo_views,
    _first_manifest_stereo_paths,
    _issue,
    _loaded_asset_hashes,
    _loaded_photo_hashes,
    _normal_sha256,
    _project_dashboard_pipes,
    _read_json_object,
    _report_latest_photo_hashes,
    _report_manifest_hash,
    _report_model_hashes,
    _require_mapping,
    _require_nonempty_string,
    _safe_manifest_asset_path,
    _safe_reason_codes,
    _safe_visibility,
    _schema_two_pipe_state_issues,
    _schema_two_view_issues,
    _sha256_file,
    build_dashboard_model,
    summarize_stereo_diagnostics,
)

_LOGGER = get_logger("gui")


class _PipeTwinApplication:
    """Tk adapter.  It is instantiated only by :func:`launch_gui`."""

    def __init__(
        self,
        root: Any,
        manifest_path: str | Path | None,
        report_path: str | Path | None,
    ) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.root = root
        self.manifest_path = Path(manifest_path).resolve() if manifest_path else None
        self.report_path = Path(report_path).resolve() if report_path else None
        self.manifest: dict[str, Any] = {}
        self.report: dict[str, Any] | None = None
        self.dashboard: dict[str, Any] = _empty_dashboard(message="载入现场清单，或打开合成示例开始使用。")
        self.selected_pipe_id: str | None = None
        self.projection_mode = tk.StringVar(value="elevation")
        self.banner_text = tk.StringVar(value="正在载入……")
        self.summary_text = tk.StringVar(value="")
        self.detail_text = tk.StringVar(value="")
        self._worker_messages: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._analysis_running = False
        self._photo_paths: dict[str, Path] = {}
        self._photo_images: dict[str, Any] = {}
        self._photo_geometry: dict[str, tuple[float, float, int, int]] = {}
        self.dxf_elevation: DxfElevation | None = None
        self.dxf_layer_colors: dict[str, str] = {}
        self.dxf_diameter_colors: dict[str, str] = {}
        self.dxf_bindings: dict[str, str] = {}
        self.selected_dxf_entity_id: str | None = None
        self._manifest_sha256: str | None = None
        self._model_actual_sha256: str | None = None
        self._photo_actual_sha256: dict[str, str] = {}

        self.root.title("管件测量工作台 · 外径 / 中心距 / 净距 / 前后状态")
        self.root.geometry("1480x900")
        self.root.minsize(1180, 780)
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei UI", 10))
        style.configure("Treeview", rowheight=28)
        self._build_widgets()
        if self.manifest_path:
            self._load_sources(self.manifest_path, self.report_path)
        else:
            # Without CLI sources the last saved workbench session is reopened;
            # the restore helper repaints the dashboard itself when there is
            # nothing to restore.
            self._restore_workbench_session()
        self.root.after(80, self._poll_worker)

    def _build_widgets(self) -> None:
        tk, ttk = self.tk, self.ttk
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="基础立面评估", command=self._open_elevation).pack(side="left", padx=3)
        ttk.Label(toolbar, text="现场识别入口：导入 DXF/STL → 双目抓拍 → 自动匹配 → 状态刷新",
                  foreground="#355371").pack(side="left", padx=(14, 3))
        # The legacy dashboard actions remain callable for report rendering and
        # command-line integrations, but are intentionally no longer exposed
        # as operator buttons.  Keep the widget object for those guarded paths.
        self.run_button = ttk.Button(toolbar, text="运行双目识别与测量", command=self._run_analysis)
        ttk.Label(toolbar, text="模型视图：").pack(side="left", padx=(18, 2))
        projection = ttk.Combobox(
            toolbar,
            textvariable=self.projection_mode,
            values=("elevation", "isometric", "dxf"),
            state="readonly",
            width=11,
        )
        projection.pack(side="left")
        projection.bind("<<ComboboxSelected>>", lambda _event: self._draw_model())

        self.banner = tk.Label(
            self.root,
            textvariable=self.banner_text,
            anchor="w",
            padx=10,
            pady=7,
            bg="#FFF3CD",
            fg="#5F4500",
        )
        self.banner.pack(fill="x")

        self.main_tabs = ttk.Notebook(self.root)
        self.main_tabs.pack(fill="both", expand=True)
        measurement_tab = ttk.Frame(self.main_tabs)
        cad_tab = ttk.Frame(self.main_tabs)
        self.main_tabs.add(measurement_tab, text="测量工作台")
        self.main_tabs.add(cad_tab, text="CAD 与识别证据")
        from .measurement_gui import MeasurementPanel

        self.measurement_panel = MeasurementPanel(measurement_tab, self)
        panes = ttk.Panedwindow(cad_tab, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=8, pady=8)
        left = ttk.Frame(panes)
        right = ttk.Frame(panes, width=560)
        panes.add(left, weight=3)
        panes.add(right, weight=2)

        notebook = ttk.Notebook(left)
        notebook.pack(fill="both", expand=True)
        model_tab = ttk.Frame(notebook)
        stereo_tab = ttk.Frame(notebook)
        notebook.add(model_tab, text="CAD 状态视图")
        notebook.add(stereo_tab, text="左右照片")
        preview_toolbar = ttk.Frame(stereo_tab)
        preview_toolbar.pack(fill="x")
        for role, label in (("left", "左"), ("right", "右")):
            ttk.Button(preview_toolbar, text=f"选择{label}图（仅预览）", command=lambda r=role: self._choose_photo(r)).pack(side="left", padx=3)

        self.model_canvas = tk.Canvas(model_tab, bg="#F4F6F8", highlightthickness=0)
        self.model_canvas.pack(fill="both", expand=True)
        self.model_canvas.bind("<Configure>", lambda _event: self._draw_model())
        self.model_canvas.bind("<Button-1>", self._select_canvas_pipe)

        stereo_panes = ttk.Panedwindow(stereo_tab, orient="horizontal")
        stereo_panes.pack(fill="both", expand=True)
        self.photo_canvases: dict[str, Any] = {}
        for role, title in (("left", "左相机"), ("right", "右相机")):
            frame = ttk.LabelFrame(stereo_panes, text=title, padding=4)
            canvas = tk.Canvas(frame, bg="#202124", highlightthickness=0)
            canvas.pack(fill="both", expand=True)
            canvas.bind("<Configure>", lambda _event, r=role: self._render_photo(r))
            self.photo_canvases[role] = canvas
            stereo_panes.add(frame, weight=1)

        ttk.Label(right, textvariable=self.summary_text, anchor="w").pack(
            fill="x", pady=(0, 6)
        )
        columns = (
            "state",
            "diameter",
            "layer",
            "left",
            "right",
            "reason",
        )
        self.tree = ttk.Treeview(right, columns=columns, show="tree headings", selectmode="browse")
        self.tree.heading("#0", text="pipe_id")
        self.tree.heading("state", text="状态")
        self.tree.heading("diameter", text="设计外径/mm")
        self.tree.heading("layer", text="层")
        self.tree.heading("left", text="左目")
        self.tree.heading("right", text="右目")
        self.tree.heading("reason", text="依据")
        self.tree.column("#0", width=220, stretch=True)
        self.tree.column("state", width=115, anchor="center")
        self.tree.column("diameter", width=72, anchor="e")
        self.tree.column("layer", width=70, anchor="center")
        self.tree.column("left", width=125, anchor="center")
        self.tree.column("right", width=125, anchor="center")
        self.tree.column("reason", width=230, stretch=True)
        for state, presentation in STATE_PRESENTATION.items():
            self.tree.tag_configure(state, foreground=presentation["color"])
        scroll_y = ttk.Scrollbar(right, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll_y.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll_y.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._select_tree_pipe)

        detail = ttk.Label(
            cad_tab,
            textvariable=self.detail_text,
            anchor="w",
            justify="left",
            padding=(10, 5),
        )
        detail.pack(fill="x")

    def _current_photo_hashes(self) -> dict[str, str]:
        """Refresh hashes for the manifest-bound pair, never preview overrides."""

        if not self.manifest:
            return {}
        try:
            return _loaded_photo_hashes(self.manifest_path, self.manifest)
        except (OSError, ValueError):
            return {role: "" for role in _STEREO_VIEW_ROLES}

    def _clear_loaded_state(
        self,
        *,
        manifest: object = None,
        manifest_path: Path | None = None,
        reason_code: str = "NO_ACTIVE_SOURCES",
        message: str = "No manifest/report is loaded",
        preserve_manifest: bool = False,
    ) -> None:
        """Drop stale report state and repaint a safe dashboard after an error."""

        candidate_manifest = manifest if isinstance(manifest, dict) else None
        if preserve_manifest and candidate_manifest is not None:
            candidate_path = (
                Path(manifest_path).resolve()
                if manifest_path is not None
                else self.manifest_path
            )
            try:
                # A parsed manifest can still be displayed as an all-unknown
                # CAD list even when its file was removed immediately after
                # parsing.  Hash each source independently and let the normal
                # binding checks record unavailable assets.
                try:
                    manifest_hash = _sha256_file(candidate_path)
                except OSError:
                    manifest_hash = None
                model_spec = candidate_manifest.get("model")
                model_path = (
                    _safe_manifest_asset_path(candidate_path, model_spec.get("path"))
                    if isinstance(model_spec, Mapping)
                    else None
                )
                try:
                    model_hash = (
                        _sha256_file(model_path)
                        if model_path is not None and model_path.is_file()
                        else ""
                    )
                except OSError:
                    model_hash = ""
                photo_hashes = _loaded_photo_hashes(candidate_path, candidate_manifest)
                safe_dashboard = build_dashboard_model(
                    candidate_manifest,
                    None,
                    manifest_sha256=manifest_hash,
                    model_actual_sha256=model_hash or None,
                    photo_actual_sha256=photo_hashes,
                )
            except Exception:
                safe_dashboard = _empty_dashboard(
                    reason_code=reason_code,
                    message=message,
                )
            self.manifest = candidate_manifest
            self.manifest_path = candidate_path
            self._manifest_sha256 = (
                manifest_hash if "manifest_hash" in locals() else None
            )
            self._model_actual_sha256 = (
                model_hash if "model_hash" in locals() else None
            )
            self._photo_actual_sha256 = (
                photo_hashes if "photo_hashes" in locals() else {}
            )
        else:
            self.manifest = {}
            if manifest_path is not None:
                self.manifest_path = Path(manifest_path).resolve()
            self._manifest_sha256 = None
            self._model_actual_sha256 = None
            self._photo_actual_sha256 = {}
            safe_dashboard = _empty_dashboard(
                reason_code=reason_code,
                message=message,
            )
        self.report = None
        self.report_path = None
        self.dashboard = safe_dashboard
        self.selected_pipe_id = None
        self._photo_paths.clear()
        self._photo_images.clear()
        self._photo_geometry.clear()
        if preserve_manifest and self.manifest:
            self._photo_paths.update(
                _first_manifest_stereo_paths(self.manifest_path, self.manifest)
            )
        # During construction or headless tests the Tk widgets may not yet
        # exist.  State is still cleared; repaint only when the adapter is
        # fully initialized.
        if hasattr(self, "tree"):
            self._refresh_dashboard()
            for role in _STEREO_VIEW_ROLES:
                self._render_photo(role)

    def _load_sources(self, manifest_path: Path, report_path: Path | None) -> None:
        manifest_path = Path(manifest_path).resolve()
        report_path = Path(report_path).resolve() if report_path is not None else None
        log_event(_LOGGER, "gui_sources_load_start", manifest=manifest_path, report=report_path)
        try:
            manifest = _read_json_object(manifest_path, "manifest")
            if isinstance(manifest.get("analysis"), Mapping) and manifest["analysis"].get("mode") == "elevation_auto":
                # Automatic elevation reports have their own presentation contract;
                # keep the CAD dashboard's geometry validation unchanged.
                self._open_elevation(manifest_path)
                return
            report = _read_json_object(report_path, "report") if report_path else None
            manifest_sha256, model_actual_sha256 = _loaded_asset_hashes(
                manifest_path, manifest
            )
            photo_actual_sha256 = _loaded_photo_hashes(manifest_path, manifest)
            dashboard = build_dashboard_model(
                manifest,
                report,
                manifest_sha256=manifest_sha256,
                model_actual_sha256=model_actual_sha256,
                photo_actual_sha256=photo_actual_sha256,
            )
        except Exception as error:
            # Never leave a previously loaded decisive result on screen after
            # a failed source switch.  If the new manifest parsed, retain its
            # CAD list but rebuild it without a report (all pipes UNKNOWN);
            # otherwise replace the dashboard with an empty fail-closed model.
            self._clear_loaded_state(
                manifest=locals().get("manifest"),
                manifest_path=manifest_path,
                reason_code="LOAD_FAILED",
                message=f"载入失败：{error}",
                preserve_manifest=isinstance(locals().get("manifest"), dict),
            )
            self.messagebox.showerror("载入失败", str(error))
            return
        self.manifest_path = manifest_path
        self.report_path = report_path
        self.manifest = manifest
        self.report = report
        self.dashboard = dashboard
        self._restore_dxf_from_manifest(manifest_path, manifest)
        self._manifest_sha256 = manifest_sha256
        self._model_actual_sha256 = model_actual_sha256
        self._photo_actual_sha256 = photo_actual_sha256
        self.selected_pipe_id = None
        self._photo_paths = _first_manifest_stereo_paths(self.manifest_path, self.manifest)
        self._photo_images.clear()
        self._photo_geometry.clear()
        self._refresh_dashboard()
        log_event(
            _LOGGER,
            "gui_sources_loaded",
            manifest=manifest_path,
            report=report_path,
            pipe_count=len(self.dashboard.get("pipes", [])),
            binding_valid=self.dashboard.get("binding_valid"),
        )
        for role in ("left", "right"):
            self._render_photo(role)

    def _restore_workbench_session(self) -> None:
        """Reopen the last saved manifest so a new session starts warm."""

        from .workbench_profile import load_profile

        try:
            profile, problem = load_profile()
        except Exception as error:  # a broken profile must never block startup
            self.banner.configure(bg="#FFF3CD", fg="#5F4500")
            self.banner_text.set("工作台配置未载入：" + str(error))
            log_event(_LOGGER, "gui_session_restore_failed", error=str(error))
            return
        if problem:
            self.banner.configure(bg="#FFF3CD", fg="#5F4500")
            self.banner_text.set("工作台配置未载入：" + problem)
            return
        if not profile:
            self._refresh_dashboard()
            return
        last = str(profile.get("last_manifest_path", "") or "")
        if last and Path(last).is_file():
            try:
                self._load_sources(Path(last), None)
            except Exception as error:
                # _load_sources already fails closed and reports to the
                # operator; log so the restore attempt stays auditable.
                log_event(_LOGGER, "gui_session_restore_failed", manifest=last, error=str(error))
            else:
                log_event(_LOGGER, "gui_session_restored", manifest=last)
                note = f"已恢复上次工作台会话：{last}"
                current = self.banner_text.get()
                # Append instead of replacing so the binding verdict that
                # _refresh_dashboard just painted stays visible.
                self.banner_text.set(f"{note}｜{current}" if current else note)
                return
        self._refresh_dashboard()

    def _restore_dxf_from_manifest(self, manifest_path: Path, manifest: Mapping[str, Any]) -> None:
        elevation = manifest.get("elevation")
        if not isinstance(elevation, Mapping) or elevation.get("format") != "dxf":
            return
        path = _safe_manifest_asset_path(manifest_path, elevation.get("path"))
        try:
            document = read_dxf_elevation(path) if path is not None else None
            if document is None or str(elevation.get("sha256", "")).lower() != document.source_sha256:
                raise ValueError("DXF 文件哈希与 manifest 不一致")
            self.dxf_elevation = document
            self.dxf_layer_colors = dict(document.layers)
            self.dxf_diameter_colors = {str(k): str(v).upper() for k, v in (elevation.get("diameter_colors") or {}).items()} if isinstance(elevation.get("diameter_colors"), Mapping) else {}
            self.dxf_bindings = {str(k): str(v) for k, v in (elevation.get("entity_bindings") or {}).items()} if isinstance(elevation.get("entity_bindings"), Mapping) else {}
        except (OSError, ValueError):
            self.dxf_elevation = None
            self.dxf_layer_colors = {}
            self.dxf_diameter_colors = {}
            self.dxf_bindings = {}

    def _choose_manifest(self) -> None:
        selected = self.filedialog.askopenfilename(
            title="选择双目采集清单",
            filetypes=(("JSON", "*.json"), ("All files", "*.*")),
        )
        if selected:
            self._load_sources(Path(selected), None)

    def _choose_report(self) -> None:
        if not self.manifest:
            self.messagebox.showinfo("先载入清单", "请先载入现场清单或打开示例，再载入对应识别结果。")
            return
        selected = self.filedialog.askopenfilename(
            title="选择识别结果",
            filetypes=(("JSON", "*.json"), ("All files", "*.*")),
        )
        if not selected:
            return
        current_photo_hashes = self._current_photo_hashes()
        try:
            report = _read_json_object(selected, "report")
            dashboard = build_dashboard_model(
                self.manifest,
                report,
                manifest_sha256=self._manifest_sha256,
                model_actual_sha256=self._model_actual_sha256,
                photo_actual_sha256=current_photo_hashes,
            )
        except Exception as error:
            # Keep the CAD manifest useful, but discard the old report and its
            # dashboard immediately.  Rebuilding with ``None`` guarantees no
            # old INSTALLED/NOT_INSTALLED state survives a bad report load.
            self._clear_loaded_state(
                manifest=self.manifest,
                manifest_path=self.manifest_path,
                reason_code="REPORT_LOAD_FAILED",
                message=f"结果载入失败：{error}",
                preserve_manifest=True,
            )
            self.messagebox.showerror("结果载入失败", str(error))
            return
        self.report_path = Path(selected).resolve()
        self.report = report
        self.dashboard = dashboard
        self._photo_actual_sha256 = current_photo_hashes
        self.selected_pipe_id = None
        self._refresh_dashboard()

    def _choose_photo(self, role: str) -> None:
        selected = self.filedialog.askopenfilename(
            title=f"选择{'左' if role == 'left' else '右'}相机照片（仅预览）",
            filetypes=(
                ("Images", "*.png *.jpg *.jpeg"),
                ("All files", "*.*"),
            ),
        )
        if selected:
            self._photo_paths[role] = Path(selected).resolve()
            self._render_photo(role)

    def _refresh_dashboard(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        counts = self.dashboard["counts"]
        self.summary_text.set(
            f"安装 {counts['INSTALLED']}  |  未安装 {counts['NOT_INSTALLED']}  |  "
            f"不确定 {counts['UNKNOWN']}"
        )
        for pipe in self.dashboard["pipes"]:
            visibility = pipe["visibility_by_view"]
            left = self._view_label(visibility.get("left"))
            right = self._view_label(visibility.get("right"))
            reasons = ", ".join(pipe["reason_codes"]) or pipe["state_basis"]
            diameter = pipe["nominal_diameter_mm"]
            self.tree.insert(
                "",
                "end",
                iid=pipe["pipe_id"],
                text=pipe["pipe_id"],
                values=(
                    f"{pipe['state_symbol']} {pipe['installation_state_zh']}",
                    f"{diameter:g}" if diameter is not None else "—",
                    pipe["layer_id"] or "—",
                    left,
                    right,
                    reasons,
                ),
                tags=(pipe["installation_state"],),
            )
        if self.dashboard["binding_valid"]:
            self.banner.configure(bg="#D7F3E3", fg="#174D2D")
            self.banner_text.set(
                f"结果与 CAD 模型绑定有效：{self.dashboard['model']['path']}"
            )
        else:
            self.banner.configure(bg="#FFF3CD", fg="#5F4500")
            messages = "；".join(
                issue["message"] for issue in self.dashboard["binding_issues"]
            )
            self.banner_text.set(f"安全降级：全部管道显示为不确定。{messages}")
        self._draw_model()
        self._update_detail()
        if hasattr(self, "measurement_panel"):
            self.measurement_panel.refresh()
        if not self.manifest:
            self.banner_text.set("开始：载入现场清单 / 现场数据录入；也可打开合成示例熟悉操作。")
        elif "SYNTHETIC" in self.manifest.get("validation_scope", ""):
            self.banner_text.set("合成示例数据 · 可用于软件操作与算法回归；现场精度需要真实双目照片与实测样本验证。")

    @staticmethod
    def _view_label(record: object) -> str:
        if not isinstance(record, Mapping):
            return "未报告"
        value = record.get("occlusion_state", record.get("visibility", "未报告"))
        return str(value)

    def _draw_model(self) -> None:
        if not self.dashboard:
            return
        canvas = self.model_canvas
        width = max(canvas.winfo_width(), 400)
        height = max(canvas.winfo_height(), 300)
        canvas.delete("all")
        mode = self.projection_mode.get()
        if mode == "dxf":
            self._draw_dxf()
            return
        projected = _project_dashboard_pipes(self.dashboard, width, height, mode)
        canvas.create_text(
            16,
            14,
            anchor="nw",
            text=(
                "CAD 状态立面（X/Y）" if mode == "elevation" else "CAD 状态等轴示意"
            ),
            fill="#263238",
            font=("Segoe UI", 12, "bold"),
        )
        legend_x = 18
        for state in INSTALLATION_STATES:
            style = STATE_PRESENTATION[state]
            canvas.create_text(
                legend_x,
                42,
                anchor="nw",
                text=f"{style['symbol']} {style['label_zh']} ({state})",
                fill=style["color"],
                font=("Segoe UI", 10, "bold"),
            )
            legend_x += 165

        for pipe in projected:
            (x1, y1), (x2, y2) = pipe["canvas_centerline"]
            selected = pipe["pipe_id"] == self.selected_pipe_id
            if selected:
                canvas.create_line(
                    x1,
                    y1,
                    x2,
                    y2,
                    width=pipe["canvas_width"] + 7,
                    fill="#1565C0",
                    capstyle="round",
                )
            canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                width=pipe["canvas_width"] + 2,
                fill="#263238",
                capstyle="round",
            )
            line_id = canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                width=pipe["canvas_width"],
                fill=pipe["state_color"],
                dash=pipe["state_dash"] or None,
                capstyle="round",
                tags=("cad_pipe", f"pipe::{pipe['pipe_id']}"),
            )
            canvas.tag_bind(
                line_id,
                "<Button-1>",
                lambda _event, pipe_id=pipe["pipe_id"]: self._select_pipe(pipe_id),
            )
            canvas.create_text(
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
                text=f"{pipe['state_symbol']} {pipe['pipe_id']}",
                fill="#111111",
                font=("Segoe UI", 9, "bold"),
                tags=("cad_pipe_label", f"pipe::{pipe['pipe_id']}"),
            )
        if not projected:
            canvas.create_text(
                width / 2,
                height / 2,
                text="清单中没有可绘制的 centerline_world_mm\n状态仍可在右侧列表查看",
                justify="center",
                fill="#616161",
                font=("Segoe UI", 12),
            )

    def _draw_dxf(self) -> None:
        document = self.dxf_elevation
        canvas = self.model_canvas
        width = max(canvas.winfo_width(), 400)
        height = max(canvas.winfo_height(), 300)
        if document is None:
            canvas.create_text(width / 2, height / 2, text="请先导入 DXF 侧立面", fill="#616161")
            return
        drawable = [point for entity in document.entities for point in (arc_points(entity) if entity.kind in {"ARC", "CIRCLE"} else entity.points)]
        min_x = min(point[0] for point in drawable); max_x = max(point[0] for point in drawable)
        min_y = min(point[1] for point in drawable); max_y = max(point[1] for point in drawable)
        margin = 48.0
        scale = min((width - 2 * margin) / max(max_x - min_x, 1.0), (height - 2 * margin) / max(max_y - min_y, 1.0))
        def xy(point: tuple[float, float]) -> tuple[float, float]:
            return (margin + (point[0] - min_x) * scale, height - margin - (point[1] - min_y) * scale)
        canvas.create_text(16, 14, anchor="nw", text="DXF 侧立面（图层颜色）", fill="#263238", font=("Segoe UI", 12, "bold"))
        for entity in document.entities:
            diameter_key = self._dxf_entity_diameter_key(entity)
            color = self.dxf_diameter_colors.get(diameter_key, entity.color)
            if entity.kind == "CIRCLE":
                cx, cy = xy(entity.points[0]); radius = float(entity.radius or 0) * scale
                item = canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline=color, width=2, tags=("dxf_entity", f"dxf::{entity.entity_id}"))
            else:
                vertices = arc_points(entity) if entity.kind == "ARC" else entity.points
                item = canvas.create_line([coordinate for point in vertices for coordinate in xy(point)], fill=color, width=2, smooth=entity.kind == "ARC", tags=("dxf_entity", f"dxf::{entity.entity_id}"))
            canvas.tag_bind(item, "<Button-1>", lambda _event, entity_id=entity.entity_id: self._select_dxf_entity(entity_id))
            if entity.points:
                middle = entity.points[len(entity.points) // 2]
                label_x, label_y = xy(middle)
                binding = self.dxf_bindings.get(entity.entity_id)
                label = f"{entity.entity_id} → {binding}" if binding else entity.entity_id
                canvas.create_text(label_x + 4, label_y - 4, anchor="sw", text=label, fill=color, font=("Segoe UI", 8, "bold"), tags=("dxf_entity", f"dxf::{entity.entity_id}"))
        canvas.create_text(16, 38, anchor="nw", text=f"文件：{document.source_path.name} · 图层 {len(document.layers)} · 实体 {len(document.entities)}", fill="#455A64")

    def _import_dxf(self) -> None:
        selected = self.filedialog.askopenfilename(title="导入 DXF 侧立面", filetypes=(("DXF", "*.dxf"), ("All files", "*.*")))
        if not selected:
            return
        try:
            document = read_dxf_elevation(selected)
        except Exception as error:
            log_event(_LOGGER, "dxf_import_failed", path=selected, error=str(error))
            self.messagebox.showerror("DXF 导入失败", str(error))
            return
        self.dxf_elevation = document
        self.dxf_layer_colors = dict(document.layers)
        self.dxf_diameter_colors = {}
        self.projection_mode.set("dxf")
        self._draw_model()
        log_event(_LOGGER, "dxf_import_finished", path=document.source_path, entity_count=len(document.entities), layer_count=len(document.layers))

    def _automate_dxf_setup(self) -> None:
        """Run the common DXF import → numbering → manifest-draft workflow."""

        if self.dxf_elevation is None:
            self._import_dxf()
        if self.dxf_elevation is None:
            return
        self._create_dxf_manifest_draft()
        log_event(_LOGGER, "dxf_automated_setup_finished", path=self.dxf_elevation.source_path)

    @staticmethod
    def _dxf_entity_diameter_key(entity: Any) -> str:
        diameter = float(entity.radius or 0.0) * 2.0 if entity.kind in {"CIRCLE", "ARC"} and entity.radius is not None else 1.0
        return f"{diameter:.6f}"

    def _configure_dxf_diameter_colors(self) -> None:
        if self.dxf_elevation is None:
            self.messagebox.showinfo("未导入 DXF", "请先导入 DXF 侧立面。")
            return
        from tkinter import colorchooser
        window = self.tk.Toplevel(self.root)
        window.title("按管径统一颜色")
        window.transient(self.root)
        keys = sorted({self._dxf_entity_diameter_key(entity) for entity in self.dxf_elevation.entities}, key=float)
        variables: dict[str, Any] = {}
        for row, key in enumerate(keys):
            self.ttk.Label(window, text=f"管径 {float(key):g} mm").grid(row=row, column=0, padx=10, pady=5, sticky="w")
            variable = self.tk.StringVar(value=self.dxf_diameter_colors.get(key, next((entity.color for entity in self.dxf_elevation.entities if self._dxf_entity_diameter_key(entity) == key), "#FFFFFF")))
            variables[key] = variable
            self.ttk.Entry(window, textvariable=variable, width=12).grid(row=row, column=1, padx=5, pady=5)
            self.ttk.Button(window, text="选择", command=lambda var=variable: var.set(colorchooser.askcolor(color=var.get(), parent=window)[1] or var.get())).grid(row=row, column=2, padx=5, pady=5)
        def save() -> None:
            self.dxf_diameter_colors = {key: value.get().upper() for key, value in variables.items() if value.get()}
            self.projection_mode.set("dxf")
            self._draw_model()
            log_event(_LOGGER, "dxf_diameter_colors_changed", diameter_count=len(self.dxf_diameter_colors))
            window.destroy()
        self.ttk.Button(window, text="应用", command=save).grid(row=len(keys), column=0, columnspan=3, pady=10)

    def _create_dxf_manifest_draft(self) -> None:
        if self.dxf_elevation is None:
            self.messagebox.showinfo("未导入 DXF", "请先导入 DXF 侧立面。")
            return
        try:
            pipes, skipped = catalog_from_dxf(
                self.dxf_elevation.source_path,
                axis_world=(0.0, 0.0, 1.0),
                diameter_colors=self.dxf_diameter_colors,
            )
        except (OSError, ValueError) as error:
            self.messagebox.showerror("无法生成草稿", str(error))
            return
        if not pipes:
            self.messagebox.showerror("无法生成草稿", "DXF 中没有可转换为管道目录的图元。")
            return
        selected = self.filedialog.asksaveasfilename(title="保存 DXF manifest 草稿", initialfile=f"{self.dxf_elevation.source_path.stem}.manifest.json", initialdir=str(self.dxf_elevation.source_path.parent), defaultextension=".json", filetypes=(("JSON", "*.json"),))
        if not selected:
            return
        selected_path = Path(selected).resolve()
        relative = os.path.relpath(self.dxf_elevation.source_path, selected_path.parent).replace("\\", "/")
        if relative.startswith("../") or relative == "..":
            self.messagebox.showerror("保存失败", "DXF 文件必须位于 manifest 草稿目录或其子目录内。")
            return
        payload = {"schema_version": "dxf-elevation-draft-v1", "dataset_id": f"dxf-draft-{self.dxf_elevation.source_sha256[:12]}", "model_revision": self.dxf_elevation.source_sha256[:16], "validation_scope": "DXF_ELEVATION_DRAFT", "scope": {"layer_model": "elevation", "layer_id": "DXF", "identity_features": ["dxf_entity_id", "layer_id"], "metric_calibrated": False, "supports_stereo": False, "supports_occlusion_reasoning": False}, "model": {"path": relative, "sha256": self.dxf_elevation.source_sha256, "unit": "millimeter", "source_unit": self.dxf_elevation.source_unit, "format": "dxf", "geometry": "circle_layout_xy", "longitudinal_extent_known": False, "axis_world": [0.0, 0.0, 1.0], "pipe_half_length_mm": 1000.0, "pipes": pipes}, "elevation": {"format": "dxf", "path": relative, "sha256": self.dxf_elevation.source_sha256, "diameter_colors": dict(self.dxf_diameter_colors), "entity_bindings": {entity["pipe_id"]: entity["pipe_id"] for entity in pipes}, "skipped_entity_ids": [entity.entity_id for entity in skipped]}}
        selected_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._load_sources(selected_path, None)
        self.projection_mode.set("dxf")
        self._draw_model()
        log_event(_LOGGER, "dxf_manifest_draft_created", manifest=selected_path, pipe_count=len(pipes))

    def _select_dxf_entity(self, entity_id: str) -> None:
        self.selected_dxf_entity_id = entity_id
        self.projection_mode.set("dxf")
        self._draw_model()
        log_event(_LOGGER, "dxf_entity_selected", entity_id=entity_id)

    def _bind_dxf_entity(self) -> None:
        if self.dxf_elevation is None or not self.selected_dxf_entity_id:
            self.messagebox.showinfo("未选择DXF图元", "请先在 DXF 侧立面中点击一条线、圆弧或外轮廓圆。")
            return
        selected = self.tree.selection()
        if not selected:
            self.messagebox.showinfo("未选择管道", "请先在右侧管道列表中选择目标管道。")
            return
        pipe_id = str(selected[0])
        self.dxf_bindings[self.selected_dxf_entity_id] = pipe_id
        self._draw_model()
        log_event(_LOGGER, "dxf_entity_bound", entity_id=self.selected_dxf_entity_id, pipe_id=pipe_id)

    def _save_dxf_bindings(self) -> None:
        if self.dxf_elevation is None or not self.dxf_bindings:
            self.messagebox.showinfo("没有DXF映射", "请先导入 DXF 并至少绑定一个图元。")
            return
        if not self.manifest_path or not self.manifest:
            self.messagebox.showinfo("没有manifest", "请先载入或创建 manifest。")
            return
        try:
            updated = copy.deepcopy(self.manifest)
            selected = self.filedialog.asksaveasfilename(title="保存带DXF映射的manifest", initialfile=self.manifest_path.name, initialdir=str(self.manifest_path.parent), defaultextension=".json", filetypes=(("JSON", "*.json"),))
            if not selected:
                return
            selected_path = Path(selected).resolve()
            relative = os.path.relpath(self.dxf_elevation.source_path, selected_path.parent).replace("\\", "/")
            if relative.startswith("../") or relative == "..":
                raise ValueError("DXF 文件必须位于要保存的 manifest 目录或其子目录内，才能写入相对路径。")
            updated["elevation"] = {
                "format": "dxf",
                "path": relative,
                "sha256": self.dxf_elevation.source_sha256,
                "diameter_colors": dict(self.dxf_diameter_colors),
                "entity_bindings": dict(self.dxf_bindings),
            }
            selected_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            log_event(_LOGGER, "dxf_bindings_saved", manifest=selected, binding_count=len(self.dxf_bindings))
            self.messagebox.showinfo("已保存", f"DXF 映射已写入：\n{selected}\n\n报告需重新分析以匹配新的 manifest 哈希。")
        except (OSError, ValueError) as error:
            self.messagebox.showerror("保存映射失败", str(error))

    def _select_canvas_pipe(self, event: Any) -> None:
        items = self.model_canvas.find_overlapping(
            event.x - 3, event.y - 3, event.x + 3, event.y + 3
        )
        for item in reversed(items):
            for tag in self.model_canvas.gettags(item):
                if tag.startswith("pipe::"):
                    self._select_pipe(tag.split("::", 1)[1])
                    return

    def _select_tree_pipe(self, _event: Any) -> None:
        selection = self.tree.selection()
        if selection:
            self._select_pipe(selection[0], update_tree=False)

    def _select_pipe(self, pipe_id: str, *, update_tree: bool = True) -> None:
        self.selected_pipe_id = pipe_id
        if update_tree and self.tree.exists(pipe_id):
            self.tree.selection_set(pipe_id)
            self.tree.see(pipe_id)
        self._draw_model()
        for role in ("left", "right"):
            self._render_photo(role)
        self._update_detail()
        if hasattr(self, "measurement_panel"):
            self.measurement_panel.pipe_id.set(pipe_id)
            self.measurement_panel.selection_changed()

    def _selected_pipe(self) -> Mapping[str, Any] | None:
        return next(
            (
                pipe
                for pipe in self.dashboard.get("pipes", [])
                if pipe.get("pipe_id") == self.selected_pipe_id
            ),
            None,
        )

    def _update_detail(self) -> None:
        pipe = self._selected_pipe()
        if pipe is None:
            self.detail_text.set("选择一根管道可查看 CAD GUID、物理颜色和判定依据。")
            return
        reasons = ", ".join(pipe["reason_codes"]) or "无原因码"
        self.detail_text.set(
            f"{pipe['pipe_id']}  |  CAD对象 {pipe['cad_object_id'] or pipe['cad_uuid'] or '—'}  |  "
            f"材料颜色 {pipe['material_color_srgb'] or pipe['appearance_color'] or '—'}  |  "
            f"状态 {pipe['installation_state_zh']} ({pipe['installation_state']})  |  "
            f"依据 {pipe['state_basis']} / {reasons}"
        )

    def _render_photo(self, role: str) -> None:
        canvas = self.photo_canvases.get(role)
        if canvas is None:
            return
        canvas.delete("all")
        path = self._photo_paths.get(role)
        if path is None:
            canvas.create_text(
                max(canvas.winfo_width(), 300) / 2,
                max(canvas.winfo_height(), 250) / 2,
                text=f"未选择{'左' if role == 'left' else '右'}相机照片",
                fill="#DADCE0",
                font=("Segoe UI", 12),
            )
            return
        try:
            import cv2
            import numpy as np

            raw = path.read_bytes()
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("OpenCV cannot decode this image")
            source_height, source_width = image.shape[:2]
            available_width = max(canvas.winfo_width() - 20, 200)
            available_height = max(canvas.winfo_height() - 42, 160)
            scale = min(
                available_width / source_width,
                available_height / source_height,
                1.0,
            )
            display_width = max(1, round(source_width * scale))
            display_height = max(1, round(source_height * scale))
            if (display_width, display_height) != (source_width, source_height):
                image = cv2.resize(
                    image,
                    (display_width, display_height),
                    interpolation=cv2.INTER_AREA,
                )
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise ValueError("OpenCV cannot encode the preview")
            photo = self.tk.PhotoImage(
                data=base64.b64encode(encoded.tobytes()).decode("ascii"),
                format="png",
            )
            center_x = max(canvas.winfo_width(), 300) / 2
            center_y = max(canvas.winfo_height(), 250) / 2 + 10
            canvas.create_image(center_x, center_y, image=photo)
            self._photo_images[role] = photo
            offset_x = center_x - display_width / 2
            offset_y = center_y - display_height / 2
            self._photo_geometry[role] = (offset_x, offset_y, source_width, source_height)
            canvas.create_text(
                8,
                7,
                anchor="nw",
                text=path.name,
                fill="#FFFFFF",
                font=("Segoe UI", 10, "bold"),
            )
            self._draw_photo_selection(role, scale)
        except Exception as error:
            canvas.create_text(
                max(canvas.winfo_width(), 300) / 2,
                max(canvas.winfo_height(), 250) / 2,
                text=f"照片预览失败\n{error}",
                justify="center",
                fill="#FFCDD2",
                font=("Segoe UI", 11),
            )

    def _draw_photo_selection(self, role: str, scale: float) -> None:
        pipe = self._selected_pipe()
        if pipe is None:
            return
        record = pipe.get("visibility_by_view", {}).get(role)
        if not isinstance(record, Mapping):
            return
        bbox = record.get("amodal_bbox_xywh")
        if (
            not isinstance(bbox, Sequence)
            or isinstance(bbox, (str, bytes))
            or len(bbox) != 4
            or any(type(value) not in (int, float) for value in bbox)
        ):
            return
        x, y, width, height = (float(value) for value in bbox)
        if width <= 0 or height <= 0:
            return
        offset_x, offset_y, _, _ = self._photo_geometry[role]
        canvas = self.photo_canvases[role]
        canvas.create_rectangle(
            offset_x + x * scale,
            offset_y + y * scale,
            offset_x + (x + width) * scale,
            offset_y + (y + height) * scale,
            outline=pipe["state_color"],
            width=3,
            dash=(5, 3) if pipe["installation_state"] == "UNKNOWN" else None,
        )
        canvas.create_text(
            offset_x + x * scale,
            offset_y + y * scale - 4,
            anchor="sw",
            text=f"{pipe['pipe_id']} · {pipe['installation_state_zh']}",
            fill=pipe["state_color"],
            font=("Segoe UI", 9, "bold"),
        )

    def _run_analysis(self) -> None:
        if self._analysis_running:
            return
        if not self.manifest_path:
            self.messagebox.showinfo("尚未载入数据", "请先载入现场清单或打开合成示例。")
            return
        if self.manifest.get("validation_scope") == "DXF_ELEVATION_DRAFT":
            self.messagebox.showinfo("DXF 草稿不能直接双目分析", "请在基础立面评估中直接导入该 DXF，补充真实双目标定和左右照片；无需 3DM/3MF。")
            return
        try:
            options = self.measurement_panel.analysis_options()
        except ValueError as error:
            self.messagebox.showerror("测量设置无效", str(error))
            return
        self._analysis_request = (self.manifest_path, copy.deepcopy(self.manifest), options, self._manifest_sha256)
        self._analysis_running = True
        self.run_button.configure(state="disabled")
        self.banner.configure(bg="#DCEBFA", fg="#164B75")
        self.banner_text.set("正在分析双目照片、局部管径、中心距、净距与前后关系……")
        log_event(_LOGGER, "gui_analysis_start", manifest=self.manifest_path)
        thread = threading.Thread(target=self._analysis_worker, daemon=True)
        thread.start()

    def _analysis_worker(self) -> None:
        try:
            manifest_path, manifest, options, _digest = self._analysis_request
            capture = manifest.get("capture")
            capture_kind = capture.get("kind") if isinstance(capture, Mapping) else None
            if (
                manifest.get("schema_version") == "2.0"
                or capture_kind == "stereo_still_capture_set"
            ):
                from .stereo_analyzer import analyze_stereo_capture

                report = analyze_stereo_capture(manifest_path, measurement_options=options)
            else:
                from .pipeline import analyze_manifest

                report = analyze_manifest(manifest_path)
            self._worker_messages.put(("ok", report))
        except Exception as error:
            self._worker_messages.put(("error", error))

    def _poll_worker(self) -> None:
        try:
            kind, payload = self._worker_messages.get_nowait()
        except queue.Empty:
            self.root.after(80, self._poll_worker)
            return
        self._analysis_running = False
        self.run_button.configure(state="normal")
        if (self._analysis_request[0] != self.manifest_path
                or self._analysis_request[3] != self._manifest_sha256):
            self.measurement_panel.notice.set("上一次分析完成；当前数据已切换，请分析当前照片。")
            self.root.after(80, self._poll_worker)
            return
        if kind == "error":
            log_event(_LOGGER, "gui_analysis_failed", error=str(payload))
            # A failed rerun must not leave the previous decisive report on
            # screen.  The manifest remains available for diagnostics, but
            # all pipe states are rebuilt as UNKNOWN before showing the error.
            self._clear_loaded_state(
                manifest=self.manifest,
                manifest_path=self.manifest_path,
                reason_code="ANALYSIS_FAILED",
                message=f"识别失败：{payload}",
                preserve_manifest=True,
            )
            self.banner.configure(bg="#F8D7DA", fg="#721C24")
            self.banner_text.set(f"识别失败：{payload}")
            self.messagebox.showerror("识别失败", str(payload))
        else:
            try:
                dashboard = build_dashboard_model(
                    self.manifest,
                    payload,
                    manifest_sha256=self._manifest_sha256,
                    model_actual_sha256=self._model_actual_sha256,
                    photo_actual_sha256=self._current_photo_hashes(),
                )
            except Exception as error:
                # Treat an invalid worker result exactly like an analysis
                # failure: discard any prior report instead of leaving stale
                # INSTALLED/NOT_INSTALLED colours visible.
                self._clear_loaded_state(
                    manifest=self.manifest,
                    manifest_path=self.manifest_path,
                    reason_code="REPORT_INVALID",
                    message=f"识别结果无效：{error}",
                    preserve_manifest=True,
                )
                self.messagebox.showerror("识别结果无效", str(error))
            else:
                self.report = payload
                self.dashboard = dashboard
                self.selected_pipe_id = None
                self._refresh_dashboard()
                diagnostic = summarize_stereo_diagnostics(self.manifest, payload)
                if diagnostic:
                    self.banner.configure(bg="#FFF3CD", fg="#5F4500")
                    self.banner_text.set(diagnostic)
                    self.measurement_panel.notice.set(diagnostic)
                else:
                    self.measurement_panel.notice.set("分析完成。查看逐管状态和间距/前后关系，可录入实测样本或导出报告。")
                log_event(
                    _LOGGER,
                    "gui_analysis_finished",
                    pipe_count=len(self.dashboard.get("pipes", [])),
                    counts=self.dashboard.get("counts"),
                    diagnostic=diagnostic,
                )
        self.root.after(80, self._poll_worker)

    def _open_demo(self) -> None:
        path = Path(__file__).resolve().parents[1] / "test_model" / "field_stereo_demo_manifest.json"
        self._load_sources(path, None)
        self.main_tabs.select(0)

    def _input_capture(self) -> None:
        from .capture_gui import CaptureInputDialog

        CaptureInputDialog(self)

    def _open_elevation(self, manifest_path: Path | None = None) -> None:
        from .elevation_gui import ElevationCaptureDialog

        try:
            current = getattr(self, "elevation_dialog", None)
            if current is not None and not current.closed:
                if current.busy:
                    current.window.lift()
                    return
                if manifest_path is not None:
                    current.load_session(manifest_path)
                current.window.lift()
            else:
                imported_dxf = self.dxf_elevation.source_path if manifest_path is None and self.dxf_elevation is not None else None
                self.elevation_dialog = ElevationCaptureDialog(self, manifest_path, model_path=imported_dxf)
        except (OSError, ValueError) as error:
            self.messagebox.showerror("基础立面评估", str(error))

    def _open_calibration_wizard(self) -> None:
        """Open the chessboard wizard; it needs only the app, not a manifest."""

        try:
            from .calibration_wizard import ChessboardWizardDialog

            ChessboardWizardDialog(self)
        except (ImportError, OSError, ValueError) as error:
            # ImportError keeps the toolbar usable when the wizard module is
            # unavailable; OSError/ValueError are its fail-closed reports.
            self.messagebox.showerror("标定向导", str(error))

    def _quick_capture(self) -> None:
        """Capture from the saved rig and analyse it in one step."""

        from .workbench_profile import capture_state_from_profile, load_profile

        profile, problem = load_profile()
        if problem or not profile:
            message = "尚未保存工作台配置；请先完成现场数据录入并保存工作台配置。"
            if problem:
                message += f"\n{problem}"
            self.messagebox.showinfo("一键抓拍", message)
            return
        state = capture_state_from_profile(profile)
        calibration = state["calibration_current"]
        missing = []
        if calibration is None:
            missing.append("尚未完成双目标定。")
        elif not state["calibration_path"] or not Path(state["calibration_path"]).is_file():
            missing.append("标定文件不存在，请重新保存工作台配置。")
        if not state["model_path"] or not Path(state["model_path"]).is_file():
            missing.append("CAD 模型不存在。")
        if not state["pipes"]:
            missing.append("管件目录为空。")
        if missing:
            # The regular input dialog is prefilled from the same profile, so
            # the operator can repair the missing piece and come back.
            self.messagebox.showinfo("一键抓拍", "请先补全现场配置：\n" + "\n".join(missing))
            self._input_capture()
            return
        try:
            from .calibration_wizard import rectifier_for_calibration

            rectifier = rectifier_for_calibration(calibration, profile)
        except ValueError as error:  # ChessboardCalibrationError is a ValueError
            self.messagebox.showerror("一键抓拍", str(error))
            return
        QuickCaptureDialog(self, profile, state, calibration, rectifier)


class QuickCaptureDialog:
    """Capture from the saved rig, build the dataset, and start the analysis.

    Everything the operator configured once lives in the workbench profile, so
    this dialog only opens the camera, takes one audited pair, and hands the
    resulting manifest back to the dashboard's normal analysis path.
    """

    def __init__(
        self,
        app: Any,
        profile: Mapping[str, Any],
        state: Mapping[str, Any],
        calibration: Mapping[str, Any],
        rectifier: Any | None,
    ) -> None:
        from .stereo_camera import LAYOUT_SEPARATE

        self.app = app
        self.profile = profile
        self.state = state
        self.calibration = calibration
        self.rectifier = rectifier
        self.camera = dict(state["camera"])
        self.separate_devices = self.camera["layout"] == LAYOUT_SEPARATE
        tk, ttk = app.tk, app.ttk
        self.window = tk.Toplevel(app.root)
        self.window.title("一键抓拍并分析")
        self.window.geometry("880x620")
        self.window.minsize(760, 540)
        self.window.transient(app.root)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.session: Any | None = None
        self.after_id: str | None = None
        self.open_after_id: str | None = None
        self._opening = False
        self._closing = False
        self.preview_images: list[Any] = []
        self.message = tk.StringVar(value=self._reminder())

        main = ttk.Frame(self.window, padding=12)
        main.pack(fill="both", expand=True)
        controls = ttk.Frame(main)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Button(controls, text="打开预览", command=self.start).pack(side="left", padx=3)
        ttk.Button(controls, text="停止", command=self.stop).pack(side="left", padx=3)
        ttk.Button(controls, text="抓拍并分析", command=self.capture).pack(side="left", padx=3)
        ttk.Button(controls, text="关闭", command=self.close).pack(side="left", padx=3)
        ttk.Label(main, textvariable=self.message, wraplength=830, foreground="#355371").pack(
            fill="x", pady=(0, 8)
        )
        previews = ttk.Frame(main)
        previews.pack(fill="both", expand=True)
        previews.columnconfigure(0, weight=1)
        previews.columnconfigure(1, weight=1)
        previews.rowconfigure(1, weight=1)
        ttk.Label(previews, text="左目").grid(row=0, column=0, pady=4)
        ttk.Label(previews, text="右目").grid(row=0, column=1, pady=4)
        self.left_preview = ttk.Label(previews, anchor="center")
        self.right_preview = ttk.Label(previews, anchor="center")
        self.left_preview.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        self.right_preview.grid(row=1, column=1, sticky="nsew", padx=(4, 0))
        ttk.Label(
            main,
            text=(
                "抓拍后自动建立现场数据集并运行双目识别；"
                "标定要求每目 {width}×{height}，时间差不得超过 {delta:g} ms。"
            ).format(
                width=self.calibration["left_camera"]["width"],
                height=self.calibration["left_camera"]["height"],
                delta=float(self.calibration["max_sync_delta_ms"]),
            ),
            foreground="#4A6178",
        ).pack(fill="x", pady=(10, 0))

    def _reminder(self) -> str:
        model_name = Path(str(self.state["model_path"])).name
        calibration_id = str(self.calibration.get("calibration_id", ""))
        summary = (
            f"模型 {model_name} · 管件 {len(self.state['pipes'])} 根 · 标定 {calibration_id}"
        )
        if self.rectifier is not None:
            return summary + " · 预览与抓拍前会按极线矫正配方处理原始相机帧。"
        return summary + " · 相机输出须为已完成极线矫正的图像。"

    def _photo(self, frame: Any) -> Any:
        import cv2

        height, width = frame.shape[:2]
        scale = min(400 / width, 440 / height, 1.0)
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
            raise ValueError("无法生成相机预览")
        return self.app.tk.PhotoImage(
            data=base64.b64encode(encoded).decode("ascii"), format="png"
        )

    def start(self) -> None:
        from .logging_config import get_logger, log_event
        from .stereo_camera import StereoCameraSession, run_in_background

        if self._opening or self.open_after_id is not None:
            self.message.set("正在处理上一个相机任务，请稍候…")
            return
        self.stop()
        self._opening = True
        self.message.set("正在打开相机；部分设备在 DirectShow 下需要数秒，请稍候…")
        layout = self.camera["layout"]
        left_index = self.camera["left_index"]
        right_index = None if self.separate_devices else self.camera["right_index"]
        log_event(
            get_logger("gui.quick_capture"),
            "quick_capture_camera_open_start",
            layout=layout,
            left_index=left_index,
            right_index=right_index,
        )
        outcome: dict[str, Any] = {}

        def _task() -> Any:
            session = StereoCameraSession(
                layout=layout,
                left_index=left_index,
                right_index=right_index,
                eye_width=int(self.calibration["left_camera"]["width"]),
                eye_height=int(self.calibration["left_camera"]["height"]),
            )
            session.open()
            return session

        def _on_result(result: Any, error: Exception | None) -> None:
            if error is not None:
                outcome["done"] = (None, error)
                return
            if self._closing:
                result.close()
                outcome["done"] = (None, ValueError("窗口已关闭"))
                return
            outcome["done"] = (result, None)

        run_in_background(_task, _on_result)
        self._poll_open(outcome)

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

        if error is not None:
            log_event(get_logger("gui.quick_capture"), "quick_capture_camera_open_failed", error=str(error))
            self.message.set(f"相机打开失败：{error}")
            self.app.messagebox.showerror("相机打开失败", str(error), parent=self.window)
            return
        self.session = session
        log_event(get_logger("gui.quick_capture"), "quick_capture_camera_open_finished")
        self.message.set("相机已打开，正在显示实时预览。")
        self._update_preview()

    def _update_preview(self) -> None:
        import numpy as np

        if self.session is None:
            return
        try:
            pair = self.session.read_pair()
            left = pair.left if self.rectifier is None else self.rectifier.rectify("left", pair.left)
            right = (
                pair.right
                if self.rectifier is None
                else self.rectifier.rectify("right", pair.right)
            )
            self.preview_images = [self._photo(left), self._photo(right)]
            self.left_preview.configure(image=self.preview_images[0])
            self.right_preview.configure(image=self.preview_images[1])
            if np.array_equal(pair.left, pair.right):
                self.message.set(
                    "相机仍在预热，或左右画面完全相同；请等待出现两个不同视角后再抓拍。"
                )
            else:
                source = "矫正后帧" if self.rectifier is not None else "相机输出帧"
                self.message.set(
                    f"实时预览（{source}）：左右主机时间差 {pair.sync_delta_ms:.3f} ms；"
                    "确认左右顺序正确后点击抓拍并分析。"
                )
            self.after_id = self.window.after(80, self._update_preview)
        except Exception as error:  # native cv2 errors included, never silent
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

    def _intake_directory(self) -> Path:
        """Create a fresh camera_intake folder, retrying an id collision."""

        from datetime import datetime
        from uuid import uuid4

        collision: FileExistsError | None = None
        for _attempt in range(3):
            capture_id = f"camera-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"
            directory = _WORKBENCH_ROOT / "camera_intake" / capture_id
            try:
                directory.mkdir(parents=True, exist_ok=False)
            except FileExistsError as error:
                collision = error
                continue
            return directory
        raise collision if collision is not None else OSError("无法创建抓拍目录")

    def capture(self) -> None:
        import cv2
        import numpy as np

        if self._opening:
            self.message.set("相机仍在后台打开；打开后请再点击“抓拍并分析”。")
            return
        if self.session is None:
            self.start()
            self.message.set("相机正在后台打开；打开后请点击“抓拍并分析”。")
            return
        try:
            pair = self.session.read_pair()
            if np.array_equal(pair.left, pair.right):
                raise ValueError(
                    "左右画面完全相同，可能仍在预热或当前不是双目输出；"
                    "请等待实时预览出现两个不同视角后重试。"
                )
            max_sync_delta_ms = float(self.calibration["max_sync_delta_ms"])
            if pair.sync_delta_ms > max_sync_delta_ms:
                raise ValueError(
                    f"本次左右抓拍时间差 {pair.sync_delta_ms:.3f} ms 超过标定允许的 "
                    f"{max_sync_delta_ms:g} ms"
                )
            frames = {"left": pair.left, "right": pair.right}
            for role in ("left", "right"):
                if self.rectifier is not None:
                    frame = self.rectifier.rectify(role, frames[role])
                    height = int(self.calibration["left_camera"]["height"])
                    width = int(self.calibration["left_camera"]["width"])
                    if frame.shape[:2] != (height, width):
                        raise ValueError(
                            f"{role} 目矫正后尺寸 {frame.shape[1]}×{frame.shape[0]} 与标定的 "
                            f"{width}×{height} 不一致，不能继续分析。"
                        )
                    frames[role] = frame
            directory = self._intake_directory()
            paths = {role: directory / f"{role}.png" for role in ("left", "right")}
            for role, frame in frames.items():
                ok, payload = cv2.imencode(".png", frame)
                if not ok:
                    raise ValueError(f"无法编码{role}相机图像")
                paths[role].write_bytes(payload.tobytes())

            from .capture_gui import create_capture_dataset

            model_path = Path(str(self.state["model_path"]))
            previous_manifest = (
                Path(str(self.state["last_manifest_path"]))
                if self.state["capture_history"]
                and self.state["last_manifest_path"]
                and Path(str(self.state["last_manifest_path"])).is_file()
                else None
            )
            manifest_path = create_capture_dataset(
                output_root=_WORKBENCH_ROOT / "captures",
                model_path=model_path,
                pipes=self.state["pipes"],
                calibration=self.calibration,
                left_path=paths["left"],
                right_path=paths["right"],
                left_time=pair.left_captured_at,
                right_time=pair.right_captured_at,
                pair_confirmed=not self.separate_devices,
                previous_manifest=previous_manifest,
                stl_unit=self.state["stl_unit"] if model_path.suffix.lower() == ".stl" else None,
                timestamp_sources={
                    "left": pair.timestamp_source,
                    "right": pair.timestamp_source,
                },
                camera_capture_provenance=pair.provenance,
            )
        except Exception as error:
            # Keep the dialog (and its camera session) open so the operator can
            # retry; a dataset is mandatory before any analysis may run.
            # Native cv2 errors are included — they must surface, not vanish.
            self.app.messagebox.showerror("一键抓拍失败", str(error), parent=self.window)
            return
        self.stop()
        self.window.destroy()
        self.app._load_sources(manifest_path, None)
        self.app.main_tabs.select(0)
        self.app._run_analysis()


def _start_application(root, manifest_path=None, report_path=None):
    """Use the current field workflow by default; explicit old inputs retain their viewer."""
    from types import SimpleNamespace
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    current = manifest_path is None and report_path is None
    if manifest_path is not None:
        payload = _read_json_object(Path(manifest_path), 'manifest')
        current = (payload.get('analysis') or {}).get('mode') == 'elevation_auto'
    if not current:
        return _PipeTwinApplication(root, manifest_path, report_path)
    from .elevation_gui import ElevationCaptureDialog
    style = ttk.Style(root)
    if 'clam' in style.theme_names():
        style.theme_use('clam')
    style.configure('.', font=('Microsoft YaHei UI', 10))
    style.configure('Treeview', rowheight=25)
    app = SimpleNamespace(root=root, tk=tk, ttk=ttk, filedialog=filedialog, messagebox=messagebox)
    dialog = ElevationCaptureDialog(app, Path(manifest_path) if manifest_path else None, window=root)
    if report_path is not None:
        dialog.message.set('已载入现场；请重新评估当前照片以生成匹配的求解状态。历史报告文件保留。')
    return dialog


def launch_gui(
    manifest_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> None:
    """Launch the field workbench directly; explicit legacy inputs use their viewer.

    Closing the primary window returns control to the office client, which owns
    the remote services. Scene changes invalidate results until reevaluation.
    """

    log_event(_LOGGER, "gui_launch", manifest=manifest_path, report=report_path)
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError(
            "The pipe dashboard requires an interactive desktop session with Tk 8.6"
        ) from error
    try:
        application = _start_application(root, manifest_path, report_path)
        root._pipe_twin_application = application
    except Exception:
        root.destroy()
        raise
    root.mainloop()


__all__ = [
    "INSTALLATION_STATES",
    "STATE_PRESENTATION",
    "build_dashboard_model",
    "launch_gui",
]
