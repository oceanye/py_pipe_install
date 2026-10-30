"""Dependency-free 3-D STL/DXF layout viewer for elevation registration."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Callable
import numpy as np

from .camera_view import CAMERA_SIDE_PRESETS, camera_side_label, preview_rotation
from .relative_scene import STATE_COLORS, STATE_LABELS, model_states, scene_sections, draw_scene
from .assist_display import assist_preview_states, hypotheses, hypothesis_assignments


def model_section_basis(lines: np.ndarray, axis: np.ndarray | None = None) -> np.ndarray:
    """Stable model right/up/towards-viewer basis for the pipe cross-section."""
    if axis is None:
        directions = lines[:, 1] - lines[:, 0]
        directions = directions / np.linalg.norm(directions, axis=1)[:, None]
        _, _, vectors = np.linalg.svd(directions, full_matrices=True)
        axis = vectors[0].copy()
        if axis[int(np.argmax(np.abs(axis)))] < 0:
            axis = -axis
    return preview_rotation(np.asarray(axis, float).tolist())


class ElevationModelViewer:
    """Orbitable STL/DXF layout preview; axis picks remain model coordinates."""
    def __init__(self, owner: Any, *, model_path: Path, pipes: list[dict[str, Any]], axis_world: list[float] | None = None, on_axis: Callable[[list[float]], None] | None = None, report: dict[str, Any] | None = None, camera_side_world: list[float] | None = None, on_camera_side: Callable | None = None) -> None:
        self.owner, self.pipes, self.on_axis = owner, pipes, on_axis
        self.on_camera_side = on_camera_side
        self.camera_side_world = camera_side_world
        tk, ttk = owner.app.tk, owner.app.ttk
        self.window = tk.Toplevel(owner.window); self.window.title("现场相对模型 · CAD 管位与方向联动"); self.window.geometry("1320x820")
        self.window.minsize(1000, 660)
        panes = ttk.Panedwindow(self.window, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=8, pady=8)
        model_panel = ttk.LabelFrame(panes, text="现场建模与观察方向")
        section_panel = ttk.LabelFrame(panes, text="模型管道编号 · 沿共同长度轴观察的截面")
        panes.add(model_panel, weight=1); panes.add(section_panel, weight=1)
        self.scene_tabs = ttk.Notebook(model_panel)
        self.scene_tabs.pack(fill='both', expand=True)
        scene_panel, direction_panel = ttk.Frame(self.scene_tabs), ttk.Frame(self.scene_tabs)
        self.scene_tabs.add(scene_panel, text='现场相对断面')
        self.scene_tabs.add(direction_panel, text='三维方向示意')
        self.scene_zone = tk.StringVar()
        self.scene_zone_box = ttk.Combobox(scene_panel, textvariable=self.scene_zone, state='readonly')
        self.scene_zone_box.pack(fill='x', padx=5, pady=3)
        self.scene_zone_box.bind('<<ComboboxSelected>>', lambda _e: self.draw())
        self.scene_canvas = tk.Canvas(scene_panel, width=580, height=540, bg='#F2F5F8', highlightthickness=0)
        self.scene_canvas.pack(fill='both', expand=True)
        self.scene_hits = []
        self.scene_canvas.bind('<Button-1>', self.pick_scene_pipe)
        self.canvas = tk.Canvas(direction_panel, width=580, height=540, bg="#101923", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.section_canvas = tk.Canvas(section_panel, width=580, height=540, bg="#F2F5F8", highlightthickness=0)
        self.section_canvas.pack(fill="both", expand=True)
        self.selection_info = tk.StringVar()
        ttk.Label(self.window, textvariable=self.selection_info, foreground="#355371").pack(fill="x", padx=8)
        controls = ttk.Frame(self.window); controls.pack(fill="x", padx=8)
        ttk.Label(controls, text="管长方向").pack(side="left")
        self.direction_mode = tk.StringVar(value="自动")
        direction_values = ("自动", "Z", "自定义") if getattr(owner, "model_kind", "") == "dxf" else ("自动", "X", "Y", "Z", "自定义")
        ttk.Combobox(controls, textvariable=self.direction_mode, values=direction_values, state="readonly", width=10).pack(side="left", padx=5)
        self.custom = tk.StringVar(value="0,0,1"); ttk.Entry(controls, textvariable=self.custom, width=16).pack(side="left")
        ttk.Button(controls, text="应用方向", command=self.apply_direction).pack(side="left", padx=5)
        ttk.Button(controls, text="反向", command=self.reverse_axis).pack(side="left")
        view_controls = ttk.Frame(self.window); view_controls.pack(fill="x", padx=8, pady=6)
        ttk.Label(view_controls, text="相机观察侧").pack(side="left")
        self.view_mode = tk.StringVar(value=camera_side_label(camera_side_world))
        self.view_choices = dict(CAMERA_SIDE_PRESETS)
        self.view_choices[self.view_mode.get()] = camera_side_world
        self.view_box = ttk.Combobox(view_controls, textvariable=self.view_mode,
            values=tuple(self.view_choices), state="readonly", width=26)
        self.view_box.pack(side="left", padx=5)
        self.view_box.bind("<<ComboboxSelected>>", lambda _e: self.preview_side())
        ttk.Button(view_controls, text="应用相机观察侧", command=self.apply_camera_side).pack(side="left", padx=5)
        ttk.Button(view_controls, text="恢复已应用方向", command=self.restore_camera_side).pack(side="left", padx=5)
        self.info = tk.StringVar(value="右键拖动只改变预览，滚轮缩放。X/Y/Z 为模型坐标；选择观察侧后点击应用保存。")
        ttk.Label(self.window, textvariable=self.info, wraplength=830).pack(fill="x", padx=8, pady=(3, 8))
        self.zoom, self.yaw, self.pitch, self.drag_start = 1.0, 0.0, 0.0, None
        self.base_rotation = np.eye(3)
        self.projected_lines = np.empty((0, 2, 2))
        self.section_hits: list[tuple[str, float, float, float]] = []
        self._snapshot_owner()
        self._load_geometry(pipes, axis_world, report)
        if self.axis is not None:
            self.direction_mode.set("自定义")
            self.custom.set(",".join(f"{v:g}" for v in self.axis))
        self.canvas.bind("<ButtonPress-3>", self.begin_orbit); self.canvas.bind("<B3-Motion>", self.orbit); self.canvas.bind("<MouseWheel>", self.wheel)
        self.canvas.bind("<Button-1>", self.pick_model_pipe)
        self.section_canvas.bind("<Button-1>", self.pick_section_pipe)
        for canvas in (self.canvas, self.section_canvas, self.scene_canvas):
            canvas.bind("<Configure>", lambda _e: self.draw())
        self.window.protocol("WM_DELETE_WINDOW", self.close); self.preview_side()
        if not any(s['items'] for s in scene_sections(self.report)):
            self.scene_tabs.select(1)

    def _snapshot_owner(self) -> None:
        self.snapshot_hashes = dict(getattr(self.owner, "image_hashes", {}))
        self.snapshot_generation = int(getattr(self.owner, "generation", 0))
        self.snapshot_model = self.owner.fields["model"].get()

    def _load_geometry(self, pipes, axis_world, report) -> None:
        self.pipes = pipes
        self.axis = np.asarray(axis_world, dtype=float) if axis_world is not None else None
        self.source_lines = np.asarray([p["centerline_world_mm"] for p in pipes], dtype=float) if pipes else np.empty((0, 2, 3))
        self.section_basis = model_section_basis(self.source_lines, self.axis) if pipes else np.eye(3)
        self.report = report or {}; self.display_lines = self.source_lines.copy()
        self.display_axis = self.section_basis[2].copy(); self.registration_rotation = None
        cloud = np.asarray(((self.report.get("local_surface") or {}).get("point_cloud") or {}).get("points_camera_mm") or [], dtype=float)
        self.display_cloud = cloud if cloud.ndim == 2 and cloud.shape[1:] == (3,) else np.empty((0, 3))
        reg = self.report.get("registration") if isinstance(self.report, dict) else None
        if isinstance(reg, dict) and reg.get("status") == "MATCHED" and reg.get("rotation_model_to_camera") is not None:
            rotation = np.asarray(reg["rotation_model_to_camera"], dtype=float); translation = np.asarray(reg.get("translation_model_to_camera_mm", [0, 0, 0]), dtype=float)
            if rotation.shape == (3, 3) and translation.shape == (3,):
                self.registration_rotation = rotation
                self.display_lines = self.source_lines @ rotation.T + translation
                self.display_axis = rotation @ self.display_axis
        elif len(self.display_cloud):
            # Without a solved pose, model and camera points have different
            # frames.  Keep the local cloud out of this model canvas.
            self.display_cloud = np.empty((0, 3))
    def sync_from_owner(self) -> None:
        """Refresh after an explicit axis/side edit; discard obsolete pose overlays."""
        settings = self.owner.registration_settings
        self._load_geometry(self.owner.pipes, settings.get("axis_world"), self.owner.report)
        self.direction_mode.set("自定义" if self.axis is not None else "自动")
        if self.axis is not None:
            self.custom.set(",".join(f"{v:g}" for v in self.axis))
        self.camera_side_world = settings.get("camera_side_world")
        label = camera_side_label(self.camera_side_world)
        self.view_choices = dict(CAMERA_SIDE_PRESETS)
        self.view_choices[label] = self.camera_side_world
        self.view_box.configure(values=tuple(self.view_choices))
        self.view_mode.set(label)
        self._snapshot_owner()
        self.preview_side()
        self.info.set("已同步主窗口方向；点击左右任一管道，与主列表同步选择。")

    def focus_section(self) -> None:
        self.section_canvas.focus_set()
        self.info.set("左右视图和主列表共享管道选择；截面固定沿管长轴。" + self.camera_side_status())

    def has_pending_camera_side(self) -> bool:
        selected = self.view_choices[self.view_mode.get()]
        applied = self.owner.registration_settings.get("camera_side_world")
        same = ((selected is None and applied is None) or
                (selected is not None and applied is not None
                 and np.allclose(selected, applied, atol=1e-9, rtol=0)))
        return not same

    def restore_camera_side(self) -> None:
        if getattr(self.owner, 'busy', False):
            return
        self.sync_from_owner()
        self.info.set('已恢复当前匹配方向；预览选择已撤回。' + self.camera_side_status())

    def camera_side_status(self) -> str:
        """Distinguish the preview arrow from the constraint sent to matching."""
        if not self._is_current():
            return "现场输入已改变，请重新打开方向预览。"
        if self.has_pending_camera_side():
            return f"仅预览：{self.view_mode.get()}；尚未用于匹配，请点击“应用相机观察侧”。"
        applied = self.owner.registration_settings.get("camera_side_world")
        if applied is None:
            return "模型匹配观察侧：未指定（自动判断）。"
        return f"模型匹配观察侧已应用：{self.view_mode.get()}。"

    def _rotation(self) -> np.ndarray:
        cy, sy, cp, sp = math.cos(self.yaw), math.sin(self.yaw), math.cos(self.pitch), math.sin(self.pitch)
        return np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], float) @ np.asarray([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], float) @ self.base_rotation

    def preview_side(self) -> None:
        side = self.view_choices[self.view_mode.get()]
        self.yaw = self.pitch = 0.
        self.base_rotation = np.eye(3)
        if side is not None:
            self.base_rotation = preview_rotation(side)
            if self.registration_rotation is not None:
                self.base_rotation = self.base_rotation @ self.registration_rotation.T
        self.draw()
        self.info.set(self.camera_side_status())

    def apply_camera_side(self) -> None:
        if not self._is_current() or getattr(self.owner, "busy", False):
            self.info.set("现场输入已改变或正在分析，请重新打开方向预览。")
            return
        if self.on_camera_side:
            self.on_camera_side(self.view_choices[self.view_mode.get()])
        self.info.set(self.camera_side_status() + "编号截面箭头与主窗口已同步；点击⑤保存并评估更新匹配。")

    def _frame(self) -> tuple[np.ndarray, float]:
        groups = [self.display_lines.reshape(-1, 3)]
        if len(self.display_cloud): groups.append(self.display_cloud[::max(1, len(self.display_cloud)//1200)])
        all_points = np.vstack([g for g in groups if len(g)]) if any(len(g) for g in groups) else np.zeros((1, 3)); rotated = all_points @ self._rotation().T; center = rotated.mean(axis=0); extent = max(float(np.ptp(rotated[:, :2], axis=0).max()), 1.0); w, h = max(self.canvas.winfo_width(), 100), max(self.canvas.winfo_height(), 100)
        return center, min(w, h) * .78 / extent * self.zoom

    def _project(self, points: np.ndarray) -> np.ndarray:
        if not len(points): return np.empty((0, 2))
        center, scale = self._frame(); q = points @ self._rotation().T - center; w, h = max(self.canvas.winfo_width(), 100), max(self.canvas.winfo_height(), 100); return np.column_stack((w/2 + q[:, 0]*scale, h/2 - q[:, 1]*scale))

    def draw(self) -> None:
        # A newly completed assessment must update an already open viewer.
        if getattr(self.owner, 'report', self.report) is not self.report and self._is_current():
            self._load_geometry(self.pipes, self.axis, self.owner.report)
            if any(s['items'] for s in scene_sections(self.report)):
                self.scene_tabs.select(0)
        self.installation_states = model_states(self.report) if self._is_current() else {}
        if self._is_current():
            self.installation_states = assist_preview_states(self.report, self.installation_states,
                getattr(self.owner, 'assist_hypothesis_index', 0))
        self.canvas.delete("all"); lines = self._project(self.display_lines.reshape(-1, 3)).reshape((-1, 2, 2)) if len(self.display_lines) else np.empty((0, 2, 2))
        self.projected_lines = lines
        selected = self.owner.selected_id()
        for index in sorted(range(len(lines)), key=lambda i: self.pipes[i]["pipe_id"] == selected):
            line, pipe = lines[index], self.pipes[index]
            color = STATE_COLORS[self.installation_states.get(pipe['pipe_id'], 'UNKNOWN')]
            active = pipe["pipe_id"] == selected
            tags = (f"pipe:{pipe['pipe_id']}", "selected" if active else "pipe")
            if active:
                self.canvas.create_line(*line[0], *line[1], fill="#FFD166", width=11, tags=tags)
            self.canvas.create_line(*line[0], *line[1], fill=color, width=5, tags=tags)
            self.canvas.create_text(line[0][0]+4, line[0][1]-5, text=pipe["pipe_id"], fill="#FFD166" if active else "#DCE9F5", anchor="sw", tags=tags)
        if len(self.display_cloud):
            for x, y in self._project(self.display_cloud[::max(1, len(self.display_cloud)//1200)]): self.canvas.create_oval(x, y, x+1, y+1, fill="#66C2FF", outline="")
        if self.display_axis is not None and len(self.display_lines):
            centre = self.display_lines.reshape(-1, 3).mean(axis=0); axis = np.asarray(self.display_axis, float); axis /= max(np.linalg.norm(axis), 1e-9); length = max(float(np.ptp(self.display_lines.reshape(-1, 3), axis=0).max()), 30)*.35; projected = self._project(np.vstack((centre-axis*length, centre+axis*length))); self.canvas.create_line(*projected[0], *projected[1], fill="#FFD166", width=3, arrow="last")
        # A fixed-size model-coordinate triad remains readable after zooming
        # or switching to the matched camera-coordinate overlay.
        rotation = self._rotation()
        if self.registration_rotation is not None:
            rotation = rotation @ self.registration_rotation
        origin = np.array([75., 140.])
        for index, (name, color) in enumerate(zip("XYZ", ("#FF7777", "#77DD99", "#79BFFF"))):
            vector = rotation[:, index]
            end = origin + 48 * vector[:2] * [1, -1]
            self.canvas.create_line(*origin, *end, fill=color, width=2, arrow="last")
            label = f"+{name}" + (" 朝向你" if vector[2] > .99 else " 背向你" if vector[2] < -.99 else "")
            if np.linalg.norm(end-origin) < 8:
                end += [12, 18]
            self.canvas.create_text(*end, text=label, fill=color, anchor="sw")
        self.canvas.create_text(12, 12, text="模型坐标 · 预览可旋转（不改变已保存观察侧）", fill="#DCE9F5", anchor="nw")
        self.canvas.create_text(12, 32, text=f"待应用：{self.view_mode.get()}", fill="#FFD166", anchor="nw")
        self.draw_section(selected)
        scenes = scene_sections(self.report) if self._is_current() else []
        self.scene_zone_box.configure(values=[s['zone_id'] for s in scenes])
        scene = next((s for s in scenes if s['zone_id'] == self.scene_zone.get()), scenes[0] if scenes else None)
        if scene:
            self.scene_zone.set(scene['zone_id'])
        mapping = hypothesis_assignments(self.report, getattr(self.owner, 'assist_hypothesis_index', 0)) if self._is_current() else {}
        self.scene_hits = draw_scene(self.scene_canvas, scene, selected_pipe=selected, advisory_assignments=mapping)
        spec = next((p for p in self.pipes if p["pipe_id"] == selected), None)
        self.selection_info.set(f"当前选择：{selected} · 模型外径 {spec['nominal_diameter_mm']:g} mm · {STATE_LABELS[self.installation_states.get(selected, 'UNKNOWN')]}（点击编号只作选择，不创建照片身份）" if spec else "点击任一视图中的管道，与主列表同步选择。")
        if not self._is_current():
            self.info.set("现场输入已改变，请重新打开联动视图；旧图不能用于选择或应用设置。")

    def draw_section(self, selected: str | None) -> None:
        canvas = self.section_canvas
        canvas.delete("all"); self.section_hits = []
        if not self.pipes:
            return
        w, h = max(canvas.winfo_width(), 100), max(canvas.winfo_height(), 100)
        centers = (self.source_lines.mean(axis=1) @ self.section_basis.T)[:, :2]
        radii = np.asarray([p["nominal_diameter_mm"] / 2 for p in self.pipes])
        low, high = (centers-radii[:, None]).min(axis=0), (centers+radii[:, None]).max(axis=0)
        scale = max(.01, min(max(w-150, 20)/max(high[0]-low[0], 1), max(h-220, 20)/max(high[1]-low[1], 1)))
        center = np.array([w/2, h/2+15])
        positions = center + (centers-(low+high)/2) * scale * [1, -1]
        for index in sorted(range(len(self.pipes)), key=lambda i: self.pipes[i]["pipe_id"] == selected):
            pipe, (x, y) = self.pipes[index], positions[index]
            radius = max(3., radii[index]*scale)
            active = pipe["pipe_id"] == selected
            tags = (f"pipe:{pipe['pipe_id']}", "selected" if active else "pipe")
            state = self.installation_states.get(pipe['pipe_id'], 'UNKNOWN')
            canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill=STATE_COLORS[state], outline="#D47D00" if active else "#273B51", width=4 if active else 1, tags=(*tags, f'state:{state}'))
            canvas.create_text(x, y-radius-5, text=pipe["pipe_id"], anchor="s", fill="#A45C00" if active else "#273B51", tags=tags)
            self.section_hits.append((pipe["pipe_id"], float(x), float(y), radius))
        canvas.create_text(12, 12, anchor="nw", text=f"固定截面 · 从管长轴正侧看，轴={np.round(self.section_basis[2], 3).tolist()}", fill="#273B51")
        canvas.create_text(12, 34, anchor="nw", text=self.camera_side_status(), width=max(w-24, 50),
                           fill="#355371", tags="camera_side_status")
        origin = np.array([48., 110.])
        for index, (name, color) in enumerate(zip("XYZ", ("#AA3333", "#237541", "#2464A0"))):
            vector = self.section_basis[:, index]
            end = origin+30*vector[:2]*[1, -1]
            if np.linalg.norm(end-origin) > 4:
                canvas.create_line(*origin, *end, fill=color, width=2, arrow="last")
                canvas.create_text(*end, text=f"+{name}", fill=color, anchor="sw")
            else:
                canvas.create_text(*(origin+[2, 16]), text=f"+{name} " + ("朝向你" if vector[2] > 0 else "背向你"), fill=color, anchor="nw")
        side = self.view_choices[self.view_mode.get()]
        if side is not None:
            projected = self.section_basis @ side
            transverse = float(np.linalg.norm(projected[:2]))
            if transverse > 1.e-6:
                direction = projected[:2] / transverse * [1, -1]
                reach = min((w/2-34)/max(abs(direction[0]), .001), (h/2-90)/max(abs(direction[1]), .001))
                start = center + direction * max(30, reach)
                end = start-direction*45
                canvas.create_line(*start, *end, fill="#995200", width=3, arrow="last", tags="camera_direction")
                canvas.create_text(*start, text="相机", anchor="s", fill="#995200", tags="camera_direction")
                caption = "箭头：相机朝截面的视线投影（允许斜视）"
            else:
                caption = "相机沿管长轴朝纸内看（与截面同侧）" if projected[2] > 0 else "相机沿管长轴朝纸外看（从截面背侧）"
            canvas.create_text(12, h-48, anchor="nw", text=caption, fill="#995200", tags="camera_direction")
        else:
            canvas.create_text(12, h-48, anchor="nw", text="相机侧未指定；求解前不猜测观察方向。", fill="#355371")
        choices = hypotheses(self.report) if self._is_current() else []
        legend = (f"辅助方案 {getattr(self.owner, 'assist_hypothesis_index', 0)+1}/{len(choices)}：黄为候选，待确认；灰为未确认"
                  if choices else "绿：相对就位  蓝：已识别安装  橙：不符  灰：待确认")
        canvas.create_text(12, h-25, anchor="nw", text=legend, fill="#355371", tags='installation-legend')

    def pick_scene_pipe(self, event) -> None:
        if not self._is_current():
            return
        hits = [r for r, x, y, radius in self.scene_hits if (event.x-x)**2+(event.y-y)**2 <= radius**2]
        if not hits:
            return
        row = hits[0]
        if row['pipe_id']:
            self.owner.select_model_pipe(row['pipe_id'])
        elif row.get('assist_pipe_id'):
            self.owner.select_model_pipe(row['assist_pipe_id'])
        text = f"{row['label']}：估计外径 {row['diameter_mm']:.2f} mm，来源 {row['diameter_source']}。"
        text += (f"对应 {row['pipe_id']}。" if row['pipe_id'] else
                 f"辅助候选 {row['assist_pipe_id']}，待确认；不是尺寸验收。" if row.get('assist_pipe_id') else
                 '尚未唯一对应 CAD 管号，不点亮确定安装位置。')
        self.selection_info.set(text)

    def _pick(self, candidates: list[str]) -> None:
        if not self._is_current():
            self.info.set("现场输入已改变，请重新打开联动视图。")
            return
        if candidates:
            # Repeated clicks cycle through coincident projected pipes.
            current = self.owner.selected_id()
            target = candidates[(candidates.index(current)+1) % len(candidates)] if current in candidates else candidates[0]
            self.owner.select_model_pipe(target)

    def pick_model_pipe(self, event: Any) -> None:
        point = np.array([event.x, event.y], float)
        candidates = []
        for pipe, (start, end) in zip(self.pipes, self.projected_lines):
            delta = end-start
            t = np.clip(float((point-start) @ delta)/max(float(delta @ delta), 1.e-9), 0, 1)
            if np.linalg.norm(point-(start+t*delta)) <= 10:
                candidates.append(pipe["pipe_id"])
        candidates.extend(pid for pid in self._tagged_pipes_at(self.canvas, event) if pid not in candidates)
        self._pick(candidates)

    @staticmethod
    def _tagged_pipes_at(canvas, event) -> list[str]:
        return list(dict.fromkeys(tag[5:] for item in canvas.find_overlapping(event.x-1, event.y-1, event.x+1, event.y+1)
                                  for tag in canvas.gettags(item) if tag.startswith("pipe:")))

    def pick_section_pipe(self, event: Any) -> None:
        point = np.array([event.x, event.y], float)
        candidates = [pid for pid, x, y, radius in self.section_hits
                    if np.linalg.norm(point-[x, y]) <= radius+5
                    or (abs(event.x-x) <= 25 and y-radius-24 <= event.y <= y-radius)]
        candidates.extend(pid for pid in self._tagged_pipes_at(self.section_canvas, event) if pid not in candidates)
        self._pick(candidates)

    def apply_preset(self) -> bool:
        mode = self.direction_mode.get(); values = {"X": [1,0,0], "Y": [0,1,0], "Z": [0,0,1]}
        if mode == "自动":
            if not len(self.source_lines): return False
            lines = self.source_lines[:,1] - self.source_lines[:,0]; lines = lines/np.linalg.norm(lines, axis=1)[:,None]; lines[np.dot(lines, lines[0]) < 0] *= -1; vector = np.mean(lines, axis=0)
        elif mode in values: vector = np.asarray(values[mode], float)
        elif mode == "自定义":
            try: vector = np.asarray([float(v.strip()) for v in self.custom.get().split(",")], float)
            except ValueError: self.info.set("自定义方向格式应为 x,y,z"); return False
        else: vector = self.axis
        if vector is None or np.asarray(vector).shape != (3,) or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) <= 1e-9: self.info.set("方向必须是有限的非零三维向量"); return False
        self.axis = np.asarray(vector, float)/np.linalg.norm(vector)
        self.section_basis = model_section_basis(self.source_lines, self.axis)
        self.display_axis = self.registration_rotation @ self.axis if self.registration_rotation is not None else self.axis
        self.info.set(f"已选择模型管长方向 {np.round(self.axis, 4).tolist()}；点击应用方向保存。")
        self.draw()
        return True

    def apply_direction(self) -> None:
        if not self._is_current() or getattr(self.owner, "busy", False): self.info.set("现场输入已改变或正在分析，请重新打开模型预览。"); return
        if not self.apply_preset(): return
        if self.axis is not None and self.on_axis: self.on_axis((np.asarray(self.axis, float)/np.linalg.norm(self.axis)).tolist())
    def apply(self) -> None:
        self.apply_direction()
    def reverse_axis(self) -> None:
        if self.axis is None:
            self.info.set("请先选择方向")
            return
        self.axis = -np.asarray(self.axis, float)
        self.direction_mode.set("自定义")
        self.custom.set(",".join(f"{v:g}" for v in self.axis))
        self.section_basis = model_section_basis(self.source_lines, self.axis)
        self.display_axis = self.registration_rotation @ self.axis if self.registration_rotation is not None else self.axis
        self.info.set(f"已反向模型管长方向 {np.round(self.axis, 4).tolist()}；点击应用方向保存。")
        self.draw()
    def _is_current(self) -> bool:
        fields = getattr(self.owner, "fields", {})
        model = fields.get("model").get() if fields.get("model") is not None else ""
        return (not getattr(self.owner, "closed", False) and int(getattr(self.owner, "generation", 0)) == self.snapshot_generation and model == self.snapshot_model and dict(getattr(self.owner, "image_hashes", {})) == self.snapshot_hashes)
    def begin_orbit(self, event: Any) -> None: self.drag_start = (event.x, event.y, self.yaw, self.pitch)
    def orbit(self, event: Any) -> None:
        if self.drag_start is None: return
        x, y, yaw, pitch = self.drag_start; self.yaw = yaw+(event.x-x)*.01; self.pitch = max(-1.5, min(1.5, pitch+(event.y-y)*.01)); self.draw()
    def wheel(self, event: Any) -> None: self.zoom = max(.35, min(6, self.zoom*(1.15 if event.delta > 0 else 1/1.15))); self.draw()
    def close(self) -> None:
        if self.window.winfo_exists(): self.window.destroy()


__all__ = ["ElevationModelViewer"]
