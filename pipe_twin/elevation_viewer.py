"""Dependency-free 3-D STL/DXF layout viewer for elevation registration."""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Callable
import numpy as np


class ElevationModelViewer:
    """Orbitable STL/DXF layout preview; axis picks remain model coordinates."""
    def __init__(self, owner: Any, *, model_path: Path, pipes: list[dict[str, Any]], axis_world: list[float] | None = None, on_axis: Callable[[list[float]], None] | None = None, report: dict[str, Any] | None = None) -> None:
        self.owner, self.pipes, self.on_axis = owner, pipes, on_axis
        tk, ttk = owner.app.tk, owner.app.ttk
        self.window = tk.Toplevel(owner.window); self.window.title("立面模型匹配 · 管长方向"); self.window.geometry("800x680")
        self.canvas = tk.Canvas(self.window, width=780, height=540, bg="#101923", highlightthickness=0); self.canvas.pack(fill="both", expand=True, padx=8, pady=8)
        controls = ttk.Frame(self.window); controls.pack(fill="x", padx=8)
        ttk.Label(controls, text="方向").pack(side="left")
        self.direction_mode = tk.StringVar(value="自动")
        direction_values = ("自动", "Z", "自定义") if getattr(owner, "model_kind", "") == "dxf" else ("自动", "X", "Y", "Z", "两点拾取", "自定义")
        ttk.Combobox(controls, textvariable=self.direction_mode, values=direction_values, state="readonly", width=10).pack(side="left", padx=5)
        self.custom = tk.StringVar(value="0,0,1"); ttk.Entry(controls, textvariable=self.custom, width=16).pack(side="left")
        ttk.Button(controls, text="应用方向", command=self.apply_direction).pack(side="left", padx=5)
        ttk.Button(controls, text="反向", command=self.reverse_axis).pack(side="left")
        observations = ((report or {}).get("local_surface") or {}).get("observations") or []
        obs_ids = [str(item.get("observation_id")) for item in observations if item.get("observation_id")]
        if obs_ids:
            self.observation_var = tk.StringVar(value=obs_ids[0]); self.pipe_var = tk.StringVar(value=str(pipes[0].get("pipe_id", "")))
            ttk.Label(controls, text="基准").pack(side="left", padx=(12, 2))
            ttk.Combobox(controls, textvariable=self.observation_var, values=obs_ids, state="readonly", width=12).pack(side="left")
            ttk.Label(controls, text="→").pack(side="left")
            ttk.Combobox(controls, textvariable=self.pipe_var, values=[str(p.get("pipe_id")) for p in pipes], state="readonly", width=12).pack(side="left")
            ttk.Button(controls, text="绑定", command=self.bind_anchor).pack(side="left", padx=3)
        ttk.Button(controls, text="清除基准对应", command=self.clear_anchors).pack(side="right")
        self.info = tk.StringVar(value=("DXF 圆形截面按模型 ±Z 作为管轴；相机俯视角度由双目配准估计。右键拖动旋转，滚轮缩放。" if getattr(owner, "model_kind", "") == "dxf" else "两点拾取：左键点击同一根管道的两个端点；右键拖动旋转，滚轮缩放。"))
        ttk.Label(self.window, textvariable=self.info).pack(fill="x", padx=8, pady=(3, 8))
        self.zoom, self.yaw, self.pitch, self.drag_start = 1.0, 0.0, 0.0, None
        self.snapshot_hashes = dict(getattr(owner, "image_hashes", {})); self.snapshot_generation = int(getattr(owner, "generation", 0)); self.snapshot_model = str(getattr(owner, "fields", {}).get("model").get()) if getattr(owner, "fields", {}).get("model") is not None else ""
        self.selected: list[tuple[int, int]] = []; self.axis = np.asarray(axis_world, dtype=float) if axis_world is not None else None
        self.source_lines = np.asarray([np.asarray(p["centerline_world_mm"], dtype=float) for p in pipes], dtype=float) if pipes else np.empty((0, 2, 3))
        self.report = report or {}; self.display_lines = self.source_lines.copy(); self.display_axis = self.axis; self.registration_rotation = None
        cloud = np.asarray(((self.report.get("local_surface") or {}).get("point_cloud") or {}).get("points_camera_mm") or [], dtype=float)
        self.display_cloud = cloud if cloud.ndim == 2 and cloud.shape[1:] == (3,) else np.empty((0, 3))
        reg = self.report.get("registration") if isinstance(self.report, dict) else None
        if isinstance(reg, dict) and reg.get("status") == "MATCHED" and reg.get("rotation_model_to_camera") is not None:
            rotation = np.asarray(reg["rotation_model_to_camera"], dtype=float); translation = np.asarray(reg.get("translation_model_to_camera_mm", [0, 0, 0]), dtype=float)
            if rotation.shape == (3, 3) and translation.shape == (3,):
                self.registration_rotation = rotation
                self.display_lines = self.source_lines @ rotation.T + translation
                if self.axis is not None: self.display_axis = rotation @ self.axis
        elif len(self.display_cloud):
            # Without a solved pose, model and camera points have different
            # frames.  Keep the local cloud out of this model canvas.
            self.display_cloud = np.empty((0, 3))
        self.canvas.bind("<Button-1>", self.pick); self.canvas.bind("<ButtonPress-3>", self.begin_orbit); self.canvas.bind("<B3-Motion>", self.orbit); self.canvas.bind("<MouseWheel>", self.wheel); self.canvas.bind("<Configure>", lambda _e: self.draw()); self.window.protocol("WM_DELETE_WINDOW", self.close); self.draw()

    def _rotation(self) -> np.ndarray:
        cy, sy, cp, sp = math.cos(self.yaw), math.sin(self.yaw), math.cos(self.pitch), math.sin(self.pitch)
        return np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], float) @ np.asarray([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], float)

    def _frame(self) -> tuple[np.ndarray, float]:
        groups = [self.display_lines.reshape(-1, 3)]
        if len(self.display_cloud): groups.append(self.display_cloud[::max(1, len(self.display_cloud)//1200)])
        all_points = np.vstack([g for g in groups if len(g)]) if any(len(g) for g in groups) else np.zeros((1, 3)); rotated = all_points @ self._rotation().T; center = rotated.mean(axis=0); extent = max(float(np.ptp(rotated[:, :2], axis=0).max()), 1.0); w, h = max(self.canvas.winfo_width(), 100), max(self.canvas.winfo_height(), 100)
        return center, min(w, h) * .78 / extent * self.zoom

    def _project(self, points: np.ndarray) -> np.ndarray:
        if not len(points): return np.empty((0, 2))
        center, scale = self._frame(); q = points @ self._rotation().T - center; w, h = max(self.canvas.winfo_width(), 100), max(self.canvas.winfo_height(), 100); return np.column_stack((w/2 + q[:, 0]*scale, h/2 - q[:, 1]*scale))

    def draw(self) -> None:
        self.canvas.delete("all"); lines = self._project(self.display_lines.reshape(-1, 3)).reshape((-1, 2, 2)) if len(self.display_lines) else np.empty((0, 2, 2))
        for index, line in enumerate(lines):
            pipe = self.pipes[index]; color = pipe.get("color_srgb", "#B0B0B0"); self.canvas.create_line(*line[0], *line[1], fill=color, width=5)
            for endpoint, point in enumerate(line):
                selected = (index, endpoint) in self.selected; self.canvas.create_oval(point[0]-5, point[1]-5, point[0]+5, point[1]+5, fill="#FFD166" if selected else color, outline="#FFFFFF")
            self.canvas.create_text(line[0][0]+4, line[0][1]-5, text=str(pipe.get("pipe_id", index+1)), fill="#DCE9F5", anchor="sw")
        if len(self.display_cloud):
            for x, y in self._project(self.display_cloud[::max(1, len(self.display_cloud)//1200)]): self.canvas.create_oval(x, y, x+1, y+1, fill="#66C2FF", outline="")
        if self.display_axis is not None and len(self.display_lines):
            centre = self.display_lines.reshape(-1, 3).mean(axis=0); axis = np.asarray(self.display_axis, float); axis /= max(np.linalg.norm(axis), 1e-9); length = max(float(np.ptp(self.display_lines.reshape(-1, 3), axis=0).max()), 30)*.35; projected = self._project(np.vstack((centre-axis*length, centre+axis*length))); self.canvas.create_line(*projected[0], *projected[1], fill="#FFD166", width=3, arrow="last")

    def _nearest_endpoint(self, event: Any) -> tuple[int, int] | None:
        if not len(self.display_lines): return None
        projected = self._project(self.display_lines.reshape(-1, 3)).reshape((-1, 2, 2)); distance = np.linalg.norm(projected - np.asarray([event.x, event.y]), axis=2); hit = np.unravel_index(int(np.argmin(distance)), distance.shape); return (int(hit[0]), int(hit[1])) if float(distance[hit]) <= 35 else None

    def pick(self, event: Any) -> None:
        if self.direction_mode.get() not in {"两点拾取", "自动"}: return
        hit = self._nearest_endpoint(event)
        if hit is None: return
        self.selected = [x for x in self.selected if x != hit] + [hit]; self.selected = self.selected[-2:]
        if len(self.selected) == 2 and self.selected[0][0] == self.selected[1][0] and self.selected[0][1] != self.selected[1][1]:
            vector = self.source_lines[self.selected[1][0], self.selected[1][1]] - self.source_lines[self.selected[0][0], self.selected[0][1]]; norm = float(np.linalg.norm(vector)); self.axis = vector/norm if norm > 1e-6 else self.axis; self.info.set(f"已从模型源坐标选择管长方向 {np.round(self.axis, 4).tolist()}；点击应用方向保存。")
        elif len(self.selected) == 2: self.info.set("请选择同一根管道的两个端点，避免把截面方向误存为管长方向。")
        self.draw()

    def apply_preset(self) -> None:
        mode = self.direction_mode.get(); values = {"X": [1,0,0], "Y": [0,1,0], "Z": [0,0,1]}
        if mode == "自动":
            if not len(self.source_lines): return
            lines = self.source_lines[:,1] - self.source_lines[:,0]; lines = lines/np.linalg.norm(lines, axis=1)[:,None]; lines[np.dot(lines, lines[0]) < 0] *= -1; vector = np.mean(lines, axis=0)
        elif mode in values: vector = np.asarray(values[mode], float)
        elif mode == "自定义":
            try: vector = np.asarray([float(v.strip()) for v in self.custom.get().split(",")], float)
            except ValueError: self.info.set("自定义方向格式应为 x,y,z"); return
        else: vector = self.axis
        if vector is None or np.asarray(vector).shape != (3,) or not np.all(np.isfinite(vector)) or np.linalg.norm(vector) <= 1e-9: self.info.set("方向必须是有限的非零三维向量"); return
        self.axis = np.asarray(vector, float)/np.linalg.norm(vector); self.display_axis = self.registration_rotation @ self.axis if self.registration_rotation is not None else self.axis; self.selected = []; self.info.set(f"已选择模型管长方向 {np.round(self.axis, 4).tolist()}；点击应用方向保存。"); self.draw()

    def apply_direction(self) -> None:
        if not self._is_current(): self.info.set("现场输入已改变，请关闭窗口后重新打开模型预览。"); return
        self.apply_preset()
        if self.axis is not None and self.on_axis: self.on_axis((np.asarray(self.axis, float)/np.linalg.norm(self.axis)).tolist())
        self.close()
    def apply(self) -> None:
        self.apply_direction()
    def reverse_axis(self) -> None:
        if self.axis is None:
            self.info.set("请先选择方向")
            return
        self.axis = -np.asarray(self.axis, float); self.display_axis = self.registration_rotation @ self.axis if self.registration_rotation is not None else self.axis; self.info.set(f"已反向模型管长方向 {np.round(self.axis, 4).tolist()}"); self.draw()
    def clear_anchors(self) -> None:
        if hasattr(self.owner, "registration_settings"):
            self.owner.registration_settings["anchors"] = {}
            if hasattr(self.owner, "invalidate"): self.owner.invalidate()
            self.snapshot_generation = int(getattr(self.owner, "generation", self.snapshot_generation))
            self.info.set("已清除基准对应；重新分析时使用自动身份匹配。")
    def bind_anchor(self) -> None:
        if not hasattr(self, "observation_var") or not hasattr(self.owner, "registration_settings") or not self._is_current():
            self.info.set("现场输入已改变，请重新打开模型预览。"); return
        observation, pipe = self.observation_var.get().strip(), self.pipe_var.get().strip()
        if not observation or not pipe: return
        anchors = dict(self.owner.registration_settings.get("anchors") or {})
        if pipe in anchors.values() and anchors.get(observation) != pipe:
            self.info.set("同一模型管道只能绑定一个观察；请先清除旧基准对应。"); return
        anchors[observation] = pipe; self.owner.registration_settings["anchors"] = anchors; self.owner.invalidate(); self.snapshot_hashes = dict(getattr(self.owner, "image_hashes", {})); self.snapshot_generation = int(getattr(self.owner, "generation", self.snapshot_generation)); self.info.set(f"已绑定 {observation} → {pipe}；该对应只对当前照片哈希有效。")

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
