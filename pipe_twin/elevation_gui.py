"""A small field workflow for persistent, rectified stereo image regions."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import queue
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .pipeline import atomic_write_text
from .logging_config import get_logger, log_event
from .elevation_zones import MAX_ZONES, normalize_zone_settings
from .model_zones import project_model_zone, propose_model_zones
from .camera_view import CAMERA_SIDE_PRESETS, camera_side_label, normalize_camera_side


ROOT = Path(__file__).resolve().parents[1] / "outputs" / "elevation_workbench"
STATE_COLORS = {"INSTALLED": "#228B55", "NOT_INSTALLED": "#D84A40", "UNKNOWN": "#CC941C"}
PALETTE = ("#E74C3C", "#3498DB", "#FFFFFF", "#2ECC71", "#F1C40F", "#9B59B6")
LOGGER = get_logger("elevation_gui")
REASON_TEXT = {
    "BURST_IMAGE_SIZE_CHANGED": "序列中图像尺寸改变",
    "BURST_EXPOSURE_UNSTABLE": "曝光变化过大，该帧已排除",
    "BURST_MOTION_UNVERIFIABLE": "可跟踪纹理不足，无法确认相机和场景静止",
    "BURST_SCENE_MOVED": "帧间位移检查未通过，可能有运动或噪声；请稳定画面后重试",
    "BURST_FRAME_QUALITY_FAILED": "该帧清晰度或曝光未通过",
    "BURST_DUPLICATE_FRAME": "重复画面，不计作新增样本",
    "BURST_FRAME_DEPTH_INVALID": "该帧没有可用双目深度",
    "BURST_REFERENCE_OR_SCENE_UNSTABLE": "末帧质量或场景稳定性不足，本序列未用于测量",
    "BURST_TOO_FEW_VALID_FRAMES": "有效且不同的帧不足 3 对",
    "BURST_NO_CONSISTENT_DEPTH": "没有获得足够的多帧一致深度",
    "CAMERA_SIDE_CONFLICT": "所选相机观察侧与双目几何冲突，请核对模型坐标方向",
    "OBSERVATIONS_EXCEED_PRESENT_PIPE_COUNT": "观测数量超过已知现场管数，请核对误检",
    "LOW_LIGHT": "图像偏暗，请增加照明或调整曝光",
    "DARK_REGION_DOMINANT": "目标区域大部分接近黑色",
    "LOW_TEXTURE": "目标纹理不足，双目匹配困难",
    "LOW_STEREO_COVERAGE": "有效深度覆盖不足",
    "ELEVATION_IDENTITY_AMBIGUOUS": "同色管区域重叠，无法唯一对应",
    "REFERENCE_DEPTH_REQUIRED_FOR_NEGATIVE": "缺少参考深度，无法区分空位与遮挡",
    "PAIR_UNHEALTHY": "同步、图像或深度质量不足",
    "INSUFFICIENT_COLOR_DEPTH_GEOMETRY": "颜色、深度或管径证据不足",
    "COLOR_DEPTH_SAME_COMPONENT": "管道颜色与深度重合",
    "LEFT_RIGHT_CONSISTENT": "左右目一致",
    "DIAMETER_WIDTH_MATCH": "管径匹配",
    "REFERENCE_DEPTH_KNOWN": "已有参考深度",
    "VALID_FREE_SPACE": "预期位置后方为空位",
    "NO_FOREGROUND_COLOR": "无前景遮挡证据",
    "ELEVATION_INSUFFICIENT_OR_OCCLUDED_EVIDENCE": "遮挡或有效证据不足",
    "ELEVATION_REPEATED_FREE_SPACE": "连续独立照片确认空位",
    "LOCAL_CYLINDER_MATCHED_TO_STL": "局部管道与模型几何匹配",
    "LOCAL_DEPTH_GEOMETRY_MATCHED_TO_MODEL": "仅双目深度局部几何与模型匹配",
    "REPEATED_REGISTERED_FREE_SPACE": "连续独立照片确认预期位置为空",
    "NO_LOCAL_SURFACE_OBSERVATIONS": "没有通过三维验证的管段",
    "INSUFFICIENT_VALID_DEPTH": "有效深度不足",
    "NO_DOMINANT_DISPARITY_SUPPORT": "管面双目匹配缺少一致深度",
    "DISCONTINUOUS_AXIAL_DEPTH_SUPPORT": "沿管长的有效深度不连续",
    "CYLINDER_FIT_RESIDUAL_TOO_LARGE": "圆柱拟合误差过大",
    "LOCAL_AXIS_SUPPORT_TOO_SHORT": "断面附近采样范围不足以稳定估计管轴方向",
    "PROJECTED_AXIS_DISAGREEMENT": "三维轴与图像方向不一致",
    "LOW_HORIZONTAL_TEXTURE": "沿视差方向纹理不足",
    "LOW_SURFACE_HORIZONTAL_TEXTURE": "整个候选管面多数区域沿视差方向纹理不足",
    "SMOOTH_AXIS_NEAR_EPIPOLAR": "光滑管面长方向接近视差方向，沿管长匹配容易出现多解",
    "CANDIDATE_WIDTH_CLIPPED": "候选横截面碰到图像或框选边界，请调整取景保留完整管宽",
    "LOW_PATCH_CORRELATION": "左右管面图像对应较差",
    "AMBIGUOUS_PATCH_CORRESPONDENCE": "多个位置图像相似，匹配不唯一",
    "COLLINEAR_WITHOUT_TWO_ANCHORS": "可见管道布局不足以唯一对应",
    "ALTERNATIVE_POSES_WITHIN_AMBIGUITY_GATE": "存在多个近似模型对应",
    "FOREGROUND_OCCLUSION": "前景遮挡",
    "CAPTURE_OR_DEPTH_UNHEALTHY": "同步或深度质量不足",
    "OUT_OF_VIEW_OR_INSUFFICIENT_SURFACE_EVIDENCE": "出视野或局部表面证据不足",
    "SEARCH_LIMIT": "模型匹配搜索范围不足",
    "LOCAL_SURFACE_SEARCH_TRUNCATED": "局部候选过多，请缩小视野或调整拍摄",
    "MULTIPLE_COMMON_AXIS_GROUPS": "存在多组不同管长方向，无法唯一匹配",
    "ANCHOR_AXIS_MISMATCH": "基准管长方向不一致",
    "ANCHOR_DIAMETER_OR_COLOR_MISMATCH": "基准管径或颜色不匹配",
    "NO_RIGID_MATCH_WITHIN_RESIDUAL_GATE": "未找到满足几何误差的模型对应",
    "SEARCH_SPACE_TRUNCATED": "模型匹配搜索范围不足",
}


def recognition_summary(report: dict) -> str:
    """Report proposal and metric stages separately, without counting pipes."""
    surface = report.get("local_surface") or {}
    candidates = surface.get("image_candidates",[])
    counts = Counter(c.get("role") for c in candidates)
    text = f"图像候选：左 {counts['left']} / 右 {counts['right']}；合格三维管段 {len(surface.get('observations',[]))}"
    reasons = Counter(r.get("reason") for r in surface.get("rejected",[]))
    if reasons:
        text += "；" + "、".join(f"{REASON_TEXT.get(k,k)} {v} 次" for k,v in reasons.most_common(2))
    from .section_display import pooling_summary
    pooled = pooling_summary(report)
    if pooled:
        text += '；'+pooled
    return text


def recognition_diagnostic_lines(report: dict | None) -> list[str]:
    """Separate image similarity, measured depth support and cylinder gates."""
    lines = ["目标是共同断面的管径、轴心间距和净距，不测量整根管长。",
             "断面附近保留小段采样带，用于估计管轴方向和验证圆弧；不是整根管道长度要求。",
             "图像候选不是实物管数；重叠分区会重复列出同一管面。",
             "图像唯一匹配比例来自有限抽样，仅作诊断，不代表测量合格率。",
             "高光 H 只统计多通道近饱和像素，不能覆盖全部反光。"]
    if not report:
        return lines + ["", "请先点击④保存并评估。"]
    lines += ["", recognition_summary(report)]
    temporal = report.get("temporal_fusion")
    if temporal:
        status = "已融合" if temporal.get("status") == "VALID" else "未通过"
        lines += ["", f"多帧融合：{status}，有效 {temporal.get('accepted_frames', 0)}/{temporal.get('requested_frames', 0)} 对",
                  "本序列只算一次评估；时间稳定性不代表绝对尺寸准确。"]
        for role, label in (("left", "左"), ("right", "右")):
            value = temporal.get(role) or {}
            lines.append(f"{label}目：每点至少 {value.get('required_support_frames', '—')} 帧支持；"
                         f"视差波动 MAD 中位数 {value.get('median_disparity_mad_px', '—')} px")
            if value.get("majority_cluster_recovered_pixels") is not None:
                lines.append(f"  时间检查重新找回 {value['majority_cluster_recovered_pixels']} 个像素；"
                             f"支持帧不足 {value.get('insufficient_observation_pixels', 0)} 个，"
                             f"匹配仍不稳定 {value.get('inconsistent_observation_pixels', 0)} 个"
                             "（左右复核前的全图统计，包含背景，不代表管面合格点数）。")
            if value.get("post_consistency_valid_fraction") is not None:
                lines.append(f"  全图有效深度：时间检查后 {100*value['pre_consistency_valid_fraction']:.1f}% → "
                             f"左右复核后 {100*value['post_consistency_valid_fraction']:.1f}%。")
        reasons = list(temporal.get("reason_codes", []))
        reasons.extend(code for frame in temporal.get("frames", []) for code in frame.get("reason_codes", []))
        if reasons:
            lines.append("序列提示：" + "；".join(REASON_TEXT.get(code, code) for code in dict.fromkeys(reasons)))
        frames = temporal.get("frames") or []
        if frames and all(all((frame.get("motion") or {}).get(role, {}).get("passed")
                              for role in ("left", "right")) for frame in frames):
            lines.append("本序列通过静止检查；管面深度不一致是匹配证据问题，不表示设备或管道发生移动。")
        else:
            lines.append("运动/曝光不稳定时保持机位、待画面稳定后重新连续采集；纹理不足时可能无法确认静止。")
    else:
        lines.append("本次使用单对图像，未进行多帧融合。")
    def percent(value):
        return "未计算" if value is None else f"{100*value:.1f}%"
    for candidate in (report.get("local_surface") or {}).get("image_candidates",[]):
        eye = "左目" if candidate.get("role") == "left" else "右目"
        lines += ["", f"{candidate.get('candidate_id')} · {eye} · 区域 {candidate.get('region_px')}",
                  f"用于拟合的有效深度：{percent(candidate.get('valid_depth_fraction'))}"]
        diagnostic = candidate.get("correspondence_diagnostic") or {}
        surface_texture = diagnostic.get("surface_texture") or {}
        if surface_texture:
            lines.append(f"全候选管面弱纹理比例：{percent(surface_texture.get('low_texture_fraction'))}"
                         f"（内部抽样 {surface_texture.get('sample_count',0)} 点，包含没有深度的区域）")
        angle = candidate.get('axis_to_epipolar_deg')
        if angle is not None:
            lines.append(f"管面长方向与视差方向夹角：{angle:.1f}°（0° 为平行）")
        lines.append(f"图像唯一匹配：{percent(diagnostic.get('sampled_unique_match_fraction'))}"
                     f"（抽样 {diagnostic.get('comparison_count',0)} 点）；高光 H：{percent(diagnostic.get('highlight_fraction'))}")
        warnings = diagnostic.get("warning_codes",[])
        if warnings:
            lines.append("图像提示：" + "；".join(REASON_TEXT.get(code,code) for code in warnings))
        for attempt in candidate.get("attempts",[]):
            label = "整段" if attempt["window_id"] == "FULL" else f"局部 {attempt['window_id']}"
            if attempt.get("fit_count"):
                lines.append(f"{label}：单眼拟合通过，仍需左右共同区间验证及模型匹配")
            for rejected in attempt.get("rejections",[]):
                text = f"{label}：{REASON_TEXT.get(rejected.get('reason'),rejected.get('reason'))}"
                trend = rejected.get("disparity_filter") or {}
                if trend.get("retained_fraction") is not None:
                    text += f"；主导深度支持 {percent(trend['retained_fraction'])}"
                fit = rejected.get("fit_diagnostics") or {}
                if fit.get("fit_rms_mm") is not None:
                    text += f"；拟合误差 {fit['fit_rms_mm']:.2f} mm / 上限 {fit['maximum_fit_rms_mm']:.2f} mm"
                if fit.get("minimum_axial_support_mm") is not None:
                    text += (f"；局部轴向采样 {fit['axial_support_mm']:.1f} mm / "
                             f"本方法需 {fit['minimum_axial_support_mm']:.1f} mm（非实物管长）")
                lines.append(text)
        if not candidate.get("attempts"):
            lines.append("旧报告未记录局部分段诊断；重新评估可生成。")
    lines += ["", "纹理不足或匹配多解：用少量可移除的哑光随机纹理做对照；高光变化不等于深度改善。",
              "光滑管道接近视差方向时，可调整整台双目相机的安装朝向再取样；仅旋转显示图像无效。",
              "取景应包含需要对比的多根管道，避开前景木条，并在图像和框选边界保留完整管宽。",
              "深度支持足够但拟合仍失败：继续检查双目矫正、深度误差及圆柱形状，不宜放宽尺寸容差。"]
    return lines


def relative_layout_summary_lines(report: dict | None) -> list[str]:
    lines = ["只比较共同断面：各管直径、轴心间距、表面净距及相对排列，不要求整根管长。",
             "相机到管道的距离只用于三维重建，不作为模型匹配目标。",
             "以下净距为共同管轴法向截面的轴心距减两半径；负值表示截面重叠。"]
    registration = (report or {}).get("registration") or {}
    registrations = ((report or {}).get("zone_registration_audit") or {}).get("registrations") or [registration]
    if registration.get("status") in {"ZONE_MATCH_CONFLICT","ZONE_RESULT_CONFLICT"}:
        registrations = []
    available = False
    for reg in registrations:
        layout = reg.get("relative_layout")
        if reg.get("status") != "MATCHED" or not layout:
            continue
        available = True
        if not layout.get("axis_centers_resolved",True):
            lines += ["", "本组使用局部条带估计，圆柱轴心未确认，暂不输出轴心距或表面净距对比。"]
            continue
        lines += ["", f"{reg.get('zone_id','全幅')} · 管径（模型 / 实测 / 偏差，mm）"]
        for item in layout.get("matches",[]):
            if item.get("observed_diameter_mm") is not None:
                lines.append(f"{item['pipe_id']}：{item['model_diameter_mm']:.2f} / {item['observed_diameter_mm']:.2f} / {item['diameter_error_mm']:+.2f}")
        lines.append("管间关系（模型 / 实测 / 偏差，mm）")
        for pair in layout.get("pairwise",[]):
            lines.append(f"{pair['pipe_id_a']} ↔ {pair['pipe_id_b']}  轴心距：{pair['model_distance_mm']:.2f} / {pair['observed_distance_mm']:.2f} / {pair['error_mm']:+.2f}")
            if pair.get("observed_surface_gap_mm") is not None:
                lines.append(f"    表面净距：{pair['model_surface_gap_mm']:.2f} / {pair['observed_surface_gap_mm']:.2f} / {pair['surface_gap_error_mm']:+.2f}")
    if not available:
        lines += ["", "尚无可靠的管间尺寸对比：需要先获得合格三维管段并确定模型对应。",
                  "图像候选框不等于实测管道，模型尺寸不会填入实测结果。"]
    if report:
        lines += ["", recognition_summary(report)]
        from .section_display import section_summary_lines, section_evaluations
        if section_evaluations(report):
            lines += ['', *section_summary_lines(report)]
    return lines


def measurement_summary_lines(spec: dict, result: dict | None) -> list[str]:
    """Human-readable reference points; never label a raw median as an axis."""
    measurement = (result or {}).get("measurement") or {}
    lines = [f"管道：{spec['pipe_id']}", f"模型/配置外径：{spec['nominal_diameter_mm']:.2f} mm",
             "距离原点：左目矫正相机光心；Z 深度与空间直线距离分别列出。",
             "截面位置：左右共同观测管段的中部，不代表整根管道端点。"]
    measured = measurement.get("measured_section") if measurement.get("status") == "MEASURED" else None
    if not measured:
        lines += ["当前实测外径：未获得", "中心线 / 最近表面：未获得可靠实测"]
    def metric(value):
        return f"{value / 1000:.4f} m（{value:.1f} mm）"
    for label, geometry in (("双目拟合", measured), ("模型配准预测（非实测）", measurement.get("model_predicted_section"))):
        if geometry is None:
            continue
        lines += ["", label, f"外径：{geometry['diameter_mm']:.2f} mm",
                  "中心线 Z 深度：" + metric(geometry["centerline"]["depth_z_mm"]),
                  "中心线空间距离：" + metric(geometry["centerline"]["range_mm"]),
                  "该截面表面最小 Z 深度：" + metric(geometry["minimum_surface_depth_z_mm"]),
                  "该截面最近表面空间距离：" + metric(geometry["nearest_surface_range_mm"])]
        tangents = geometry.get("silhouette_tangent_points") or []
        if tangents:
            lines.append("两侧投影轮廓切点 Z 深度：" + " / ".join(metric(p["depth_z_mm"]) for p in tangents))
    samples = measurement.get("surface_samples") or {}
    if samples:
        lines += ["", "原始可见表面匹配点（仅诊断，不是中心线或顶点）："]
        for role, data in samples.items():
            value = data.get("depth_z_median_mm")
            lines.append(f"{'左目' if role == 'left' else '右目'}深度中位数：" + (metric(value) if value is not None else "无"))
    lines += ["", "模型尺寸不会自动转成实测值；颜色仅辅助对应。"]
    return lines


def image_region(start: tuple[float, float], end: tuple[float, float],
                 transform: tuple[float, float, float], size: tuple[int, int]) -> list[int] | None:
    """Map a viewport drag to source pixels for a group assessment region."""
    scale, ox, oy = transform
    if not math.isfinite(scale) or scale <= 0:
        return None
    left, right = sorted(((start[0] - ox) / scale, (end[0] - ox) / scale))
    top, bottom = sorted(((start[1] - oy) / scale, (end[1] - oy) / scale))
    x0, y0 = max(0, math.floor(left)), max(0, math.floor(top))
    x1, y1 = min(size[0], math.ceil(right)), min(size[1], math.ceil(bottom))
    return [x0, y0, x1 - x0, y1 - y0] if x1 - x0 >= 8 and y1 - y0 >= 8 else None


class RegionCanvas:
    def __init__(self, parent: Any, owner: Any, role: str, *, assessment_regions: bool = False) -> None:
        self.owner, self.role = owner, role
        self.assessment_regions = assessment_regions
        self.canvas = owner.app.tk.Canvas(parent, bg="#152235", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.image: np.ndarray | None = None
        self.photo = None
        self.transform = (1.0, 0.0, 0.0)
        self.zoom = 1.0
        self.pan = [0.0, 0.0]
        self.pan_start = None
        self.region_start: tuple[float, float] | None = None
        from .axial_selection_gui import AxialBandController
        self.band_controller = AxialBandController(self) if role == 'left' and assessment_regions else None
        self.canvas.bind("<Configure>", lambda _e: self.draw())
        self.canvas.bind("<MouseWheel>", self.wheel)
        self.canvas.bind("<ButtonPress-3>", self.start_pan)
        self.canvas.bind("<B3-Motion>", self.move_pan)
        if role == "left" and assessment_regions:
            self.canvas.bind("<ButtonPress-1>", self.start_region)
            self.canvas.bind("<B1-Motion>", self.move_region)
            self.canvas.bind("<ButtonRelease-1>", self.finish_region)

    def set_image(self, image: np.ndarray | None) -> None:
        if self.assessment_regions and self.role == "left" and self.owner.region_selection_active:
            self.owner._set_region_selection(False)
        self.image = image
        if self.band_controller:
            self.owner._set_axial_selection(False)
            band = self.owner.registration_settings.get('aggregation_band')
            if image is not None and band and tuple(band['image_size']) != image.shape[1::-1]:
                self.owner.registration_settings.pop('aggregation_band', None)
                self.owner.invalidate()
                self.owner.message.set('照片尺寸改变，已清除汇聚范围；请重新指定管长方向。')
            ground = self.owner.registration_settings.get('ground_region')
            if image is not None and ground and tuple(ground['image_size']) != image.shape[1::-1]:
                self.owner.registration_settings.pop('ground_region', None)
                self.owner.invalidate()
        self.zoom, self.pan = 1.0, [0.0, 0.0]
        self.draw()

    def draw(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        cw, ch = max(canvas.winfo_width(), 100), max(canvas.winfo_height(), 100)
        if self.image is None:
            canvas.create_text(cw / 2, ch / 2, text="选择矫正照片或连接双目抓拍", fill="white")
            return
        h, w = self.image.shape[:2]
        scale = min(cw / w, ch / h) * self.zoom
        ox, oy = (cw - w * scale) / 2 + self.pan[0], (ch - h * scale) / 2 + self.pan[1]
        if (scale, ox, oy) != self.transform:
            self.cancel_region_drag()
        self.transform = (scale, ox, oy)
        # Render only the viewport, so zooming does not allocate giant bitmaps.
        rendered = cv2.warpAffine(self.image, np.array([[scale, 0, ox], [0, scale, oy]], dtype=float), (cw, ch), borderValue=(53, 34, 21))
        ok, payload = cv2.imencode(".png", rendered)
        if ok:
            self.photo = self.owner.app.tk.PhotoImage(data=base64.b64encode(payload).decode("ascii"), format="png")
            canvas.create_image(0, 0, image=self.photo, anchor="nw")
        selected = self.owner.selected_id()
        drawn_observations: set[str] = set()
        for spec in self.owner.pipes:
            box = spec.get(f"{self.role}_region_px")
            if not box:
                evidence = (self.owner.results.get(spec["pipe_id"], {}) or {}).get("current_evidence", {})
                box = (evidence.get(self.role, {}) or {}).get("region_xywh")
                label_override = None
                if not box:
                    surface = (self.owner.report or {}).get("local_surface", {}) if isinstance(self.owner.report, dict) else {}
                    for observation in surface.get("observations", []) if isinstance(surface, dict) else []:
                        if observation.get("pipe_id") == spec.get("pipe_id") or (observation.get("pipe_id") is None and observation.get("color_srgb") == spec.get("color_srgb")):
                            box = observation.get(f"{self.role}_region_px")
                            label_override = observation.get("observation_id")
                            if box:
                                if label_override:
                                    drawn_observations.add(str(label_override))
                                break
                if not box: continue
            else:
                label_override = None
            x, y, bw, bh = box
            result = self.owner.results.get(spec["pipe_id"], {})
            color = STATE_COLORS.get(result.get("installation_state"), "#58B6F2")
            canvas.create_rectangle(ox + x * scale, oy + y * scale, ox + (x + bw) * scale, oy + (y + bh) * scale,
                                    outline=color, width=3 if spec["pipe_id"] == selected else 1)
            label = label_override or spec.get("name") or spec["pipe_id"]
            if result:
                label += " · " + result.get("installation_state_zh", "不确定")
            canvas.create_text(ox + x * scale + 2, max(12, oy + y * scale - 3), anchor="sw", text=label, fill=color)
        # Before a rigid model pose is solved, show every detected local surface
        # with its OBS id.  Matching by model colour is only a convenience and
        # can hide observations when the physical pipe colour differs.
        report = self.owner.report if isinstance(self.owner.report, dict) else {}
        # Image proposals deliberately have no model ID or measurement. Show
        # their own IDs and failure stage even when every model row is UNKNOWN.
        for candidate in (report.get("local_surface") or {}).get("image_candidates",[]):
            if candidate.get("role") != self.role:
                continue
            x,y,bw,bh = candidate["region_px"]
            canvas.create_rectangle(ox+x*scale,oy+y*scale,ox+(x+bw)*scale,oy+(y+bh)*scale,
                                    outline="#FFD166",width=2,dash=(5,3),tags="image-candidate")
            label = str(candidate.get("candidate_id","")) + " · 图像候选"
            reasons = candidate.get("reason_codes") or []
            if reasons:
                label += " · " + REASON_TEXT.get(reasons[0],reasons[0])
            canvas.create_text(ox+x*scale+2,oy+y*scale+2,anchor="nw",text=label,
                               fill="#FFD166",tags="image-candidate")
        if self.owner.report is not None:
            registration = report.get("registration") if isinstance(report, dict) else None
            if not isinstance(registration, dict) or registration.get("status") != "MATCHED":
                surface = report.get("local_surface") if isinstance(report, dict) else None
                for observation in (surface or {}).get("observations", []) if isinstance(surface, dict) else []:
                    oid = str(observation.get("observation_id") or "")
                    if not oid or oid in drawn_observations:
                        continue
                    box = observation.get(f"{self.role}_region_px")
                    if not box:
                        continue
                    x, y, bw, bh = box
                    color = observation.get("color_srgb") or "#66C2FF"
                    canvas.create_rectangle(ox + x * scale, oy + y * scale,
                                            ox + (x + bw) * scale, oy + (y + bh) * scale,
                                            outline=color, width=2, dash=(3, 2))
                    canvas.create_text(ox + x * scale + 2, max(12, oy + y * scale - 3),
                                       anchor="sw", text=oid, fill=color)

        from .section_display import boundary_overlay_segments
        for x1, y1, x2, y2 in boundary_overlay_segments(report, self.role):
            canvas.create_line(ox+x1*scale, oy+y1*scale, ox+x2*scale, oy+y2*scale,
                               fill='#67E8EF', width=2, tags='pooled-boundary')

        # Group search bounds are independent from automatic pipe identities.
        if self.assessment_regions and self.role == "left" and self.owner.zone_settings.get("scope") == "zones":
            for zone in self.owner.zone_settings.get("zones", []):
                if not zone.get("enabled") or not zone.get("roi_rect_px"):
                    continue
                x, y, bw, bh = zone["roi_rect_px"]
                canvas.create_rectangle(ox + x * scale, oy + y * scale,
                                        ox + (x + bw) * scale, oy + (y + bh) * scale,
                                        outline="#67E8A5", width=2, dash=(7, 3), tags="assessment-zone")
                canvas.create_text(ox + x * scale + 4, oy + y * scale + 4,
                                   anchor="nw", text=f"{zone['zone_id']} · 评估区域", fill="#67E8A5",
                                   tags="assessment-zone")
        if self.band_controller:
            self.band_controller.draw()
            ground = self.owner.registration_settings.get('ground_region')
            if ground:
                x, y, bw, bh = ground['rect_px']
                canvas.create_rectangle(ox+x*scale, oy+y*scale, ox+(x+bw)*scale, oy+(y+bh)*scale,
                                        outline='#E7A9FF', width=2, tags='ground-region')
                canvas.create_text(ox+x*scale+4, oy+y*scale+4, anchor='nw', text='地面检查区域',
                                   fill='#E7A9FF', tags='ground-region')

    def cancel_region_drag(self) -> None:
        self.region_start = None
        self.canvas.delete("pending-zone")
        if self.band_controller:
            self.band_controller.cancel()

    def start_region(self, event: Any) -> None:
        if self.band_controller and self.band_controller.start(event):
            return
        if (self.role != "left" or self.image is None or self.owner.busy
                or not self.owner.region_selection_active):
            return
        scale, ox, oy = self.transform
        h, w = self.image.shape[:2]
        if not (ox <= event.x < ox + w * scale and oy <= event.y < oy + h * scale):
            return
        self.canvas.focus_set()
        self.region_start = (event.x, event.y)

    def move_region(self, event: Any) -> None:
        if self.band_controller and self.band_controller.move(event):
            return
        self.canvas.delete("pending-zone")
        if self.region_start is None or self.image is None or self.owner.busy:
            return
        h, w = self.image.shape[:2]
        rect = image_region(self.region_start, (event.x, event.y), self.transform, (w, h))
        if rect:
            scale, ox, oy = self.transform
            x, y, bw, bh = rect
            self.canvas.create_rectangle(ox + x * scale, oy + y * scale,
                                         ox + (x + bw) * scale, oy + (y + bh) * scale,
                                         outline="#FFE07A", width=2, dash=(4, 2), tags="pending-zone")

    def finish_region(self, event: Any) -> None:
        if self.band_controller and self.band_controller.finish(event):
            return
        start = self.region_start
        self.cancel_region_drag()
        if (start is None or self.image is None or self.owner.busy
                or not self.owner.region_selection_active):
            return
        h, w = self.image.shape[:2]
        rect = image_region(start, (event.x, event.y), self.transform, (w, h))
        if rect is None:
            self.owner.message.set("框选区域太小，请在左图重新拖出至少 32×32 原图像素的区域。")
            return
        if self.owner.region_selection_target == 'ground':
            self.owner.set_ground_region(rect)
        else:
            self.owner.add_left_region(rect)

    def wheel(self, event: Any) -> None:
        if self.region_start is not None or (self.band_controller and self.band_controller.dragging):
            return
        old = self.zoom
        self.zoom = min(8.0, max(1.0, old * (1.2 if event.delta > 0 else 1 / 1.2)))
        if self.zoom == 1:
            self.pan = [0.0, 0.0]
        self.draw()

    def start_pan(self, event: Any) -> None:
        self.cancel_region_drag()
        self.pan_start = (event.x, event.y, *self.pan)

    def move_pan(self, event: Any) -> None:
        if self.pan_start:
            x, y, px, py = self.pan_start
            self.pan = [px + event.x - x, py + event.y - y]
            self.draw()


class ZoneEditorDialog:
    """Small group-ROI editor; it never assigns an ROI to an individual pipe."""

    def __init__(self, owner: Any) -> None:
        self.owner = owner
        self.generation = owner.generation
        self.app = owner.app
        self.tk, self.ttk = self.app.tk, self.app.ttk
        image = owner.views["left"].image
        if image is None:
            raise ValueError("请先载入左目矫正图，再配置分区")
        self.image = image
        self.height, self.width = image.shape[:2]
        self.scale = min(1.0, 760.0 / self.width, 520.0 / self.height)
        self.display_width = max(1, int(round(self.width * self.scale)))
        self.display_height = max(1, int(round(self.height * self.scale)))
        self.zones = copy.deepcopy(owner.zone_settings.get("zones", []))
        self.drag_start: tuple[float, float] | None = None
        self.pending_rect: list[int] | None = None
        self.selected_index: int | None = None
        self.window = self.tk.Toplevel(owner.window)
        self.window.title("分区分析 · 左目矫正图")
        self.window.geometry("1040x700")
        self.window.transient(owner.window)
        self.window.protocol("WM_DELETE_WINDOW", self.window.destroy)
        layout = self.ttk.Frame(self.window, padding=10)
        layout.pack(fill="both", expand=True)
        self.canvas = self.tk.Canvas(layout, width=self.display_width, height=self.display_height,
                                     bg="#152235", highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        side = self.ttk.Frame(layout, padding=(10, 0, 0, 0))
        side.pack(side="right", fill="y")
        self.label = self.tk.StringVar(value="")
        self.ttk.Label(side, text="在左图拖出管道组的矩形区域；分区可以重叠。", wraplength=250).pack(anchor="w", pady=(0, 8))
        self.ttk.Label(side, text="分区名称（可选）").pack(anchor="w")
        self.label_entry = self.ttk.Entry(side, textvariable=self.label, width=28)
        self.label_entry.pack(anchor="w", pady=(2, 8))
        self.listbox = self.tk.Listbox(side, height=15, width=30, exportselection=False)
        self.listbox.pack(fill="y", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._selected)
        buttons = self.ttk.Frame(side)
        buttons.pack(fill="x", pady=8)
        self.ttk.Button(buttons, text="新增当前框选", command=self._add).pack(fill="x", pady=2)
        self.ttk.Button(buttons, text="绑定框选到所选候选", command=self._bind_pending).pack(fill="x", pady=2)
        self.ttk.Button(buttons, text="根据 STL 自动生成候选", command=self._auto_propose).pack(fill="x", pady=2)
        self.ttk.Button(buttons, text="确认/启用所选候选", command=self._confirm).pack(fill="x", pady=2)
        self.ttk.Button(buttons, text="删除所选分区", command=self._delete).pack(fill="x", pady=2)
        self.ttk.Button(buttons, text="保存已确认分区", command=self._save).pack(fill="x", pady=(10, 2))
        self.ttk.Button(buttons, text="取消", command=self.window.destroy).pack(fill="x", pady=2)
        self.status = self.tk.StringVar(value="请拖出一个矩形")
        self.ttk.Label(side, textvariable=self.status, wraplength=250, foreground="#355371").pack(anchor="w", pady=6)
        self._render_image()
        self._refresh_list()
        self.canvas.bind("<ButtonPress-1>", self._start)
        self.canvas.bind("<B1-Motion>", self._drag)
        self.canvas.bind("<ButtonRelease-1>", self._finish)

    def _render_image(self) -> None:
        rendered = cv2.resize(self.image, (self.display_width, self.display_height), interpolation=cv2.INTER_AREA)
        ok, payload = cv2.imencode(".png", rendered)
        if not ok:
            raise ValueError("无法渲染分区预览")
        self.photo = self.app.tk.PhotoImage(data=base64.b64encode(payload).decode("ascii"), format="png")

    def _pixel(self, x: float, y: float) -> tuple[int, int]:
        return (max(0, min(self.width, int(round(x / self.scale)))),
                max(0, min(self.height, int(round(y / self.scale)))))

    def _start(self, event: Any) -> None:
        self.drag_start = (event.x, event.y)
        self.pending_rect = None

    def _drag(self, event: Any) -> None:
        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        self.pending_rect = [*self._pixel(min(x0, x1), min(y0, y1)),
                             self._pixel(max(x0, x1), max(y0, y1))[0] - self._pixel(min(x0, x1), min(y0, y1))[0],
                             self._pixel(max(x0, x1), max(y0, y1))[1] - self._pixel(min(x0, x1), min(y0, y1))[1]]
        self._draw()

    def _finish(self, event: Any) -> None:
        self._drag(event)
        self.drag_start = None
        if self.pending_rect:
            try:
                normalized = normalize_zone_settings({"scope": "zones", "zones": [{
                    "zone_id": "Z01", "label": "Z01", "enabled": True,
                    "coordinate_space": "rectified_left", "roi_rect_px": self.pending_rect,
                }]}, image_size=(self.width, self.height))
                self.pending_rect = normalized["zones"][0]["roi_rect_px"]
                self.status.set(f"已框选 {self.pending_rect}，点击“新增当前框选”")
            except ValueError as error:
                self.pending_rect = None
                self.status.set(str(error))
        self._draw()

    def _draw(self) -> None:
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        for index, zone in enumerate(self.zones):
            rect = zone.get("roi_rect_px")
            if not rect:
                continue
            x, y, width, height = rect
            selected = index == self.selected_index
            self.canvas.create_rectangle(x * self.scale, y * self.scale,
                                         (x + width) * self.scale, (y + height) * self.scale,
                                         outline="#FFD166" if selected else "#58B6F2",
                                         width=3 if selected else 2)
            self.canvas.create_text(x * self.scale + 4, max(12, y * self.scale + 14),
                                    text=zone["zone_id"], anchor="w", fill="#FFFFFF")
        if self.pending_rect:
            x, y, width, height = self.pending_rect
            self.canvas.create_rectangle(x * self.scale, y * self.scale,
                                         (x + width) * self.scale, (y + height) * self.scale,
                                         outline="#FF9F1C", width=2, dash=(5, 3))

    def _refresh_list(self) -> None:
        self.listbox.delete(0, "end")
        for zone in self.zones:
            if zone.get("source") == "stl_parallel":
                state = "已启用" if zone.get("enabled") else "待确认"
                mapping_source = {"unmapped": "未映射", "matched_section": "模型投影", "user": "人工框选"}.get(
                    zone.get("roi_source", "unmapped"), zone.get("roi_source", "unmapped"))
                ids = ",".join(zone.get("model_pipe_ids", []))
                text = f"{zone['zone_id']} · {zone['label']} · STL {state}/{mapping_source} · {ids}"
            else:
                text = f"{zone['zone_id']} · {zone['label']} · {zone.get('roi_rect_px')}"
            self.listbox.insert("end", text)
        self._draw()

    def _selected(self, _event: Any = None) -> None:
        selection = self.listbox.curselection()
        self.selected_index = int(selection[0]) if selection else None
        if self.selected_index is not None:
            self.label.set(self.zones[self.selected_index]["label"])
        self._draw()

    def _add(self) -> None:
        if self.pending_rect is None:
            self.status.set("请先在左图拖出矩形")
            return
        if len(self.zones) >= MAX_ZONES:
            self.status.set(f"分区最多支持{MAX_ZONES}个")
            return
        used = {zone["zone_id"] for zone in self.zones}
        index = 1
        while f"Z{index:02d}" in used:
            index += 1
        zone_id = f"Z{index:02d}"
        label = self.label.get().strip() or zone_id
        self.zones.append({"zone_id": zone_id, "label": label, "enabled": True,
                           "coordinate_space": "rectified_left", "roi_rect_px": list(self.pending_rect)})
        self.pending_rect = None
        self.label.set("")
        self.selected_index = len(self.zones) - 1
        self._refresh_list()
        self.listbox.selection_set(self.selected_index)
        self.status.set(f"已加入{zone_id}；可继续框选或保存")

    def _bind_pending(self) -> None:
        """Attach the current image rectangle to the selected STL proposal."""
        if self.pending_rect is None:
            self.status.set("请先在左图拖出矩形")
            return
        if self.selected_index is None or self.selected_index >= len(self.zones):
            self.status.set("请先在列表中选择一个 STL 候选")
            return
        zone = self.zones[self.selected_index]
        if zone.get("source") != "stl_parallel":
            self.status.set("只有 STL 自动候选可以绑定模型分区")
            return
        zone["roi_rect_px"] = list(self.pending_rect)
        zone["roi_source"] = "user"
        zone["confirmed"] = False
        zone["enabled"] = False
        self.pending_rect = None
        self._refresh_list()
        self.listbox.selection_set(self.selected_index)
        log_event(LOGGER, "elevation_zone_model_roi_bound", zone_id=zone["zone_id"],
                  model_pipe_ids=list(zone.get("model_pipe_ids", [])), roi_rect_px=list(zone["roi_rect_px"]))
        self.status.set(f"已将框选范围绑定到{zone['zone_id']}；请点击“确认/启用所选候选”")

    def _auto_propose(self) -> None:
        if not self.owner.pipes:
            self.status.set("请先导入 STL 或 DXF，建立管道目录")
            return
        try:
            proposals = propose_model_zones(self.owner.pipes)
            calibration = None
            report = self.owner.report if isinstance(self.owner.report, Mapping) else None
            if report is not None:
                try:
                    from .stereo_analyzer import _calibration_from_manifest
                    calibration = _calibration_from_manifest(self.owner.current_calibration())
                except (OSError, ValueError, KeyError):
                    calibration = None
            if calibration is not None:
                for proposal in proposals:
                    rect = project_model_zone(proposal, self.owner.pipes, calibration, report)
                    if rect is not None:
                        proposal["roi_rect_px"] = rect
                        proposal["roi_source"] = "matched_section"
            # Regenerating proposals replaces drafts and stale model hashes;
            # confirmed candidates for the same catalogue and manual ROIs stay.
            catalog_hash = proposals[0].get("model_catalog_sha256") if proposals else None
            kept = [zone for zone in self.zones
                    if zone.get("source") != "stl_parallel"
                    or (zone.get("confirmed") and zone.get("model_catalog_sha256") == catalog_hash)]
            used = {str(zone.get("zone_id")) for zone in kept}
            existing_groups = {tuple(zone.get("model_pipe_ids", [])) for zone in kept
                               if zone.get("source") == "stl_parallel"}
            fresh = []
            for proposal in proposals:
                group = tuple(proposal.get("model_pipe_ids", []))
                if group in existing_groups:
                    continue
                zone_id = str(proposal["zone_id"])
                suffix = 1
                while zone_id in used:
                    suffix += 1
                    zone_id = f"{proposal['zone_id']}-{suffix}"
                proposal["zone_id"] = zone_id
                used.add(zone_id)
                existing_groups.add(group)
                fresh.append(proposal)
            if len(kept) + len(fresh) > MAX_ZONES:
                raise ValueError(f"模型候选与现有分区合计超过{MAX_ZONES}个，请先删除无用分区")
            self.zones = kept + fresh
            self.selected_index = None
            self._refresh_list()
            projected_fresh = sum(zone.get("roi_source") == "matched_section" for zone in fresh)
            log_event(LOGGER, "elevation_zone_model_candidates_generated",
                      model_pipe_count=len(self.owner.pipes), candidate_count=len(fresh),
                      projected_count=projected_fresh,
                      candidate_ids=[zone["zone_id"] for zone in fresh])
            self.status.set(f"已生成{len(fresh)}个 STL 平行管组候选，其中{projected_fresh}个已投影到当前照片；请逐项确认或删除")
        except (OSError, ValueError) as error:
            self.status.set(str(error))

    def _confirm(self) -> None:
        if self.selected_index is None or self.selected_index >= len(self.zones):
            self.status.set("请先选择一个候选")
            return
        zone = self.zones[self.selected_index]
        if zone.get("source") != "stl_parallel":
            zone["enabled"] = True
            self._refresh_list()
            log_event(LOGGER, "elevation_zone_confirmed", zone_id=zone["zone_id"], source="manual")
            self.status.set(f"已确认{zone['zone_id']}")
            return
        if not zone.get("roi_rect_px"):
            self.status.set("该候选尚未绑定照片范围，请先框选并点击“绑定框选到所选候选”")
            return
        zone["confirmed"] = True
        zone["enabled"] = True
        try:
            normalize_zone_settings({"scope": "zones", "zones": self.zones},
                                    image_size=(self.width, self.height))
        except ValueError as error:
            zone["confirmed"] = False
            zone["enabled"] = False
            self.status.set(str(error))
            return
        self._refresh_list()
        self.listbox.selection_set(self.selected_index)
        log_event(LOGGER, "elevation_zone_confirmed", zone_id=zone["zone_id"], source="stl_parallel",
                  model_pipe_ids=list(zone.get("model_pipe_ids", [])), roi_source=zone.get("roi_source"))
        self.status.set(f"已确认并启用{zone['zone_id']}；保存后用于分区分析")

    def _delete(self) -> None:
        if self.selected_index is None:
            return
        zone = self.zones.pop(self.selected_index)
        log_event(LOGGER, "elevation_zone_deleted", zone_id=zone.get("zone_id"),
                  source=zone.get("source", "manual"), model_pipe_ids=list(zone.get("model_pipe_ids", [])))
        self.selected_index = None
        self._refresh_list()

    def _save(self) -> None:
        if self.owner.busy or self.owner.generation != self.generation:
            self.status.set("现场输入或评估区域已改变，请关闭并重新打开分区管理。")
            return
        try:
            settings = normalize_zone_settings({**self.owner.zone_settings, "scope": "zones", "zones": self.zones},
                                               image_size=(self.width, self.height))
        except ValueError as error:
            self.status.set(str(error))
            return
        self.owner.zone_settings = settings
        self.owner.scope_label.set("分区分析")
        self.owner.invalidate()
        self.owner.redraw()
        log_event(LOGGER, "elevation_zones_saved", scope=settings["scope"],
                  zones=[{"zone_id": zone["zone_id"], "source": zone.get("source", "manual"),
                          "enabled": zone["enabled"]} for zone in settings["zones"]])
        self.owner.message.set("分区已保存；各分区继承全局直径/颜色匹配策略，保存并评估后执行。")
        self.window.destroy()


class ElevationCaptureDialog:
    """STL/DXF pipe layout → stereo capture → save/analyze/reopen."""

    # The field workflow is deliberately automatic.  Pipe identity comes from
    # the DXF/STL geometry and the stereo local surface. Optional group search
    # bounds never assign individual pipe identities.
    analysis_mode = "elevation_auto"

    def __init__(self, app: Any, manifest_path: Path | None = None, *, output_root: Path | None = None,
                 restore: bool = True, model_path: Path | None = None) -> None:
        from .workbench_profile import load_profile

        self.app = app
        self.output_root = Path(output_root) if output_root is not None else ROOT
        self.profile, self.profile_problem = load_profile()
        self.calibration_override = None
        self.pose_adjustment = {"mode": "keep"}
        self.pipes: list[dict[str, Any]] = []
        self.results: dict[str, dict[str, Any]] = {}
        self.report = None
        self.last_manifest: Path | None = None
        self.generation = 0
        self.busy = False
        self.closed = False
        self.calibration_dialog = None
        self.camera_dialog = None
        self.model_viewer = None
        self.quality_dialog = None
        self.messages: queue.Queue = queue.Queue()
        self.timestamp_sources = {role: "MANIFEST_OPERATOR_CONFIRMED" for role in ("left", "right")}
        self.camera_capture_provenance = {}
        self.temporal_frames: list[dict] = []
        self.image_paths: dict[str, str] = {}
        self.image_hashes: dict[str, str] = {}
        self.status_refresh_metadata: dict[str, str] | None = None
        self.registration_settings: dict[str, Any] = {"axis_world": None, "anchors": {}, "local_observation_mode": "auto"}
        self.model_kind = ""
        self.model_half_length_mm = 1000.0
        self.analysis_settings: dict[str, Any] = {}
        self.zone_settings: dict[str, Any] = normalize_zone_settings()
        self.region_selection_active = False
        self.region_selection_target = 'assessment'
        self.axial_selection_active = False
        tk, ttk = app.tk, app.ttk
        self.window = tk.Toplevel(app.root)
        self.window.title("基础立面评估 · 双目管道状态")
        available_width = max(640, self.window.winfo_screenwidth()-80)
        available_height = max(480, self.window.winfo_screenheight()-100)
        self.window.geometry(f"{min(1120, available_width)}x{min(860, available_height)}")
        self.window.minsize(min(940, available_width), min(660, available_height))
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.fields = {key: tk.StringVar() for key in ("model", "calibration", "left", "right", "left_time", "right_time")}
        self.stl_unit = tk.StringVar(value="millimeter")
        self.confirmed = tk.BooleanVar(value=False)
        self.history = tk.BooleanVar(value=True)
        self.local_observation_label = tk.StringVar(value="自动（圆柱/图像引导）")
        self.message = tk.StringVar(value="首次：导入 STL 或 DXF → 选择双目标定 → 设置管长方向 → 双目抓拍 → 自动匹配立面。固定机位以后可直接重复抓拍。")
        self.calibration_status = tk.StringVar(value="未加载标定；重投影通过后仍需独立尺寸复核")
        self.summary = tk.StringVar(value="尚未建立管道目录")
        self.matching_preset = tk.StringVar(value="原始灰度")
        self.disparity_count = tk.StringVar(value="256")
        self.diameter_tolerance_mm = tk.StringVar(value="3.0")
        self.diameter_tolerance_ratio = tk.StringVar(value="0.10")
        self.color_delta_lab = tk.StringVar(value="45.0")
        self.color_filter_enabled = tk.BooleanVar(value=False)
        self.present_pipe_count = tk.StringVar(value="")
        self.scope_label = tk.StringVar(value="全幅自动")
        self.camera_side_label = tk.StringVar(value=camera_side_label(None))

        frame = ttk.Frame(self.window, padding=12)
        frame.pack(fill="both", expand=True)
        self.controls = ttk.Frame(frame)
        self.controls.pack(fill="x")
        bar = ttk.Frame(self.controls)
        bar.pack(fill="x", pady=(0, 7))
        ttk.Button(bar, text="① 导入 STL/DXF", command=self.browse_model).pack(side="left", padx=3)
        ttk.Combobox(bar, textvariable=self.stl_unit, values=("millimeter", "centimeter", "meter", "inch"), state="readonly", width=11).pack(side="left")
        ttk.Button(bar, text="② 相机标定", command=self.browse_calibration).pack(side="left", padx=3)
        ttk.Button(bar, text="重新标定 / 质量检查", command=self.open_calibration).pack(side="left", padx=3)
        ttk.Button(bar, text="打开基础现场", command=self.browse_session).pack(side="right", padx=3)
        action_bar = ttk.Frame(self.controls)
        action_bar.pack(fill="x", pady=(0, 7))
        ttk.Button(action_bar, text="③ 双目抓拍", command=self.capture_camera).pack(side="left", padx=3)
        self.run_button = ttk.Button(action_bar, text="④ 保存并评估", command=lambda: self.submit(True))
        self.run_button.pack(side="left", padx=6)
        ttk.Button(action_bar, text="状态刷新", command=self.refresh_status).pack(side="left", padx=3)
        self.save_button = ttk.Button(action_bar, text="仅保存基础现场", command=lambda: self.submit(False))
        self.save_button.pack(side="left", padx=12)
        ttk.Label(action_bar, text="已抓拍：点击④开始分析", foreground="#355371").pack(side="left", padx=6)
        method_bar = ttk.Frame(self.controls)
        method_bar.pack(fill="x", pady=(0, 4))
        ttk.Label(method_bar, text="局部建模").pack(side="left", padx=(3, 3))
        self.local_observation_box = ttk.Combobox(
            method_bar, textvariable=self.local_observation_label,
            values=("自动（圆柱/图像引导）", "仅双目深度几何", "图像引导圆柱", "平行局部条带（颜色辅助）"),
            state="readonly", width=22)
        self.local_observation_box.pack(side="left")
        self.local_observation_box.bind("<<ComboboxSelected>>", lambda _e: self._local_observation_selected())
        ttk.Label(method_bar, text="分析范围").pack(side="left", padx=(10, 3))
        self.scope_box = ttk.Combobox(method_bar, textvariable=self.scope_label,
                                      values=("全幅自动", "分区分析"), state="readonly", width=12)
        self.scope_box.pack(side="left")
        self.scope_box.bind("<<ComboboxSelected>>", lambda _e: self._scope_selected())
        self.zone_button = ttk.Button(method_bar, text="分区管理", command=self.open_zone_manager)
        self.zone_button.pack(side="left", padx=3)
        self.region_button = ttk.Button(method_bar, text="左图框选区域", command=self.toggle_region_selection)
        self.region_button.pack(side="left", padx=3)
        self.axial_button = ttk.Button(method_bar, text='管长方向 / 汇聚范围', command=self.toggle_axial_selection)
        self.axial_button.pack(side='left', padx=3)
        ttk.Button(method_bar, text='清除汇聚范围', command=self.clear_axial_band).pack(side='left', padx=3)
        self.window.bind("<Escape>", lambda _e: self.cancel_image_selections())
        ground_bar = ttk.Frame(self.controls)
        ground_bar.pack(fill='x', pady=(0, 3))
        self.ground_button = ttk.Button(ground_bar, text='左图框选地面', command=self.toggle_ground_selection)
        self.ground_button.pack(side='left', padx=3)
        ttk.Button(ground_bar, text='地面距离 / 姿态检查', command=self.show_ground_check).pack(side='left', padx=3)
        ttk.Button(ground_bar, text='清除地面区域', command=self.clear_ground_region).pack(side='left', padx=3)
        ttk.Label(ground_bar, text='框选裸露地面；与管段汇聚范围独立。', foreground='#355371').pack(side='left', padx=6)
        self.settings_button = ttk.Button(ground_bar, text='展开拍摄 / 评估设置', command=self.toggle_capture_settings)
        self.settings_button.pack(side='right', padx=3)
        ttk.Label(self.controls, textvariable=self.calibration_status, foreground="#355371").pack(anchor="w")
        # Keep photo selection usable at the minimum window size. Infrequent
        # controls remain available in one expandable area, with values intact.
        self.capture_settings = ttk.Frame(self.controls)
        self.capture_settings_expanded = False
        photos = ttk.Frame(self.capture_settings)
        photos.pack(fill="x", pady=6)
        for role, name in (("left", "左"), ("right", "右")):
            ttk.Button(photos, text=f"导入{name}矫正图", command=lambda r=role: self.browse_photo(r)).pack(side="left", padx=3)
            ttk.Entry(photos, textvariable=self.fields[f"{role}_time"], width=29).pack(side="left")
        ttk.Checkbutton(self.capture_settings, text="左右图来自同一次同步拍摄；机位、镜头设置和区域位置正确", variable=self.confirmed).pack(anchor="w")
        matching_bar = ttk.Frame(self.capture_settings)
        matching_bar.pack(fill="x", pady=3)
        ttk.Label(matching_bar, text="深度处理").pack(side="left")
        ttk.Combobox(matching_bar, textvariable=self.matching_preset, values=("原始灰度", "弱光降噪"),
                     state="readonly", width=12).pack(side="left", padx=6)
        ttk.Label(matching_bar, text="视差搜索范围（像素，16 的倍数）").pack(side="left")
        ttk.Entry(matching_bar, textvariable=self.disparity_count, width=7).pack(side="left", padx=6)
        ttk.Label(matching_bar, text="近距离需扩大范围；弱光处理保留原始照片与颜色", foreground="#355371").pack(side="left")
        policy_bar = ttk.Frame(self.capture_settings)
        policy_bar.pack(fill="x", pady=3)
        ttk.Label(policy_bar, text="直径容差 mm").pack(side="left")
        ttk.Entry(policy_bar, textvariable=self.diameter_tolerance_mm, width=7).pack(side="left", padx=(4, 10))
        ttk.Label(policy_bar, text="相对容差").pack(side="left")
        ttk.Entry(policy_bar, textvariable=self.diameter_tolerance_ratio, width=7).pack(side="left", padx=(4, 10))
        ttk.Checkbutton(policy_bar, text="启用颜色筛选（辅助）", variable=self.color_filter_enabled).pack(side="left", padx=(4, 10))
        ttk.Label(policy_bar, text="颜色容差 Lab").pack(side="left")
        ttk.Entry(policy_bar, textvariable=self.color_delta_lab, width=7).pack(side="left", padx=(4, 6))
        ttk.Label(policy_bar, text="直径始终为主筛选；STL 颜色需先按管径确认", foreground="#355371").pack(side="left")
        scene_bar = ttk.Frame(self.capture_settings)
        scene_bar.pack(fill="x", pady=3)
        ttk.Label(scene_bar, text="现场实际管数（未知可留空）").pack(side="left")
        ttk.Entry(scene_bar, textvariable=self.present_pipe_count, width=5).pack(side="left", padx=6)
        ttk.Label(scene_bar, text="模型目录可多于现场实物；数量不代替管道身份确认", foreground="#355371").pack(side="left")
        ttk.Button(scene_bar, text="颜色与反光检查", command=self.open_quality_check).pack(side="right", padx=4)

        view_bar = ttk.Frame(self.capture_settings)
        view_bar.pack(fill="x", pady=3)
        ttk.Label(view_bar, text="相机从哪侧看（模型坐标）").pack(side="left")
        self.camera_side_box = ttk.Combobox(view_bar, textvariable=self.camera_side_label,
            values=tuple(CAMERA_SIDE_PRESETS), state="readonly", width=25)
        self.camera_side_box.pack(side="left", padx=6)
        self.camera_side_box.bind("<<ComboboxSelected>>", lambda _e: self._camera_side_selected())
        ttk.Button(view_bar, text="查看方向示意", command=self.open_model_viewer).pack(side="left", padx=4)
        ttk.Label(view_bar, text="允许斜视；管长方向另设。不确定时保留自动判断。", foreground="#355371").pack(side="left")

        # Reserve results and status before the expanding photo pane consumes
        # space, so shorter displays do not lose the bottom of the workflow.
        footer = ttk.Frame(frame)
        footer.pack(side="bottom", fill="x")
        panes = ttk.Panedwindow(frame, orient="horizontal")
        panes.pack(fill="both", expand=True, pady=8)
        self.views = {}
        for role, title in (("left", "左目 · 评估区域与匹配结果"), ("right", "右目 · 自动匹配结果")):
            view_frame = ttk.LabelFrame(panes, text=title, padding=3)
            panes.add(view_frame, weight=1)
            self.views[role] = RegionCanvas(view_frame, self, role, assessment_regions=True)
        ttk.Label(footer, text="左图画管长方向：两条橙色垂线之间汇聚；拖动端点可调范围。青色线为稳定边界。方向与范围随现场保存。", foreground="#355371").pack(anchor="w")

        table_frame = ttk.Frame(footer)
        table_frame.pack(fill="x", pady=6)
        columns = ("diameter", "color", "left", "right", "reference", "state", "position", "reason")
        self.tree = ttk.Treeview(table_frame, columns=columns, height=5, selectmode="browse")
        self.tree.heading("#0", text="管道编号")
        self.tree.column("#0", width=90, stretch=False)
        for column, label, width in zip(columns, ("模型管径 mm", "颜色", "左证据", "右证据", "深度", "安装证据", "断面就位", "说明"), (90, 80, 60, 60, 60, 90, 135, 310)):
            self.tree.heading(column, text=label)
            self.tree.column(column, width=width, stretch=column == "reason")
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.tree.pack(side="left", fill="x", expand=True)
        for state, color in STATE_COLORS.items():
            self.tree.tag_configure(state, foreground=color)
        self.tree.bind("<<TreeviewSelect>>", lambda _e: self.redraw())
        self.tree.bind("<Double-1>", lambda _e: self.edit_pipe())
        row = ttk.Frame(footer)
        row.pack(fill="x")
        actions = [("设置颜色", self.edit_pipe), ("同径统一颜色", self.apply_diameter_color),
                   ("管长与观察方向", self.open_model_viewer), ("查看模型管道编号", self.show_catalog),
                   ("断面尺寸与就位", self.show_measurement), ("管段汇聚", self.show_section_aggregation)]
        for name, action in actions:
            ttk.Button(row, text=name, command=action).pack(side="left", padx=2)
        bottom = ttk.Frame(footer)
        bottom.pack(fill="x", pady=(8, 4))
        ttk.Checkbutton(bottom, text="沿用上次场景历史（相机未移动）", variable=self.history).pack(side="left")
        ttk.Button(bottom, text="识别诊断", command=self.show_recognition_diagnostics).pack(side="left", padx=8)
        ttk.Label(footer, textvariable=self.summary, wraplength=860).pack(anchor="w")
        ttk.Label(footer, textvariable=self.message, wraplength=860, foreground="#355371").pack(fill="x", pady=3)

        for key, variable in self.fields.items():
            variable.trace_add("write", lambda *_args, k=key: self._field_changed(k))
        self.stl_unit.trace_add("write", lambda *_args: self.unit_changed())
        self.confirmed.trace_add("write", lambda *_args: self.invalidate())
        self.matching_preset.trace_add("write", lambda *_args: self.invalidate())
        self.disparity_count.trace_add("write", lambda *_args: self.invalidate())
        self.diameter_tolerance_mm.trace_add("write", lambda *_args: self.invalidate())
        self.diameter_tolerance_ratio.trace_add("write", lambda *_args: self.invalidate())
        self.color_delta_lab.trace_add("write", lambda *_args: self.invalidate())
        self.color_filter_enabled.trace_add("write", lambda *_args: self.invalidate())
        self.present_pipe_count.trace_add("write", lambda *_args: self.invalidate())
        if self.profile:
            self.fields["calibration"].set(self.profile.get("calibration_path", ""))
            self.calibration_override = copy.deepcopy(self.profile.get("calibration_current"))
        if manifest_path is not None:
            try:
                self.load_session(manifest_path)
            except Exception:
                self.window.destroy()
                raise
        elif restore:
            try:
                active_profile = copy.deepcopy(self.profile)
                recent = json.loads((self.output_root / "last_scene.json").read_text(encoding="utf-8"))
                self.load_session(Path(recent["manifest_path"]))
                active = (active_profile or {}).get("calibration_current")
                if (active and active_profile.get("calibration_path")
                        and active.get("calibration_id") != self.calibration_override.get("calibration_id")):
                    # Auto-resume model/settings with the newly selected calibration.
                    # Explicitly opening a historical package still uses its own
                    # calibration. Old rectified photos cannot be reused here.
                    self.profile = active_profile
                    self.fields["calibration"].set(active_profile["calibration_path"])
                    self.calibration_override = copy.deepcopy(active)
                    self.camera_capture_provenance = {}
                    self.status_refresh_metadata = None
                    self.summary.set("已恢复模型；当前标定已更新，等待新照片")
                    self.message.set("当前标定已更新，请点击③双目抓拍，再保存并评估。历史照片仍保留在原现场包中。")
            except FileNotFoundError:
                pass
            except (OSError, ValueError, KeyError) as error:
                self.message.set(f"上次基础现场未恢复：{error}")
        if manifest_path is None and model_path is not None:
            # The main GUI may already have an imported DXF.  Carry that
            # selection into the elevation dialog so the operator does not
            # have to browse for the same layout twice.
            self.load_model(Path(model_path).resolve())
        self.refresh_calibration_status()
        self.poll_id = self.window.after(100, self.poll)

    def invalidate(self) -> None:
        self.generation += 1
        if self.region_selection_active:
            self._set_region_selection(False)
        if self.axial_selection_active or self.views['left'].band_controller.dragging:
            self._set_axial_selection(False)
        if self.report is not None or self.results:
            self.report, self.results = None, {}
            self.refresh_table()
            self.summary.set("输入已改变，请重新评估")

    def _field_changed(self, key: str) -> None:
        self.invalidate()
        if key in {"left", "right", "left_time", "right_time", "calibration"}:
            self.temporal_frames = []
        if key in {"left", "right", "calibration"} and self.quality_dialog is not None and not self.quality_dialog.closed:
            self.quality_dialog.clear_inputs()
        if key == "calibration":
            self.calibration_override = None
            self.registration_settings.pop('aggregation_band', None)
            self.registration_settings.pop('ground_region', None)
            self.zone_settings = normalize_zone_settings()
            self.scope_label.set("全幅自动")
            self._clear_registration_anchors()
            self.confirmed.set(False)
            for spec in self.pipes:
                spec.pop("left_region_px", None)
                spec.pop("right_region_px", None)
            for role, view in self.views.items():
                self.fields[role].set("")
                self.fields[f"{role}_time"].set("")
                view.set_image(None)
            self.image_paths.clear()
            self.image_hashes.clear()
            self.last_manifest = None
            self.refresh_table()
        if key in {"left", "right"}:
            self.confirmed.set(False)

    def unit_changed(self) -> None:
        if self.pipes and self.fields["model"].get():
            self.pipes.clear()
            self.model_kind = ""
            self.last_manifest = None
            self.invalidate()
            self.refresh_table()
            self.message.set("模型单位已改变，请重新导入 STL 或 DXF")

    def _scope_selected(self) -> None:
        if self.scope_label.get() == "分区分析":
            if not self.zone_settings.get("zones"):
                self.message.set("分区分析需要至少一个区域；点击“左图框选区域”，在左图拖框。")
            self.zone_settings["scope"] = "zones"
        else:
            self.zone_settings = normalize_zone_settings()
            self.message.set("已切换为全幅自动分析；分区配置已停用。")
        self.invalidate()
        self.redraw()

    def _set_region_selection(self, active: bool, *, target='assessment') -> None:
        if active:
            self._set_axial_selection(False)
        self.region_selection_active = active
        self.region_selection_target = target if active else 'assessment'
        self.region_button.configure(text="取消框选（Esc）" if active and target == 'assessment' else "左图框选区域")
        self.ground_button.configure(text='取消地面框选（Esc）' if active and target == 'ground' else '左图框选地面')
        view = self.views["left"]
        view.cancel_region_drag()
        view.canvas.configure(cursor="crosshair" if active else "")

    def _set_axial_selection(self, active: bool) -> None:
        self.axial_selection_active = active
        self.axial_button.configure(text='取消方向绘制（Esc）' if active else '管长方向 / 汇聚范围')
        view = self.views['left']
        view.cancel_region_drag()
        view.canvas.configure(cursor='crosshair' if active else '')

    def cancel_image_selections(self) -> None:
        self._set_region_selection(False)
        self._set_axial_selection(False)
        self.redraw()

    def toggle_axial_selection(self) -> None:
        if self.busy:
            return
        if self.axial_selection_active:
            self.cancel_image_selections()
            return
        if self.views['left'].image is None:
            self.message.set('请先抓拍或导入左目矫正图，再指定管长方向。')
            return
        self._set_region_selection(False)
        self._set_axial_selection(True)
        self.message.set('在左图沿管长拖一条线：两端生成垂线，两线之间汇聚。建议避开支架和管端、选择至少 384 px；完成后可拖动橙色端点调整范围。')

    def set_axial_band(self, payload: dict) -> None:
        if self.busy or self.views['left'].image is None:
            return
        from .axial_selection import normalize_axial_band, band_geometry
        image = self.views['left'].image
        band = normalize_axial_band(payload, image_size=image.shape[1::-1])
        self.registration_settings['aggregation_band'] = band
        self.invalidate()
        self.redraw()
        length = band_geometry(band)[2]
        note = '；范围较短，可能不足以通过子段稳定性检查' if length < 384 else ''
        self.message.set(f'已指定汇聚范围 {length:.0f} px{note}。点击④保存并评估；所画方向仅用于图像取样，三维方向仍由双目解算。')
        log_event(LOGGER, 'elevation_axial_band_changed', aggregation_band=band)

    def clear_axial_band(self) -> None:
        if self.busy:
            return
        self.registration_settings.pop('aggregation_band', None)
        self.invalidate()
        self.redraw()
        self.message.set('已清除手动汇聚范围，恢复自动管段汇聚。')

    def toggle_capture_settings(self) -> None:
        self.capture_settings_expanded = not self.capture_settings_expanded
        if self.capture_settings_expanded:
            self.capture_settings.pack(fill='x')
        else:
            self.capture_settings.pack_forget()
        self.settings_button.configure(text='收起拍摄 / 评估设置' if self.capture_settings_expanded else '展开拍摄 / 评估设置')

    def toggle_ground_selection(self) -> None:
        if self.busy:
            return
        if self.region_selection_active and self.region_selection_target == 'ground':
            self.cancel_image_selections()
            return
        if self.views['left'].image is None:
            self.message.set('请先抓拍或导入左目矫正图，再框选地面。')
            return
        self._set_region_selection(True, target='ground')
        self.message.set('在左图框选一块裸露地面，避开支架、管道和杂物。应有足够宽度与高度；框选后点击④保存并评估。')

    def set_ground_region(self, rect: list[int]) -> None:
        if self.busy or self.views['left'].image is None:
            return
        from .ground_check import normalize_ground_region
        ground = normalize_ground_region({'image_size': list(self.views['left'].image.shape[1::-1]), 'rect_px': rect})
        self.registration_settings['ground_region'] = ground
        self.invalidate()
        self.redraw()
        self.message.set('已保存地面区域。点击④保存并评估，再打开“地面距离 / 姿态检查”。')
        log_event(LOGGER, 'elevation_ground_region_changed', ground_region=ground)

    def clear_ground_region(self) -> None:
        if not self.busy:
            self.registration_settings.pop('ground_region', None)
            self.invalidate()
            self.redraw()

    def show_ground_check(self) -> None:
        from .ground_check import ground_summary_lines
        window = self.app.tk.Toplevel(self.window)
        window.title('地面距离与相机姿态检查')
        window.geometry('840x450')
        text = self.app.tk.Text(window, wrap='word', padx=12, pady=12)
        text.pack(fill='both', expand=True)
        check = (self.report or {}).get('ground_check')
        if check is None and self.registration_settings.get('ground_region'):
            check = {'status': 'PENDING_EVALUATION'}
        text.insert('1.0', '\n'.join(ground_summary_lines(check)))
        text.configure(state='disabled')

    def toggle_region_selection(self) -> None:
        if self.busy:
            return
        if self.region_selection_active:
            self._set_region_selection(False)
            self.message.set("已取消框选；现有评估区域保持不变。")
            return
        if self.views["left"].image is None:
            self.message.set("请先双目抓拍或导入左目矫正图，再框选评估区域。")
            return
        if len(self.zone_settings.get("zones", [])) >= MAX_ZONES:
            self.message.set(f"评估区域最多 {MAX_ZONES} 个，请在“分区管理”中删除不需要的区域。")
            return
        self._set_region_selection(True)
        self.message.set("在左图拖框选断面附近的一小段，跨过待比较的各管并保留完整管宽；不必框整根管长。滚轮缩放、右键平移；Esc 取消。")

    def add_left_region(self, rect: list[int]) -> None:
        """Add a group ROI through the existing persisted zone contract."""
        image = self.views["left"].image
        if self.busy or image is None:
            return
        zones = copy.deepcopy(self.zone_settings.get("zones", []))
        used = {zone["zone_id"] for zone in zones}
        index = 1
        while f"Z{index:02d}" in used:
            index += 1
        zone_id = f"Z{index:02d}"
        zones.append({"zone_id": zone_id, "label": zone_id, "enabled": True,
                      "coordinate_space": "rectified_left", "roi_rect_px": list(rect)})
        try:
            settings = normalize_zone_settings({**self.zone_settings, "scope": "zones", "zones": zones},
                                               image_size=(image.shape[1], image.shape[0]))
        except ValueError as error:
            self.message.set(str(error))
            return
        self.zone_settings = settings
        self.scope_label.set("分区分析")
        self.invalidate()
        self.redraw()
        log_event(LOGGER, "elevation_left_region_added", zone_id=zone_id, roi_rect_px=rect)
        self.message.set(f"已加入 {zone_id}，切换为分区分析。点击④保存并评估；修改或删除请用“分区管理”。")

    def open_zone_manager(self) -> None:
        if self.busy:
            return
        self._set_region_selection(False)
        try:
            if self.views["left"].image is None:
                raise ValueError("请先载入左目矫正图，再配置分区")
            ZoneEditorDialog(self)
        except (OSError, ValueError) as error:
            self.show_error(error)

    def _restore_zone_settings(self, payload: Mapping[str, Any] | None) -> None:
        image = self.views["left"].image
        size = (image.shape[1], image.shape[0]) if image is not None else None
        self.zone_settings = normalize_zone_settings(payload, image_size=size)
        self.scope_label.set("分区分析" if self.zone_settings["scope"] == "zones"
                             else "全幅自动")

    def _matching_settings_from_controls(self) -> dict[str, Any]:
        from .matching_config import normalize_matching_settings

        try:
            payload = {
                "diameter_filter_enabled": True,
                "diameter_tolerance_mm": float(self.diameter_tolerance_mm.get()),
                "diameter_tolerance_ratio": float(self.diameter_tolerance_ratio.get()),
                "color_filter_enabled": bool(self.color_filter_enabled.get()),
                "color_filter_mode": "hint",
                "color_delta_lab": float(self.color_delta_lab.get()),
            }
        except (TypeError, ValueError) as error:
            raise ValueError("直径/颜色匹配设置必须是有效数字") from error
        return normalize_matching_settings(payload)

    def _restore_matching_settings(self, payload: Mapping[str, Any] | None) -> None:
        from .matching_config import normalize_matching_settings

        settings = normalize_matching_settings(payload)
        self.diameter_tolerance_mm.set(str(settings["diameter_tolerance_mm"]))
        self.diameter_tolerance_ratio.set(str(settings["diameter_tolerance_ratio"]))
        self.color_delta_lab.set(str(settings["color_delta_lab"]))
        self.color_filter_enabled.set(bool(settings["color_filter_enabled"]))

    def selected_id(self) -> str | None:
        selected = self.tree.selection()
        return str(selected[0]) if selected else None

    def open_quality_check(self) -> None:
        if self.busy:
            return
        from .capture_quality_gui import CaptureQualityDialog
        if self.quality_dialog is not None and not self.quality_dialog.closed:
            self.quality_dialog.refresh_inputs()
            self.quality_dialog.window.lift()
        else:
            self.quality_dialog = CaptureQualityDialog(self)

    def selected_pipe(self) -> dict[str, Any] | None:
        return next((row for row in self.pipes if row["pipe_id"] == self.selected_id()), None)

    def redraw(self) -> None:
        for view in self.views.values():
            view.draw()
        if self.model_viewer is not None and self.model_viewer.window.winfo_exists():
            self.model_viewer.draw()

    def select_model_pipe(self, pipe_id: str) -> None:
        """Navigation only: selecting a model pipe never creates an association."""
        if self.tree.exists(pipe_id):
            self.tree.selection_set(pipe_id)
            self.tree.see(pipe_id)
            self.redraw()

    def _local_observation_selected(self) -> None:
        """Persist the local 3-D observation basis in the scene manifest."""
        labels = {
            "自动（圆柱/图像引导）": "auto",
            "仅双目深度几何": "geometry_only",
            "图像引导圆柱": "image_guided",
            "平行局部条带（颜色辅助）": "parallel_strip",
        }
        self.registration_settings["local_observation_mode"] = labels.get(
            self.local_observation_label.get(), "auto")
        self.invalidate()
        if self.registration_settings["local_observation_mode"] == "geometry_only":
            self.message.set("仅双目深度几何：沿共同管轴建立左右目共同局部截断，颜色不参与候选生成。")
        elif self.registration_settings["local_observation_mode"] == "image_guided":
            self.message.set("图像引导圆柱：按图像长条生成候选，剔除异常视差，再独立验证双目圆柱；不按模型颜色筛选。")
        elif self.registration_settings["local_observation_mode"] == "parallel_strip":
            self.message.set("平行局部条带：沿共同管轴测量可见条带，颜色仅作候选辅助。")
        else:
            self.message.set("自动局部建模：优先圆柱表面；颜色关闭时回退到图像引导圆柱，颜色开启时回退到平行条带。")

    def refresh_table(self) -> None:
        selected = self.selected_id()
        self.tree.delete(*self.tree.get_children())
        for spec in self.pipes:
            row = self.results.get(spec["pipe_id"], {})
            evidence = row.get("current_evidence", {}) if isinstance(row, dict) else {}
            left_region = None
            right_region = None
            if isinstance(evidence, dict):
                left_region = (evidence.get("left") or {}).get("region_xywh") or left_region
                right_region = (evidence.get("right") or {}).get("region_xywh") or right_region
            reasons = row.get("reason_codes", [])
            pending_text = "待匹配"
            explanation = "、".join(REASON_TEXT.get(str(code), str(code)) for code in reasons) or REASON_TEXT.get(row.get("state_basis"), pending_text)
            if row.get("state_basis") == "REFERENCE_DEPTH_REQUIRED_FOR_NEGATIVE":
                explanation = REASON_TEXT[row["state_basis"]]
            elif row.get("installation_state") == "UNKNOWN" and row.get("current_evidence", {}).get("free_space_candidate"):
                explanation = "当前为空位候选；需至少两次间隔 ≥1 秒的连续独立抓拍"
            if isinstance(evidence, dict) and evidence.get("measured_diameter_mm") is not None:
                diameter_text = f"实测外径 {float(evidence['measured_diameter_mm']):.1f} mm"
                distance = evidence.get("observed_distance_to_left_camera_mm")
                distance_text = f"；截面中心线距左目 {float(distance):.0f} mm" if isinstance(distance, (int, float)) else ""
                explanation = f"{diameter_text}{distance_text}；{explanation}"
            position = row.get('section_position') or {}
            self.tree.insert("", "end", iid=spec["pipe_id"], text=spec["pipe_id"], tags=(row.get("installation_state", "UNKNOWN"),), values=(f"{spec['nominal_diameter_mm']:g}", spec["color_srgb"], "自动" if left_region else "待匹配", "自动" if right_region else "待匹配", "自动", row.get("installation_state_zh", "待评估"), position.get('state_zh', '待评估'), explanation))
        if self.pipes:
            self.tree.selection_set(selected if selected in self.tree.get_children() else self.pipes[0]["pipe_id"])
        self.redraw()

    def show_measurement(self) -> None:
        spec = self.selected_pipe()
        if spec is None:
            return
        tk, ttk = self.app.tk, self.app.ttk
        dialog = tk.Toplevel(self.window)
        dialog.title("断面尺寸与就位 · 模型与图像估计")
        dialog.geometry("850x660")
        text = tk.Text(dialog, wrap="word", padx=15, pady=15, font=("Microsoft YaHei", 10))
        text.pack(fill="both", expand=True)
        lines = relative_layout_summary_lines(self.report)
        lines += ["", "—— 所选管道的相机坐标诊断 ——", ""]
        lines += measurement_summary_lines(spec,self.results.get(spec["pipe_id"]))
        text.insert("1.0", "\n".join(lines))
        text.configure(state="disabled")
        ttk.Button(dialog, text="关闭", command=dialog.destroy).pack(pady=6)

    def show_section_aggregation(self) -> None:
        from .section_display import section_summary_lines
        tk, ttk = self.app.tk, self.app.ttk
        dialog = tk.Toplevel(self.window)
        dialog.title('管段汇聚 · 边界、尺寸与相对就位')
        dialog.geometry('980x700')
        text = tk.Text(dialog, wrap='word', padx=15, pady=15, font=('Microsoft YaHei', 10))
        scroll = ttk.Scrollbar(dialog, orient='vertical', command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        text.pack(fill='both', expand=True)
        text.insert('1.0', '\n'.join(section_summary_lines(self.report, detailed=True)))
        text.configure(state='disabled')

    def show_recognition_diagnostics(self) -> None:
        tk, ttk = self.app.tk, self.app.ttk
        dialog = tk.Toplevel(self.window)
        dialog.title("识别诊断 · 图像、深度与圆柱拟合")
        dialog.geometry("900x660")
        frame = ttk.Frame(dialog)
        frame.pack(fill="both", expand=True)
        text = tk.Text(frame, wrap="word", padx=15, pady=15, font=("Microsoft YaHei",10))
        scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
        text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)
        text.insert("1.0", "\n".join(recognition_diagnostic_lines(self.report)))
        text.configure(state="disabled")
        ttk.Button(dialog, text="关闭", command=dialog.destroy).pack(pady=6)

    def browse_model(self) -> None:
        if self.busy:
            return
        path = self.app.filedialog.askopenfilename(parent=self.window, title="选择 STL 或 DXF 管道模型", filetypes=(("STL/DXF", "*.stl *.dxf"), ("STL", "*.stl"), ("DXF", "*.dxf")))
        if path:
            try:
                self.load_model(Path(path))
            except (OSError, ValueError) as error:
                self.show_error(error)

    def load_model(self, path: Path) -> None:
        suffix = path.suffix.lower()
        if suffix == ".dxf":
            from .dxf_elevation import catalog_from_dxf
            pipes, skipped = catalog_from_dxf(path, axis_world=(0.0, 0.0, 1.0),
                                              unitless_unit=self.stl_unit.get())
            self.model_kind = "dxf"
            self.model_half_length_mm = 1000.0
        elif suffix == ".stl":
            from .capture_gui import catalog_from_model
            pipes, skipped = catalog_from_model(path, stl_unit=self.stl_unit.get())
            self.model_kind = "stl"
        else:
            raise ValueError("模型必须是 STL 或 DXF 文件")
        if not pipes:
            raise ValueError("模型没有可识别的独立直管组件")
        self.fields["model"].set(str(path.resolve()))
        diameters = sorted({round(row["nominal_diameter_mm"], 1) for row in pipes})
        for row in pipes:
            if suffix == ".stl":
                row["color_srgb"] = PALETTE[diameters.index(round(row["nominal_diameter_mm"], 1)) % len(PALETTE)]
                row["color_source"] = "stl_synthetic_by_diameter"
            else:
                row["color_source"] = str(row.get("color_source") or "dxf")
            row["axis"] = "auto"
        self.pipes = pipes
        self.status_refresh_metadata = None
        self._analyze_after_camera_capture = False
        self.registration_settings = {"axis_world": [0.0, 0.0, 1.0] if suffix == ".dxf" else None,
                                      "anchors": {}, "local_observation_mode": "auto"}
        self._restore_camera_side_control()
        self.zone_settings = normalize_zone_settings()
        self.scope_label.set("全幅自动")
        self.local_observation_label.set("自动（圆柱/图像引导）")
        self.last_manifest = None
        self.confirmed.set(False)
        self.invalidate()
        self.refresh_table()
        skipped_count = len(skipped)
        self.summary.set(f"识别 {len(pipes)} 根管；未纳入 {skipped_count} 个图元。" + (" DXF 对象/图层颜色已保留；请核对共同管长方向。" if suffix == ".dxf" else " STL 无标准颜色；当前颜色按管径生成，启用筛选前请按管径确认现场颜色。"))

    def edit_pipe(self) -> None:
        spec = self.selected_pipe()
        if self.busy or spec is None:
            return
        tk, ttk = self.app.tk, self.app.ttk
        dialog = tk.Toplevel(self.window)
        dialog.title(f"{spec['pipe_id']} · 管道配置")
        dialog.transient(self.window)
        box = ttk.Frame(dialog, padding=15)
        box.pack(fill="both", expand=True)
        fields = {}
        fields_spec = (("color_srgb", "目标颜色 #RRGGBB", spec["color_srgb"]),)
        for i, (key, label, value) in enumerate(fields_spec):
            fields[key] = tk.StringVar(value=str(value))
            ttk.Label(box, text=label).grid(row=i, column=0, sticky="w", pady=5)
            ttk.Entry(box, textvariable=fields[key], width=24).grid(row=i, column=1, pady=5)
        def choose_color() -> None:
            from tkinter import colorchooser
            color = colorchooser.askcolor(fields["color_srgb"].get(), parent=dialog)[1]
            if color:
                fields["color_srgb"].set(color.upper())
        ttk.Button(box, text="选颜色", command=choose_color).grid(row=0, column=2, padx=6)
        ttk.Label(box, text="模型管径、身份和照片区域由 STL/DXF 与双目局部点云确定；此处只调整显示颜色。").grid(row=len(fields_spec), column=0, columnspan=3, pady=9)
        def apply() -> None:
            import re
            try:
                if self.busy:
                    raise ValueError("正在分析，完成后再修改配置")
                diameter = float(spec["nominal_diameter_mm"])
                color = fields["color_srgb"].get().strip().upper()
                if not math.isfinite(diameter) or diameter <= 0 or not re.fullmatch(r"#[0-9A-F]{6}", color):
                    raise ValueError("请输入正数管径和 #RRGGBB 颜色")
                spec.update(nominal_diameter_mm=diameter, color_srgb=color, color_source="user")
                self._clear_registration_anchors()
                self.invalidate()
                self.refresh_table()
                dialog.destroy()
            except ValueError as error:
                self.show_error(error)
        ttk.Button(box, text="应用", command=apply).grid(row=1, column=2)

    def apply_diameter_color(self) -> None:
        if self.busy:
            return
        spec = self.selected_pipe()
        if spec:
            for row in self.pipes:
                if abs(row["nominal_diameter_mm"] - spec["nominal_diameter_mm"]) <= 0.2:
                    row["color_srgb"] = spec["color_srgb"]
                    row["color_source"] = "user"
            self._clear_registration_anchors()
            self.invalidate()
            self.refresh_table()

    def show_catalog(self) -> None:
        self.open_model_viewer()
        if self.model_viewer is not None and self.model_viewer.window.winfo_exists():
            self.model_viewer.focus_section()

    def _restore_camera_side_control(self) -> None:
        label = camera_side_label(self.registration_settings.get("camera_side_world"))
        self.camera_side_label.set(label)
        self.camera_side_box.configure(values=tuple(dict.fromkeys((*CAMERA_SIDE_PRESETS, label))))

    def _camera_side_selected(self) -> None:
        label = self.camera_side_label.get()
        if label in CAMERA_SIDE_PRESETS:
            self._set_camera_side(CAMERA_SIDE_PRESETS[label])

    def _set_camera_side(self, side: list[float] | None) -> None:
        if self.busy:
            self._restore_camera_side_control()
            return
        side = normalize_camera_side(side)
        if side == self.registration_settings.get("camera_side_world"):
            return
        live_viewer = (self.model_viewer if self.model_viewer is not None
                       and self.model_viewer.window.winfo_exists() and self.model_viewer._is_current() else None)
        if side is None:
            self.registration_settings.pop("camera_side_world", None)
        else:
            self.registration_settings["camera_side_world"] = side
        self._restore_camera_side_control()
        self._clear_registration_anchors()
        self.invalidate()
        if live_viewer is not None:
            live_viewer.sync_from_owner()
        self.message.set(f"相机观察方向：{camera_side_label(side)}。改变观察侧后需重新评估；下次保存会开始新历史。")

    def open_model_viewer(self) -> None:
        if self.busy:
            return
        if not self.pipes:
            self.message.set("请先导入 STL 或 DXF，再设置共同管长方向。")
            return
        from .elevation_viewer import ElevationModelViewer
        if self.model_viewer is not None and self.model_viewer.window.winfo_exists():
            if self.model_viewer._is_current():
                self.model_viewer.window.lift(); return
            self.model_viewer.close()
        path = self.fields["model"].get().strip()
        if not path:
            self.message.set("请先导入 STL 或 DXF，再设置共同管长方向。")
            return
        self.model_viewer = ElevationModelViewer(self, model_path=Path(path), pipes=self.pipes,
                                                 axis_world=self.registration_settings.get("axis_world"),
                                                 camera_side_world=self.registration_settings.get("camera_side_world"),
                                                 report=self.report,
                                                 on_camera_side=self._set_camera_side,
                                                 on_axis=self._set_axis_world)

    def _set_axis_world(self, axis: list[float]) -> None:
        live_viewer = (self.model_viewer if self.model_viewer is not None
                       and self.model_viewer.window.winfo_exists() and self.model_viewer._is_current() else None)
        vector = np.asarray(axis, dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-9 or not np.all(np.isfinite(vector)):
            self.show_error(ValueError("管长方向必须是有限的非零三维向量")); return
        axis = vector / norm
        if self.model_kind == "dxf" and not np.allclose(np.abs(axis), [0.0, 0.0, 1.0], atol=1e-9, rtol=0):
            self.show_error(ValueError("当前 DXF 是 XY 平面圆形管道布置，管轴应保持模型 ±Z；俯视角度由双目配准估计。")); return
        if self.model_kind == "dxf":
            # DXF stores cross-section centres in XY and has no 3-D pipe
            # endpoints.  Keep the centre fixed while changing only the
            # common axis used to construct the registration centreline.
            for pipe in self.pipes:
                line = np.asarray(pipe.get("centerline_world_mm"), dtype=float)
                if line.shape == (2, 3):
                    center = line.mean(axis=0)
                    pipe["centerline_world_mm"] = np.stack((center - axis * self.model_half_length_mm,
                                                               center + axis * self.model_half_length_mm)).tolist()
        self.registration_settings["axis_world"] = axis.tolist()
        self.registration_settings["anchors"] = {}
        self.invalidate()
        if live_viewer is not None:
            live_viewer.sync_from_owner()
        self.message.set(f"已保存共同管长方向：{np.round(axis, 4).tolist()}；下一次评估将自动匹配 {self.model_kind.upper() or '模型'}。")

    def _clear_registration_anchors(self) -> None:
        self.registration_settings["anchors"] = {}

    def browse_calibration(self) -> None:
        if self.busy:
            return
        path = self.app.filedialog.askopenfilename(parent=self.window, title="选择真实相机标定", filetypes=(("JSON", "*.json"),))
        if path:
            try:
                from .capture_gui import load_calibration_json, field_calibration_problem
                calibration = load_calibration_json(path)
                problem = field_calibration_problem(calibration)
                if problem:
                    raise ValueError(problem)
                self.fields["calibration"].set(path)
                self.calibration_override = calibration
                bundle = json.loads(Path(path).read_text(encoding="utf-8-sig"))
                self.profile = dict(self.profile or {}, rectification_recipe=bundle.get("rectification_recipe"))
                self.refresh_calibration_status()
            except (OSError, ValueError) as error:
                self.show_error(error)

    def current_calibration(self) -> dict:
        from .capture_gui import load_calibration_json, field_calibration_problem
        calibration = self.calibration_override or load_calibration_json(self.fields["calibration"].get())
        problem = field_calibration_problem(calibration)
        if problem:
            raise ValueError(problem)
        return calibration

    def refresh_calibration_status(self) -> None:
        try:
            from .stereo_analyzer import _calibration_from_manifest
            calibration = self.current_calibration()
            parsed = _calibration_from_manifest(calibration)
            from .calibration_quality import provisional_calibration_allowed
            if (parsed.validated or provisional_calibration_allowed(calibration)) and parsed.rectified:
                registration = (
                    "CAD配准已确认"
                    if parsed.registration_validated
                    else "CAD配准未确认（立面相对布局仍可继续；完整绝对坐标需另行配准）"
                )
                from .calibration_quality import calibration_quality_text
                self.calibration_status.set(
                    f"已加载标定 · {parsed.calibration_id} · "
                    f"{parsed.left.width}×{parsed.left.height} 每目 · 基线 {parsed.baseline_mm:g} mm · {registration}\n"
                    + calibration_quality_text(calibration))
            else:
                self.calibration_status.set("标定文件已载入，但尚未满足现场双目门禁；请完成一次标定向导")
        except (OSError, ValueError) as error:
            self.calibration_status.set(str(error))

    def open_calibration(self) -> None:
        if not self.busy:
            if self.camera_dialog is not None and self.camera_dialog.window.winfo_exists():
                self.camera_dialog.close()
            from .calibration_wizard import ChessboardWizardDialog
            if self.calibration_dialog is not None and self.calibration_dialog.window.winfo_exists():
                self.calibration_dialog.window.lift()
            else:
                self.calibration_dialog = ChessboardWizardDialog(self.app, owner=self)

    def _persist_profile(self, sections: tuple, *, camera: dict | None = None, **_kwargs: Any) -> None:
        # Reuse device selection; the elevation scene itself is stored in its
        # portable manifest rather than in the global camera profile.
        if camera is not None:
            from .workbench_profile import update_profile
            update_profile({"camera": camera})
            self.profile = dict(self.profile or {}, camera=copy.deepcopy(camera))

    def capture_camera(self, *, analyze_after_capture: bool = False) -> None:
        if self.busy:
            return
        try:
            from .calibration_wizard import rectifier_for_calibration
            from .capture_gui import StereoCameraDialog
            calibration = self.current_calibration()
            rectifier = rectifier_for_calibration(calibration, self.profile)
            if self.calibration_dialog is not None and self.calibration_dialog.window.winfo_exists():
                # The wizard archives accepted pairs. Release its UVC session
                # (including an in-flight open) before opening measurement.
                self.calibration_dialog.close()
            self._analyze_after_camera_capture = bool(analyze_after_capture)
            if self.camera_dialog is not None and self.camera_dialog.window.winfo_exists():
                self.camera_dialog.on_capture = self._camera_capture_finished
                self.camera_dialog.window.lift()
            else:
                self.camera_dialog = StereoCameraDialog(self, calibration, rectifier, on_capture=self._camera_capture_finished)
        except (OSError, ValueError) as error:
            self.show_error(error)

    def _camera_capture_finished(self, _paths: Any, _pair: Any) -> None:
        """Adopt a fresh pair, optionally starting a status refresh analysis."""
        self.load_images()
        if getattr(self, "_analyze_after_camera_capture", False):
            self._analyze_after_camera_capture = False
            # StereoCameraDialog closes itself immediately after this callback;
            # defer submit so the camera is released before depth computation.
            self.window.after(50, lambda: self.submit(True))

    def refresh_status(self) -> None:
        """Capture a new physical scene and append a timestamped match run."""
        if self.busy:
            return
        self.status_refresh_metadata = {
            "action": "STATUS_REFRESH",
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }
        self.message.set("状态刷新已请求：请等待相机预热并完成新的左右目抓拍，随后自动重新匹配。")
        self.capture_camera(analyze_after_capture=True)

    def browse_photo(self, role: str) -> None:
        if self.busy:
            return
        path = self.app.filedialog.askopenfilename(parent=self.window, title="选择已矫正照片", filetypes=(("图像", "*.png *.jpg *.jpeg"),))
        if path:
            try:
                from .capture_gui import photo_file_time
                capture_time = photo_file_time(path)
                self.status_refresh_metadata = None
                self.fields[role].set(path)
                self.fields[f"{role}_time"].set(capture_time)
                self.timestamp_sources[role] = "MANIFEST_OPERATOR_CONFIRMED"
                self.camera_capture_provenance = {}
                self.load_images()
            except (OSError, ValueError) as error:
                self.show_error(error)

    def load_images(self) -> None:
        for role, view in self.views.items():
            path = self.fields[role].get().strip()
            if not path:
                view.set_image(None)
                self.image_paths.pop(role, None)
                self.image_hashes.pop(role, None)
                continue
            data = Path(path).read_bytes()
            image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                view.set_image(None)
                self.image_paths.pop(role, None)
                self.image_hashes.pop(role, None)
                raise ValueError(f"无法读取 {role} 图像")
            if view.image is not None and view.image.shape != image.shape:
                self.last_manifest = None
                self.zone_settings = normalize_zone_settings()
                self.scope_label.set("全幅自动")
                self.message.set("照片尺寸改变，评估区域已清除；请载入与标定尺寸一致的左右图并重新框选。")
            digest = hashlib.sha256(data).hexdigest()
            if self.image_hashes.get(role) not in (None, digest):
                # Observation anchors refer to one exact capture.  They must
                # never silently carry over to a new pair.
                self.registration_settings["anchors"] = {}
            self.image_paths[role] = str(Path(path).resolve())
            self.image_hashes[role] = digest
            view.set_image(image)
        self.invalidate()
        self.refresh_table()
        if self.quality_dialog is not None and not self.quality_dialog.closed:
            self.quality_dialog.refresh_inputs()

    def browse_session(self) -> None:
        if self.busy:
            return
        path = self.app.filedialog.askopenfilename(parent=self.window, title="打开已保存的基础现场", filetypes=(("基础现场 JSON", "*.json"),))
        if path:
            try:
                self.load_session(Path(path))
            except (OSError, ValueError) as error:
                self.show_error(error)

    def load_session(self, path: Path) -> None:
        from .elevation_dataset import load_elevation_dataset
        path = Path(path).resolve()
        if path.is_dir():
            path /= "manifest.json"
        self.invalidate()
        self.status_refresh_metadata = None
        self._analyze_after_camera_capture = False
        loaded = load_elevation_dataset(path)
        manifest = loaded["manifest"]
        from .stereo_analyzer import _analysis_config
        self.analysis_settings = {k: v for k, v in (manifest.get("analysis") or {}).items()
                                  if k not in {"mode", "elevation_auto"}}
        normalized_analysis = _analysis_config(self.analysis_settings)
        stereo_matching = normalized_analysis["stereo_matching"]
        self.matching_preset.set("弱光降噪" if stereo_matching["preprocessing"] == "low_light" else "原始灰度")
        self.disparity_count.set(str(stereo_matching["num_disparities"]))
        self._restore_matching_settings(normalized_analysis["matching"])
        loaded_mode = str(loaded.get("mode") or (manifest.get("analysis") or {}).get("mode") or "")
        if loaded_mode != "elevation_auto":
            raise ValueError("当前版本只支持 DXF/STL 自动立面现场；请重新导入模型并建立现场")
        restored_registration = copy.deepcopy(loaded.get("registration_settings") or {
            "axis_world": None, "anchors": {}, "local_observation_mode": "auto"})
        # Field traces and image loading can invalidate anchors.  Keep the
        # verified package settings aside and restore them only after all
        # snapshot checks have completed below.
        self.registration_settings = {"axis_world": None, "anchors": {}, "local_observation_mode": "auto"}
        self.pipes = []
        self.fields["calibration"].set(str(path))
        self.calibration_override = loaded["calibration"]
        self.profile = dict(self.profile or {}, rectification_recipe=loaded.get("rectification_recipe"))
        for key in ("left", "right"):
            self.fields[key].set(str(loaded[f"{key}_path"]))
            self.fields[f"{key}_time"].set(loaded[f"{key}_time"])
        self.stl_unit.set(manifest.get("model", {}).get("source_unit", "millimeter"))
        model = manifest.get("model", {}).get("path")
        self.fields["model"].set(str(path.parent / model) if model else "")
        self.model_kind = Path(str(model)).suffix.lower().lstrip(".") if model else ""
        self.model_half_length_mm = float((manifest.get("model") or {}).get("pipe_half_length_mm", 1000.0))
        # Reset images before loading so old image dimensions cannot discard
        # regions already verified by the dataset loader.
        for view in self.views.values():
            view.set_image(None)
        self.pipes = copy.deepcopy(loaded["pipe_specs"])
        self.last_manifest = path.resolve()
        self.camera_capture_provenance = {}
        self.image_paths.clear(); self.image_hashes.clear()
        views = manifest["capture"]["capture_groups"][-1]["views"]
        self.timestamp_sources = {role: views[role]["timestamp_source"] for role in ("left", "right")}
        self.camera_capture_provenance = {role: {k: v for k, v in views[role].items()
             if k.startswith("capture_") or k == "side_by_side_order"} for role in ("left", "right")}
        self.load_images()
        self.temporal_frames = [
            {**{f"{role}_path": str(path.parent / frame["views"][role]["path"]) for role in ("left", "right")},
             **{f"{role}_time": frame["views"][role]["captured_at"] for role in ("left", "right")},
             "sha256": {role: frame["views"][role]["sha256"] for role in ("left", "right")},
             "timestamp_sources": {role: frame["views"][role]["timestamp_source"] for role in ("left", "right")},
             "camera_capture_provenance": {role: {k: v for k, v in frame["views"][role].items()
                   if k.startswith("capture_") or k == "side_by_side_order"} for role in ("left", "right")}}
            for frame in manifest["capture"]["capture_groups"][-1].get("temporal_frames", [])]
        self._restore_zone_settings(loaded.get("scope_settings"))
        self.registration_settings = restored_registration
        self._restore_camera_side_control()
        local_labels = {
            "auto": "自动（圆柱/图像引导）",
            "geometry_only": "仅双目深度几何",
            "image_guided": "图像引导圆柱",
            "parallel_strip": "平行局部条带（颜色辅助）",
        }
        self.local_observation_label.set(local_labels.get(
            restored_registration.get("local_observation_mode", "auto"), "自动（圆柱/图像引导）"))
        self.present_pipe_count.set(str(restored_registration.get("present_pipe_count", "")))
        self.confirmed.set(False)
        self.refresh_table()
        self.refresh_calibration_status()
        self.summary.set(f"已恢复 {len(self.pipes)} 根管；自动匹配会重新生成照片区域")
        self.message.set(f"已打开 {path}。可直接重新分析，自动匹配 {self.model_kind.upper() or '模型'}。")

    def submit(self, analyze: bool) -> None:
        if self.busy:
            return
        try:
            calibration = copy.deepcopy(self.current_calibration())
            if not self.confirmed.get():
                raise ValueError("请确认左右图来自同一次同步拍摄，且相机机位未改变")
            if not self.pipes:
                raise ValueError("请先导入 STL 或 DXF 管道目录")
            left, right = self.views["left"].image, self.views["right"].image
            if left is None or right is None:
                raise ValueError("请先载入完整的左右照片或完成一次双目抓拍")
            if left.shape != right.shape:
                raise ValueError("左右照片尺寸不一致，请重新载入同一组矫正照片")
            for role in ("left", "right"):
                if self.image_paths.get(role) != str(Path(self.fields[role].get()).resolve()):
                    raise ValueError("照片尚未正确载入预览，请重新选择")
                if hashlib.sha256(Path(self.fields[role].get()).read_bytes()).hexdigest() != self.image_hashes.get(role):
                    self.invalidate()
                    raise ValueError("照片文件在预览后已改变，请重新载入并核对区域")
            mode = "elevation_auto"
            if not self.fields["model"].get().strip():
                raise ValueError("自动匹配模式必须提供 STL 或 DXF 模型")
            from .stereo_analyzer import _analysis_config
            settings = copy.deepcopy(self.analysis_settings)
            try:
                count = int(self.disparity_count.get())
            except ValueError as error:
                raise ValueError("视差搜索范围需为 16 的正整数倍") from error
            settings["stereo_matching"] = {**settings.get("stereo_matching", {}),
                "num_disparities": count,
                "preprocessing": "low_light" if self.matching_preset.get() == "弱光降噪" else "none"}
            settings["matching"] = self._matching_settings_from_controls()
            settings = _analysis_config(settings)
            if count + settings["stereo_matching"]["min_disparity"] >= left.shape[1]:
                raise ValueError("视差搜索范围必须小于每目图像宽度")
            scope_settings = normalize_zone_settings(
                self.zone_settings, image_size=(left.shape[1], left.shape[0]))
            arguments = dict(output_root=self.output_root / "captures", calibration=calibration,
                analysis_settings=settings,
                left_path=Path(self.fields["left"].get()), right_path=Path(self.fields["right"].get()),
                left_time=self.fields["left_time"].get(), right_time=self.fields["right_time"].get(),
                pipe_specs=copy.deepcopy(self.pipes), pair_confirmed=True,
                previous_manifest=self.last_manifest if self.history.get() else None,
                model_path=Path(self.fields["model"].get()) if self.fields["model"].get() else None,
                stl_unit=self.stl_unit.get(), timestamp_sources=copy.deepcopy(self.timestamp_sources),
                rectification_recipe=copy.deepcopy((self.profile or {}).get("rectification_recipe")),
                camera_capture_provenance=copy.deepcopy(self.camera_capture_provenance),
                status_refresh=copy.deepcopy(self.status_refresh_metadata),
                temporal_frames=copy.deepcopy(getattr(self, "temporal_frames", [])),
                scope_settings=scope_settings,
                _preview_hashes=copy.deepcopy(self.image_hashes))
            registration = copy.deepcopy(self.registration_settings)
            count_text = self.present_pipe_count.get().strip()
            registration.pop("present_pipe_count", None)
            if count_text:
                try:
                    registration["present_pipe_count"] = int(count_text)
                except ValueError as error:
                    raise ValueError("现场实际管数需为正整数") from error
            from .elevation_dataset import normalize_registration_settings
            arguments.update(mode=mode, registration_settings=normalize_registration_settings(registration))
            self.invalidate()
            self.busy = True
            self._set_controls(False)
            generation = self.generation
            log_event(LOGGER, "elevation_gui_run_start", analyze=analyze, pipe_count=len(self.pipes))
            self.message.set("正在保存并分析双目深度……" if analyze else "正在保存基础现场……")
            threading.Thread(target=self._worker, args=(arguments, analyze, generation), daemon=True).start()
        except (OSError, ValueError) as error:
            self.show_error(error)

    def _worker(self, arguments: dict, analyze: bool, generation: int) -> None:
        try:
            from .elevation_dataset import create_elevation_dataset, elevation_history_compatible, load_elevation_dataset
            expected_hashes = arguments.pop("_preview_hashes")
            previous = arguments["previous_manifest"]
            history_kwargs = dict(calibration=arguments["calibration"], pipe_specs=arguments["pipe_specs"], model_path=arguments["model_path"], stl_unit=arguments["stl_unit"], rectification_recipe=arguments["rectification_recipe"], analysis_settings=arguments["analysis_settings"], scope_settings=arguments["scope_settings"])
            history_kwargs.update(mode="elevation_auto", registration_settings=arguments.get("registration_settings"))
            reset_history = bool(previous and not elevation_history_compatible(previous, **history_kwargs))
            if reset_history:
                arguments["previous_manifest"] = None
            path = create_elevation_dataset(**arguments)
            loaded = load_elevation_dataset(path)
            latest = loaded["manifest"]["capture"]["capture_groups"][-1]["views"]
            if any(latest[role]["sha256"] != expected_hashes[role] for role in ("left", "right")):
                raise ValueError("保存期间照片发生变化，请重新载入")
            from .stereo_analyzer import analyze_stereo_capture
            report = analyze_stereo_capture(path, report_output_path=path.parent / "report.json", evidence_dir=path.parent / "evidence",
                progress=lambda text: self.messages.put((generation, "progress", text))) if analyze else None
            self.messages.put((generation, "ok", (path, report, reset_history)))
        except Exception as error:
            log_event(LOGGER, "elevation_gui_run_failed", error=str(error), saved_manifest=str(locals().get("path", "")))
            if "path" in locals():
                error = ValueError(f"{error}\n现场包已保存：{path}；可打开该现场重试分析。")
            self.messages.put((generation, "error", error))

    def _set_controls(self, enabled: bool) -> None:
        def visit(widget: Any) -> None:
            for child in widget.winfo_children():
                if hasattr(child, "state"):
                    child.state(["!disabled"] if enabled else ["disabled"])
                visit(child)
        visit(self.controls)
        for button in (self.run_button, self.save_button):
            button.configure(state="normal" if enabled else "disabled")

    def poll(self) -> None:
        if self.closed:
            return
        try:
            generation, kind, payload = self.messages.get_nowait()
        except queue.Empty:
            self.poll_id = self.window.after(100, self.poll)
            return
        if kind == "progress":
            if generation == self.generation:
                self.message.set(payload)
            self.poll_id = self.window.after(100, self.poll)
            return
        self.busy = False
        self._set_controls(True)
        if generation != self.generation:
            self.invalidate()
            self.message.set("输入在计算时发生改变；已丢弃旧结果，请重新评估。")
        elif kind == "error":
            self.invalidate()
            self.show_error(payload)
        else:
            path, report, reset_history = payload
            refresh_metadata = copy.deepcopy(self.status_refresh_metadata)
            self.status_refresh_metadata = None
            self.last_manifest = path
            self.report = report
            self.results = {row["pipe_id"]: row for row in report["pipes"]} if report else {}
            self.refresh_table()
            counts = report["counts"] if report else {}
            self.summary.set(f"安装 {counts.get('INSTALLED', 0)} · 未安装 {counts.get('NOT_INSTALLED', 0)} · 不确定 {counts.get('UNKNOWN', 0)}" if report else "基础现场已保存")
            inventory = report.get("scene_inventory", {}) if report else {}
            if inventory.get("present_pipe_count") is not None:
                self.summary.set(f"现场实物 {inventory['present_pipe_count']} 根；模型候选 {inventory['model_candidate_count']} 个管位 · 已对应 {counts.get('INSTALLED', 0)} · 待确认 {counts.get('UNKNOWN', 0)}")
            if report:
                self.summary.set(self.summary.get()+"\n"+recognition_summary(report))
                temporal = report.get("temporal_fusion")
                if temporal:
                    status = "已融合" if temporal["status"] == "VALID" else "未通过"
                    self.summary.set(self.summary.get()+f" · 多帧 {temporal['accepted_frames']}/{temporal['requested_frames']} 对（{status}）")
            self.message.set(f"已保存：{path.parent}。JSON 报告和叠图在同一目录。" if report else f"已保存：{path}")
            if report and (refresh_metadata or report.get("status_refresh")):
                recorded = report.get("status_refresh") or refresh_metadata or {}
                audits = report.get("capture_audit", {}).get("groups", [])
                captured_at = audits[-1].get("captured_at") if audits else None
                self.message.set(
                    f"状态刷新完成：请求 {recorded.get('requested_at', '未记录')}；"
                    f"拍摄 {captured_at or '未记录'}；分析 {report.get('generated_at', '未记录')}。"
                )
            if report:
                audits = report.get("capture_audit", {}).get("groups", [])
                latest = audits[-1] if audits else {}
                warnings = set(latest.get("depth_audit", {}).get("warning_codes", []))
                for quality in (latest.get("quality") or {}).values():
                    warnings.update(quality.get("warning_codes", []))
                if warnings:
                    self.message.set(self.message.get() + " " + "；".join(REASON_TEXT.get(w, w) for w in sorted(warnings)))
            if reset_history:
                self.message.set(self.message.get() + " 配置已改变，已开始新历史。")
            log_event(LOGGER, "elevation_gui_run_finished", manifest=str(path), counts=counts, history_reset=reset_history)
            try:
                atomic_write_text(self.output_root / "last_scene.json", json.dumps({"manifest_path": str(path)}, ensure_ascii=False))
            except OSError as error:
                self.message.set(f"现场已保存，但无法记录恢复位置：{error}")
        self.poll_id = self.window.after(100, self.poll)

    def show_error(self, error: Exception) -> None:
        self.message.set(str(error))
        self.app.messagebox.showerror("基础立面评估", str(error), parent=self.window)

    def close(self) -> None:
        self.closed = True
        for dialog in (self.camera_dialog, self.calibration_dialog, self.model_viewer, self.quality_dialog):
            if dialog is not None and dialog.window.winfo_exists():
                dialog.close()
        if getattr(self, "poll_id", None):
            self.window.after_cancel(self.poll_id)
        self.window.destroy()


__all__ = ["ElevationCaptureDialog", "RegionCanvas", "image_region"]
