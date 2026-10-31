"""Pure presentation helpers for the elevation workflow.

This module deliberately has no Tk dependency.  It formats recognition and
measurement evidence for the GUI while keeping the window/controller module
focused on state and user interaction.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


REASON_TEXT = {
    "BURST_IMAGE_SIZE_CHANGED": "序列中图像尺寸改变",
    "BURST_EXPOSURE_UNSTABLE": "曝光变化过大，该帧已排除",
    "BURST_MOTION_UNVERIFIABLE": "可跟踪纹理不足，无法确认相机和场景静止",
    "BURST_SCENE_MOVED": "图像变化检查未通过；可能涉及运动、曝光、背景或反光变化，不能仅据此认定管道移动",
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
    candidates = surface.get("image_candidates", [])
    counts = Counter(c.get("role") for c in candidates)
    text = f"图像候选：左 {counts['left']} / 右 {counts['right']}；合格三维管段 {len(surface.get('observations', []))}"
    reasons = Counter(r.get("reason") for r in surface.get("rejected", []))
    if reasons:
        text += "；" + "、".join(
            f"{REASON_TEXT.get(k, k)} {v} 次" for k, v in reasons.most_common(2)
        )
    from .section_display import pooling_summary

    pooled = pooling_summary(report)
    if pooled:
        text += ";" + pooled
    return text


def recognition_diagnostic_lines(report: dict | None) -> list[str]:
    """Separate image similarity, measured depth support and cylinder gates."""
    lines = [
        "目标是共同断面的管径、轴心间距和净距，不测量整根管长。",
        "断面附近保留小段采样带，用于估计管轴方向和验证圆弧；不是整根管道长度要求。",
        "图像候选不是实物管数；重叠分区会重复列出同一管面。",
        "图像唯一匹配比例来自有限抽样，仅作诊断，不代表测量合格率。",
        "高光 H 只统计多通道近饱和像素，不能覆盖全部反光。",
    ]
    if not report:
        return lines + ["", "请先点击⑤保存并评估。"]
    lines += ["", recognition_summary(report)]
    temporal = report.get("temporal_fusion")
    if temporal:
        status = "已融合" if temporal.get("status") == "VALID" else "未通过"
        lines += [
            "",
            f"多帧融合：{status}，有效 {temporal.get('accepted_frames', 0)}/{temporal.get('requested_frames', 0)} 对",
            "本序列只算一次评估；时间稳定性不代表绝对尺寸准确。",
        ]
        for role, label in (("left", "左"), ("right", "右")):
            value = temporal.get(role) or {}
            lines.append(
                f"{label}目：每点至少 {value.get('required_support_frames', '—')} 帧支持；"
                f"视差波动 MAD 中位数 {value.get('median_disparity_mad_px', '—')} px"
            )
            if value.get("majority_cluster_recovered_pixels") is not None:
                lines.append(
                    f"  时间检查重新找回 {value['majority_cluster_recovered_pixels']} 个像素；"
                    f"支持帧不足 {value.get('insufficient_observation_pixels', 0)} 个，"
                    f"匹配仍不稳定 {value.get('inconsistent_observation_pixels', 0)} 个"
                    "（左右复核前的全图统计，包含背景，不代表管面合格点数）。"
                )
            if value.get("post_consistency_valid_fraction") is not None:
                lines.append(
                    f"  全图有效深度：时间检查后 {100*value['pre_consistency_valid_fraction']:.1f}% → "
                    f"左右复核后 {100*value['post_consistency_valid_fraction']:.1f}%。"
                )
        reasons = list(temporal.get("reason_codes", []))
        reasons.extend(
            code
            for frame in temporal.get("frames", [])
            for code in frame.get("reason_codes", [])
        )
        if reasons:
            lines.append("序列提示：" + "；".join(REASON_TEXT.get(code, code) for code in dict.fromkeys(reasons)))
        frames = temporal.get("frames") or []
        if frames and all(
            all((frame.get("motion") or {}).get(role, {}).get("passed") for role in ("left", "right"))
            for frame in frames
        ):
            lines.append("本序列通过静止检查；管面深度不一致是匹配证据问题，不表示设备或管道发生移动。")
        else:
            lines.append("运动/曝光不稳定时保持机位、待画面稳定后重新连续采集；纹理不足时可能无法确认静止。")
    else:
        lines.append("本次使用单对图像，未进行多帧融合。")

    def percent(value):
        return "未计算" if value is None else f"{100*value:.1f}%"

    for candidate in (report.get("local_surface") or {}).get("image_candidates", []):
        eye = "左目" if candidate.get("role") == "left" else "右目"
        lines += [
            "",
            f"{candidate.get('candidate_id')} · {eye} · 区域 {candidate.get('region_px')}",
            f"用于拟合的有效深度：{percent(candidate.get('valid_depth_fraction'))}",
        ]
        diagnostic = candidate.get("correspondence_diagnostic") or {}
        surface_texture = diagnostic.get("surface_texture") or {}
        if surface_texture:
            lines.append(
                f"全候选管面弱纹理比例：{percent(surface_texture.get('low_texture_fraction'))}"
                f"（内部抽样 {surface_texture.get('sample_count', 0)} 点，包含没有深度的区域）"
            )
        angle = candidate.get("axis_to_epipolar_deg")
        if angle is not None:
            lines.append(f"管面长方向与视差方向夹角：{angle:.1f}°（0° 为平行）")
        lines.append(
            f"图像唯一匹配：{percent(diagnostic.get('sampled_unique_match_fraction'))}"
            f"（抽样 {diagnostic.get('comparison_count', 0)} 点）；高光 H：{percent(diagnostic.get('highlight_fraction'))}"
        )
        warnings = diagnostic.get("warning_codes", [])
        if warnings:
            lines.append("图像提示：" + "；".join(REASON_TEXT.get(code, code) for code in warnings))
        for attempt in candidate.get("attempts", []):
            label = "整段" if attempt["window_id"] == "FULL" else f"局部 {attempt['window_id']}"
            if attempt.get("fit_count"):
                lines.append(f"{label}：单眼拟合通过，仍需左右共同区间验证及模型匹配")
            for rejected in attempt.get("rejections", []):
                text = f"{label}：{REASON_TEXT.get(rejected.get('reason'), rejected.get('reason'))}"
                trend = rejected.get("disparity_filter") or {}
                if trend.get("retained_fraction") is not None:
                    text += f"；主导深度支持 {percent(trend['retained_fraction'])}"
                fit = rejected.get("fit_diagnostics") or {}
                if fit.get("fit_rms_mm") is not None:
                    text += f"；拟合误差 {fit['fit_rms_mm']:.2f} mm / 上限 {fit['maximum_fit_rms_mm']:.2f} mm"
                if fit.get("minimum_axial_support_mm") is not None:
                    text += (
                        f"；局部轴向采样 {fit['axial_support_mm']:.1f} mm / "
                        f"本方法需 {fit['minimum_axial_support_mm']:.1f} mm（非实物管长）"
                    )
                lines.append(text)
        if not candidate.get("attempts"):
            lines.append("旧报告未记录局部分段诊断；重新评估可生成。")
    lines += [
        "",
        "纹理不足或匹配多解：用少量可移除的哑光随机纹理做对照；高光变化不等于深度改善。",
        "光滑管道接近视差方向时，可调整整台双目相机的安装朝向再取样；仅旋转显示图像无效。",
        "取景应包含需要对比的多根管道，避开前景木条，并在图像和框选边界保留完整管宽。",
        "深度支持足够但拟合仍失败：继续检查双目矫正、深度误差及圆柱形状，不宜放宽尺寸容差。",
    ]
    return lines


def relative_layout_summary_lines(report: dict | None) -> list[str]:
    lines = [
        "只比较共同断面：各管直径、轴心间距、表面净距及相对排列，不要求整根管长。",
        "相机到管道的距离只用于三维重建，不作为模型匹配目标。",
        "以下净距为共同管轴法向截面的轴心距减两半径；负值表示截面重叠。",
    ]
    registration = (report or {}).get("registration") or {}
    registrations = ((report or {}).get("zone_registration_audit") or {}).get("registrations") or [registration]
    if registration.get("status") in {"ZONE_MATCH_CONFLICT", "ZONE_RESULT_CONFLICT"}:
        registrations = []
    available = False
    for reg in registrations:
        layout = reg.get("relative_layout")
        if reg.get("status") != "MATCHED" or not layout:
            continue
        available = True
        if not layout.get("axis_centers_resolved", True):
            lines += ["", "本组使用局部条带估计，圆柱轴心未确认，暂不输出轴心距或表面净距对比。"]
            continue
        lines += ["", f"{reg.get('zone_id', '全幅')} · 管径（模型 / 实测 / 偏差，mm）"]
        for item in layout.get("matches", []):
            if item.get("observed_diameter_mm") is not None:
                lines.append(
                    f"{item['pipe_id']}：{item['model_diameter_mm']:.2f} / "
                    f"{item['observed_diameter_mm']:.2f} / {item['diameter_error_mm']:+.2f}"
                )
        lines.append("管间关系（模型 / 实测 / 偏差，mm）")
        for pair in layout.get("pairwise", []):
            lines.append(
                f"{pair['pipe_id_a']} ↔ {pair['pipe_id_b']}  轴心距："
                f"{pair['model_distance_mm']:.2f} / {pair['observed_distance_mm']:.2f} / {pair['error_mm']:+.2f}"
            )
            if pair.get("observed_surface_gap_mm") is not None:
                lines.append(
                    f"    表面净距：{pair['model_surface_gap_mm']:.2f} / "
                    f"{pair['observed_surface_gap_mm']:.2f} / {pair['surface_gap_error_mm']:+.2f}"
                )
    if not available:
        lines += [
            "",
            "尚无可靠的管间尺寸对比：需要先获得合格三维管段并确定模型对应。",
            "图像候选框不等于实测管道，模型尺寸不会填入实测结果。",
        ]
    if report:
        lines += ["", recognition_summary(report)]
        from .section_display import section_summary_lines, section_evaluations

        if section_evaluations(report):
            lines += ["", *section_summary_lines(report)]
    return lines


def measurement_summary_lines(spec: dict, result: dict | None) -> list[str]:
    """Human-readable reference points; never label a raw median as an axis."""
    measurement = (result or {}).get("measurement") or {}
    lines = [
        f"管道：{spec['pipe_id']}",
        f"模型/配置外径：{spec['nominal_diameter_mm']:.2f} mm",
        "距离原点：左目矫正相机光心；Z 深度与空间直线距离分别列出。",
        "截面位置：左右共同观测管段的中部，不代表整根管道端点。",
    ]
    measured = measurement.get("measured_section") if measurement.get("status") == "MEASURED" else None
    if not measured:
        lines += ["当前实测外径：未获得", "中心线 / 最近表面：未获得可靠实测"]

    def metric(value):
        return f"{value / 1000:.4f} m（{value:.1f} mm）"

    for label, geometry in (
        ("双目拟合", measured),
        ("模型配准预测（非实测）", measurement.get("model_predicted_section")),
    ):
        if geometry is None:
            continue
        lines += [
            "",
            label,
            f"外径：{geometry['diameter_mm']:.2f} mm",
            "中心线 Z 深度：" + metric(geometry["centerline"]["depth_z_mm"]),
            "中心线空间距离：" + metric(geometry["centerline"]["range_mm"]),
            "该截面表面最小 Z 深度：" + metric(geometry["minimum_surface_depth_z_mm"]),
            "该截面最近表面空间距离：" + metric(geometry["nearest_surface_range_mm"]),
        ]
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


def image_region(
    start: tuple[float, float],
    end: tuple[float, float],
    transform: tuple[float, float, float],
    size: tuple[int, int],
) -> list[int] | None:
    """Map a viewport drag to source pixels for a group assessment region."""
    scale, ox, oy = transform
    if not math.isfinite(scale) or scale <= 0:
        return None
    left, right = sorted(((start[0] - ox) / scale, (end[0] - ox) / scale))
    top, bottom = sorted(((start[1] - oy) / scale, (end[1] - oy) / scale))
    x0, y0 = max(0, math.floor(left)), max(0, math.floor(top))
    x1, y1 = min(size[0], math.ceil(right)), min(size[1], math.ceil(bottom))
    return [x0, y0, x1 - x0, y1 - y0] if x1 - x0 >= 8 and y1 - y0 >= 8 else None


__all__ = [
    "REASON_TEXT",
    "image_region",
    "measurement_summary_lines",
    "recognition_diagnostic_lines",
    "recognition_summary",
    "relative_layout_summary_lines",
]
