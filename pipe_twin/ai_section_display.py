"""Read-only presentation helpers for the optional AI section candidate channel.

The AI channel is deliberately namespaced under ``report['ai_section_assist']``.
These helpers never derive candidates from the strict installation result or
from the legacy installation-assist hypotheses.  A missing provider result is
shown as an explicit not-run state so the GUI cannot imply that local rules
were produced by an AI model.
"""
from __future__ import annotations

from collections.abc import Mapping

STATUS_LABELS = {
    "NOT_RUN": "未运行",
    "SUPPORTED_VISIBLE": "支持定位",
    "AMBIGUOUS": "候选有歧义",
    "UNRESOLVED": "未能定位",
    "RECAPTURE_RECOMMENDED": "建议重采",
    "INSUFFICIENT": "证据不足",
    "CANDIDATES_READY": "候选已生成",
    "VALID": "已完成",
    "PARTIAL": "部分完成",
    "REJECTED": "结果被拒绝",
}
QUALITY_LABELS = {
    "HIGH": "高",
    "MEDIUM": "中",
    "LOW": "低",
    "UNKNOWN": "未评估",
}
REASON_LABELS = {
    "AI_PROVIDER_NOT_CONFIGURED": "尚未配置 AI 提供方。",
    "AI_RESULT_NOT_AVAILABLE": "当前报告没有 AI 断面候选结果。",
    "SECTION_POINT_AMBIGUOUS": "断面点位存在重叠或遮挡，无法唯一定位。",
    "SECTION_POINT_NOT_VISIBLE": "模型断面点位在当前观察方向不可见。",
    "OBSERVATION_DIRECTION_MISSING": "缺少观察方向，不能将图像候选映射到模型断面。",
    "OBSERVATION_DIRECTION_PARALLEL_TO_PIPE_AXIS": "观察侧沿管轴，无法确定横截面朝向，请调整模型观察方向。",
    "AI_AGGREGATION_BAND_MISSING": "尚未指定管长方向与汇聚范围，请在左图画出后重新评估。",
    "AI_SUPPORT_WITHOUT_FIELD_IMAGE": "AI 未引用本次实拍图，不能确认该点位。",
    "LOW_EVIDENCE_QUALITY": "证据质量较低，仅作候选参考。",
}

_EVIDENCE_STATUS_LABELS = {
    "VALID": "已通过",
    "PARTIAL": "部分通过",
    "ANALYZED": "已分析",
    "UNRESOLVED": "未解析",
    "INVALID": "无效",
    "REJECTED": "已拒绝",
}


def ai_section_data(report):
    """Return a valid, read-only AI result or ``{}``.

    The function intentionally requires the section-only scope marker.  This
    prevents a future provider response containing installation labels from
    accidentally being displayed as this feature's output.
    """
    value = (report or {}).get("ai_section_assist") or {}
    if not isinstance(value, dict):
        return {}
    # ``SECTION_ONLY`` is the GUI-facing alias used by the current report
    # contract; accept the stricter core spelling during the transition.
    if value.get("scope") not in ("SECTION_ONLY", "CROSS_SECTION_CANDIDATE_ONLY"):
        return {}
    return value


def _reason_text(item):
    codes = item.get("reason_codes") or []
    if not codes and item.get("reason"):
        return str(item["reason"])
    return "；".join(REASON_LABELS.get(str(code), str(code)) for code in codes) or "—"


def _position(item):
    point = item.get("point_id") or item.get("section_point_id") or item.get("model_point_id")
    position = item.get("section_position") or item.get("model_position")
    if isinstance(position, dict):
        position = position.get("label") or position.get("name") or position.get("index")
    if point and position not in (None, ""):
        return f"{point} · {position}"
    return str(point or position or "待指定")


def _control_section_lines(control, *, label):
    """Format a control section without confusing pixels with metric stations."""
    if not isinstance(control, Mapping) or not control:
        return []
    status = str(control.get("status") or "UNRESOLVED")
    if status != "DEFINED":
        reasons = control.get("reason_codes") or []
        reason = "；".join(REASON_LABELS.get(str(code), str(code)) for code in reasons)
        suffix = f"（{reason}）" if reason else ""
        return [f"{label}：未稳定确定{suffix}"]
    coordinate_space = control.get("coordinate_space")
    if coordinate_space == "rectified_left_pixel":
        return [
            f"{label}：左图汇聚范围中点垂线（像素），位置 {control.get('station_px')}。",
            "这是照片上的断面采样位置；尚未形成毫米空间站位。",
        ]
    interval = control.get("interval_mm") or []
    station = control.get("station_mm")
    try:
        station_text = f"{float(station):.1f} mm"
    except (TypeError, ValueError):
        station_text = "已定义"
    return [f"{label}：公共支撑区间中点 {station_text}（区间 {interval}）。"]


def _report_evidence_lines(report, data):
    """Summarize full-replay evidence beside the independent AI result.

    The values are read-only context.  In particular, local geometry counts
    are never converted into AI candidate states or installation states.
    """
    if not isinstance(report, Mapping):
        return []
    lines = []
    temporal = report.get("temporal_fusion")
    if isinstance(temporal, Mapping):
        status = str(temporal.get("status") or "UNRESOLVED")
        requested = temporal.get("requested_frames")
        accepted = temporal.get("accepted_frames")
        if requested is not None or accepted is not None:
            lines.append(
                f"时序融合：{_EVIDENCE_STATUS_LABELS.get(status, status)}"
                f"（{accepted if accepted is not None else '—'}/"
                f"{requested if requested is not None else '—'} 帧）。"
            )
        else:
            lines.append(f"时序融合：{_EVIDENCE_STATUS_LABELS.get(status, status)}。")
        reasons = temporal.get("reason_codes") or []
        if reasons:
            lines.append("时序提示：" + "；".join(str(code) for code in reasons) + "。")

    section = report.get("section_evaluation")
    if isinstance(section, Mapping):
        status = str(section.get("status") or "UNRESOLVED")
        candidates = section.get("candidates")
        count = len(candidates) if isinstance(candidates, list) else None
        conditional = section.get("conditional_section_count")
        summary = f"断面汇聚：{_EVIDENCE_STATUS_LABELS.get(status, status)}"
        if count is not None:
            summary += f"，候选 {count} 个"
        if conditional is not None:
            summary += f"，条件候选 {conditional} 个"
        lines.append(summary + "。")
        pooled = section.get("control_section")
        # Prefer the AI namespace when it has already bound its own control
        # section; otherwise expose the pooled section as an available input.
        ai_control = data.get("control_section") if isinstance(data, Mapping) else None
        if not (isinstance(ai_control, Mapping) and ai_control.get("status") == "DEFINED"):
            lines.extend(_control_section_lines(pooled, label="可供 AI 引用的主流程控制断面"))

    surface = report.get("local_surface")
    if isinstance(surface, Mapping):
        observations = surface.get("observations")
        image_candidates = surface.get("image_candidates")
        audit = surface.get("audit")
        metric_count = audit.get("metric_cylinder_count") if isinstance(audit, Mapping) else None
        if isinstance(observations, list) or isinstance(image_candidates, list) or metric_count is not None:
            observed_text = len(observations) if isinstance(observations, list) else "—"
            image_text = len(image_candidates) if isinstance(image_candidates, list) else "—"
            metric_text = metric_count if metric_count is not None else "—"
            lines.append(
                f"局部双目证据：可用观测 {observed_text}，图像候选 {image_text}，"
                f"可匹配圆柱 {metric_text}。"
            )
    return lines


def section_candidate_rows(report):
    """Build display rows without changing *report* or strict state."""
    data = ai_section_data(report)
    if not data:
        return [{
            "key": "__ai_section_not_run__",
            "label": "AI 断面候选",
            "pipe_id": "—",
            "position": "—",
            "status": "NOT_RUN",
            "quality": "UNKNOWN",
            "reason": REASON_LABELS["AI_RESULT_NOT_AVAILABLE"] + " " + REASON_LABELS["AI_PROVIDER_NOT_CONFIGURED"],
            "values": ("—", "—", STATUS_LABELS["NOT_RUN"], QUALITY_LABELS["UNKNOWN"], "未运行；AI_PROVIDER_NOT_CONFIGURED"),
        }]
    rows = []
    candidates = data.get("candidates") or data.get("point_assessments") or data.get("observations") or []
    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            continue
        status = str(item.get("section_status") or item.get("status") or "UNRESOLVED")
        quality = str(item.get("evidence_quality") or item.get("quality") or "UNKNOWN")
        observation = item.get("observation_id") or item.get("point_id") or item.get("label") or f"A{index + 1}"
        pipe_id = str(item.get("pipe_id") or item.get("candidate_pipe_id") or "待对应")
        position = _position(item)
        # Core responses keep fixed section coordinates below model_section.
        if position == "待指定" and isinstance(item.get("model_section"), dict):
            position = _position(item["model_section"])
        reason = _reason_text(item)
        key = f"{observation}:{pipe_id}:{index}"
        detail = (
            f"{observation}：{STATUS_LABELS.get(status, status)}\n"
            f"模型管号：{pipe_id}\n"
            f"模型断面点位：{position}\n"
            f"证据质量：{QUALITY_LABELS.get(quality, quality)}\n"
            f"说明：{reason}\n"
            "范围：仅评估模型断面候选；不判断三维管段，也不写入安装验收状态。"
        )
        rows.append({
            **item,
            "key": key,
            "label": str(observation),
            "pipe_id": pipe_id,
            "position": position,
            "status": status,
            "quality": quality,
            "reason": reason,
            "detail": detail,
            "values": (pipe_id, position, STATUS_LABELS.get(status, status),
                       QUALITY_LABELS.get(quality, quality), reason),
        })
    if not rows:
        status = str(data.get('status') or 'INSUFFICIENT')
        if status not in ('NOT_RUN', 'REJECTED', 'UNRESOLVED', 'RECAPTURE_RECOMMENDED'):
            status = 'INSUFFICIENT'
        reason = _reason_text(data)
        if reason == '—':
            reason = 'AI 尚未运行。' if status == 'NOT_RUN' else 'AI 结果没有可显示的候选。'
        return [{
            "key": "__ai_section_empty__",
            "label": "AI 断面候选",
            "pipe_id": "—",
            "position": "—",
            "status": status,
            "quality": "UNKNOWN",
            "reason": reason,
            "values": ("—", "—", STATUS_LABELS[status], QUALITY_LABELS["UNKNOWN"], reason),
        }]
    return rows


def ai_section_summary_lines(report):
    data = ai_section_data(report)
    if not data:
        return [
            "AI 断面候选 · 未运行",
            "范围：仅定位模型断面中的候选点位；不判断三维管段或安装完成状态。",
            "当前报告没有 AI 结果；AI_PROVIDER_NOT_CONFIGURED。",
            "本页与主线安装判定、辅助安装识别相互独立。",
        ]
    status = str(data.get("status") or "INSUFFICIENT")
    lines = [
        "AI 断面候选 · " + STATUS_LABELS.get(status, status),
        "范围：仅定位模型断面中的候选点位；不判断三维管段或安装完成状态。",
        f"提供方：{data.get('provider') or '未记录'}；契约：{data.get('schema_version') or '未记录'}。",
        f"模型：{data.get('model') or '未记录'}；结果生成时间：{data.get('generated_at') or '未记录'}。",
    ]
    lines.extend(_report_evidence_lines(report, data))
    if data.get('observation_scope') == 'TEMPORAL_BURST_REFERENCE_SET':
        count = data.get('temporal_frame_count')
        suffix = f'（{count} 组代表性双目帧）' if count else ''
        lines.append('输入范围：整组连续双目图像' + suffix + '；按同一目标的相机位移观察处理。')
    elif data.get('observation_scope') == 'LATEST_REFERENCE_PAIR_ONLY_NOT_TEMPORAL_FUSION':
        lines.append('输入范围：仅当前参考双目对；不代表整组多帧融合，也不是重新采集。')
    captured = sorted({str(asset['captured_at']) for asset in data.get('asset_refs', [])
                       if isinstance(asset, dict) and asset.get('kind') == 'FIELD_IMAGE' and asset.get('captured_at')})
    if captured:
        lines.append('实拍时间：' + ' / '.join(captured) + '。')
    control = data.get("control_section") or {}
    control_label = "图像控制断面" if control.get("coordinate_space") == "rectified_left_pixel" else "AI 输入控制断面"
    lines.extend(_control_section_lines(control, label=control_label))
    pooled = data.get('pooled_control_section')
    if isinstance(pooled, Mapping):
        lines.extend(_control_section_lines(pooled, label='AI 已绑定的主流程控制断面'))
    codes = data.get("reason_codes") or []
    lines.extend("提示：" + REASON_LABELS.get(str(code), str(code)) for code in codes)
    for row in section_candidate_rows(report):
        lines.extend(["", row.get("detail") or f"{row['label']}：{row['reason']}"])
    lines.append("AI 候选只读展示，不会覆盖或改变严格安装状态。")
    return lines
