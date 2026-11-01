"""Read-only presentation helpers for the optional AI section candidate channel.

The AI channel is deliberately namespaced under ``report['ai_section_assist']``.
These helpers never derive candidates from the strict installation result or
from the legacy installation-assist hypotheses.  A missing provider result is
shown as an explicit not-run state so the GUI cannot imply that local rules
were produced by an AI model.
"""
from __future__ import annotations

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
    "LOW_EVIDENCE_QUALITY": "证据质量较低，仅作候选参考。",
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
        return [{
            "key": "__ai_section_empty__",
            "label": "AI 断面候选",
            "pipe_id": "—",
            "position": "—",
            "status": "INSUFFICIENT",
            "quality": "UNKNOWN",
            "reason": "AI 结果没有可显示的候选。",
            "values": ("—", "—", STATUS_LABELS["INSUFFICIENT"], QUALITY_LABELS["UNKNOWN"], "没有候选"),
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
    ]
    control = data.get("control_section") or {}
    if isinstance(control, dict):
        if control.get("status") == "DEFINED":
            interval = control.get("interval_mm") or []
            station = control.get("station_mm")
            try:
                station_text = f"{float(station):.1f} mm"
            except (TypeError, ValueError):
                station_text = "已定义"
            lines.append(f"控制断面：管段汇聚共同支撑区间中点 {station_text}（区间 {interval}）。")
        elif control:
            lines.append("控制断面：管段汇聚共同支撑区间未能稳定确定。")
    codes = data.get("reason_codes") or []
    lines.extend("提示：" + REASON_LABELS.get(str(code), str(code)) for code in codes)
    for row in section_candidate_rows(report):
        lines.extend(["", row.get("detail") or f"{row['label']}：{row['reason']}"])
    lines.append("AI 候选只读展示，不会覆盖或改变严格安装状态。")
    return lines
