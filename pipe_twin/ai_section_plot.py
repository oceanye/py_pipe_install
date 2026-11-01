"""Read-only AI state styles for the existing CAD cross-section geometry.

The caller supplies current CAD pipes and owns source freshness checks. This
module never reads local installation hypotheses, fits, or measured centres.
It deliberately returns no geometry: AI candidate coordinates and nominal CAD
diameters must not become a second, apparently measured, field cross-section.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import re

from .ai_section_display import REASON_LABELS, ai_section_data


_STYLES = {
    'SUPPORTED_VISIBLE': ('图像支持', '#BDE8D8', '#17644B', (), 2),
    'AMBIGUOUS': ('有歧义', '#FFF1CF', '#895A0D', (6, 3), 2),
    'UNRESOLVED': ('未解', '#E7EDF2', '#596579', (2, 3), 1),
    'RECAPTURE_RECOMMENDED': ('建议重采', '#F4E4F6', '#804184', (6, 2, 2, 2), 2),
    'NOT_RUN': ('未运行', '#F2F5F8', '#596579', (2, 3), 1),
    'REJECTED': ('结果无效', '#F2F5F8', '#596579', (2, 3), 1),
    'STALE': ('已失效', '#F2F5F8', '#596579', (2, 3), 1),
}
_POINT_STATES = tuple(list(_STYLES)[:4])
_POINT_ID = re.compile(r'P[0-9]{2,}')
_CAPTION = '圆心和直径来自 CAD；颜色与线型仅表示 AI 图像候选，未解不表示未安装。'
_LEGEND = '实线：图像支持　长虚线：有歧义\n点线：未解　点划线：建议重采'


def _row(pipe_id, state, *, point_id='', quality='UNKNOWN', reason=''):
    label, fill, outline, dash, width = _STYLES[state]
    return {
        'pipe_id': pipe_id, 'point_id': point_id, 'status': state,
        'label': label, 'quality': quality, 'reason': reason,
        'fill': fill, 'outline': outline, 'dash': dash, 'width': width,
    }


def _reason(item):
    codes = item.get('reason_codes')
    if isinstance(codes, Sequence) and not isinstance(codes, (str, bytes)):
        return '；'.join(REASON_LABELS.get(code, code) for code in codes if isinstance(code, str))
    value = item.get('reason')
    return value if isinstance(value, str) else ''


def ai_section_plot(report, pipes, *, current=True):
    """Describe one AI overlay by pipe ID without changing CAD or report data.

    ``current=False`` suppresses every saved candidate. A rejected/not-run run
    also suppresses any leftover candidate rows. Supported rows in a partial
    accepted run remain available; missing, malformed or conflicting rows are
    unresolved. Selection belongs to the viewer and must be drawn separately
    from these state styles.
    """
    ids = list(dict.fromkeys(p['pipe_id'] for p in pipes
                            if isinstance(p, Mapping) and isinstance(p.get('pipe_id'), str)))
    data = ai_section_data(report if isinstance(report, Mapping) else None)
    run_status = data.get('run_status')
    decision = data.get('status')
    invalid = None
    reason = ''
    if not current:
        invalid, reason = 'STALE', '现场输入已改变，请重新评估。'
    elif not data or run_status == 'NOT_RUN' or decision == 'NOT_RUN':
        invalid, reason = 'NOT_RUN', 'AI 尚未运行；请点击⑤ AI 断面评估。'
    elif decision in ('STALE', 'INVALIDATED') or run_status in ('STALE', 'INVALIDATED'):
        invalid, reason = 'STALE', 'AI 结果已失效，请重新评估。'
    elif (run_status is not None and run_status not in ('VALID', 'PARTIAL')
          or decision in ('REJECTED', 'INVALID', 'FAILED', 'ERROR', 'CANCELLED')):
        invalid, reason = 'REJECTED', 'AI 结果未被接纳，未显示旧候选。'
    elif run_status is None and decision not in (*_POINT_STATES, 'VALID', 'PARTIAL', 'CANDIDATES_READY', 'INSUFFICIENT'):
        invalid, reason = 'REJECTED', '缺少可用的 AI 运行状态。'

    # Prefer an explicit empty canonical collection over a leftover alias.
    raw = next((data[k] for k in ('candidates', 'point_assessments', 'observations') if k in data), [])
    if invalid is None and (not isinstance(raw, Sequence) or isinstance(raw, (str, bytes))):
        invalid, reason = 'REJECTED', 'AI 候选列表无效。'
    if invalid is not None:
        return {
            'status': invalid, 'available': False,
            'by_pipe': {pid: _row(pid, invalid, reason=reason) for pid in ids},
            'counts': {state: 0 for state in _POINT_STATES},
            'summary': f'AI 断面候选 · {_STYLES[invalid][0]}。{reason}',
            'legend': _LEGEND, 'caption': _CAPTION,
        }

    candidates = [item for item in raw if isinstance(item, Mapping)]
    pipe_counts = Counter(item.get('pipe_id') for item in candidates if isinstance(item.get('pipe_id'), str))
    point_counts = Counter(item.get('point_id') for item in candidates if isinstance(item.get('point_id'), str))
    by_pipe = {pid: _row(pid, 'UNRESOLVED', reason='本次 AI 没有提供此管位的有效判断。') for pid in ids}
    for item in candidates:
        pid, point = item.get('pipe_id'), item.get('point_id')
        if not isinstance(pid, str) or pid not in by_pipe:
            continue
        state = item.get('section_status', item.get('status'))
        quality = item.get('evidence_quality', item.get('quality', 'UNKNOWN'))
        note = _reason(item)
        if not isinstance(point, str) or not _POINT_ID.fullmatch(point):
            state, point, note = 'UNRESOLVED', '', 'AI 断面点位编号缺失或无效。'
        elif pipe_counts[pid] > 1 or point_counts[point] > 1:
            state, note = 'UNRESOLVED', 'AI 管号或断面点位重复，无法唯一对应。'
        elif not isinstance(state, str) or state not in _POINT_STATES:
            state, note = 'UNRESOLVED', 'AI 候选状态无效。'
        by_pipe[pid] = _row(pid, state, point_id=point,
                           quality=quality if isinstance(quality, str) else 'UNKNOWN', reason=note)
    counts = {state: sum(row['status'] == state for row in by_pipe.values()) for state in _POINT_STATES}
    summary = (f"AI 断面候选 · 支持 {counts['SUPPORTED_VISIBLE']} / 歧义 {counts['AMBIGUOUS']} / "
               f"未解 {counts['UNRESOLVED']} / 重采 {counts['RECAPTURE_RECOMMENDED']}")
    return {'status': run_status or 'VALID', 'available': True, 'by_pipe': by_pipe,
            'counts': counts, 'summary': summary, 'legend': _LEGEND, 'caption': _CAPTION}


__all__ = ['ai_section_plot']
