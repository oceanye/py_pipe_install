"""Read-only presentation of coarse installation hypotheses, separate from acceptance."""
from __future__ import annotations

STATUS_LABELS = {'LIKELY': '疑似已安装', 'AMBIGUOUS': '多个对应可选',
                 'INSUFFICIENT': '证据不足', 'NO_OBSERVATION': '未观察到'}
COLORS = {'BLUE': '蓝', 'RED': '红', 'WHITE': '白', 'GREEN': '绿', 'YELLOW': '黄', 'UNKNOWN': '未分类'}
SIZE_LABELS = {'SMALL': '细', 'MEDIUM': '中', 'LARGE': '粗', 'SAME_SIZE': '同档', 'UNRESOLVED': '待比较'}
REASONS = {'MULTIPLE_PLAUSIBLE_LAYOUTS': '多个设计位置组合接近，请切换方案对照。',
           'RELATIVE_LAYOUT_CONFLICT': '相对关系与设计差异较大，请核对观察侧和管段范围。',
           'TOO_FEW_LAYOUT_OBSERVATIONS': '可用管道太少，暂不足以区分排列组合。',
           'SEARCH_TRUNCATED': '候选组合太多，尚未完成全部比较。',
           'NO_SINGLE_CURRENT_IMAGE_SECTION': '尚无当前单一区域的图像候选，请重新评估。',
           'NO_COMPATIBLE_LAYOUT_HYPOTHESIS': '尚未找到符合观察侧的相对排列。',
           'NO_COMPATIBLE_MODEL_SIZE_OR_COLOR_CLASS': '颜色或粗细档位与模型不符，请核对采购规格提示。'}


def assist_data(report):
    value = (report or {}).get('installation_assist') or {}
    return value if value.get('advisory_only') is True else {}


def hypotheses(report):
    return assist_data(report).get('ranked_hypotheses') or []


def hypothesis_assignments(report, index=0):
    choices = hypotheses(report)
    if not choices or not isinstance(index, int) or not 0 <= index < len(choices):
        return {}
    rows = choices[index].get('assignments') or []
    if (any(not r.get('observation_id') or not r.get('pipe_id') for r in rows)
            or len({r['observation_id'] for r in rows}) != len(rows)
            or len({r['pipe_id'] for r in rows}) != len(rows)):
        return {}
    return {r['observation_id']: r['pipe_id'] for r in rows}


def relation_label(assignment):
    layer, count = assignment.get('model_layer_index_from_camera'), assignment.get('model_layer_count')
    if layer is not None and count:
        layer_text = ('单层' if count == 1 else '近层' if layer == 1 else '远层' if layer == count
                      else '中层' if count == 3 else f'第 {layer}/{count} 层')
    else:
        layer_text = assignment.get('model_layer_label') or '层位待定'
    lateral = assignment.get('model_lateral_rank')
    return layer_text+(f' · 横向位 {lateral}' if lateral is not None else '')


def assist_rows(report, index=0):
    mapping = hypothesis_assignments(report, index)
    choices = hypotheses(report)
    coordinates = {r['observation_id']: r for r in choices[index].get('assignments', [])} if choices and 0 <= index < len(choices) else {}
    rows = []
    for number, item in enumerate(assist_data(report).get('observations') or []):
        identity = item['observation_id']
        label = item.get('label') or f'A{number+1}'
        candidates = list(item.get('candidate_pipe_ids') or [])
        chosen = mapping.get(identity)
        relation = relation_label(coordinates.get(identity, {}))
        status = STATUS_LABELS.get(item.get('status'), '待评估')
        if chosen and item.get('suggested_pipe_id') and chosen != item['suggested_pipe_id']:
            status = '备选对照，待确认'
        colour = COLORS.get(item.get('color_class'), item.get('color_class') or '未分类')
        size = SIZE_LABELS.get(item.get('relative_size_class'), item.get('relative_size_class') or '待比较')
        detail = [f'{label}：{status}', f'颜色：{colour}；相对大小：{size}。',
                  '当前候选方案：'+(str(chosen)+'（待确认）' if chosen else '无对应'),
                  '模型相对位置：'+relation+'。层位从指定观察侧近到远排列，不是实测地面高度。',
                  '可选模型管号：'+('、'.join(candidates) if candidates else '暂无'),
                  '依据颜色、粗细、排列和层间关系辅助识别；不要求毫米尺寸验收通过。',
                  '方案排序不是正确率；切换方案仅供比较，不代表确认安装或尺寸合格。']
        if item.get('image_width_px') is not None:
            detail.append(f"图像可见宽度约 {item['image_width_px']:.1f} px；受距离和遮挡影响。")
        if item.get('prior_diameter_mm') is not None:
            detail.append(f"采用采购外径提示 {item['prior_diameter_mm']:g} mm，仅作识别先验。")
        detail += ['提示：'+REASONS.get(reason, str(reason)) for reason in item.get('reason_codes') or []]
        rows.append({**item, 'label': label, 'key': identity,
                     'preview_pipe_id': chosen, 'detail': '\n'.join(detail),
                     'values': (colour, str(size), chosen or '待对应', relation, ' / '.join(candidates) or '—', status)})
    return rows


def assist_summary_lines(report, index=0):
    data = assist_data(report)
    if not data:
        return ['辅助安装识别：请点击⑤保存并评估。', '历史报告需要重新评估才能生成辅助候选。']
    counts = data.get('counts') or {}
    lines = ['辅助安装识别 · '+STATUS_LABELS.get(data.get('status'), '待评估'),
             '目标：通过上下层、左右关系和粗细差异，协助区分已安装的模型管位。',
             '黄色表示当前候选方案；多个方案接近时请结合现场比较，尚未观察到不等于未安装。',
             f"疑似已安装 {counts.get('LIKELY', 0)}；对应有歧义 {counts.get('AMBIGUOUS', 0)}；未观察到 {counts.get('NO_OBSERVATION', 0)}。"]
    if data.get('known_diameter_used_for_localization'):
        lines.append('采用已知管径辅助定位；管径来自识别先验，不是本次实测，也不覆盖原测量。')
    relations = data.get('relation_summary') or []
    lines.extend([relations] if isinstance(relations, str) else [str(r) for r in relations])
    lines.extend('提示：'+REASONS.get(reason, str(reason)) for reason in data.get('reason_codes') or [])
    choices = hypotheses(report)
    if choices and 0 <= index < len(choices):
        selected = choices[index]
        lines += ['', f'当前预览：方案 {index+1} / {len(choices)}（待确认，不是验收结果）']
        if selected.get('difference_score') is not None:
            lines.append(f"关系差异分数 {selected['difference_score']:.3f}；越小越相符，不是概率。")
        if selected.get('near_best') is False:
            lines.append('此方案弱于首选，仅供对照，不纳入疑似已安装。')
        for row in selected.get('assignments') or []:
            label = next((r['label'] for r in assist_rows(report, index) if r['key'] == row['observation_id']), row['observation_id'])
            lines.append(f"{label} → {row['pipe_id']} · {relation_label(row)}")
        lines.append('层位按指定观察侧从近到远；下视时通常对应上/中/下层。横向位按指定观察侧的设计排列。')
        relations = selected.get('relation_summary') or []
        lines.extend([relations] if isinstance(relations, str) else [str(r) for r in relations])
    lines += ['', '颜色与采购管径只用于识别先验，不会覆盖实测管径；精确尺寸/就位结果保留在其他分页。']
    for row in assist_rows(report, index):
        lines += ['', row['detail']]
    return lines


def assist_preview_states(report, strict_states, index=0):
    states = dict(strict_states)
    data = assist_data(report)
    if data.get('status') not in ('LIKELY', 'AMBIGUOUS'):
        return states
    observations = {o['observation_id']: o for o in data.get('observations') or []}
    for oid, pid in hypothesis_assignments(report, index).items():
        if states.get(pid, 'UNKNOWN') == 'UNKNOWN':
            observation = observations.get(oid, {})
            states[pid] = ('SUSPECTED_INSTALLED' if observation.get('status') == 'LIKELY'
                          and observation.get('suggested_pipe_id') == pid else 'ASSIST_CANDIDATE')
    return states
