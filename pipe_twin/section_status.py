"""Per-observation solve status, independent of unresolved CAD assignments."""
from __future__ import annotations

from .section_display import REASONS, section_evaluations

PAIRING_REASONS = {
    'INSUFFICIENT_INDEPENDENT_SILHOUETTES': '稳定的独立轮廓不足，无法自由求解圆心和半径',
    'PROJECTED_WIDTH_MISMATCH': '左右可见宽度不一致',
    'AMBIGUOUS_STEREO_CANDIDATES': '有多个可能的左右对应，不能唯一配对',
    'NO_COMPATIBLE_STEREO_CANDIDATE': '未找到同时满足几何检查的右目候选',
    'NO_STABLE_BOUNDARY': '没有通过子段一致性检查的边界',
}
STATUS_REASONS = {'PIXEL_UNCERTAINTY_UNRESOLVED': '几何条件过弱，像素误差估计无法稳定求解'}


def pairing_detail_lines(audit):
    if not audit or audit.get('status') == 'PAIRED':
        return []
    lines = [PAIRING_REASONS.get(audit.get('status'), '左右候选尚未唯一配对')+'。']
    for other in audit.get('nearby_candidates', [])[:3]:
        lines.append(f"邻近右目候选 {other['candidate_id']}：左 {audit['stable_boundary_count']} 条、"
                     f"右 {other['stable_boundary_count']} 条稳定轮廓，合计 {other['combined_stable_boundary_count']} 条。")
        codes = other.get('reason_codes', [])
        lines.extend(PAIRING_REASONS.get(code, code)+'。' for code in codes)
        if 'INSUFFICIENT_INDEPENDENT_SILHOUETTES' in codes:
            lines.append('自由圆心与半径至少需要 3 条独立切线约束；增加同一条边的像素不能补出缺失的另一条轮廓。')
    return lines


def _box(stage, role, box):
    if box is None:
        return None
    x, y, w, h = box
    points = [[x, y], [x+w, y], [x, y+h], [x+w, y+h]]
    refinement = stage.get('epipolar_refinement') or {}
    if role == 'right' and refinement.get('applied'):
        from .epipolar_refinement import transform_points
        points = transform_points(points, refinement['right_refined_to_original'])
    xs, ys = list(zip(*points))
    offsets = stage.get('zone_offsets') or {}
    return [min(xs)+offsets.get(role+'_x', 0), min(ys)+offsets.get(role+'_y', 0),
            max(xs)-min(xs), max(ys)-min(ys)]


def solver_status_rows(report):
    rows = []
    for stage_index, stage in enumerate(section_evaluations(report)):
        zone = stage.get('zone_id', '全幅')
        views = {role: {v['candidate_id']: v for v in stage.get('views', {}).get(role, [])}
                 for role in ('left', 'right')}
        model = (stage.get('model') or {}).get('candidates', [])
        labels = {m['id']: f'S{i+1}' for i, m in enumerate(m for m in model if m.get('estimated_center_uv_mm') is not None)}
        registration = stage.get('registration') or {}
        matches = registration.get('matches', []) if registration.get('status') == 'MATCHED' else []
        if len({m['pipe_id'] for m in matches}) != len(matches) or len({m['observation_id'] for m in matches}) != len(matches):
            matches = []
        assigned = {m['observation_id']: m['pipe_id'] for m in matches}
        candidates = list(stage.get('candidates', []))
        paired = {c.get('left_candidate_id') for c in candidates}
        # Keep useful unpaired image evidence visible, but never count it as a solved pipe.
        for view in views['left'].values():
            if (view['candidate_id'] not in paired and view.get('aggregation', {}).get('stable_boundary_count', 0)
                    and not set(view.get('reason_codes', [])) & {'TRANSVERSE_BOUNDARY_CLIPPED', 'VISIBLE_BAND_TOO_NARROW'}):
                candidates.append({'candidate_id': view['candidate_id'], 'left_candidate_id': view['candidate_id'],
                    'left_region_px': view.get('region_px'), 'reason_codes': ['STEREO_SECTION_PAIR_UNRESOLVED']})
        for index, candidate in enumerate(candidates):
            identity = candidate['candidate_id']
            pairing = next((row for row in stage.get('pairing_audit', {}).get('left', [])
                            if row['candidate_id'] == candidate.get('left_candidate_id')), None)
            left = views['left'].get(candidate.get('left_candidate_id'), {})
            right = views['right'].get(candidate.get('right_candidate_id'), {})
            edge_counts = [v.get('aggregation', {}).get('stable_boundary_count', 0) for v in (left, right)]
            fit = candidate.get('fit') or {}
            solved = fit.get('status') == 'CONDITIONAL_FIT' and identity in labels
            diameter = fit.get('trial_diameter_mm') if solved else None
            temporal = candidate.get('temporal_consistency') or {}
            spread = temporal.get('diameter')
            reasons = list(candidate.get('reason_codes', []))
            if solved and not candidate.get('window_sensitivity', {}).get('stable'):
                reasons.append('POOLED_DIMENSIONS_UNSTABLE')
            assessment = stage.get('position_assessment') or {}
            if assessment.get('calibration_validated') is False:
                reasons.append('CALIBRATION_NOT_VALIDATED')
            if stage.get('epipolar_refinement', {}).get('applied'):
                reasons.append('EPIPOLAR_REFINEMENT_NOT_METRIC_VALIDATED')
            reasons = list(dict.fromkeys(reasons))
            if temporal:
                repeat = ('通过' if temporal['diameter_repeatable'] else '波动' if spread else '不足')
                if spread:
                    repeat += f" {spread['min_mm']:.1f}～{spread['max_mm']:.1f}"
                frames = f"{temporal['solved_frames']}/{temporal['sampled_frames']}"
                layout = '通过' if temporal['layout_repeatable'] else '待稳定'
            else:
                repeat, frames, layout = '未复核', '单帧', '未复核'
                if (stage.get('temporal_sections') or {}).get('status') == 'BURST_REJECTED':
                    repeat, frames = '连拍未通过', '无有效连拍'
            label = labels.get(identity, f'C{index+1}')
            current = '断面已解（条件估计）' if solved else '断面未解'
            if not solved and pairing and pairing.get('status') == 'INSUFFICIENT_INDEPENDENT_SILHOUETTES':
                current = '断面未解（稳定轮廓不足）'
            if temporal and temporal.get('reason_codes'):
                current = '断面已解，重复性待改善' if solved else '断面未解'
            correction = stage.get('ground_correction') or {}
            if correction.get('trial_mode'):
                current = ('倾角试验 · ' if correction.get('applied') else '倾角试验未应用 · ')+current
            source_text = ('仅使用当前连拍；表中尺寸来自最后一帧，未将历史或采购尺寸补入拟合。' if temporal else
                           '本次多帧已复核；该候选尚无可比较的重复解。'
                           if (stage.get('temporal_sections') or {}).get('status') == 'CHECKED' else
                           '本次连拍未通过；以下是单帧条件结果，未完成重复性复核。'
                           if (stage.get('temporal_sections') or {}).get('status') == 'BURST_REJECTED' else
                           '本次仅有单帧断面结果，未完成多帧重复性复核；采购尺寸未参与拟合。')
            detail = [f'{zone} / {label}：{current}。',
                      source_text,
                      '重复性仅反映本次帧间变化，不代表绝对精度或已完成安装。']
            if temporal:
                detail += [f"解出 {frames} 帧，至少要求 {temporal['required_frames']} 帧。"]
                if spread:
                    detail += [f"外径极差 {spread['span_mm']:.2f} mm；重复性门限 {temporal['diameter_span_limit_mm']:.2f} mm。"]
                for pair in stage.get('temporal_sections', {}).get('pairwise', []):
                    if identity in pair['ids'] and pair.get('distance'):
                        ids = ' / '.join(labels.get(i, i) for i in pair['ids'])
                        distance = pair['distance']
                        detail.append(f"{ids} 心距范围 {distance['min_mm']:.1f}～{distance['max_mm']:.1f} mm；极差 {distance['span_mm']:.1f} mm。")
            detail.extend(STATUS_REASONS.get(r, REASONS.get(r, r)) for r in reasons)
            if not solved:
                detail.extend(pairing_detail_lines(pairing))
            if correction.get('trial_mode'):
                from .ground_correction import correction_summary
                detail.append(correction_summary(correction))
                comparison = stage.get('ground_trial_comparison') or {}
                for change in comparison.get('candidates', []):
                    if change['candidate_id'] == identity:
                        a, b = change['before_diameter_mm'], change['after_diameter_mm']
                        if a is not None and b is not None:
                            detail.append(f'同图对照外径：默认修正规则 {a:.2f} → 候选地面试验 {b:.2f} mm。')
                for change in comparison.get('pairs', []):
                    if identity in change['ids']:
                        ids = ' / '.join(labels.get(i, i) for i in change['ids'])
                        detail.append(f"{ids} 同图心距：{change['before_center_distance_mm']:.2f} → {change['after_center_distance_mm']:.2f} mm。")
                detail.append('前后变化仅用于测试效果，不能据此认定更准确。')
            if not assigned.get(identity):
                detail.append('尚未唯一对应 CAD 管号，安装状态继续待判定。')
            rows.append({'key': f'{stage_index}:{identity}', 'zone_id': zone, 'candidate_id': identity,
                'label': label, 'color_srgb': left.get('color_srgb') or '#8B9BAB',
                'left_region_px': _box(stage, 'left', candidate.get('left_region_px')),
                'right_region_px': _box(stage, 'right', candidate.get('right_region_px')),
                'solved': solved, 'diameter_mm': diameter, 'temporal_consistency': temporal,
                'pipe_id': assigned.get(identity), 'reason_codes': reasons, 'detail': '\n'.join(detail),
                'values': [f"左 {edge_counts[0]} / 右 {edge_counts[1]}",
                    f'{diameter:.2f}（估）' if diameter is not None else '未解出', frames, repeat, layout,
                    assigned.get(identity) or '待对应', current]})
    return rows
