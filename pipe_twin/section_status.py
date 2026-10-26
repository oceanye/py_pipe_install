"""Per-observation solve status, independent of unresolved CAD assignments."""
from __future__ import annotations

from .section_display import REASONS, section_evaluations


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
            if temporal and temporal.get('reason_codes'):
                current = '断面已解，重复性待改善' if solved else '断面未解'
            correction = stage.get('ground_correction') or {}
            if correction.get('trial_mode'):
                current = ('倾角试验 · ' if correction.get('applied') else '倾角试验未应用 · ')+current
            detail = [f'{zone} / {label}：{current}。',
                      '仅使用当前连拍；表中尺寸来自最后一帧，未将历史或采购尺寸补入拟合。',
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
            detail.extend(REASONS.get(r, r) for r in reasons)
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
