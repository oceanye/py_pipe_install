"""Read-only GUI presentation of pooled boundaries and relative sections."""
from __future__ import annotations

REASONS = {
    'TEMPORAL_SECTION_SUPPORT_INSUFFICIENT': '当前连拍中可独立解算的帧数不足',
    'TEMPORAL_SECTION_DIAMETER_UNSTABLE': '当前连拍的外径变化超出重复性门限',
    'TEMPORAL_SECTION_LAYOUT_UNSTABLE': '当前连拍的相对心距不稳定或比较帧数不足',
    'STEREO_SECTION_PAIR_UNRESOLVED': '已有图像边界，左右候选尚未唯一配对',
    'PARTIAL_STEREO_BOUNDARY_SUPPORT': '仅三条稳定轮廓参与拟合；缺失边界未补齐，尚不用于就位验收',
    'EPIPOLAR_REFINEMENT_NOT_METRIC_VALIDATED': '图像微校正改善了对齐，毫米精度尚未独立验证',
    'CALIBRATION_NOT_VALIDATED': '当前为暂用标定，尚未验证尺寸精度',
    'CAPTURE_UNHEALTHY': '拍摄质量未通过',
    'SECTION_SEARCH_TRUNCATED': '候选搜索未完成',
    'SECTION_MODEL_CORRESPONDENCE_UNRESOLVED': '尚未唯一对应到模型管号',
    'SECTION_MODEL_CORRESPONDENCE_NOT_ONE_TO_ONE': '模型对应重复',
    'NO_UNIQUE_VISIBLE_SECTION': '没有可唯一定位的可见断面',
    'POOLED_BOUNDARY_UNSTABLE': '子段边界不一致',
    'POOLED_DIMENSIONS_UNSTABLE': '不同子段解出的尺寸或位置不一致',
    'NO_RELATIVE_SECTION_REFERENCE': '缺少另一根可比较的管道',
    'SECTION_UNCERTAINTY_CROSSES_TOLERANCE': '误差区间跨越容差，暂不能判定',
    'SECTION_DIMENSIONS_OUTSIDE_TOLERANCE': '断面尺寸或间距超出容差',
    'SECTION_DIAMETERS_AND_RELATIVE_GAPS_AGREE': '断面尺寸及相对间距符合当前容差',
    'ZONE_POSITION_CONFLICT': '不同分区的就位结果冲突',
    'MULTIPLE_BANDS_OR_OCCLUDED_INSTANCES': '外包络可能包含多根管或遮挡',
    'POSSIBLE_SHARED_PIPE_INSTANCE': '分离片段可能属于同一根管，不重复计数',
    'CYLINDER_PROJECTION_INCONSISTENT': '共同圆柱模型与图像边界不一致',
    'COMMON_AXIS_UNSTABLE': '二维边界虽稳定，共同三维轴向仍不稳定',
    'COMMON_AXIS_UNRESOLVED': '共同三维轴向证据不足',
    'TRANSVERSE_BOUNDARY_CLIPPED': '管宽边界超出画面',
    'VISIBLE_BAND_TOO_NARROW': '可见条带过窄',
    'AXIAL_IMAGE_PARAMETERIZATION_UNSUPPORTED': '当前轴向投影不适用此汇聚方式',
    'GROUND_AXIS_CONFLICT': '水平管假设与图像边界不一致',
}


def section_evaluations(report):
    if not report:
        return []
    return report.get('section_evaluations') or ([report['section_evaluation']] if report.get('section_evaluation') else [])


def pooling_summary(report):
    stages = section_evaluations(report)
    if not stages:
        return ''
    count = sum(s.get('boundary_pair_count', 0) for s in stages)
    states = (report.get('position_assessment') or {}).get('counts', {})
    return (f"汇聚边界 {count} 对；断面相对就位 {states.get('IN_POSITION', 0)} / "
            f"不符 {states.get('OUT_OF_POSITION', 0)} / 待判定 {states.get('UNKNOWN', 0)}")


def section_summary_lines(report, *, detailed=False):
    lines = ['流程：沿管轴汇聚可见边界 → 检查子段一致性 → 估算断面尺寸 → 对比模型相对就位。',
             '汇聚不会恢复遮挡；稳定边界不等于尺寸准确。同一根管的分离片段不作为两根管计数。',
             '只比较管径、轴心距和净距；相机距离和管长不作为模型验收目标。',
             '当前间距筛查容差 5 mm；误差区间跨越容差时待判定。结论仅限可见共同断面的相对关系。']
    stages = section_evaluations(report)
    if not stages:
        return lines + ['', '请点击④保存并评估，生成管段汇聚结果。']
    lines += ['', pooling_summary(report)]
    for stage in stages:
        lines += ['', f"{stage.get('zone_id', '全幅')}：稳定双目边界 {stage.get('boundary_pair_count', 0)} 对"]
        temporal = stage.get('temporal_sections') or {}
        if temporal.get('status') == 'CHECKED':
            lines += [f"当前连拍：逐帧独立复核 {temporal['sampled_frames']} 帧；只检查重复性，不证明绝对精度。"]
        band = stage.get('aggregation_band')
        if band:
            lines += [f"手动汇聚范围：沿左图指定方向 {band['span_px']:.1f} px，两条垂线之间取样。"]
            if stage.get('section_revision') in ('pooled-sections-common-station-v2', 'pooled-sections-ground-constrained-v3'):
                lines += ['右图使用对应视差搜索带；鼠标仅提供大致方向，实际边缘方向从图像重新估计。',
                          '三维轴向和毫米尺寸仍需解算；图像内倾斜与纵深倾斜分别处理。']
            else:
                lines += ['右图使用对应视差搜索带；所画方向是二维取样方向，三维轴向和毫米尺寸仍需解算。']
        if stage.get('section_revision') in ('pooled-sections-common-station-v2', 'pooled-sections-ground-constrained-v3'):
            lines += ['子段按两眼共同轴向区间对齐；圆心差在同一法向断面上比较，不把沿管长的位移算作误差。']
        if stage.get('ground_correction'):
            from .ground_correction import correction_summary
            lines.append(correction_summary(stage['ground_correction']))
        refinement = stage.get('epipolar_refinement', {})
        if refinement.get('applied'):
            before = refinement['heldout_before_median_p95_px']
            after = refinement['heldout_after_median_p95_px']
            lines.append(f'断面图像微校正：独立验证点垂直误差 P95 {before[1]:.2f} → {after[1]:.2f} px；原照片、标定和深度图保留，尺寸精度仍待验证。')
        for candidate in stage.get('candidates', []):
            fit = candidate.get('fit', {})
            diameter = fit.get('trial_diameter_mm')
            reasons = candidate.get('reason_codes', [])
            boundary_label = '三条稳定轮廓' if candidate.get('boundary_status') == 'PARTIAL' else '边界稳定'
            text = f"{candidate['candidate_id']}：{boundary_label}；"
            if diameter is None:
                text += '断面尺寸未解出'
            else:
                text += f"条件估计外径 {diameter:.2f} / 半径 {diameter/2:.2f} mm"
                if reasons:
                    text += '（未通过尺寸检查，仅作诊断）'
            if reasons:
                text += '；'+'、'.join(REASONS.get(r, r) for r in reasons)
            lines.append(text)
            temporal = candidate.get('temporal_consistency') or {}
            if temporal:
                spread = temporal.get('diameter')
                lines.append(f"  连拍解出 {temporal['solved_frames']}/{temporal['sampled_frames']} 帧；外径重复性{'通过' if temporal['diameter_repeatable'] else '未通过'}。")
                if spread:
                    lines.append(f"  外径范围 {spread['min_mm']:.2f}～{spread['max_mm']:.2f} mm，极差 {spread['span_mm']:.2f} mm。")
            sensitivity = candidate.get('window_sensitivity') or {}
            if 'center_difference_mm' in sensitivity:
                lines.append(f"  子段间变化：直径 {sensitivity['diameter_difference_mm']:.2f} mm，断面圆心 {sensitivity['center_difference_mm']:.2f} mm")
                components = sensitivity.get('center_difference_components_mm')
                if components:
                    lines.append(f'  圆心差分量：断面横向 {components[0]:.2f} mm，另一法向方向 {components[1]:.2f} mm；不表示实物移动。')
            elif sensitivity.get('reason') in ('NO_COMMON_AXIAL_SUPPORT', 'AXIAL_STATION_MAPPING_UNRESOLVED'):
                lines.append('  两眼共同轴向采样不足，无法比较子段尺寸。')
        for pair in (stage.get('model') or {}).get('pairs', []):
            lines.append(f"条件断面 {' / '.join(pair['ids'])}：轴心距 {pair['center_distance_mm']:.2f} / 净距 {pair['surface_clearance_mm']:.2f} mm（非已验收值）")
        if detailed:
            for role, label in [('left', '左'), ('right', '右')]:
                for item in stage.get('views', {}).get(role, []):
                    lines.append(f"{label} {item['candidate_id']}：{'边界稳定' if item['boundary_stable'] else '边界未稳定'}")
                    for edge in item['aggregation']['boundaries']:
                        p95 = edge.get('holdout_p95_px')
                        metric = '—' if p95 is None else f'{p95:.2f}'
                        lines.append(f"  {edge['edge']}：有效子段 {edge.get('accepted_windows', 0)}/{edge.get('total_windows', 0)}；交叉检查 P95 {metric} px（门限 2 px）")
                    lines += ['  '+REASONS.get(r, r) for r in item.get('reason_codes', [])]
            lines.append(f"规则版本：{stage.get('revision')}；规则摘要：{stage.get('policy_sha256')}")
    position = (report or {}).get('position_assessment') or {}
    lines += ['', '模型管道的断面相对就位：']
    for row in position.get('pipes', []):
        lines.append(f"{row['pipe_id']}：{row['state_zh']}；"+'、'.join(REASONS.get(r, r) for r in row.get('reason_codes', [])))
    for stage in stages:
        for pair in stage.get('position_assessment', {}).get('pairwise', []):
            lines.append(f"{' / '.join(pair['pipe_ids'])}：心距 模型 {pair['model_center_distance_mm']:.2f} / 估计 {pair['observed_center_distance_mm']:.2f} mm；净距 模型 {pair['model_surface_clearance_mm']:.2f} / 估计 {pair['observed_surface_clearance_mm']:.2f} mm")
    return lines


def boundary_overlay_segments(report, role):
    """Draw only accepted observed blocks, with zone offsets restored."""
    lines = []
    for stage in section_evaluations(report):
        offsets = stage.get('zone_offsets', {})
        dx, dy = offsets.get(role+'_x', 0), offsets.get(role+'_y', 0)
        for item in stage.get('views', {}).get(role, []):
            for edge in item['aggregation']['boundaries']:
                if edge['status'] != 'STABLE':
                    continue
                slope, intercept = edge['line_slope_intercept']
                for window in edge['windows']:
                    if window['accepted']:
                        u, v = window['axial_interval_px']
                        points = [[u, slope*u+intercept], [v, slope*v+intercept]]
                        refinement = stage.get('epipolar_refinement', {})
                        if role == 'right' and refinement.get('applied'):
                            from .epipolar_refinement import transform_points
                            points = transform_points(points, refinement['right_refined_to_original'])
                        lines.append((points[0][0]+dx, points[0][1]+dy, points[1][0]+dx, points[1][1]+dy))
    return lines
