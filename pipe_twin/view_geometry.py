"""Explain section observability in the report's own rectified camera frame.

These are setup diagnostics, not a measurement acceptance gate. A baseline's
component across a pipe axis concerns silhouette section fitting specifically;
it is not a replacement for the physical baseline in textured stereo matching.
"""
from __future__ import annotations

import math
import numpy as np


def section_view_geometry(axis, baseline_mm):
    try:
        direction = np.array(axis, dtype=float, copy=True)
        baseline = float(baseline_mm)
        if (direction.shape != (3,) or not np.isfinite(direction).all()
                or np.linalg.norm(direction) < 1e-12 or not math.isfinite(baseline) or baseline <= 0):
            raise ValueError('Invalid section geometry')
    except (TypeError, ValueError):
        return {'status': 'GEOMETRY_UNRESOLVED', 'accepted_measurement': False}
    direction /= np.linalg.norm(direction)
    fraction = float(np.linalg.norm(np.cross([1., 0., 0.], direction)))
    fraction = float(np.clip(fraction, 0., 1.))
    return {'status': 'CONDITIONAL_GEOMETRY', 'physical_baseline_mm': baseline,
            'baseline_perpendicular_to_axis_mm': baseline*fraction,
            'baseline_cross_axis_fraction': fraction,
            'baseline_axis_acute_angle_deg': math.degrees(math.asin(fraction)),
            'axis_camera': direction.tolist(), 'coordinate_frame': 'LEFT_RECTIFIED_CAMERA_MM',
            'scope': 'SILHOUETTE_SECTION_GEOMETRY_ONLY', 'accepted_measurement': False}


def report_view_geometry(report):
    report = report or {}
    stages = report.get('section_evaluations') or ([report['section_evaluation']]
               if report.get('section_evaluation') else [])
    audit = report.get('calibration_audit') or {}
    old_baseline = ((audit.get('quality_audit') or {}).get('rectification_geometry') or {}).get(
        'horizontal_encoded_baseline_mm')
    results = []
    for stage in stages:
        direction = stage.get('axis') or {}
        baselines = [c.get('fit', {}).get('physical_baseline_mm') for c in stage.get('candidates', [])]
        baselines = [float(b) for b in baselines if isinstance(b, (float, int)) and math.isfinite(b) and b > 0]
        baseline = baselines[0] if baselines else old_baseline
        if baselines and not np.allclose(baselines, baseline, rtol=1e-8, atol=1e-8):
            value = {'status': 'REPORT_BASELINE_CONFLICT', 'accepted_measurement': False}
        elif direction.get('status') != 'STABLE':
            value = {'status': 'AXIS_UNRESOLVED', 'accepted_measurement': False}
        else:
            value = section_view_geometry(direction.get('axis'), baseline)
        value.update(zone_id=stage.get('zone_id'), source_generated_at=report.get('generated_at'),
                     calibration_id=audit.get('calibration_id'),
                     axis_source=direction.get('source', 'IMAGE_BOUNDARIES'),
                     ground_boundary_check_passed=(stage.get('ground_correction') or {}).get('boundary_check_passed'),
                     ground_trial=bool((stage.get('ground_correction') or {}).get('trial_mode')))
        results.append(value)
    return results


def view_geometry_summary_lines(report):
    lines = ['机位几何检查（来自本次评估照片，不是实时角度）',
             '本页检查双目连线横跨管轴的程度；光轴相对地面的倾斜见“地面倾角”页。', '']
    if report and report.get('generated_at'):
        lines.append('报告时间：'+str(report['generated_at']))
    values = report_view_geometry(report)
    if not values:
        lines.append('请先抓拍并评估；需要从图像求出共同管轴后才能计算断面有效基线。')
    for value in values:
        if value.get('zone_id'):
            lines.append('区域：'+str(value['zone_id']))
        if value['status'] != 'CONDITIONAL_GEOMETRY':
            lines.append('管轴或本报告的物理基线尚未确定，暂不显示几何数值。')
            continue
        lines += [f"物理双目基线：{value['physical_baseline_mm']:.2f} mm",
                  f"垂直管轴的有效分量：{value['baseline_perpendicular_to_axis_mm']:.2f} mm",
                  f"双目连线与管轴锐夹角：{value['baseline_axis_acute_angle_deg']:.2f}°",
                  f"横跨程度：{value['baseline_cross_axis_fraction']:.1%}（0% 沿管轴，100% 横跨管轴）"]
        if value['ground_trial']:
            lines.append('轴向使用地面试验约束，上述数值也属于条件诊断。')
        if value['ground_boundary_check_passed'] is False:
            lines.append('本次地面约束与管轴轮廓检查未通过；不能据此确认机位或尺寸已可靠。')
    lines += ['', '如何调整：',
              '先用实时地面调姿检查向下倾斜；再让两镜头连线更多地横跨管轴，保持管道在两眼内可见。',
              '调到垂直地面不会自动改善双目连线方向；相机绕光轴转动也会改变管道在图像中的方向。',
              '调整机位后重新抓拍、重画本次范围并评估；用同物、同光对照管径和相对心距的重复性。',
              '', '含义与限制：',
              '有效分量越大，通常越有利于用两眼轮廓求断面；这不是精度保证，也不改变物理基线。',
              '地面距离不能代替管道距离；更多相同机位帧不能增加新的观察方向或看穿遮挡。',
              '管径、相对位置和安装状态仍须通过各自检查，不能凭此页的数值点亮 CAD。']
    return lines
