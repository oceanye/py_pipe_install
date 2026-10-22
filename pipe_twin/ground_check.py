"""Operator-marked ground plane check using existing stereo samples only."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import numpy as np

REVISION = 'stereo-ground-plane-check-v1'
POLICY = {'maximum_points': 4096, 'ransac_trials': 96, 'plane_tolerance_mm': 5.,
          'minimum_points': 256, 'minimum_valid_fraction': .25, 'minimum_plane_fraction': .7,
          'minimum_spatial_cells': 8, 'minimum_second_spread_mm': 20.,
          'maximum_holdout_rms_mm': 3., 'maximum_eye_angle_deg': 2., 'maximum_eye_offset_mm': 10.,
          'minimum_roll_normal_projection': .08715574274765817}  # sin(5 degrees)
POLICY_SHA256 = hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()


def normalize_ground_region(payload, *, image_size=None):
    if payload is None:
        return None
    if not isinstance(payload, Mapping) or set(payload)-{'image_size', 'rect_px', 'coordinate_space', 'selection_source'}:
        raise ValueError('地面检查区域配置无效')
    size, rect = payload.get('image_size'), payload.get('rect_px')
    if payload.get('coordinate_space', 'rectified_left') != 'rectified_left':
        raise ValueError('地面区域必须使用左目矫正图')
    if not isinstance(size, (list, tuple)) or len(size) != 2 or any(type(v) is not int or v < 32 for v in size):
        raise ValueError('地面区域缺少有效图像尺寸')
    if image_size is not None and tuple(size) != tuple(image_size):
        raise ValueError('照片尺寸改变，请重新框选地面')
    if not isinstance(rect, (list, tuple)) or len(rect) != 4 or any(type(v) is not int for v in rect):
        raise ValueError('地面区域必须包含四个整数')
    x, y, w, h = rect
    if min(x, y) < 0 or min(w, h) < 32 or x+w > size[0] or y+h > size[1]:
        raise ValueError('地面区域应在图内且至少为 32×32 原图像素')
    source = payload.get('selection_source', 'OPERATOR')
    if source not in ('OPERATOR', 'ASSISTANT_DIAGNOSTIC'):
        raise ValueError('地面选区来源无效')
    result = {'coordinate_space': 'rectified_left', 'image_size': list(size), 'rect_px': list(rect)}
    if source != 'OPERATOR':
        result['selection_source'] = source
    return result


def _plane(points):
    center = points.mean(axis=0)
    _, singular, vectors = np.linalg.svd(points-center, full_matrices=False)
    normal = vectors[-1]; offset = -float(normal@center)
    if offset < 0:
        normal, offset = -normal, -offset
    return normal, offset, singular/np.sqrt(len(points))


def _fit_plane(points, cells):
    result = {'status': 'INSUFFICIENT_GROUND_SUPPORT', 'sample_count': len(points)}
    if len(points) < POLICY['minimum_points']:
        return result, None
    train = (cells % 4+cells//4) % 2 == 0
    if min(train.sum(), (~train).sum()) < 64:
        return result, None
    rng = np.random.default_rng(221026)
    training = points[train]
    best = None; score = 0
    for _ in range(POLICY['ransac_trials']):
        a, b, c = training[rng.choice(len(training), 3, replace=False)]
        normal = np.cross(b-a, c-a); length = np.linalg.norm(normal)
        if length < 1e-8:
            continue
        normal /= length
        selected = np.abs((training-a)@normal) <= POLICY['plane_tolerance_mm']
        if selected.sum() > score:
            best, score = selected, int(selected.sum())
    if best is None or score < 64:
        return result, None
    normal, offset, _ = _plane(training[best])
    residual = points@normal+offset
    inliers = np.abs(residual) <= POLICY['plane_tolerance_mm']
    holdout = residual[~train]
    # Holdout majority has to agree with the training plane, before refitting.
    holdout_inliers = abs(holdout) <= POLICY['plane_tolerance_mm']
    rms = float(np.sqrt(np.mean(holdout[holdout_inliers]**2))) if holdout_inliers.any() else float('inf')
    cell_count = len(np.unique(cells[inliers]))
    result.update(inlier_fraction=float(inliers.mean()), holdout_inlier_fraction=float(holdout_inliers.mean()),
                  holdout_rms_mm=rms if np.isfinite(rms) else None, occupied_cells=cell_count)
    if (inliers.mean() < POLICY['minimum_plane_fraction'] or holdout_inliers.mean() < POLICY['minimum_plane_fraction']
            or cell_count < POLICY['minimum_spatial_cells'] or rms > POLICY['maximum_holdout_rms_mm']):
        result['status'] = 'GROUND_PLANE_INCONSISTENT'
        return result, None
    normal, offset, spread = _plane(points[inliers])
    if spread[1] < POLICY['minimum_second_spread_mm']:
        result['status'] = 'GROUND_SUPPORT_TOO_NARROW'
        return result, None
    result.update(status='PLANE_FIT', normal_camera=normal.tolist(), offset_mm=offset,
                  second_spread_mm=float(spread[1]))
    return result, inliers


def _backproject(pixels, depth, intrinsic, origin):
    rays = np.c_[pixels, np.ones(len(pixels))]@np.linalg.inv(intrinsic).T
    return rays*depth[:, None]+origin


def check_ground(group, calibration, region, *, section_stages=()):
    result = {'revision': REVISION, 'policy': dict(POLICY), 'policy_sha256': POLICY_SHA256,
              'status': 'NOT_CONFIGURED', 'accepted_measurement': False, 'total_accuracy_known': False,
              'pipe_axis_constrained_by_ground': False, 'calibration_validated': bool(calibration.validated)}
    if region is None:
        return result
    region = normalize_ground_region(region, image_size=(calibration.left.width, calibration.left.height))
    source = ('ASSISTANT_SUGGESTED_GROUND_NOT_CONFIRMED' if region.get('selection_source') == 'ASSISTANT_DIAGNOSTIC'
              else 'OPERATOR_MARKED_GROUND')
    result.update(selection=region, surface_identity_source=source,
                  status='INSUFFICIENT_GROUND_DEPTH')
    if not group.get('pair_healthy', False):
        return result | {'status': 'CAPTURE_UNHEALTHY'}
    depth = group['depth']; x, y, w, h = region['rect_px']
    z = np.asarray(depth.left_depth_mm)[y:y+h, x:x+w]
    valid = np.asarray(depth.left_valid)[y:y+h, x:x+w] & np.isfinite(z) & (z > 0)
    result['valid_depth_fraction'] = float(valid.mean())
    yy, xx = np.nonzero(valid)
    if len(xx) < POLICY['minimum_points'] or valid.mean() < POLICY['minimum_valid_fraction']:
        return result
    sample = np.linspace(0, len(xx)-1, min(len(xx), POLICY['maximum_points']), dtype=int)
    xx, yy = xx[sample], yy[sample]
    cells = np.minimum(3, xx*4//w)+4*np.minimum(3, yy*4//h)
    kl, kr = calibration.left.rectified_intrinsic, calibration.right.rectified_intrinsic
    points = _backproject(np.c_[xx+x, yy+y], z[yy, xx], kl, np.zeros(3))
    left, inliers = _fit_plane(points, cells)
    result['left_plane'] = left
    if inliers is None:
        return result | {'status': left['status']}
    base = np.array([calibration.baseline_mm, 0., 0.])
    projected = (points[inliers]-base)@kr.T
    uv = np.round(projected[:, :2]/projected[:, 2:]).astype(int)
    eligible = (uv[:, 0] >= 0) & (uv[:, 0] < calibration.right.width) & (uv[:, 1] >= 0) & (uv[:, 1] < calibration.right.height)
    uv, right_cells = uv[eligible], cells[inliers][eligible]
    # A right pixel counts once, even when several left samples round to it.
    _, unique = np.unique(uv, axis=0, return_index=True)
    uv, right_cells = uv[unique], right_cells[unique]
    rz = np.asarray(depth.right_depth_mm)[uv[:, 1], uv[:, 0]]
    good = np.asarray(depth.right_valid)[uv[:, 1], uv[:, 0]] & np.isfinite(rz) & (rz > 0)
    result['right_confirmed_fraction'] = float(good.sum()/max(1, inliers.sum()))
    if result['right_confirmed_fraction'] < .5:
        return result | {'status': 'INSUFFICIENT_RIGHT_GROUND_SUPPORT'}
    right_points = _backproject(uv[good], rz[good], kr, base)
    right, _ = _fit_plane(right_points, right_cells[good])
    result['right_plane'] = right
    if right['status'] != 'PLANE_FIT':
        return result | {'status': 'RIGHT_GROUND_PLANE_INCONSISTENT'}
    n = np.array(left['normal_camera']); nr = np.array(right['normal_camera'])
    angle = math.degrees(math.acos(float(np.clip(n@nr, -1, 1))))
    difference = abs(left['offset_mm']-right['offset_mm'])
    result.update(left_right_normal_angle_deg=angle, left_right_plane_offset_difference_mm=difference)
    if angle > POLICY['maximum_eye_angle_deg'] or difference > POLICY['maximum_eye_offset_mm']:
        return result | {'status': 'LEFT_RIGHT_GROUND_INCONSISTENT'}
    roll_observable = float(np.hypot(n[0], n[1])) >= POLICY['minimum_roll_normal_projection']
    result.update(status='CONSISTENT_GROUND_PLANE', camera_to_ground_perpendicular_mm=left['offset_mm'],
        ground_depth_median_mm=float(np.median(points[inliers, 2])),
        optical_axis_to_ground_deg=math.degrees(math.asin(float(np.clip(abs(n[2]), 0, 1)))),
        image_roll_relative_to_ground_deg=math.degrees(math.atan2(n[0], -n[1])) if roll_observable else None,
        roll_observable=roll_observable, pipe_axis_checks=[])
    for stage in section_stages:
        axis = (stage.get('axis') or {}).get('axis')
        if axis is not None and stage['axis'].get('status') == 'STABLE':
            axis = np.asarray(axis)/np.linalg.norm(axis)
            result['pipe_axis_checks'].append({'zone_id': stage.get('zone_id'),
                'angle_to_ground_deg': math.degrees(math.asin(float(np.clip(abs(n@axis), 0, 1)))),
                'axis_status': stage['axis'].get('status'), 'scope': 'DIAGNOSTIC_NOT_A_FORCED_CONSTRAINT'})
    return result


def ground_summary_lines(check):
    lines = ['请在左图框选确定的裸露地面，避开管道、支架和其他物体。',
             '垂直离地距离与沿视线的深度是两个不同量；水平放管不等于两端到相机等距。']
    if not check or check.get('status') == 'NOT_CONFIGURED':
        return lines+['尚未设置地面区域。框选后点击④保存并评估。']
    if check.get('status') == 'PENDING_EVALUATION':
        return lines+['已设置地面区域；点击④保存并评估，通过检查后自动约束水平管轴并重新计算断面。']
    if check.get('surface_identity_source') == 'ASSISTANT_SUGGESTED_GROUND_NOT_CONFIRMED':
        lines.append('当前选区是离线分析建议，尚未经你确认是地面；请核对紫色框。')
    if check.get('correction_revision'):
        from .ground_correction import correction_summary
        for item in check.get('corrections', []):
            lines.append((item.get('zone_id') or '全幅')+'：'+correction_summary(item))
    labels = {'INSUFFICIENT_GROUND_DEPTH': '地面有效深度不足', 'CAPTURE_UNHEALTHY': '拍摄质量未通过',
        'INSUFFICIENT_GROUND_SUPPORT': '地面二维采样覆盖不足', 'GROUND_PLANE_INCONSISTENT': '地面样本不符合稳定平面',
        'GROUND_SUPPORT_TOO_NARROW': '地面区域过窄，法向不可靠', 'INSUFFICIENT_RIGHT_GROUND_SUPPORT': '右目对应地面支持不足',
        'RIGHT_GROUND_PLANE_INCONSISTENT': '右目地面平面未通过', 'LEFT_RIGHT_GROUND_INCONSISTENT': '左右目地面结果不一致'}
    status = check.get('status')
    if status != 'CONSISTENT_GROUND_PLANE':
        return lines+['检查未通过：'+labels.get(status, str(status)), '尚不输出确定的地面距离或相机倾斜角。']
    roll = check.get('image_roll_relative_to_ground_deg')
    roll_text = f'画面相对地面横滚：{roll:.2f}°' if roll is not None else '横滚不可可靠观测：光轴接近垂直地面。'
    lines += [f"相机到地面的垂直距离（暂估）：{check['camera_to_ground_perpendicular_mm']:.1f} mm",
              f"选区沿相机 Z 方向深度中位数：{check['ground_depth_median_mm']:.1f} mm",
              f"光轴与地面的夹角：{check['optical_axis_to_ground_deg']:.2f}°（接近 0° 表示光轴平行于地面）",
              roll_text,
              f"左右平面法向差：{check['left_right_normal_angle_deg']:.2f}°；距离差：{check['left_right_plane_offset_difference_mm']:.1f} mm"]
    for row in check.get('pipe_axis_checks', []):
        suffix = '约束后的角度，不是独立核验' if row.get('scope') == 'CONSTRAINED_AXIS' else '水平管应接近 0°'
        lines.append(f"{row.get('zone_id') or '全幅'}管轴与地面夹角：{row['angle_to_ground_deg']:.2f}°（{suffix}）")
    if not check.get('calibration_validated'):
        lines.append('当前为暂用标定，绝对距离与角度精度尚未验证。')
    return lines+(['地面修正用于管段汇聚与断面模型，不修改相机标定；应用修正不表示安装就位已通过。']
                  if check.get('correction_revision') else ['此报告仅做地面检查；重新评估后可应用地面修正。'])
