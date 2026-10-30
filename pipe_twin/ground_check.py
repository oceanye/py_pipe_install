"""Operator-marked ground plane check using existing stereo samples only."""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import cv2
import numpy as np

REVISION = 'stereo-ground-plane-check-v4'
POLICY = {'maximum_points': 4096, 'ransac_trials': 96, 'plane_tolerance_mm': 5.,
          'minimum_points': 256, 'minimum_valid_fraction': .25, 'minimum_plane_fraction': .7,
          'minimum_spatial_cells': 8, 'minimum_second_spread_mm': 20.,
          'maximum_holdout_rms_mm': 3., 'maximum_eye_angle_deg': 2., 'maximum_eye_offset_mm': 10.,
          'minimum_roll_normal_projection': .08715574274765817}  # sin(5 degrees)
POLICY_SHA256 = hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()
SUBPIXEL_POLICY = {'revision': 'ground-bidirectional-subpixel-v1', 'window_px': 21, 'pyramid_levels': 1,
                  'maximum_epipolar_error_px': .5, 'maximum_roundtrip_error_px': .5,
                  'maximum_seed_change_px': 2., 'minimum_retained_fraction': .25}


def _refine_ground_samples(images, pixels, depths, intrinsic, baseline, role):
    """Refine observed correspondences before fitting; no planar depth filling."""
    source = images[role]
    other = images['right' if role == 'left' else 'left']
    gray = lambda image: cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    source, other = gray(source), gray(other)
    audit = {'policy': dict(SUBPIXEL_POLICY), 'input_sample_count': len(pixels),
             'kept_sample_count': 0, 'retained_fraction': 0., 'status': 'INSUFFICIENT_SUBPIXEL_SUPPORT',
             'selection_uses_plane': False, 'inferred_pixels': 0}
    empty = np.zeros(len(pixels), bool)
    if not len(pixels):
        return np.empty((0, 3)), empty, audit
    sign = -1. if role == 'left' else 1.
    seed_disparity = float(intrinsic[0, 0])*baseline/np.asarray(depths)
    start = np.asarray(pixels, np.float32).reshape(-1, 1, 2)
    guess = start.copy(); guess[:, 0, 0] += sign*seed_disparity
    options = dict(winSize=(SUBPIXEL_POLICY['window_px'],)*2, maxLevel=SUBPIXEL_POLICY['pyramid_levels'],
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .001),
                   flags=cv2.OPTFLOW_USE_INITIAL_FLOW, minEigThreshold=1e-4)
    try:
        matched, forward, _ = cv2.calcOpticalFlowPyrLK(source, other, start, guess, **options)
        if matched is None:
            return np.empty((0, 3)), empty, audit
        finite = np.isfinite(matched).all(axis=(1, 2))
        safe_matched = matched.copy(); safe_matched[~finite] = 0.
        returned, reverse, _ = cv2.calcOpticalFlowPyrLK(other, source, safe_matched, start.copy(), **options)
        if returned is None:
            return np.empty((0, 3)), empty, audit
    except cv2.error:
        audit['status'] = 'SUBPIXEL_MATCH_FAILED'
        return np.empty((0, 3)), empty, audit
    finite &= np.isfinite(returned).all(axis=(1, 2))
    difference = matched[:, 0]-start[:, 0]
    disparity = sign*difference[:, 0]
    border = SUBPIXEL_POLICY['window_px']//2
    h, w = source.shape
    inside = ((start[:, 0, 0] >= border) & (start[:, 0, 0] < w-border)
              & (start[:, 0, 1] >= border) & (start[:, 0, 1] < h-border)
              & (matched[:, 0, 0] >= border) & (matched[:, 0, 0] < w-border)
              & (matched[:, 0, 1] >= border) & (matched[:, 0, 1] < h-border))
    reasons = {
        'TRACKING_FAILED': ~(forward.ravel().astype(bool) & reverse.ravel().astype(bool) & finite),
        'PATCH_OUTSIDE_IMAGE': ~inside,
        'EPIPOLAR_MISMATCH': abs(difference[:, 1]) > SUBPIXEL_POLICY['maximum_epipolar_error_px'],
        'ROUNDTRIP_MISMATCH': np.linalg.norm(returned[:, 0]-start[:, 0], axis=1) > SUBPIXEL_POLICY['maximum_roundtrip_error_px'],
        'DISPARITY_SEED_MISMATCH': abs(disparity-seed_disparity) > SUBPIXEL_POLICY['maximum_seed_change_px'],
        'NON_POSITIVE_DISPARITY': disparity <= 0,
    }
    valid = ~np.logical_or.reduce(list(reasons.values()))
    audit.update(kept_sample_count=int(valid.sum()), retained_fraction=float(valid.mean()),
                 rejection_counts={key: int(mask.sum()) for key, mask in reasons.items()},
                 rejection_counts_overlap=True)
    if valid.sum() >= POLICY['minimum_points'] and valid.mean() >= SUBPIXEL_POLICY['minimum_retained_fraction']:
        audit['status'] = 'REFINED_MATCHES'
    z = float(intrinsic[0, 0])*baseline/disparity[valid]
    origin = np.array([baseline if role == 'right' else 0., 0., 0.])
    return _backproject(np.asarray(pixels)[valid], z, intrinsic, origin), valid, audit


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
    result = {'status': 'INSUFFICIENT_GROUND_SUPPORT', 'sample_count': len(points),
              'fit_scope': 'TRAINING_CELLS_ONLY', 'validation_scope': 'SPATIAL_HELD_OUT_CELLS',
              'failure_reasons': [], 'quality_checks': []}
    if len(points) < POLICY['minimum_points']:
        result['failure_reasons'] = ['GROUND_TOO_FEW_SAMPLES']
        return result, None
    train = (cells % 4+cells//4) % 2 == 0
    result.update(training_sample_count=int(train.sum()), holdout_sample_count=int((~train).sum()))
    if min(train.sum(), (~train).sum()) < 64:
        result['failure_reasons'] = ['GROUND_SPATIAL_SPLIT_SUPPORT_LOW']
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
        result['failure_reasons'] = ['GROUND_NO_TRAINING_CONSENSUS']
        return result, None
    normal, offset, spread = _plane(training[best])
    residual = points@normal+offset
    inliers = np.abs(residual) <= POLICY['plane_tolerance_mm']
    holdout = residual[~train]
    # Holdout majority must agree with the training plane. Keep this same plane
    # for output and correction: refitting on holdout would invalidate its audit.
    holdout_inliers = abs(holdout) <= POLICY['plane_tolerance_mm']
    rms = float(np.sqrt(np.mean(holdout[holdout_inliers]**2))) if holdout_inliers.any() else float('inf')
    cell_count = len(np.unique(cells[inliers]))
    result.update(inlier_fraction=float(inliers.mean()), training_inlier_fraction=float(inliers[train].mean()),
                  holdout_inlier_fraction=float(holdout_inliers.mean()),
                  holdout_rms_mm=rms if np.isfinite(rms) else None, occupied_cells=cell_count)
    # Preserve a rejected hypothesis for inspection, never as an accepted plane.
    result['candidate_plane'] = {'normal_camera': normal.tolist(), 'offset_mm': offset}
    checks = [
        ('GROUND_PLANE_SUPPORT_FRACTION_LOW', float(inliers.mean()), POLICY['minimum_plane_fraction'], '>='),
        ('GROUND_HOLDOUT_SUPPORT_FRACTION_LOW', float(holdout_inliers.mean()), POLICY['minimum_plane_fraction'], '>='),
        ('GROUND_SPATIAL_COVERAGE_LOW', cell_count, POLICY['minimum_spatial_cells'], '>='),
        ('GROUND_HOLDOUT_RMS_HIGH', rms if np.isfinite(rms) else None, POLICY['maximum_holdout_rms_mm'], '<='),
    ]
    result['quality_checks'] = [dict(reason_code=code, value=value, limit=limit, comparison=comparison,
        passed=bool(value is not None and (value >= limit if comparison == '>=' else value <= limit)))
        for code, value, limit, comparison in checks]
    result['failure_reasons'] = [item['reason_code'] for item in result['quality_checks'] if not item['passed']]
    if result['failure_reasons']:
        result['status'] = 'GROUND_PLANE_INCONSISTENT'
        return result, None
    result['quality_checks'].append(dict(reason_code='GROUND_SECOND_SPREAD_LOW', value=float(spread[1]),
        limit=POLICY['minimum_second_spread_mm'], comparison='>=',
        passed=bool(spread[1] >= POLICY['minimum_second_spread_mm'])))
    if spread[1] < POLICY['minimum_second_spread_mm']:
        result['status'] = 'GROUND_SUPPORT_TOO_NARROW'
        result['failure_reasons'] = ['GROUND_SECOND_SPREAD_LOW']
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
    images = {role: group.get(role) for role in ('left', 'right')}
    refine = any(image is not None for image in images.values())
    if refine and (not np.allclose(kl, kr, atol=1e-9, rtol=0)
                   or (calibration.left.height, calibration.left.width)
                   != (calibration.right.height, calibration.right.width)):
        return result | {'status': 'GROUND_RECTIFIED_GEOMETRY_UNSUPPORTED'}
    if refine and not all(isinstance(image, np.ndarray) and image.dtype == np.uint8
                 and (image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 3))
                 and image.shape[:2] == (calibration.left.height, calibration.left.width)
                 for image in images.values()):
        return result | {'status': 'GROUND_IMAGE_PAIR_INVALID'}
    if refine:
        result['raw_depth_plane'] = _fit_plane(points, cells)[0]
        points, kept, audit = _refine_ground_samples(images, np.c_[xx+x, yy+y], z[yy, xx],
                                                    kl, calibration.baseline_mm, 'left')
        result['subpixel_refinement'] = {'scope': 'OBSERVED_STEREO_PATCHES_NOT_PLANAR_DEPTH_FILLING',
                                        'left': audit}
        result['estimated_refined_depth_fraction'] = result['valid_depth_fraction']*audit['retained_fraction']
        xx, yy, cells = xx[kept], yy[kept], cells[kept]
        if (audit['status'] != 'REFINED_MATCHES'
                or result['estimated_refined_depth_fraction'] < POLICY['minimum_valid_fraction']):
            return result | {'status': 'INSUFFICIENT_GROUND_SUBPIXEL_SUPPORT'}
    left, inliers = _fit_plane(points, cells)
    result['left_plane'] = left
    from .ground_visualization import observed_support
    plane = left if left['status'] == 'PLANE_FIT' else left.get('candidate_plane')
    if plane:
        if refine:
            # Show only the measured refined samples. Never paint an entire
            # ROI or reuse noisy dense pixels as refined-plane evidence.
            support_z = np.full(z.shape, np.nan)
            support_valid = np.zeros(z.shape, bool)
            support_z[yy, xx] = points[:, 2]; support_valid[yy, xx] = True
        else:
            support_z, support_valid = z, valid
        result['observed_support'] = observed_support(support_z, support_valid, kl, region, plane, POLICY['plane_tolerance_mm'])
        if refine:
            result['observed_support']['sampling'] = 'SPARSE_REFINED_STEREO_MATCHES'
    if inliers is None:
        return result | {'status': left['status']}
    base = np.array([calibration.baseline_mm, 0., 0.])
    # With patch refinement, the independent right-plane solve samples all
    # retained left correspondences, not only the left plane's inliers.
    selected_points = points if refine else points[inliers]
    selected_cells = cells if refine else cells[inliers]
    projected = (selected_points-base)@kr.T
    uv = np.round(projected[:, :2]/projected[:, 2:]).astype(int)
    eligible = (uv[:, 0] >= 0) & (uv[:, 0] < calibration.right.width) & (uv[:, 1] >= 0) & (uv[:, 1] < calibration.right.height)
    uv, right_cells = uv[eligible], selected_cells[eligible]
    # A right pixel counts once, even when several left samples round to it.
    _, unique = np.unique(uv, axis=0, return_index=True)
    uv, right_cells = uv[unique], right_cells[unique]
    rz = np.asarray(depth.right_depth_mm)[uv[:, 1], uv[:, 0]]
    good = np.asarray(depth.right_valid)[uv[:, 1], uv[:, 0]] & np.isfinite(rz) & (rz > 0)
    result['right_confirmed_fraction'] = float(good.sum()/max(1, len(selected_points)))
    if result['right_confirmed_fraction'] < .5:
        return result | {'status': 'INSUFFICIENT_RIGHT_GROUND_SUPPORT'}
    if refine:
        right_points, kept_right, right_audit = _refine_ground_samples(images, uv[good], rz[good],
                                                                      kr, calibration.baseline_mm, 'right')
        result['subpixel_refinement']['right'] = right_audit
        right_cells = right_cells[good][kept_right]
        result['raw_right_confirmed_fraction'] = result['right_confirmed_fraction']
        result['right_confirmed_fraction'] *= right_audit['retained_fraction']
        if right_audit['status'] != 'REFINED_MATCHES' or result['right_confirmed_fraction'] < .5:
            return result | {'status': 'INSUFFICIENT_RIGHT_GROUND_SUBPIXEL_SUPPORT'}
    else:
        right_points = _backproject(uv[good], rz[good], kr, base)
        right_cells = right_cells[good]
    right, _ = _fit_plane(right_points, right_cells)
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
        return lines+['尚未设置地面区域。框选后点击“倾角检查”，或点击⑤保存并评估。']
    if check.get('status') == 'PENDING_EVALUATION':
        return lines+['已设置地面区域；点击“倾角检查”单独计算。点击⑤保存并评估，通过检查后自动约束水平管轴并重新计算断面。']
    from .ground_visualization import tilt_label_lines
    lines += tilt_label_lines(check)
    if check.get('source_scope') == 'REFERENCE_STEREO_PAIR_ONLY':
        lines.append('地面仅使用当前末帧双目照片；原连拍未通过，本次地面检查不恢复多帧或安装验收。')
        if check.get('source_capture_id'):
            lines.append('地面来源帧：'+str(check['source_capture_id']))
        if check.get('source_burst_reason_codes'):
            lines.append('连拍未通过原因：'+'、'.join(check['source_burst_reason_codes']))
    if check.get('subpixel_refinement'):
        lines.append('地面采用图像块双向子像素匹配；涂色仅为真实稀疏匹配样本，不填充整块地面。')
        for role, name in [('left', '左目'), ('right', '右目')]:
            audit = check['subpixel_refinement'].get(role)
            if audit:
                lines.append(f"{name}精化保留 {audit['kept_sample_count']}/{audit['input_sample_count']} 点"
                             f"（{audit['retained_fraction']:.1%}）；筛选不使用平面拟合残差。")
    policy = check.get('policy') or POLICY
    coverage = check.get('valid_depth_fraction')
    if coverage is not None:
        lines.append(f"选区有效深度：{coverage:.1%}；要求至少 {policy['minimum_valid_fraction']:.0%}。")
    failure_labels = {
        'GROUND_PLANE_SUPPORT_FRACTION_LOW': '平面一致点不足',
        'GROUND_HOLDOUT_SUPPORT_FRACTION_LOW': '留出一致点不足',
        'GROUND_SPATIAL_COVERAGE_LOW': '平面空间覆盖不足',
        'GROUND_HOLDOUT_RMS_HIGH': '留出残差过大',
        'GROUND_SECOND_SPREAD_LOW': '平面第二方向跨度不足',
        'GROUND_TOO_FEW_SAMPLES': '有效采样点不足',
        'GROUND_SPATIAL_SPLIT_SUPPORT_LOW': '训练或留出分区采样不足',
        'GROUND_NO_TRAINING_CONSENSUS': '训练点未找到平面共识',
    }
    for key, name in [('left_plane', '左目'), ('right_plane', '右目')]:
        plane = check.get(key) or {}
        if 'inlier_fraction' in plane:
            lines.append(f"{name}平面一致点 {plane['inlier_fraction']:.1%}；留出一致点 "
                         f"{plane['holdout_inlier_fraction']:.1%}（均需 ≥{policy['minimum_plane_fraction']:.0%}，"
                         f"点到平面 ≤{policy['plane_tolerance_mm']:g} mm）。")
            rms = plane.get('holdout_rms_mm')
            lines.append(f"{name}留出一致点 RMS：" + (f"{rms:.2f} mm" if rms is not None else '无可用值')
                         + f"；要求 ≤{policy['maximum_holdout_rms_mm']:g} mm。")
        failures = plane.get('failure_reasons', [])
        if failures:
            lines.append(name+'未通过项目：'+'、'.join(failure_labels.get(code, code) for code in failures)+'。')
    if check.get('surface_identity_source') == 'ASSISTANT_SUGGESTED_GROUND_NOT_CONFIRMED':
        lines.append('当前选区是离线分析建议，尚未经你确认是地面；请核对紫色框。')
    if check.get('correction_revision'):
        from .ground_correction import correction_summary
        for item in check.get('corrections', []):
            lines.append((item.get('zone_id') or '全幅')+'：'+correction_summary(item))
    labels = {'INSUFFICIENT_GROUND_DEPTH': '地面有效深度不足', 'CAPTURE_UNHEALTHY': '拍摄质量未通过',
        'GROUND_IMAGE_PAIR_INVALID': '地面精化图像缺失或尺寸不匹配',
        'GROUND_RECTIFIED_GEOMETRY_UNSUPPORTED': '地面精化要求左右矫正内参及图像尺寸一致',
        'INSUFFICIENT_GROUND_SUBPIXEL_SUPPORT': '左目地面子像素匹配支持不足',
        'INSUFFICIENT_RIGHT_GROUND_SUBPIXEL_SUPPORT': '右目地面子像素匹配支持不足',
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
