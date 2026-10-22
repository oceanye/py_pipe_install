"""Constrain a horizontal pipe axis to an independently observed ground plane."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import numpy as np

from .section_diagnostics import section_basis

REVISION = 'ground-constrained-pipe-axis-v1'
POLICY = {'maximum_nullspace_ratio': .25, 'maximum_boundary_p95_px': 2.,
          'minimum_boundaries': 3, 'minimum_direction_strength': 1e-6}
POLICY_SHA256 = hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()


def _angle(a, b):
    return math.degrees(math.acos(float(np.clip(abs(np.dot(a, b)), 0, 1))))


def correct_pipe_axis(observations, intrinsic, unconstrained, ground):
    """Refit line direction in the ground tangent plane, not a post-hoc rotation.

    Pixels must use the same rectified intrinsic (right crop offsets restored).
    A correction affects subsequent free-radius fits; it supplies no depth,
    camera calibration, nominal radius or missing contour pixels.
    """
    audit = {'revision': REVISION, 'policy': dict(POLICY), 'policy_sha256': POLICY_SHA256,
             'status': 'NOT_CONFIGURED', 'applied': False,
             'assumption': 'PIPE_AXES_PARALLEL_TO_GROUND',
             'calibration_modified': False, 'nominal_dimensions_used': False}
    if not ground or ground.get('status') == 'NOT_CONFIGURED':
        return unconstrained, audit
    audit['ground_status'] = ground.get('status')
    if ground.get('status') != 'CONSISTENT_GROUND_PLANE':
        return unconstrained, audit | {'status': 'GROUND_CHECK_FAILED'}
    if ground.get('surface_identity_source') != 'OPERATOR_MARKED_GROUND':
        return unconstrained, audit | {'status': 'GROUND_REGION_UNCONFIRMED'}
    if len(observations) < POLICY['minimum_boundaries'] or {o['role'] for o in observations} != {'left', 'right'}:
        return unconstrained, audit | {'status': 'NO_USABLE_PIPE_BOUNDARIES'}
    normal = np.array((ground.get('left_plane') or {}).get('normal_camera'), float, copy=True)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-9:
        return unconstrained, audit | {'status': 'GROUND_NORMAL_INVALID'}
    normal /= np.linalg.norm(normal)
    k = np.asarray(intrinsic, float)
    lines = []
    for observation in observations:
        points = np.asarray(observation['points'], float)
        if len(points) < 2 or not np.isfinite(points).all() or np.ptp(points[:, 0]) < 1e-6:
            return unconstrained, audit | {'status': 'NO_USABLE_PIPE_BOUNDARIES'}
        slope, intercept = np.polyfit(points[:, 0], points[:, 1], 1)
        lines.append([slope, -1., intercept])
    normals = np.asarray(lines)@k
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    basis = section_basis(normal)
    _, singular, vectors = np.linalg.svd(normals@basis, full_matrices=False)
    if singular[0] < POLICY['minimum_direction_strength']:
        return unconstrained, audit | {'status': 'GROUND_AXIS_UNOBSERVABLE'}
    ratio = float(singular[-1]/singular[0])
    axis = basis@vectors[-1]
    reference = unconstrained.get('axis')
    if (reference is not None and axis@reference < 0) or (reference is None and axis[np.argmax(abs(axis))] < 0):
        axis = -axis
    audit.update(normal_camera=normal.tolist(), proposed_axis=axis.tolist(), nullspace_ratio=ratio,
                 axis_change_deg=_angle(axis, np.asarray(reference)) if reference is not None else None,
                 boundary_residuals=[])
    # The horizontal assumption must still explain each actual image boundary.
    # A line through the observed centre and constrained vanishing point gives
    # a finite-image residual, even when the vanishing point is at infinity.
    for observation in observations:
        points = np.asarray(observation['points'], float)
        center_ray = np.linalg.solve(k, np.r_[points.mean(axis=0), 1.])
        line = np.linalg.solve(k.T, np.cross(center_ray, axis))
        length = np.linalg.norm(line[:2])
        if length < 1e-12:
            return unconstrained, audit | {'status': 'GROUND_AXIS_UNOBSERVABLE'}
        residual = abs(np.c_[points, np.ones(len(points))]@line)/length
        audit['boundary_residuals'].append({'role': observation['role'], 'edge': observation['edge'],
                                          'p95_px': float(np.percentile(residual, 95))})
    if ratio > POLICY['maximum_nullspace_ratio'] or any(
            r['p95_px'] > POLICY['maximum_boundary_p95_px'] for r in audit['boundary_residuals']):
        return unconstrained, audit | {'status': 'GROUND_AXIS_CONFLICT'}
    corrected = {'status': 'STABLE', 'axis': axis.tolist(), 'source': 'GROUND_PLANE_CONSTRAINED',
                 'singular_values': singular.tolist(), 'nullspace_ratio': ratio,
                 'line_residual_degrees': np.degrees(np.arcsin(np.clip(abs(normals@axis), 0, 1))).tolist()}
    return corrected, audit | {'status': 'APPLIED', 'applied': True}


def finish_ground_correction(ground, stages):
    """Attach current corrected section results without refitting the ground."""
    if ground is None:
        return None
    result = copy.deepcopy(ground)
    corrections = [{'zone_id': stage.get('zone_id'), **stage.get('ground_correction', {
        'status': 'NO_USABLE_PIPE_BOUNDARIES', 'applied': False})} for stage in stages]
    applied = sum(bool(c.get('applied')) for c in corrections)
    result.update(correction_revision=REVISION, corrections=corrections,
                  pipe_axis_constrained_by_ground=bool(applied), pipe_axis_checks=[])
    result['correction_status'] = ('APPLIED' if applied and applied == len(corrections) else
                                   'PARTIALLY_APPLIED' if applied else 'NOT_APPLIED')
    if result.get('status') == 'CONSISTENT_GROUND_PLANE':
        normal = np.asarray(result['left_plane']['normal_camera'])
        for stage in stages:
            axis = stage.get('axis') or {}
            if axis.get('status') == 'STABLE':
                result['pipe_axis_checks'].append({'zone_id': stage.get('zone_id'), 'axis_status': 'STABLE',
                    'angle_to_ground_deg': math.degrees(math.asin(float(np.clip(abs(normal@axis['axis']), 0, 1)))),
                    'scope': 'CONSTRAINED_AXIS' if stage.get('ground_correction', {}).get('applied') else 'UNCONSTRAINED_AXIS'})
    return result


def correction_summary(correction):
    labels = {'NOT_CONFIGURED': '未设置地面区域', 'GROUND_CHECK_FAILED': '地面平面检查未通过',
        'GROUND_REGION_UNCONFIRMED': '请重新框选并确认实际地面', 'NO_USABLE_PIPE_BOUNDARIES': '缺少可用双目管道边界',
        'GROUND_NORMAL_INVALID': '地面法向无效', 'GROUND_AXIS_UNOBSERVABLE': '地面与边缘不足以确定管轴',
        'GROUND_AXIS_CONFLICT': '水平管假设与图像边界不一致'}
    if correction.get('applied'):
        delta = correction.get('axis_change_deg')
        suffix = f'（方向变化 {delta:.3f}°）' if delta is not None else ''
        return '已应用地面修正：按管轴平行地面重新求解'+suffix+'。'
    return '未应用地面修正：'+labels.get(correction.get('status'), str(correction.get('status')))+'；保留自由轴向结果。'
