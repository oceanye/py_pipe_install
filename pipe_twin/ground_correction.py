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
    trial = bool(ground and ground.get('candidate_trial_enabled'))
    audit['trial_mode'] = trial
    if not ground or ground.get('status') == 'NOT_CONFIGURED':
        return unconstrained, audit
    audit['ground_status'] = ground.get('status')
    if ground.get('status') != 'CONSISTENT_GROUND_PLANE' and not trial:
        return unconstrained, audit | {'status': 'GROUND_CHECK_FAILED'}
    if ground.get('surface_identity_source') != 'OPERATOR_MARKED_GROUND' and not trial:
        return unconstrained, audit | {'status': 'GROUND_REGION_UNCONFIRMED'}
    if len(observations) < POLICY['minimum_boundaries'] or {o['role'] for o in observations} != {'left', 'right'}:
        return unconstrained, audit | {'status': 'NO_USABLE_PIPE_BOUNDARIES'}
    plane = ground.get('left_plane') or {}
    if trial and plane.get('normal_camera') is None:
        plane = plane.get('candidate_plane') or {}
    normal = np.array(plane.get('normal_camera'), float, copy=True)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-9:
        return unconstrained, audit | {'status': 'GROUND_NORMAL_INVALID'}
    normal /= np.linalg.norm(normal)
    if trial:
        audit.update(scope='EXPERIMENTAL_GROUND_CONSTRAINT_NOT_ACCEPTED_MEASUREMENT',
                     trial_revision='candidate-ground-hypothesis-v1',
                     ground_quality_bypassed=ground.get('status') != 'CONSISTENT_GROUND_PLANE',
                     ground_identity_bypassed=ground.get('surface_identity_source') != 'OPERATOR_MARKED_GROUND',
                     original_ground_status=ground.get('status'))
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
    boundary_passed = not any(r['p95_px'] > POLICY['maximum_boundary_p95_px'] for r in audit['boundary_residuals'])
    audit['boundary_check_passed'] = boundary_passed
    if ratio > POLICY['maximum_nullspace_ratio'] or (not boundary_passed and not trial):
        return unconstrained, audit | {'status': 'GROUND_AXIS_CONFLICT'}
    if trial:
        # An explicit trial may inspect a conflicting finite hypothesis. Keep
        # the failed check and block metric acceptance; strict mode is unchanged.
        audit['boundary_consistency_bypassed'] = not boundary_passed
    corrected = {'status': 'STABLE', 'axis': axis.tolist(), 'source': 'GROUND_PLANE_CONSTRAINED',
                 'singular_values': singular.tolist(), 'nullspace_ratio': ratio,
                 'line_residual_degrees': np.degrees(np.arcsin(np.clip(abs(normals@axis), 0, 1))).tolist()}
    return corrected, audit | {'status': 'TRIAL_APPLIED' if trial else 'APPLIED', 'applied': True}


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
    if any(c.get('trial_mode') for c in corrections):
        result['correction_status'] = 'TRIAL_'+result['correction_status']
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
        prefix = '试验采用候选地面（未验证）：' if correction.get('trial_mode') else '已应用地面修正：'
        conflict = '；部分轮廓一致性未通过，保留冲突供对照' if correction.get('boundary_consistency_bypassed') else ''
        return prefix+'按管轴平行地面重新求解'+suffix+conflict+'。'
    return '未应用地面修正：'+labels.get(correction.get('status'), str(correction.get('status')))+'；保留自由轴向结果。'


def compare_ground_trial(before, after):
    """Compare two solves of identical images; changes do not imply accuracy."""
    from .temporal_sections import associate
    mapping = associate(after, before)
    old = {c['candidate_id']: c for c in before.get('candidates', [])}
    rows = []
    for c in after.get('candidates', []):
        a = old.get(mapping.get(c['candidate_id']), {}).get('fit', {})
        b = c.get('fit', {})
        rows.append({'candidate_id': c['candidate_id'], 'before_candidate_id': mapping.get(c['candidate_id']),
            'before_diameter_mm': a.get('trial_diameter_mm') if a.get('status') == 'CONDITIONAL_FIT' else None,
            'after_diameter_mm': b.get('trial_diameter_mm') if b.get('status') == 'CONDITIONAL_FIT' else None})
    old_pairs = {frozenset(p['ids']): p for p in (before.get('model') or {}).get('pairs', [])}
    pairs = []
    for p in (after.get('model') or {}).get('pairs', []):
        previous = old_pairs.get(frozenset(mapping.get(i) for i in p['ids']))
        if previous:
            pairs.append({'ids': p['ids'], 'before_center_distance_mm': previous['center_distance_mm'],
                          'after_center_distance_mm': p['center_distance_mm']})
    return {'scope': 'SAME_IMAGES_BEFORE_AFTER_NOT_ACCURACY_VALIDATION',
            'baseline_ground_correction': before.get('ground_correction'),
            'baseline_axis': before.get('axis'), 'trial_axis': after.get('axis'),
            'candidates': rows, 'pairs': pairs}
