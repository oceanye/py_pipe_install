"""Relative section placement decisions after boundary and dimension checks.

Visibility is not installation. Unmatched/occluded pipes remain UNKNOWN;
this module never declares NOT_INSTALLED or evaluates absolute CAD pose.
"""
from __future__ import annotations

import hashlib
import json
from itertools import combinations
import numpy as np

from .elevation_registration import register_elevation
from .section_diagnostics import section_basis

POLICY = {'revision': 'relative-section-position-v1', 'position_tolerance_mm': 5.,
          'clearance_tolerance_mm': 5., 'uncertainty_scenario': 'COHERENT_EDGE_OFFSETS',
          'confidence_multiplier': 1.96, 'scope': 'RELATIVE_COMMON_SECTION_ONLY'}
POLICY_SHA256 = hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()
STATE_ZH = {'IN_POSITION': '断面相对就位', 'OUT_OF_POSITION': '断面位置/尺寸不符', 'UNKNOWN': '就位待判定'}


def _interval_check(error, halfwidth, tolerance):
    if halfwidth is None or not np.isfinite(halfwidth) or halfwidth < 0:
        return 'UNKNOWN'
    if abs(error)+halfwidth <= tolerance:
        return 'IN_POSITION'
    if abs(error)-halfwidth > tolerance:
        return 'OUT_OF_POSITION'
    return 'UNKNOWN'


def compare_section_position(model, candidates, pipe_specs, registration, *, calibration_validated,
                             capture_healthy, truncated=False, matching=None):
    """Compare uniquely assigned sections, including uncertainty and gaps."""
    settings = {'diameter_tolerance_mm': 3., 'diameter_tolerance_ratio': .1} | dict(matching or {})
    gate_reasons = []
    if not calibration_validated:
        gate_reasons.append('CALIBRATION_NOT_VALIDATED')
    if not capture_healthy:
        gate_reasons.append('CAPTURE_UNHEALTHY')
    if truncated:
        gate_reasons.append('SECTION_SEARCH_TRUNCATED')
    if registration.get('status') != 'MATCHED':
        gate_reasons.append('SECTION_MODEL_CORRESPONDENCE_UNRESOLVED')
    observations = {c['id']: c for c in (model or {}).get('candidates', [])}
    quality = {c['candidate_id']: c for c in candidates}
    by_model = {m['pipe_id']: m['observation_id'] for m in registration.get('matches', [])}
    # Even malformed repeated correspondence must not create a positive state.
    if len(by_model) != len(registration.get('matches', [])) or len(set(by_model.values())) != len(by_model):
        gate_reasons.append('SECTION_MODEL_CORRESPONDENCE_NOT_ONE_TO_ONE')
    specs = {p['pipe_id']: p for p in pipe_specs}
    pair_lookup = {frozenset(p['ids']): p for p in (model or {}).get('pairs', [])}
    checks = {p: [] for p in specs}
    pair_rows = []
    if pipe_specs:
        line = np.asarray(pipe_specs[0]['centerline_world_mm'], float)
        basis = section_basis(line[1]-line[0])
    for first, second in combinations([p for p in by_model if p in specs], 2):
        pair = pair_lookup.get(frozenset([by_model[first], by_model[second]]))
        if pair is None:
            continue
        centers = [np.asarray(specs[p]['centerline_world_mm'], float).mean(axis=0) for p in (first, second)]
        expected = float(np.linalg.norm((centers[1]-centers[0])@basis))
        expected_gap = expected-sum(float(specs[p]['nominal_diameter_mm'])/2 for p in (first, second))
        error = pair['center_distance_mm']-expected
        gap_error = pair['surface_clearance_mm']-expected_gap
        uncertainty = pair.get('pixel_error', {}).get('scenarios', {}).get(POLICY['uncertainty_scenario'], {})
        center_half = uncertainty.get('center_distance_normal_95_halfwidth_mm')
        gap_half = uncertainty.get('surface_clearance_normal_95_halfwidth_mm')
        states = [_interval_check(error, center_half, POLICY['position_tolerance_mm']),
                  _interval_check(gap_error, gap_half, POLICY['clearance_tolerance_mm'])]
        row = {'pipe_ids': [first, second], 'model_center_distance_mm': expected,
               'observed_center_distance_mm': pair['center_distance_mm'], 'center_distance_error_mm': error,
               'model_surface_clearance_mm': expected_gap, 'observed_surface_clearance_mm': pair['surface_clearance_mm'],
               'surface_clearance_error_mm': gap_error, 'center_distance_95_halfwidth_mm': center_half,
               'surface_clearance_95_halfwidth_mm': gap_half, 'dimension_checks': states}
        pair_rows.append(row)
        for pipe_id in (first, second):
            checks[pipe_id].extend(states)
    rows = []
    for pipe_id, spec in specs.items():
        oid = by_model.get(pipe_id)
        measured, candidate = observations.get(oid), quality.get(oid)
        reasons = list(gate_reasons)
        own = {'pipe_id': pipe_id, 'observation_id': oid, 'observed_diameter_mm': None,
               'model_diameter_mm': float(spec['nominal_diameter_mm']), 'diameter_error_mm': None}
        if measured is None or candidate is None or measured.get('diameter_mm') is None:
            reasons.append('NO_UNIQUE_VISIBLE_SECTION')
        else:
            own['observed_diameter_mm'] = measured['diameter_mm']
            own['diameter_error_mm'] = measured['diameter_mm']-float(spec['nominal_diameter_mm'])
            reasons.extend(candidate.get('reason_codes', []))
            if candidate.get('boundary_status') != 'STABLE':
                reasons.append('POOLED_BOUNDARY_UNSTABLE')
            if not candidate.get('window_sensitivity', {}).get('stable'):
                reasons.append('POOLED_DIMENSIONS_UNSTABLE')
            sigma = candidate.get('pixel_error', {}).get('scenarios', {}).get(POLICY['uncertainty_scenario'], {}).get('diameter_sigma_mm')
            half = None if sigma is None else POLICY['confidence_multiplier']*sigma
            tolerance = max(settings['diameter_tolerance_mm'], settings['diameter_tolerance_ratio']*own['model_diameter_mm'])
            own.update(diameter_95_halfwidth_mm=half, diameter_tolerance_mm=tolerance)
            checks[pipe_id].append(_interval_check(own['diameter_error_mm'], half, tolerance))
            if not any(pipe_id in pair['pipe_ids'] for pair in pair_rows):
                reasons.append('NO_RELATIVE_SECTION_REFERENCE')
        state = 'UNKNOWN'
        if not reasons:
            if 'OUT_OF_POSITION' in checks[pipe_id]:
                state = 'OUT_OF_POSITION'; reasons = ['SECTION_DIMENSIONS_OUTSIDE_TOLERANCE']
            elif checks[pipe_id] and all(s == 'IN_POSITION' for s in checks[pipe_id]):
                state = 'IN_POSITION'; reasons = ['SECTION_DIAMETERS_AND_RELATIVE_GAPS_AGREE']
            else:
                reasons = ['SECTION_UNCERTAINTY_CROSSES_TOLERANCE']
        rows.append(own | {'state': state, 'state_zh': STATE_ZH[state], 'reason_codes': sorted(set(reasons))})
    return {'policy': dict(POLICY), 'policy_sha256': POLICY_SHA256, 'pipes': rows, 'pairwise': pair_rows,
            'counts': {s: sum(r['state'] == s for r in rows) for s in STATE_ZH},
            'absolute_installation_pose_assessed': False, 'absence_assessed': False,
            'calibration_validated': bool(calibration_validated), 'capture_healthy': bool(capture_healthy),
            'production_authority': False}


def assess_pooled_sections(pooled, pipe_specs, calibration, *, capture_healthy, settings, matching):
    model = pooled.get('model')
    model_by_id = {c['id']: c for c in (model or {}).get('candidates', [])}
    observations = []
    for item in pooled.get('candidates', []):
        section = model_by_id.get(item['candidate_id'])
        if (section is None or section.get('diameter_mm') is None or item.get('reason_codes')
                or not item.get('window_sensitivity', {}).get('stable')):
            continue
        observations.append({'observation_id': item['candidate_id'],
            'center_camera_mm': section['center_in_common_plane_camera_mm'],
            'axis_camera': model['section']['normal_camera'], 'diameter_mm': section['diameter_mm'],
            'color_identity_validated': False})
    ids = {o['observation_id'] for o in observations}
    anchors = {k: v for k, v in settings.get('anchors', {}).items() if k in ids}
    registration = register_elevation(pipe_specs, observations, settings.get('axis_world'), anchors=anchors,
        camera_side_world=settings.get('camera_side_world'), config={
            'diameter_tolerance_mm': matching.get('diameter_tolerance_mm', 3.),
            'diameter_tolerance_ratio': matching.get('diameter_tolerance_ratio', .1)})
    pooled['ignored_anchor_ids'] = sorted(set(settings.get('anchors', {}))-set(anchors))
    present_count = settings.get('present_pipe_count')
    if present_count is not None and len(observations) > present_count:
        registration = {'status': 'SCENE_INVENTORY_CONFLICT', 'matches': [],
                        'reason_codes': ['OBSERVATIONS_EXCEED_PRESENT_PIPE_COUNT']}
    pooled['registration'] = registration
    pooled['position_assessment'] = compare_section_position(model, pooled.get('candidates', []), pipe_specs, registration,
        calibration_validated=bool(calibration.validated), capture_healthy=capture_healthy,
        truncated=pooled.get('truncated', False), matching=matching)
    return pooled


def merge_position_assessments(assessments, pipe_specs):
    rows = []
    for spec in pipe_specs:
        candidates = [r for a in assessments for r in a.get('pipes', []) if r['pipe_id'] == spec['pipe_id']]
        definitive = [r for r in candidates if r['state'] != 'UNKNOWN']
        states = {r['state'] for r in definitive}
        if len(states) > 1:
            row = {'pipe_id': spec['pipe_id'], 'state': 'UNKNOWN', 'state_zh': STATE_ZH['UNKNOWN'], 'reason_codes': ['ZONE_POSITION_CONFLICT']}
        elif definitive:
            row = dict(definitive[0])
        elif candidates:
            row = dict(candidates[0])
        else:
            row = {'pipe_id': spec['pipe_id'], 'state': 'UNKNOWN', 'state_zh': STATE_ZH['UNKNOWN'], 'reason_codes': ['NO_UNIQUE_VISIBLE_SECTION']}
        rows.append(row)
    return {'policy': dict(POLICY), 'policy_sha256': POLICY_SHA256, 'pipes': rows,
            'counts': {s: sum(r['state'] == s for r in rows) for s in STATE_ZH}, 'scope': 'PER_ZONE_COMMON_SECTIONS',
            'absolute_installation_pose_assessed': False, 'absence_assessed': False, 'production_authority': False}
