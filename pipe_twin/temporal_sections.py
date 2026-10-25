"""Repeatability checks for independently reconstructed static-burst sections.

Only the current synchronized burst is eligible. The last frame supplies the
displayed geometry; prior frames neither fill missing contours nor become new
current pipes. Relative distances avoid comparing unrelated axial stations.
"""
from __future__ import annotations

import hashlib
import json
import math
from itertools import combinations
import numpy as np

POLICY = {'revision': 'static-section-repeatability-v1', 'maximum_frames': 7,
          'minimum_frames': 3, 'minimum_fraction': .8, 'minimum_box_iou': .5,
          'diameter_span_mm': 1., 'diameter_span_ratio': .05, 'pair_distance_span_mm': 5.}
POLICY_SHA256 = hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()


def sample_indices(count):
    return np.linspace(0, count-1, min(count, POLICY['maximum_frames'])).round().astype(int).tolist() if count else []


def _iou(first, second):
    a, b = np.asarray(first, float), np.asarray(second, float)
    intersection = np.maximum(0, np.minimum(a[:2]+a[2:], b[:2]+b[2:])-np.maximum(a[:2], b[:2])).prod()
    return float(intersection/max(a[2:].prod()+b[2:].prod()-intersection, 1))


def associate(reference, other):
    """Unique stereo geometry + appearance, never candidate number or CAD size."""
    from .pooled_sections import _compatible_appearance
    first, second = reference.get('candidates', []), other.get('candidates', [])
    views_a = {v['candidate_id']: v for v in reference.get('views', {}).get('left', [])}
    views_b = {v['candidate_id']: v for v in other.get('views', {}).get('left', [])}
    matrix = np.zeros((len(first), len(second)), bool)
    for i, a in enumerate(first):
        for j, b in enumerate(second):
            va, vb = views_a.get(a['left_candidate_id']), views_b.get(b['left_candidate_id'])
            matrix[i, j] = (va is not None and vb is not None and _compatible_appearance(va, vb)
                and all(_iou(a[r+'_region_px'], b[r+'_region_px']) >= POLICY['minimum_box_iou'] for r in ('left', 'right')))
    return {first[i]['candidate_id']: second[j]['candidate_id'] for i, j in zip(*np.nonzero(matrix))
            if matrix[i].sum() == 1 and matrix[:, j].sum() == 1}


def _range(values):
    return {'median_mm': float(np.median(values)), 'min_mm': float(min(values)),
            'max_mm': float(max(values)), 'span_mm': float(max(values)-min(values))} if values else None


def check_sections(latest, stages, sources, *, requested_frames):
    """Attach diagnostics, leaving every fitted centre, radius and covariance intact."""
    if len(stages) != len(sources) or len(stages) < POLICY['minimum_frames'] or stages[-1] is not latest:
        raise ValueError('SECTION_BURST_REQUIRES_CURRENT_REFERENCE_AND_THREE_FRAMES')
    count = len(stages); required = max(3, math.ceil(count*POLICY['minimum_fraction']))
    maps = [associate(latest, stage) for stage in stages]
    candidates = [{c['candidate_id']: c for c in s.get('candidates', [])} for s in stages]
    models = [{c['id']: c for c in (s.get('model') or {}).get('candidates', [])} for s in stages]
    pairs = []
    latest_ids = [c['candidate_id'] for c in latest.get('candidates', [])]
    for a, b in combinations(latest_ids, 2):
        distances = []
        for mapping, model in zip(maps, models):
            x, y = model.get(mapping.get(a)), model.get(mapping.get(b))
            if x and y and x.get('estimated_center_uv_mm') is not None and y.get('estimated_center_uv_mm') is not None:
                distances.append(float(np.linalg.norm(np.asarray(x['estimated_center_uv_mm'])-y['estimated_center_uv_mm'])))
        values = _range(distances)
        pairs.append({'ids': [a, b], 'sample_count': len(distances), 'distance': values,
                      'repeatable': bool(len(distances) >= required and values['span_mm'] <= POLICY['pair_distance_span_mm'])})
    for row in latest.get('candidates', []):
        identity = row['candidate_id']; values, frame_results = [], []
        for index, (mapping, by_id) in enumerate(zip(maps, candidates)):
            match = by_id.get(mapping.get(identity))
            fit = (match or {}).get('fit', {})
            diameter = fit.get('trial_diameter_mm') if fit.get('status') == 'CONDITIONAL_FIT' and match.get('included_in_section_model', True) else None
            if diameter is not None:
                values.append(float(diameter))
            frame_results.append({'capture_id': sources[index]['capture_id'], 'candidate_id': mapping.get(identity),
                                  'diameter_mm': diameter, 'reason_codes': (match or {}).get('reason_codes', [])})
        spread = _range(values)
        tolerance = max(POLICY['diameter_span_mm'], POLICY['diameter_span_ratio']*spread['median_mm']) if spread else None
        repeatable = bool(len(values) >= required and spread['span_mm'] <= tolerance)
        related = [p for p in pairs if identity in p['ids']]
        position = bool(related and all(p['repeatable'] for p in related))
        reasons = []
        if len(values) < required:
            reasons.append('TEMPORAL_SECTION_SUPPORT_INSUFFICIENT')
        elif not repeatable:
            reasons.append('TEMPORAL_SECTION_DIAMETER_UNSTABLE')
        if not position:
            reasons.append('TEMPORAL_SECTION_LAYOUT_UNSTABLE')
        row['temporal_consistency'] = {'sampled_frames': count, 'solved_frames': len(values),
            'required_frames': required, 'diameter': spread, 'diameter_span_limit_mm': tolerance,
            'diameter_repeatable': repeatable, 'layout_repeatable': position,
            'frames': frame_results, 'reason_codes': reasons, 'accuracy_validated': False}
        row['reason_codes'] = sorted(set(row.get('reason_codes', [])+reasons))
    latest['temporal_sections'] = {'status': 'CHECKED', 'policy': dict(POLICY), 'policy_sha256': POLICY_SHA256,
        'requested_frames': requested_frames, 'sampled_frames': count, 'required_frames': required,
        'sources': sources, 'pairwise': pairs, 'geometry_source': 'CURRENT_REFERENCE_FRAME',
        'scope': 'REPEATABILITY_NOT_ABSOLUTE_ACCURACY', 'historical_frames_used': False}
    return latest
