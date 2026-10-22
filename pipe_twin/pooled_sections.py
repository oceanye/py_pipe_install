"""Image boundaries -> conditional sections -> relative installation assessment.

This stage is independent of dense-depth coverage and runs before the legacy
surface recognizer. It reports its own evidence, never synthesizes depth or
silently promotes a colour envelope to an accepted cylinder observation.
"""
from __future__ import annotations

import cv2
import numpy as np

from .axial_aggregation import aggregate_boundaries, POLICY, REVISION, POLICY_SHA256
from .image_regions import propose_regions
from .projection_diagnostics import fit_silhouettes, pixel_error_scenarios
from .section_diagnostics import axis_from_image_lines
from .section_model import build_section_model


def _observed_edges(image, proposal, role):
    x, y, width, height = proposal['region_px']
    crop = image[y:y+height, x:x+width]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    appearance = proposal['appearance']
    if appearance == 'neutral':
        raw = (hsv[:, :, 1] < 65) & (hsv[:, :, 2] >= 100)
    else:
        hue = int(appearance.split('-')[1]); delta = abs(hsv[:, :, 0].astype(int)-hue)
        raw = (np.minimum(delta, 180-delta) <= 15) & (hsv[:, :, 1] >= 120) & (hsv[:, :, 2] >= 40)
    # Morphology only bounds the search; edges come from the original pixels.
    gate = cv2.dilate(proposal['mask'], np.ones((5, 5), np.uint8)).astype(bool)
    raw &= gate
    points = {'top': [], 'bottom': []}
    profiles = []
    trim = max(8, int(proposal['width_px']*.15))
    for column in range(trim, max(trim, width-trim)):
        rows = np.flatnonzero(raw[:, column])
        if len(rows) < max(3, .35*proposal['width_px']):
            continue
        low, high = rows[0], rows[-1]
        points['top'].append([x+column, y+low-.5])
        points['bottom'].append([x+column, y+high+.5])
        if high-low >= 8 and appearance != 'neutral':
            indexes = np.linspace(low, high, 100).round().astype(int)
            profiles.append((hsv[indexes, column, 1] < 100) & (hsv[indexes, column, 2] >= 180))
    highlights = 0
    if profiles:
        # A long envelope can span separate cylinders. Look within disjoint
        # blocks too: averaging across ends can erase their separate highlights.
        for block in np.array_split(profiles, max(1, len(profiles)//128)):
            active = np.mean(block, axis=0) >= .5
            active[:5] = False; active[-5:] = False
            transitions = np.diff(np.r_[False, active, False].astype(int))
            highlights = max(highlights, int(sum(end-start >= 2 for start, end in zip(
                np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)))))
    edges = [{'role': role, 'edge': edge, 'points': np.asarray(pixels, float).reshape(-1, 2)} for edge, pixels in points.items()]
    reasons = []
    if abs(proposal['axis_image'][0]) < .3:
        reasons.append('AXIAL_IMAGE_PARAMETERIZATION_UNSUPPORTED')
    if proposal.get('transverse_boundary_contacts'):
        reasons.append('TRANSVERSE_BOUNDARY_CLIPPED')
    if proposal['width_px'] < POLICY['minimum_width_px']:
        reasons.append('VISIBLE_BAND_TOO_NARROW')
    if highlights >= 2:
        reasons.append('MULTIPLE_BANDS_OR_OCCLUDED_INSTANCES')
    return {'role': role, 'region_px': list(proposal['region_px']), 'appearance': appearance,
            'color_srgb': proposal['sampled_color_srgb'], 'width_px': proposal['width_px'],
            'edges': edges, 'reason_codes': reasons, 'highlight_band_count': highlights}


def _join_disjoint(items):
    """Combine compatible visible pieces; the intervening gap remains empty."""
    combined = []
    for item in items:
        item['source_regions_px'] = [item['region_px']]
        merged = False
        for previous in combined:
            if previous['appearance'] != item['appearance']:
                continue
            joins = True
            for first, second in zip(previous['edges'], item['edges']):
                a, b = first['points'], second['points']
                if min(len(a), len(b)) < 48:
                    joins = False; break
                a0, a1 = a[:, 0].min(), a[:, 0].max()
                b0, b1 = b[:, 0].min(), b[:, 0].max()
                if min(a1, b1) >= max(a0, b0) or max(a0, b0)-min(a1, b1) > 512:
                    joins = False; break
                line_a = np.polyfit(a[:, 0], a[:, 1], 1)
                line_b = np.polyfit(b[:, 0], b[:, 1], 1)
                station = (max(a0, b0)+min(a1, b1))/2
                if abs((line_a-line_b)@[station, 1]) > 2 or abs(line_a[0]-line_b[0]) > .02:
                    joins = False; break
            if not joins:
                continue
            for first, second in zip(previous['edges'], item['edges']):
                first['points'] = np.vstack([first['points'], second['points']])
            previous['source_regions_px'] += item['source_regions_px']
            previous['reason_codes'] = sorted(set(previous['reason_codes']+item['reason_codes']))
            boxes = np.asarray(previous['source_regions_px'])
            low = boxes[:, :2].min(axis=0); high = (boxes[:, :2]+boxes[:, 2:]).max(axis=0)
            previous['region_px'] = [*low.tolist(), *(high-low).tolist()]
            merged = True
            break
        if not merged:
            combined.append(item)
    return combined


def aggregate_image(image, role):
    proposals, audit = propose_regions(image)
    items = _join_disjoint([_observed_edges(image, p, role) for p in proposals])
    for index, item in enumerate(items):
        item['candidate_id'] = f'POOL-{role[0].upper()}{index+1:02d}'
        item['aggregation'], item['pooled_edges'] = aggregate_boundaries(item['edges'])
        item['full_span_audit'] = item['aggregation']
        attempts = [{'span': 'FULL', 'stable_boundary_count': item['aggregation']['stable_boundary_count']}]
        # Extend only while independent substrip checks remain consistent.
        # A failed full span stays in the audit; a locally stable part does not
        # certify the rejected ends, occluded gap or whole physical pipe.
        best_length = 0.
        if item['aggregation']['stable_boundary_count'] < 2 and all(len(e['points']) for e in item['edges']):
            low = max(e['points'][:, 0].min() for e in item['edges'])
            high = min(e['points'][:, 0].max() for e in item['edges'])+1
            for length in POLICY['adaptive_lengths_px']:
                if length >= high-low:
                    continue
                for start in np.linspace(low, high-length, POLICY['adaptive_positions']):
                    selected = [{**e, 'points': e['points'][(e['points'][:, 0] >= start) & (e['points'][:, 0] < start+length)]} for e in item['edges']]
                    aggregation, pooled = aggregate_boundaries(selected)
                    attempts.append({'span': [float(start), float(start+length)],
                                     'stable_boundary_count': aggregation['stable_boundary_count']})
                    if aggregation['stable_boundary_count'] == 2 and length > best_length:
                        item['aggregation'], item['pooled_edges'] = aggregation, pooled
                        best_length = length
        item['span_attempts'] = attempts
        item['boundary_stable'] = item['aggregation']['stable_boundary_count'] == 2
        item['boundary_status'] = 'STABLE' if item['boundary_stable'] else 'UNSTABLE'
    return items, audit


def _compact(item):
    return {k: v for k, v in item.items() if k not in ('edges', 'pooled_edges')}


def _bounded(observations):
    """Keep actual samples across the whole interval; never interpolate gaps."""
    output = []
    for obs in observations:
        points = obs['points'][np.argsort(obs['points'][:, 0])]
        count = min(len(points), POLICY['maximum_fit_points_per_boundary'])
        output.append({**obs, 'points': points[np.linspace(0, len(points)-1, count).round().astype(int)]})
    return output


def _possible_same_instance(first, second):
    if first['appearance'] != second['appearance']:
        return False
    a, b = first['region_px'], second['region_px']
    if min(a[0]+a[2], b[0]+b[2]) >= max(a[0], b[0]):
        return False
    station = (min(a[0]+a[2], b[0]+b[2])+max(a[0], b[0]))/2
    for x, y in zip(first['aggregation']['boundaries'], second['aggregation']['boundaries']):
        delta = np.asarray(x['line_slope_intercept'])-y['line_slope_intercept']
        if abs(delta@[station, 1]) > max(8., .4*min(first['width_px'], second['width_px'])):
            return False
    return True


def _compatible(first, second, offset=(0., 0.)):
    if not (first['boundary_stable'] and second['boundary_stable']):
        return False
    if first['appearance'] != second['appearance']:
        return False
    if abs(first['width_px']-second['width_px']) > max(6, .25*max(first['width_px'], second['width_px'])):
        return False
    x, y, w, h = first['region_px']; u, v, s, t = second['region_px']
    u += offset[0]; v += offset[1]
    if min(x+w, u+s)-max(x, u) < .4*min(w, s):
        return False
    # Rectified parallel axes: this is a bounded proposal association only,
    # not point correspondence or depth. Multiple choices stay ambiguous.
    station = (max(x, u)+min(x+w, u+s))/2
    for a, b in zip(first['aggregation']['boundaries'], second['aggregation']['boundaries']):
        right_line = np.asarray(b['line_slope_intercept']).copy()
        right_line[1] += offset[1]-right_line[0]*offset[0]
        delta = np.asarray(a['line_slope_intercept'])-right_line
        if abs(delta@[station, 1]) > max(24, first['width_px']*.4):
            return False
    return True


def analyze_pooled_sections(left, right, calibration):
    views, proposal_audit = {}, {}
    for role, image in [('left', left), ('right', right)]:
        views[role], proposal_audit[role] = aggregate_image(image, role)
    result = {'revision': REVISION, 'policy_sha256': POLICY_SHA256, 'policy': dict(POLICY),
              'stages': ['BOUNDARY_AGGREGATION', 'SECTION_DIMENSIONS', 'MODEL_RELATIVE_POSITION'],
              'occlusion_recovered': False, 'nominal_dimensions_used_in_fitting': False,
              'views': {role: [_compact(i) for i in items] for role, items in views.items()},
              'proposal_audit': proposal_audit, 'candidates': [], 'model': None,
              'truncated': any(a['truncated'] for a in proposal_audit.values())}
    k = np.asarray(calibration.left.rectified_intrinsic, float)
    kr = np.asarray(calibration.right.rectified_intrinsic, float)
    # The projection fitter uses a shared rectified K. Translate right image
    # coordinates to that K after any asymmetric ROI crop; never recenter eyes.
    if not np.allclose(k[:2, :2], kr[:2, :2], atol=1e-7):
        return result | {'status': 'UNSUPPORTED_RECTIFIED_INTRINSICS'}
    offset = k[:2, 2]-kr[:2, 2]
    comparisons = np.array([[_compatible(a, b, offset) for b in views['right']] for a in views['left']], bool)
    if comparisons.ndim != 2 or not comparisons.size:
        return result | {'status': 'NO_STABLE_STEREO_BOUNDARIES'}
    pair_indexes = [(i, j) for i, j in zip(*np.nonzero(comparisons))
                    if comparisons[i].sum() == 1 and comparisons[:, j].sum() == 1]
    all_lines = []
    paired_observations = []
    for i, j in pair_indexes:
        observations = [*views['left'][i]['pooled_edges'],
                        *[{**o, 'points': o['points']+offset} for o in views['right'][j]['pooled_edges']]]
        paired_observations.append((i, j, observations))
        # Compound/occluded envelopes cannot define a cylinder's common axis.
        for obs in observations if not (views['left'][i]['reason_codes'] or views['right'][j]['reason_codes']) else []:
            slope, intercept = np.polyfit(obs['points'][:, 0], obs['points'][:, 1], 1)
            all_lines.append([slope, -1., intercept])
    try:
        axis_audit = axis_from_image_lines(all_lines, k)
    except ValueError:
        axis_audit = {'status': 'COMMON_AXIS_UNRESOLVED'}
    axis = None
    if 'axis' in axis_audit:
        singular = axis_audit['singular_values']
        ratio = float(singular[-1]/singular[-2])
        axis_audit['nullspace_ratio'] = ratio
        axis_audit['status'] = 'STABLE' if ratio <= POLICY['maximum_axis_nullspace_ratio'] else 'COMMON_AXIS_UNSTABLE'
        if axis_audit['status'] == 'STABLE':
            axis = np.asarray(axis_audit['axis'])
    result['axis'] = axis_audit
    model_items = []
    for i, j, observations in paired_observations:
        a, b = views['left'][i], views['right'][j]
        candidate = {'candidate_id': f'{a["candidate_id"]}-{b["candidate_id"]}',
                     'left_candidate_id': a['candidate_id'], 'right_candidate_id': b['candidate_id'],
                     'left_region_px': a['region_px'], 'right_region_px': b['region_px'],
                     'boundary_status': 'STABLE', 'reason_codes': sorted(set(a['reason_codes']+b['reason_codes']))}
        reference = a['aggregation']['boundaries'][0]['reference_u_px']
        fit = {'status': 'OUTLINE_IDENTITY_OR_OCCLUSION_UNRESOLVED'}
        error, sensitivity = {}, {}
        if axis is None:
            candidate['reason_codes'].append(axis_audit['status'])
            fit = {'status': axis_audit['status']}
        if not candidate['reason_codes'] and axis is not None:
            fit = fit_silhouettes(_bounded(observations), k, calibration.baseline_mm, axis, reference)
            if fit['status'] == 'CONDITIONAL_FIT':
                error = pixel_error_scenarios(_bounded(observations), k, calibration.baseline_mm, axis, reference,
                                               sigma_px=POLICY['assumed_coherent_edge_sigma_px'])
                if fit['reprojection_rms_px'] > POLICY['maximum_reprojection_rms_px']:
                    candidate['reason_codes'].append('CYLINDER_PROJECTION_INCONSISTENT')
                halves = []
                for parity in (0, 1):
                    divided = []
                    for obs in observations:
                        p = obs['points']; first = float(p[:, 0].min())
                        keep = (np.floor((p[:, 0]-first)/POLICY['window_px']).astype(int) % 2) == parity
                        divided.append({**obs, 'points': p[keep]})
                    halves.append(fit_silhouettes(_bounded(divided), k, calibration.baseline_mm, axis, reference))
                if all(f['status'] == 'CONDITIONAL_FIT' for f in halves):
                    diameter_delta = abs(halves[0]['trial_diameter_mm']-halves[1]['trial_diameter_mm'])
                    center_delta = float(np.linalg.norm(np.array(halves[0]['section_center_parameters_mm'])-halves[1]['section_center_parameters_mm']))
                    stable = diameter_delta <= max(1., .05*fit['trial_diameter_mm']) and center_delta <= 5.
                    sensitivity = {'diameter_difference_mm': diameter_delta, 'center_difference_mm': center_delta,
                                   'stable': bool(stable), 'method': 'DISJOINT_ALTERNATING_AXIAL_BLOCKS'}
                else:
                    sensitivity = {'stable': False, 'reason': 'SUBSET_FIT_UNRESOLVED'}
                if not sensitivity['stable']:
                    candidate['reason_codes'].append('POOLED_DIMENSIONS_UNSTABLE')
        candidate.update(fit=fit, pixel_error=error, window_sensitivity=sensitivity,
                         dimension_status='CONDITIONAL' if fit['status'] == 'CONDITIONAL_FIT' else 'UNRESOLVED')
        result['candidates'].append(candidate)
        model_items.append({'id': candidate['candidate_id'], 'label': candidate['candidate_id'],
                            'color': a['color_srgb'], 'reference_diameter_mm': None, 'fit': fit, 'pixel_error': error})
    # Disjoint pieces may belong to the same pipe behind an obstruction. Keep
    # every fit in the audit, but never make their separation an inter-pipe gap.
    omitted = set()
    for n, (i, _, _) in enumerate(paired_observations):
        for m in range(n+1, len(paired_observations)):
            j = paired_observations[m][0]
            if _possible_same_instance(views['left'][i], views['left'][j]):
                sizes = [sum(e.get('retained_columns', 0) for e in views['left'][v]['aggregation']['boundaries']) for v in (i, j)]
                omitted.add(m if sizes[0] >= sizes[1] else n)
                for index in (n, m):
                    result['candidates'][index]['reason_codes'].append('POSSIBLE_SHARED_PIPE_INSTANCE')
    for index, candidate in enumerate(result['candidates']):
        candidate['included_in_section_model'] = index not in omitted
    matched = {x for i, j in pair_indexes for x in (views['left'][i]['candidate_id'], views['right'][j]['candidate_id'])}
    result['unpaired_candidate_ids'] = [c['candidate_id'] for items in views.values() for c in items if c['candidate_id'] not in matched]
    if axis is not None:
        result['model'] = build_section_model([m for index, m in enumerate(model_items) if index not in omitted], axis,
            metadata={'nominal_dimensions_used_in_fitting': False, 'occlusion_recovered': False})
    return result | {'status': 'ANALYZED', 'boundary_pair_count': len(pair_indexes),
                     'conditional_section_count': sum(c['dimension_status'] == 'CONDITIONAL' for c in result['candidates'])}
