"""Image boundaries -> conditional sections -> relative installation assessment.

This stage is independent of dense-depth coverage and runs before the legacy
surface recognizer. It reports its own evidence, never synthesizes depth or
silently promotes a colour envelope to an accepted cylinder observation.
"""
from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np

from .axial_aggregation import aggregate_boundaries, POLICY, REVISION, POLICY_SHA256
from .image_regions import propose_regions
from .projection_diagnostics import fit_silhouettes, pixel_error_scenarios
from .section_diagnostics import axis_from_image_lines
from .section_model import build_section_model
from .axial_consistency import synchronize_axial_blocks
from .ground_correction import correct_pipe_axis


EDGE_EXTRACTION_REVISION = 'colour-supported-local-gradient-v1'
EDGE_EXTRACTION_POLICY = {'search_radius_px': 3., 'gaussian_sigma_px': .7, 'axial_sigma_px': 4.,
    'axial_kernel_radius_px': 12, 'minimum_colour_gradient': 4.,
    'hue_half_width': 15., 'saturation_min': 120., 'value_min': 40.,
    'hue_margin_scale': 4., 'value_margin_scale': 3.,
    'localization': 'COLOUR_MEMBERSHIP_ZERO_CROSSING',
    'neutral_refinement': False, 'fill_missing_columns': False,
    'filtered_pixel_samples_independent': False,
    'geometric_line_used_for_localization': False}
EDGE_EXTRACTION_POLICY_SHA256 = hashlib.sha256(json.dumps(EDGE_EXTRACTION_POLICY, sort_keys=True).encode()).hexdigest()


def _colour_gradient(image, proposal, sampling_mask=None):
    """Localize the outer colour transition, without a fitted-line prior.

    The continuous margin has exactly the same HSV decision boundary as the
    original mask. Local axial averaging suppresses isolated colour noise;
    a directed zero crossing then supplies a subpixel edge. No radius, CAD
    size, fitted straight line, missing column or occluded gap is supplied.
    """
    if proposal['appearance'] == 'neutral':
        return None
    hue = int(proposal['appearance'].split('-')[1])
    x, y, w, h = proposal['region_px']
    radius = EDGE_EXTRACTION_POLICY['axial_kernel_radius_px']
    pad = radius+2
    x0, y0 = max(0, x-pad), max(0, y-pad)
    x1, y1 = min(image.shape[1], x+w+pad), min(image.shape[0], y+h+pad)
    hsv = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).astype(np.float32)
    delta = np.abs(hsv[:, :, 0]-hue)
    delta = np.minimum(delta, 180-delta)
    score = np.minimum(np.minimum(hsv[:, :, 1]-EDGE_EXTRACTION_POLICY['saturation_min'],
        EDGE_EXTRACTION_POLICY['hue_margin_scale']*(EDGE_EXTRACTION_POLICY['hue_half_width']-delta)),
        EDGE_EXTRACTION_POLICY['value_margin_scale']*(hsv[:, :, 2]-EDGE_EXTRACTION_POLICY['value_min']))
    ax, ay = np.asarray(proposal['axis_image'], float)/np.linalg.norm(proposal['axis_image'])
    rows, columns = np.mgrid[-radius:radius+1, -radius:radius+1]
    along, across = columns*ax+rows*ay, -columns*ay+rows*ax
    kernel = np.exp(-.5*((along/EDGE_EXTRACTION_POLICY['axial_sigma_px'])**2
                         +(across/EDGE_EXTRACTION_POLICY['gaussian_sigma_px'])**2))
    kernel /= kernel.sum()
    kernel = kernel.astype(np.float32)
    if sampling_mask is None:
        score = cv2.filter2D(score, -1, kernel)
    else:
        allowed = sampling_mask[y0:y1, x0:x1].astype(np.float32)
        weights = cv2.filter2D(allowed, -1, kernel)
        score = cv2.filter2D(score*allowed, -1, kernel)/np.maximum(weights, 1e-6)
    # Values retain original integer pixel centres; interpolation between
    # adjacent samples therefore preserves the threshold's half-pixel rule.
    return score, x0, y0


def _gradient_edge(gradient, x, seed_y, edge):
    if gradient is None:
        return seed_y, False
    score, x0, y0 = gradient
    column = x-x0
    seed = seed_y-y0-.5
    radius = EDGE_EXTRACTION_POLICY['search_radius_px']
    low = max(1, int(np.ceil(seed-radius)))
    high = min(len(score)-2, int(np.floor(seed+radius)))
    if high < low:
        return seed_y, False
    sign = 1. if edge == 'top' else -1.
    profile = score[:, column]*sign
    choices = []
    for index in range(low, high+1):
        delta = float(profile[index+1]-profile[index])
        if profile[index] <= 0 < profile[index+1] and delta >= EDGE_EXTRACTION_POLICY['minimum_colour_gradient']:
            located = y0+index-float(profile[index])/delta
            if abs(located-seed_y) <= radius:
                choices.append(located)
    return (min(choices, key=lambda value: abs(value-seed_y)), True) if choices else (seed_y, False)


def _observed_edges(image, proposal, role, sampling_mask=None):
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
    interior = None
    if sampling_mask is not None:
        selected = sampling_mask[y:y+height, x:x+width]
        raw &= selected
        # The cut planes are axial ends, never observed cylinder silhouettes.
        # Erode the selection to discard false top/bottom points on a cut.
        interior = cv2.erode(sampling_mask.astype(np.uint8), np.ones((5, 5), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0)[y:y+height, x:x+width]
    points = {'top': [], 'bottom': []}
    gradient = _colour_gradient(image, proposal, sampling_mask)
    extraction = {'revision': EDGE_EXTRACTION_REVISION, 'policy': dict(EDGE_EXTRACTION_POLICY),
                  'policy_sha256': EDGE_EXTRACTION_POLICY_SHA256,
                  'source': 'LOCAL_COLOUR_GRADIENT' if gradient is not None else 'ORIGINAL_NEUTRAL_THRESHOLD',
                  'edges': {edge: {'refined_columns': 0, 'threshold_columns': 0, 'shifts': []}
                            for edge in points}}
    profiles = []
    trim = max(8, int(proposal['width_px']*.15))
    for column in range(trim, max(trim, width-trim)):
        rows = np.flatnonzero(raw[:, column])
        if len(rows) < max(3, .35*proposal['width_px']):
            continue
        low, high = rows[0], rows[-1]
        if interior is not None and not (interior[low, column] and interior[high, column]):
            continue
        search_rows = np.flatnonzero(proposal['mask'][:, column])
        for edge, seed in [('top', y+low-.5), ('bottom', y+high+.5)]:
            search_seed = (y+search_rows[0]-.5 if edge == 'top' else y+search_rows[-1]+.5) if len(search_rows) else seed
            located, refined = _gradient_edge(gradient, x+column, search_seed, edge)
            if not refined or abs(located-seed) > EDGE_EXTRACTION_POLICY['search_radius_px']:
                located = seed
                refined = False
            # An adjusted sample must remain inside the actual selection.
            # Neither a cut-plane nor pixels beyond it can become a silhouette.
            if sampling_mask is not None:
                row = int(round(located))
                if not (0 <= row < sampling_mask.shape[0] and sampling_mask[row, x+column]):
                    located, refined = seed, False
            points[edge].append([x+column, located])
            extraction['edges'][edge]['refined_columns' if refined else 'threshold_columns'] += 1
            if refined:
                extraction['edges'][edge]['shifts'].append(located-seed)
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
    for audit in extraction['edges'].values():
        shifts = audit.pop('shifts')
        audit['median_shift_px'] = float(np.median(shifts)) if shifts else None
        audit['absolute_shift_p95_px'] = float(np.percentile(np.abs(shifts), 95)) if shifts else None
    return {'role': role, 'region_px': list(proposal['region_px']), 'appearance': appearance,
            'color_srgb': proposal['sampled_color_srgb'], 'width_px': proposal['width_px'],
            'edges': edges, 'reason_codes': reasons, 'highlight_band_count': highlights,
            'edge_extraction': extraction}


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
            if previous.get('edge_extraction') is not None and item.get('edge_extraction') is not None:
                sources = previous['edge_extraction'].get('source_extractions')
                if sources is None:
                    sources = [{'region_px': list(previous['region_px']), 'extraction': previous['edge_extraction']}]
                sources = sources+[{'region_px': list(item['region_px']), 'extraction': item['edge_extraction']}]
                previous['edge_extraction'] = {'revision': EDGE_EXTRACTION_REVISION,
                    'policy_sha256': EDGE_EXTRACTION_POLICY_SHA256, 'source': 'DISJOINT_VISIBLE_PIECES',
                    'source_extractions': sources}
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


def aggregate_image(image, role, *, sampling_mask=None, axis_hint_image=None):
    proposals, audit = propose_regions(image, sampling_mask=sampling_mask, axis_hint_image=axis_hint_image)
    items = _join_disjoint([_observed_edges(image, p, role, sampling_mask) for p in proposals])
    for index, item in enumerate(items):
        item['candidate_id'] = f'POOL-{role[0].upper()}{index+1:02d}'
        item['aggregation'], item['pooled_edges'] = aggregate_boundaries(item['edges'])
        item['full_span_audit'] = item['aggregation']
        attempts = [{'span': 'FULL', 'stable_boundary_count': item['aggregation']['stable_boundary_count']}]
        # Extend only while independent substrip checks remain consistent.
        # A failed full span stays in the audit; a locally stable part does not
        # certify the rejected ends, occluded gap or whole physical pipe.
        best_count = item['aggregation']['stable_boundary_count']
        best_length = max((float(np.ptp(e['points'][:, 0]))+1 for e in item['pooled_edges']), default=0.)
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
                    score = aggregation['stable_boundary_count']
                    if score > best_count or (score == best_count and score > 0 and length > best_length):
                        item['aggregation'], item['pooled_edges'] = aggregation, pooled
                        best_count = score
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
    boundaries = [item['aggregation']['boundaries'] for item in (first, second)]
    # A single surviving tangent cannot distinguish two cylinders sharing an
    # apparent edge. Missing/unstable edges must neither crash this check nor
    # cause a second physical pipe to be merged out of the section model.
    if any({b['edge'] for b in rows if b['status'] == 'STABLE'
            and b.get('line_slope_intercept') is not None} != {'top', 'bottom'}
           for rows in boundaries):
        return False
    station = (min(a[0]+a[2], b[0]+b[2])+max(a[0], b[0]))/2
    first_edges, second_edges = [{b['edge']: b for b in rows} for rows in boundaries]
    for edge in ('top', 'bottom'):
        x, y = first_edges[edge], second_edges[edge]
        delta = np.asarray(x['line_slope_intercept'])-y['line_slope_intercept']
        if abs(delta@[station, 1]) > max(8., .4*min(first['width_px'], second['width_px'])):
            return False
    return True


def _compatibility_reasons(first, second, offset=(0., 0.)):
    reasons = []
    stable_counts = [item['aggregation']['stable_boundary_count'] for item in (first, second)]
    # Three independent tangents can determine a conditional free circle.
    # An unstable fourth contour is omitted, never replaced by a prediction.
    if min(stable_counts) < 1 or sum(stable_counts) < 3:
        reasons.append('INSUFFICIENT_INDEPENDENT_SILHOUETTES')
    if not _compatible_appearance(first, second):
        reasons.append('APPEARANCE_MISMATCH')
    if abs(first['width_px']-second['width_px']) > max(6, .25*max(first['width_px'], second['width_px'])):
        reasons.append('PROJECTED_WIDTH_MISMATCH')
    x, y, w, h = first['region_px']; u, v, s, t = second['region_px']
    u += offset[0]; v += offset[1]
    if min(x+w, u+s)-max(x, u) < .4*min(w, s):
        reasons.append('INSUFFICIENT_AXIAL_IMAGE_OVERLAP')
    # Rectified parallel axes: this is a bounded proposal association only,
    # not point correspondence or depth. Multiple choices stay ambiguous.
    station = (max(x, u)+min(x+w, u+s))/2
    for a, b in zip(first['aggregation']['boundaries'], second['aggregation']['boundaries']):
        if a['status'] != 'STABLE' or b['status'] != 'STABLE':
            continue
        right_line = np.asarray(b['line_slope_intercept']).copy()
        right_line[1] += offset[1]-right_line[0]*offset[0]
        delta = np.asarray(a['line_slope_intercept'])-right_line
        if abs(delta@[station, 1]) > max(24, first['width_px']*.4):
            reasons.append('TRANSVERSE_POSITION_MISMATCH')
            break
    return reasons


def _compatible(first, second, offset=(0., 0.)):
    return not _compatibility_reasons(first, second, offset)


def _pairing_audit(views, reasons, comparisons):
    """Explain rejected/ambiguous pairings without inventing missing tangents."""
    audit = {'revision': 'stereo-boundary-pairing-v1',
             'minimum_independent_boundaries': 3, 'nominal_dimensions_used': False}
    distant = {'APPEARANCE_MISMATCH', 'INSUFFICIENT_AXIAL_IMAGE_OVERLAP', 'TRANSVERSE_POSITION_MISMATCH'}
    for role, other in [('left', 'right'), ('right', 'left')]:
        rows = []
        for index, item in enumerate(views[role]):
            compatible, nearby, unique = [], [], False
            for opposite, counterpart in enumerate(views[other]):
                i, j = (index, opposite) if role == 'left' else (opposite, index)
                codes = reasons[i][j]
                if comparisons[i, j]:
                    compatible.append(counterpart['candidate_id'])
                    unique = comparisons[i].sum() == 1 and comparisons[:, j].sum() == 1
                if not distant.intersection(codes):
                    nearby.append({'candidate_id': counterpart['candidate_id'],
                        'stable_boundary_count': counterpart['aggregation']['stable_boundary_count'],
                        'combined_stable_boundary_count': item['aggregation']['stable_boundary_count']
                            + counterpart['aggregation']['stable_boundary_count'],
                        'reason_codes': codes})
            stable = item['aggregation']['stable_boundary_count']
            status = ('PAIRED' if unique and len(compatible) == 1 else
                      'AMBIGUOUS_STEREO_CANDIDATES' if compatible else
                      'NO_STABLE_BOUNDARY' if stable == 0 else
                      'INSUFFICIENT_INDEPENDENT_SILHOUETTES' if nearby and all(
                          'INSUFFICIENT_INDEPENDENT_SILHOUETTES' in c['reason_codes'] for c in nearby) else
                      'NO_COMPATIBLE_STEREO_CANDIDATE')
            rows.append({'candidate_id': item['candidate_id'], 'stable_boundary_count': stable,
                         'status': status, 'compatible_candidate_ids': compatible,
                         'nearby_candidates': nearby})
        audit[role] = rows
    return audit


def _compatible_appearance(first, second):
    """Overlapping hue bins are proposal labels, not different physical colours."""
    a, b = first['appearance'], second['appearance']
    if a == b:
        return True
    if not (a.startswith('hue-') and b.startswith('hue-')):
        return False
    delta = abs(int(a.split('-')[1])-int(b.split('-')[1]))
    if min(delta, 180-delta) > 15:
        return False
    colours = []
    for item in (first, second):
        value = item.get('color_srgb', '')
        if not isinstance(value, str) or len(value) != 7 or not value.startswith('#'):
            return False
        try:
            colours.append([int(value[i:i+2], 16) for i in (5, 3, 1)])
        except ValueError:
            return False
    hsv = cv2.cvtColor(np.array([colours], np.uint8), cv2.COLOR_BGR2HSV)[0].astype(int)
    difference = abs(hsv[0, 0]-hsv[1, 0])
    return bool(min(difference, 180-difference) <= 15 and hsv[:, 1].min() >= 80)


def analyze_pooled_sections(left, right, calibration, *, aggregation_band=None, image_offsets=None,
                            disparity_range=(0., 256.), ground_result=None, refine_epipolar=False):
    views, proposal_audit = {}, {}
    masks, band_audit = {}, None
    k = np.asarray(calibration.left.rectified_intrinsic, float)
    kr = np.asarray(calibration.right.rectified_intrinsic, float)
    refinement = {'applied': False, 'status': 'NOT_REQUESTED'}
    if refine_epipolar:
        # Dense depth/ground keep their own original evidence. This operation
        # belongs only to the independent silhouette reconstruction stage.
        if image_offsets or left.shape != right.shape or not np.allclose(k, kr):
            refinement = {'applied': False, 'status': 'FULL_COMMON_INTRINSICS_REQUIRED'}
        else:
            from .epipolar_refinement import refine_right_image
            right, refinement = refine_right_image(left, right, k)
    if aggregation_band is not None:
        from .axial_selection import normalize_axial_band, band_sampling_masks
        aggregation_band = normalize_axial_band(aggregation_band,
            image_size=None if image_offsets else (left.shape[1], left.shape[0]))
        offsets = image_offsets or {}
        full_vertical_offset = k[1, 2]-kr[1, 2]+offsets.get('left_y', 0)-offsets.get('right_y', 0)
        masks, band_audit = band_sampling_masks(aggregation_band, left.shape, right.shape,
            offsets=offsets, disparity_range=disparity_range, right_vertical_offset=full_vertical_offset)
    for role, image in [('left', left), ('right', right)]:
        views[role], proposal_audit[role] = (aggregate_image(image, role, sampling_mask=masks[role],
            axis_hint_image=band_audit['axis_image']) if band_audit else aggregate_image(image, role))
    result = {'revision': REVISION, 'policy_sha256': POLICY_SHA256, 'policy': dict(POLICY),
              'section_revision': 'pooled-sections-ground-constrained-v3',
              'edge_extraction_revision': EDGE_EXTRACTION_REVISION,
              'edge_extraction_policy_sha256': EDGE_EXTRACTION_POLICY_SHA256,
              'edge_extraction_policy': dict(EDGE_EXTRACTION_POLICY),
              'stages': ['BOUNDARY_AGGREGATION', 'SECTION_DIMENSIONS', 'MODEL_RELATIVE_POSITION'],
              'occlusion_recovered': False, 'nominal_dimensions_used_in_fitting': False,
              'views': {role: [_compact(i) for i in items] for role, items in views.items()},
              'proposal_audit': proposal_audit, 'candidates': [], 'model': None,
              'epipolar_refinement': refinement,
              'ground_correction': correct_pipe_axis([], k, {}, ground_result)[1],
              'aggregation_band': band_audit,
              'truncated': any(a['truncated'] for a in proposal_audit.values())}
    # The projection fitter uses a shared rectified K. Translate right image
    # coordinates to that K after any asymmetric ROI crop; never recenter eyes.
    if not np.allclose(k[:2, :2], kr[:2, :2], atol=1e-7):
        return result | {'status': 'UNSUPPORTED_RECTIFIED_INTRINSICS'}
    offset = k[:2, 2]-kr[:2, 2]
    reasons = [[_compatibility_reasons(a, b, offset) for b in views['right']] for a in views['left']]
    comparisons = np.array([[not codes for codes in row] for row in reasons], bool).reshape(
        len(views['left']), len(views['right']))
    result['pairing_audit'] = _pairing_audit(views, reasons, comparisons)
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
    if 'axis' in axis_audit:
        singular = axis_audit['singular_values']
        ratio = float(singular[-1]/singular[-2])
        axis_audit['nullspace_ratio'] = ratio
        axis_audit['status'] = 'STABLE' if ratio <= POLICY['maximum_axis_nullspace_ratio'] else 'COMMON_AXIS_UNSTABLE'
    result['unconstrained_axis'] = axis_audit
    trusted_observations = [obs for i, j, observations in paired_observations
        if not (views['left'][i]['reason_codes'] or views['right'][j]['reason_codes']) for obs in observations]
    axis_audit, correction = correct_pipe_axis(trusted_observations, k, axis_audit, ground_result)
    result['ground_correction'] = correction
    axis = np.asarray(axis_audit['axis']) if axis_audit.get('status') == 'STABLE' else None
    result['axis'] = axis_audit
    model_items = []
    for i, j, observations in paired_observations:
        a, b = views['left'][i], views['right'][j]
        candidate = {'candidate_id': f'{a["candidate_id"]}-{b["candidate_id"]}',
                     'left_candidate_id': a['candidate_id'], 'right_candidate_id': b['candidate_id'],
                     'left_region_px': a['region_px'], 'right_region_px': b['region_px'],
                     'boundary_status': 'STABLE', 'reason_codes': sorted(set(a['reason_codes']+b['reason_codes']))}
        candidate['observed_boundary_count'] = len(observations)
        reference = float(np.mean(a['pooled_edges'][0]['points'][:, 0]))
        fit = {'status': 'OUTLINE_IDENTITY_OR_OCCLUSION_UNRESOLVED'}
        error, sensitivity = {}, {}
        common_support = None
        if axis is None:
            candidate['reason_codes'].append(axis_audit['status'])
            fit = {'status': axis_audit['status']}
        if not candidate['reason_codes'] and axis is not None:
            fit = fit_silhouettes(_bounded(observations), k, calibration.baseline_mm, axis, reference)
            if fit['status'] == 'CONDITIONAL_FIT':
                common, partitions, common_support = synchronize_axial_blocks(observations, k, calibration.baseline_mm, fit)
                if common:
                    fit = fit_silhouettes(_bounded(common), k, calibration.baseline_mm, axis, reference)
            if fit['status'] == 'CONDITIONAL_FIT':
                fitted_observations = common if common else observations
                try:
                    error = pixel_error_scenarios(_bounded(fitted_observations), k, calibration.baseline_mm, axis, reference,
                                                  sigma_px=POLICY['assumed_coherent_edge_sigma_px'])
                except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
                    # A nearly parallel baseline may admit a finite trial fit
                    # but lose its solution under a subpixel perturbation.
                    # Keep that candidate diagnostic and withhold acceptance;
                    # it must not abort the assessment of all other pipes.
                    error = {'status': 'PIXEL_UNCERTAINTY_UNRESOLVED',
                             'total_accuracy_known': False, 'scenarios': {}, 'reason': str(exc)}
                    candidate['reason_codes'].append('PIXEL_UNCERTAINTY_UNRESOLVED')
                if correction['applied']:
                    error['ground_normal_uncertainty_included'] = False
                if fit['reprojection_rms_px'] > POLICY['maximum_reprojection_rms_px']:
                    candidate['reason_codes'].append('CYLINDER_PROJECTION_INCONSISTENT')
                halves = []
                for divided in partitions:
                    halves.append(fit_silhouettes(_bounded(divided), k, calibration.baseline_mm, axis, reference))
                if len(halves) == 2 and all(f['status'] == 'CONDITIONAL_FIT' for f in halves):
                    diameter_delta = abs(halves[0]['trial_diameter_mm']-halves[1]['trial_diameter_mm'])
                    center_delta = float(np.linalg.norm(np.array(halves[0]['section_center_parameters_mm'])-halves[1]['section_center_parameters_mm']))
                    stable = diameter_delta <= max(1., .05*fit['trial_diameter_mm']) and center_delta <= 5.
                    sensitivity = {'diameter_difference_mm': diameter_delta, 'center_difference_mm': center_delta,
                                   'center_difference_components_mm': (np.array(halves[0]['section_center_parameters_mm'])-halves[1]['section_center_parameters_mm']).tolist(),
                                   'stable': bool(stable), 'method': 'COMMON_STATION_ALTERNATING_BLOCKS'}
                else:
                    sensitivity = {'stable': False, 'reason': 'SUBSET_FIT_UNRESOLVED' if partitions else common_support['status']}
                if not sensitivity['stable']:
                    candidate['reason_codes'].append('POOLED_DIMENSIONS_UNSTABLE')
        candidate.update(fit=fit, pixel_error=error, window_sensitivity=sensitivity, common_axial_support=common_support,
                         dimension_status='CONDITIONAL' if fit['status'] == 'CONDITIONAL_FIT' else 'UNRESOLVED')
        if correction['status'] == 'GROUND_AXIS_CONFLICT' or correction.get('boundary_consistency_bypassed'):
            candidate['reason_codes'].append('GROUND_AXIS_CONFLICT')
        if correction.get('trial_mode') and correction.get('applied'):
            candidate['reason_codes'].append('GROUND_CORRECTION_EXPERIMENTAL')
        if len(observations) < 4:
            candidate['boundary_status'] = 'PARTIAL'
            candidate['reason_codes'].append('PARTIAL_STEREO_BOUNDARY_SUPPORT')
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
        included = [m for index, m in enumerate(model_items) if index not in omitted]
        origin_id = next((m['id'] for m in included if m['fit']['status'] == 'CONDITIONAL_FIT'), None)
        result['model'] = build_section_model(included, axis, origin_id=origin_id,
            metadata={'nominal_dimensions_used_in_fitting': False, 'occlusion_recovered': False,
                      'ground_correction': correction,
                      'epipolar_refinement': refinement,
                      'candidate_quality': {c['candidate_id']: c['reason_codes'] for c in result['candidates']}})
    return result | {'status': 'ANALYZED', 'boundary_pair_count': len(pair_indexes),
                     'conditional_section_count': sum(c['dimension_status'] == 'CONDITIONAL' for c in result['candidates'])}
