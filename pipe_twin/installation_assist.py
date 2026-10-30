"""Coarse, model-assisted identity suggestions, independent of metric acceptance.

Colours/nominal sizes are priors for identity, never measured diameters. The
same current image can remain useful when stereo measurement gates fail.
Candidate ambiguity is computed over all near-best assignments, not just the
few hypotheses displayed in the GUI. Unseen model members are not absent.
"""
from __future__ import annotations

import colorsys
import itertools
import math
import numpy as np

from .image_line_geometry import canonical_image_axis, normal_from_axis
from .section_diagnostics import section_basis
from .installation_assist_config import SUPPORTED_COLOR_CLASSES

REVISION = 'relative-layout-installation-assist-v1'
POLICY = {'maximum_assignments': 50000, 'maximum_displayed_hypotheses': 5,
          'near_best_difference': .12, 'maximum_relative_difference': .8,
          'minimum_layout_observations': 3, 'view_sample_step_deg': 5,
          'hint_diameter_tolerance_mm': 3., 'hint_diameter_tolerance_ratio': .15}


def color_class(value):
    try:
        r, g, b = [int(value[i:i+2], 16)/255 for i in (1, 3, 5)]
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
    except (TypeError, ValueError, IndexError):
        return 'UNKNOWN'
    if s < .3:
        return 'WHITE' if v > .55 else 'UNKNOWN'
    if h < .075 or h > .92:
        return 'RED'
    if .50 < h < .77:
        return 'BLUE'
    if .18 < h < .49:
        return 'GREEN'
    if .075 <= h <= .18:
        return 'YELLOW'
    return 'UNKNOWN'


def _unit(vector):
    value = np.asarray(vector, float)
    if value.shape != (3,) or not np.isfinite(value).all() or np.linalg.norm(value) < 1e-9:
        raise ValueError('INVALID_ASSIST_DIRECTION')
    return value/np.linalg.norm(value)


def _model(specs, settings):
    values, ids = [], set()
    axis = settings.get('axis_world')
    for spec in specs:
        points = np.asarray(spec['centerline_world_mm'], float)
        if points.shape != (2, 3) or not np.isfinite(points).all():
            raise ValueError('INVALID_ASSIST_MODEL')
        direction = _unit(points[1]-points[0])
        axis = direction if axis is None else _unit(axis)
        if abs(axis@direction) < math.cos(math.radians(2)):
            raise ValueError('ASSIST_REQUIRES_PARALLEL_MODEL')
        diameter = float(spec['nominal_diameter_mm'])
        if spec['pipe_id'] in ids or not np.isfinite(diameter) or diameter <= 0:
            raise ValueError('INVALID_ASSIST_MODEL')
        ids.add(spec['pipe_id'])
        synthetic = str(spec.get('color_source', '')).startswith('stl_synthetic')
        values.append({'pipe_id': spec['pipe_id'], 'center': points.mean(axis=0),
            'diameter': diameter, 'color_class': 'UNKNOWN' if synthetic else color_class(spec.get('color_srgb')),
            'synthetic_color_ignored': synthetic})
    return values, axis


def _transverse(view, axis):
    normal = normal_from_axis(axis)
    boundaries = (view.get('aggregation') or {}).get('boundaries', [])
    pixels = [b.get('reference_pixel') for b in boundaries if b.get('status') == 'STABLE' and b.get('reference_pixel') is not None]
    if pixels:
        pixel = np.mean(pixels, axis=0)
    else:
        x, y, w, h = view['region_px']
        pixel = np.array([x+w/2, y+h/2])
    return float(pixel@normal)


def _observations(stage):
    views = stage.get('views') or {}
    left = {v['candidate_id']: v for v in views.get('left', [])}
    right = {v['candidate_id']: v for v in views.get('right', [])}
    pairs = {c.get('left_candidate_id'): c for c in stage.get('candidates', [])}
    model = (stage.get('model') or {}).get('candidates', [])
    labels = {c['id']: f'S{i+1}' for i, c in enumerate(model) if c.get('estimated_center_uv_mm') is not None}
    eligible = [v for v in left.values() if ((v.get('aggregation') or {}).get('stable_boundary_count', 0) > 0
                or v.get('boundary_stable')) and not set(v.get('reason_codes') or []) &
                {'TRANSVERSE_BOUNDARY_CLIPPED', 'VISIBLE_BAND_TOO_NARROW', 'MULTIPLE_BANDS_OR_OCCLUDED_INSTANCES'}]
    if not eligible:
        return []
    directions = [canonical_image_axis(v.get('image_axis', [1., 0.])) for v in eligible]
    _, vectors = np.linalg.eigh(sum(np.outer(a, a) for a in directions))
    axis = canonical_image_axis(vectors[:, -1])
    values = []
    offsets = stage.get('zone_offsets') or {}
    for view in eligible:
        pair = pairs.get(view['candidate_id'], {})
        if (pair.get('included_in_section_model') is False
                or 'POSSIBLE_SHARED_PIPE_INSTANCE' in pair.get('reason_codes', [])):
            continue
        identity = pair.get('candidate_id', view['candidate_id'])
        def box(role, source):
            if not source:
                return None
            x, y, w, h = source['region_px']
            refinement = stage.get('epipolar_refinement') or {}
            if role == 'right' and refinement.get('applied'):
                from .epipolar_refinement import transform_points
                corners = transform_points([[x, y], [x+w, y], [x, y+h], [x+w, y+h]],
                                           refinement['right_refined_to_original'])
                low, high = np.min(corners, axis=0), np.max(corners, axis=0)
                x, y = low.tolist()
                w, h = (high-low).tolist()
            return [x+offsets.get(role+'_x', 0), y+offsets.get(role+'_y', 0), w, h]
        values.append({'observation_id': identity, 'label': labels.get(identity, f'C{len(values)+1}'),
            'zone_id': stage.get('zone_id', '全幅'), 'left_region_px': box('left', view),
            'right_region_px': box('right', right.get(pair.get('right_candidate_id'))),
            'color_class': color_class(view.get('color_srgb')), 'image_width_px': float(view['width_px']),
            'image_transverse_px': _transverse(view, axis), 'image_axis': axis.tolist(),
            'source': 'CURRENT_POOLED_IMAGE_BOUNDARIES', 'metric_fit_required': False,
            'measurement_reason_codes': pair.get('reason_codes', []), 'status': 'INSUFFICIENT',
            'suggested_pipe_id': None, 'candidate_pipe_ids': [], 'reason_codes': []})
    return values


def _candidate_sets(observations, model, hints):
    candidates = []
    classes = sorted({round(m['diameter'], 2) for m in model})
    for observation in observations:
        hint = hints.get(observation['color_class'])
        choices = []
        if hint is not None:
            hint = float(hint)
            if not np.isfinite(hint) or hint <= 0:
                raise ValueError('INVALID_ASSIST_DIAMETER_HINT')
            choices = [i for i, m in enumerate(model) if abs(m['diameter']-hint) <= max(
                POLICY['hint_diameter_tolerance_mm'], hint*POLICY['hint_diameter_tolerance_ratio'])]
            observation.update(prior_diameter_mm=hint, diameter_source='USER_STATED_PRIOR_NOT_MEASUREMENT')
            nearest = min(classes, key=lambda d: abs(d-hint))
            index = classes.index(nearest)
            observation['relative_size_class'] = ('SMALL' if index == 0 else 'LARGE' if index == len(classes)-1 else 'MEDIUM') if len(classes) > 1 else 'SAME_SIZE'
        else:
            choices = [i for i, m in enumerate(model) if m['color_class'] == 'UNKNOWN'
                       or observation['color_class'] == 'UNKNOWN' or m['color_class'] == observation['color_class']]
            observation.update(prior_diameter_mm=None, diameter_source='IMAGE_WIDTH_ONLY', relative_size_class='UNRESOLVED')
        observation['candidate_pipe_ids'] = [model[i]['pipe_id'] for i in choices]
        if not choices:
            observation['reason_codes'].append('NO_COMPATIBLE_MODEL_SIZE_OR_COLOR_CLASS')
        candidates.append(choices)
    return candidates


def _image_hypothesis(observations, points, diameters, axis_world, side):
    """Weak perspective image ordering with unknown obliqueness/scale/offset."""
    basis = section_basis(axis_world)
    if side is None:
        view0 = basis[:, 0]
        angles = np.arange(0., 360., POLICY['view_sample_step_deg'])
    else:
        view0 = side-axis_world*(side@axis_world)
        if np.linalg.norm(view0) < 1e-8:
            return None
        view0 /= np.linalg.norm(view0)
        angles = np.arange(-85., 86., POLICY['view_sample_step_deg'])
    perp = np.cross(axis_world, view0)
    radians = np.radians(angles)
    views = np.cos(radians)[:, None]*view0+np.sin(radians)[:, None]*perp
    directions = np.cross(axis_world, views)
    directions = np.r_[directions, -directions]
    views = np.r_[views, views]
    x = points@directions.T
    x -= x.mean(axis=0)
    observed = np.array([o['image_transverse_px'] for o in observations])
    target = observed-observed.mean()
    variance = (x*x).sum(axis=0)
    scale = (target@x)/np.maximum(variance, 1e-12)
    widths = np.array([o['image_width_px'] for o in observations])
    valid = (variance > 1e-8) & (scale > 1e-8)
    residual = np.sqrt(np.mean((target[:, None]-x*scale)**2, axis=0))/max(float(np.median(widths)), 1.)
    expected_width = np.maximum(diameters[:, None]*scale, 1e-8)
    size = np.sqrt(np.mean(np.log(widths[:, None]/expected_width)**2, axis=0))
    score = .75*residual+.25*size
    score[~valid] = np.inf
    best = int(np.argmin(score))
    if not np.isfinite(score[best]):
        return None
    return {'difference_score': float(score[best]), 'layout_difference': float(residual[best]),
            'size_difference': float(size[best]), 'method': 'WEAK_PERSPECTIVE_IMAGE_LAYOUT',
            'camera_side_world_hypothesis': views[best].tolist(), 'scale_is_measurement': False}


def _prior_hypothesis(observations, points, axis_world, side, prior, characteristic):
    indexes = [i for i, o in enumerate(observations) if o['observation_id'] in prior]
    if len(indexes) < POLICY['minimum_layout_observations']:
        return None
    values = [prior[observations[i]['observation_id']] for i in indexes]
    axis_camera = _unit(values[0]['axis_camera'])
    if any(abs(_unit(v['axis_camera'])@axis_camera) < .999999 for v in values):
        return None
    bw, bc = section_basis(axis_world), section_basis(axis_camera)
    p = points[indexes]@bw
    q = np.array([v['section_center_uv_mm'] for v in values])
    p -= p.mean(axis=0); q -= q.mean(axis=0)
    # No scale adjustment and no tight millimetre acceptance threshold. A
    # model-neighbour-normalized difference ranks topology hypotheses only.
    if not np.isfinite(characteristic):
        return None
    u, _, vt = np.linalg.svd(p.T@q)
    results = []
    for parity in (1., -1.):
        r = u@np.diag([1., parity])@vt
        rotation = bc@r.T@bw.T+np.linalg.det(r)*np.outer(axis_camera, axis_world)
        view = rotation.T@np.array([0., 0., -1.])
        if side is not None and view@side <= 1e-6:
            continue
        rms = float(np.sqrt(np.mean(np.sum((p@r-q)**2, axis=1))))
        results.append({'difference_score': rms/characteristic, 'layout_difference': rms/characteristic,
            'method': 'USER_DIAMETER_PRIOR_RELATIVE_SECTION', 'size_difference': None,
            'conditional_layout_rms_mm': rms, 'not_a_metric_acceptance_test': True,
            'camera_side_world_hypothesis': view.tolist(), 'scale_adjusted': False,
            'prior_position_count': len(indexes)})
    return min(results, key=lambda r: r['difference_score']) if results else None


def _model_labels(model, axis, side):
    if not model:
        return {}
    sizes = sorted({round(m['diameter'], 2) for m in model})
    labels = {}
    transverse = None
    if side is not None:
        cross = np.cross(axis, side)
        if np.linalg.norm(cross) > 1e-8:
            transverse = cross/np.linalg.norm(cross)
    levels = sorted({round(float(m['center']@side), 2) for m in model}, reverse=True) if side is not None else []
    columns = sorted({round(float(m['center']@transverse), 2) for m in model}) if transverse is not None else []
    for m in model:
        index = sizes.index(round(m['diameter'], 2))
        layer = levels.index(round(float(m['center']@side), 2))+1 if levels else None
        layer_text = ('靠相机层' if layer == 1 else '远离相机层' if layer == len(levels) else f'中间层 {layer}') if layer else '层位待确定'
        if len(levels) == 1:
            layer_text = '同一设计层'
        lateral = columns.index(round(float(m['center']@transverse), 2))+1 if columns else None
        labels[m['pipe_id']] = {'model_layer_index': layer, 'model_layer_label': layer_text,
            'model_layer_index_from_camera': layer, 'model_layer_count': len(levels) if levels else None,
            'model_lateral_rank': lateral, 'model_lateral_count': len(columns) if columns else None,
            'model_layer_basis': 'DESIGN_COORDINATE_ALONG_SPECIFIED_CAMERA_SIDE' if layer else 'UNRESOLVED',
            'model_lateral_order': lateral,
            'model_nominal_diameter_mm': m['diameter'],
            'model_size_class': ('SMALL' if index == 0 else 'LARGE' if index == len(sizes)-1 else 'MEDIUM') if len(sizes) > 1 else 'SAME_SIZE'}
    return labels


def _relations(observations):
    result = []
    for first, second in itertools.combinations(observations, 2):
        difference = second['image_transverse_px']-first['image_transverse_px']
        result.append({'observation_ids': [first['observation_id'], second['observation_id']],
            'relation': 'IMAGE_TRANSVERSE_ORDER', 'second_on_positive_normal_side': bool(difference > 0),
            'image_normal': normal_from_axis(first['image_axis']).tolist(),
            'world_vertical_order_known': False,
            'relative_width_ratio': second['image_width_px']/first['image_width_px']})
    return result


def _assignments(choices, audit):
    """Budget every explored branch, including collisions; no giant products."""
    used, current = set(), []
    def visit(index):
        if index == len(choices):
            yield tuple(current)
            return
        for candidate in choices[index]:
            if audit['branches'] >= POLICY['maximum_assignments']:
                audit['truncated'] = True
                return
            audit['branches'] += 1
            if candidate in used:
                continue
            used.add(candidate); current.append(candidate)
            yield from visit(index+1)
            current.pop(); used.remove(candidate)
            if audit['truncated']:
                return
    yield from visit(0)


def analyze_installation_assist(report, pipe_specs, settings=None, *, calibration=None):
    """Return advisory suggestions without altering reports, inputs or strict states."""
    settings = dict(settings or {})
    options = settings.get('installation_assist') or settings
    hints = options.get('appearance_diameter_hints') or {}
    if not isinstance(hints, dict) or any(key not in SUPPORTED_COLOR_CLASSES for key in hints):
        raise ValueError('INVALID_ASSIST_COLOR_HINTS')
    if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in hints.values()):
        raise ValueError('INVALID_ASSIST_DIAMETER_HINT')
    model, axis = _model(pipe_specs, settings)
    side = _unit(settings['camera_side_world']) if settings.get('camera_side_world') is not None else None
    model_labels = _model_labels(model, axis, side)
    output = {'revision': REVISION, 'policy': dict(POLICY), 'advisory_only': True,
        'accepted_measurement': False, 'installation_acceptance': False, 'status': 'INSUFFICIENT',
        'observations': [], 'pipes': [{'pipe_id': m['pipe_id'], 'status': 'NO_OBSERVATION', 'observation_ids': [],
                                     **model_labels[m['pipe_id']]} for m in model],
        'ranked_hypotheses': [], 'reason_codes': [], 'pairwise_relations': [],
        'hint_source': options.get('hint_source'), 'appearance_diameter_hints': dict(hints),
        'synthetic_model_colors_ignored': any(m['synthetic_color_ignored'] for m in model),
        'counts': {'LIKELY': 0, 'AMBIGUOUS': 0, 'NO_OBSERVATION': len(model)},
        'known_diameter_used_for_localization': False, 'diameter_hint_source': options.get('hint_source'),
        'score_definition': 'RELATIVE_DIFFERENCE_LOWER_IS_BETTER_NOT_PROBABILITY',
        'relation_summary': ['图像横向顺序不自动等于世界上下层；设计层位随候选对应显示。',
                             '未观察到不表示未安装；本结果只辅助辨认，不判断尺寸或就位合格。']}
    stages = report.get('section_evaluations') or ([report['section_evaluation']] if report.get('section_evaluation') else [])
    if len(stages) != 1 or not model:
        output['reason_codes'] = ['NO_SINGLE_CURRENT_IMAGE_SECTION']
        return output
    stage = stages[0]
    observations = _observations(stage)
    output['observations'] = observations
    output['pairwise_relations'] = _relations(observations)
    candidates = _candidate_sets(observations, model, hints)
    active = [i for i, c in enumerate(candidates) if c]
    if len(active) < POLICY['minimum_layout_observations']:
        output['reason_codes'] = ['TOO_FEW_LAYOUT_OBSERVATIONS']
        return output
    selected = [observations[i] for i in active]
    choices = [candidates[i] for i in active]
    prior, prior_audit = {}, {'status': 'NOT_AVAILABLE'}
    if calibration is not None and hints:
        from .assist_prior_geometry import prior_positions
        prior_candidates = prior_positions(stage, calibration, hints)
        prior = {key: value for key, value in prior_candidates.items() if value.get('status') == 'PRIOR_CONDITIONAL_CENTER'}
        prior_audit = {'status': 'AVAILABLE' if prior else 'UNAVAILABLE', 'candidates': prior_candidates}
    output['prior_geometry'] = prior_audit
    use_prior = sum(o['observation_id'] in prior for o in selected) >= POLICY['minimum_layout_observations']
    if use_prior:
        qualified = [i for i, observation in enumerate(selected) if observation['observation_id'] in prior]
        for observation in selected:
            if observation['observation_id'] not in prior:
                observation['reason_codes'] = ['PRIOR_POSITION_UNAVAILABLE']
        selected, choices = [selected[i] for i in qualified], [choices[i] for i in qualified]
    output['position_source'] = 'USER_DIAMETER_PRIOR_RELATIVE_SECTION' if use_prior else 'IMAGE_LAYOUT_ONLY'
    output['method'] = output['position_source']
    output['known_diameter_used_for_localization'] = use_prior
    all_points = np.array([m['center'] for m in model])
    distances = np.linalg.norm(all_points[:, None]-all_points[None, :], axis=2)
    distances[distances < 1e-6] = np.inf
    characteristic = float(np.median(distances.min(axis=1)))
    if len(selected) > len(model):
        output['reason_codes'] = ['MORE_OBSERVATIONS_THAN_MODEL_MEMBERS']
        return output
    hypotheses, evaluated = [], 0
    search = {'branches': 0, 'truncated': False}
    for assignment in _assignments(choices, search):
        evaluated += 1
        points = np.array([model[i]['center'] for i in assignment])
        diameters = np.array([model[i]['diameter'] for i in assignment])
        result = (_prior_hypothesis(selected, points, axis, side, prior, characteristic) if use_prior else
                  _image_hypothesis(selected, points, diameters, axis, side))
        if result is None:
            continue
        result['assignments'] = [{'observation_id': o['observation_id'], 'pipe_id': model[i]['pipe_id'],
            'model_center_mm': model[i]['center'].tolist(), 'design_diameter_mm': model[i]['diameter'],
            'model_depth_coordinate_toward_camera_mm': float(model[i]['center']@side) if side is not None else None}
            for o, i in zip(selected, assignment)]
        for value in result['assignments']:
            value.update(model_labels[value['pipe_id']])
        hypotheses.append(result)
    hypotheses.sort(key=lambda h: (h['difference_score'], [a['pipe_id'] for a in h['assignments']]))
    truncated = search['truncated']
    output.update(evaluated_assignments=evaluated, search_branches=search['branches'], search_truncated=truncated)
    if not hypotheses:
        output['reason_codes'] = ['NO_COMPATIBLE_LAYOUT_HYPOTHESIS']
        return output
    best = hypotheses[0]['difference_score']
    plausible = [h for h in hypotheses if h['difference_score'] <= best+POLICY['near_best_difference']]
    output['plausible_hypothesis_count'] = len(plausible)
    output['ranked_hypotheses'] = [dict(h, rank=i+1, near_best=h in plausible)
        for i, h in enumerate(hypotheses[:POLICY['maximum_displayed_hypotheses']])]
    good = best <= POLICY['maximum_relative_difference'] and not truncated
    if not good:
        output['reason_codes'] = ['SEARCH_TRUNCATED' if truncated else 'RELATIVE_LAYOUT_CONFLICT']
    for observation in selected:
        alternatives = sorted({a['pipe_id'] for h in plausible for a in h['assignments'] if a['observation_id'] == observation['observation_id']})
        observation['candidate_pipe_ids'] = alternatives if not truncated else observation['candidate_pipe_ids']
        observation['status'] = 'LIKELY' if good and len(alternatives) == 1 else 'AMBIGUOUS' if good else 'INSUFFICIENT'
        observation['suggested_pipe_id'] = alternatives[0] if observation['status'] == 'LIKELY' else None
        observation['reason_codes'] = ([] if observation['status'] == 'LIKELY' else
            ['MULTIPLE_PLAUSIBLE_LAYOUTS'] if good else list(output['reason_codes']))
        if observation['observation_id'] in prior:
            observation['prior_position'] = prior[observation['observation_id']]
        if observation['suggested_pipe_id']:
            observation.update(model_labels[observation['suggested_pipe_id']])
    for row in output['pipes']:
        matching = [o for o in observations if row['pipe_id'] in o['candidate_pipe_ids'] and o['status'] != 'INSUFFICIENT']
        row['observation_ids'] = [o['observation_id'] for o in matching]
        row['status'] = ('LIKELY' if any(o['suggested_pipe_id'] == row['pipe_id'] for o in matching)
                         else 'AMBIGUOUS' if matching else 'NO_OBSERVATION')
    output['counts'] = {status: sum(p['status'] == status for p in output['pipes']) for status in ('LIKELY', 'AMBIGUOUS', 'NO_OBSERVATION')}
    output['status'] = 'LIKELY' if output['counts']['LIKELY'] else 'AMBIGUOUS' if good else 'INSUFFICIENT'
    return output
