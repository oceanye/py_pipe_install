"""Coarse centres from observed boundary summaries and an explicit size prior.

This deliberately answers a different question from free radius measurement.
Samples on retained observation-line runs represent those lines; they are not
new independent pixels, hidden contours, or the free cylinder's predicted lines.
"""
from __future__ import annotations

import math
import numpy as np

from .image_line_geometry import canonical_image_axis, point_at_station
from .projection_diagnostics import fit_silhouettes
from .section_diagnostics import section_basis

REVISION = 'diameter-prior-observed-boundary-centres-v1'


def _colour_class(appearance):
    if appearance == 'neutral':
        return 'WHITE'
    try:
        hue = float(str(appearance).split('hue-', 1)[1]) % 180
    except (ValueError, IndexError):
        return None
    if hue <= 15 or hue >= 165:
        return 'RED'
    if 90 <= hue <= 135:
        return 'BLUE'
    return None


def _prior(appearance, hints):
    value = hints.get(appearance, hints.get(_colour_class(appearance)))
    source = 'USER_STATED_PRIOR'
    if isinstance(value, dict):
        source = value.get('source', source)
        value = value.get('diameter_mm', value.get('prior_diameter_mm'))
    if isinstance(value, bool):
        raise ValueError('DIAMETER_PRIOR_UNAVAILABLE')
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError('DIAMETER_PRIOR_UNAVAILABLE') from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError('DIAMETER_PRIOR_UNAVAILABLE')
    return value, str(source)


def _line_samples(boundary, view, offset):
    if boundary.get('status') != 'STABLE':
        raise ValueError('BOUNDARY_NOT_STABLE')
    oriented = 'retained_axial_runs_px' in boundary
    runs = boundary.get('retained_axial_runs_px' if oriented else 'retained_column_runs_px')
    if not runs:
        raise ValueError('OBSERVED_BOUNDARY_SUPPORT_UNAVAILABLE')
    line = boundary.get('line_homogeneous')
    if line is None:
        slope = np.asarray(boundary.get('line_slope_intercept'), float)
        if slope.shape != (2,):
            raise ValueError('OBSERVED_BOUNDARY_LINE_UNAVAILABLE')
        line = [-slope[0], 1., -slope[1]]
    line = np.asarray(line, float)
    if line.shape != (3,) or not np.isfinite(line).all() or np.linalg.norm(line[:2]) < 1e-12:
        raise ValueError('OBSERVED_BOUNDARY_LINE_UNAVAILABLE')
    image_axis = canonical_image_axis(boundary.get('image_axis', view.get('image_axis', [1., 0.])))
    station_axis = image_axis if oriented else np.array([1., 0.])
    # Select only retained bins. Disjoint runs stay disjoint, including when the
    # sample budget reduces the number of line representatives.
    intervals = []
    for low, high in runs:
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError('INVALID_OBSERVED_BOUNDARY_SUPPORT')
        if intervals and low < intervals[-1][1]:
            raise ValueError('INVALID_OBSERVED_BOUNDARY_SUPPORT')
        intervals.append([low, high])
    intervals = np.asarray(intervals)
    lengths = intervals[:, 1]-intervals[:, 0]
    ends = np.cumsum(lengths)
    margin = min(.5, ends[-1]/4)
    distance = np.linspace(margin, ends[-1]-margin, min(128, max(2, int(math.ceil(ends[-1])))))
    index = np.searchsorted(ends, distance, side='right')
    stations = intervals[index, 0]+distance-np.r_[0., ends[:-1]][index]
    points = np.array([point_at_station(line, station_axis, s) for s in stations])+offset
    return points, image_axis, len(runs)


def prior_positions(stage, calibration, appearance_diameter_hints):
    """Return candidate-ID keyed conditional centres; never mutate the stage.

    ``calibration`` is the original full-image rectified calibration. Zone
    offsets are restored to it. Hints are keyed by observed appearance or
    RED/BLUE/WHITE, and contain a diameter or {diameter_mm, source}. Synthetic
    STL display colours and predicted cylinder lines are never read here.
    """
    candidates = stage.get('candidates') or []
    result = {}
    if not candidates:
        return result
    views = {role: {v['candidate_id']: v for v in (stage.get('views') or {}).get(role, [])}
             for role in ('left', 'right')}
    correction = stage.get('ground_correction') or {}
    axis_data = stage.get('axis') or {}
    axis_source = 'SECTION_AXIS'
    if (correction.get('trial_mode') or correction.get('boundary_check_passed') is False
            or axis_data.get('status') != 'STABLE'):
        free = stage.get('unconstrained_axis') or {}
        if free.get('status') == 'STABLE':
            axis_data, axis_source = free, 'FREE_IMAGE_AXIS'
    ids = [str(c.get('candidate_id', '')) for c in candidates]
    for candidate in candidates:
        identity = str(candidate.get('candidate_id', ''))
        entry = {'revision': REVISION, 'status': 'PRIOR_CENTER_UNAVAILABLE',
                 'accepted_measurement': False, 'installation_evidence': False,
                 'source': 'OBSERVED_BOUNDARY_LINES_WITH_DIAMETER_PRIOR',
                 'independent_pixel_samples': False, 'predicted_cylinder_lines_used': False,
                 'axis_source': axis_source, 'reason_codes': []}
        result[identity] = entry
        try:
            if not identity or ids.count(identity) != 1:
                raise ValueError('DUPLICATE_OR_MISSING_OBSERVATION_ID')
            if calibration is None:
                raise ValueError('CALIBRATION_UNAVAILABLE')
            if (stage.get('epipolar_refinement') or {}).get('applied'):
                raise ValueError('REFINED_IMAGE_COORDINATES_NOT_SUPPORTED_FOR_PRIOR_CENTER')
            if axis_data.get('status') != 'STABLE':
                raise ValueError('IMAGE_AXIS_UNRESOLVED')
            axis = np.asarray(axis_data['axis'], float)
            left = views['left'].get(candidate.get('left_candidate_id'), {})
            right = views['right'].get(candidate.get('right_candidate_id'), {})
            diameter, source = _prior(left.get('appearance'), appearance_diameter_hints or {})
            entry.update(prior_diameter_mm=diameter, diameter_source=source,
                         axis_camera=axis.tolist())
            k = np.array(calibration.left.rectified_intrinsic, float, copy=True)
            kr = np.asarray(calibration.right.rectified_intrinsic, float)
            if not np.allclose(k, kr, atol=1e-7, rtol=0):
                raise ValueError('FULL_RECTIFIED_INTRINSICS_DIFFER')
            common = canonical_image_axis(left.get('image_axis', [1., 0.]))
            observations, runs = [], []
            offsets = stage.get('zone_offsets') or {}
            for role, view in (('left', left), ('right', right)):
                offset = np.array([offsets.get(role+'_x', 0.), offsets.get(role+'_y', 0.)])
                for boundary in (view.get('aggregation') or {}).get('boundaries', []):
                    if boundary.get('status') != 'STABLE' or boundary.get('boundary_kind') == 'occlusion':
                        continue
                    points, image_axis, count = _line_samples(boundary, view, offset)
                    edge = boundary['edge']
                    if image_axis@common < 0:
                        edge = 'bottom' if edge == 'top' else 'top'
                    observations.append({'role': role, 'edge': edge, 'points': points,
                                         'image_axis': common.tolist()})
                    runs.append({'role': role, 'edge': edge, 'retained_runs': count,
                                 'line_representatives': len(points)})
            if not any(o['role'] == 'left' for o in observations):
                raise ValueError('OBSERVED_LEFT_BOUNDARY_UNAVAILABLE')
            reference = np.mean(np.vstack([o['points'] for o in observations if o['role'] == 'left']), axis=0)
            fit = fit_silhouettes(observations, k, calibration.baseline_mm, axis,
                                  image_axis=common, reference_pixel=reference,
                                  known_diameter_mm=diameter)
            if fit['status'] != 'CONDITIONAL_FIT':
                raise ValueError(fit['status'])
            params = np.asarray(fit['section_center_parameters_mm'])
            center = section_basis(axis)@params
            entry.update(status='PRIOR_CONDITIONAL_CENTER', section_center_uv_mm=params.tolist(),
                         center_camera_mm=center.tolist(),
                         reprojection_rms_px=fit['reprojection_rms_px'],
                         boundary_support=runs, coordinate_frame='LEFT_RECTIFIED_CAMERA_MM',
                         limits=['DIAMETER_IS_SUPPLIED_PRIOR_NOT_MEASUREMENT',
                                 'LINE_SUMMARY_NOT_INDEPENDENT_RAW_PIXELS',
                                 'NOT_A_METRIC_OR_INSTALLATION_ACCEPTANCE'])
        except (ValueError, KeyError, TypeError, np.linalg.LinAlgError) as error:
            entry['reason_codes'] = [str(error)]
    return result
