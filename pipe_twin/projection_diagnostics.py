"""Conditional silhouette fitting for a straight, locally uniform cylinder.

Offline diagnostic only. Image outlines are tangent rays, not stereo matches
of the same material surface point. Radius and section centre are fitted by
forward projection without using CAD dimensions or dense depth.
"""
from __future__ import annotations

import numpy as np
import cv2

from .section_diagnostics import section_basis
from .image_line_geometry import canonical_image_axis, normal_from_axis, fit_image_line

REVISION = 'direction-independent-cylinder-projection-v2'


def pool_strip_mask(mask, x0, x1, q0, q1, slope):
    """Average observed paint support along the axis, retaining image position.

    q=(v-slope*(u-u_reference))/sqrt(1+slope**2). The transform is recorded,
    never re-centred independently in the two views. Bilinear resampling is
    only for 2D image evidence, not missing 3D depth. Multiple intervals stay
    separate; occupancy is not a calibrated confidence or physical pipe count.
    """
    if mask.ndim != 2 or not 0 <= x0 < x1 <= mask.shape[1] or q1 <= q0:
        raise ValueError('INVALID_STRIP')
    reference = (x0+x1-1)/2
    q = np.arange(q0, q1+.125, .25)
    xx, qq = np.meshgrid(np.arange(x0, x1), q)
    yy = slope*(xx-reference)+qq*np.sqrt(1+slope*slope)
    valid = (yy >= 0) & (yy <= mask.shape[0]-1)
    sampled = cv2.remap(mask.astype(np.float32), xx.astype(np.float32), yy.astype(np.float32),
                        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    counts = valid.sum(axis=1)
    profile = np.divide((sampled*valid).sum(axis=1), counts, out=np.zeros(len(q)), where=counts > 0)
    inside = profile >= .5
    transitions = np.diff(np.r_[False, inside, False].astype(int))
    intervals = []
    for start, end in zip(np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)):
        lo, hi = float(q[start]), float(q[end-1])
        if start > 0:
            lo = float(q[start-1]+(.5-profile[start-1])/(profile[start]-profile[start-1])*(q[start]-q[start-1]))
        if end < len(q):
            hi = float(q[end-1]+(.5-profile[end-1])/(profile[end]-profile[end-1])*(q[end]-q[end-1]))
        intervals.append({'low_q_px': lo, 'high_q_px': hi, 'width_px': hi-lo,
                          'clipped': bool(start == 0 or end == len(q))})
    return {'reference_u_px': reference, 'slope': float(slope), 'columns': [x0, x1],
            'q_px': q.tolist(), 'paint_fraction': profile.tolist(), 'intervals': intervals,
            'scope': 'IMAGE_PROJECTION_ONLY_NO_METRIC_SCALE'}


def _image_reference(observations, reference_u, image_axis, reference_pixel):
    """Resolve a shared sampling frame and an image transversal, never metric length."""
    if image_axis is None:
        directions = []
        for observation in observations:
            if observation.get('image_axis') is not None:
                direction = canonical_image_axis(observation['image_axis'])
            else:
                line = fit_image_line(observation['points'])
                direction = canonical_image_axis([-line[1], line[0]])
            directions.append(direction)
        _, vectors = np.linalg.eigh(sum(np.outer(a, a) for a in directions))
        image_axis = vectors[:, -1]
    image_axis = canonical_image_axis(image_axis)
    if reference_pixel is not None:
        pixel = np.asarray(reference_pixel, float)
        direction = image_axis
    elif reference_u is not None:
        # Preserve the historical u-column reference for callers requesting it.
        pixel = np.array([float(reference_u), 0.])
        direction = np.array([1., 0.])
    else:
        pixels = np.vstack([o['points'] for o in observations if o['role'] == 'left'])
        pixel = np.mean(pixels, axis=0)
        direction = image_axis
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        raise ValueError('INVALID_REFERENCE_PIXEL')
    return image_axis, pixel, direction


def _reference_station(center, axis, intrinsic, reference_pixel, reference_direction):
    image_plane = np.r_[reference_direction, -reference_direction@reference_pixel]
    plane = np.asarray(intrinsic).T @ image_plane
    plane /= np.linalg.norm(plane)
    denominator = plane @ axis
    if abs(denominator) < 1e-10:
        raise ValueError('REFERENCE_STATION_UNOBSERVABLE')
    return center-axis*((plane@center)/denominator)


def projected_lines(section_center, radius, basis, intrinsic, camera_center, reference_u=None,
                    *, image_axis=None, reference_pixel=None):
    """Tangent lines ordered by the shared image normal, including vertical pipes.

    Coefficients and all samples stay in original rectified pixel coordinates.
    The ordering transversal is only a local line-label convention.
    """
    q = np.asarray(section_center)-basis.T@np.asarray(camera_center)
    distance = np.linalg.norm(q)
    if radius <= 0 or distance <= radius or not np.isfinite(distance):
        raise ValueError('CAMERA_INSIDE_CYLINDER_OR_INVALID_RADIUS')
    g = q/distance
    t = np.array([-g[1], g[0]])
    ratio = radius/distance
    normals = basis@np.column_stack((ratio*g+np.sqrt(1-ratio**2)*t,
                                    ratio*g-np.sqrt(1-ratio**2)*t))
    lines = (np.linalg.inv(intrinsic).T@normals).T
    lengths = np.linalg.norm(lines[:, :2], axis=1)
    if np.any(lengths < 1e-12):
        raise ValueError('DEGENERATE_PROJECTED_TANGENT')
    lines /= lengths[:, None]
    image_normal = normal_from_axis([1., 0.] if image_axis is None else image_axis)
    reference = np.asarray(reference_pixel if reference_pixel is not None
                           else [0. if reference_u is None else reference_u, 0.], float)
    denominator = lines[:, :2] @ image_normal
    if np.any(np.abs(denominator) < 1e-8):
        raise ValueError('IMAGE_REFERENCE_STATION_UNOBSERVABLE')
    transverse = -(lines[:, :2]@reference+lines[:, 2])/denominator
    return lines[np.argsort(transverse)]


def fit_silhouettes(observations, intrinsic, baseline_mm, axis, reference_u=None, *, known_diameter_mm=None,
                    image_axis=None, reference_pixel=None):
    """Fit a conditional cylinder; preserve all coordinate offsets.

    Each observation provides role, edge ('top'/'bottom'), and Nx2 image points.
    A boundary explicitly labelled 'occlusion' is excluded. Three independent
    tangent planes across both eyes can suffice for the three unknowns; two
    cannot. With an explicitly supplied known diameter, only two centre
    parameters remain; two independent tangents may then suffice. The supplied
    diameter is recorded as a constraint, never an independent measurement.
    Hidden predicted contours never become observed evidence.
    This diagnostic assumes equal rectified intrinsics and a +X baseline.
    It does not establish segmentation, calibration quality, or field accuracy.
    """
    k = np.asarray(intrinsic, float)
    if known_diameter_mm is not None and (not np.isfinite(known_diameter_mm) or known_diameter_mm <= 0):
        raise ValueError('INVALID_KNOWN_DIAMETER')
    fixed_radius = None if known_diameter_mm is None else float(known_diameter_mm)/2
    parameter_count = 3 if fixed_radius is None else 2
    a = np.array(axis, dtype=float, copy=True)
    if k.shape != (3, 3) or not np.isfinite(k).all() or baseline_mm <= 0:
        raise ValueError('INVALID_CAMERA_GEOMETRY')
    basis = section_basis(a)
    a /= np.linalg.norm(a)
    base = np.array([baseline_mm, 0., 0.])
    perpendicular = float(np.linalg.norm(np.cross(base, a)))
    result = {'revision': REVISION, 'scope': 'CONDITIONAL_PROJECTION_NOT_MEASUREMENT', 'accepted_diameter_mm': None,
              'axis': a.tolist(), 'physical_baseline_mm': float(baseline_mm),
              'baseline_perpendicular_to_axis_mm': perpendicular,
              'diameter_source': 'IMAGE_FIT' if fixed_radius is None else 'USER_KNOWN_CONSTRAINT',
              'known_diameter_mm': known_diameter_mm}
    if perpendicular < 1e-6:
        return result | {'status': 'UNOBSERVABLE_PARALLEL_BASELINE'}
    allowed = {('left', 'top'), ('left', 'bottom'), ('right', 'top'), ('right', 'bottom')}
    keys = [(o['role'], o['edge']) for o in observations]
    if not set(keys) <= allowed or len(set(keys)) != len(keys):
        raise ValueError('INVALID_OR_DUPLICATE_BOUNDARY')
    ignored = [o for o in observations if o.get('boundary_kind') == 'occlusion']
    observations = [o for o in observations if o.get('boundary_kind') != 'occlusion']
    result['ignored_occlusion_boundaries'] = [[o['role'], o['edge']] for o in ignored]
    result['observed_boundary_count'] = len(observations)
    if len(observations) < parameter_count or len({o['role'] for o in observations}) < 2:
        return result | {'status': 'INSUFFICIENT_INDEPENDENT_SILHOUETTES'}
    image_axis, reference, reference_direction = _image_reference(
        observations, reference_u, image_axis, reference_pixel)
    result.update(image_axis=image_axis.tolist(), reference_pixel_px=reference.tolist(),
                  reference_direction_image=reference_direction.tolist(),
                  reference_definition='IMAGE_TRANSVERSE_LINE' if reference_pixel is not None or reference_u is None
                  else 'LEGACY_IMAGE_U_COLUMN')
    matrices, rhs, normalized = [], [], []
    for obs in observations:
        points = np.asarray(obs['points'], float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
            raise ValueError('INVALID_EDGE_POINTS')
        normal = k.T@fit_image_line(points, image_axis)
        normal -= a*(normal@a)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            return result | {'status': 'UNOBSERVABLE_TANGENT_PLANES'}
        normal /= norm
        if obs['edge'] == 'bottom':
            normal = -normal
        camera = np.zeros(3) if obs['role'] == 'left' else base
        matrices.append([*(normal@basis), -1.])
        rhs.append(normal@camera)
        normalized.append((obs['role'], 0 if obs['edge'] == 'top' else 1, points))
    matrices, rhs = np.asarray(matrices), np.asarray(rhs)
    if fixed_radius is not None:
        matrices, rhs = matrices[:, :2], rhs+fixed_radius
    initial, _, rank, singular = np.linalg.lstsq(matrices, rhs, rcond=None)
    if rank < parameter_count:
        return result | {'status': 'UNOBSERVABLE_TANGENT_PLANES'}

    def residual(p):
        radius = p[2] if fixed_radius is None else fixed_radius
        predicted = {role: projected_lines(p[:2], radius, basis, k, camera,
                                          image_axis=image_axis, reference_pixel=reference)
                     for role, camera in [('left', np.zeros(3)), ('right', base)]}
        errors = []
        for role, index, points in normalized:
            line = predicted[role][index]
            errors.extend(points@line[:2]+line[2])
        return np.array(errors)

    p = initial.copy()
    damping = 1e-3
    try:
        residual(p)
    except ValueError:
        return result | {'status': 'INVALID_INITIAL_GEOMETRY'}
    for _ in range(45):
        error = residual(p)
        jac = np.column_stack([(residual(p+np.eye(parameter_count)[j]*.001)-error)/.001
                               for j in range(parameter_count)])
        normal = jac.T@jac
        step = np.linalg.solve(normal+damping*np.diag(np.maximum(np.diag(normal), 1e-9)), -jac.T@error)
        candidate = p+step
        try:
            improves = np.sum(residual(candidate)**2) < np.sum(error**2)
        except ValueError:
            improves = False
        if improves:
            p = candidate
            damping = max(damping/3, 1e-9)
            if np.linalg.norm(step) < 1e-7:
                break
        else:
            damping = min(damping*10, 1e12)
    center = basis@p[:2]  # The point on the axis closest to the left origin.
    try:
        station = _reference_station(center, a, k, reference, reference_direction)
    except ValueError as error:
        return result | {'status': str(error)}
    if station[2] <= 0:
        return result | {'status': 'CYLINDER_BEHIND_CAMERA'}
    radius = p[2] if fixed_radius is None else fixed_radius
    return result | {'status': 'CONDITIONAL_FIT', 'trial_diameter_mm': float(2*radius),
                     'section_center_parameters_mm': p[:2].tolist(),
                     # Legacy consumers retain the same point under the old key.
                     'center_at_reference_column_mm': station.tolist(),
                     'center_at_reference_pixel_mm': station.tolist(),
                     'reference_left_column_px': float(reference_u) if reference_u is not None else None,
                     'reprojection_rms_px': float(np.sqrt(np.mean(residual(p)**2))),
                     'linear_condition': float(singular[0]/singular[-1]),
                     'predicted_lines': {role: projected_lines(p[:2], radius, basis, k, camera,
                        image_axis=image_axis, reference_pixel=reference).tolist()
                                         for role, camera in [('left', np.zeros(3)), ('right', base)]}}


def pixel_error_scenarios(observations, intrinsic, baseline_mm, axis, reference_u=None, sigma_px=.5,
                          *, known_diameter_mm=None, image_axis=None, reference_pixel=None):
    """Propagate explicitly assumed pixel noise; never a total accuracy claim.

    The IID scenario assumes each observed pixel has independent transverse
    noise. The coherent-edge scenario assumes each whole edge has an offset
    of sigma_px, independently across edges. More samples reduce only the
    first scenario. Calibration, occlusion and segmentation bias are excluded.
    """
    if not np.isfinite(sigma_px) or sigma_px <= 0:
        raise ValueError('INVALID_PIXEL_SIGMA')
    intrinsic = np.asarray(intrinsic, dtype=float)
    observations = [o for o in observations if o.get('boundary_kind') != 'occlusion']
    fitted = fit_silhouettes(observations, intrinsic, baseline_mm, axis, reference_u,
                             known_diameter_mm=known_diameter_mm, image_axis=image_axis,
                             reference_pixel=reference_pixel)
    if fitted['status'] != 'CONDITIONAL_FIT':
        return {'status': fitted['status'], 'total_accuracy_known': False}
    image_axis = np.asarray(fitted['image_axis'])
    image_normal = normal_from_axis(image_axis)
    reference = np.asarray(fitted['reference_pixel_px'])
    reference_direction = np.asarray(fitted['reference_direction_image'])

    def shifted(index, offset):
        values = [{**o, 'points': np.array(o['points'], dtype=float, copy=True)} for o in observations]
        values[index]['points'] += offset*image_normal
        fit = fit_silhouettes(values, intrinsic, baseline_mm, axis, reference_u,
                              known_diameter_mm=known_diameter_mm, image_axis=image_axis,
                              reference_pixel=reference_pixel)
        if fit['status'] != 'CONDITIONAL_FIT':
            raise ValueError('PIXEL_PERTURBATION_UNOBSERVABLE')
        return np.array([*fit['section_center_parameters_mm'], fit['trial_diameter_mm']/2,
                         fit['trial_diameter_mm'], fit['center_at_reference_pixel_mm'][2]])

    jac = np.column_stack([(shifted(i, .05)-shifted(i, -.05))/.1 for i in range(len(observations))])
    counts = np.array([len(o['points']) for o in observations], dtype=int)
    # Propagate independent point noise through the actual least-squares
    # problem, including each point's coordinate. Counting samples alone or
    # treating edge-slope errors as a mean offset would be incomplete.
    basis = section_basis(axis)
    unit_axis = np.asarray(axis, float)/np.linalg.norm(axis)
    params = np.asarray(fitted['section_center_parameters_mm'])
    if known_diameter_mm is None:
        params = np.r_[params, fitted['trial_diameter_mm']/2]
    parameter_count = len(params)

    def radius(p):
        return p[2] if known_diameter_mm is None else known_diameter_mm/2

    def point_residual(p):
        values, normal_weights = [], []
        for obs in observations:
            camera = [0., 0., 0.] if obs['role'] == 'left' else [baseline_mm, 0., 0.]
            lines = projected_lines(p[:2], radius(p), basis, np.asarray(intrinsic), camera,
                                    image_axis=image_axis, reference_pixel=reference)
            line = lines[0 if obs['edge'] == 'top' else 1]
            points = np.asarray(obs['points'])
            values.extend(points@line[:2]+line[2])
            normal_weights.extend([line[:2]@image_normal]*len(points))
        return np.asarray(values), np.asarray(normal_weights)

    residual, normal_weights = point_residual(params)
    point_jac = np.column_stack([(point_residual(params+np.eye(parameter_count)[i]*.001)[0]-residual)/.001
                                 for i in range(parameter_count)])
    inverse = np.linalg.inv(point_jac.T@point_jac)
    covariance_params = inverse@(point_jac.T@(normal_weights[:, None]**2*point_jac))@inverse*sigma_px**2

    def output_values(p):
        center = basis@p[:2]
        station = _reference_station(center, unit_axis, intrinsic, reference, reference_direction)
        return np.array([*p[:2], radius(p), 2*radius(p), station[2]])

    output_jac = np.column_stack([(output_values(params+np.eye(parameter_count)[i]*.001)-output_values(params))/.001
                                  for i in range(parameter_count)])
    iid_covariance = output_jac@covariance_params@output_jac.T
    scenarios = {}
    for label, effective_counts in [('IID_PIXELS_OPTIMISTIC', counts), ('COHERENT_EDGE_OFFSETS', np.ones(len(counts)))]:
        cov = iid_covariance if label == 'IID_PIXELS_OPTIMISTIC' else sigma_px**2*jac@jac.T
        std = np.sqrt(np.maximum(0, np.diag(cov)))
        scenarios[label] = {'assumed_effective_samples_per_edge': effective_counts.tolist(),
                            'section_parameter_order': ['u_mm', 'v_mm', 'radius_mm'],
                            'section_parameter_covariance_mm2': cov[:3, :3].tolist(),
                            'diameter_sigma_mm': float(std[3]), 'radius_sigma_mm': float(std[3]/2),
                            'center_z_sigma_mm': float(std[4]),
                            'normal_95_halfwidth_diameter_mm': float(1.96*std[3]),
                            'normal_95_halfwidth_radius_mm': float(.98*std[3]),
                            'normal_95_halfwidth_center_z_mm': float(1.96*std[4])}
    return {'revision': REVISION, 'status': 'ASSUMPTION_CONDITIONAL_LINEAR_PROPAGATION', 'total_accuracy_known': False,
            'assumed_normal_pixel_sigma_px': float(sigma_px), 'pixel_noise_direction_image': image_normal.tolist(),
            'pixel_noise_definition': 'SHARED_IMAGE_AXIS_NORMAL', 'point_counts_per_edge': counts.tolist(),
            'edge_order': [[o['role'], o['edge']] for o in observations],
            'diameter_and_z_jacobian_mm_per_px': jac[3:].tolist(),
            'section_parameters_jacobian_mm_per_px': jac[:3].tolist(), 'scenarios': scenarios,
            'excluded': ['CALIBRATION_SCALE_AND_RECTIFICATION', 'MISIDENTIFIED_SILHOUETTE',
                         'OCCLUSION_CLASSIFICATION', 'AXIS_ERROR', 'GENERAL_PIXEL_NOISE_CORRELATION']
                        + (['SUPPLIED_DIAMETER_TOLERANCE'] if known_diameter_mm is not None else [])}
