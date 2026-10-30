"""Compare silhouettes on common conditional axial stations, not pixel origins.

The left/right silhouette rays touch different surface points. A provisional
cylinder supplies only an axial coordinate for partitioning existing samples;
it supplies no measured depth or hidden pixels. Centres are compared in the
same axis-normal basis, so axial displacement is never counted as section error.
"""
from __future__ import annotations

import numpy as np

from .axial_aggregation import POLICY
from .section_diagnostics import section_basis

REVISION = 'common-axial-stations-v2'


def conditional_axial_stations(observation, intrinsic, baseline_mm, fit):
    axis = np.asarray(fit['axis'], float)
    axis = axis/np.linalg.norm(axis)
    basis = section_basis(axis)
    camera = np.array([baseline_mm, 0., 0.]) if observation['role'] == 'right' else np.zeros(3)
    points = np.asarray(observation['points'], float)
    rays = np.c_[points, np.ones(len(points))]@np.linalg.inv(intrinsic).T
    radial = rays@basis
    centre = np.asarray(fit['section_center_parameters_mm'])-camera@basis
    denominator = np.sum(radial*radial, axis=1)
    if np.any(denominator < 1e-12):
        raise ValueError('AXIAL_RAY_DEGENERATE')
    # Closest approach of the ray to the fitted axis. For a true silhouette
    # this is the tangent generator's station; no cross-eye point matching.
    distance = (radial@centre)/denominator
    station = camera@axis+distance*(rays@axis)
    if not np.isfinite(station).all() or np.any(distance <= 0):
        raise ValueError('AXIAL_STATION_UNRESOLVED')
    return station


def synchronize_axial_blocks(observations, intrinsic, baseline_mm, fit):
    audit = {'revision': REVISION, 'status': 'NO_COMMON_AXIAL_SUPPORT',
             'mapping_source': 'CONDITIONAL_CYLINDER_FOR_PARTITION_ONLY',
             'supplies_measured_depth': False, 'windows': [],
             'centre_comparison': 'COMMON_AXIS_NORMAL_PLANE',
             'axial_displacement_in_center_error': False}
    if fit.get('status') != 'CONDITIONAL_FIT' or len(observations) < 3:
        return [], [], audit
    try:
        stations = [conditional_axial_stations(o, intrinsic, baseline_mm, fit) for o in observations]
    except (ValueError, np.linalg.LinAlgError):
        return [], [], audit | {'status': 'AXIAL_STATION_MAPPING_UNRESOLVED'}
    if not all(len(s) for s in stations):
        return [], [], audit
    low, high = max(s.min() for s in stations), min(s.max() for s in stations)
    if high <= low:
        return [], [], audit
    audit['common_interval_mm'] = [float(low), float(high)]
    # Keep approximately 128 image pixels per independent block; the actual
    # boundaries are common physical stations, including unequal eye crops.
    spans = []
    for obs, station in zip(observations, stations):
        inside = (station >= low) & (station <= high)
        points = np.asarray(obs['points'])[inside]
        if len(points) < POLICY['minimum_points_per_window']*POLICY['minimum_windows']:
            return [], [], audit
        # Order by the recovered axial station, not by image X: a vertical
        # silhouette can have constant X while covering a long actual strip.
        order = np.argsort(station[inside])
        spans.append(float(np.linalg.norm(points[order[-1]]-points[order[0]])))
    count = min(POLICY['maximum_windows'], int(np.ceil(min(spans)/POLICY['window_px'])))
    if count < POLICY['minimum_windows']:
        return [], [], audit
    edges = np.linspace(low, high, count+1)
    indices = [np.clip(np.searchsorted(edges, s, side='right')-1, 0, count-1) for s in stations]
    inside = [(s >= low) & (s <= high) for s in stations]
    accepted = []
    for index in range(count):
        counts = [int(np.count_nonzero(mask & (ids == index))) for mask, ids in zip(inside, indices)]
        enough = min(counts) >= POLICY['minimum_points_per_window']
        audit['windows'].append({'index': index, 'interval_mm': edges[index:index+2].tolist(),
                                 'point_counts': counts, 'accepted': enough})
        if enough:
            accepted.append(index)
    audit.update(total_windows=count, accepted_windows=len(accepted), window_edges_mm=edges.tolist(),
                 boundary_keys=[o['role']+':'+o['edge'] for o in observations])
    if len(accepted) < POLICY['minimum_windows'] or len(accepted)/count < POLICY['minimum_window_fraction']:
        return [], [], audit
    common, halves = [], [[], []]
    for obs, ids, mask in zip(observations, indices, inside):
        selected = mask & np.isin(ids, accepted)
        points = np.asarray(obs['points'])
        common.append({**obs, 'points': points[selected]})
        for parity in (0, 1):
            halves[parity].append({**obs, 'points': points[selected & (ids % 2 == parity)]})
    if any(len(o['points']) < POLICY['minimum_points_per_window'] for half in halves for o in half):
        return [], [], audit
    audit['status'] = 'COMMON_AXIAL_SUPPORT'
    return common, halves, audit
