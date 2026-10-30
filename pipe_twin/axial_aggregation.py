"""Versioned axial boundary pooling before any metric/model comparison.

No CAD diameter, camera distance in CAD or pipe length is used. Each edge is
tested in disjoint image strips; only consistent original edge samples are
combined. Occluded/missing columns are not filled and remain in the audit.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

REVISION = 'axial-boundary-aggregation-v1'
POLICY = {'window_px': 128, 'minimum_windows': 3, 'maximum_windows': 16,
          'minimum_points_per_window': 48, 'minimum_window_fraction': .75,
          'maximum_edge_rms_px': 1.25, 'maximum_window_offset_px': 1.5,
          'maximum_holdout_p95_px': 2., 'minimum_width_px': 8.,
          'adaptive_lengths_px': [384, 640, 1024], 'adaptive_positions': 3,
          'adaptive_selection': 'stable-edge-count-then-longest-v2',
          'maximum_fit_points_per_boundary': 128, 'maximum_reprojection_rms_px': 2.,
          'maximum_axis_nullspace_ratio': .25,
          'assumed_coherent_edge_sigma_px': .5}
POLICY_SHA256 = hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()


def _fit(points):
    x, y = np.asarray(points, float).T
    origin = x.mean()
    matrix = np.column_stack([x-origin, np.ones(len(x))])
    keep = np.ones(len(x), bool)
    for _ in range(4):
        slope, intercept = np.linalg.lstsq(matrix[keep], y[keep], rcond=None)[0]
        error = y-matrix@[slope, intercept]
        scale = 1.4826*np.median(np.abs(error-np.median(error)))
        keep = np.abs(error) <= max(.75, 3*scale)
        if keep.sum() < 2:
            raise ValueError('EDGE_UNRESOLVED')
    slope, intercept = np.linalg.lstsq(matrix[keep], y[keep], rcond=None)[0]
    error = y-matrix@[slope, intercept]
    return np.array([slope, intercept-slope*origin]), error, keep


def _column_runs(points):
    """Half-open runs of actual sampled columns, with holes left untouched."""
    columns = np.unique(np.floor(np.asarray(points)[:, 0]).astype(int))
    if not len(columns):
        return []
    starts = np.r_[0, np.flatnonzero(np.diff(columns) > 1)+1]
    ends = np.r_[starts[1:]-1, len(columns)-1]
    return [[int(columns[a]), int(columns[b])+1] for a, b in zip(starts, ends)]


def aggregate_boundaries(observations):
    """Pool top/bottom edge samples independently for one proposed instance.

    Input keys are role, edge, points (u,v); optional boundary_kind=occlusion.
    All image offsets are retained. Substrips are disjoint, not repeated
    overlapping windows masquerading as independent samples.
    """
    output, pooled = [], []
    seen = set()
    for obs in observations:
        key = (obs['role'], obs['edge'])
        if key in seen or key[0] not in ('left', 'right') or key[1] not in ('top', 'bottom'):
            raise ValueError('INVALID_OR_DUPLICATE_BOUNDARY')
        seen.add(key)
        audit = {'role': key[0], 'edge': key[1], 'boundary_kind': obs.get('boundary_kind', 'visible'),
                 'status': 'UNSTABLE', 'reason_codes': [], 'windows': []}
        if obs.get('boundary_kind') == 'occlusion':
            audit.update(status='OCCLUDED', reason_codes=['OCCLUDED_BOUNDARY_NOT_FILLED'])
            output.append(audit)
            continue
        points = np.asarray(obs['points'], float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
            raise ValueError('INVALID_EDGE_POINTS')
        # One sample per axial pixel; duplicates add no support.
        if len(points):
            _, indices = np.unique(np.floor(points[:, 0]), return_index=True)
            points = points[np.sort(indices)]
        audit['source_columns'] = len(points)
        audit['observed_column_runs_px'] = _column_runs(points)
        if len(points) < POLICY['minimum_points_per_window']*POLICY['minimum_windows']:
            audit['reason_codes'] = ['TOO_FEW_AXIAL_WINDOWS']
            output.append(audit)
            continue
        x0, x1 = float(points[:, 0].min()), float(points[:, 0].max())+1
        count = min(POLICY['maximum_windows'], max(1, int(np.ceil((x1-x0)/POLICY['window_px']))))
        # The cap expands disjoint blocks rather than silently ignoring ends.
        boundaries = np.linspace(x0, x1, count+1)
        global_line, _, _ = _fit(points)
        kept, window_sets = [], []
        for index, (low, high) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            subset = points[(points[:, 0] >= low) & (points[:, 0] < high)]
            window = {'index': index, 'axial_interval_px': [float(low), float(high)], 'columns': len(subset),
                      'accepted': False, 'reason': 'MISSING_OR_OCCLUDED_COLUMNS'}
            if len(subset) >= POLICY['minimum_points_per_window']:
                line, residual, inliers = _fit(subset)
                rms = float(np.sqrt(np.mean(residual[inliers]**2)))
                mid = float(subset[:, 0].mean())
                offset = float(abs((line-global_line)@[mid, 1.]))
                fraction = float(inliers.mean())
                good = (rms <= POLICY['maximum_edge_rms_px'] and offset <= POLICY['maximum_window_offset_px']
                        and fraction >= POLICY['minimum_window_fraction'])
                window.update(accepted=good, edge_rms_px=rms, offset_from_global_px=offset,
                              inlier_fraction=fraction, reason='CONSISTENT' if good else 'LOCAL_BOUNDARY_INCONSISTENT')
                if good:
                    kept.append(subset[inliers]); window_sets.append(subset[inliers])
            audit['windows'].append(window)
        accepted = len(kept)
        audit.update(total_windows=count, accepted_windows=accepted)
        if accepted < POLICY['minimum_windows'] or accepted/count < POLICY['minimum_window_fraction']:
            audit['reason_codes'] = ['TOO_FEW_CONSISTENT_AXIAL_WINDOWS']
            output.append(audit)
            continue
        joined = np.vstack(kept)
        line, residual, inliers = _fit(joined)
        # Each alternating half predicts the other half at its real positions.
        holdout_errors = []
        for parity in (0, 1):
            train = np.vstack(window_sets[parity::2])
            test = np.vstack(window_sets[1-parity::2])
            predicted, _, _ = _fit(train)
            holdout_errors.extend(abs(test[:, 1]-test[:, 0]*predicted[0]-predicted[1]))
        p95 = float(np.percentile(holdout_errors, 95))
        audit.update(line_slope_intercept=line.tolist(), retained_columns=int(inliers.sum()),
                     retained_column_runs_px=_column_runs(joined[inliers]),
                     pooled_rms_px=float(np.sqrt(np.mean(residual[inliers]**2))), holdout_p95_px=p95,
                     observed_axial_interval_px=[x0, x1], reference_u_px=(x0+x1-1)/2,
                     edge_v_at_reference_px=float(line@[(x0+x1-1)/2, 1.]))
        if p95 > POLICY['maximum_holdout_p95_px']:
            audit['reason_codes'] = ['CROSS_WINDOW_BOUNDARY_UNSTABLE']
        else:
            audit.update(status='STABLE', reason_codes=[])
            pooled.append({**obs, 'points': joined[inliers]})
        output.append(audit)
    return {'revision': REVISION, 'policy_sha256': POLICY_SHA256, 'policy': dict(POLICY),
            'scope': 'IMAGE_BOUNDARIES_NOT_METRIC_OR_OCCLUSION_RECOVERY',
            'boundaries': output, 'stable_boundary_count': sum(r['status'] == 'STABLE' for r in output),
            'occluded_boundary_count': sum(r['status'] == 'OCCLUDED' for r in output)}, pooled
