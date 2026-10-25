"""Small, held-out checked pitch/roll correction for section reconstruction.

This is a derived image operation, not a new calibration. Baseline, focal
length, yaw and the left coordinate frame are unchanged. A free horizontal
shift would manufacture disparity, so it is deliberately not estimated.
"""
from __future__ import annotations

import cv2
import numpy as np

REVISION = 'held-out-epipolar-pitch-roll-v1'


def transform_points(points, homography):
    points = np.asarray(points, float)
    q = np.c_[points, np.ones(len(points))] @ np.asarray(homography, float).T
    return q[:, :2] / q[:, 2:3]


def estimate_rotation(left, right, intrinsic, image_size):
    """Fit two angular DOFs; spatially interleaved tiles are held out entirely."""
    left, right, k = np.asarray(left, float), np.asarray(right, float), np.asarray(intrinsic, float)
    audit = {'revision': REVISION, 'applied': False, 'status': 'INSUFFICIENT_FEATURES',
             'calibration_updated': False, 'yaw_estimated': False,
             'baseline_or_focal_length_estimated': False, 'feature_count': len(left)}
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 2:
        raise ValueError('INVALID_EPIPOLAR_POINTS')
    if len(left) < 100 or not np.isfinite(left).all() or not np.isfinite(right).all():
        return audit
    width, height = image_size
    cells = np.floor(left / [width / 8, height / 6]).astype(int)
    train = (cells[:, 0] + cells[:, 1]) % 2 == 0
    if min(train.sum(), (~train).sum()) < 40 or len(np.unique(cells, axis=0)) < 10:
        return audit | {'status': 'INSUFFICIENT_SPATIAL_COVERAGE'}
    inverse = np.linalg.inv(k)

    def homography(parameters):
        rotation = cv2.Rodrigues(np.array([parameters[0], 0., parameters[1]]))[0]
        return k @ rotation @ inverse

    def residual(parameters):
        return transform_points(right, homography(parameters))[:, 1] - left[:, 1]

    parameters = np.zeros(2)
    for _ in range(12):
        errors = residual(parameters)
        jacobian = np.column_stack([(residual(parameters + np.eye(2)[i]*1.e-6)-errors)/1.e-6
                                    for i in range(2)])
        weight = np.minimum(1., .75 / np.maximum(np.abs(errors[train]), 1.e-9))
        design = jacobian[train] * np.sqrt(weight[:, None])
        if np.linalg.cond(design) > 100:
            return audit | {'status': 'ILL_CONDITIONED_FEATURE_GEOMETRY'}
        step = np.linalg.lstsq(design, errors[train]*np.sqrt(weight), rcond=None)[0]
        parameters -= step
        if np.linalg.norm(step) < 1.e-9:
            break
    before, after = np.abs(residual(np.zeros(2))[~train]), np.abs(residual(parameters)[~train])
    audit.update(train_count=int(train.sum()), validation_count=int((~train).sum()),
                 rotation_pitch_roll_deg=np.degrees(parameters).tolist(),
                 heldout_before_median_p95_px=np.percentile(before, [50, 95]).tolist(),
                 heldout_after_median_p95_px=np.percentile(after, [50, 95]).tolist())
    if np.abs(np.degrees(parameters)).max() > .2:
        return audit | {'status': 'CORRECTION_TOO_LARGE'}
    if (np.median(after) > .85*np.median(before) or np.median(before)-np.median(after) < .1
            or np.percentile(after, 95) > 2. or np.percentile(after, 95) > .9*np.percentile(before, 95)):
        return audit | {'status': 'NO_VALIDATED_IMPROVEMENT'}
    h = homography(parameters)
    corners = np.array([[0., 0.], [width-1., 0.], [0., height-1.], [width-1., height-1.]])
    shift = transform_points(corners, h)-corners
    audit['maximum_corner_shift_px'] = np.abs(shift).max(axis=0).tolist()
    if np.abs(shift).max() > 6.:
        return audit | {'status': 'CORRECTION_TOO_LARGE'}
    return audit | {'status': 'VALIDATED_IMAGE_REFINEMENT', 'applied': True,
                    'right_original_to_refined': h.tolist(),
                    'right_refined_to_original': np.linalg.inv(h).tolist(),
                    'metric_accuracy_validated': False}


def refine_right_image(left, right, intrinsic):
    """Use mutual distinctive matches; no CAD geometry or pipe dimensions."""
    detector = cv2.SIFT_create(nfeatures=6500)
    a, da = detector.detectAndCompute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), None)
    b, db = detector.detectAndCompute(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), None)
    if da is None or db is None or min(len(da), len(db)) < 2:
        return right, {'revision': REVISION, 'status': 'INSUFFICIENT_FEATURES', 'applied': False}
    matcher = cv2.BFMatcher()
    forward = [row[0] for row in matcher.knnMatch(da, db, k=2)
               if len(row) == 2 and row[0].distance < .7*row[1].distance]
    reverse = {row[0].queryIdx: row[0].trainIdx for row in matcher.knnMatch(db, da, k=2)
               if len(row) == 2 and row[0].distance < .7*row[1].distance}
    matches = [m for m in forward if reverse.get(m.trainIdx) == m.queryIdx]
    lp = np.array([a[m.queryIdx].pt for m in matches], float).reshape(-1, 2)
    rp = np.array([b[m.trainIdx].pt for m in matches], float).reshape(-1, 2)
    keep = (lp[:, 0]-rp[:, 0] > 5) & (lp[:, 0]-rp[:, 0] < 256) & (np.abs(lp[:, 1]-rp[:, 1]) < 15)
    lp, rp = lp[keep], rp[keep]
    if len(lp) >= 100:
        _, inliers = cv2.findFundamentalMat(lp, rp, cv2.USAC_MAGSAC, 1., .999, 5000)
        if inliers is not None:
            lp, rp = lp[inliers.ravel() != 0], rp[inliers.ravel() != 0]
    audit = estimate_rotation(lp, rp, intrinsic, (left.shape[1], left.shape[0]))
    if not audit['applied']:
        return right, audit
    return cv2.warpPerspective(right, np.asarray(audit['right_original_to_refined']),
                               (right.shape[1], right.shape[0])), audit
