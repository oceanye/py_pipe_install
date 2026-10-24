"""Read-only section diagnostics; never an accepted production measurement.

The caller supplies a conditional common axis, not a CAD radius. Projecting a
short axial neighbourhood into its normal plane preserves a straight cylinder's
section, but does not prove that the input contains only one cylinder.
"""
from __future__ import annotations

import numpy as np


def axis_from_image_lines(lines, intrinsic):
    """Recover an unoriented common direction from calibrated image lines.

All lines must be silhouettes/edges parallel to the same physical direction.
Their backprojected planes contain that direction even across translated eyes.
At least two distinct planes are necessary; a low residual is not calibration
validation. Returns singular values so ill conditioning remains inspectable.
"""
    lines = np.asarray(lines, dtype=float)
    k = np.asarray(intrinsic, dtype=float)
    if lines.ndim != 2 or lines.shape[1] != 3 or len(lines) < 2 or k.shape != (3, 3):
        raise ValueError("INVALID_LINES_OR_INTRINSIC")
    normals = lines @ k
    norms = np.linalg.norm(normals, axis=1)
    if not np.isfinite(normals).all() or np.any(norms < 1e-12):
        raise ValueError("INVALID_LINE_NORMAL")
    normals /= norms[:, None]
    _, singular, vectors = np.linalg.svd(normals, full_matrices=True)
    if singular[1] < 1e-6:
        raise ValueError("AXIS_UNOBSERVABLE")
    axis = vectors[-1]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    angles = np.degrees(np.arcsin(np.clip(np.abs(normals @ axis), 0, 1)))
    return {"axis": axis.tolist(), "singular_values": singular.tolist(),
            "line_residual_degrees": angles.tolist()}


def section_basis(axis):
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) < 1e-9:
        raise ValueError("INVALID_AXIS")
    axis = axis / np.linalg.norm(axis)
    # For horizontal pipes, u is image-down and v points approximately away
    # from the camera. Use a fallback for axes nearly parallel to camera Z.
    reference = np.array([0., 0., 1.]) if abs(axis[2]) < .95 else np.array([0., 1., 0.])
    u = np.cross(reference, axis)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return np.column_stack((u, v))


def circle_trial(section_points):
    """Bounded robust circle hypothesis, exposing failures and all-point errors.

No physical pipe count, CAD radius, missing-depth interpolation, axial length
gate, or production acceptance side effect. Geometry passing is only one
necessary check; stereo agreement, segmentation and calibration remain pending.
"""
    points = np.asarray(section_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("INVALID_SECTION_POINTS")
    result = {"scope": "DIAGNOSTIC_NOT_MEASUREMENT", "point_count": len(points),
              "measured_diameter_mm": None, "geometry_passed": False, "reasons": []}
    if len(points) < 80:
        result["reasons"].append("TOO_FEW_POINTS")
    if len(points) < 6:
        return result
    sample = points[np.linspace(0, len(points)-1, min(len(points), 1600), dtype=int)]
    origin = np.mean(sample, axis=0)
    q = sample - origin

    def seed(p):
        matrix = np.column_stack((2*p, np.ones(len(p))))
        sol, _, rank, _ = np.linalg.lstsq(matrix, np.sum(p*p, axis=1), rcond=None)
        r2 = sol[2] + sol[:2] @ sol[:2]
        if rank < 3 or r2 <= 0 or not np.isfinite(r2):
            return None
        return np.r_[sol[:2], np.sqrt(r2)]

    def errors(p):
        return np.linalg.norm(q-p[:2], axis=1)-p[2]

    def score(p):
        e = np.abs(errors(p))
        limit = min(1.5, p[2]*.05)
        return (int(np.count_nonzero(e <= limit)), -float(np.median(e)))

    rng = np.random.default_rng(24021)
    candidates = [seed(q)] + [seed(q[rng.choice(len(q), 3, replace=False)]) for _ in range(96)]
    candidates = [p for p in candidates if p is not None]
    if not candidates:
        result["reasons"].append("CURVATURE_UNRESOLVED")
        return result
    p = max(candidates, key=score)
    # Refine a bounded consensus, retaining the original denominator. Never
    # discard bad points before counting the required 75% support.
    for _ in range(15):
        residual = errors(p)
        keep = np.abs(residual) <= min(3., max(.2, p[2]*.1))
        if np.count_nonzero(keep) < 6:
            break
        delta = p[:2]-q[keep]
        distances = np.maximum(np.linalg.norm(delta, axis=1), 1e-9)
        jac = np.column_stack((delta/distances[:, None], -np.ones(len(delta))))
        change, *_ = np.linalg.lstsq(jac, -residual[keep], rcond=None)
        candidate = p+change
        if candidate[2] <= 0 or not np.isfinite(candidate).all():
            break
        if np.sum(errors(candidate)[keep]**2) > np.sum(residual[keep]**2):
            break
        p = candidate
        if np.linalg.norm(change) < 1e-7:
            break
    center, radius = p[:2]+origin, float(p[2])
    residual = np.linalg.norm(points-center, axis=1)-radius
    limit = min(1.5, .05*radius)
    inliers = np.abs(residual) <= limit
    fraction = float(np.mean(inliers))
    result.update(trial_center_mm=center.tolist(), trial_diameter_mm=2*radius,
                  inlier_fraction=fraction, radial_limit_mm=limit,
                  all_point_rms_mm=float(np.sqrt(np.mean(residual**2))),
                  absolute_residual_p50_p95_mm=np.percentile(np.abs(residual), [50, 95]).tolist(),
                  inlier_rms_mm=float(np.sqrt(np.mean(residual[inliers]**2))) if inliers.any() else None)
    if fraction < .75:
        result["reasons"].append("LOW_CIRCLE_CONSENSUS")
    arc, occupied = 0., 0
    if inliers.any():
        offsets = points[inliers]-center
        angles = np.sort(np.arctan2(offsets[:, 1], offsets[:, 0]) % (2*np.pi))
        gaps = np.diff(np.r_[angles, angles[0]+2*np.pi])
        start = angles[(int(np.argmax(gaps))+1) % len(angles)]
        unwrapped = (angles-start) % (2*np.pi)
        span = 2*np.pi-float(gaps.max())
        arc = float(np.degrees(span))
        bins = np.minimum(9, (unwrapped/max(span, 1e-9)*10).astype(int))
        occupied = int(np.count_nonzero(np.bincount(bins, minlength=10) >= 3))
    result.update(arc_degrees=arc, angular_bins_with_three_points=occupied)
    if arc < 70:
        result["reasons"].append("INSUFFICIENT_ARC")
    if occupied < 6:
        result["reasons"].append("UNEVEN_ARC_SUPPORT")
    result["geometry_passed"] = not result["reasons"]
    return result
