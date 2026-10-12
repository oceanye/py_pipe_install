"""Independent image holdout checks; pixel fit cannot certify metric scale."""
from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np

REVISION = "stereo-holdout-quality-v1"
POLICY = {"minimum_holdout_pairs": 3, "holdout_fraction": .2,
          "max_vertical_rms_px": .5, "max_vertical_p95_px": 1.,
          "max_cross_reprojection_rms_px": 1., "max_pair_cross_rms_px": 1.5}
POLICY_SHA256 = hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()


def holdout_quality(pairs, *, recipe: dict, pattern, square_mm: float, image_size) -> dict:
    """Fit only each board's left pose; predict its unseen right image.

    Intrinsics, distortion and the stereo transform are never refitted here.
    The physical board scale is shared with training and thus not independent.
    """
    from .calibration_wizard import _object_points, align_pair_orientation, transform_board_observation
    matrices = {key: np.asarray(recipe[key], np.float64) for key in
                ("K1", "D1", "K2", "D2", "R1", "R2", "P1", "P2")}
    k1, d1, k2, d2 = [matrices[k] for k in ("K1", "D1", "K2", "D2")]
    r1, r2, p1, p2 = [matrices[k] for k in ("R1", "R2", "P1", "P2")]
    rotation = r2.T @ r1
    translation = r2.T @ (np.linalg.solve(p2[:, :3], p2[:, 3]) -
                           np.linalg.solve(p1[:, :3], p1[:, 3]))
    objects = _object_points(pattern, square_mm).astype(np.float64)
    records, vertical, cross = [], [], []
    for index, (left, right) in enumerate(pairs):
        record = {"holdout_index": index+1, "passed": False}
        records.append(record)
        try:
            right = transform_board_observation(right, transform=recipe.get("right_frame_transform", "none"),
                                                image_size=image_size, pattern=pattern)
            right = align_pair_orientation(left, right, pattern=pattern)
            lp, rp = [np.asarray(o.corners_px, np.float64).reshape(-1, 1, 2) for o in (left, right)]
            if not np.isfinite(lp).all() or not np.isfinite(rp).all():
                raise ValueError("nonfinite corners")
            for points in (lp.reshape(-1, 2), rp.reshape(-1, 2)):
                if len(points) != len(objects) or np.any(points < 0) or np.any(points[:, 0] >= image_size[0]) or np.any(points[:, 1] >= image_size[1]):
                    raise ValueError("holdout board outside image or wrong corner count")
            ok, rv, tv = cv2.solvePnP(objects, lp, k1, d1)
            if not ok:
                raise ValueError("left pose failed")
            left_rotation = cv2.Rodrigues(rv)[0]
            right_rotation = rotation @ left_rotation
            right_translation = rotation @ tv.reshape(3) + translation
            points_left = objects @ left_rotation.T + tv.reshape(3)
            points_right = objects @ right_rotation.T + right_translation
            if np.any(points_left[:, 2] <= 0) or np.any(points_right[:, 2] <= 0):
                raise ValueError("board behind camera")
            predicted = cv2.projectPoints(objects, cv2.Rodrigues(right_rotation)[0], right_translation, k2, d2)[0]
            cross_error = np.linalg.norm((predicted-rp).reshape(-1, 2), axis=1)
            ul = cv2.undistortPoints(lp, k1, d1, R=r1, P=p1).reshape(-1, 2)
            ur = cv2.undistortPoints(rp, k2, d2, R=r2, P=p2).reshape(-1, 2)
            dy = ul[:, 1]-ur[:, 1]
            rms = float(np.sqrt(np.mean(cross_error**2)))
            yrms = float(np.sqrt(np.mean(dy**2)))
            yp95 = float(np.percentile(np.abs(dy), 95))
            positive = float(np.mean(ul[:, 0] > ur[:, 0]))
            passed = (rms <= POLICY["max_pair_cross_rms_px"] and yrms <= POLICY["max_vertical_rms_px"]
                      and yp95 <= POLICY["max_vertical_p95_px"] and positive >= .98)
            record.update(passed=bool(passed), cross_reprojection_rms_px=rms,
                          vertical_rms_px=yrms, vertical_p95_px=yp95, positive_disparity_fraction=positive)
            vertical.extend(dy.tolist())
            cross.extend(cross_error.tolist())
        except (cv2.error, ValueError) as error:
            record["error"] = str(error)
    rms = float(np.sqrt(np.mean(np.square(cross)))) if cross else None
    return {"algorithm_revision": REVISION, "policy": dict(POLICY), "policy_sha256": POLICY_SHA256,
            "status": "PASSED" if len(records) >= POLICY["minimum_holdout_pairs"] and
            all(r["passed"] for r in records) and rms is not None and rms <= POLICY["max_cross_reprojection_rms_px"] else "FAILED",
            "pair_count": len(records), "pairs": records, "cross_reprojection_rms_px": rms,
            "vertical_rms_px": float(np.sqrt(np.mean(np.square(vertical)))) if vertical else None,
            "vertical_p95_px": float(np.percentile(np.abs(vertical), 95)) if vertical else None,
            "scale_verified_independently": False,
            "validation_scope": "HELD_OUT_IMAGES_SAME_BOARD_SCALE_NOT_ABSOLUTE_ACCURACY"}


def calibration_quality_text(calibration: dict) -> str:
    audit = calibration.get("quality_audit") or {}
    holdout = audit.get("holdout") or {}
    if not audit:
        return "旧标定未附质量审计；尺寸尺度尚需实物复核，建议重新标定"
    square = audit.get("square_mm")
    label = f"实测格长 {square:g} mm；" if isinstance(square, (float, int)) else ""
    if audit.get("solve_mode") == "baseline_anchored_centered_pinhole":
        return label + "简化零畸变模型，仅供流程诊断；请补拍多姿态重新标定"
    if holdout.get("status") == "PASSED":
        return label + f"留出验证通过（垂直 P95 {holdout['vertical_p95_px']:.3f} px）；毫米尺度待独立实物复核"
    return label + "未通过独立留出验证；尺寸尺度尚需实物复核"
