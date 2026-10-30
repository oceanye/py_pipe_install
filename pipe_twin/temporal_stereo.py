"""Bounded fixed-camera burst fusion, before cylinder fitting.

Only measured, bidirectionally valid disparities are accumulated. No spatial
hole filling, image warping, CAD dimensions or previous sessions are used.
"""
from __future__ import annotations

import hashlib
import json
import math
import warnings
from datetime import datetime
from typing import Callable

import cv2
import numpy as np

from .stereo_analyzer import StereoDepthResult

REVISION = "static-burst-photometric-consensus-v3"
POLICY = {"max_frames": 24, "min_frames": 3, "max_duration_s": 30,
          "min_support_fraction": .5, "inlier_delta_px": 1., "max_mad_px": .6,
          "motion_median_px": .35, "motion_p95_px": .9,
          "min_tracks": 24, "min_track_tiles": 6, "max_brightness_shift": 12.,
          "max_changed_fraction": .02, "max_tile_difference": 20.,
          "motion_photometric_normalization": "GLOBAL_MEDIAN_OFFSET_BEFORE_LK",
          "require_observed_majority": True, "majority_interval_reseed": True}
POLICY_SHA256 = hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()


def validate_burst_records(group: dict, calibration, manifest_path) -> list[dict]:
    """Validate nested frames with the same contracts as ordinary captures."""
    from .stereo_analyzer import _capture_groups_from_manifest

    frames = group.get("temporal_frames")
    if frames is None:
        return []
    if not isinstance(frames, list) or not POLICY["min_frames"] <= len(frames) <= POLICY["max_frames"]:
        raise ValueError("连续采集必须包含 3～24 对图像")
    payload = {"kind": "stereo_still_capture_set", "camera_layout": "stereo",
               "capture_group_id": "burst", "interval_minutes": 5, "capture_groups": frames}
    _, _, normalized = _capture_groups_from_manifest(manifest_path, payload, calibration)
    if any(not frame["sync_valid"] for frame in normalized):
        raise ValueError("连续采集中存在左右不同步的帧")
    if any(frame["views"]["left"]["sha256"] == frame["views"]["right"]["sha256"] for frame in normalized):
        raise ValueError("连续采集中出现完全相同的左右照片")
    duration = (datetime.fromisoformat(normalized[-1]["captured_at"]) -
                datetime.fromisoformat(normalized[0]["captured_at"])).total_seconds()
    if duration > POLICY["max_duration_s"]:
        raise ValueError("连续采集跨度超过 30 秒，请重新采集短序列")
    if any(normalized[-1]["views"][role][key] != group["views"][role][key]
           for role in ("left", "right") for key in ("sha256", "captured_at")):
        raise ValueError("连续采集末帧与当前预览照片不一致")
    # A change of device, exposure or layout must start a new burst.
    for role in ("left", "right"):
        def settings(view):
            return {k: v for k, v in view.items() if k.startswith("capture_") or k == "side_by_side_order"}
        reference = settings(normalized[-1]["views"][role])
        if settings(group["views"][role]) != reference:
            raise ValueError("连续采集与代表帧的相机设置不一致")
        if any(settings(frame["views"][role]) != reference for frame in normalized):
            raise ValueError("连续采集过程中相机/快门设置改变，请重新采集")
    return normalized


def static_pair_check(reference: np.ndarray, image: np.ndarray) -> dict:
    """Conservative image-motion check; no alignment that could hide motion."""
    if image.shape != reference.shape:
        return {"passed": False, "reason": "BURST_IMAGE_SIZE_CHANGED"}
    a, b = [cv2.cvtColor(value, cv2.COLOR_BGR2GRAY) for value in (reference, image)]
    brightness = float(np.median(b.astype(np.float32) - a))
    if abs(brightness) > POLICY["max_brightness_shift"]:
        return {"passed": False, "reason": "BURST_EXPOSURE_UNSTABLE", "brightness_shift": brightness}
    # LK assumes brightness constancy. Apply the same measured offset already
    # removed by the pixel-difference check to the tracking copy only. Even a
    # stationary textured patch can otherwise acquire spurious displacement.
    # No coordinates are aligned; original images and depth stay unchanged.
    tracking_b = np.rint(np.clip(b.astype(np.float32)-brightness, 0, 255)).astype(np.uint8)
    points = cv2.goodFeaturesToTrack(a, maxCorners=500, qualityLevel=.01, minDistance=12)
    if points is None or len(points) < POLICY["min_tracks"]:
        return {"passed": False, "reason": "BURST_MOTION_UNVERIFIABLE"}
    tracked, status, _ = cv2.calcOpticalFlowPyrLK(a, tracking_b, points, None)
    if tracked is None:
        return {"passed": False, "reason": "BURST_MOTION_UNVERIFIABLE"}
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(tracking_b, a, tracked, None)
    if back is None:
        return {"passed": False, "reason": "BURST_MOTION_UNVERIFIABLE"}
    valid = status.ravel().astype(bool) & back_status.ravel().astype(bool)
    valid &= np.linalg.norm((back-points).reshape(-1, 2), axis=1) < .5
    positions = points.reshape(-1, 2)[valid]
    h, w = a.shape
    tiles = {(min(3, int(x*4/w)), min(2, int(y*3/h))) for x, y in positions}
    if len(positions) < POLICY["min_tracks"] or len(tiles) < POLICY["min_track_tiles"]:
        return {"passed": False, "reason": "BURST_MOTION_UNVERIFIABLE", "track_count": len(positions)}
    displacement = np.linalg.norm((tracked-points).reshape(-1, 2)[valid], axis=1)
    median, p95 = [float(v) for v in np.percentile(displacement, (50, 95))]
    difference = np.abs(cv2.GaussianBlur(b, (5, 5), 0).astype(np.float32) -
                        cv2.GaussianBlur(a, (5, 5), 0).astype(np.float32) - brightness)
    tile_difference = max(float(np.mean(tile)) for row in np.array_split(difference, 6)
                          for tile in np.array_split(row, 8, axis=1) if tile.size)
    changed = float(np.mean(difference > 25))
    passed = (median <= POLICY["motion_median_px"] and p95 <= POLICY["motion_p95_px"]
              and changed <= POLICY["max_changed_fraction"] and tile_difference <= POLICY["max_tile_difference"])
    return {"passed": bool(passed), "reason": None if passed else "BURST_SCENE_MOVED",
            "track_count": len(positions), "track_tiles": len(tiles),
            "median_motion_px": median, "p95_motion_px": p95,
            "brightness_shift": brightness, "changed_fraction": changed,
            "tracking_brightness_offset_removed": brightness,
            "maximum_tile_difference": tile_difference}


def _majority_seed(stack, selected, minimum):
    """Find a bounded majority interval only where the ordinary median fails.

    A one-sided tail of gross mismatches can displace the median to the edge
    of the real cluster. Search the shortest interval containing a strict
    majority of measured frames, not a CAD-derived disparity. Its midpoint
    is only a seed: the output remains the median of agreeing observations.
    """
    ordered = np.sort(stack[:, selected], axis=0)
    needed = minimum[selected]
    columns = np.arange(ordered.shape[1])
    best_width = np.full(ordered.shape[1], np.inf, np.float32)
    seed = np.full(ordered.shape[1], np.nan, np.float32)
    for start in range(len(stack)):
        end = start + needed - 1
        in_bounds = end < len(stack)
        high = ordered[np.minimum(end, len(stack)-1), columns]
        low = ordered[start]
        width = high-low
        better = in_bounds & np.isfinite(width) & (width < best_width)
        best_width[better] = width[better]
        seed[better] = (low[better]+high[better])*.5
    seed[best_width > 2*POLICY["inlier_delta_px"]] = np.nan
    return seed


def _fuse_disparities(samples: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    """Tile the reduction to bound peak memory at full camera resolution."""
    shape = samples[0].shape
    output = np.full(shape, np.nan, np.float32)
    required = max(POLICY["min_frames"], math.ceil(len(samples)*POLICY["min_support_fraction"]))
    counts, spreads = [], []
    recovered = 0
    insufficient = 0
    inconsistent = 0
    for y in range(0, shape[0], 64):
        stack = np.stack([value[y:y+64] for value in samples])
        finite = np.isfinite(stack)
        available = finite.sum(axis=0)
        # The frame quorum is unchanged. Additionally, two equally supported
        # competing layers must never masquerade as one stable surface.
        minimum = np.maximum(required, available//2+1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # unsupported pixels stay NaN
            center = np.nanmedian(stack, axis=0)
            inliers = finite & (np.abs(stack-center) <= POLICY["inlier_delta_px"])
            count = np.sum(inliers, axis=0)
            retry = (available >= minimum) & (count < minimum)
            if np.any(retry):
                center[retry] = _majority_seed(stack, retry, minimum)
                inliers = finite & (np.abs(stack-center) <= POLICY["inlier_delta_px"])
                count = np.sum(inliers, axis=0)
            retained = np.where(inliers, stack, np.nan)
            median = np.nanmedian(retained, axis=0)
            # A reseeded interval may be asymmetric. Require its samples to
            # agree with the final measured median as well, not just the seed.
            inliers &= np.abs(stack-median) <= POLICY["inlier_delta_px"]
            count = np.sum(inliers, axis=0)
            retained = np.where(inliers, stack, np.nan)
            median = np.nanmedian(retained, axis=0)
            mad = np.nanmedian(np.abs(retained-median), axis=0)
        valid = (count >= minimum) & (mad <= POLICY["max_mad_px"])
        recovered += int(np.count_nonzero(valid & retry))
        insufficient += int(np.count_nonzero(available < required))
        inconsistent += int(np.count_nonzero((available >= required) & ~valid))
        output[y:y+64] = np.where(valid, median, np.nan)
        counts.extend(count[valid][::32].tolist())
        spreads.extend(mad[valid][::32].tolist())
    return output, {"required_support_frames": required,
                    "median_support_frames": float(np.median(counts)) if counts else None,
                    "median_disparity_mad_px": float(np.median(spreads)) if spreads else None,
                    "p95_disparity_mad_px": float(np.percentile(spreads, 95)) if spreads else None,
                    "majority_cluster_recovered_pixels": recovered,
                    "insufficient_observation_pixels": insufficient,
                    "inconsistent_observation_pixels": inconsistent,
                    "pre_consistency_valid_fraction": float(np.mean(np.isfinite(output)))}


def fuse_stereo_burst(frames: list[dict], *, load_images: Callable, compute_depth: Callable,
                      quality_check: Callable, calibration, progress: Callable | None = None) -> StereoDepthResult:
    """Stream frames, retaining only disparity arrays; reference is the last pair.

    The caller has hash-checked the manifest, calibration, timestamps and devices.
    A detected scene move invalidates the whole burst; missing quality merely
    excludes a frame. Failure never silently falls back to an earlier frame.
    """
    if not POLICY["min_frames"] <= len(frames) <= POLICY["max_frames"]:
        raise ValueError("连续采集必须包含 3～24 对图像")
    reference = load_images(frames[-1])
    shape = reference["left"].shape[:2]
    audit = {"algorithm_revision": REVISION, "policy": dict(POLICY), "policy_sha256": POLICY_SHA256,
             "validation_scope": "TEMPORAL_REPEATABILITY_NOT_ABSOLUTE_ACCURACY",
             "requested_frames": len(frames), "reference_capture_id": frames[-1]["capture_id"],
             "accepted_frames": 0, "frames": [], "reason_codes": []}
    samples = {role: [] for role in ("left", "right")}
    seen = set()
    last_depth_audit = {}
    fatal = False
    for index, frame in enumerate(frames):
        if progress is not None:
            progress(index+1, len(frames))
        images = reference if index == len(frames)-1 else load_images(frame)
        signature = tuple(hashlib.sha256(images[r].tobytes()).hexdigest() for r in ("left", "right"))
        record = {"capture_id": frame["capture_id"], "accepted": False, "reason_codes": []}
        audit["frames"].append(record)
        quality = {role: quality_check(images[role]) for role in ("left", "right")}
        record["quality"] = quality
        if not all(value["passed"] for value in quality.values()):
            record["reason_codes"].append("BURST_FRAME_QUALITY_FAILED")
            fatal |= index == len(frames)-1
            continue
        motion = {role: static_pair_check(reference[role], images[role]) for role in ("left", "right")}
        record["motion"] = motion
        if not all(value["passed"] for value in motion.values()):
            codes = list(dict.fromkeys(value["reason"] for value in motion.values() if not value["passed"]))
            record["reason_codes"].extend(codes)
            fatal |= "BURST_SCENE_MOVED" in codes or index == len(frames)-1
            continue
        if signature in seen:
            record["reason_codes"].append("BURST_DUPLICATE_FRAME")
            continue
        seen.add(signature)
        depth = compute_depth(images["left"], images["right"])
        last_depth_audit = depth.audit
        record["single_frame_depth_audit"] = depth.audit
        if depth.audit.get("status") != "VALID":
            record["reason_codes"].append("BURST_FRAME_DEPTH_INVALID")
            continue
        for role in ("left", "right"):
            z = getattr(depth, f"{role}_depth_mm")
            valid = getattr(depth, f"{role}_valid") & np.isfinite(z) & (z > 0)
            disparity = np.full(shape, np.nan, np.float32)
            disparity[valid] = getattr(calibration, role).fx * calibration.baseline_mm / z[valid]
            samples[role].append(disparity)
        record["accepted"] = True
        audit["accepted_frames"] += 1
    if fatal:
        audit["reason_codes"].append("BURST_REFERENCE_OR_SCENE_UNSTABLE")
    if audit["accepted_frames"] < POLICY["min_frames"]:
        audit["reason_codes"].append("BURST_TOO_FEW_VALID_FRAMES")
    if audit["reason_codes"]:
        fused = {role: np.full(shape, np.nan, np.float32) for role in samples}
    else:
        fused = {}
        for role in samples:
            fused[role], audit[role] = _fuse_disparities(samples[role])
            samples[role].clear()
        # Recheck the fused maps in both directions; do not combine unrelated
        # medians from the two eyes into a synthetic stereo correspondence.
        masks = {}
        rows, columns = np.indices(shape)
        for role, other, sign in (("left", "right", -1), ("right", "left", 1)):
            disparity = fused[role]
            valid = np.isfinite(disparity)
            target = np.rint(columns + sign*np.nan_to_num(disparity)).astype(np.int32)
            valid &= (target >= 0) & (target < shape[1])
            opposite = np.full(shape, np.nan, np.float32)
            opposite[valid] = fused[other][rows[valid], target[valid]]
            limit = float(last_depth_audit.get("left_right_consistency_px", 1.))
            masks[role] = valid & np.isfinite(opposite) & (np.abs(disparity-opposite) <= limit)
        for role in fused:
            fused[role][~masks[role]] = np.nan
            audit[role]["post_consistency_valid_fraction"] = float(np.mean(masks[role]))
    depths = {}
    for role, disparity in fused.items():
        valid = np.isfinite(disparity) & (disparity > 0)
        z = np.full(shape, np.nan, np.float32)
        z[valid] = getattr(calibration, role).fx*calibration.baseline_mm/disparity[valid]
        depths[role] = z
    valid = {role: np.isfinite(z) for role, z in depths.items()}
    status = "VALID" if all(np.any(v) for v in valid.values()) else "INVALID"
    if status == "INVALID" and not audit["reason_codes"]:
        audit["reason_codes"].append("BURST_NO_CONSISTENT_DEPTH")
    audit["status"] = status
    depth_audit = dict(last_depth_audit)
    depth_audit.update(status=status, reason_codes=audit["reason_codes"], temporal_fusion=audit,
                       valid_left_fraction=float(np.mean(valid["left"])),
                       valid_right_fraction=float(np.mean(valid["right"])))
    return StereoDepthResult(depths["left"], depths["right"], valid["left"], valid["right"], depth_audit)
