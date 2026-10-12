"""Bounded image strip proposals and measured-disparity outlier rejection.

Image proposals are diagnostics, never identities or measurements. No CAD
colour, diameter, pipe count, or manually supplied image direction is used.
Only original stereo samples survive the disparity filter; holes stay holes.
"""
from __future__ import annotations

import hashlib
import json
import math

import cv2
import numpy as np

REVISION = "image-strips-disparity-consensus-v1"
POLICY = {"maximum_regions": 24, "minimum_pixels": 120, "minimum_length_px": 48,
          "minimum_aspect": 2.0, "minimum_fill": 0.55,
          "axis_tolerance_deg": 12.0, "disparity_residual_px": 2.0,
          "minimum_disparity_inlier_fraction": 0.6, "minimum_axial_bins": 6,
          "ransac_trials": 96, "maximum_samples": 4096, "chromatic_saturation_min": 120}
POLICY_SHA256 = hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()


def _axis(mask):
    rows, cols = np.nonzero(mask)
    pts = np.column_stack((cols, rows)).astype(np.float32)
    rect = cv2.minAreaRect(pts)
    (_, _), (w, h), angle = rect
    if h > w:
        angle += 90
    angle = math.radians(angle)
    return np.asarray([math.cos(angle), math.sin(angle)]), max(w, h), min(w, h)


def propose_regions(image):
    """Separate elongated paint/neutral regions, independently of the model."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be uint8 BGR")
    # Median suppresses narrow highlights for proposal geometry only. The
    # original image and stereo depth are never blurred for measurement.
    hsv = cv2.cvtColor(cv2.medianBlur(image, 5), cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    chromatic = (hsv[:, :, 1] >= POLICY["chromatic_saturation_min"]) & (hsv[:, :, 2] >= 40)
    masks = [(f"hue-{hue}", chromatic & (np.minimum(abs(hsv[:, :, 0].astype(float)-hue),
                                                     180-abs(hsv[:, :, 0].astype(float)-hue)) <= 15))
             for hue in range(0, 180, 30)]
    masks.append(("neutral", (hsv[:, :, 1] < 65) & (gray >= max(100, threshold))))
    proposals = []
    component_limit_hit = False
    for kind, raw in masks:
        mask = cv2.morphologyEx(raw.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # Long edge consensus finds a direction even if two pipes touch via a
        # short transverse connector. Weight by capped length, not one blob.
        lines = cv2.createLineSegmentDetector().detect(mask * 255)[0]
        if lines is None:
            continue
        lines = lines.reshape(-1, 4)
        delta = lines[:, 2:] - lines[:, :2]
        lengths = np.linalg.norm(delta, axis=1)
        good = lengths >= POLICY["minimum_length_px"]
        if not np.any(good):
            continue
        angles = np.arctan2(delta[good, 1], delta[good, 0]) % np.pi
        weights = np.minimum(lengths[good], 300)
        ranked = np.argsort(-lengths[good],kind="stable")[:256]
        angles, weights = angles[ranked], weights[ranked]
        scores = []
        for a in angles:
            error = np.abs(np.sin(angles-a))
            scores.append(float(weights[error < math.sin(math.radians(8))].sum()))
        angle = float(angles[int(np.argmax(scores))])
        length = max(21, min(81, int(max(image.shape[:2]) * 0.08)))
        kernel = np.zeros((length, length), np.uint8)
        center = (length-1)/2
        dx, dy = math.cos(angle)*center, math.sin(angle)*center
        cv2.line(kernel, (round(center-dx), round(center-dy)),
                 (round(center+dx), round(center+dy)), 1, 1)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(opened, 8)
        # Work on bounded largest components. Explicitly account for overflow.
        eligible = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= POLICY["minimum_pixels"]]
        eligible.sort(key=lambda i: (-int(stats[i, cv2.CC_STAT_AREA]), i))
        component_limit_hit |= len(eligible) > 48
        for i in eligible[:48]:
            x, y, w, h, area = map(int, stats[i])
            local = (labels[y:y+h, x:x+w] == i).astype(np.uint8)
            axis, major, minor = _axis(local)
            if major < POLICY["minimum_length_px"] or minor < 4 or major / minor < 1.3:
                continue
            if area / max(1, (major+1)*(minor+1)) < POLICY["minimum_fill"]:
                continue
            # Fill image holes only, to keep white highlights inside a paint
            # proposal. Missing depth is still missing later.
            contours, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(local, contours, -1, 1, cv2.FILLED)
            rgb = np.median(image[y:y+h, x:x+w][local.astype(bool)], axis=0)[::-1].round().astype(int)
            proposals.append({"region_px": [x,y,w,h], "axis_image": axis.tolist(),
                              "length_px": float(major), "width_px": float(minor),
                              "appearance": kind, "sampled_color_srgb": "#%02X%02X%02X" % tuple(rgb),
                              "mask": local, "score": float(major * math.sqrt(minor))})
    # Join two paint strips only across a bright, neutral highlight. A dark
    # gap between same-colour pipes must remain a gap. Use actual mask pixels,
    # not bounding-box overlap, so slanted strips remain well separated.
    for kind in sorted({p["appearance"] for p in proposals if p["appearance"] != "neutral"}):
        members = [p for p in proposals if p["appearance"] == kind]
        if len(members) < 2:
            continue
        combined = np.zeros(image.shape[:2],np.uint8)
        for p in members:
            x,y,w,h = p["region_px"]
            combined[y:y+h,x:x+w] |= p["mask"]
        joined = cv2.morphologyEx(combined,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
        bright = (hsv[:,:,1] < 100) & (hsv[:,:,2] >= 180)
        joined = combined | ((joined > 0) & bright).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(joined,8)
        merged = []
        for i in range(1,count):
            x,y,w,h,area = map(int,stats[i])
            local = (labels[y:y+h,x:x+w] == i).astype(np.uint8)
            axis,major,minor = _axis(local)
            if major < POLICY["minimum_length_px"] or minor < 4:
                continue
            rgb = np.median(image[y:y+h,x:x+w][local.astype(bool)],axis=0)[::-1].round().astype(int)
            merged.append({"region_px":[x,y,w,h],"axis_image":axis.tolist(),"length_px":float(major),
                           "width_px":float(minor),"appearance":kind,
                           "sampled_color_srgb":"#%02X%02X%02X" % tuple(rgb),"mask":local,
                           "score":float(major*math.sqrt(minor))})
        proposals = [p for p in proposals if p["appearance"] != kind] + merged
    proposals = [p for p in proposals if p["length_px"] / p["width_px"] >= POLICY["minimum_aspect"]]
    proposals.sort(key=lambda p: -p["score"])
    unique = []
    for p in proposals:
        x,y,w,h = p["region_px"]
        duplicate = False
        for q in unique:
            a,b,c,d = q["region_px"]
            intersection = max(0,min(x+w,a+c)-max(x,a))*max(0,min(y+h,b+d)-max(y,b))
            if intersection / max(1,min(w*h,c*d)) > .7:
                duplicate = True
                break
        if not duplicate:
            unique.append(p)
    # Multiple independent strips vote for an axis. Keep ties visible instead
    # of promoting a single very long clutter region to the common direction.
    if len(unique) >= 2:
        axes = np.asarray([p["axis_image"] for p in unique])
        agreement = np.abs(axes @ axes.T) >= math.cos(math.radians(POLICY["axis_tolerance_deg"]))
        counts = agreement.sum(axis=1)
        if max(counts) >= 2:
            winners = np.flatnonzero(counts == max(counts))
            keep = agreement[winners].any(axis=0)
            unique = [p for p,k in zip(unique,keep) if k]
    truncated = component_limit_hit or len(unique) > POLICY["maximum_regions"]
    unique = unique[:POLICY["maximum_regions"]]
    unique.sort(key=lambda p: (p["region_px"][1], p["region_px"][0]))
    return unique, {"revision":REVISION, "policy_sha256":POLICY_SHA256,
                    "scope":"IMAGE_CANDIDATES_NOT_PIPE_IDENTITY_OR_MEASUREMENT",
                    "truncated":truncated, "candidate_count":len(unique)}


def filter_disparity(depth, support, camera, baseline_mm, axis_image):
    """Reject off-surface disparities with a bounded, deterministic consensus.

    A coarse plane only selects measured pixels; it is never used to fill
    depth or supply cylinder geometry. Curvature still needs an independent
    strict cylinder fit and an agreeing fit in the other eye.
    """
    rows, cols = np.nonzero(support & np.isfinite(depth) & (depth > 0))
    keep = np.zeros(depth.shape, bool)
    n = len(rows)
    audit = {"input_points":n, "retained_points":0, "reason":"INSUFFICIENT_VALID_DEPTH"}
    if n < 80:
        return keep, audit
    pixels = np.column_stack((cols,rows)).astype(float)
    origin = pixels.mean(axis=0)
    scale = max(1.,float(np.ptp(pixels,axis=0).max()))
    xy = (pixels-origin)/scale
    design = np.column_stack((xy,np.ones(n)))
    disparities = float(camera.fx)*float(baseline_mm)/depth[rows,cols]
    indexes = np.linspace(0,n-1,min(n,POLICY["maximum_samples"]),dtype=int)
    sample, values = design[indexes], disparities[indexes]
    rng = np.random.default_rng(0)
    best, best_count = None, 0
    for _ in range(POLICY["ransac_trials"]):
        selected = rng.choice(len(indexes),3,replace=False)
        matrix = sample[selected]
        if abs(np.linalg.det(matrix)) < 1e-5:
            continue
        params = np.linalg.solve(matrix,values[selected])
        inliers = np.abs(sample @ params-values) <= POLICY["disparity_residual_px"]
        count = int(inliers.sum())
        if count > best_count:
            best, best_count = params, count
    if best is None:
        return keep, audit
    # Refine once, on consensus samples only; never replace measured depths.
    inliers = np.abs(sample @ best-values) <= POLICY["disparity_residual_px"]
    best = np.linalg.lstsq(sample[inliers],values[inliers],rcond=None)[0]
    accepted = np.abs(design @ best-disparities) <= POLICY["disparity_residual_px"]
    fraction = float(accepted.mean())
    axial = pixels @ np.asarray(axis_image)
    bins = np.minimum(7,((axial-axial.min()) / max(1.,float(np.ptp(axial))) * 8).astype(int))
    covered = sum(np.count_nonzero(accepted & (bins == i)) >= 8 for i in range(8))
    audit.update(retained_points=int(accepted.sum()), retained_fraction=fraction,
                 axial_bins_with_support=covered, maximum_disparity_residual_px=POLICY["disparity_residual_px"])
    if fraction < POLICY["minimum_disparity_inlier_fraction"]:
        audit["reason"] = "NO_DOMINANT_DISPARITY_SUPPORT"
    elif covered < POLICY["minimum_axial_bins"]:
        audit["reason"] = "DISCONTINUOUS_AXIAL_DEPTH_SUPPORT"
    else:
        audit["reason"] = None
        keep[rows[accepted],cols[accepted]] = True
    return keep, audit
