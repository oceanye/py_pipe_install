"""Bounded image strip proposals and measured-disparity outlier rejection.

Image proposals are diagnostics, never identities or measurements. No CAD
colour, diameter or pipe count is used. An optional image direction and mask
constrain proposal extraction, without supplying metric or 3-D information.
Only original stereo samples survive the disparity filter; holes stay holes.
"""
from __future__ import annotations

import hashlib
import json
import math

import cv2
import numpy as np

REVISION = "image-strips-overlapping-hue-v4"
POLICY = {"maximum_regions": 24, "minimum_pixels": 120, "minimum_length_px": 48,
          "minimum_aspect": 2.0, "minimum_fill": 0.55,
          "axis_tolerance_deg": 12.0, "disparity_residual_px": 2.0,
          "minimum_disparity_inlier_fraction": 0.6, "minimum_axial_bins": 6,
          "ransac_trials": 96, "maximum_samples": 4096, "chromatic_saturation_min": 120,
          "highlight_gap_max_px": 21, "highlight_gap_width_fraction": 0.5,
          "maximum_local_windows": 3, "local_window_length_fraction": 0.5,
          "local_window_min_length_px": 160, "local_window_width_multiple": 4.0,
          "direction_kernel_odd": True, "clipped_minimum_aspect": 1.3,
          "clipped_boundary_margin_px": 2, "global_axis_filter": False,
          "hue_center_step": 15, "hue_half_width": 15}
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


def propose_regions(image, *, sampling_mask=None, axis_hint_image=None):
    """Separate elongated paint/neutral regions, independently of the model."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be uint8 BGR")
    if sampling_mask is not None:
        sampling_mask = np.asarray(sampling_mask, bool)
        if sampling_mask.shape != image.shape[:2]:
            raise ValueError('sampling mask must match image dimensions')
    if axis_hint_image is not None:
        axis_hint_image = np.asarray(axis_hint_image, float)
        if axis_hint_image.shape != (2,) or not np.isfinite(axis_hint_image).all() or np.linalg.norm(axis_hint_image) < 1e-9:
            raise ValueError('invalid image axis hint')
        axis_hint_image = axis_hint_image/np.linalg.norm(axis_hint_image)
    # Median suppresses narrow highlights for proposal geometry only. The
    # original image and stereo depth are never blurred for measurement.
    hsv = cv2.cvtColor(cv2.medianBlur(image, 5), cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    chromatic = (hsv[:, :, 1] >= POLICY["chromatic_saturation_min"]) & (hsv[:, :, 2] >= 40)
    # Overlap equal-width colour windows instead of joining broad colour bins:
    # a small shade change at H=105 must not split one blue surface. Each
    # window still spans only 30 hue units, and spatial gaps remain gaps.
    # Keep the original centers first for stable labels when masks tie.
    centers = list(range(0, 180, 30)) + list(range(POLICY["hue_center_step"], 180, 30))
    hue_values = hsv[:, :, 0].astype(np.int16)
    masks = [(f"hue-{hue}", chromatic & (np.minimum(abs(hue_values-hue),
                  180-abs(hue_values-hue)) <= POLICY["hue_half_width"])) for hue in centers]
    masks.append(("neutral", (hsv[:, :, 1] < 65) & (gray >= max(100, threshold))))
    proposals = []
    component_limit_hit = False
    for kind, raw in masks:
        if sampling_mask is not None:
            raw &= sampling_mask
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
        if axis_hint_image is not None:
            angle = math.atan2(axis_hint_image[1], axis_hint_image[0])
        # A centred odd kernel avoids a one-pixel shift of the paint mask.
        # Such a shift leaves a false dark seam alongside a real highlight.
        length = max(21, min(81, int(max(image.shape[:2]) * 0.08))) | 1
        kernel = np.zeros((length, length), np.uint8)
        center = (length-1)/2
        dx, dy = math.cos(angle)*center, math.sin(angle)*center
        cv2.line(kernel, (round(center-dx), round(center-dy)),
                 (round(center+dx), round(center+dy)), 1, 1)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0)
        if sampling_mask is not None:
            opened &= sampling_mask.astype(np.uint8)
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
        # A fixed 9px kernel splits wider pipes at a broad highlight. Scale
        # the bridge to the thinner paint fragment, with a hard pixel cap;
        # still only bridge bright neutral pixels, never a dark pipe gap.
        gap = min(POLICY["highlight_gap_max_px"], max(9, int(min(
            p["width_px"] for p in members) * POLICY["highlight_gap_width_fraction"])))
        gap += 1 - gap % 2
        joined = cv2.morphologyEx(combined,cv2.MORPH_CLOSE,np.ones((gap,gap),np.uint8))
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
    elongated = []
    for p in proposals:
        x,y,w,h = p["region_px"]
        ax,ay = p["axis_image"]
        edge = POLICY["clipped_boundary_margin_px"]
        clipped = ((abs(ax) >= .9 and x <= edge and x+w >= image.shape[1]-edge)
                   or (abs(ay) >= .9 and y <= edge and y+h >= image.shape[0]-edge))
        p["axially_clipped"] = bool(clipped)
        contacts = []
        if x <= edge:
            contacts.append("left")
        if y <= edge:
            contacts.append("top")
        if x+w >= image.shape[1]-edge:
            contacts.append("right")
        if y+h >= image.shape[0]-edge:
            contacts.append("bottom")
        p["boundary_contacts"] = contacts
        transverse = {"top", "bottom"} if abs(ax) >= .9 else {"left", "right"} if abs(ay) >= .9 else set()
        p["transverse_boundary_contacts"] = [side for side in contacts if side in transverse]
        p["axis_to_epipolar_deg"] = math.degrees(math.atan2(abs(ay), abs(ax)))
        # A user crop can truncate both ends of a thick pipe. Retain its
        # image proposal; the unchanged metric length gate still applies.
        minimum_aspect = POLICY["clipped_minimum_aspect"] if clipped else POLICY["minimum_aspect"]
        if p["length_px"] / p["width_px"] >= minimum_aspect:
            elongated.append(p)
    proposals = elongated
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
    # Direction votes are diagnostics only. Numerous short carpet/stand
    # fragments can outvote the real pipes differently in the two eyes.
    # Preserve alternate directions under the existing score/budget cap;
    # measured cylinder axes and stereo pairing decide geometric consistency.
    if len(unique) >= 2:
        axes = np.asarray([p["axis_image"] for p in unique])
        agreement = np.abs(axes @ axes.T) >= math.cos(math.radians(POLICY["axis_tolerance_deg"]))
        counts = agreement.sum(axis=1)
        for p, count in zip(unique, counts):
            p["axis_consensus_count"] = int(count)
    truncated = component_limit_hit or len(unique) > POLICY["maximum_regions"]
    unique = unique[:POLICY["maximum_regions"]]
    unique.sort(key=lambda p: (p["region_px"][1], p["region_px"][0]))
    return unique, {"revision":REVISION, "policy_sha256":POLICY_SHA256,
                    "scope":"IMAGE_CANDIDATES_NOT_PIPE_IDENTITY_OR_MEASUREMENT",
                    "global_axis_filter_applied": False,
                    "truncated":truncated, "candidate_count":len(unique)}


def axial_windows(region):
    """Bounded overlapping masks for independently measured local segments.

    The image width sets a minimum useful segment length, never a CAD radius.
    Windows only restrict the original proposal; depth holes stay holes.
    """
    mask = region["mask"].astype(bool)
    rows, cols = np.nonzero(mask)
    if len(rows) < POLICY["minimum_pixels"]:
        return []
    axis = np.asarray(region["axis_image"], dtype=float)
    if axis[int(np.argmax(np.abs(axis)))] < 0:
        axis = -axis
    station = cols * axis[0] + rows * axis[1]
    low, high = float(station.min()), float(station.max())
    span = high - low
    length = max(POLICY["local_window_min_length_px"],
                 POLICY["local_window_width_multiple"] * region["width_px"],
                 POLICY["local_window_length_fraction"] * span)
    if length >= .8 * span:
        return []
    result = []
    for index, start in enumerate(np.linspace(low, high-length, POLICY["maximum_local_windows"])):
        selected = (station >= start) & (station <= start+length)
        local = np.zeros(mask.shape, np.uint8)
        local[rows[selected], cols[selected]] = 1
        result.append({"window_id": f"W{index+1:02d}", "mask": local,
                       "axial_interval_px": [float(start),float(start+length)]})
    return result


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
