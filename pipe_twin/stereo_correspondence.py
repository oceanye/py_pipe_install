"""Bounded, read-only image checks of already computed stereo samples.

These diagnostics neither create depth nor certify calibration or measurement.
Repeated/flat patches can agree in both SGBM directions without being unique.
"""
from __future__ import annotations

import hashlib
import json

import cv2
import numpy as np

REVISION = "sampled-stereo-patch-check-v1"
POLICY = {"maximum_samples": 512, "patch_half_width": 5, "patch_half_height": 4,
          "minimum_horizontal_texture": 1.5, "minimum_correlation": .8,
          "minimum_uniqueness_margin": .05,
          "alternative_offsets_px": [-32, -16, -8, -4, 4, 8, 16, 32],
          "minimum_samples": 32, "highlight_channel_min": 250}
POLICY_SHA256 = hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest()


def patch_correspondence(image, peer_image, depth, valid, camera, peer_camera,
                         baseline_mm, role, region):
    """Probe measured correspondences in either eye, including cropped views.

    ZNCC is invariant to local gain/offset, but not to view-dependent reflection.
    Alternative matches test ambiguity, not the existence of a true surface.
    Horizontal texture is used because rectified disparity is horizontal.
    """
    if role not in {"left", "right"}:
        raise ValueError("role must be left or right")
    x, y, w, h = region["region_px"]
    mask = region["mask"].astype(bool)
    patch = image[y:y+h, x:x+w]
    highlights = np.count_nonzero(patch >= POLICY["highlight_channel_min"], axis=2) >= 2
    eligible = mask & valid[y:y+h, x:x+w] & np.isfinite(depth[y:y+h, x:x+w]) & (depth[y:y+h, x:x+w] > 0)
    rows, cols = np.nonzero(eligible)
    audit = {"revision": REVISION, "policy_sha256": POLICY_SHA256,
             "scope": "SAMPLED_IMAGE_DIAGNOSTIC_NOT_METRIC_AUTHORITY",
             "measurement_authority": False, "eligible_depth_points": len(rows),
             "sample_count": 0, "highlight_fraction": float(highlights[mask].mean()) if mask.any() else 0.,
             "warning_codes": []}
    if not len(rows):
        return audit | {"status": "INSUFFICIENT_SAMPLES"}
    indexes = np.linspace(0, len(rows)-1, min(len(rows), POLICY["maximum_samples"]), dtype=int)
    xx, yy = cols[indexes]+x, rows[indexes]+y
    z = depth[yy, xx]
    # Backproject in this camera, translate by the signed baseline, project
    # in the peer. Principal-point differences matter for asymmetric ROIs.
    k, peer_k = camera.rectified_intrinsic, peer_camera.rectified_intrinsic
    peer_x = (xx-k[0,2]) / k[0,0] * peer_k[0,0] + peer_k[0,2]
    peer_x += (-1 if role == "left" else 1) * peer_k[0,0] * baseline_mm / z
    peer_y = (yy-k[1,2]) / k[1,1] * peer_k[1,1] + peer_k[1,2]
    rx, ry = POLICY["patch_half_width"], POLICY["patch_half_height"]
    inside = ((xx >= rx) & (xx < image.shape[1]-rx) & (yy >= ry) & (yy < image.shape[0]-ry)
              & (peer_x >= rx) & (peer_x < peer_image.shape[1]-rx-1)
              & (peer_y >= ry) & (peer_y < peer_image.shape[0]-ry-1))
    xx, yy, peer_x, peer_y = xx[inside], yy[inside], peer_x[inside], peer_y[inside]
    audit["sample_count"] = len(xx)
    if len(xx) < POLICY["minimum_samples"]:
        return audit | {"status": "INSUFFICIENT_SAMPLES"}
    offsets_y, offsets_x = np.mgrid[-ry:ry+1, -rx:rx+1]
    offsets_x, offsets_y = offsets_x.reshape(1,-1), offsets_y.reshape(1,-1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    peer_gray = cv2.cvtColor(peer_image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def samples(source, u, v):
        return cv2.remap(source, (u[:,None]+offsets_x).astype(np.float32),
                         (v[:,None]+offsets_y).astype(np.float32), cv2.INTER_LINEAR)

    def normalized(values):
        values = values-values.mean(axis=1,keepdims=True)
        return values / np.maximum(np.linalg.norm(values,axis=1,keepdims=True), 1e-6)

    a, b = samples(gray,xx,yy), samples(peer_gray,peer_x,peer_y)
    aa = normalized(a)
    correlation = np.sum(aa*normalized(b), axis=1)
    alternatives = np.full(len(xx), -1., dtype=np.float32)
    alternative_count = np.zeros(len(xx), dtype=int)
    for offset in POLICY["alternative_offsets_px"]:
        u = peer_x+offset
        available = (u >= rx) & (u < peer_image.shape[1]-rx-1)
        score = np.sum(aa*normalized(samples(peer_gray,u,peer_y)), axis=1)
        alternatives[available] = np.maximum(alternatives[available],score[available])
        alternative_count += available
    texture = np.minimum(np.std(a.reshape(-1,2*ry+1,2*rx+1),axis=2).mean(axis=1),
                         np.std(b.reshape(-1,2*ry+1,2*rx+1),axis=2).mean(axis=1))
    margin = correlation-alternatives
    tested = alternative_count > 0
    if tested.sum() < POLICY["minimum_samples"]:
        return audit | {"status": "INSUFFICIENT_SAMPLES"}
    textured = texture >= POLICY["minimum_horizontal_texture"]
    similar = correlation >= POLICY["minimum_correlation"]
    unique = margin >= POLICY["minimum_uniqueness_margin"]
    warnings = []
    if np.mean(~textured[tested]) > .5:
        warnings.append("LOW_HORIZONTAL_TEXTURE")
    if np.mean(~similar[tested]) > .5:
        warnings.append("LOW_PATCH_CORRELATION")
    if np.mean(~unique[tested]) > .5:
        warnings.append("AMBIGUOUS_PATCH_CORRESPONDENCE")
    return audit | {"status": "DIAGNOSTIC_ONLY", "comparison_count": int(tested.sum()),
                    "horizontal_texture_median": float(np.median(texture[tested])),
                    "low_texture_fraction": float(np.mean(~textured[tested])),
                    "correlation_median": float(np.median(correlation[tested])),
                    "uniqueness_margin_median": float(np.median(margin[tested])),
                    "sampled_unique_match_fraction": float(np.mean((textured & similar & unique)[tested])),
                    "warning_codes": warnings}
