"""Compare saved stereo processing without opening a camera or changing sources."""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipe_twin.calibration_wizard import align_pair_orientation, detect_board_corners
from pipe_twin.elevation_dataset import load_elevation_dataset
from pipe_twin.photo_capture import load_photo_snapshot
from pipe_twin.stereo_analyzer import (
    STEREO_ALGORITHM_REVISION, _analysis_config, _calibration_from_manifest,
    _compute_stereo_depth, _quality,
)


def board_reference(images, pattern):
    """Use sparse matched board corners as a local reference, never as depth input."""
    observations = [detect_board_corners(images[role], pattern=pattern) for role in ("left", "right")]
    if any(item is None for item in observations):
        return None, {"detected": False}
    observations[1] = align_pair_orientation(*observations, pattern=pattern)
    left, right = [np.asarray(item.corners_px).reshape(-1, 2) for item in observations]
    disparity = left[:, 0] - right[:, 0]
    design = np.c_[left, np.ones(len(left))]
    coef = np.linalg.lstsq(design, disparity, rcond=None)[0]
    residual = float(np.sqrt(np.mean((design @ coef - disparity) ** 2)))
    vertical = float(np.percentile(np.abs(left[:, 1] - right[:, 1]), 95))
    audit = {"detected": True, "corner_disparity_median_px": float(np.median(disparity)),
             "plane_fit_rms_px": residual, "vertical_residual_p95_px": vertical,
             "reference_scope": "sparse corners in one board region; not independent depth ground truth"}
    # A badly aligned board cannot be used to score dense disparity.
    if np.any(disparity <= 0) or residual > 1.0 or vertical > 3.0:
        audit["usable"] = False
        return None, audit
    yy, xx = np.indices(images["left"].shape[:2])
    mask = np.zeros(xx.shape, np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(left.astype(np.int32)), 1)
    mask = cv2.erode(mask, np.ones((11, 11), np.uint8)).astype(bool)
    audit["usable"] = bool(mask.any())
    return (mask, coef[0] * xx + coef[1] * yy + coef[2]), audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new local output directory")
    parser.add_argument("--num-disparities", type=int, default=256)
    parser.add_argument("--board", type=int, nargs=2, metavar=("COLUMNS", "ROWS"))
    args = parser.parse_args()
    if args.board and any(value < 3 or value > 30 for value in args.board):
        parser.error("board dimensions must be between 3 and 30 inner corners")
    path = args.manifest.resolve()
    loaded = load_elevation_dataset(path)
    manifest = loaded["manifest"]
    calibration = _calibration_from_manifest(loaded["calibration"])
    saved = _analysis_config(manifest["analysis"])
    variants = {"saved": saved}
    for name, preprocessing in (("wide", "none"), ("low_light", "low_light")):
        config = copy.deepcopy(saved)
        config["stereo_matching"].update(num_disparities=args.num_disparities, preprocessing=preprocessing)
        variants[name] = _analysis_config(config)
    # Never overwrite a previous report, source photo or manifest.
    args.output.mkdir(parents=True, exist_ok=False)
    audit = {"algorithm_revision": STEREO_ALGORITHM_REVISION, "opencv_version": cv2.__version__,
             "source_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
             "variants": variants, "groups": [], "production_authority": False}
    sections = []
    for index, group in enumerate(manifest["capture"]["capture_groups"]):
        images, integrity = {}, {}
        for role in ("left", "right"):
            images[role], integrity[role] = load_photo_snapshot(path, group["views"][role])
        reference, board = board_reference(images, tuple(args.board)) if args.board else (None, {"requested": False})
        row = {"capture_id": group["capture_id"], "photos": integrity, "board": board,
               "quality": {r: _quality(im, saved) for r, im in images.items()}, "results": []}
        source_name = f"{index:03d}-source.jpg"
        cv2.imencode(".jpg", images["left"])[1].tofile(args.output / source_name)
        sections.append(f'<section><h2>{html.escape(group["capture_id"])}</h2><img src="{source_name}" alt="原始左目">')
        for name, config in variants.items():
            start = time.perf_counter()
            depth = _compute_stereo_depth(images["left"], images["right"], calibration, config)
            result = {"variant": name, "elapsed_seconds": time.perf_counter() - start, "depth_audit": depth.audit, "regions": []}
            for spec in loaded["pipe_specs"]:
                region = {"pipe_id": spec["pipe_id"]}
                for role in ("left", "right"):
                    box = spec.get(role + "_region_px")
                    if box:
                        x, y, w, h = box
                        region[role] = {"valid_depth_fraction": float(getattr(depth, role + "_valid")[y:y+h, x:x+w].mean()),
                                        "quality": _quality(images[role][y:y+h, x:x+w], saved)}
                result["regions"].append(region)
            if reference is not None:
                mask, expected = reference
                valid = mask & depth.left_valid
                errors = np.abs(calibration.left.fx * calibration.baseline_mm / depth.left_depth_mm[valid] - expected[valid])
                result["board"] = {"coverage": float(depth.left_valid[mask].mean()),
                    "disparity_error_p50_p90_px": np.percentile(errors, [50, 90]).tolist() if errors.size else None,
                    "bad_over_3px_fraction": float(np.mean(errors > 3)) if errors.size else None,
                    "warning": "DENSE_BOARD_MISMATCH" if errors.size and np.mean(errors > 3) > .1 else "NO_DENSE_REFERENCE" if not errors.size else None}
            preview = cv2.applyColorMap(np.nan_to_num(np.clip((depth.left_depth_mm - 600) / 1800 * 255, 0, 255)).astype(np.uint8), cv2.COLORMAP_TURBO)
            preview[~depth.left_valid] = 0
            name_jpg = f"{index:03d}-{name}.jpg"
            cv2.imencode(".jpg", preview)[1].tofile(args.output / name_jpg)
            sections.append(f'<h3>{name}</h3><img src="{name_jpg}" alt="{name}"><pre>{html.escape(json.dumps(result, ensure_ascii=False, indent=2))}</pre>')
            row["results"].append(result)
            print(json.dumps({"frame": index, "variant": name, "coverage": depth.audit["valid_left_fraction"], "board": result.get("board")}), flush=True)
        audit["groups"].append(row)
        sections.append("</section>")
    (args.output / "comparison.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    report = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>双目弱光对比</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:32px auto;background:#f5f7fa}section{background:white;padding:20px;margin:16px 0}img{max-width:100%}pre{white-space:pre-wrap;font-size:13px}</style><h1>双目弱光离线对比</h1><p>黑色为无有效深度；颜色范围 0.6–2.4 m。覆盖率增加不等于精度提高。棋盘参考来自当前角点，只核对局部；重复纹理仍可能错配，不能作为验收或新的完整标定。</p>' + ''.join(sections) + '</html>'
    (args.output / "report.html").write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
