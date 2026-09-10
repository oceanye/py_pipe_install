"""Replay a chessboard calibration diagnostic NPZ without a camera."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipe_twin.calibration_wizard import replay_calibration_diagnostic


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("diagnostic", type=Path)
    parser.add_argument(
        "--expected-baseline-mm",
        type=float,
        help="override the saved measured lens-centre baseline",
    )
    parser.add_argument(
        "--max-rms-px", type=float,
        help="explicit RMS threshold override; otherwise reuse saved settings",
    )
    parser.add_argument("--output", type=Path, help="save an audit report JSON (not a validated calibration)")
    args = parser.parse_args()
    result = replay_calibration_diagnostic(
        args.diagnostic,
        expected_baseline_mm=args.expected_baseline_mm,
        max_reprojection_rms_px=args.max_rms_px,
    )
    summary = {
        "source": str(args.diagnostic.resolve()),
        "solve_options": {
            key: result.audit[key]
            for key in ("expected_baseline_mm", "max_reprojection_rms_px", "minimum_pairs", "max_sync_delta_ms", "layout", "image_size_px", "pattern_inner_corners", "square_mm", "opencv_version")
        },
        "solve_mode": result.audit["solve_mode"],
        "baseline_used_as_constraint": result.audit["solve_mode"] == "baseline_anchored_centered_pinhole",
        "validated": result.validated,
        "rejection_reasons": result.rejection_reasons,
        "input_pair_count": result.audit["input_pair_count"],
        "used_pair_count": result.pair_count,
        "discarded_pair_indices": result.audit["discarded_pair_indices"],
        "left_rms_px": result.left_rms_px,
        "right_rms_px": result.right_rms_px,
        "stereo_rms_px": result.stereo_rms_px,
        "baseline_mm": result.audit["baseline_mm"],
        "diversity": result.audit["diversity"],
        "right_frame_transform": result.audit["right_frame_transform"],
        "right_frame_transform_candidates": result.audit[
            "right_frame_transform_candidates"
        ],
        "outlier_pruning": result.audit["outlier_pruning"],
        "intrinsic_model_candidates": result.audit["intrinsic_model_candidates"],
        "image_pose_geometry": result.audit["image_pose_geometry"],
        "calibration_pose_geometry": result.audit["calibration_pose_geometry"],
        "rectification_geometry": result.audit["rectification_geometry"],
    }
    if args.output:
        from pipe_twin.pipeline import atomic_write_text, ensure_paths_distinct

        ensure_paths_distinct(diagnostic=args.diagnostic, output=args.output)
        atomic_write_text(args.output, json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result.validated else 2


if __name__ == "__main__":
    raise SystemExit(main())
