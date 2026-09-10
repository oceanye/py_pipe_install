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
    args = parser.parse_args()
    result = replay_calibration_diagnostic(args.diagnostic)
    summary = {
        "validated": result.validated,
        "rejection_reasons": result.rejection_reasons,
        "input_pair_count": result.audit["input_pair_count"],
        "used_pair_count": result.pair_count,
        "discarded_pair_indices": result.audit["discarded_pair_indices"],
        "left_rms_px": result.left_rms_px,
        "right_rms_px": result.right_rms_px,
        "stereo_rms_px": result.stereo_rms_px,
        "baseline_mm": result.audit["baseline_mm"],
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
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if result.validated else 2


if __name__ == "__main__":
    raise SystemExit(main())
