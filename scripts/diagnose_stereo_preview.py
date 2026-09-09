"""Exercise the field stereo preview without opening Tk.

This small field diagnostic distinguishes camera open/read failures from
chessboard detector failures and prints one JSON summary for the issue log.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipe_twin.calibration_wizard import detect_board_corners
from pipe_twin.stereo_camera import StereoCameraSession


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--eye-width", type=int, default=1920)
    parser.add_argument("--eye-height", type=int, default=1080)
    parser.add_argument(
        "--layout",
        choices=("side_by_side_left_right", "side_by_side_right_left"),
        default="side_by_side_left_right",
    )
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--detect-every", type=int, default=3)
    parser.add_argument("--pattern-columns", type=int, default=8)
    parser.add_argument("--pattern-rows", type=int, default=6)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.seconds <= 0 or args.detect_every <= 0:
        raise SystemExit("--seconds and --detect-every must be positive")
    summary: dict[str, Any] = {
        "camera_index": args.index,
        "layout": args.layout,
        "requested_stream": [args.eye_width * 2, args.eye_height],
        "frames": 0,
        "detection_attempts": 0,
        "left_detections": 0,
        "right_detections": 0,
        "cv_errors": [],
    }

    def report(role: str):
        def _record(stage: str, error: Exception) -> None:
            summary["cv_errors"].append(
                {
                    "role": role,
                    "stage": stage,
                    "type": type(error).__name__,
                    "error": str(error),
                }
            )

        return _record

    started = time.monotonic()
    with StereoCameraSession(
        layout=args.layout,
        left_index=args.index,
        right_index=None,
        eye_width=args.eye_width,
        eye_height=args.eye_height,
    ) as session:
        while time.monotonic() - started < args.seconds:
            pair = session.read_pair()
            summary["frames"] += 1
            summary["delivered_eye_shape"] = list(pair.left.shape)
            if summary["frames"] % args.detect_every:
                continue
            summary["detection_attempts"] += 1
            pattern = (args.pattern_columns, args.pattern_rows)
            if detect_board_corners(
                pair.left,
                pattern=pattern,
                on_cv_error=report("left"),
                sb_accuracy=False,
            ) is not None:
                summary["left_detections"] += 1
            if detect_board_corners(
                pair.right,
                pattern=pattern,
                on_cv_error=report("right"),
                sb_accuracy=False,
            ) is not None:
                summary["right_detections"] += 1

    summary["elapsed_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
