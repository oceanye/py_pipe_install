"""Command-line entry points for fixture inspection and replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .model_3mf import inspect_3mf
from .pipeline import analyze_manifest, atomic_write_text, ensure_paths_distinct


def _write_json(path: str | Path, payload: dict) -> Path:
    return atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pipe_twin",
        description="Audit the current single-layer pipe digital-twin fixture.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-model", help="inspect a 3MF model")
    inspect_parser.add_argument("model", help="path to the 3MF file")
    inspect_parser.add_argument("--output", help="optional JSON output path")

    analyze_parser = subparsers.add_parser("analyze", help="replay a manifest-bound MKV")
    analyze_parser.add_argument("--manifest", required=True, help="path to manifest.json")
    analyze_parser.add_argument("--output", required=True, help="JSON report output path")
    analyze_parser.add_argument(
        "--observations",
        help="optional frame-level JSONL evidence output path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inspect-model":
        if args.output:
            ensure_paths_distinct(model=args.model, output=args.output)
        result = inspect_3mf(args.model)
        if args.output:
            output = _write_json(args.output, result)
            print(f"3MF inspection written to {output}")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    report = analyze_manifest(
        args.manifest,
        args.observations,
        report_output_path=args.output,
    )
    output = _write_json(args.output, report)
    evaluation = report["evaluation"]
    print(
        f"M0 replay {'PASS' if report['passed'] else 'FAIL'}: "
        f"{evaluation['correct_frame_count']}/{evaluation['evaluated_frame_count']} frames, "
        f"max event error={evaluation['event_boundary_error_frames_max']} frame(s)"
    )
    print(f"Report written to {output}")
    if args.observations:
        print(f"Frame evidence written to {Path(args.observations).resolve()}")
    return 0 if report["passed"] else 1
