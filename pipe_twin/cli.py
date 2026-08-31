"""Command-line entry points for fixture inspection, replay, and simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .model_3mf import inspect_3mf
from .pipeline import analyze_manifest, atomic_write_text, ensure_paths_distinct
from .synthetic_stereo import generate_synthetic_stereo


def _write_json(path: str | Path, payload: dict) -> Path:
    return atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pipe_twin",
        description="Audit and generate traceable pipe digital-twin fixtures.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-model", help="inspect a 3MF model")
    inspect_parser.add_argument("model", help="path to the 3MF file")
    inspect_parser.add_argument("--output", help="optional JSON output path")

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="analyze a manifest-bound legacy video or scheduled still photos",
    )
    analyze_parser.add_argument("--manifest", required=True, help="path to manifest.json")
    analyze_parser.add_argument("--output", required=True, help="JSON report output path")
    analyze_parser.add_argument(
        "--observations",
        help="optional frame/photo-level JSONL evidence output path",
    )

    simulate_parser = subparsers.add_parser(
        "simulate-stereo",
        help="render deterministic stereo/depth/instance and elevation truth",
    )
    simulate_parser.add_argument(
        "--manifest",
        required=True,
        help="path to a synthetic_cad_truth manifest",
    )
    simulate_parser.add_argument(
        "--output-dir",
        required=True,
        help="directory for generated truth products",
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

    if args.command == "simulate-stereo":
        result = generate_synthetic_stereo(args.manifest, args.output_dir)
        output = Path(args.output_dir).resolve()
        print(
            f"Synthetic stereo dataset {result['dataset_id']} written to {output} "
            f"({len(result['instance_catalog'])} instances)"
        )
        return 0

    report = analyze_manifest(
        args.manifest,
        args.observations,
        report_output_path=args.output,
    )
    output = Path(args.output).resolve()
    evaluation = report["evaluation"]
    if report["report_type"].endswith("still-capture"):
        result_label = (
            "PASS"
            if report["passed"] is True
            else "FAIL"
            if report["passed"] is False
            else "NOT_EVALUATED"
        )
        print(
            f"M0 still-photo analysis {result_label}: "
            f"{evaluation['correct_capture_count']}/"
            f"{evaluation['evaluated_capture_count']} evaluated capture(s)"
        )
    else:
        print(
            f"M0 replay {'PASS' if report['passed'] else 'FAIL'}: "
            f"{evaluation['correct_frame_count']}/{evaluation['evaluated_frame_count']} frames, "
            f"max event error={evaluation['event_boundary_error_frames_max']} frame(s)"
        )
    print(f"Report written to {output}")
    if args.observations:
        print(f"Image evidence written to {Path(args.observations).resolve()}")
    return 1 if report["passed"] is False else 0
