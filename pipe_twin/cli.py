"""Command-line entry points for fixture inspection, replay, and simulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .cad_model import load_cad_scene
from .model_3mf import inspect_3mf
from .pipeline import analyze_manifest, atomic_write_text, ensure_paths_distinct
from .synthetic_stereo import generate_synthetic_stereo
from .logging_config import get_logger, log_event

_LOGGER = get_logger("cli")


def _write_json(path: str | Path, payload: dict) -> Path:
    return atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def _inspect_model(
    path: str | Path,
    *,
    required_object_ids: list[str] | None = None,
    stl_unit: str | None = None,
) -> dict:
    """Return the established 3MF audit or a normalized 3DM/STL mesh audit."""

    model_path = Path(path)
    if model_path.suffix.lower() == ".3mf" and not required_object_ids:
        return inspect_3mf(model_path)
    scene = load_cad_scene(
        model_path,
        required_object_ids=required_object_ids,
        stl_unit=stl_unit,
    )
    return {
        "format": "Rhino 3DM" if scene.source_format == "3dm" else scene.source_format,
        "source_path": str(scene.source_path),
        "source_sha256": scene.source_sha256,
        "source_unit": scene.source_unit,
        "unit_scale_to_mm": scene.unit_scale_to_mm,
        "normalized_unit": "millimeter",
        "object_count": scene.object_count,
        "total_vertex_count": sum(item.vertex_count for item in scene.objects),
        "total_triangle_count": sum(item.triangle_count for item in scene.objects),
        "all_meshes_watertight": all(item.watertight for item in scene.objects),
        "objects": [
            {
                "object_id": item.object_id,
                "guid": item.guid,
                "pipe_id": item.pipe_id,
                "name": item.name,
                "layer_path": item.layer_path,
                "color_srgb": item.color_srgb,
                "geometry_type": item.geometry_type,
                "mesh_source": item.mesh_source,
                "vertex_count": item.vertex_count,
                "triangle_count": item.triangle_count,
                "bbox_min_mm": list(item.bbox_min_mm),
                "bbox_max_mm": list(item.bbox_max_mm),
                "watertight": item.watertight,
            }
            for item in scene.objects
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pipe_twin",
        description="Audit and generate traceable pipe digital-twin fixtures.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect-model", help="inspect a mesh-backed 3MF, Rhino 3DM, or STL model"
    )
    inspect_parser.add_argument("model", help="path to the 3MF, 3DM, or STL file")
    inspect_parser.add_argument(
        "--stl-unit",
        choices=("millimeter", "centimeter", "meter", "inch"),
        help="required source coordinate unit for STL, because STL stores no unit metadata",
    )
    inspect_parser.add_argument(
        "--object-id",
        dest="object_ids",
        action="append",
        help=(
            "inspect only this manifest-bound CAD object identity; repeat for multiple "
            "Rhino GUIDs, 3MF object IDs, or STL component IDs"
        ),
    )
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

    stereo_parser = subparsers.add_parser(
        "analyze-stereo",
        help="analyze manifest-bound stereo still photos against a CAD model",
    )
    stereo_parser.add_argument("--manifest", required=True, help="stereo manifest JSON")
    stereo_parser.add_argument("--output", required=True, help="installation report JSON")
    stereo_parser.add_argument(
        "--evidence-dir",
        help="optional directory for status-coloured CAD overlay images",
    )

    adapt_parser = subparsers.add_parser(
        "adapt-calibration",
        help="convert OpenCV K/D/R/T plus a CAD pose to manifest calibration JSON",
    )
    adapt_parser.add_argument("--source", required=True, help="OpenCV/GLM calibration JSON")
    adapt_parser.add_argument("--output", required=True, help="manifest calibration JSON output")
    adapt_parser.add_argument("--calibration-id", required=True)
    adapt_parser.add_argument("--validated", action="store_true")
    adapt_parser.add_argument("--registration-validated", action="store_true")

    validate_parser = subparsers.add_parser(
        "validate-calibration",
        help="validate a manifest-bound calibration and print repair guidance",
    )
    validate_parser.add_argument(
        "--calibration", required=True, help="calibration JSON or full stereo manifest"
    )
    validate_parser.add_argument("--output", help="optional JSON diagnostics output")

    gui_parser = subparsers.add_parser(
        "gui", help="open the local CAD-bound pipe status dashboard"
    )
    gui_parser.add_argument("--manifest", help="optional manifest JSON; omit for an empty measurement workbench")
    gui_parser.add_argument("--report", help="optional recognition report JSON")
    return parser


def _main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_event(_LOGGER, "command_start", command=args.command)
    if args.command == "adapt-calibration":
        from .calibration_adapter import adapt_opencv_stereo_calibration, load_json

        source = load_json(args.source)
        source = source.get("opencv_stereo_calibration", source)
        calibration = adapt_opencv_stereo_calibration(
            source,
            calibration_id=args.calibration_id,
            validated=args.validated,
            registration_validated=args.registration_validated,
        )
        _write_json(args.output, calibration)
        print(f"Calibration written to {Path(args.output).resolve()}")
        return 0
    if args.command == "validate-calibration":
        from .calibration_adapter import load_json, validate_calibration

        diagnostics = validate_calibration(load_json(args.calibration))
        if args.output:
            _write_json(args.output, diagnostics)
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
        return 0 if diagnostics["valid"] else 1
    if args.command == "inspect-model":
        if args.output:
            ensure_paths_distinct(model=args.model, output=args.output)
        result = _inspect_model(
            args.model,
            required_object_ids=args.object_ids,
            stl_unit=args.stl_unit,
        )
        if args.output:
            output = _write_json(args.output, result)
            print(f"CAD inspection written to {output}")
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

    if args.command == "analyze-stereo":
        from .stereo_analyzer import analyze_stereo_capture

        report = analyze_stereo_capture(
            args.manifest,
            report_output_path=args.output,
            evidence_dir=args.evidence_dir,
        )
        counts = report["counts"]
        print(
            "Stereo CAD analysis completed: "
            f"installed={counts['INSTALLED']}, "
            f"not_installed={counts['NOT_INSTALLED']}, "
            f"unknown={counts['UNKNOWN']}"
        )
        print(f"Report written to {Path(args.output).resolve()}")
        if args.evidence_dir:
            print(f"Evidence overlays written to {Path(args.evidence_dir).resolve()}")
        return 0

    if args.command == "gui":
        from .gui import launch_gui

        launch_gui(args.manifest, args.report)
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


def main(argv: Sequence[str] | None = None) -> int:
    try:
        result = _main(argv)
    except Exception:
        _LOGGER.exception("command_failed", extra={"event": "command_failed", "fields": {}})
        raise
    log_event(_LOGGER, "command_finished", exit_code=result)
    return result
