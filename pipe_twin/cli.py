"""Command-line entry points for fixture inspection, replay, and simulation."""

from __future__ import annotations

import argparse
import json
import time
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

    legacy_parser = subparsers.add_parser(
        "adapt-legacy-calibration",
        help="safely convert a legacy camera_config.py after explicit unit and CAD pose review",
    )
    legacy_parser.add_argument("--source", required=True, help="legacy camera_config.py")
    legacy_parser.add_argument("--output", required=True, help="manifest calibration JSON output")
    legacy_parser.add_argument("--calibration-id", required=True)
    legacy_parser.add_argument(
        "--translation-unit",
        required=True,
        choices=("mm", "cm", "m"),
        help="unit of the legacy T vector; the adapter will not infer this",
    )
    legacy_parser.add_argument(
        "--left-pose-json",
        required=True,
        help="JSON object with rotation_world_to_camera and center_world_mm",
    )
    legacy_parser.add_argument("--validated", action="store_true")
    legacy_parser.add_argument("--registration-validated", action="store_true")

    validate_parser = subparsers.add_parser(
        "validate-calibration",
        help="validate a manifest-bound calibration and print repair guidance",
    )
    validate_parser.add_argument(
        "--calibration", required=True, help="calibration JSON or full stereo manifest"
    )
    validate_parser.add_argument("--output", help="optional JSON diagnostics output")

    calibrate_parser = subparsers.add_parser(
        "calibrate-stereo",
        help="automatically calibrate a stereo rig from paired chessboard photo folders",
    )
    calibrate_parser.add_argument("--left-dir", required=True, help="left-camera chessboard photos")
    calibrate_parser.add_argument("--right-dir", required=True, help="right-camera chessboard photos")
    calibrate_parser.add_argument("--output", required=True, help="generated calibration JSON")
    calibrate_parser.add_argument("--calibration-id", default="FIELD-AUTO-STEREO")
    calibrate_parser.add_argument("--board-columns", type=int, default=9, help="inner corners (default: 9)")
    calibrate_parser.add_argument("--board-rows", type=int, default=6, help="inner corners (default: 6)")
    calibrate_parser.add_argument("--square-size-mm", type=float, default=25.0, help="checker square edge in mm (default: 25)")
    calibrate_parser.add_argument("--min-pairs", type=int, default=8, help="minimum usable pairs (default: 8)")
    calibrate_parser.add_argument(
        "--max-rms-px", type=float, default=1.5,
        help="reject poor calibration above this RMS reprojection error; 0 uses the 1.5 px safe default",
    )
    calibrate_parser.add_argument(
        "--expected-baseline-mm",
        type=float,
        default=0.0,
        help="measured lens-centre distance in mm; 0 means unknown",
    )

    capture_parser = subparsers.add_parser(
        "capture-stereo",
        help="unattended capture of paired raw stereo PNGs from a UVC camera",
    )
    capture_parser.add_argument("--output-dir", required=True, help="empty directory for PNGs and capture.json")
    capture_parser.add_argument("--left-index", type=int, default=0, help="camera index (side-by-side stream by default)")
    capture_parser.add_argument("--right-index", type=int, help="right camera index when using separate devices")
    capture_parser.add_argument(
        "--layout", choices=("side_by_side_left_right", "side_by_side_right_left", "separate_devices"),
        default="side_by_side_left_right",
    )
    capture_parser.add_argument("--eye-width", type=int, default=640)
    capture_parser.add_argument("--eye-height", type=int, default=480)
    capture_parser.add_argument("--count", type=int, help="number of pairs; defaults to one unless --duration-s is set")
    capture_parser.add_argument("--interval-s", type=float, default=0.0, help="delay between captures")
    capture_parser.add_argument("--duration-s", type=float, help="capture until this duration elapses")
    capture_parser.add_argument("--detect-chessboard", action="store_true", help="record optional 11x7 chessboard detection status")
    capture_parser.add_argument("--board-columns", type=int, default=11)
    capture_parser.add_argument("--board-rows", type=int, default=7)

    fetch_parser = subparsers.add_parser(
        "fetch-stereo", help="download a completed remote run and verify all file hashes over HTTP",
    )
    fetch_parser.add_argument("--url", required=True, help="run directory URL containing evidence_manifest.json")
    fetch_parser.add_argument("--output-dir", required=True, help="local directory for the complete run")
    fetch_parser.add_argument("--timeout-s", type=float, default=45.0)
    fetch_parser.add_argument("--workers", type=int, default=4)
    agent_parser = subparsers.add_parser(
        "serve-capture",
        help="serve an authenticated, bounded remote stereo-capture control plane",
    )
    agent_parser.add_argument("--bind", default="127.0.0.1", help="bind address; use the Tailscale address on the office PC")
    agent_parser.add_argument("--port", type=int, default=8770)
    agent_parser.add_argument("--token-file", required=True, help="file containing one bearer token line")
    agent_parser.add_argument("--output-root", required=True, help="fixed root for generated remote run directories")
    agent_parser.add_argument("--file-base-url", help="read-only 8765 base URL used to build run_url responses")
    agent_parser.add_argument("--left-index", type=int, default=0)
    agent_parser.add_argument("--right-index", type=int)
    agent_parser.add_argument(
        "--layout",
        choices=("side_by_side_left_right", "side_by_side_right_left", "separate_devices"),
        default="side_by_side_left_right",
    )
    agent_parser.add_argument("--eye-width", type=int, default=1920)
    agent_parser.add_argument("--eye-height", type=int, default=1080)
    agent_parser.add_argument("--backend", type=int)
    agent_parser.add_argument("--default-count", type=int, default=1)
    agent_parser.add_argument("--max-count", type=int, default=30)
    agent_parser.add_argument("--default-interval-s", type=float, default=0.0)
    agent_parser.add_argument("--max-interval-s", type=float, default=60.0)
    agent_parser.add_argument("--max-duration-s", type=float, default=300.0)

    office_parser = subparsers.add_parser(
        "office-client",
        help="start the office GUI together with the remote capture and evidence services",
    )
    office_parser.add_argument("--bind", help="Tailscale IPv4 address; detected automatically when omitted")
    office_parser.add_argument("--capture-port", type=int, default=8770)
    office_parser.add_argument("--file-port", type=int, default=8765)
    office_parser.add_argument("--token-file", help="office token file; generated outside the evidence root")
    office_parser.add_argument("--output-root", help="office evidence root; defaults to D:\\pipe_twin_runs on Windows")
    office_parser.add_argument("--manifest")
    office_parser.add_argument("--report")
    office_parser.add_argument("--no-gui", action="store_true", help="keep services alive without opening Tk")

    remote_parser = subparsers.add_parser(
        "remote-capture",
        help="submit a bounded capture job to a remote capture agent",
    )
    remote_parser.add_argument("--agent-url", required=True, help="capture-agent base URL, for example http://100.103.31.118:8770")
    remote_parser.add_argument("--token-file", required=True)
    remote_parser.add_argument("--count", type=int)
    remote_parser.add_argument("--interval-s", type=float)
    remote_parser.add_argument("--duration-s", type=float)
    remote_parser.add_argument("--detect-chessboard", action="store_true")
    remote_parser.add_argument("--board-columns", type=int, default=8)
    remote_parser.add_argument("--board-rows", type=int, default=6)
    remote_parser.add_argument("--timeout-s", type=float, default=15.0)
    remote_parser.add_argument("--wait", action="store_true", help="poll until the job reaches a terminal state")
    remote_parser.add_argument("--poll-s", type=float, default=1.0)


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
    if args.command == "adapt-legacy-calibration":
        from .calibration_adapter import adapt_legacy_camera_config, load_json

        pose = load_json(args.left_pose_json)
        required_pose = {"rotation_world_to_camera", "center_world_mm"}
        if set(pose) != required_pose:
            raise ValueError(
                "--left-pose-json must contain exactly rotation_world_to_camera and center_world_mm"
            )
        calibration = adapt_legacy_camera_config(
            args.source,
            calibration_id=args.calibration_id,
            translation_unit=args.translation_unit,
            left_camera_pose=pose,
            validated=args.validated,
            registration_validated=args.registration_validated,
        )
        _write_json(args.output, calibration)
        print(f"Legacy calibration converted to {Path(args.output).resolve()}")
        return 0
    if args.command == "validate-calibration":
        from .calibration_adapter import load_json, validate_calibration

        diagnostics = validate_calibration(load_json(args.calibration))
        if args.output:
            _write_json(args.output, diagnostics)
        print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
        return 0 if diagnostics["valid"] else 1
    if args.command == "calibrate-stereo":
        from .calibration_wizard import (
            CalibrationWizardError,
            calibrate_stereo_from_folders,
            write_calibration,
        )

        try:
            calibration = calibrate_stereo_from_folders(
                args.left_dir,
                args.right_dir,
                board_columns=args.board_columns,
                board_rows=args.board_rows,
                square_size_mm=args.square_size_mm,
                calibration_id=args.calibration_id,
                min_pairs=args.min_pairs,
                max_rms_px=None if args.max_rms_px <= 0 else args.max_rms_px,
                expected_baseline_mm=(args.expected_baseline_mm or None),
            )
        except CalibrationWizardError as error:
            raise SystemExit(f"自动双目标定失败：{error}") from error
        output = write_calibration(args.output, calibration)
        quality = calibration["source_audit"]["auto_calibration"]
        print(
            f"自动双目标定完成：有效 {quality['accepted_pairs']}/{quality['candidate_pairs']} 对，"
            f"RMS={quality['rms_stereo_px']:.3f}px"
        )
        print(f"标定 JSON 已写入 {output.resolve()}；下一步请用 QR/位姿完成 CAD 配准。")
        return 0
    if args.command == "capture-stereo":
        from .remote_capture import capture_stereo_pairs

        manifest = capture_stereo_pairs(
            args.output_dir,
            left_index=args.left_index,
            right_index=args.right_index,
            layout=args.layout,
            eye_width=args.eye_width,
            eye_height=args.eye_height,
            count=args.count,
            interval_s=args.interval_s,
            duration_s=args.duration_s,
            detect_chessboard=args.detect_chessboard,
            board_columns=args.board_columns,
            board_rows=args.board_rows,
        )
        print(f"已采集并保存 {json.loads(manifest.read_text(encoding='utf-8'))['pair_count']} 对照片：{manifest}")
        return 0
    if args.command == "fetch-stereo":
        from .remote_fetch import fetch_stereo_run

        def show_progress(done: int, total: int, name: str) -> None:
            if done == total or done % 10 == 0:
                print(f"远程文件校验 {done}/{total}: {name}", flush=True)

        report_path = fetch_stereo_run(args.url, args.output_dir, timeout_s=args.timeout_s,
                                       workers=args.workers, progress=show_progress)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        print(f"远程数据下载校验通过：{len(report['files'])} 个文件，{report['verified_bytes']} 字节；报告：{report_path}")
        return 0
    if args.command == "serve-capture":
        from .capture_agent import CaptureAgentConfig, serve_capture_agent

        config = CaptureAgentConfig(
            output_root=Path(args.output_root),
            left_index=args.left_index,
            right_index=args.right_index,
            layout=args.layout,
            eye_width=args.eye_width,
            eye_height=args.eye_height,
            backend=args.backend,
            default_count=args.default_count,
            max_count=args.max_count,
            default_interval_s=args.default_interval_s,
            max_interval_s=args.max_interval_s,
            max_duration_s=args.max_duration_s,
            file_base_url=args.file_base_url,
        )
        serve_capture_agent(
            bind=args.bind,
            port=args.port,
            token_file=args.token_file,
            config=config,
        )
        return 0
    if args.command == "office-client":
        from .office_client import OfficeClient, OfficeClientConfig

        config = OfficeClientConfig(
            bind=args.bind or "",
            capture_port=args.capture_port,
            file_port=args.file_port,
            token_file=Path(args.token_file).expanduser() if args.token_file else None,
            output_root=Path(args.output_root).expanduser() if args.output_root else None,
        )
        return OfficeClient(config).run(
            manifest=args.manifest,
            report=args.report,
            gui=not args.no_gui,
        )
    if args.command == "remote-capture":
        from .capture_agent import get_remote_capture, read_token, submit_remote_capture

        request = {}
        for key, value in (
            ("count", args.count),
            ("interval_s", args.interval_s),
            ("duration_s", args.duration_s),
        ):
            if value is not None:
                request[key] = value
        if args.detect_chessboard:
            request.update(
                detect_chessboard=True,
                board_columns=args.board_columns,
                board_rows=args.board_rows,
            )
        token = read_token(args.token_file)
        result = submit_remote_capture(args.agent_url, token, request, timeout_s=args.timeout_s)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not args.wait:
            return 0
        job_id = result.get("job_id")
        if not isinstance(job_id, str):
            raise RuntimeError("remote capture response did not contain a job_id")
        if isinstance(args.poll_s, bool) or not 0.1 <= float(args.poll_s) <= 60.0:
            raise ValueError("poll-s must be between 0.1 and 60 seconds")
        while True:
            time.sleep(float(args.poll_s))
            result = get_remote_capture(args.agent_url, token, job_id, timeout_s=args.timeout_s)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            if result.get("state") in {"COMPLETED", "FAILED"}:
                return 0 if result.get("state") == "COMPLETED" else 1

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
            f"Stereo {report.get('mode', 'CAD')} analysis completed: "
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
