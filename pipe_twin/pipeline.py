"""End-to-end replay pipeline for the current single-layer fixture."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import __version__
from .detector import ColorDiameterDetector, validate_m0_manifest
from .model_3mf import inspect_3mf
from .photo_capture import load_photo_snapshot, resolve_photo_path
from .state import confirmed_intervals, debounce_states, normalize_state


class AssetIntegrityError(ValueError):
    """Raised when a manifest-bound asset does not match its SHA-256."""


def ensure_paths_distinct(**named_paths: str | Path | None) -> None:
    """Reject aliases between any supplied input and output paths."""

    entries = [(name, Path(value)) for name, value in named_paths.items() if value is not None]
    for index, (left_name, left_path) in enumerate(entries):
        left_resolved = left_path.resolve(strict=False)
        left_key = os.path.normcase(str(left_resolved))
        for right_name, right_path in entries[index + 1 :]:
            right_resolved = right_path.resolve(strict=False)
            same_path = left_key == os.path.normcase(str(right_resolved))
            if not same_path and left_path.exists() and right_path.exists():
                try:
                    same_path = os.path.samefile(left_path, right_path)
                except OSError:
                    same_path = False
            if same_path:
                raise ValueError(
                    f"Path collision: {left_name} and {right_name} refer to {left_resolved}"
                )


def atomic_write_text(path: str | Path, content: str) -> Path:
    """Atomically replace a text output after its complete content is durable."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return target.resolve()


def atomic_write_text_bundle(outputs: dict[str | Path, str]) -> dict[Path, Path]:
    """Stage related text outputs and roll them back together on commit failure."""

    entries = [(Path(path), content) for path, content in outputs.items()]
    if not entries:
        return {}

    resolved_targets = [target.resolve(strict=False) for target, _ in entries]
    normalized = [os.path.normcase(str(target)) for target in resolved_targets]
    if len(set(normalized)) != len(normalized):
        raise ValueError("Output bundle paths must be distinct")
    for index, left in enumerate(resolved_targets):
        for right in resolved_targets[index + 1 :]:
            if left in right.parents or right in left.parents:
                raise ValueError("Output bundle paths must not contain one another")
    for target, _ in entries:
        if target.is_symlink():
            raise ValueError(f"Output bundle target must not be a symbolic link: {target}")
        if target.exists() and not target.is_file():
            raise IsADirectoryError(target)

    staged: dict[Path, Path] = {}
    backups: dict[Path, Path] = {}
    promoted: list[Path] = []
    try:
        for target, content in entries:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as stream:
                staged[target] = Path(stream.name)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())

        for target, _ in entries:
            if not target.exists():
                continue
            with tempfile.NamedTemporaryFile(
                prefix=f".{target.name}.",
                suffix=".backup",
                dir=target.parent,
                delete=False,
            ) as stream:
                backup = Path(stream.name)
            backup.unlink()
            backups[target] = backup
            os.replace(target, backup)

        for target, _ in entries:
            promoted.append(target)
            os.replace(staged[target], target)
    except BaseException as error:
        rollback_errors: list[BaseException] = []
        for target in reversed(promoted):
            try:
                target.unlink(missing_ok=True)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
        for target, backup in reversed(list(backups.items())):
            if not backup.exists():
                continue
            try:
                os.replace(backup, target)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
        if rollback_errors:
            raise RuntimeError(
                "Output bundle commit failed and rollback was incomplete"
            ) from error
        raise
    finally:
        for temporary_path in staged.values():
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass

    # Reaching this point is the bundle commit point. Backup cleanup is
    # deliberately outside the rollback region: an interruption here must
    # never remove already-promoted outputs or partially restore old ones.
    for backup in backups.values():
        try:
            backup.unlink(missing_ok=True)
        except OSError:
            pass

    return {target: target.resolve() for target, _ in entries}


def _commit_analysis_outputs(
    report: dict[str, Any],
    report_output_path: str | Path | None,
    observations_path: str | Path | None,
    observation_lines: list[str] | None,
) -> None:
    outputs: dict[str | Path, str] = {}
    if report_output_path is not None:
        outputs[report_output_path] = (
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
    if observations_path is not None and observation_lines is not None:
        content = "\n".join(observation_lines)
        if content:
            content += "\n"
        outputs[observations_path] = content
    atomic_write_text_bundle(outputs)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json_snapshot(path: str | Path) -> tuple[dict[str, Any], str]:
    """Parse JSON and hash the exact same immutable byte snapshot."""

    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("The JSON root must be an object")
    return payload, digest


def _resolve_asset(manifest_path: Path, relative_path: str) -> Path:
    asset = (manifest_path.parent / relative_path).resolve()
    if not asset.is_file():
        raise FileNotFoundError(asset)
    return asset


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.name


def _validate_hash(path: Path, expected: str) -> dict[str, Any]:
    actual = sha256_file(path)
    return {
        "path": str(path),
        "expected_sha256": expected.lower(),
        "actual_sha256": actual,
        "verified": actual == expected.lower(),
        "size_bytes": path.stat().st_size,
    }


def _validate_model(inspection: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    expected_model = manifest["model"]
    errors: list[str] = []
    if inspection["unit"] != expected_model["unit"]:
        errors.append(f"3MF unit is {inspection['unit']!r}, expected {expected_model['unit']!r}")
    if inspection["object_count"] != int(expected_model["expected_object_count"]):
        errors.append(
            f"3MF object count is {inspection['object_count']}, "
            f"expected {expected_model['expected_object_count']}"
        )

    by_id = {item["object_id"]: item for item in inspection["objects"]}
    for expected in expected_model["pipes"]:
        pipe_id = expected["pipe_id"]
        actual = by_id.get(str(expected["cad_object_id"]))
        if actual is None:
            errors.append(f"{pipe_id}: missing CAD object {expected['cad_object_id']}")
            continue
        checks = (
            ("uuid", actual.get("uuid"), expected.get("cad_uuid")),
            ("name", actual.get("name"), expected.get("cad_name")),
            ("color", actual.get("color_srgb"), expected.get("nominal_color_srgb")),
        )
        for label, actual_value, expected_value in checks:
            if actual_value != expected_value:
                errors.append(
                    f"{pipe_id}: {label} is {actual_value!r}, expected {expected_value!r}"
                )
        if abs(float(actual["diameter_mm"]) - float(expected["nominal_diameter_mm"])) > 0.1:
            errors.append(
                f"{pipe_id}: diameter {actual['diameter_mm']:.6f} mm is outside the ±0.1 mm gate"
            )
        if abs(float(actual["length_mm"]) - float(expected["nominal_length_mm"])) > 0.1:
            errors.append(
                f"{pipe_id}: length {actual['length_mm']:.6f} mm is outside the ±0.1 mm gate"
            )
        if not actual["watertight"]:
            errors.append(f"{pipe_id}: mesh is not closed and watertight")
    return errors


def _compress_states(states: Sequence[Iterable[str]], fps: float) -> list[dict[str, Any]]:
    if not states:
        return []
    intervals: list[dict[str, Any]] = []
    previous: tuple[str, ...] | None = None
    for frame_index, state_value in enumerate(states):
        state = normalize_state(state_value)
        if state == previous:
            continue
        if intervals:
            intervals[-1]["end_frame"] = frame_index - 1
            intervals[-1]["end_seconds"] = (frame_index - 1) / fps
        intervals.append(
            {
                "start_frame": frame_index,
                "start_seconds": frame_index / fps,
                "visible_pipe_ids": list(state),
            }
        )
        previous = state
    intervals[-1]["end_frame"] = len(states) - 1
    intervals[-1]["end_seconds"] = (len(states) - 1) / fps
    return intervals


def _expected_frame_states(
    expected_intervals: Sequence[dict[str, Any]], frame_count: int
) -> list[tuple[str, ...]]:
    ordered = sorted(expected_intervals, key=lambda item: int(item["start_frame"]))
    states: list[tuple[str, ...]] = []
    interval_index = 0
    for frame_index in range(frame_count):
        while (
            interval_index + 1 < len(ordered)
            and int(ordered[interval_index + 1]["start_frame"]) <= frame_index
        ):
            interval_index += 1
        if ordered and int(ordered[0]["start_frame"]) <= frame_index:
            states.append(normalize_state(ordered[interval_index]["visible_pipe_ids"]))
        else:
            states.append(tuple())
    return states


def _sequence_difference(
    detected: Sequence[tuple[str, ...]], expected: Sequence[tuple[str, ...]]
) -> int:
    shared = min(len(detected), len(expected))
    return sum(detected[index] != expected[index] for index in range(shared)) + abs(
        len(detected) - len(expected)
    )


def _percentiles(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p05": None, "p50": None, "p95": None}
    data = np.asarray(values, dtype=float)
    return {
        "count": int(len(data)),
        "p05": float(np.percentile(data, 5)),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
    }


def _serialize_observation(
    manifest: dict[str, Any],
    frame_index: int,
    fps: float,
    detection: dict[str, Any],
) -> str:
    video = manifest["video"]
    record = {
        "schema_version": "1.0",
        "frame_id": f"frame-{frame_index:06d}",
        "frame_index": frame_index,
        "timestamp_seconds": frame_index / fps,
        "camera_id": video.get("camera_id", "cad-screen-capture-0"),
        "capture_group_id": video.get("capture_group_id", manifest["dataset_id"]),
        "calibration_id": video.get("calibration_id"),
        "layer_id": manifest["scope"].get("layer_id", "L0"),
        "metric_calibrated": False,
        "visible_pipe_ids": detection["visible_pipe_ids"],
        "installation_state": "UNKNOWN",
        "observations": list(detection["observations"].values()),
        "algorithm_version": __version__,
    }
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def _analyze_video_manifest(
    manifest_path: str | Path,
    observations_path: str | Path | None = None,
    report_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Replay the legacy manifest-bound 3MF and MKV assets."""

    manifest_argument = Path(manifest_path)
    manifest_file = manifest_argument.resolve()
    manifest, manifest_sha256 = load_json_snapshot(manifest_file)
    validate_m0_manifest(manifest)
    model_path = _resolve_asset(manifest_file, manifest["model"]["path"])
    video_path = _resolve_asset(manifest_file, manifest["video"]["path"])
    ensure_paths_distinct(
        manifest=manifest_file,
        model=model_path,
        video=video_path,
        observations=observations_path,
        report=report_output_path,
    )
    model_integrity = _validate_hash(model_path, manifest["model"]["sha256"])
    video_integrity = _validate_hash(video_path, manifest["video"]["sha256"])
    if not model_integrity["verified"] or not video_integrity["verified"]:
        raise AssetIntegrityError("An input asset does not match the SHA-256 bound by the manifest")
    model_integrity["path"] = str(manifest["model"]["path"])
    video_integrity["path"] = str(manifest["video"]["path"])

    model_inspection = inspect_3mf(model_path)
    model_errors = _validate_model(model_inspection, manifest)
    detector = ColorDiameterDetector(manifest)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    metadata = {
        "width": int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))),
        "height": int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "container_frame_count": int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))),
    }
    fps = metadata["fps"]
    if fps <= 0:
        capture.release()
        raise RuntimeError("Video reports a non-positive frame rate")

    raw_states: list[tuple[str, ...]] = []
    measurements: dict[str, dict[str, list[float]]] = {
        pipe["pipe_id"]: {"diameter_px": [], "projected_diameter_estimate_mm": [], "score": []}
        for pipe in manifest["model"]["pipes"]
    }
    observation_lines: list[str] | None = [] if observations_path is not None else None

    try:
        frame_index = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            detection = detector.detect(frame)
            raw_states.append(normalize_state(detection["visible_pipe_ids"]))
            for pipe_id, observation in detection["observations"].items():
                if observation["visibility"] != "VISIBLE":
                    continue
                measurements[pipe_id]["diameter_px"].append(float(observation["diameter_px"]))
                measurements[pipe_id]["projected_diameter_estimate_mm"].append(
                    float(observation["projected_diameter_estimate_mm"])
                )
                measurements[pipe_id]["score"].append(float(observation["match_score"]))
            if observation_lines is not None:
                observation_lines.append(
                    _serialize_observation(manifest, frame_index, fps, detection)
                )
            frame_index += 1
    finally:
        capture.release()

    decoded_frame_count = len(raw_states)
    config = manifest["video"]
    minimum_stable_frames = int(config["minimum_stable_frames"])
    stable_states = debounce_states(raw_states, minimum_stable_frames, fps)
    raw_intervals = _compress_states(raw_states, fps)
    stable_intervals = confirmed_intervals(stable_states, minimum_stable_frames, fps)
    expected_intervals = config["expected_visibility_intervals"]
    expected_states = _expected_frame_states(expected_intervals, decoded_frame_count)

    correct_frames = sum(
        detected == expected for detected, expected in zip(raw_states, expected_states, strict=True)
    )
    frame_accuracy = correct_frames / decoded_frame_count if decoded_frame_count else 0.0
    per_pipe_accuracy: dict[str, float] = {}
    for pipe in manifest["model"]["pipes"]:
        pipe_id = pipe["pipe_id"]
        correct = sum(
            (pipe_id in detected) == (pipe_id in expected)
            for detected, expected in zip(raw_states, expected_states, strict=True)
        )
        per_pipe_accuracy[pipe_id] = correct / decoded_frame_count if decoded_frame_count else 0.0

    expected_sequence = [normalize_state(item["visible_pipe_ids"]) for item in expected_intervals]
    stable_sequence = [normalize_state(item["visible_pipe_ids"]) for item in stable_intervals]
    raw_sequence = [normalize_state(item["visible_pipe_ids"]) for item in raw_intervals]
    false_transitions = _sequence_difference(stable_sequence, expected_sequence)

    evidence_boundary_errors: list[int] = []
    confirmation_boundary_errors: list[int] = []
    if stable_sequence == expected_sequence:
        evidence_boundary_errors = [
            abs(int(actual["start_frame"]) - int(expected["start_frame"]))
            for actual, expected in zip(stable_intervals, expected_intervals, strict=True)
        ]
        confirmation_boundary_errors = [
            abs(int(actual["confirmation_frame"]) - int(expected["start_frame"]))
            for actual, expected in zip(stable_intervals, expected_intervals, strict=True)
        ]

    metadata_errors: list[str] = []
    expected_metadata = {
        "width": int(config["expected_width"]),
        "height": int(config["expected_height"]),
        "container_frame_count": int(config["expected_frame_count"]),
    }
    for key, expected_value in expected_metadata.items():
        if metadata[key] != expected_value:
            metadata_errors.append(f"{key} is {metadata[key]}, expected {expected_value}")
    if abs(fps - float(config["expected_fps"])) > 0.01:
        metadata_errors.append(f"fps is {fps}, expected {config['expected_fps']}")
    if decoded_frame_count != int(config["expected_frame_count"]):
        metadata_errors.append(
            f"decoded frame count is {decoded_frame_count}, expected {config['expected_frame_count']}"
        )

    measurement_summary = {
        pipe_id: {
            "diameter_px": _percentiles(values["diameter_px"]),
            "projected_diameter_estimate_mm": _percentiles(
                values["projected_diameter_estimate_mm"]
            ),
            "match_score": _percentiles(values["score"]),
            "diameter_source": "model_prior",
            "projected_diameter_method": "uncalibrated_aspect_ratio",
            "scale_source": "nominal_length_prior",
            "metric_calibrated": False,
        }
        for pipe_id, values in measurements.items()
    }
    maximum_evidence_boundary_error = max(evidence_boundary_errors, default=None)
    maximum_confirmation_boundary_error = max(confirmation_boundary_errors, default=None)
    acceptance = manifest["acceptance"]
    passed = (
        not model_errors
        and not metadata_errors
        and frame_accuracy >= float(acceptance["visibility_frame_accuracy_min"])
        and maximum_confirmation_boundary_error is not None
        and maximum_confirmation_boundary_error
        <= int(acceptance["event_boundary_error_frames_max"])
        and false_transitions <= int(acceptance["stable_false_transition_count_max"])
        and raw_sequence == expected_sequence
    )

    if hashlib.sha256(manifest_file.read_bytes()).hexdigest() != manifest_sha256:
        raise AssetIntegrityError("Manifest changed during replay; no report was committed")
    if sha256_file(model_path) != manifest["model"]["sha256"].lower():
        raise AssetIntegrityError("3MF model changed during replay; no report was committed")
    if sha256_file(video_path) != manifest["video"]["sha256"].lower():
        raise AssetIntegrityError("MKV video changed during replay; no report was committed")

    report = {
        "schema_version": "1.0",
        "report_type": "single-layer-color-diameter-m0-replay",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "software_version": __version__,
        "dataset_id": manifest["dataset_id"],
        "model_revision": manifest["model_revision"],
        "passed": passed,
        "capability": {
            "level": "M0",
            "layer_model": "single",
            "identity_features": manifest["scope"]["identity_features"],
            "metric_calibrated": False,
            "installation_state_inferred": False,
            "stereo_supported": False,
            "multi_layer_supported": False,
            "multi_camera_3dgs_reserved_only": True,
        },
        "inputs": {
            "manifest": {
                "path": _display_path(manifest_file),
                "sha256": manifest_sha256,
            },
            "model": model_integrity,
            "video": video_integrity,
        },
        "model_audit": {
            "passed": not model_errors,
            "validation_errors": model_errors,
            "inspection": model_inspection,
        },
        "video_audit": {
            "passed": not metadata_errors,
            "metadata": metadata,
            "decoded_frame_count": decoded_frame_count,
            "duration_seconds": decoded_frame_count / fps,
            "validation_errors": metadata_errors,
        },
        "measurement_summary": measurement_summary,
        "visibility_timeline": stable_intervals,
        "raw_visibility_timeline": raw_intervals,
        "evaluation": {
            "visibility_frame_accuracy": frame_accuracy,
            "correct_frame_count": correct_frames,
            "evaluated_frame_count": decoded_frame_count,
            "per_pipe_visibility_accuracy": per_pipe_accuracy,
            "stable_false_transition_count": false_transitions,
            "event_evidence_boundary_errors_frames": evidence_boundary_errors,
            "event_confirmation_boundary_errors_frames": confirmation_boundary_errors,
            "event_evidence_boundary_error_frames_max": maximum_evidence_boundary_error,
            "event_boundary_error_frames_max": maximum_confirmation_boundary_error,
            "expected_interval_count": len(expected_intervals),
            "detected_interval_count": len(stable_intervals),
            "acceptance_thresholds": acceptance,
        },
        "state_semantics": {
            "visible": "The fixture produced direct image evidence for this pipe candidate.",
            "not_observed": "No qualified image candidate; this does not prove removal or absence.",
            "installation_state": "UNKNOWN for every observation in this M0 replay.",
        },
        "limitations": manifest["limitations"],
    }
    _commit_analysis_outputs(
        report,
        report_output_path,
        observations_path,
        observation_lines,
    )
    return report


def _serialize_still_observation(
    manifest: dict[str, Any],
    manifest_sha256: str,
    capture_record: dict[str, Any],
    view: dict[str, Any],
    source_sha256: str,
    sample_index: int,
    detection: dict[str, Any],
) -> str:
    capture = manifest["capture"]
    record = {
        "schema_version": "1.1",
        "observation_id": f"{capture_record['capture_id']}:mono",
        "capture_id": capture_record["capture_id"],
        "sample_index": sample_index,
        "view_role": "mono",
        "camera_id": view["camera_id"],
        "capture_group_id": capture["capture_group_id"],
        "dataset_id": manifest["dataset_id"],
        "model_revision": manifest["model_revision"],
        "manifest_sha256": manifest_sha256,
        "captured_at": view["captured_at"],
        "timestamp_source": view["timestamp_source"],
        "calibration_id": capture.get("calibration_id"),
        "orientation_policy": view["orientation_policy"],
        "source_sha256": source_sha256,
        "layer_id": manifest["scope"].get("layer_id", "L0"),
        "metric_calibrated": False,
        "visible_pipe_ids": detection["visible_pipe_ids"],
        "installation_state": "UNKNOWN",
        "observations": list(detection["observations"].values()),
        "algorithm_version": __version__,
    }
    return json.dumps(record, ensure_ascii=False, separators=(",", ":"))


def _photo_quality_metrics(image: np.ndarray) -> dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    percentiles = np.percentile(gray, (5, 50, 95))
    return {
        "status": "UNVALIDATED",
        "field_quality_policy_validated": False,
        "focus_laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "luminance_p05": float(percentiles[0]),
        "luminance_p50": float(percentiles[1]),
        "luminance_p95": float(percentiles[2]),
        "dark_clipped_fraction": float(np.mean(gray <= 5)),
        "bright_clipped_fraction": float(np.mean(gray >= 250)),
    }


def _analyze_still_capture_manifest(
    manifest_path: str | Path,
    observations_path: str | Path | None = None,
    report_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze a manifest-bound set of scheduled still photographs."""

    manifest_file = Path(manifest_path).resolve()
    manifest, manifest_sha256 = load_json_snapshot(manifest_file)
    validate_m0_manifest(manifest)
    if "capture" not in manifest:
        raise ValueError("Expected a still-capture manifest")

    model_path = _resolve_asset(manifest_file, manifest["model"]["path"])
    capture_config = manifest["capture"]
    named_paths: dict[str, str | Path | None] = {
        "manifest": manifest_file,
        "model": model_path,
        "observations": observations_path,
        "report": report_output_path,
    }
    for capture_record in capture_config["captures"]:
        view = capture_record["views"]["mono"]
        named_paths[f"photo_{capture_record['capture_id']}"] = resolve_photo_path(
            manifest_file, view["path"]
        )
    ensure_paths_distinct(**named_paths)

    model_integrity = _validate_hash(model_path, manifest["model"]["sha256"])
    if not model_integrity["verified"]:
        raise AssetIntegrityError("The 3MF model does not match its manifest SHA-256")
    model_integrity["path"] = str(manifest["model"]["path"])

    model_inspection = inspect_3mf(model_path)
    model_errors = _validate_model(model_inspection, manifest)
    detector = ColorDiameterDetector(manifest)
    measurements: dict[str, dict[str, list[float]]] = {
        pipe["pipe_id"]: {
            "diameter_px": [],
            "projected_diameter_estimate_mm": [],
            "score": [],
        }
        for pipe in manifest["model"]["pipes"]
    }
    observation_lines: list[str] | None = [] if observations_path is not None else None
    photo_results: list[dict[str, Any]] = []
    evaluated_results: list[bool] = []

    for sample_index, capture_record in enumerate(capture_config["captures"]):
        view = capture_record["views"]["mono"]
        try:
            image, integrity = load_photo_snapshot(manifest_file, view)
        except ValueError as error:
            raise AssetIntegrityError(str(error)) from error
        detection = detector.detect(image)
        for pipe_id, observation in detection["observations"].items():
            if observation["visibility"] != "VISIBLE":
                continue
            measurements[pipe_id]["diameter_px"].append(float(observation["diameter_px"]))
            measurements[pipe_id]["projected_diameter_estimate_mm"].append(
                float(observation["projected_diameter_estimate_mm"])
            )
            measurements[pipe_id]["score"].append(float(observation["match_score"]))

        expected = capture_record.get("expected_visible_pipe_ids")
        exact_match: bool | None = None
        if expected is not None:
            exact_match = normalize_state(expected) == normalize_state(
                detection["visible_pipe_ids"]
            )
            evaluated_results.append(exact_match)
        photo_results.append(
            {
                "capture_id": capture_record["capture_id"],
                "view_role": "mono",
                "camera_id": view["camera_id"],
                "captured_at": view["captured_at"],
                "timestamp_source": view["timestamp_source"],
                "asset": integrity,
                "quality": _photo_quality_metrics(image),
                "visible_pipe_ids": detection["visible_pipe_ids"],
                "expected_visible_pipe_ids": expected,
                "exact_match": exact_match,
                "installation_state": "UNKNOWN",
                "observations": list(detection["observations"].values()),
            }
        )
        if observation_lines is not None:
            observation_lines.append(
                _serialize_still_observation(
                    manifest,
                    manifest_sha256,
                    capture_record,
                    view,
                    integrity["actual_sha256"],
                    sample_index,
                    detection,
                )
            )

    if len(evaluated_results) == len(photo_results):
        evaluation_status = "EVALUATED"
        passed: bool | None = not model_errors and all(evaluated_results)
    elif evaluated_results:
        evaluation_status = "PARTIALLY_EVALUATED"
        passed = False if model_errors or not all(evaluated_results) else None
    else:
        evaluation_status = "NOT_EVALUATED"
        passed = False if model_errors else None

    measurement_summary = {
        pipe_id: {
            "diameter_px": _percentiles(values["diameter_px"]),
            "projected_diameter_estimate_mm": _percentiles(
                values["projected_diameter_estimate_mm"]
            ),
            "match_score": _percentiles(values["score"]),
            "diameter_source": "model_prior",
            "projected_diameter_method": "uncalibrated_aspect_ratio",
            "scale_source": "nominal_length_prior",
            "metric_calibrated": False,
        }
        for pipe_id, values in measurements.items()
    }
    duplicate_content_sha256_count = len(photo_results) - len(
        {result["asset"]["actual_sha256"] for result in photo_results}
    )

    if hashlib.sha256(manifest_file.read_bytes()).hexdigest() != manifest_sha256:
        raise AssetIntegrityError("Manifest changed during photo analysis; no report was committed")
    if sha256_file(model_path) != manifest["model"]["sha256"].lower():
        raise AssetIntegrityError("3MF model changed during photo analysis; no report was committed")
    for capture_record in capture_config["captures"]:
        view = capture_record["views"]["mono"]
        photo_path = resolve_photo_path(manifest_file, view["path"])
        if sha256_file(photo_path) != str(view["sha256"]).lower():
            raise AssetIntegrityError(
                f"Photo changed during analysis: {capture_record['capture_id']}"
            )

    report = {
        "schema_version": "1.1",
        "report_type": "single-layer-color-diameter-m0-still-capture",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "software_version": __version__,
        "dataset_id": manifest["dataset_id"],
        "model_revision": manifest["model_revision"],
        "passed": passed,
        "intake_passed": not model_errors,
        "field_acceptance_passed": None,
        "capability": {
            "level": "M0",
            "input_kind": "scheduled_still_capture_set",
            "camera_layout": "mono",
            "metric_calibrated": False,
            "field_quality_policy_validated": False,
            "installation_state_inferred": False,
            "stereo_supported": False,
            "multi_layer_supported": False,
        },
        "inputs": {
            "manifest": {
                "path": _display_path(manifest_file),
                "sha256": manifest_sha256,
            },
            "model": model_integrity,
            "photos": [result["asset"] for result in photo_results],
        },
        "model_audit": {
            "passed": not model_errors,
            "validation_errors": model_errors,
            "inspection": model_inspection,
        },
        "capture_audit": {
            "capture_group_id": capture_config["capture_group_id"],
            "interval_minutes": capture_config["interval_minutes"],
            "camera_layout": capture_config["camera_layout"],
            "capture_count": len(photo_results),
            "continuous_video_used": False,
            "duplicate_content_sha256_count": duplicate_content_sha256_count,
            "duplicate_content_is_failure": False,
            "photos": photo_results,
        },
        "measurement_summary": measurement_summary,
        "evaluation": {
            "status": evaluation_status,
            "scope": "manifest_oracle_exact_visible_pipe_ids",
            "unit": "capture",
            "evaluated_capture_count": len(evaluated_results),
            "correct_capture_count": sum(evaluated_results),
            "exact_match_accuracy": (
                sum(evaluated_results) / len(evaluated_results)
                if evaluated_results
                else None
            ),
            "video_event_metrics_applicable": False,
            "field_acceptance_applicable": False,
        },
        "state_semantics": {
            "visible": "This photograph produced direct image evidence for the pipe candidate.",
            "not_observed": "No qualified image candidate; this does not prove non-installation.",
            "installation_state": "UNKNOWN for every M0 still-photo observation.",
        },
        "limitations": manifest["limitations"],
    }
    _commit_analysis_outputs(
        report,
        report_output_path,
        observations_path,
        observation_lines,
    )
    return report


def analyze_manifest(
    manifest_path: str | Path,
    observations_path: str | Path | None = None,
    report_output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Analyze a manifest-bound legacy video or scheduled still capture set."""

    manifest, _ = load_json_snapshot(Path(manifest_path).resolve())
    validate_m0_manifest(manifest)
    if "video" in manifest:
        return _analyze_video_manifest(
            manifest_path,
            observations_path,
            report_output_path,
        )
    return _analyze_still_capture_manifest(
        manifest_path,
        observations_path,
        report_output_path,
    )
