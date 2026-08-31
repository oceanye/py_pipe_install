from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from pipe_twin.cli import main
from pipe_twin.detector import ColorDiameterDetector
from pipe_twin.pipeline import AssetIntegrityError, analyze_manifest, sha256_file


ROOT = Path(__file__).resolve().parents[1]
BASE_MANIFEST_PATH = ROOT / "test_model" / "manifest.json"
MODEL_PATH = ROOT / "test_model" / "管道群.3mf"

BLUE_ID = "pipe-blue-d45"
RED_ID = "pipe-red-d40"
WHITE_ID = "pipe-white-d20"
ALL_PIPE_IDS = [BLUE_ID, RED_ID, WHITE_ID]


def _synthetic_photo() -> np.ndarray:
    image = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    specs = (
        ((250, 54, 0), 45, 300),
        ((0, 0, 255), 40, 450),
        ((255, 254, 254), 20, 600),
    )
    for bgr, diameter, y0 in specs:
        image[y0 : y0 + diameter, 400:950] = bgr
    return image


def _write_png(path: Path, image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise AssertionError("OpenCV failed to encode the test photograph")
    content = encoded.tobytes()
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _write_jpeg(path: Path, image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    if not success:
        raise AssertionError("OpenCV failed to encode the test photograph")
    content = encoded.tobytes()
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _still_manifest(photo_name: str, photo_sha256: str) -> dict:
    manifest = json.loads(BASE_MANIFEST_PATH.read_text(encoding="utf-8"))
    video = manifest.pop("video")
    manifest["schema_version"] = "1.1"
    manifest["dataset_id"] = "pipe-group-single-layer-still-smoke-v1"
    manifest["model"]["path"] = str(MODEL_PATH.resolve())
    manifest["capture"] = {
        "kind": "still_capture_set",
        "camera_layout": "mono",
        "capture_group_id": "inspection-20260901-0001",
        "calibration_id": None,
        "interval_minutes": 30,
        "analysis": {
            key: copy.deepcopy(video[key])
            for key in (
                "analysis_roi_xyxy",
                "minimum_component_area_px",
                "minimum_long_side_px",
                "minimum_aspect_ratio",
                "diameter_absolute_tolerance_mm",
                "diameter_relative_tolerance",
                "delta_e76_tolerance",
                "color_rules",
            )
        },
        "captures": [
            {
                "capture_id": "capture-000001",
                "expected_visible_pipe_ids": ALL_PIPE_IDS,
                "views": {
                    "mono": {
                        "camera_id": "field-camera-01",
                        "path": photo_name,
                        "sha256": photo_sha256,
                        "expected_width": 1920,
                        "expected_height": 1080,
                        "captured_at": "2026-09-01T00:30:00.123+08:00",
                        "timestamp_source": "MANIFEST_OPERATOR_CONFIRMED",
                        "orientation_policy": "RAW_PIXELS_NO_EXIF_TRANSFORM",
                    }
                },
            }
        ],
    }
    manifest["limitations"] = [
        "This is a synthetic still-photo intake smoke fixture, not a field-camera acceptance set.",
        "A still photograph with no detection never proves NOT_INSTALLED.",
    ]
    return manifest


class StillPhotoCaptureTests(unittest.TestCase):
    def test_manifest_bound_jpeg_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            photo = root / "capture.jpg"
            digest = _write_jpeg(photo, _synthetic_photo())
            manifest = _still_manifest(photo.name, digest)
            manifest["capture"]["captures"][0]["expected_visible_pipe_ids"] = None
            manifest_path = root / "capture_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = analyze_manifest(manifest_path)

            self.assertTrue(report["intake_passed"])
            self.assertIsNone(report["passed"])
            self.assertEqual(
                report["capture_audit"]["photos"][0]["asset"]["encoded_format"],
                "jpeg",
            )

    def test_manifest_bound_photo_uses_one_snapshot_without_video_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            photo = root / "capture.png"
            digest = _write_png(photo, _synthetic_photo())
            manifest = _still_manifest(photo.name, digest)
            manifest_path = root / "capture_manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            report_path = root / "report.json"
            observations_path = root / "observations.jsonl"

            exit_code = main(
                [
                    "analyze",
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(report_path),
                    "--observations",
                    str(observations_path),
                ]
            )

            self.assertEqual(exit_code, 0)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(
                report["report_type"],
                "single-layer-color-diameter-m0-still-capture",
            )
            self.assertTrue(report["passed"])
            self.assertTrue(report["intake_passed"])
            self.assertIsNone(report["field_acceptance_passed"])
            self.assertFalse(report["capture_audit"]["continuous_video_used"])
            self.assertEqual(report["capture_audit"]["interval_minutes"], 30)
            self.assertEqual(report["evaluation"]["status"], "EVALUATED")
            self.assertFalse(report["evaluation"]["video_event_metrics_applicable"])
            self.assertFalse(report["evaluation"]["field_acceptance_applicable"])
            self.assertNotIn("event_boundary_error_frames_max", report["evaluation"])
            photo_result = report["capture_audit"]["photos"][0]
            self.assertEqual(set(photo_result["visible_pipe_ids"]), set(ALL_PIPE_IDS))
            self.assertEqual(photo_result["installation_state"], "UNKNOWN")
            self.assertEqual(photo_result["quality"]["status"], "UNVALIDATED")

            observation = json.loads(observations_path.read_text(encoding="utf-8"))
            self.assertEqual(observation["capture_id"], "capture-000001")
            self.assertEqual(observation["view_role"], "mono")
            self.assertEqual(
                observation["captured_at"], "2026-09-01T00:30:00.123+08:00"
            )
            self.assertEqual(observation["source_sha256"], digest)
            self.assertEqual(
                observation["manifest_sha256"], sha256_file(manifest_path)
            )
            self.assertEqual(observation["model_revision"], manifest["model_revision"])
            self.assertNotIn("frame_index", observation)
            self.assertNotIn("timestamp_seconds", observation)

    def test_no_oracle_is_not_misreported_as_algorithm_pass_or_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            photo = root / "capture.png"
            digest = _write_png(photo, _synthetic_photo())
            manifest = _still_manifest(photo.name, digest)
            manifest["capture"]["captures"][0]["expected_visible_pipe_ids"] = None
            manifest_path = root / "capture_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = analyze_manifest(manifest_path)

            self.assertIsNone(report["passed"])
            self.assertTrue(report["intake_passed"])
            self.assertEqual(report["evaluation"]["status"], "NOT_EVALUATED")
            self.assertEqual(report["evaluation"]["evaluated_capture_count"], 0)

    def test_identical_photo_bytes_are_diagnostic_not_an_intake_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_photo = root / "capture-1.png"
            digest = _write_png(first_photo, _synthetic_photo())
            second_photo = root / "capture-2.png"
            second_photo.write_bytes(first_photo.read_bytes())
            manifest = _still_manifest(first_photo.name, digest)
            second_capture = copy.deepcopy(manifest["capture"]["captures"][0])
            second_capture["capture_id"] = "capture-000002"
            second_capture["views"]["mono"]["path"] = second_photo.name
            second_capture["views"]["mono"]["captured_at"] = (
                "2026-09-01T01:00:00.123+08:00"
            )
            manifest["capture"]["captures"].append(second_capture)
            manifest_path = root / "capture_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = analyze_manifest(manifest_path)

            self.assertTrue(report["passed"])
            self.assertEqual(report["capture_audit"]["capture_count"], 2)
            self.assertEqual(
                report["capture_audit"]["duplicate_content_sha256_count"], 1
            )
            self.assertFalse(report["capture_audit"]["duplicate_content_is_failure"])

    def test_partial_oracle_still_reports_a_known_mismatch_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_photo = root / "capture-1.png"
            digest = _write_png(first_photo, _synthetic_photo())
            second_photo = root / "capture-2.png"
            second_digest = _write_png(second_photo, _synthetic_photo())
            manifest = _still_manifest(first_photo.name, digest)
            manifest["capture"]["captures"][0]["expected_visible_pipe_ids"] = [BLUE_ID]
            second_capture = copy.deepcopy(manifest["capture"]["captures"][0])
            second_capture["capture_id"] = "capture-000002"
            second_capture["expected_visible_pipe_ids"] = None
            second_capture["views"]["mono"]["path"] = second_photo.name
            second_capture["views"]["mono"]["sha256"] = second_digest
            second_capture["views"]["mono"]["captured_at"] = (
                "2026-09-01T01:00:00.123+08:00"
            )
            manifest["capture"]["captures"].append(second_capture)
            manifest_path = root / "capture_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = analyze_manifest(manifest_path)

            self.assertFalse(report["passed"])
            self.assertEqual(report["evaluation"]["status"], "PARTIALLY_EVALUATED")
            self.assertEqual(report["evaluation"]["evaluated_capture_count"], 1)

    def test_hash_or_decode_failure_never_commits_partial_outputs(self) -> None:
        for case in ("hash", "decode"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                photo = root / "capture.png"
                if case == "hash":
                    digest = _write_png(photo, _synthetic_photo())
                    expected_digest = "0" * 64
                else:
                    photo.write_bytes(b"not-a-valid-image")
                    digest = sha256_file(photo)
                    expected_digest = digest
                manifest = _still_manifest(photo.name, expected_digest)
                manifest_path = root / "capture_manifest.json"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                report_path = root / "report.json"
                observations_path = root / "observations.jsonl"

                with self.assertRaises(AssetIntegrityError):
                    analyze_manifest(
                        manifest_path,
                        observations_path=observations_path,
                        report_output_path=report_path,
                    )

                self.assertFalse(report_path.exists())
                self.assertFalse(observations_path.exists())

    def test_capture_contract_rejects_ambiguous_or_untrusted_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            photo = root / "capture.png"
            digest = _write_png(photo, _synthetic_photo())
            base = _still_manifest(photo.name, digest)
            invalid_manifests = []

            both_sources = copy.deepcopy(base)
            both_sources["video"] = json.loads(
                BASE_MANIFEST_PATH.read_text(encoding="utf-8")
            )["video"]
            invalid_manifests.append(both_sources)

            null_video_key = copy.deepcopy(base)
            null_video_key["video"] = None
            invalid_manifests.append(null_video_key)

            no_timezone = copy.deepcopy(base)
            no_timezone["capture"]["captures"][0]["views"]["mono"]["captured_at"] = (
                "2026-09-01T00:30:00"
            )
            invalid_manifests.append(no_timezone)

            stereo_without_contract = copy.deepcopy(base)
            stereo_without_contract["capture"]["camera_layout"] = "stereo"
            invalid_manifests.append(stereo_without_contract)

            out_of_range = copy.deepcopy(base)
            out_of_range["capture"]["interval_minutes"] = 61
            invalid_manifests.append(out_of_range)

            missing_capture_group = copy.deepcopy(base)
            missing_capture_group["capture"]["capture_group_id"] = ""
            invalid_manifests.append(missing_capture_group)

            absolute_photo_path = copy.deepcopy(base)
            absolute_photo_path["capture"]["captures"][0]["views"]["mono"]["path"] = (
                "C:/untrusted/capture.png"
            )
            invalid_manifests.append(absolute_photo_path)

            escaping_photo_path = copy.deepcopy(base)
            escaping_photo_path["capture"]["captures"][0]["views"]["mono"]["path"] = (
                "../capture.png"
            )
            invalid_manifests.append(escaping_photo_path)

            unsupported_schema = copy.deepcopy(base)
            unsupported_schema["schema_version"] = "1.0"
            invalid_manifests.append(unsupported_schema)

            untrusted_timestamp_source = copy.deepcopy(base)
            untrusted_timestamp_source["capture"]["captures"][0]["views"]["mono"][
                "timestamp_source"
            ] = "GUESSED_FROM_FILENAME"
            invalid_manifests.append(untrusted_timestamp_source)

            empty_analysis = copy.deepcopy(base)
            empty_analysis["capture"]["analysis"] = {}
            invalid_manifests.append(empty_analysis)

            excessive_decoded_size = copy.deepcopy(base)
            excessive_decoded_size["capture"]["captures"][0]["views"]["mono"][
                "expected_width"
            ] = 100_000
            excessive_decoded_size["capture"]["captures"][0]["views"]["mono"][
                "expected_height"
            ] = 100_000
            invalid_manifests.append(excessive_decoded_size)

            for index, manifest in enumerate(invalid_manifests):
                with self.subTest(index=index):
                    manifest_path = root / f"invalid-{index}.json"
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        ColorDiameterDetector(manifest_path)

    def test_report_cannot_overwrite_a_bound_photo(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            photo = root / "capture.png"
            digest = _write_png(photo, _synthetic_photo())
            manifest = _still_manifest(photo.name, digest)
            manifest_path = root / "capture_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaises(ValueError):
                main(
                    [
                        "analyze",
                        "--manifest",
                        str(manifest_path),
                        "--output",
                        str(photo),
                    ]
                )

            self.assertEqual(sha256_file(photo), digest)

    def test_invalid_report_target_does_not_leave_observations_behind(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            photo = root / "capture.png"
            digest = _write_png(photo, _synthetic_photo())
            manifest = _still_manifest(photo.name, digest)
            manifest_path = root / "capture_manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report_directory = root / "report-directory"
            report_directory.mkdir()
            observations_path = root / "observations.jsonl"

            with self.assertRaises(IsADirectoryError):
                main(
                    [
                        "analyze",
                        "--manifest",
                        str(manifest_path),
                        "--output",
                        str(report_directory),
                        "--observations",
                        str(observations_path),
                    ]
                )

            self.assertFalse(observations_path.exists())


if __name__ == "__main__":
    unittest.main()
