from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from pipe_twin.stereo_analyzer import (
    StereoAnalysisError,
    StereoDepthResult,
    _analysis_config,
    _calibration_from_manifest,
    _compute_stereo_depth,
    analyze_stereo_capture,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_pipe_3mf(path: Path) -> str:
    segments = 12
    vertices: list[tuple[float, float, float]] = []
    for x in (-100.0, 100.0):
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            vertices.append((x, 15.0 * math.cos(angle), 1000.0 + 15.0 * math.sin(angle)))
    triangles: list[tuple[int, int, int]] = []
    for index in range(segments):
        following = (index + 1) % segments
        triangles.extend(
            (
                (index, segments + index, segments + following),
                (index, segments + following, following),
            )
        )
    for index in range(1, segments - 1):
        triangles.append((0, index + 1, index))
        triangles.append((segments, segments + index, segments + index + 1))

    vertex_xml = "".join(
        f'<vertex x="{x}" y="{y}" z="{z}"/>' for x, y, z in vertices
    )
    triangle_xml = "".join(
        f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in triangles
    )
    model_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<model unit="millimeter" '
        'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
        '<resources><colorgroup id="2"><color color="#FF0000"/></colorgroup>'
        '<object id="1" name="pipe" type="model" pid="2" pindex="0"><mesh>'
        f'<vertices>{vertex_xml}</vertices><triangles>{triangle_xml}</triangles>'
        '</mesh></object></resources><build><item objectid="1"/></build></model>'
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("3D/3dmodel.model", model_xml)
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()


def _write_png(path: Path, image: np.ndarray) -> str:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise AssertionError("OpenCV could not encode test image")
    content = encoded.tobytes()
    path.write_bytes(content)
    return hashlib.sha256(content).hexdigest()


def _view(path: Path, digest: str, camera_id: str, captured_at: str) -> dict:
    return {
        "camera_id": camera_id,
        "path": path.name,
        "sha256": digest,
        "expected_width": 120,
        "expected_height": 100,
        "captured_at": captured_at,
        "timestamp_source": "MANIFEST_OPERATOR_CONFIRMED",
        "orientation_policy": "RAW_PIXELS_NO_EXIF_TRANSFORM",
    }


def _manifest(
    root: Path,
    *,
    count: int,
    draw_pipe: bool,
    unique_content: bool = True,
) -> tuple[Path, dict]:
    model_path = root / "pipe.3mf"
    model_hash = _write_pipe_3mf(model_path)
    groups = []
    for index in range(count):
        left = np.full((100, 120, 3), 128, dtype=np.uint8)
        right = left.copy()
        if draw_pipe:
            cv2.line(left, (30, 50), (90, 50), (0, 0, 255), 9, cv2.LINE_8)
            cv2.line(right, (15, 50), (75, 50), (0, 0, 255), 9, cv2.LINE_8)
        if unique_content and index:
            left[0, 0] = (128 + index, 128, 128)
            right[0, 0] = (128, 128 + index, 128)
        left_path = root / f"capture-{index}-left.png"
        right_path = root / f"capture-{index}-right.png"
        left_hash = _write_png(left_path, left)
        right_hash = _write_png(right_path, right)
        minute = index * 30
        hour, minute = divmod(minute, 60)
        timestamp = f"2026-09-03T{hour:02d}:{minute:02d}:00.000+08:00"
        groups.append(
            {
                "capture_id": f"pair-{index + 1:03d}",
                "views": {
                    "left": _view(left_path, left_hash, "camera-left", timestamp),
                    "right": _view(right_path, right_hash, "camera-right", timestamp),
                },
            }
        )

    manifest = {
        "schema_version": "2.0",
        "dataset_id": "stereo-unit-test",
        "model_revision": "pipe-v1",
        "model": {
            "path": model_path.name,
            "sha256": model_hash,
            "unit": "millimeter",
            "pipes": [
                {
                    "instance_id": 1,
                    "pipe_id": "pipe-red-d30",
                    "cad_object_id": "1",
                    "layer_id": "front",
                    "color_class": "red",
                    "color_srgb": "#FF0000",
                    "nominal_diameter_mm": 30.0,
                    "centerline_world_mm": [
                        [-100.0, 0.0, 1000.0],
                        [100.0, 0.0, 1000.0],
                    ],
                }
            ],
        },
        "stereo_calibration": {
            "calibration_id": "unit-exact-v1",
            "validated": True,
            "registration_validated": True,
            "rectified": True,
            "baseline_mm": 50.0,
            "max_sync_delta_ms": 5.0,
            "left_camera": {
                "camera_id": "camera-left",
                "width": 120,
                "height": 100,
                "fx": 300.0,
                "fy": 300.0,
                "cx": 60.0,
                "cy": 50.0,
                "rotation_world_to_camera": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "center_world_mm": [0.0, 0.0, 0.0],
                "D": [0, 0, 0, 0, 0],
            },
            "right_camera": {
                "camera_id": "camera-right",
                "width": 120,
                "height": 100,
                "fx": 300.0,
                "fy": 300.0,
                "cx": 60.0,
                "cy": 50.0,
                "rotation_world_to_camera": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "center_world_mm": [50.0, 0.0, 0.0],
                "D": [0, 0, 0, 0, 0],
            },
        },
        "capture": {
            "kind": "stereo_still_capture_set",
            "camera_layout": "stereo",
            "capture_group_id": "inspection-current",
            "interval_minutes": 30,
            "capture_groups": groups,
        },
        "analysis": {
            "minimum_focus_laplacian_variance": 0.0,
            "minimum_luminance_p05": 0.0,
            "maximum_luminance_p95": 255.0,
            "minimum_amodal_pixels": 10,
            "minimum_valid_depth_fraction": 0.80,
            "minimum_target_depth_fraction": 0.80,
            "minimum_color_support_fraction": 0.60,
            "maximum_width_relative_error": 0.40,
            "depth_tolerance_mm": 20.0,
            "occlusion_margin_mm": 20.0,
            "free_space_margin_mm": 50.0,
            "minimum_free_space_fraction": 0.80,
            "fully_occluded_fraction": 0.90,
            "color_delta_e76_tolerance": 30.0,
            "minimum_repeated_absence_captures": 2,
            "minimum_depth_mm": 100.0,
            "maximum_depth_mm": 2000.0,
            "stereo_matching": {
                "min_disparity": 0,
                "num_disparities": 32,
                "block_size": 5,
            },
        },
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def _depth(value: float, *, valid: bool = True) -> StereoDepthResult:
    shape = (100, 120)
    depth = np.full(shape, value if valid else np.nan, dtype=np.float32)
    mask = np.full(shape, valid, dtype=bool)
    return StereoDepthResult(
        left_depth_mm=depth,
        right_depth_mm=depth.copy(),
        left_valid=mask,
        right_valid=mask.copy(),
        audit={
            "status": "VALID" if valid else "INVALID",
            "reason_codes": [] if valid else ["NO_BIDIRECTIONAL_VALID_DISPARITY"],
            "valid_left_fraction": float(valid),
            "valid_right_fraction": float(valid),
        },
    )


class StereoAnalyzerTests(unittest.TestCase):
    def test_left_right_inconsistent_disparity_is_invalid(self) -> None:
        class FakeMatcher:
            def __init__(self, disparity: float):
                self.encoded = np.full((100, 120), disparity * 16, dtype=np.int16)

            def compute(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
                return self.encoded

        with tempfile.TemporaryDirectory() as temporary:
            _, manifest = _manifest(Path(temporary), count=1, draw_pipe=True)
            calibration = _calibration_from_manifest(manifest["stereo_calibration"])
            config = _analysis_config(manifest["analysis"])
            image = np.full((100, 120, 3), 128, dtype=np.uint8)
            with mock.patch(
                "pipe_twin.stereo_analyzer.cv2.StereoSGBM_create",
                side_effect=[FakeMatcher(15.0), FakeMatcher(-5.0)],
            ):
                depth = _compute_stereo_depth(image, image, calibration, config)

            self.assertFalse(np.any(depth.left_valid))
            self.assertFalse(np.any(depth.right_valid))
            self.assertEqual(depth.audit["status"], "INVALID")

    def test_current_color_width_and_depth_evidence_is_installed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = _manifest(Path(temporary), count=1, draw_pipe=True)
            with mock.patch(
                "pipe_twin.stereo_analyzer._compute_stereo_depth",
                return_value=_depth(1000.0),
            ):
                report = analyze_stereo_capture(manifest_path)

            self.assertEqual(report["counts"], {"INSTALLED": 1, "NOT_INSTALLED": 0, "UNKNOWN": 0})
            pipe = report["pipes"][0]
            self.assertEqual(pipe["pipe_id"], "pipe-red-d30")
            self.assertEqual(pipe["cad_object_id"], "1")
            self.assertEqual(pipe["installation_state"], "INSTALLED")
            self.assertEqual(pipe["state_basis"], "DIRECT_STEREO_CAD_EVIDENCE")
            for role in ("left", "right"):
                view = pipe["visibility_by_view"][role]
                self.assertTrue(view["color_gate_passed"])
                self.assertTrue(view["width_gate_passed"])
                self.assertGreater(view["target_depth_fraction"], 0.9)
                self.assertIsNotNone(view["amodal_bbox_xywh"])
                self.assertIsNotNone(view["visible_bbox_xywh"])

    def test_two_clear_stereo_absences_are_not_installed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = _manifest(Path(temporary), count=2, draw_pipe=False)
            with mock.patch(
                "pipe_twin.stereo_analyzer._compute_stereo_depth",
                return_value=_depth(1300.0),
            ):
                report = analyze_stereo_capture(manifest_path)

            pipe = report["pipes"][0]
            self.assertEqual(pipe["installation_state"], "NOT_INSTALLED")
            self.assertEqual(len(pipe["negative_evidence_capture_ids"]), 2)
            self.assertEqual(pipe["reason_codes"], ["ALL_NEGATIVE_EVIDENCE_GATES_PASSED"])

    def test_duplicate_photo_pair_does_not_count_as_repeated_absence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = _manifest(
                Path(temporary), count=2, draw_pipe=False, unique_content=False
            )
            with mock.patch(
                "pipe_twin.stereo_analyzer._compute_stereo_depth",
                return_value=_depth(1300.0),
            ):
                report = analyze_stereo_capture(manifest_path)

            pipe = report["pipes"][0]
            self.assertEqual(pipe["installation_state"], "UNKNOWN")
            self.assertEqual(len(pipe["negative_evidence_capture_ids"]), 2)
            self.assertEqual(pipe["negative_evidence_unique_capture_count"], 1)

    def test_too_close_retries_do_not_count_as_independent_absences(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = _manifest(root, count=2, draw_pipe=False)
            manifest["capture"]["capture_groups"][1]["views"]["left"][
                "captured_at"
            ] = "2026-09-03T00:00:00.001+08:00"
            manifest["capture"]["capture_groups"][1]["views"]["right"][
                "captured_at"
            ] = "2026-09-03T00:00:00.001+08:00"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch(
                "pipe_twin.stereo_analyzer._compute_stereo_depth",
                return_value=_depth(1300.0),
            ):
                report = analyze_stereo_capture(manifest_path)

            pipe = report["pipes"][0]
            self.assertEqual(pipe["installation_state"], "UNKNOWN")
            self.assertEqual(pipe["negative_evidence_unique_capture_count"], 1)

    def test_historical_absence_cannot_override_inconclusive_current_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = _manifest(Path(temporary), count=3, draw_pipe=False)
            with mock.patch(
                "pipe_twin.stereo_analyzer._compute_stereo_depth",
                side_effect=[_depth(1300.0), _depth(1300.0), _depth(1000.0, valid=False)],
            ):
                report = analyze_stereo_capture(manifest_path)

            pipe = report["pipes"][0]
            self.assertEqual(pipe["installation_state"], "UNKNOWN")
            self.assertEqual(pipe["qualified_negative_evidence_capture_ids"], [])

    def test_current_repeated_absence_can_replace_historical_positive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = _manifest(root, count=3, draw_pipe=False)
            # Make only the first pair visibly contain the mesh-bound pipe.
            for role, x1, x2 in (("left", 30, 90), ("right", 15, 75)):
                view = manifest["capture"]["capture_groups"][0]["views"][role]
                image_path = root / view["path"]
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                assert image is not None
                cv2.line(image, (x1, 50), (x2, 50), (0, 0, 255), 9, cv2.LINE_8)
                digest = _write_png(image_path, image)
                view["sha256"] = digest
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch(
                "pipe_twin.stereo_analyzer._compute_stereo_depth",
                side_effect=[_depth(1000.0), _depth(1300.0), _depth(1300.0)],
            ):
                report = analyze_stereo_capture(manifest_path)

            pipe = report["pipes"][0]
            self.assertEqual(pipe["positive_evidence_capture_ids"], ["pair-001"])
            self.assertEqual(pipe["qualified_negative_evidence_capture_ids"], [
                "pair-002",
                "pair-003",
            ])
            self.assertEqual(pipe["installation_state"], "NOT_INSTALLED")

    def test_one_absence_is_unknown_not_not_installed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = _manifest(Path(temporary), count=1, draw_pipe=False)
            with mock.patch(
                "pipe_twin.stereo_analyzer._compute_stereo_depth",
                return_value=_depth(1300.0),
            ):
                report = analyze_stereo_capture(manifest_path)

            pipe = report["pipes"][0]
            self.assertEqual(pipe["installation_state"], "UNKNOWN")
            self.assertEqual(pipe["state_basis"], "INSUFFICIENT_REPEATED_NEGATIVE_EVIDENCE")

    def test_fully_occluded_in_both_views_is_always_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = _manifest(Path(temporary), count=2, draw_pipe=False)
            with mock.patch(
                "pipe_twin.stereo_analyzer._compute_stereo_depth",
                return_value=_depth(800.0),
            ):
                report = analyze_stereo_capture(manifest_path)

            pipe = report["pipes"][0]
            self.assertEqual(pipe["installation_state"], "UNKNOWN")
            self.assertEqual(pipe["reason_codes"], ["FULLY_OCCLUDED_IN_BOTH_VIEWS"])
            self.assertEqual(
                {view["occlusion_state"] for view in pipe["visibility_by_view"].values()},
                {"FULLY_OCCLUDED"},
            )
            for view in pipe["visibility_by_view"].values():
                self.assertFalse(view["assessable"])
                self.assertFalse(view["width_assessable"])
                self.assertEqual(view["width_gate_status"], "OCCLUDED")

    def test_invalid_disparity_and_bad_sync_fail_to_unknown(self) -> None:
        for case in ("depth", "sync"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest_path, manifest = _manifest(root, count=1, draw_pipe=True)
                if case == "sync":
                    manifest["capture"]["capture_groups"][0]["views"]["right"][
                        "captured_at"
                    ] = "2026-09-03T00:00:00.100+08:00"
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    mocked_depth = _depth(1000.0)
                else:
                    mocked_depth = _depth(1000.0, valid=False)
                with mock.patch(
                    "pipe_twin.stereo_analyzer._compute_stereo_depth",
                    return_value=mocked_depth,
                ):
                    report = analyze_stereo_capture(manifest_path)
                self.assertEqual(report["pipes"][0]["installation_state"], "UNKNOWN")

    def test_raw_non_rectified_calibration_is_rejected_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, manifest = _manifest(root, count=1, draw_pipe=True)
            manifest["stereo_calibration"]["rectified"] = False
            with self.assertRaisesRegex(StereoAnalysisError, "rectified must be true"):
                _calibration_from_manifest(manifest["stereo_calibration"])

    def test_stereo_matching_resource_bounds_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, manifest = _manifest(root, count=1, draw_pipe=True)
            manifest["analysis"]["stereo_matching"]["num_disparities"] = 10_000
            with self.assertRaisesRegex(StereoAnalysisError, "num_disparities"):
                _analysis_config(manifest["analysis"])

    def test_overflowing_manifest_matrix_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, manifest = _manifest(root, count=1, draw_pipe=True)
            # A JSON integer can exceed float64 even though it is syntactically
            # valid.  Matrix parsing must turn that into a typed manifest error,
            # not leak OverflowError from NumPy.
            manifest["model"]["pipes"][0]["centerline_world_mm"][0][0] = 10**400
            manifest_path = root / "overflow.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(StereoAnalysisError, "centerline_world_mm"):
                analyze_stereo_capture(manifest_path)

    def test_untextured_stereo_never_becomes_negative_free_space(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path, _ = _manifest(Path(temporary), count=2, draw_pipe=False)

            report = analyze_stereo_capture(manifest_path)

            pipe = report["pipes"][0]
            self.assertEqual(pipe["installation_state"], "UNKNOWN")
            self.assertEqual(pipe["negative_evidence_capture_ids"], [])
            for group in report["capture_audit"]["groups"]:
                self.assertEqual(group["depth_audit"]["status"], "INVALID")

    def test_amodal_mask_is_from_cad_mesh_not_sidecar_centerline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = _manifest(root, count=1, draw_pipe=True)
            with mock.patch(
                "pipe_twin.stereo_analyzer._compute_stereo_depth",
                return_value=_depth(1000.0),
            ):
                baseline = analyze_stereo_capture(manifest_path)

            manifest["model"]["pipes"][0]["centerline_world_mm"] = [
                [-100.0, 10.0, 1000.0],
                [100.0, 10.0, 1000.0],
            ]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with mock.patch(
                "pipe_twin.stereo_analyzer._compute_stereo_depth",
                return_value=_depth(1000.0),
            ):
                shifted_sidecar = analyze_stereo_capture(manifest_path)

            for role in ("left", "right"):
                first = baseline["pipes"][0]["visibility_by_view"][role]
                second = shifted_sidecar["pipes"][0]["visibility_by_view"][role]
                self.assertEqual(first["projection_geometry_source"], "cad_triangle_mesh")
                self.assertEqual(first["amodal_pixels_in_frame"], second["amodal_pixels_in_frame"])

    def test_missing_right_view_is_rejected_before_report_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = _manifest(root, count=1, draw_pipe=True)
            del manifest["capture"]["capture_groups"][0]["views"]["right"]
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report_path = root / "report.json"

            with self.assertRaises(StereoAnalysisError):
                analyze_stereo_capture(manifest_path, report_output_path=report_path)

            self.assertFalse(report_path.exists())

    def test_capture_id_cannot_escape_evidence_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, manifest = _manifest(root, count=1, draw_pipe=True)
            manifest["capture"]["capture_groups"][0]["capture_id"] = "../escape"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaises(StereoAnalysisError):
                analyze_stereo_capture(manifest_path, evidence_dir=root / "evidence")

            self.assertFalse((root / "escape_left_overlay.png").exists())

    def test_report_hardlink_to_manifest_is_rejected_before_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = _manifest(root, count=1, draw_pipe=True)
            report_path = root / "report-hardlink.json"
            try:
                os.link(manifest_path, report_path)
            except OSError as error:  # pragma: no cover - unusual filesystem
                self.skipTest(f"hard links are unavailable: {error}")
            original = manifest_path.read_bytes()

            with self.assertRaisesRegex(StereoAnalysisError, "Path collision"):
                analyze_stereo_capture(manifest_path, report_output_path=report_path)

            self.assertEqual(manifest_path.read_bytes(), original)

    def test_report_and_overlay_contract_is_gui_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path, _ = _manifest(root, count=1, draw_pipe=True)
            report_path = root / "report.json"
            evidence_dir = root / "evidence"
            with mock.patch(
                "pipe_twin.stereo_analyzer._compute_stereo_depth",
                return_value=_depth(1000.0),
            ):
                returned = analyze_stereo_capture(
                    manifest_path,
                    report_output_path=report_path,
                    evidence_dir=evidence_dir,
                )

            persisted = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(returned, persisted)
            self.assertEqual(set(persisted["pipes"][0]["visibility_by_view"]), {"left", "right"})
            self.assertEqual(len(persisted["evidence_files"]), 2)
            for item in persisted["evidence_files"]:
                self.assertTrue(Path(item["path"]).is_file())

    def test_pipe_group2_golden_stereo_is_eight_installed_one_unknown(self) -> None:
        report = analyze_stereo_capture(
            ROOT / "test_model" / "field_stereo_demo_manifest.json"
        )

        self.assertEqual(
            report["counts"],
            {"INSTALLED": 8, "NOT_INSTALLED": 0, "UNKNOWN": 1},
        )
        by_instance = {item["instance_id"]: item for item in report["pipes"]}
        self.assertEqual(by_instance[7]["installation_state"], "UNKNOWN")
        self.assertEqual(
            by_instance[7]["reason_codes"], ["FULLY_OCCLUDED_IN_BOTH_VIEWS"]
        )
        self.assertFalse(by_instance[7]["visibility_by_view"]["left"]["assessable"])
        self.assertEqual(
            by_instance[4]["visibility_by_view"]["left"]["width_gate_status"],
            "OCCLUSION_LIMITED",
        )


if __name__ == "__main__":
    unittest.main()
