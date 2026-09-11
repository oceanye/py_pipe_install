from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
import cv2

from pipe_twin.elevation_depth import ElevationDepthError, analyze_elevation_groups
from pipe_twin.stereo_analyzer import StereoDepthResult
from test_stereo_analyzer import _manifest


class _Depth:
    def __init__(self, shape: tuple[int, int], valid: bool = True) -> None:
        self.left_depth_mm = np.full(shape, 1000.0, dtype=np.float32)
        self.right_depth_mm = np.full(shape, 1000.0, dtype=np.float32)
        self.left_valid = np.full(shape, valid, dtype=bool)
        self.right_valid = np.full(shape, valid, dtype=bool)


class _Camera:
    fx = 250.0
    camera_id = "cam"


class _Calibration:
    validated = True
    rectified = True
    baseline_mm = 5.0
    left = right = _Camera()


def _group(image: np.ndarray, name: str, *, valid: bool = True, signature: str | None = None, captured_at: str | None = None) -> dict:
    return {
        "left": image.copy(),
        "right": image.copy(),
        "depth": _Depth(image.shape[:2], valid),
        "pair_healthy": True,
        "capture_id": name,
        "pair_signature": signature or name,
        "captured_at": captured_at,
    }


class ElevationDepthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = [{
            "pipe_id": "P001", "color_srgb": "#FF0000", "nominal_diameter_mm": 20,
            "left_region_px": [5, 5, 10, 10], "right_region_px": [5, 5, 10, 10], "expected_depth_mm": 500,
        }]
        self.image = np.zeros((24, 32, 3), dtype=np.uint8)

    def test_installed_uses_colour_and_stereo_depth_without_registration(self) -> None:
        self.image[5:10, 5:15] = (0, 0, 255)
        spec = [dict(self.spec[0], expected_depth_mm=1000)]
        report = analyze_elevation_groups([_group(self.image, "capture-1")], calibration=_Calibration(), pipe_specs=spec)
        self.assertFalse(report["registration_required"])
        self.assertEqual(report["pipes"][0]["installation_state"], "INSTALLED")

    def test_two_independent_empty_captures_are_not_installed(self) -> None:
        second = self.image.copy(); second[0, 0] = 1
        report = analyze_elevation_groups(
            [_group(self.image, "capture-1", captured_at="2026-01-01T00:00:00+00:00"), _group(second, "capture-2", captured_at="2026-01-01T00:00:02+00:00")],
            calibration=_Calibration(), pipe_specs=self.spec,
        )
        self.assertEqual(report["pipes"][0]["installation_state"], "NOT_INSTALLED")

    def test_repeated_same_image_does_not_count_twice(self) -> None:
        report = analyze_elevation_groups(
            [_group(self.image, "capture-1", signature="same", captured_at="2026-01-01T00:00:00+00:00"),
             _group(self.image, "capture-2", signature="same", captured_at="2026-01-01T00:00:02+00:00")],
            calibration=_Calibration(), pipe_specs=self.spec,
        )
        self.assertEqual(report["pipes"][0]["installation_state"], "UNKNOWN")

    def test_unhealthy_current_pair_invalidates_old_negative_history(self) -> None:
        old = self.image.copy(); old[0, 0] = 1
        failed = self.image.copy(); failed[0, 0] = 2
        report = analyze_elevation_groups(
            [_group(self.image, "capture-1", captured_at="2026-01-01T00:00:00+00:00"),
             _group(old, "capture-2", captured_at="2026-01-01T00:00:02+00:00"),
             _group(failed, "capture-3", valid=False, captured_at="2026-01-01T00:00:04+00:00")],
            calibration=_Calibration(), pipe_specs=self.spec,
        )
        self.assertEqual(report["pipes"][0]["installation_state"], "UNKNOWN")

    def test_missing_reference_depth_cannot_be_negative(self) -> None:
        spec = [dict(self.spec[0])]; spec[0].pop("expected_depth_mm")
        old = self.image.copy(); old[0, 0] = 1
        report = analyze_elevation_groups(
            [_group(self.image, "capture-1", captured_at="2026-01-01T00:00:00+00:00"),
             _group(old, "capture-2", captured_at="2026-01-01T00:00:02+00:00")],
            calibration=_Calibration(), pipe_specs=spec,
        )
        self.assertEqual(report["pipes"][0]["installation_state"], "UNKNOWN")

    def test_negative_capture_count_one_is_rejected(self) -> None:
        with self.assertRaises(ElevationDepthError):
            analyze_elevation_groups([_group(self.image, "capture-1")], calibration=_Calibration(), pipe_specs=self.spec, config={"minimum_negative_captures": 1})

    def test_expected_depth_mismatch_is_not_installed(self) -> None:
        image = self.image.copy(); image[5:10, 5:15] = (0, 0, 255)
        spec = [dict(self.spec[0], expected_depth_mm=200)]
        report = analyze_elevation_groups([_group(image, "capture-1")], calibration=_Calibration(), pipe_specs=spec)
        self.assertEqual(report["pipes"][0]["installation_state"], "UNKNOWN")

    def test_large_left_right_epipolar_offset_is_unknown(self) -> None:
        left = self.image.copy(); right = self.image.copy()
        left[5:10, 5:25] = (0, 0, 255); right[12:17, 5:25] = (0, 0, 255)
        group = _group(left, "capture-1"); group["right"] = right
        specs = [dict(self.spec[0], left_region_px=[5, 5, 20, 10], right_region_px=[5, 12, 20, 10], expected_depth_mm=1000)]
        report = analyze_elevation_groups([group], calibration=_Calibration(), pipe_specs=specs)
        self.assertEqual(report["pipes"][0]["installation_state"], "UNKNOWN")
        evidence = report["pipes"][0]["current_evidence"]
        self.assertTrue(all(evidence[eye]["gates"][gate] for eye in ("left", "right") for gate in ("color", "depth", "joint", "component", "width")))
        self.assertFalse(evidence["left_right_position_consistent"])

    def test_sparse_background_depth_cannot_prove_an_empty_region(self) -> None:
        second = self.image.copy(); second[0, 0] = 1
        groups = [_group(self.image, "capture-1", captured_at="2026-01-01T00:00:00+00:00"),
                  _group(second, "capture-2", captured_at="2026-01-01T00:00:02+00:00")]
        for group in groups:
            for eye in ("left", "right"):
                mask = getattr(group["depth"], f"{eye}_valid")
                mask[:] = False
                mask[5:8, 5:15] = True
        report = analyze_elevation_groups(groups, calibration=_Calibration(), pipe_specs=self.spec)
        row = report["pipes"][0]
        self.assertEqual(row["installation_state"], "UNKNOWN")
        self.assertTrue(row["current_evidence"]["left"]["gates"]["depth"])
        self.assertAlmostEqual(row["current_evidence"]["left"]["free_space_fraction"], .3)

    def test_foreground_depth_stops_prior_empty_state(self) -> None:
        second = self.image.copy(); second[0, 0] = 1
        front = self.image.copy(); front[0, 0] = 2
        groups = [_group(self.image, "capture-1", captured_at="2026-01-01T00:00:00+00:00"),
                  _group(second, "capture-2", captured_at="2026-01-01T00:00:02+00:00"),
                  _group(front, "capture-3", captured_at="2026-01-01T00:00:04+00:00")]
        groups[-1]["depth"].left_depth_mm[:] = 300
        groups[-1]["depth"].right_depth_mm[:] = 300
        report = analyze_elevation_groups(groups, calibration=_Calibration(), pipe_specs=self.spec)
        row = report["pipes"][0]
        self.assertEqual(row["installation_state"], "UNKNOWN")
        self.assertTrue(row["current_evidence"]["left"]["foreground_occlusion"])

    def test_interrupted_negative_history_is_not_combined(self) -> None:
        first = self.image.copy(); first[0, 0] = 1
        middle = self.image.copy(); middle[5:10, 5:15] = (0, 0, 255); middle[0, 0] = 2
        latest = self.image.copy(); latest[0, 0] = 3
        times = ["2026-01-01T00:00:00+00:00", "2026-01-01T00:00:02+00:00", "2026-01-01T00:00:04+00:00"]
        groups = [_group(first, "capture-1", captured_at=times[0]), _group(middle, "capture-2", captured_at=times[1]), _group(latest, "capture-3", captured_at=times[2])]
        report = analyze_elevation_groups(groups, calibration=_Calibration(), pipe_specs=self.spec)
        self.assertEqual(report["pipes"][0]["installation_state"], "UNKNOWN")

    def test_manifest_entrypoint_uses_real_snapshots_and_writes_evidence(self) -> None:
        from pipe_twin.stereo_analyzer import analyze_stereo_capture
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, payload = _manifest(root, count=1, draw_pipe=True)
            payload["analysis"] = {
                "mode": "elevation_depth",
                "stereo_matching": {"num_disparities": 32, "block_size": 5},
                "elevation_depth": {"pipes": [{
                    "pipe_id": "P001", "color_srgb": "#FF0000", "nominal_diameter_mm": 30,
                    "left_region_px": [20, 40, 80, 20], "right_region_px": [5, 40, 80, 20],
                    "expected_depth_mm": 500,
                }]},
            }
            manifest_path.write_text(__import__("json").dumps(payload), encoding="utf-8")
            depth = StereoDepthResult(np.full((100, 120), 1000, np.float32), np.full((100, 120), 1000, np.float32),
                                      np.ones((100, 120), bool), np.ones((100, 120), bool), {"status": "VALID"})
            report_path = root / "report.json"; evidence = root / "evidence"
            with mock.patch("pipe_twin.stereo_analyzer._compute_stereo_depth", return_value=depth):
                report = analyze_stereo_capture(manifest_path, report_output_path=report_path, evidence_dir=evidence)
            self.assertEqual(report["mode"], "elevation_depth")
            self.assertTrue(report_path.exists())
            self.assertEqual(len(report["evidence_files"]), 2)
            self.assertTrue(all(Path(item).exists() for item in report["evidence_files"]))

    def test_create_dataset_to_router_real_sgbm_installed(self) -> None:
        """Exercise the production dataset writer and SGBM depth path together."""
        from pipe_twin.elevation_dataset import create_elevation_dataset
        from pipe_twin.stereo_analyzer import analyze_stereo_capture
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); _, payload = _manifest(root, count=1, draw_pipe=False)
            rng = np.random.default_rng(1)
            left = rng.integers(30, 220, (100, 120, 3), dtype=np.uint8)
            right = np.roll(left, -15, axis=1)
            for y in range(40, 60):
                for x in range(20, 100):
                    left[y, x] = (0, 0, 200 + ((x * 7 + y * 13) % 56))
            for y in range(40, 60):
                for x in range(5, 85):
                    right[y, x] = left[y, x + 15]
            left_path = root / "real-left.png"; right_path = root / "real-right.png"
            cv2.imwrite(str(left_path), left); cv2.imwrite(str(right_path), right)
            specs = [{"pipe_id": "P001", "color_srgb": "#FF0000", "nominal_diameter_mm": 67,
                      "left_region_px": [20, 40, 80, 20], "right_region_px": [5, 40, 80, 20],
                      "expected_depth_mm": 1000}]
            manifest_path = create_elevation_dataset(
                output_root=root / "packages", calibration=payload["stereo_calibration"],
                left_path=left_path, right_path=right_path,
                left_time="2026-01-01T00:00:00+00:00", right_time="2026-01-01T00:00:00+00:00",
                pipe_specs=specs, pair_confirmed=True)
            package = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
            package["analysis"]["stereo_matching"] = {"num_disparities": 32, "block_size": 5}
            manifest_path.write_text(__import__("json").dumps(package), encoding="utf-8")
            report = analyze_stereo_capture(manifest_path)
            self.assertEqual(report["capture_audit"]["groups"][0]["pair_healthy"], True)
            self.assertEqual(report["pipes"][0]["installation_state"], "INSTALLED")
            self.assertAlmostEqual(report["pipes"][0]["current_evidence"]["left"]["target_depth_median_mm"], 1000.0, delta=10.0)

    def test_manifest_report_cannot_overwrite_input(self) -> None:
        from pipe_twin.elevation_depth import analyze_elevation_depth_manifest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); manifest_path, payload = _manifest(root, count=1, draw_pipe=False)
            payload["analysis"] = {"mode": "elevation_depth", "elevation_depth": {"pipes": [{
                "pipe_id": "P001", "color_srgb": "#FF0000", "nominal_diameter_mm": 30,
                "left_region_px": [20, 40, 80, 20], "right_region_px": [5, 40, 80, 20], "expected_depth_mm": 500,
            }]}}
            manifest_path.write_text(__import__("json").dumps(payload), encoding="utf-8")
            with self.assertRaises(ElevationDepthError):
                analyze_elevation_depth_manifest(manifest_path, report_output_path=manifest_path)
            model_path = root / payload["model"]["path"]
            original = model_path.read_bytes()
            with self.assertRaises(ElevationDepthError):
                analyze_elevation_depth_manifest(manifest_path, report_output_path=model_path)
            self.assertEqual(model_path.read_bytes(), original)

    def test_occluded_or_invalid_depth_is_unknown(self) -> None:
        image = self.image.copy()
        image[5:15, 5:15] = (0, 0, 255)
        report = analyze_elevation_groups([_group(image, "capture-1", valid=False)], calibration=_Calibration(), pipe_specs=self.spec)
        self.assertEqual(report["pipes"][0]["installation_state"], "UNKNOWN")

    def test_duplicate_regions_are_rejected_by_the_contract(self) -> None:
        bad = [dict(self.spec[0], pipe_id="P001")]
        with self.assertRaises(ElevationDepthError):
            analyze_elevation_groups([_group(self.image, "capture-1")], calibration=_Calibration(), pipe_specs=self.spec + bad)

    def test_overlapping_same_colour_regions_are_unknown(self) -> None:
        image = self.image.copy()
        image[5:15, 5:15] = (0, 0, 255)
        second = dict(self.spec[0], pipe_id="P002", left_region_px=[6, 5, 10, 10], right_region_px=[6, 5, 10, 10])
        report = analyze_elevation_groups([_group(image, "capture-1")], calibration=_Calibration(), pipe_specs=[self.spec[0], second])
        self.assertTrue(all(row["identity_ambiguous"] for row in report["pipes"]))
        self.assertTrue(all(row["installation_state"] == "UNKNOWN" for row in report["pipes"]))


if __name__ == "__main__":
    unittest.main()
