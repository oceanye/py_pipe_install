from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from pipe_twin.capture_gui import (
    catalog_from_model,
    create_capture_dataset,
    normalize_capture_time,
    photo_file_time,
)
from pipe_twin.camera_pose import apply_camera_pose
from pipe_twin.cli import build_parser
from pipe_twin.stereo_analyzer import _load_cad_scene


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "test_model" / "field_stereo_demo_manifest.json"
STL_MODEL = ROOT / "test_model" / "管道布置.stl"


class CaptureInputTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        group = self.manifest["capture"]["capture_groups"][0]
        self.arguments = dict(model_path=MANIFEST.parent / self.manifest["model"]["path"],
            pipes=self.manifest["model"]["pipes"], calibration=self.manifest["stereo_calibration"],
            left_path=MANIFEST.parent / group["views"]["left"]["path"], right_path=MANIFEST.parent / group["views"]["right"]["path"],
            left_time="2026-09-07T10:00:00.000+08:00", right_time="2026-09-07T10:00:00.001+08:00", pair_confirmed=True)

    def test_gui_can_start_without_a_manifest(self):
        self.assertIsNone(build_parser().parse_args(["gui"]).manifest)
        parsed = build_parser().parse_args(
            ["inspect-model", str(STL_MODEL), "--stl-unit", "millimeter"]
        )
        self.assertEqual(parsed.stl_unit, "millimeter")

    def test_create_copies_original_assets_and_keeps_model_mapping(self):
        before = hashlib.sha256(self.arguments["left_path"].read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temp:
            path = create_capture_dataset(output_root=Path(temp), **self.arguments)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["model"]["pipes"], self.arguments["pipes"])
            view = manifest["capture"]["capture_groups"][-1]["views"]["left"]
            self.assertEqual(hashlib.sha256((path.parent / view["path"]).read_bytes()).hexdigest(), before)
            self.assertEqual(manifest["stereo_calibration"], self.arguments["calibration"])
        self.assertEqual(hashlib.sha256(self.arguments["left_path"].read_bytes()).hexdigest(), before)

    def test_direct_camera_timestamp_and_device_provenance_are_preserved(self):
        provenance = {
            role: {
                "capture_backend": "OpenCV",
                "capture_layout": "SINGLE_FRAME_SIDE_BY_SIDE",
                "capture_device_index": 0,
                "side_by_side_order": "LEFT_THEN_RIGHT",
                "capture_sync_method": "SAME_UVC_FRAME",
            }
            for role in ("left", "right")
        }
        with tempfile.TemporaryDirectory() as temp:
            path = create_capture_dataset(
                output_root=Path(temp),
                timestamp_sources={
                    "left": "HOST_SYSTEM_CLOCK",
                    "right": "HOST_SYSTEM_CLOCK",
                },
                camera_capture_provenance=provenance,
                **self.arguments,
            )
            manifest = json.loads(path.read_text(encoding="utf-8"))
            views = manifest["capture"]["capture_groups"][-1]["views"]
            for role in ("left", "right"):
                self.assertEqual(views[role]["timestamp_source"], "HOST_SYSTEM_CLOCK")
                self.assertEqual(views[role]["capture_device_index"], 0)
                self.assertEqual(views[role]["capture_sync_method"], "SAME_UVC_FRAME")

    def test_preserve_history_requires_identical_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            path = create_capture_dataset(output_root=Path(temp), previous_manifest=MANIFEST, **self.arguments)
            result = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(result["capture"]["capture_groups"]), 2)
            for group in result["capture"]["capture_groups"]:
                for view in group["views"].values():
                    self.assertTrue((path.parent / view["path"]).exists())
            changed = copy.deepcopy(self.arguments)
            changed["calibration"]["calibration_id"] = "different"
            with self.assertRaisesRegex(ValueError, "同一模型"):
                create_capture_dataset(output_root=Path(temp), previous_manifest=MANIFEST, **changed)

    def test_same_image_unconfirmed_pair_and_date_without_clock_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            for changed in ({"right_path": self.arguments["left_path"]}, {"pair_confirmed": False}, {"left_time": "2026-09-07"}):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    create_capture_dataset(output_root=Path(temp), **(self.arguments | changed))
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_common_local_time_formats_are_normalized_with_a_zone(self):
        local, assumed = normalize_capture_time("2026/9/7 10:00")
        self.assertTrue(assumed)
        parsed = datetime.fromisoformat(local)
        self.assertEqual((parsed.year, parsed.month, parsed.day, parsed.hour), (2026, 9, 7, 10))
        self.assertIsNotNone(parsed.utcoffset())
        explicit, assumed = normalize_capture_time("2026-09-07T10:00:00Z")
        self.assertFalse(assumed)
        self.assertTrue(explicit.endswith("+00:00"))
        self.assertIsNotNone(datetime.fromisoformat(photo_file_time(self.arguments["left_path"])).utcoffset())

        with tempfile.TemporaryDirectory() as temp:
            path = create_capture_dataset(
                output_root=Path(temp),
                **(
                    self.arguments
                    | {
                        "left_time": "2026/9/7 10:00:00",
                        "right_time": "2026年9月7日 10时00分00.001秒",
                    }
                ),
            )
            manifest = json.loads(path.read_text(encoding="utf-8"))
            views = manifest["capture"]["capture_groups"][-1]["views"]
            self.assertEqual(views["left"]["captured_at"][:23], "2026-09-07T10:00:00.000")
            self.assertEqual(views["right"]["captured_at"][:23], "2026-09-07T10:00:00.001")
            self.assertTrue(views["left"]["timestamp_timezone_assumed"])
            self.assertTrue(views["right"]["timestamp_timezone_assumed"])

    def test_model_catalog_design_only_preserves_all_nine_cylinders(self):
        pipes, skipped = catalog_from_model(self.arguments["model_path"])
        self.assertEqual(len(pipes), 9, skipped)
        self.assertFalse(skipped)
        self.assertEqual(len({p["pipe_id"] for p in pipes}), 9)
        self.assertEqual(sorted(round(p["nominal_diameter_mm"]) for p in pipes), [20, 20, 20, 20, 40, 40, 45, 45, 45])

    def test_repository_stl_catalog_finds_twelve_pipe_components(self):
        pipes, skipped = catalog_from_model(STL_MODEL, stl_unit="millimeter")
        self.assertEqual(len(pipes), 12, skipped)
        self.assertFalse(skipped)
        self.assertEqual(len({pipe["cad_object_id"] for pipe in pipes}), 12)
        self.assertTrue(all(pipe["color_srgb"] == "#B0B0B0" for pipe in pipes))
        self.assertTrue(all(pipe["nominal_diameter_mm"] > 0 for pipe in pipes))

    def test_stl_camera_pose_and_capture_package_reload_end_to_end(self):
        pipes, skipped = catalog_from_model(STL_MODEL, stl_unit="millimeter")
        self.assertFalse(skipped)
        calibration = apply_camera_pose(
            self.arguments["calibration"],
            {
                "mode": "positive_z",
                "center_world_mm": [2217.0, 1797.0, 2200.0],
                "yaw_deg": 0.0,
                "pitch_deg": 0.0,
                "roll_deg": 2.5,
                "registration_validated": True,
            },
        )
        arguments = self.arguments | {
            "model_path": STL_MODEL,
            "pipes": pipes,
            "calibration": calibration,
            "stl_unit": "millimeter",
        }
        with tempfile.TemporaryDirectory() as temp:
            path = create_capture_dataset(output_root=Path(temp), **arguments)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["model"]["path"], "model.stl")
            self.assertEqual(manifest["model"]["unit"], "millimeter")
            self.assertEqual(manifest["model"]["source_unit"], "millimeter")
            self.assertEqual(
                manifest["stereo_calibration"]["registration_adjustment"]["roll_deg"],
                2.5,
            )
            scene = _load_cad_scene(path, manifest["model"])
            self.assertEqual(scene.model_format, "stl")
            self.assertEqual(len(scene.pipes), 12)
            self.assertEqual(
                scene.object_binding_validation,
                "STL_COMPONENT_ID_AND_MESH_VALIDATED",
            )


if __name__ == "__main__":
    unittest.main()
