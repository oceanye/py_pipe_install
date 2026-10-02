from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from pipe_twin.capture_gui import (
    MODEL_ANCHOR_VIEWS,
    QrRegistrationDialog,
    StereoCameraDialog,
    _model_anchor_projection,
    _model_anchor_projection_basis,
    _model_surface_orientation,
    _orbit_view_basis,
    _pick_model_anchor_surface,
    _qr_source_image_path,
    catalog_from_model,
    create_capture_dataset,
    field_calibration_problem,
    load_calibration_json,
    normalize_capture_time,
    photo_file_time,
)
from pipe_twin.camera_pose import apply_camera_pose
from pipe_twin.cli import build_parser
from pipe_twin.stereo_analyzer import _load_cad_scene
from pipe_twin.stereo_camera import CapturedStereoPair, LAYOUT_SIDE_BY_SIDE_LR


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

    def test_calibration_loader_rejects_blank_or_directory_with_clear_message(self):
        with self.assertRaisesRegex(ValueError, "请先选择真实双目标定 JSON"):
            load_calibration_json("")
        with self.assertRaisesRegex(ValueError, "不存在或不是文件"):
            load_calibration_json(MANIFEST.parent)

    def test_calibration_loader_accepts_a_full_manifest(self):
        self.assertEqual(
            load_calibration_json(MANIFEST),
            self.manifest["stereo_calibration"],
        )

    def test_field_capture_rejects_demo_or_unvalidated_calibration(self):
        self.assertIn("合成演示", field_calibration_problem(self.arguments["calibration"]))
        real = copy.deepcopy(self.arguments["calibration"])
        real["calibration_id"] = "FIELD-USB-STEREO-001"
        real["validated"] = False
        self.assertIn("validated=true", field_calibration_problem(real))
        real["validated"] = True
        self.assertIsNone(field_calibration_problem(real))

    def test_gui_can_start_without_a_manifest(self):
        self.assertIsNone(build_parser().parse_args(["gui"]).manifest)
        office = build_parser().parse_args(["office-client", "--no-gui"])
        self.assertTrue(office.no_gui)
        self.assertEqual(office.capture_port, 8770)
        self.assertEqual(office.file_port, 8765)
        self.assertFalse(office.require_token)
        remote = build_parser().parse_args(
            ["remote-capture", "--agent-url", "http://127.0.0.1:8770", "--exposure-ms", "33.333333"]
        )
        self.assertIsNone(remote.token_file)
        self.assertAlmostEqual(remote.exposure_ms, 1000 / 30, places=5)
        self.assertFalse(remote.auto_exposure)
        auto = build_parser().parse_args(
            ["remote-capture", "--agent-url", "http://127.0.0.1:8770", "--auto-exposure"]
        )
        self.assertTrue(auto.auto_exposure)
        parsed = build_parser().parse_args(
            ["inspect-model", str(STL_MODEL), "--stl-unit", "millimeter"]
        )
        self.assertEqual(parsed.stl_unit, "millimeter")

    def test_model_anchor_picker_selects_frontmost_surface_and_view_axes(self):
        mesh = SimpleNamespace(
            object_id="pipe-surface",
            color_srgb="#B0B0B0",
            vertices_world_mm=np.asarray(
                [
                    [0.0, 0.0, 0.0],
                    [10.0, 0.0, 0.0],
                    [0.0, 10.0, 0.0],
                    [0.0, 0.0, 5.0],
                    [10.0, 0.0, 5.0],
                    [0.0, 10.0, 5.0],
                ]
            ),
            triangles=np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32),
        )
        scene = SimpleNamespace(objects=(mesh,))
        label = "从上往下（+Z → -Z）"
        projection = _model_anchor_projection(
            scene, width=600, height=400, view_label=label
        )
        expected_world = np.asarray([2.0, 2.0, 5.0])
        click = (
            float(expected_world @ projection["right_world"])
            * projection["scale"]
            + projection["offset_x"],
            -float(expected_world @ projection["up_world"])
            * projection["scale"]
            + projection["offset_y"],
        )
        selected = _pick_model_anchor_surface(projection, click)
        self.assertIsNotNone(selected)
        point, triangle_index = selected
        np.testing.assert_allclose(point, expected_world, atol=1e-8)
        self.assertEqual(triangle_index, 1)
        self.assertEqual(
            MODEL_ANCHOR_VIEWS[label],
            (projection["right_name"], projection["up_name"]),
        )

    def test_orbit_picker_derives_an_arbitrary_orthonormal_surface_basis(self):
        mesh = SimpleNamespace(
            object_id="sloped-surface",
            color_srgb="#B0B0B0",
            vertices_world_mm=np.asarray(
                [[0.0, 0.0, 0.0], [10.0, 0.0, 10.0], [0.0, 10.0, 0.0]]
            ),
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        )
        right_view, up_view = _orbit_view_basis(35.0, 20.0)
        projection = _model_anchor_projection_basis(
            SimpleNamespace(objects=(mesh,)),
            width=600,
            height=400,
            right_world=right_view,
            up_world=up_view,
            zoom=1.5,
        )
        right, up, front = _model_surface_orientation(
            projection, 0, roll_deg=31.0
        )
        np.testing.assert_allclose(np.linalg.norm(right), 1.0, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(up), 1.0, atol=1e-9)
        np.testing.assert_allclose(np.linalg.norm(front), 1.0, atol=1e-9)
        np.testing.assert_allclose(np.dot(right, up), 0.0, atol=1e-9)
        np.testing.assert_allclose(np.cross(right, up), front, atol=1e-9)
        self.assertGreater(abs(float(front[0])), 0.5)
        self.assertGreater(abs(float(front[2])), 0.5)

    def test_blank_qr_source_is_not_reported_as_dot_directory(self):
        with self.assertRaisesRegex(ValueError, "尚未抓拍二维码照片") as caught:
            _qr_source_image_path("")
        self.assertNotIn("不存在：.", str(caught.exception))

    def test_direct_camera_capture_notifies_qr_dialog_with_saved_images(self):
        pair = CapturedStereoPair(
            left=np.zeros((8, 12, 3), dtype=np.uint8),
            right=np.ones((8, 12, 3), dtype=np.uint8),
            left_captured_at="2026-09-11T17:00:00.000+08:00",
            right_captured_at="2026-09-11T17:00:00.001+08:00",
            sync_delta_ms=1.0,
            timestamp_source="HOST_SYSTEM_CLOCK",
            provenance={"left": {}, "right": {}},
        )
        dialog = object.__new__(StereoCameraDialog)
        dialog._opening = False
        dialog.exposure_preset = mock.Mock()
        dialog.exposure_preset.get.return_value = "自动曝光"
        dialog.session = SimpleNamespace(read_pair=mock.Mock(return_value=pair))
        dialog.rectifier = None
        dialog.calibration = SimpleNamespace(
            max_sync_delta_ms=10.0,
            left=SimpleNamespace(width=12, height=8),
        )
        variables = {key: mock.Mock() for key in ("left", "right", "left_time", "right_time")}
        owner = SimpleNamespace(
            fields=variables,
            timestamp_sources={},
            camera_capture_provenance={},
            confirmed=mock.Mock(),
            message=mock.Mock(),
            _persist_profile=mock.Mock(),
        )
        dialog.owner = owner
        dialog.app = SimpleNamespace(messagebox=SimpleNamespace(showerror=mock.Mock()))
        dialog.window = object()
        dialog._layout = lambda: LAYOUT_SIDE_BY_SIDE_LR
        dialog._indices = lambda: (0, None)
        dialog.close = mock.Mock()
        callback_paths = []

        def receive(paths, received_pair):
            self.assertIs(received_pair, pair)
            self.assertTrue(paths["left"].is_file())
            self.assertTrue(paths["right"].is_file())
            callback_paths.append(paths)

        dialog.on_capture = receive
        with tempfile.TemporaryDirectory() as temp, mock.patch(
            "pipe_twin.measurement_gui.OUTPUT_ROOT", Path(temp)
        ):
            dialog.capture()
        self.assertEqual(len(callback_paths), 1)
        variables["left"].set.assert_called_once()
        variables["right"].set.assert_called_once()
        owner.confirmed.set.assert_called_once_with(True)
        self.assertIsNone(owner._persist_profile.call_args.kwargs["camera"]["exposure_ms"])
        dialog.close.assert_called_once()
        dialog.app.messagebox.showerror.assert_not_called()

    def test_qr_print_export_does_not_require_cad_coordinates_first(self):
        dialog = object.__new__(QrRegistrationDialog)
        dialog.marker_id = mock.Mock()
        dialog.marker_id.get.return_value = "PIPE-TWIN-QR-001"
        dialog.marker_edge = mock.Mock()
        dialog.marker_edge.get.return_value = "120"
        dialog.message = mock.Mock()
        dialog.window = object()
        protector = mock.Mock()
        dialog.owner = SimpleNamespace(
            app=SimpleNamespace(
                measurement_panel=SimpleNamespace(_protect_output=protector),
                messagebox=SimpleNamespace(showerror=mock.Mock()),
            )
        )
        with tempfile.TemporaryDirectory() as temp:
            generated = Path(temp) / "qr.png"
            generated.write_bytes(b"png")
            with mock.patch(
                "pipe_twin.capture_gui.write_printable_qr_png",
                return_value=generated,
            ) as writer:
                dialog.export_marker()
        writer.assert_called_once()
        protector.assert_called_once()
        dialog.message.set.assert_called_once()
        dialog.owner.app.messagebox.showerror.assert_not_called()

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
                "capture_exposure_target_ms": 5.0,
                "capture_exposure_ms": 3.90625,
                "capture_exposure_status": "DRIVER_REPORTED",
            }
            for role in ("left", "right")
        }
        field_arguments = copy.deepcopy(self.arguments)
        field_arguments["calibration"]["calibration_id"] = "FIELD-USB-STEREO-001"
        with tempfile.TemporaryDirectory() as temp:
            path = create_capture_dataset(
                output_root=Path(temp),
                timestamp_sources={
                    "left": "HOST_SYSTEM_CLOCK",
                    "right": "HOST_SYSTEM_CLOCK",
                },
                camera_capture_provenance=provenance,
                **field_arguments,
            )
            manifest = json.loads(path.read_text(encoding="utf-8"))
            views = manifest["capture"]["capture_groups"][-1]["views"]
            for role in ("left", "right"):
                self.assertEqual(views[role]["timestamp_source"], "HOST_SYSTEM_CLOCK")
                self.assertEqual(views[role]["capture_device_index"], 0)
                self.assertEqual(views[role]["capture_sync_method"], "SAME_UVC_FRAME")
                self.assertEqual(views[role]["capture_exposure_ms"], 3.90625)
                self.assertEqual(views[role]["capture_exposure_target_ms"], 5.0)
                self.assertEqual(views[role]["capture_exposure_status"], "DRIVER_REPORTED")

        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "合成演示"):
                create_capture_dataset(
                    output_root=Path(temp),
                    camera_capture_provenance=provenance,
                    **self.arguments,
                )

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

    def test_high_resolution_capture_allows_more_than_512_disparities(self):
        pipes, skipped = catalog_from_model(STL_MODEL, stl_unit="millimeter")
        self.assertFalse(skipped)
        with tempfile.TemporaryDirectory() as temp:
            path = create_capture_dataset(
                output_root=Path(temp),
                **(
                    self.arguments
                    | {
                        "model_path": STL_MODEL,
                        "pipes": pipes,
                        "stl_unit": "millimeter",
                    }
                ),
            )
            manifest = json.loads(path.read_text(encoding="utf-8"))
        selected = manifest["analysis"]["stereo_matching"]["num_disparities"]
        self.assertGreater(selected, 512)
        self.assertLess(selected, 1920)
        self.assertEqual(
            manifest["analysis"]["intake_disparity_estimate"][
                "selected_num_disparities"
            ],
            selected,
        )

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
