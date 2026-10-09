from __future__ import annotations

import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from pipe_twin.dxf_elevation import catalog_from_dxf
from pipe_twin.elevation_gui import ElevationCaptureDialog, ZoneEditorDialog, image_region
from test_elevation_dataset import _calibration


class ElevationRegionTests(unittest.TestCase):
    def test_zoomed_reversed_drag_uses_original_pixels(self):
        self.assertEqual(image_region((210, 130), (30, 50), (2, 10, 10), (640, 480)), [10, 20, 90, 40])
        self.assertEqual(image_region((-100, -100), (50, 50), (1, 0, 0), (640, 480)), [0, 0, 50, 50])
        self.assertIsNone(image_region((5, 5), (8, 8), (1, 0, 0), (640, 480)))


class ElevationGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tkinter as tk
        from tkinter import ttk
        try:
            cls.root = tk.Tk()
        except tk.TclError as error:
            raise unittest.SkipTest(f"Tk display unavailable: {error}") from error
        cls.root.withdraw()
        cls.tk, cls.ttk = tk, ttk

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name) / "workbench"
        self.app = SimpleNamespace(root=self.root, tk=self.tk, ttk=self.ttk, messagebox=mock.Mock(), filedialog=mock.Mock())
        self.profile_patch = mock.patch("pipe_twin.workbench_profile.load_profile", return_value=(None, None))
        self.profile_patch.start()
        self.dialog = ElevationCaptureDialog(self.app, output_root=self.output, restore=False)
        self.dialog.window.withdraw()
        self.dialog.calibration_override = _calibration()
        model = Path(__file__).resolve().parents[1] / "test_model" / "管道布置.dxf"
        self.dialog.load_model(model)
        self.spec = copy.deepcopy(self.dialog.pipes[0])
        for role, value in (("left", 50), ("right", 60)):
            path = Path(self.tmp.name) / f"{role}.png"
            cv2.imwrite(str(path), np.full((480, 640, 3), value, np.uint8))
            self.dialog.fields[role].set(str(path))
            self.dialog.fields[f"{role}_time"].set("2026-09-12T10:00:00+08:00")
        self.dialog.load_images()
        self.dialog.refresh_table()
        self.dialog.confirmed.set(True)

    def tearDown(self):
        self.dialog.close()
        self.root.update()
        self.profile_patch.stop()
        self.tmp.cleanup()

    def finish_worker(self):
        deadline = time.monotonic() + 15
        while self.dialog.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertFalse(self.dialog.busy, "GUI worker did not finish")
        self.app.messagebox.showerror.assert_not_called()

    def test_save_reopen_and_repeat_save_without_json_editing(self):
        self.dialog.submit(False)
        self.finish_worker()
        saved = self.dialog.last_manifest
        self.assertTrue(saved.is_file())
        payload = json.loads(saved.read_text(encoding="utf-8"))
        self.assertEqual(payload["analysis"]["mode"], "elevation_auto")
        self.assertTrue(payload["model"]["path"].endswith(".dxf"))
        self.assertFalse(payload["stereo_calibration"]["registration_validated"])
        self.dialog.load_session(saved)
        self.assertNotIn("left_region_px", self.dialog.pipes[0])
        self.assertFalse(self.dialog.confirmed.get())
        self.dialog.confirmed.set(True)
        self.dialog.submit(False)
        self.finish_worker()
        repeated = json.loads(self.dialog.last_manifest.read_text(encoding="utf-8"))
        self.assertEqual(len(repeated["capture"]["capture_groups"]), 1)
        self.assertNotIn("已开始新历史", self.dialog.message.get())

    def test_saved_scene_runs_real_analysis_and_renders_unknown_for_blank_images(self):
        self.dialog.submit(True)
        self.finish_worker()
        self.assertEqual(self.dialog.results["P001"]["installation_state"], "UNKNOWN")
        self.assertIn("不确定 12", self.dialog.summary.get())
        self.assertTrue((self.dialog.last_manifest.parent / "report.json").is_file())
        self.assertGreaterEqual(len(self.dialog.report["evidence_files"]), 2)

    def test_matching_controls_roundtrip_and_preserve_other_settings(self):
        self.dialog.analysis_settings = {"minimum_depth_mm": 350, "stereo_matching": {"block_size": 7}}
        self.dialog.disparity_count.set("256")
        self.dialog.matching_preset.set("弱光降噪")
        self.dialog.submit(False)
        self.finish_worker()
        saved = self.dialog.last_manifest
        self.dialog.disparity_count.set("64")
        self.dialog.load_session(saved)
        self.assertEqual(self.dialog.disparity_count.get(), "256")
        self.assertEqual(self.dialog.matching_preset.get(), "弱光降噪")
        self.assertEqual(self.dialog.analysis_settings["minimum_depth_mm"], 350)
        self.assertEqual(self.dialog.analysis_settings["stereo_matching"]["block_size"], 7)
        self.dialog.confirmed.set(True)
        self.dialog.submit(False)
        self.finish_worker()
        analysis = json.loads(self.dialog.last_manifest.read_text(encoding="utf-8"))["analysis"]
        self.assertEqual(analysis["stereo_matching"]["num_disparities"], 256)
        self.assertEqual(analysis["minimum_depth_mm"], 350)

    def test_measurement_detail_opens_without_inventing_field_dimensions(self):
        self.dialog.show_measurement()
        self.root.update_idletasks()
        dialogs = [w for w in self.dialog.window.winfo_children() if w.winfo_class() == "Toplevel"]
        self.assertTrue(dialogs)
        widget = next(w for w in dialogs[-1].winfo_children() if w.winfo_class() == "Text")
        self.assertIn("未获得可靠实测", widget.get("1.0", "end"))
        dialogs[-1].destroy()

    def test_external_photo_change_is_rejected_and_old_result_removed(self):
        self.dialog.report = {"old": True}
        self.dialog.results = {"P001": {"installation_state": "INSTALLED"}}
        Path(self.dialog.fields["left"].get()).write_bytes(b"externally changed")
        self.dialog.submit(False)
        self.assertFalse(self.dialog.busy)
        self.assertIsNone(self.dialog.report)
        self.assertFalse(self.dialog.results)
        self.app.messagebox.showerror.assert_called_once()
        self.assertIn("预览后已改变", self.dialog.message.get())

    def test_first_run_stl_catalog_and_model_id_preview(self):
        model = Path(__file__).resolve().parents[1] / "test_model" / "管道布置.stl"
        self.dialog.load_model(model)
        self.assertTrue(self.dialog.pipes)
        self.assertTrue(self.dialog.selected_id())
        self.assertTrue(all(row.get("color_source") == "stl_synthetic_by_diameter" for row in self.dialog.pipes))
        colors_by_diameter = {}
        for row in self.dialog.pipes:
            diameter = round(row["nominal_diameter_mm"], 1)
            colors_by_diameter.setdefault(diameter, row["color_srgb"])
            self.assertEqual(row["color_srgb"], colors_by_diameter[diameter])
        self.assertTrue(all("left_region_px" not in row for row in self.dialog.pipes))
        self.dialog.show_catalog()
        self.root.update_idletasks()

    def test_stl_zone_candidates_can_be_bound_confirmed_and_saved(self):
        model = Path(__file__).resolve().parents[1] / "test_model" / "管道布置.stl"
        self.dialog.load_model(model)
        editor = ZoneEditorDialog(self.dialog)
        editor.window.withdraw()
        try:
            editor._auto_propose()
            self.assertEqual(len(editor.zones), 1)
            self.assertFalse(editor.zones[0]["confirmed"])
            self.assertFalse(editor.zones[0]["enabled"])
            editor.listbox.selection_set(0)
            editor._selected()
            editor.pending_rect = [0, 0, 640, 480]
            editor._bind_pending()
            self.assertEqual(editor.zones[0]["roi_source"], "user")
            editor._confirm()
            self.assertTrue(editor.zones[0]["confirmed"])
            self.assertTrue(editor.zones[0]["enabled"])
            editor._auto_propose()
            self.assertEqual(len(editor.zones), 1)
            self.assertTrue(editor.zones[0]["enabled"])
            editor._save()
            self.assertEqual(self.dialog.zone_settings["scope"], "zones")
            self.assertEqual(self.dialog.zone_settings["zones"][0]["model_pipe_ids"],
                             sorted(spec["pipe_id"] for spec in self.dialog.pipes))
        finally:
            if editor.window.winfo_exists():
                editor.window.destroy()

    def test_auto_mode_is_roi_free_and_direction_is_normalized(self):
        model = Path(__file__).resolve().parents[1] / "test_model" / "管道布置.stl"
        self.dialog.load_model(model)
        # The production workflow is automatic; model geometry owns the pipe
        # catalogue and the analyzer owns photo regions.
        self.dialog.pipes[0]["left_region_px"] = [20, 20, 20, 20]
        self.assertTrue(all("left_region_px" not in row for row in self.dialog.pipes[1:]))
        self.dialog._set_axis_world([0.0, 0.0, 4.0])
        self.assertEqual(self.dialog.registration_settings["axis_world"], [0.0, 0.0, 1.0])
        self.assertEqual(self.dialog.registration_settings["anchors"], {})

    def test_auto_canvas_draws_unmatched_observation_overlay(self):
        self.dialog.report = {
            "registration": {"status": "INSUFFICIENT_OBSERVATIONS"},
            "local_surface": {"observations": [{
                "observation_id": "OBS-0007", "left_region_px": [24, 30, 80, 42],
                "right_region_px": [20, 30, 78, 42], "color_srgb": "#00FF00",
            }]},
        }
        self.dialog.views["left"].set_image(np.zeros((120, 160, 3), dtype=np.uint8))
        items = self.dialog.views["left"].canvas.find_all()
        rectangles = [item for item in items if self.dialog.views["left"].canvas.type(item) == "rectangle"]
        labels = [self.dialog.views["left"].canvas.itemcget(item, "text") for item in items if self.dialog.views["left"].canvas.type(item) == "text"]
        self.assertTrue(rectangles)
        self.assertIn("OBS-0007", labels)

    def test_model_viewer_keeps_axis_in_stl_coordinates(self):
        model = Path(__file__).resolve().parents[1] / "test_model" / "管道布置.stl"
        self.dialog.load_model(model)
        self.dialog.report = {"registration": {"status": "MATCHED", "rotation_model_to_camera": [[1, 0, 0], [0, -1, 0], [0, 0, -1]], "translation_model_to_camera_mm": [0, 0, 1000]}, "local_surface": {"point_cloud": {"points_camera_mm": [[0, 0, 1000]], "colors_srgb": ["#FFFFFF"]}}}
        self.dialog.registration_settings["axis_world"] = [0, 0, 1]
        self.dialog.open_model_viewer()
        viewer = self.dialog.model_viewer
        self.assertTrue(np.allclose(viewer.display_axis, [0, 0, -1]))
        viewer.direction_mode.set("Z")
        viewer.apply_direction()
        self.assertTrue(np.allclose(self.dialog.registration_settings["axis_world"], [0, 0, 1]))

    def test_restore_keeps_verified_registration_settings_after_photo_load(self):
        manifest = {"model": {"source_unit": "millimeter", "path": None}, "capture": {"capture_groups": [{"views": {"left": {"timestamp_source": "MANIFEST_OPERATOR_CONFIRMED"}, "right": {"timestamp_source": "MANIFEST_OPERATOR_CONFIRMED"}}}]}}
        loaded = {"manifest": manifest, "mode": "elevation_auto", "registration_settings": {"axis_world": [0, 0, 1], "anchors": {}}, "calibration": _calibration(), "pipe_specs": [copy.deepcopy(self.spec)], "left_path": Path(self.tmp.name) / "left.png", "right_path": Path(self.tmp.name) / "right.png", "left_time": "2026-09-12T10:00:00+08:00", "right_time": "2026-09-12T10:00:00+08:00", "rectification_recipe": None}
        with mock.patch("pipe_twin.elevation_dataset.load_elevation_dataset", return_value=loaded), mock.patch.object(self.dialog, "load_images"):
            self.dialog.load_session(Path(self.tmp.name) / "manifest.json")
        self.assertEqual(self.dialog.registration_settings["anchors"], {})

    def test_calibration_change_requires_new_images_and_regions(self):
        self.dialog.fields["calibration"].set("new-calibration.json")
        self.assertIsNone(self.dialog.calibration_override)
        self.assertFalse(self.dialog.confirmed.get())
        self.assertFalse(self.dialog.image_hashes)
        self.assertNotIn("left_region_px", self.dialog.pipes[0])
        self.assertIsNone(self.dialog.views["left"].image)

    def test_missing_eye_provides_actionable_error(self):
        self.dialog.views["right"].set_image(None)
        self.dialog.submit(False)
        self.app.messagebox.showerror.assert_called_once()
        self.assertIn("完整的左右照片", self.dialog.message.get())

    def test_camera_preferences_keep_imported_rectification_recipe(self):
        recipe = {"calibration_id": "imported-bundle"}
        self.dialog.profile = {"rectification_recipe": recipe}
        with mock.patch("pipe_twin.workbench_profile.update_profile") as persist:
            self.dialog._persist_profile(("camera",), camera={"left_index": 2})
        persist.assert_called_once_with({"camera": {"left_index": 2}})
        self.assertEqual(self.dialog.profile["rectification_recipe"], recipe)

    def test_quality_check_operates_without_observations_and_preserves_analysis_regions(self):
        original = copy.deepcopy(self.dialog.pipes)
        self.dialog.open_quality_check()
        quality = self.dialog.quality_dialog
        quality.window.withdraw()
        quality.set_region("left", [20, 20, 40, 40])
        self.assertEqual(quality.pipes[0]["metrics"]["summary"]["status"], "INCOMPLETE")
        quality.set_region("right", [18, 20, 40, 40])
        self.assertEqual(quality.pipes[0]["metrics"]["summary"]["status"], "COMPLETE")
        self.assertEqual(self.dialog.pipes, original)
        self.assertFalse(self.dialog.results)
        saved = quality.record_path()
        self.assertTrue(saved.is_file())
        record = json.loads(saved.read_text(encoding="utf-8"))
        self.assertEqual(record["photo_sha256"], self.dialog.image_hashes)
        quality.close()
        self.dialog.open_quality_check()
        restored = self.dialog.quality_dialog
        restored.window.withdraw()
        self.assertEqual(restored.pipes[0]["left_region_px"], [20, 20, 40, 40])
        self.assertEqual(restored.pipes[0]["metrics"]["summary"]["status"], "COMPLETE")

    def test_quality_check_clears_changed_photos_and_retains_old_record(self):
        self.dialog.open_quality_check()
        quality = self.dialog.quality_dialog
        quality.window.withdraw()
        for role in ("left", "right"):
            quality.set_region(role, [20, 20, 40, 40])
        saved = quality.record_path()
        original_record = saved.read_bytes()
        cv2.imwrite(self.dialog.fields["left"].get(), np.full((480, 640, 3), 180, np.uint8))
        self.dialog.load_images()
        self.assertNotEqual(quality.record_path(), saved)
        self.assertNotIn("left_region_px", quality.pipes[0])
        self.assertIsNone(quality.pipes[0]["metrics"]["summary"]["color_coverage_percent"])
        self.assertEqual(saved.read_bytes(), original_record)

    def test_quality_mouse_selection_and_matching_policy_changes(self):
        self.dialog.open_quality_check()
        quality = self.dialog.quality_dialog
        quality.window.withdraw()
        quality.set_color("#505050")
        for role in ("left", "right"):
            view = quality.views[role]
            self.assertTrue(view.canvas.bind("<ButtonPress-1>"))
            view.transform = (1.0, 0.0, 0.0)
            view.start(SimpleNamespace(x=20, y=20))
            view.move(SimpleNamespace(x=60, y=60))
            view.finish(SimpleNamespace(x=60, y=60))
            self.assertEqual(quality.pipes[0][f"{role}_region_px"], [20, 20, 40, 40])
        self.dialog.color_delta_lab.set("1")
        self.assertEqual(quality.pipes[0]["metrics"]["summary"]["color_coverage_percent"], 0)
        self.dialog.color_delta_lab.set("100")
        self.assertEqual(quality.pipes[0]["metrics"]["summary"]["color_coverage_percent"], 100)
        self.assertIn("筛选关闭", quality.policy_status.get())
        quality.save()
        record = json.loads(quality.record_path().read_text(encoding="utf-8"))
        self.assertEqual(record["matching_context"]["color_delta_lab_used"], 100)
        self.dialog.color_delta_lab.set("")
        self.assertIsNone(quality.pipes[0]["metrics"]["summary"]["color_coverage_percent"])
        self.assertIn("请修正", quality.policy_status.get())
        self.dialog.color_delta_lab.set("45")
        self.dialog.local_observation_label.set("仅双目深度几何")
        self.assertIn("仅使用深度几何", quality.policy_status.get())

    def test_quality_color_application_invalidates_old_result_only_on_explicit_apply(self):
        originals = copy.deepcopy(self.dialog.pipes)
        self.dialog.open_quality_check()
        quality = self.dialog.quality_dialog
        quality.window.withdraw()
        self.dialog.report = {"old": True}
        self.dialog.results = {"P001": {"installation_state": "INSTALLED"}}
        quality.set_color("#ABCD12")
        self.assertEqual(self.dialog.pipes[0]["color_srgb"], originals[0]["color_srgb"])
        self.assertTrue(self.dialog.results)
        quality.apply_color()
        for row, old in zip(self.dialog.pipes, originals):
            self.assertEqual(row["color_srgb"], "#ABCD12" if abs(row["nominal_diameter_mm"] - self.spec["nominal_diameter_mm"]) <= 0.2 else old["color_srgb"])
        self.assertFalse(self.dialog.color_filter_enabled.get())
        self.assertFalse(self.dialog.results)
        self.assertIsNone(self.dialog.report)

    def test_live_dialogs_use_basic_owner_and_close_with_scene(self):
        self.dialog.open_calibration()
        wizard = self.dialog.calibration_dialog
        self.assertIn("基础立面模式标定后可直接抓拍", wizard.message.get())
        self.dialog.capture_camera()
        camera = self.dialog.camera_dialog
        self.assertIs(camera.owner, self.dialog)
        self.assertIsNone(camera.session)
        self.dialog.close()
        self.assertFalse(wizard.window.winfo_exists())
        self.assertFalse(camera.window.winfo_exists())
        # tearDown owns the final root destruction.
        self.dialog = ElevationCaptureDialog(self.app, output_root=self.output, restore=False)
        self.dialog.window.withdraw()


if __name__ == "__main__":
    unittest.main()
