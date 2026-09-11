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

from pipe_twin.elevation_gui import ElevationCaptureDialog, image_region
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
        self.spec = {"pipe_id": "P001", "nominal_diameter_mm": 20.0, "color_srgb": "#FF0000",
                     "left_region_px": [200, 100, 100, 40], "right_region_px": [170, 100, 100, 40],
                     "expected_depth_mm": 1000.0, "axis": "horizontal"}
        for role, value in (("left", 50), ("right", 60)):
            path = Path(self.tmp.name) / f"{role}.png"
            cv2.imwrite(str(path), np.full((480, 640, 3), value, np.uint8))
            self.dialog.fields[role].set(str(path))
            self.dialog.fields[f"{role}_time"].set("2026-09-12T10:00:00+08:00")
        self.dialog.load_images()
        self.dialog.pipes = [copy.deepcopy(self.spec)]
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
        self.assertEqual(payload["analysis"]["mode"], "elevation_depth")
        self.assertNotIn("path", payload["model"])
        self.assertFalse(payload["stereo_calibration"]["registration_validated"])
        self.dialog.clear_regions()
        self.dialog.load_session(saved)
        self.assertEqual(self.dialog.pipes[0]["left_region_px"], self.spec["left_region_px"])
        self.assertEqual(self.dialog.pipes[0]["expected_depth_mm"], 1000)
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
        self.assertIn("不确定 1", self.dialog.summary.get())
        self.assertTrue((self.dialog.last_manifest.parent / "report.json").is_file())
        self.assertEqual(len(self.dialog.report["evidence_files"]), 2)

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

    def test_region_change_discards_reference_and_stale_worker_result(self):
        self.dialog.report = {"old": True}
        self.dialog.results = {"P001": {"installation_state": "INSTALLED"}}
        generation = self.dialog.generation
        self.dialog.set_region("left", [210, 100, 100, 40])
        self.assertNotIn("expected_depth_mm", self.dialog.pipes[0])
        self.assertIsNone(self.dialog.report)
        self.dialog.messages.put((generation, "ok", (Path("stale.json"), {"pipes": [], "counts": {}}, False)))
        self.dialog.window.after_cancel(self.dialog.poll_id)
        self.dialog.poll()
        self.assertIsNone(self.dialog.last_manifest)
        self.assertIn("丢弃旧结果", self.dialog.message.get())

    def test_first_run_stl_catalog_and_model_id_preview(self):
        model = Path(__file__).resolve().parents[1] / "test_model" / "管道布置.stl"
        self.dialog.load_model(model)
        self.assertTrue(self.dialog.pipes)
        self.assertTrue(self.dialog.selected_id())
        self.assertTrue(all("left_region_px" not in row for row in self.dialog.pipes))
        self.dialog.show_catalog()
        self.root.update_idletasks()

    def test_calibration_change_requires_new_images_and_regions(self):
        self.dialog.fields["calibration"].set("new-calibration.json")
        self.assertIsNone(self.dialog.calibration_override)
        self.assertFalse(self.dialog.confirmed.get())
        self.assertFalse(self.dialog.image_hashes)
        self.assertNotIn("left_region_px", self.dialog.pipes[0])
        self.assertNotIn("expected_depth_mm", self.dialog.pipes[0])
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

    def test_live_dialogs_use_basic_owner_and_close_with_scene(self):
        self.dialog.open_calibration()
        wizard = self.dialog.calibration_dialog
        self.assertIn("基础模式标定后可直接抓拍", wizard.message.get())
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
