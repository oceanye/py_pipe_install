from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from pipe_twin.calibration_adapter import adapt_opencv_stereo_calibration
from pipe_twin.elevation_dataset import create_elevation_dataset, elevation_history_compatible, load_elevation_dataset


def _calibration() -> dict:
    source = {
        "K1": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        "K2": [[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]],
        "D1": [0.0] * 5, "D2": [0.0] * 5, "image_size": [640, 480],
        "R": np.eye(3).tolist(), "T": [-0.06, 0.0, 0.0], "translation_unit": "m",
        "left_camera_pose": {"rotation_world_to_camera": np.eye(3).tolist(), "center_world_mm": [0, 0, 0]},
    }
    return adapt_opencv_stereo_calibration(source, calibration_id="FIELD-USB-001", validated=True, registration_validated=False)


def _recipe(calibration: dict) -> dict:
    k = calibration["left_camera"]["K"]
    p1 = [row[:] + [0.0] for row in k]
    p2 = [row[:] + [0.0] for row in k]
    p2[0][3] = -float(k[0][0]) * 60.0
    return {"calibration_id": calibration["calibration_id"], "K1": k, "D1": [0.0] * 5,
            "K2": k, "D2": [0.0] * 5, "R1": np.eye(3).tolist(), "R2": np.eye(3).tolist(),
            "P1": p1, "P2": p2, "image_width_px": 640, "image_height_px": 480,
            "output_width_px": 640, "output_height_px": 480, "alpha": 0.0,
            "created_at": "2026-09-12T00:00:00+08:00", "definition": "test"}


class ElevationDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.left = root / "left.png"; self.right = root / "right.png"
        left = np.zeros((480, 640, 3), np.uint8); right = left.copy(); right[:, :, 2] = 32
        cv2.imwrite(str(self.left), left); cv2.imwrite(str(self.right), right)
        self.specs = [{"pipe_id": "P1", "color_srgb": "#FF0000", "nominal_diameter_mm": 20,
                       "left_region_px": [20, 20, 80, 40], "right_region_px": [18, 20, 80, 40], "axis": "horizontal"}]

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _create(self, **kwargs) -> Path:
        calibration = kwargs.pop("calibration", _calibration())
        return create_elevation_dataset(output_root=Path(self.tmp.name) / "out", calibration=calibration,
            left_path=self.left, right_path=self.right, left_time="2026-09-12T10:00:00+08:00",
            right_time="2026-09-12T10:00:00.001+08:00", pipe_specs=self.specs, pair_confirmed=True, **kwargs)

    def test_create_and_restore_without_cad_or_registration(self):
        path = self._create()
        loaded = load_elevation_dataset(path)
        self.assertEqual(loaded["manifest"]["analysis"]["mode"], "elevation_depth")
        self.assertFalse(loaded["calibration"]["registration_validated"])
        self.assertEqual(loaded["pipe_specs"][0]["pipe_id"], "P1")
        self.assertEqual(loaded["left_path"].name, "left.png")

    def test_rectification_recipe_is_persisted_and_restored(self):
        calibration = _calibration(); recipe = _recipe(calibration)
        path = create_elevation_dataset(output_root=Path(self.tmp.name) / "recipe", calibration=calibration,
            rectification_recipe=recipe, left_path=self.left, right_path=self.right,
            left_time="2026-09-12T10:00:00+08:00", right_time="2026-09-12T10:00:00.001+08:00",
            pipe_specs=self.specs, pair_confirmed=True)
        loaded = load_elevation_dataset(path)
        self.assertEqual(loaded["rectification_recipe"]["calibration_id"], "FIELD-USB-001")

    def test_live_camera_exposure_provenance_survives_package_roundtrip(self):
        for exposure in (
            {"capture_exposure_status": "AUTO"},
            {"capture_exposure_status": "DRIVER_REPORTED", "capture_exposure_target_ms": 5.0,
             "capture_exposure_ms": 3.90625},
        ):
            with self.subTest(exposure=exposure):
                provenance = {role: {"capture_device_index": 0,
                                     "capture_sync_method": "SAME_UVC_FRAME", **exposure}
                              for role in ("left", "right")}
                path = self._create(camera_capture_provenance=provenance)
                loaded = load_elevation_dataset(path)
                views = loaded["manifest"]["capture"]["capture_groups"][0]["views"]
                for role in ("left", "right"):
                    for key, value in provenance[role].items():
                        self.assertEqual(views[role][key], value)
                    if exposure["capture_exposure_status"] == "AUTO":
                        self.assertNotIn("capture_exposure_ms", views[role])

    def test_invalid_exposure_provenance_is_rejected_before_export(self):
        for key, value in (("capture_exposure_ms", float("nan")),
                           ("capture_exposure_target_ms", True),
                           ("capture_exposure_status", "FAKE_SUCCESS")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self._create(camera_capture_provenance={"left": {key: value}})

    def test_history_is_copied_and_hash_checked(self):
        first = self._create()
        second = create_elevation_dataset(output_root=Path(self.tmp.name) / "out", calibration=_calibration(),
            left_path=self.left, right_path=self.right, left_time="2026-09-12T10:01:00+08:00",
            right_time="2026-09-12T10:01:00.001+08:00", pipe_specs=self.specs, pair_confirmed=True,
            previous_manifest=first)
        manifest = json.loads(second.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["capture"]["capture_groups"]), 2)
        self.assertTrue((second.parent / "history/0000_left.png").is_file())
        (second.parent / "history/0000_left.png").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            load_elevation_dataset(second)

    def test_changed_specs_cannot_reuse_history(self):
        first = self._create()
        changed = copy.deepcopy(self.specs); changed[0]["nominal_diameter_mm"] = 25
        with self.assertRaisesRegex(ValueError, "标定、管道配置和模型"):
            create_elevation_dataset(output_root=Path(self.tmp.name) / "out", calibration=_calibration(),
                left_path=self.left, right_path=self.right, left_time="2026-09-12T10:00:00+08:00",
                right_time="2026-09-12T10:00:00.001+08:00", pipe_specs=changed, pair_confirmed=True,
                previous_manifest=first)

    def test_identical_pair_is_idempotent(self):
        first = self._create()
        second = self._create(previous_manifest=first)
        manifest = json.loads(second.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["capture"]["capture_groups"]), 1)

    def test_history_compatibility_reports_config_change(self):
        first = self._create()
        changed = copy.deepcopy(self.specs); changed[0]["nominal_diameter_mm"] = 25
        self.assertFalse(elevation_history_compatible(first, _calibration(), changed))

    def test_no_model_ignores_gui_default_stl_unit(self):
        first = self._create(stl_unit="millimeter")
        self.assertTrue(elevation_history_compatible(
            first, _calibration(), self.specs, stl_unit="millimeter"))

    def test_tampered_model_raises_before_configuration_comparison(self):
        model = Path(self.tmp.name) / "reference.stl"
        model.write_text("solid reference\nendsolid reference\n", encoding="utf-8")
        first = self._create(model_path=model, stl_unit="millimeter")
        (first.parent / "model/reference.stl").write_text("changed", encoding="utf-8")
        changed = copy.deepcopy(self.specs); changed[0]["nominal_diameter_mm"] = 25
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            elevation_history_compatible(first, _calibration(), changed,
                                         model_path=model, stl_unit="millimeter")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self._create(previous_manifest=first, model_path=model, stl_unit="millimeter")

    def test_demo_calibration_and_unconfirmed_pair_are_rejected(self):
        demo = _calibration(); demo["calibration_id"] = "SYNTHETIC-DEMO"
        with self.assertRaises(ValueError):
            self._create(calibration=demo)
        with self.assertRaisesRegex(ValueError, "同步拍摄"):
            create_elevation_dataset(output_root=Path(self.tmp.name) / "out2", calibration=_calibration(),
                left_path=self.left, right_path=self.right, left_time="2026-09-12T10:00:00+08:00",
                right_time="2026-09-12T10:00:00+08:00", pipe_specs=self.specs, pair_confirmed=False)


if __name__ == "__main__":
    unittest.main()
