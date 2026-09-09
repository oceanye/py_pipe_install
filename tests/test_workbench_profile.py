from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from pipe_twin.workbench_profile import (
    default_profile,
    load_camera_calibration_bundle,
    load_profile,
    profile_sections_from_state,
    reset_profile,
    save_profile,
    save_camera_calibration_bundle,
    update_profile,
    calibration_ids_match,
    capture_state_from_profile,
    validate_profile,
    validate_rectification_recipe,
    write_standalone_calibration,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "test_model" / "field_stereo_demo_manifest.json"


def _calibration() -> dict:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    calibration = copy.deepcopy(payload["stereo_calibration"])
    calibration["calibration_id"] = "FIELD-CHESS-TEST-001"
    return calibration


def _pipes() -> list[dict]:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return copy.deepcopy(payload["model"]["pipes"])


def _recipe(*, width: int = 1920, height: int = 1080, calibration_id: str = "FIELD-CHESS-TEST-001") -> dict:
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    focal = 2637.5783226764374
    intrinsic = [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]]
    projection_left = [[focal, 0.0, width / 2, 0.0], [0.0, focal, height / 2, 0.0], [0.0, 0.0, 1.0, 0.0]]
    projection_right = [row[:] for row in projection_left]
    projection_right[0][3] = -focal * 95.0
    return {
        "calibration_id": calibration_id,
        "K1": intrinsic,
        "D1": [0.0, 0.0, 0.0, 0.0, 0.0],
        "K2": [row[:] for row in intrinsic],
        "D2": [0.0, 0.0, 0.0, 0.0, 0.0],
        "R1": identity,
        "R2": identity,
        "P1": projection_left,
        "P2": projection_right,
        "image_width_px": width,
        "image_height_px": height,
        "output_width_px": width,
        "output_height_px": height,
        "alpha": 0.0,
        "created_at": "2026-09-08T10:00:00+00:00",
        "definition": "unit-test recipe",
    }


def _profile() -> dict:
    profile = default_profile()
    profile["model_path"] = "test_model/管道布置.stl"
    profile["pipes"] = _pipes()
    profile["calibration_path"] = "outputs/measurement_workbench/calibration_current.json"
    profile["calibration_current"] = _calibration()
    profile["rectification_recipe"] = _recipe()
    profile["qr_settings"] = {
        "marker_id": "PIPE-TWIN-QR-001",
        "marker_edge_mm": 120.0,
        "measured_marker_edge_mm": 114.5,
        "marker_center_world_mm": [10.0, 20.0, 30.0],
        "print_right_world": "+X",
        "print_up_world": "+Y",
        "max_reprojection_rms_px": 2.0,
    }
    profile["pose_adjustment"] = {
        "mode": "adjust_current",
        "center_world_mm": [0.0, 0.0, 1000.0],
        "yaw_deg": 1.5,
        "pitch_deg": -2.0,
        "roll_deg": 0.5,
        "registration_validated": True,
    }
    profile["camera"] = {"layout": "side_by_side_left_right", "left_index": 0, "right_index": 1}
    profile["last_manifest_path"] = "outputs/measurement_workbench/captures/field-1/manifest.json"
    profile["capture_history"] = True
    return profile


class WorkbenchProfileTests(unittest.TestCase):
    def test_profile_round_trip_preserves_every_section(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profile.json"
            save_profile(_profile(), path=path)
            loaded, problem = load_profile(path)
            self.assertEqual(problem, "")
            expected = validate_profile(_profile())
            expected["updated_at"] = loaded["updated_at"]
            self.assertEqual(loaded, expected)

    def test_legacy_camera_transform_migrates_into_rectification_recipe(self):
        profile = _profile()
        profile["camera"]["right_frame_transform"] = "flip_horizontal"
        profile["rectification_recipe"].pop("right_frame_transform", None)

        migrated = validate_profile(profile)

        self.assertEqual(
            migrated["rectification_recipe"]["right_frame_transform"],
            "flip_horizontal",
        )
        self.assertNotIn("right_frame_transform", migrated["camera"])

    def test_invalid_kind_or_schema_version_is_rejected(self):
        for changed in ({"kind": "other"}, {"schema_version": "9.9"}):
            with self.subTest(changed=changed):
                with self.assertRaisesRegex(ValueError, "工作台配置"):
                    validate_profile(_profile() | changed)

    def test_bad_pipe_or_calibration_fails_closed(self):
        broken_pipe = _profile()
        broken_pipe["pipes"][0]["pipe_id"] = ""
        with self.assertRaisesRegex(ValueError, "pipe_id"):
            validate_profile(broken_pipe)
        broken_calibration = _profile()
        broken_calibration["calibration_current"]["rectified"] = False
        with self.assertRaisesRegex(ValueError, "rectified"):
            validate_profile(broken_calibration)
        duplicate = _profile()
        duplicate["pipes"][1]["pipe_id"] = duplicate["pipes"][0]["pipe_id"]
        with self.assertRaisesRegex(ValueError, "重复"):
            validate_profile(duplicate)

    def test_recipe_shape_or_size_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "字段不匹配"):
            validate_rectification_recipe(_recipe() | {"extra": 1})
        with self.assertRaisesRegex(ValueError, "尺寸不一致"):
            validate_rectification_recipe(_recipe(width=640, height=480), calibration=_calibration())
        broken = _recipe()
        broken["R1"][0][0] = 2.0
        with self.assertRaisesRegex(ValueError, "正交旋转"):
            validate_rectification_recipe(broken)
        skewed = _recipe()
        skewed["K1"][0][1] = 5.0
        with self.assertRaisesRegex(ValueError, "轴倾斜"):
            validate_rectification_recipe(skewed)

    def test_recipe_projection_geometry_and_calibration_binding_are_guarded(self):
        reversed_baseline = _recipe()
        reversed_baseline["P2"][0][3] *= -1
        with self.assertRaisesRegex(ValueError, "正的水平基线"):
            validate_rectification_recipe(reversed_baseline)

        wrong_intrinsic = _recipe()
        wrong_intrinsic["P1"][0][0] += 5.0
        wrong_intrinsic["P2"][0][0] += 5.0
        with self.assertRaisesRegex(ValueError, "当前左目矫正内参"):
            validate_rectification_recipe(
                wrong_intrinsic, calibration=_calibration()
            )

    def test_safety_checkboxes_are_never_persisted(self):
        payload = _profile()
        payload["qr_settings"]["print_measured"] = True
        with self.assertRaisesRegex(ValueError, "安全确认"):
            validate_profile(payload)
        payload = _profile()
        payload["qr_settings"]["cad_confirmed"] = True
        with self.assertRaisesRegex(ValueError, "安全确认"):
            validate_profile(payload)

    def test_unreadable_profile_is_quarantined_to_recovery_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profile.json"
            path.write_text("{not json", encoding="utf-8")
            profile, problem = load_profile(path)
            self.assertIsNone(profile)
            self.assertIn("无法载入", problem)
            self.assertFalse(path.exists())
            recoveries = list(Path(temp).glob("profile_recovery_*.json"))
            self.assertEqual(len(recoveries), 1)
        profile, problem = load_profile(Path(temp) / "missing.json")
        self.assertIsNone(profile)
        self.assertEqual(problem, "")

    def test_save_rotates_previous_content_to_bak(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profile.json"
            save_profile(_profile(), path=path)
            first = path.read_text(encoding="utf-8")
            changed = _profile()
            changed["capture_history"] = False
            save_profile(changed, path=path)
            self.assertEqual(json.loads(path.with_name(path.name + ".bak").read_text(encoding="utf-8"))["capture_history"], True)
            self.assertNotEqual(path.read_text(encoding="utf-8"), first)

    def test_reset_profile_writes_default_and_keeps_old_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profile.json"
            save_profile(_profile(), path=path)
            reset_profile(path=path)
            loaded, problem = load_profile(path)
            self.assertEqual(problem, "")
            self.assertEqual(loaded["pipes"], [])
            self.assertTrue(list(Path(temp).glob("profile_recovery_*.json")))

    def test_recipe_ownership_matches_base_and_suffix_ids(self):
        self.assertTrue(calibration_ids_match("FIELD-CHESS-TEST-001", "FIELD-CHESS-TEST-001"))
        self.assertTrue(
            calibration_ids_match("FIELD-CHESS-TEST-001-qr-abc-def", "FIELD-CHESS-TEST-001")
        )
        for current, recipe in (
            ("", "FIELD-CHESS-TEST-001"),
            ("FIELD-CHESS-TEST-001", ""),
            ("OTHER-CHESS-TEST-001", "FIELD-CHESS-TEST-001"),
            ("FIELD-CHESS-TEST-001X", "FIELD-CHESS-TEST-001"),
        ):
            with self.subTest(current=current, recipe=recipe):
                self.assertFalse(calibration_ids_match(current, recipe))

    def test_capture_state_mapping_round_trips_through_profile_sections(self):
        state = capture_state_from_profile(_profile())
        self.assertEqual(state["camera"], {"layout": "side_by_side_left_right", "left_index": 0, "right_index": 1})
        sections = profile_sections_from_state(
            state,
            sections=("model_path", "stl_unit", "pipes", "qr_settings", "pose_adjustment", "camera"),
        )
        self.assertEqual(set(sections), {"model_path", "stl_unit", "pipes", "qr_settings", "pose_adjustment", "camera"})
        with self.assertRaisesRegex(ValueError, "不支持的配置段落"):
            profile_sections_from_state(state, sections=("nonsense",))
        with self.assertRaisesRegex(ValueError, "缺少配置段落"):
            profile_sections_from_state({}, sections=("model_path",))

    def test_update_profile_merges_sections_and_keeps_the_rest(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profile.json"
            save_profile(_profile(), path=path)
            update_profile({"capture_history": False, "camera": {"layout": "separate_devices", "left_index": 0, "right_index": 2}}, path=path)
            loaded, problem = load_profile(path)
            self.assertEqual(problem, "")
            self.assertFalse(loaded["capture_history"])
            self.assertEqual(loaded["camera"]["right_index"], 2)
            self.assertEqual(loaded["pipes"], _pipes())
            with self.assertRaisesRegex(ValueError, "不支持的配置段落"):
                update_profile({"nonsense": 1}, path=path)

    def test_standalone_calibration_writer_round_trips_through_loader(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "calibration_current.json"
            written = write_standalone_calibration(_calibration(), path=path)
            self.assertEqual(written, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("note", payload)
            self.assertEqual(payload["stereo_calibration"]["calibration_id"], "FIELD-CHESS-TEST-001")

    def test_portable_calibration_bundle_preserves_recipe_and_qr_settings(self):
        calibration = _calibration()
        calibration["calibration_id"] += "-qr-registration"
        calibration["registration_validated"] = True
        settings = _profile()["qr_settings"]
        with tempfile.TemporaryDirectory() as temp:
            path = save_camera_calibration_bundle(
                Path(temp) / "camera-result.json",
                calibration,
                rectification_recipe=_recipe(),
                qr_settings=settings,
            )
            loaded = load_camera_calibration_bundle(path)
        self.assertEqual(
            loaded["stereo_calibration"]["calibration_id"],
            calibration["calibration_id"],
        )
        self.assertEqual(loaded["qr_settings"]["measured_marker_edge_mm"], 114.5)
        self.assertEqual(loaded["rectification_recipe"]["calibration_id"], "FIELD-CHESS-TEST-001")
        self.assertEqual(loaded["rectification_recipe"]["right_frame_transform"], "none")

    def test_chessboard_bundle_without_rectification_recipe_is_rejected(self):
        calibration = _calibration()
        with self.assertRaisesRegex(ValueError, "极线矫正配方"):
            save_camera_calibration_bundle(
                "unused.json", calibration, rectification_recipe=None
            )


if __name__ == "__main__":
    unittest.main()
