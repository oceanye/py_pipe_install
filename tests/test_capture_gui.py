from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pipe_twin.capture_gui import catalog_from_model, create_capture_dataset
from pipe_twin.cli import build_parser


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "test_model" / "field_stereo_demo_manifest.json"


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

    def test_same_image_unconfirmed_pair_and_naive_time_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            for changed in ({"right_path": self.arguments["left_path"]}, {"pair_confirmed": False}, {"left_time": "2026-09-07T10:00:00"}):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    create_capture_dataset(output_root=Path(temp), **(self.arguments | changed))
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_model_catalog_design_only_preserves_all_nine_cylinders(self):
        pipes, skipped = catalog_from_model(self.arguments["model_path"])
        self.assertEqual(len(pipes), 9, skipped)
        self.assertFalse(skipped)
        self.assertEqual(len({p["pipe_id"] for p in pipes}), 9)
        self.assertEqual(sorted(round(p["nominal_diameter_mm"]) for p in pipes), [20, 20, 20, 20, 40, 40, 45, 45, 45])


if __name__ == "__main__":
    unittest.main()
