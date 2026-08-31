from __future__ import annotations

import hashlib
import json
import unittest
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "test_model"
MANIFEST_PATH = MODEL_DIR / "manifest.json"
GROUP2_MANIFEST_PATH = MODEL_DIR / "pipe_group2_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RepositorySmokeTests(unittest.TestCase):
    def test_required_project_files_exist(self) -> None:
        required = (
            "README.md",
            "requirements.txt",
            "pipe_twin/photo_capture.py",
            "tests/test_photo_capture.py",
            "doc/需求文档.txt",
            "doc/camera_contour_3d_pipe_migration_guide.md",
            "doc/管道数字孪生识别系统测试开发计划.md",
            "doc/管道群2模拟双目与遮挡拓扑说明.md",
            "test_model/管道群.3mf",
            "test_model/管道群.mkv",
            "test_model/manifest.json",
            "test_model/管道群2.3mf",
            "test_model/管道群2.mkv",
            "test_model/pipe_group2_manifest.json",
            "test_model/pipe_group2_synthetic_stereo/camera.json",
            "test_model/pipe_group2_synthetic_stereo/left_truth.npz",
            "test_model/pipe_group2_synthetic_stereo/right_truth.npz",
            "test_model/pipe_group2_synthetic_stereo/elevation_amodal_overlay.png",
            "test_model/pipe_group2_synthetic_stereo/elevation_view_topology.json",
            "test_model/pipe_group2_synthetic_stereo/installation_status.json",
            "test_model/pipe_group2_synthetic_stereo/dataset_manifest.json",
        )

        for relative_path in required:
            with self.subTest(path=relative_path):
                path = ROOT / relative_path
                self.assertTrue(path.is_file(), f"Missing required file: {relative_path}")
                self.assertGreater(path.stat().st_size, 0, f"Empty required file: {relative_path}")

    def test_3mf_fixture_has_basic_package_structure(self) -> None:
        model_path = MODEL_DIR / "管道群.3mf"
        self.assertTrue(zipfile.is_zipfile(model_path))

        with zipfile.ZipFile(model_path) as package:
            names = set(package.namelist())
            self.assertIn("[Content_Types].xml", names)
            self.assertIn("3D/3dmodel.model", names)
            root = ElementTree.fromstring(package.read("3D/3dmodel.model"))

        core = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"
        self.assertEqual(root.attrib.get("unit"), "millimeter")
        self.assertEqual(len(root.findall(f".//{{{core}}}object")), 3)

    def test_mkv_fixture_has_matroska_ebml_header(self) -> None:
        for name in ("管道群.mkv", "管道群2.mkv"):
            with self.subTest(name=name), (MODEL_DIR / name).open("rb") as stream:
                self.assertEqual(stream.read(4), bytes.fromhex("1A45DFA3"))

    def test_manifest_binds_current_model_and_video_by_hash(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

        for section_name in ("model", "video"):
            with self.subTest(section=section_name):
                section = manifest[section_name]
                asset_path = MODEL_DIR / section["path"]
                self.assertTrue(asset_path.is_file())
                self.assertEqual(_sha256(asset_path), section["sha256"].lower())

        self.assertEqual(manifest["model"]["expected_object_count"], 3)
        self.assertEqual(manifest["video"]["expected_frame_count"], 1727)

    def test_group2_manifest_binds_model_video_and_generated_dataset(self) -> None:
        manifest = json.loads(GROUP2_MANIFEST_PATH.read_text(encoding="utf-8"))
        model = manifest["model"]
        video = manifest["reference_video"]
        model_path = MODEL_DIR / model["path"]
        video_path = MODEL_DIR / video["path"]

        self.assertEqual(_sha256(model_path), model["sha256"].lower())
        self.assertEqual(_sha256(video_path), video["sha256"].lower())
        self.assertEqual(model["expected_object_count"], 9)
        self.assertEqual(len(model["pipes"]), 9)
        self.assertEqual(manifest["domain"], "synthetic_cad_truth")

        generated_path = MODEL_DIR / "pipe_group2_synthetic_stereo" / "dataset_manifest.json"
        generated = json.loads(generated_path.read_text(encoding="utf-8"))
        self.assertEqual(generated["source"]["manifest_sha256"], _sha256(GROUP2_MANIFEST_PATH))
        self.assertEqual(generated["source"]["model_sha256"], model["sha256"])
        self.assertEqual(len(generated["instance_catalog"]), 9)
        status_record = generated["installation_assessment"]
        self.assertEqual(status_record["path"], "installation_status.json")
        status_path = generated_path.parent / status_record["path"]
        self.assertEqual(status_path.stat().st_size, status_record["size_bytes"])
        self.assertEqual(_sha256(status_path), status_record["sha256"])

        status = json.loads(status_path.read_text(encoding="utf-8"))
        pipes = status["pipes"]
        instance_ids = [item["instance_id"] for item in pipes]
        pipe_ids = [item["pipe_id"] for item in pipes]
        self.assertEqual(len(pipes), 9)
        self.assertEqual(len(set(instance_ids)), 9)
        self.assertEqual(len(set(pipe_ids)), 9)
        self.assertEqual(
            set(instance_ids),
            {item["instance_id"] for item in generated["instance_catalog"]},
        )
        self.assertEqual(
            set(pipe_ids),
            {item["pipe_id"] for item in generated["instance_catalog"]},
        )
        observed_counts = Counter(item["installation_state"] for item in pipes)
        self.assertEqual(
            {
                state: observed_counts[state]
                for state in ("INSTALLED", "NOT_INSTALLED", "UNKNOWN")
            },
            status["counts"],
        )
        labels = {
            "INSTALLED": "安装",
            "NOT_INSTALLED": "未安装",
            "UNKNOWN": "不明",
        }
        for item in pipes:
            with self.subTest(status_pipe_id=item["pipe_id"]):
                self.assertIn(item["installation_state"], labels)
                self.assertEqual(
                    item["installation_state_zh"], labels[item["installation_state"]]
                )
                self.assertTrue(
                    set(item["positive_evidence_view_ids"]).issubset({"left", "right"})
                )
                self.assertTrue(
                    set(item["negative_evidence_view_ids"]).issubset({"left", "right"})
                )
        self.assertEqual(status["fusion_view_ids"], ["left", "right"])
        self.assertFalse(status["reference_elevation_included_in_fusion"])

    def test_manifest_keeps_mvp_capability_boundaries_explicit(self) -> None:
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        scope = manifest["scope"]

        self.assertEqual(scope["layer_model"], "single")
        self.assertFalse(scope["metric_calibrated"])
        self.assertFalse(scope["supports_stereo"])
        self.assertFalse(scope["supports_occlusion_reasoning"])
        self.assertFalse(scope["supports_3dgs_training"])
        self.assertIn("appearance_color", scope["identity_features"])

        limitations = " ".join(manifest["limitations"]).lower()
        self.assertIn("not_observed", limitations)
        self.assertIn("unknown", limitations)
        self.assertIn("d22/d50", limitations)


if __name__ == "__main__":
    unittest.main()
