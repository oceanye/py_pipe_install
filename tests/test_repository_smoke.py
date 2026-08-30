from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositorySmokeTests(unittest.TestCase):
    def test_required_project_files_exist(self) -> None:
        required = (
            "README.md",
            "requirements.txt",
            "doc/需求文档.txt",
            "doc/camera_contour_3d_pipe_migration_guide.md",
            "doc/管道数字孪生识别系统测试开发计划.md",
            "test_model/管道群.sat",
        )

        for relative_path in required:
            with self.subTest(path=relative_path):
                path = ROOT / relative_path
                self.assertTrue(path.is_file(), f"Missing required file: {relative_path}")
                self.assertGreater(path.stat().st_size, 0, f"Empty required file: {relative_path}")

    def test_sat_fixture_has_basic_acis_structure(self) -> None:
        sat_path = ROOT / "test_model/管道群.sat"
        content = sat_path.read_text(encoding="ascii")

        self.assertGreaterEqual(content.count("body "), 2)
        self.assertIn("cone-surface", content)
        self.assertTrue(content.rstrip().endswith("End-of-ACIS-data"))


if __name__ == "__main__":
    unittest.main()
