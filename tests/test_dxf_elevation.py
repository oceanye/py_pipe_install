from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipe_twin.capture_gui import catalog_from_model
from pipe_twin.dxf_elevation import DxfError, arc_points, catalog_from_dxf, read_dxf_elevation


class DxfElevationTests(unittest.TestCase):
    def test_reads_layer_colors_and_outer_contours(self) -> None:
        content = """0\nSECTION\n2\nTABLES\n0\nTABLE\n2\nLAYER\n0\nLAYER\n2\nPIPES\n62\n1\n0\nENDTAB\n0\nENDSEC\n0\nSECTION\n2\nENTITIES\n0\nLINE\n8\nPIPES\n10\n0\n20\n0\n11\n100\n21\n0\n0\nCIRCLE\n8\nPIPES\n10\n50\n20\n25\n40\n10\n0\nENDSEC\n0\nEOF\n"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "elevation.dxf"
            path.write_text(content, encoding="ascii")
            document = read_dxf_elevation(path)
        self.assertEqual(document.layers["PIPES"], "#FF0000")
        self.assertEqual([entity.kind for entity in document.entities], ["LINE", "CIRCLE"])
        self.assertEqual([entity.entity_id for entity in document.entities], ["P001", "P002"])
        self.assertEqual(document.entities[1].color, "#FF0000")
        self.assertGreaterEqual(len(arc_points(document.entities[1])), 2)

    def test_unitless_dxf_requires_an_explicit_manifest_unit(self) -> None:
        content = "0\nSECTION\n2\nENTITIES\n0\nCIRCLE\n8\n0\n10\n0\n20\n0\n40\n10\n0\nENDSEC\n0\nEOF\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unitless.dxf"
            path.write_text(content, encoding="ascii")
            with self.assertRaises(DxfError):
                catalog_from_dxf(path)
            pipes, _ = catalog_from_dxf(path, unitless_unit="centimeter")
        self.assertEqual(pipes[0]["nominal_diameter_mm"], 200.0)

    def test_real_pipe_layout_catalog_uses_circles_and_object_colors(self) -> None:
        root = Path(__file__).resolve().parents[1]
        document = root / "test_model" / "管道布置.dxf"
        pipes, skipped = catalog_from_dxf(document)
        self.assertEqual(len(pipes), 12)
        self.assertEqual(len(skipped), 4)  # the rectangular drawing border
        self.assertEqual({round(p["nominal_diameter_mm"]): sum(round(q["nominal_diameter_mm"]) == round(p["nominal_diameter_mm"]) for q in pipes) for p in pipes}, {26: 4, 41: 4, 51: 4})
        self.assertEqual({round(p["nominal_diameter_mm"]): p["color_srgb"] for p in pipes}, {26: "#FFFFFF", 41: "#FF0000", 51: "#0000FF"})
        self.assertTrue(all(p["centerline_world_mm"][0][2] < 0 < p["centerline_world_mm"][1][2] for p in pipes))
        stl, skipped_stl = catalog_from_model(root / "test_model" / "管道布置.stl", stl_unit="millimeter")
        dxf_centres = sorted((round((row["centerline_world_mm"][0][0] + row["centerline_world_mm"][1][0]) / 2, 2),
                              round((row["centerline_world_mm"][0][1] + row["centerline_world_mm"][1][1]) / 2, 2),
                              round(row["nominal_diameter_mm"], 3)) for row in pipes)
        stl_centres = sorted((round((row["centerline_world_mm"][0][0] + row["centerline_world_mm"][1][0]) / 2, 2),
                              round((row["centerline_world_mm"][0][1] + row["centerline_world_mm"][1][1]) / 2, 2),
                              round(row["nominal_diameter_mm"], 3)) for row in stl)
        self.assertEqual(dxf_centres, stl_centres)
        self.assertEqual(len(skipped_stl), 0)


if __name__ == "__main__":
    unittest.main()
