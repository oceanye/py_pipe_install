from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pipe_twin.dxf_elevation import arc_points, read_dxf_elevation


class DxfElevationTests(unittest.TestCase):
    def test_reads_layer_colors_and_outer_contours(self) -> None:
        content = """0\nSECTION\n2\nTABLES\n0\nTABLE\n2\nLAYER\n0\nLAYER\n2\nPIPES\n62\n1\n0\nENDTAB\n0\nENDSEC\n0\nSECTION\n2\nENTITIES\n0\nLINE\n8\nPIPES\n10\n0\n20\n0\n11\n100\n21\n0\n0\nCIRCLE\n8\nPIPES\n10\n50\n20\n25\n40\n10\n0\nENDSEC\n0\nEOF\n"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "elevation.dxf"
            path.write_text(content, encoding="ascii")
            document = read_dxf_elevation(path)
        self.assertEqual(document.layers["PIPES"], "#FF0000")
        self.assertEqual([entity.kind for entity in document.entities], ["LINE", "CIRCLE"])
        self.assertEqual(document.entities[1].color, "#FF0000")
        self.assertGreaterEqual(len(arc_points(document.entities[1])), 2)


if __name__ == "__main__":
    unittest.main()
