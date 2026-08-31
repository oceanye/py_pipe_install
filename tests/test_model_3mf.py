from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from pipe_twin.model_3mf import ThreeMFError, inspect_3mf


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "test_model" / "管道群.3mf"

CORE_NS = "http://schemas.microsoft.com/3dmanufacturing/core/2015/02"

BOX_MESH = """
<mesh>
  <vertices>
    <vertex x="0" y="-1" z="-1"/><vertex x="0" y="1" z="-1"/>
    <vertex x="0" y="1" z="1"/><vertex x="0" y="-1" z="1"/>
    <vertex x="10" y="-1" z="-1"/><vertex x="10" y="1" z="-1"/>
    <vertex x="10" y="1" z="1"/><vertex x="10" y="-1" z="1"/>
  </vertices>
  <triangles>
    <triangle v1="0" v2="2" v3="1"/><triangle v1="0" v2="3" v3="2"/>
    <triangle v1="4" v2="5" v3="6"/><triangle v1="4" v2="6" v3="7"/>
    <triangle v1="0" v2="1" v3="5"/><triangle v1="0" v2="5" v3="4"/>
    <triangle v1="1" v2="2" v3="6"/><triangle v1="1" v2="6" v3="5"/>
    <triangle v1="2" v2="3" v3="7"/><triangle v1="2" v2="7" v3="6"/>
    <triangle v1="3" v2="0" v3="4"/><triangle v1="3" v2="4" v3="7"/>
  </triangles>
</mesh>
"""


def _write_minimal_3mf(path: Path, model_xml: str) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as package:
        package.writestr("3D/3dmodel.model", model_xml)


class Inspect3mfContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inspection = inspect_3mf(MODEL_PATH)

    def test_reports_millimetre_units_and_three_objects(self) -> None:
        self.assertIsInstance(self.inspection, dict)
        self.assertEqual(self.inspection["unit"], "millimeter")
        self.assertEqual(self.inspection["object_count"], 3)
        self.assertEqual(len(self.inspection["objects"]), 3)

    def test_object_identity_color_and_dimensions_match_manifest_fixture(self) -> None:
        expected = {
            "1": {
                "uuid": "eb81751b-ea7c-4c5e-8e36-d7a4d29f3d46",
                "name": "实体3",
                "color_srgb": "#0036FA",
                "diameter_mm": 45.0,
                "length_mm": 500.0,
            },
            "3": {
                "uuid": "3d18358a-a006-4619-94ec-cbe5c5109a7e",
                "name": "实体4",
                "color_srgb": "#FF0000",
                "diameter_mm": 40.0,
                "length_mm": 500.0,
            },
            "5": {
                "uuid": "918c9795-d713-4098-8238-e6aeae36ed71",
                "name": "实体5",
                "color_srgb": "#FEFEFF",
                "diameter_mm": 20.0,
                "length_mm": 500.0,
            },
        }
        actual = {str(item["id"]): item for item in self.inspection["objects"]}

        self.assertEqual(set(actual), set(expected))
        for object_id, contract in expected.items():
            with self.subTest(object_id=object_id):
                item = actual[object_id]
                self.assertEqual(item["uuid"], contract["uuid"])
                self.assertEqual(item["name"], contract["name"])
                self.assertEqual(item["color_srgb"].upper(), contract["color_srgb"])
                self.assertAlmostEqual(
                    float(item["diameter_mm"]), contract["diameter_mm"], delta=0.1
                )
                self.assertAlmostEqual(
                    float(item["length_mm"]), contract["length_mm"], delta=0.001
                )

    def test_all_fixture_meshes_are_closed(self) -> None:
        for item in self.inspection["objects"]:
            with self.subTest(object_id=item["id"]):
                self.assertIn("watertight", item)
                self.assertTrue(item["watertight"])

    def test_missing_package_is_not_silently_accepted(self) -> None:
        missing = MODEL_PATH.with_name("missing-fixture.3mf")
        with self.assertRaises(FileNotFoundError):
            inspect_3mf(missing)

    def test_non_millimetre_model_is_rejected_instead_of_mislabelling_values(self) -> None:
        model_xml = f"""
        <model xmlns="{CORE_NS}" unit="inch">
          <resources><object id="1" type="model">{BOX_MESH}</object></resources>
          <build><item objectid="1"/></build>
        </model>
        """
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary) / "inch.3mf"
            _write_minimal_3mf(fixture, model_xml)
            with self.assertRaises(ThreeMFError):
                inspect_3mf(fixture)

    def test_build_transform_is_rejected_until_transform_support_is_explicit(self) -> None:
        model_xml = f"""
        <model xmlns="{CORE_NS}" unit="millimeter">
          <resources><object id="1" type="model">{BOX_MESH}</object></resources>
          <build><item objectid="1" transform="1 0 0 0 1 0 0 0 1 10 0 0"/></build>
        </model>
        """
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary) / "transformed.3mf"
            _write_minimal_3mf(fixture, model_xml)
            with self.assertRaises(ThreeMFError):
                inspect_3mf(fixture)

    def test_component_objects_are_rejected_until_component_support_is_explicit(self) -> None:
        model_xml = f"""
        <model xmlns="{CORE_NS}" unit="millimeter">
          <resources>
            <object id="1" type="model">{BOX_MESH}</object>
            <object id="2" type="model">
              <components><component objectid="1"/></components>
            </object>
          </resources>
          <build><item objectid="2"/></build>
        </model>
        """
        with tempfile.TemporaryDirectory() as temporary:
            fixture = Path(temporary) / "components.3mf"
            _write_minimal_3mf(fixture, model_xml)
            with self.assertRaises(ThreeMFError):
                inspect_3mf(fixture)


if __name__ == "__main__":
    unittest.main()
