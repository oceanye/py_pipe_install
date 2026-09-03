from __future__ import annotations

import hashlib
import tempfile
import unittest
import uuid
from pathlib import Path

from pipe_twin.cad_model import (
    CadModelError,
    load_3dm_scene,
    load_3mf_scene,
    load_cad_scene,
)

try:
    import rhino3dm
except ImportError:  # pragma: no cover - the CI dependency is installed by the project.
    rhino3dm = None


ROOT = Path(__file__).resolve().parents[1]
THREEMF_PATH = ROOT / "test_model" / "管道群.3mf"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_rhino_mesh_model(
    path: Path,
    *,
    unit: object | None = None,
    include_auxiliary_brep: bool = False,
) -> uuid.UUID:
    assert rhino3dm is not None
    model = rhino3dm.File3dm()
    if unit is not None:
        model.Settings.ModelUnitSystem = unit

    layer = rhino3dm.Layer()
    layer.Name = "Installed Pipes"
    layer.Color = (11, 22, 33, 255)
    layer_index = model.Layers.Add(layer)

    mesh = rhino3dm.Mesh()
    # A quad deliberately exercises the deterministic A-B-C / A-C-D split.
    for point in ((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0)):
        mesh.Vertices.Add(*point)
    mesh.Faces.AddFace(0, 1, 2, 3)

    object_guid = uuid.UUID("12345678-1234-5678-9abc-1234567890ab")
    attributes = rhino3dm.ObjectAttributes()
    attributes.Id = object_guid
    attributes.Name = "Pipe A"
    attributes.LayerIndex = layer_index
    attributes.ColorSource = rhino3dm.ObjectColorSource.ColorFromObject
    attributes.ObjectColor = (201, 102, 3, 255)
    attributes.SetUserString("pipe_id", "PIPE-A")
    added = model.Objects.AddMesh(mesh, attributes)
    if str(added).lower() != str(object_guid).lower():
        raise AssertionError("rhino3dm did not preserve the requested object GUID")
    if include_auxiliary_brep:
        auxiliary = rhino3dm.Brep.CreateFromBoundingBox(
            rhino3dm.BoundingBox(0, 0, 0, 1, 1, 1)
        )
        auxiliary_attributes = rhino3dm.ObjectAttributes()
        auxiliary_attributes.Id = uuid.UUID(
            "abcdefab-cdef-abcd-efab-cdefabcdefab"
        )
        auxiliary_attributes.Name = "Unbound construction solid"
        auxiliary_attributes.LayerIndex = layer_index
        model.Objects.AddBrep(auxiliary, auxiliary_attributes)
    if not model.Write(str(path), 8):
        raise AssertionError("rhino3dm could not write the temporary model")
    return object_guid


@unittest.skipUnless(rhino3dm is not None, "rhino3dm is an optional CAD reader dependency")
class Rhino3dmLoaderTests(unittest.TestCase):
    def test_native_mesh_guid_layer_color_user_string_and_unit_conversion(self) -> None:
        assert rhino3dm is not None
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "pipes.3dm"
            object_guid = _write_rhino_mesh_model(
                model_path,
                unit=rhino3dm.UnitSystem.Centimeters,
            )
            scene = load_3dm_scene(model_path)
            expected_sha256 = _sha256(model_path)

        self.assertEqual(scene.source_format, "3dm")
        self.assertEqual(scene.source_unit, "Centimeters")
        self.assertEqual(scene.unit_scale_to_mm, 10.0)
        self.assertEqual(scene.source_sha256, expected_sha256)
        self.assertEqual(scene.object_count, 1)
        item = scene.objects[0]
        self.assertEqual(item.object_id, str(object_guid))
        self.assertEqual(item.guid, str(object_guid))
        self.assertEqual(item.pipe_id, "PIPE-A")
        self.assertEqual(item.name, "Pipe A")
        self.assertEqual(item.layer_path, "Installed Pipes")
        self.assertEqual(item.color_srgb, "#C96603")
        self.assertEqual(item.geometry_type, "Mesh")
        self.assertEqual(item.mesh_source, "3dm_native_mesh")
        self.assertEqual(item.vertex_count, 4)
        self.assertEqual(item.triangle_count, 2)
        self.assertEqual(item.bbox_min_mm, (0.0, 0.0, 0.0))
        self.assertEqual(item.bbox_max_mm, (20.0, 10.0, 0.0))
        self.assertEqual(item.triangles.tolist(), [[0, 1, 2], [0, 2, 3]])
        self.assertFalse(item.vertices_world_mm.flags.writeable)
        self.assertFalse(item.triangles.flags.writeable)
        with self.assertRaises(ValueError):
            item.vertices_world_mm[0, 0] = 99.0

    def test_brep_without_cached_render_mesh_fails_closed(self) -> None:
        assert rhino3dm is not None
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "brep_without_mesh.3dm"
            model = rhino3dm.File3dm()
            layer = rhino3dm.Layer()
            layer.Name = "Pipes"
            layer_index = model.Layers.Add(layer)
            brep = rhino3dm.Brep.CreateFromBoundingBox(
                rhino3dm.BoundingBox(0, 0, 0, 10, 10, 10)
            )
            attributes = rhino3dm.ObjectAttributes()
            attributes.LayerIndex = layer_index
            attributes.Name = "BRep pipe"
            model.Objects.AddBrep(brep, attributes)
            self.assertTrue(model.Write(str(model_path), 8))

            with self.assertRaisesRegex(CadModelError, "cached render mesh"):
                load_3dm_scene(model_path)

    def test_required_guid_filter_skips_unbound_auxiliary_geometry(self) -> None:
        assert rhino3dm is not None
        with tempfile.TemporaryDirectory() as temporary:
            model_path = Path(temporary) / "pipe_with_auxiliary.3dm"
            pipe_guid = _write_rhino_mesh_model(
                model_path,
                unit=rhino3dm.UnitSystem.Millimeters,
                include_auxiliary_brep=True,
            )

            with self.assertRaisesRegex(CadModelError, "cached render mesh"):
                load_3dm_scene(model_path)
            selected = load_3dm_scene(
                model_path,
                required_object_ids={str(pipe_guid).upper()},
            )

            self.assertEqual(selected.object_count, 1)
            self.assertEqual(selected.objects[0].object_id, str(pipe_guid))
            with self.assertRaisesRegex(CadModelError, "missing required object GUIDs"):
                load_3dm_scene(
                    model_path,
                    required_object_ids={"ffffffff-ffff-ffff-ffff-ffffffffffff"},
                )

class UnifiedCadLoaderTests(unittest.TestCase):
    def test_existing_3mf_is_normalized_to_common_scene(self) -> None:
        scene = load_3mf_scene(THREEMF_PATH)
        self.assertEqual(scene.source_format, "3mf")
        self.assertEqual(scene.source_unit, "millimeter")
        self.assertEqual(scene.unit_scale_to_mm, 1.0)
        self.assertEqual(scene.source_sha256, _sha256(THREEMF_PATH))
        self.assertEqual(scene.object_count, 3)
        self.assertEqual(set(scene.by_object_id), {"1", "3", "5"})
        self.assertEqual(
            set(scene.by_guid),
            {
                "eb81751b-ea7c-4c5e-8e36-d7a4d29f3d46",
                "3d18358a-a006-4619-94ec-cbe5c5109a7e",
                "918c9795-d713-4098-8238-e6aeae36ed71",
            },
        )
        self.assertTrue(all(item.vertices_world_mm.shape[1] == 3 for item in scene.objects))
        self.assertTrue(all(item.triangles.shape[1] == 3 for item in scene.objects))
        self.assertTrue(all(item.watertight for item in scene.objects))

    def test_dispatch_and_unknown_extension_fail_closed(self) -> None:
        self.assertEqual(load_cad_scene(THREEMF_PATH).source_format, "3mf")
        selected = load_cad_scene(
            THREEMF_PATH,
            required_object_ids={"1", "5"},
        )
        self.assertEqual(set(selected.by_object_id), {"1", "5"})
        with self.assertRaisesRegex(CadModelError, "missing required object IDs"):
            load_cad_scene(THREEMF_PATH, required_object_ids={"missing"})
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.obj"
            path.write_text("not a supported CAD model", encoding="utf-8")
            with self.assertRaisesRegex(CadModelError, "expected .3mf or .3dm"):
                load_cad_scene(path)


if __name__ == "__main__":
    unittest.main()
