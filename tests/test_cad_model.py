from __future__ import annotations

import hashlib
import struct
import tempfile
import unittest
import uuid
from pathlib import Path

from pipe_twin.cad_model import (
    CadModelError,
    load_3dm_scene,
    load_3mf_scene,
    load_cad_scene,
    load_stl_scene,
)

try:
    import rhino3dm
except ImportError:  # pragma: no cover - the CI dependency is installed by the project.
    rhino3dm = None


ROOT = Path(__file__).resolve().parents[1]
THREEMF_PATH = ROOT / "test_model" / "管道群.3mf"
STL_PATH = ROOT / "test_model" / "管道布置.stl"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tetrahedron(offset: float = 0.0) -> list[list[tuple[float, float, float]]]:
    vertices = [
        (offset + 0.0, 0.0, 0.0),
        (offset + 1.0, 0.0, 0.0),
        (offset + 0.0, 1.0, 0.0),
        (offset + 0.0, 0.0, 1.0),
    ]
    return [
        [vertices[a], vertices[b], vertices[c]]
        for a, b, c in ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))
    ]


def _write_binary_stl(path: Path, triangles: list) -> None:
    payload = bytearray(b"unit test STL".ljust(80, b"\0"))
    payload.extend(struct.pack("<I", len(triangles)))
    for triangle in triangles:
        flat = [coordinate for point in triangle for coordinate in point]
        payload.extend(struct.pack("<12fH", 0.0, 0.0, 0.0, *flat, 0))
    path.write_bytes(payload)


def _write_ascii_stl(path: Path, triangles: list) -> None:
    lines = ["solid fixture"]
    for triangle in triangles:
        lines.extend(["facet normal 0 0 0", "outer loop"])
        lines.extend(f"vertex {x} {y} {z}" for x, y, z in triangle)
        lines.extend(["endloop", "endfacet"])
    lines.append("endsolid fixture")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


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
            with self.assertRaisesRegex(CadModelError, "expected .3mf, .3dm, or .stl"):
                load_cad_scene(path)


class StlLoaderTests(unittest.TestCase):
    def test_repository_binary_stl_splits_twelve_watertight_pipe_shells(self) -> None:
        scene = load_stl_scene(STL_PATH, stl_unit="millimeter")
        self.assertEqual(scene.source_format, "stl")
        self.assertEqual(scene.source_unit, "millimeter")
        self.assertEqual(scene.unit_scale_to_mm, 1.0)
        self.assertEqual(scene.source_sha256, _sha256(STL_PATH))
        self.assertEqual(scene.object_count, 12)
        self.assertEqual(sum(item.vertex_count for item in scene.objects), 576)
        self.assertEqual(sum(item.triangle_count for item in scene.objects), 1104)
        self.assertTrue(all(item.watertight for item in scene.objects))
        self.assertTrue(all(item.mesh_source == "stl_binary_connected_mesh" for item in scene.objects))
        self.assertTrue(all(item.color_srgb == "#B0B0B0" for item in scene.objects))
        self.assertEqual(len(set(scene.by_object_id)), 12)

    def test_binary_component_ids_are_stable_and_filterable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "two-components.stl"
            _write_binary_stl(path, _tetrahedron(10) + _tetrahedron(0))
            first = load_stl_scene(path, stl_unit="mm")
            second = load_stl_scene(path, stl_unit="millimeters")
            self.assertEqual(list(first.by_object_id), list(second.by_object_id))
            selected_id = first.objects[1].object_id
            selected = load_cad_scene(
                path,
                stl_unit="millimeter",
                required_object_ids={selected_id.upper()},
            )
            self.assertEqual([item.object_id for item in selected.objects], [selected_id])
            with self.assertRaisesRegex(CadModelError, "missing required component IDs"):
                load_stl_scene(path, stl_unit="millimeter", required_object_ids={"missing"})

    def test_ascii_stl_and_explicit_unit_conversion(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tetra-ascii.stl"
            _write_ascii_stl(path, _tetrahedron())
            scene = load_stl_scene(path, stl_unit="centimeter")
            self.assertEqual(scene.source_unit, "centimeter")
            self.assertEqual(scene.unit_scale_to_mm, 10.0)
            self.assertEqual(scene.object_count, 1)
            self.assertEqual(scene.objects[0].bbox_max_mm, (10.0, 10.0, 10.0))
            self.assertEqual(scene.objects[0].mesh_source, "stl_ascii_connected_mesh")
            self.assertTrue(scene.objects[0].watertight)

    def test_stl_unit_is_mandatory_and_invalid_geometry_fails_closed(self) -> None:
        with self.assertRaisesRegex(CadModelError, "do not store units"):
            load_cad_scene(STL_PATH)
        with self.assertRaisesRegex(CadModelError, "Unsupported STL unit"):
            load_stl_scene(STL_PATH, stl_unit="foot")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trailing = root / "trailing.stl"
            _write_binary_stl(trailing, _tetrahedron())
            trailing.write_bytes(trailing.read_bytes() + b"unexpected")
            with self.assertRaises(CadModelError):
                load_stl_scene(trailing, stl_unit="millimeter")
            degenerate = root / "degenerate.stl"
            triangle = [[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]]
            _write_binary_stl(degenerate, triangle)
            with self.assertRaisesRegex(CadModelError, "zero-area"):
                load_stl_scene(degenerate, stl_unit="millimeter")


if __name__ == "__main__":
    unittest.main()
