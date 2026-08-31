from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import cv2

from pipe_twin.synthetic_stereo import (
    CameraModel,
    MeshGeometry,
    SceneInstance,
    _build_scene,
    _camera_from_dict,
    _disparity_truth,
    _procedural_textured_rgb,
    _stereo_correspondence_valid,
    _validate_rectified_rig,
    compute_view_topology,
    generate_synthetic_stereo,
    load_mesh_geometries,
    render_scene,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = ROOT / "test_model"
MANIFEST_PATH = FIXTURE_DIR / "pipe_group2_manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _downsized_manifest(source: dict[str, object]) -> dict[str, object]:
    """Keep the physical rig unchanged while reducing CI raster cost."""

    manifest = copy.deepcopy(source)
    model = manifest["model"]
    model["path"] = str((FIXTURE_DIR / model["path"]).resolve())
    reference_video = manifest["reference_video"]
    reference_video["path"] = str(
        (FIXTURE_DIR / reference_video["path"]).resolve()
    )

    rig = manifest["virtual_stereo_rig"]
    for camera_name in ("left_camera", "right_camera"):
        camera = rig[camera_name]
        scale = 1.0 / 6.0
        camera["width"] = 320
        camera["height"] = 180
        for field in ("fx", "fy", "cx", "cy"):
            camera[field] = float(camera[field]) * scale

    elevation = manifest["reference_elevation"]
    elevation["width"] = 400
    elevation["height"] = 250
    elevation["scale_px_per_mm"] = 0.375
    elevation["cx"] = 200.0
    elevation["cy"] = 125.0
    return manifest


class SyntheticStereoFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.model_path = FIXTURE_DIR / cls.manifest["model"]["path"]
        cls.geometries = load_mesh_geometries(cls.model_path)
        cls.instances = _build_scene(cls.manifest, cls.geometries)
        cls.elevation_camera = _camera_from_dict(
            "reference_elevation", cls.manifest["reference_elevation"]
        )
        cls.elevation_render = render_scene(cls.instances, cls.elevation_camera)
        (
            cls.elevation_topology,
            cls.elevation_full_masks,
            cls.elevation_visible_masks,
        ) = compute_view_topology(
            cls.instances,
            cls.elevation_camera,
            cls.elevation_render,
        )

    def test_manifest_loads_nine_bound_instances(self) -> None:
        model = self.manifest["model"]
        pipes = model["pipes"]

        self.assertEqual(self.manifest["domain"], "synthetic_cad_truth")
        self.assertEqual(model["expected_object_count"], 9)
        self.assertEqual(len(self.geometries), 9)
        self.assertEqual(len(self.instances), 9)
        self.assertEqual({item.instance_id for item in self.instances}, set(range(1, 10)))
        self.assertEqual({item.mesh.object_id for item in self.instances}, set(self.geometries))

        manifest_by_id = {int(item["instance_id"]): item for item in pipes}
        self.assertEqual(len(manifest_by_id), 9, "instance_id must be unique")
        for instance in self.instances:
            with self.subTest(instance_id=instance.instance_id):
                contract = manifest_by_id[instance.instance_id]
                self.assertEqual(instance.pipe_id, contract["pipe_id"])
                self.assertEqual(instance.mesh.uuid, contract["cad_uuid"])
                self.assertEqual(instance.layer_id, contract["layer_id"])
                self.assertEqual(instance.centerline_world_mm.shape, (2, 3))

    def test_rectified_rig_rotation_baseline_and_disparity_formula(self) -> None:
        rig = self.manifest["virtual_stereo_rig"]
        left = _camera_from_dict("left", rig["left_camera"])
        right = _camera_from_dict("right", rig["right_camera"])

        np.testing.assert_allclose(
            left.rotation_world_to_camera @ left.rotation_world_to_camera.T,
            np.eye(3),
            atol=1e-12,
        )
        np.testing.assert_allclose(
            right.rotation_world_to_camera,
            left.rotation_world_to_camera,
            atol=0.0,
        )
        self.assertAlmostEqual(float(np.linalg.det(left.rotation_world_to_camera)), 1.0)

        baseline = float(np.linalg.norm(right.center_world_mm - left.center_world_mm))
        self.assertAlmostEqual(baseline, 95.0, places=9)
        self.assertAlmostEqual(baseline, float(rig["baseline_mm"]), places=9)

        point = np.asarray([[250.0, 165.212569, 0.0]], dtype=float)
        left_uv, left_depth = left.project(point)
        right_uv, right_depth = right.project(point)
        self.assertAlmostEqual(float(left_depth[0]), float(right_depth[0]), places=9)
        self.assertAlmostEqual(float(left_uv[0, 1]), float(right_uv[0, 1]), places=9)

        measured_disparity_px = float(left_uv[0, 0] - right_uv[0, 0])
        expected_disparity_px = float(left.fx) * baseline / float(left_depth[0])
        self.assertAlmostEqual(measured_disparity_px, expected_disparity_px, places=9)
        self.assertGreater(measured_disparity_px, 0.0)

        invalid_rig = copy.deepcopy(rig)
        invalid_rig["baseline_mm"] = 80.0
        with self.assertRaisesRegex(ValueError, "baseline_mm"):
            _validate_rectified_rig(invalid_rig, left, right)

    def test_perspective_triangle_depth_matches_ray_plane_intersection(self) -> None:
        camera = CameraModel(
            camera_id="analytic",
            width=200,
            height=200,
            rotation_world_to_camera=np.eye(3),
            center_world_mm=np.zeros(3),
            near_mm=0.5,
            far_mm=10.0,
            projection="perspective",
            fx=100.0,
            fy=100.0,
            cx=100.0,
            cy=100.0,
        )
        vertices = np.asarray(
            [[-1.0, -1.0, 2.0], [1.0, -1.0, 4.0], [0.0, 1.0, 3.0]],
            dtype=float,
        )
        mesh = MeshGeometry(
            object_id="triangle",
            name="analytic",
            uuid="analytic",
            color_srgb="#FFFFFF",
            measured_diameter_mm=1.0,
            measured_centerline_world_mm=np.asarray([[0, 0, 3], [1, 0, 3]], dtype=float),
            vertices_world_mm=vertices,
            triangles=np.asarray([[0, 1, 2]], dtype=np.int32),
        )
        instance = SceneInstance(
            instance_id=1,
            pipe_id="analytic",
            layer_id="front",
            color_class="white",
            nominal_diameter_mm=1.0,
            centerline_world_mm=mesh.measured_centerline_world_mm,
            mesh=mesh,
        )
        result = render_scene([instance], camera)
        row, column = 100, 100
        self.assertEqual(int(result.instance_id[row, column]), 1)

        normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
        ray = np.asarray(
            [
                (column + 0.5 - camera.cx) / camera.fx,
                (row + 0.5 - camera.cy) / camera.fy,
                1.0,
            ]
        )
        expected_z = float((normal @ vertices[0]) / (normal @ ray))
        self.assertAlmostEqual(float(result.depth_z_mm[row, column]), expected_z, places=5)

    def test_zbuffer_depth_and_instance_ownership_are_consistent(self) -> None:
        result = self.elevation_render
        background = result.instance_id == 0

        self.assertEqual(result.depth_z_mm.dtype, np.dtype(np.float32))
        self.assertEqual(result.instance_id.dtype, np.dtype(np.uint16))
        np.testing.assert_array_equal(np.isnan(result.depth_z_mm), background)
        self.assertTrue(np.all(np.isfinite(result.depth_z_mm[~background])))
        self.assertTrue(np.all(result.depth_z_mm[~background] > self.elevation_camera.near_mm))
        self.assertTrue(np.all(result.depth_z_mm[~background] < self.elevation_camera.far_mm))

        front = next(item for item in self.instances if item.instance_id == 2)
        hidden_back = next(item for item in self.instances if item.instance_id == 7)
        front_only = render_scene([front], self.elevation_camera)
        back_only = render_scene([hidden_back], self.elevation_camera)
        overlap = (front_only.instance_id == 2) & (back_only.instance_id == 7)

        self.assertGreater(int(np.count_nonzero(overlap)), 0)
        self.assertTrue(np.all(result.instance_id[overlap] == 2))
        self.assertTrue(
            np.all(front_only.depth_z_mm[overlap] < back_only.depth_z_mm[overlap])
        )
        np.testing.assert_allclose(
            result.depth_z_mm[overlap],
            front_only.depth_z_mm[overlap],
            rtol=0.0,
            atol=0.0,
        )

        reversed_render = render_scene(list(reversed(self.instances)), self.elevation_camera)
        np.testing.assert_array_equal(reversed_render.instance_id, result.instance_id)
        np.testing.assert_allclose(
            reversed_render.depth_z_mm,
            result.depth_z_mm,
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )

    def test_manifest_rejects_duplicate_cad_binding(self) -> None:
        duplicate = copy.deepcopy(self.manifest)
        duplicate["model"]["pipes"][1]["cad_object_id"] = duplicate["model"]["pipes"][0][
            "cad_object_id"
        ]
        with self.assertRaisesRegex(ValueError, "only one"):
            _build_scene(duplicate, self.geometries)

    def test_generator_rejects_source_directory_and_reference_hash_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "dedicated output"):
            generate_synthetic_stereo(MANIFEST_PATH, FIXTURE_DIR)

        manifest = _downsized_manifest(self.manifest)
        manifest["reference_video"]["sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            manifest_path = temporary_path / "invalid-video-hash.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            output = temporary_path / "generated"
            with self.assertRaisesRegex(ValueError, "Reference video hash"):
                generate_synthetic_stereo(manifest_path, output)
            self.assertFalse(output.exists())

    def test_world_anchored_texture_supports_an_opencv_sgbm_smoke(self) -> None:
        rig = self.manifest["virtual_stereo_rig"]
        left_camera = _camera_from_dict("left", rig["left_camera"])
        right_camera = _camera_from_dict("right", rig["right_camera"])
        left = render_scene(self.instances, left_camera)
        right = render_scene(self.instances, right_camera)
        left_gray = cv2.cvtColor(
            _procedural_textured_rgb(left, left_camera), cv2.COLOR_BGR2GRAY
        )
        right_gray = cv2.cvtColor(
            _procedural_textured_rgb(right, right_camera), cv2.COLOR_BGR2GRAY
        )
        matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=272,
            blockSize=5,
            P1=8 * 5 * 5,
            P2=32 * 5 * 5,
            disp12MaxDiff=2,
            preFilterCap=31,
            uniquenessRatio=5,
            speckleWindowSize=50,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )
        estimated = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        expected = _disparity_truth(
            left.depth_z_mm,
            float(left_camera.fx),
            float(rig["baseline_mm"]),
        )
        correspondence = _stereo_correspondence_valid(
            left,
            left_camera,
            right,
            right_camera,
        )
        assessable = correspondence & np.isfinite(expected)
        valid = assessable & (estimated > 0)
        absolute_error = np.abs(estimated[valid] - expected[valid])

        self.assertGreater(float(np.count_nonzero(valid) / np.count_nonzero(assessable)), 0.98)
        self.assertLess(float(np.percentile(absolute_error, 95)), 2.0)
        self.assertGreater(float(np.mean(absolute_error <= 2.0)), 0.98)

    def test_elevation_topology_matches_expected_front_to_back_relationships(self) -> None:
        topology = self.elevation_topology
        records = {int(item["instance_id"]): item for item in topology["pipes"]}
        expected_states = {
            1: "FULLY_VISIBLE",
            2: "FULLY_VISIBLE",
            3: "FULLY_VISIBLE",
            4: "PARTIALLY_OCCLUDED",
            5: "PARTIALLY_OCCLUDED",
            6: "PARTIALLY_OCCLUDED",
            7: "FULLY_OCCLUDED",
            8: "PARTIALLY_OCCLUDED",
            9: "FULLY_OCCLUDED",
        }
        expected_edges = {
            (1, 4),
            (1, 5),
            (2, 6),
            (2, 7),
            (2, 8),
            (3, 9),
        }

        self.assertEqual(topology["edge_direction"], "occluder_to_occluded_target")
        self.assertEqual(
            {instance_id: item["occlusion_state"] for instance_id, item in records.items()},
            expected_states,
        )
        actual_edges = {
            (int(item["from_instance_id"]), int(item["to_instance_id"]))
            for item in topology["occlusion_edges"]
        }
        self.assertEqual(actual_edges, expected_edges)

        layer_by_id = {item.instance_id: item.layer_id for item in self.instances}
        for source, target in actual_edges:
            with self.subTest(edge=(source, target)):
                self.assertEqual(layer_by_id[source], "front")
                self.assertEqual(layer_by_id[target], "back")

        for instance_id, item in records.items():
            with self.subTest(instance_id=instance_id):
                self.assertEqual(
                    int(item["amodal_pixels_in_frame"]),
                    int(item["visible_pixels"]) + int(item["occluded_pixels"]),
                )
                self.assertEqual(
                    int(item["occluded_pixels"]),
                    sum(int(edge["occluded_pixels"]) for edge in item["occluders"]),
                )
        self.assertEqual(records[7]["visible_pixels"], 0)
        self.assertEqual(records[9]["visible_pixels"], 0)
        self.assertFalse(records[7]["assessable"])
        self.assertFalse(records[9]["assessable"])
        self.assertEqual(
            [item["occluder_instance_id"] for item in records[7]["occluders"]], [2]
        )
        self.assertEqual(
            [item["occluder_instance_id"] for item in records[9]["occluders"]], [3]
        )

    def test_generator_writes_typed_truth_hash_catalog_and_safety_boundaries(self) -> None:
        manifest = _downsized_manifest(self.manifest)
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            manifest_path = temporary_path / "fixture.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            output = temporary_path / "generated"
            generated = generate_synthetic_stereo(manifest_path, output)
            persisted = json.loads(
                (output / "dataset_manifest.json").read_text(encoding="utf-8")
            )

            self.assertEqual(generated, persisted)
            self.assertEqual(generated["source"]["manifest_sha256"], _sha256(manifest_path))
            self.assertEqual(len(generated["instance_catalog"]), 9)
            self.assertFalse(generated["truth_semantics"]["field_calibration_validated"])
            self.assertFalse(generated["truth_semantics"]["installation_state_inferred"])
            self.assertIn("never RGB color", generated["truth_semantics"]["identity_source"])

            records = []
            for product in generated["capture"]["products"].values():
                records.extend(
                    value
                    for value in product.values()
                    if isinstance(value, dict) and "path" in value
                )
            records.extend(generated["reference_elevation"].values())
            records.append(generated["calibration"])
            for record in records:
                with self.subTest(product=record["path"]):
                    product_path = output / record["path"]
                    self.assertTrue(product_path.is_file())
                    self.assertEqual(product_path.stat().st_size, record["size_bytes"])
                    self.assertEqual(_sha256(product_path), record["sha256"])

            for camera_name in ("left", "right"):
                with self.subTest(camera=camera_name):
                    truth_path = output / f"{camera_name}_truth.npz"
                    with np.load(truth_path, allow_pickle=False) as truth:
                        depth = truth["depth_z_mm"]
                        instance_id = truth["instance_id"]
                        disparity = truth["disparity_px"]
                        correspondence = truth["stereo_correspondence_valid"]
                    self.assertEqual(depth.dtype, np.dtype(np.float32))
                    self.assertEqual(instance_id.dtype, np.dtype(np.uint16))
                    self.assertEqual(disparity.dtype, np.dtype(np.float32))
                    self.assertEqual(correspondence.dtype, np.dtype(bool))
                    self.assertEqual(depth.shape, (180, 320))
                    self.assertEqual(instance_id.shape, depth.shape)
                    np.testing.assert_array_equal(np.isnan(depth), instance_id == 0)
                    np.testing.assert_array_equal(np.isnan(disparity), instance_id == 0)
                    np.testing.assert_array_equal(correspondence & (instance_id == 0), False)
                    valid = instance_id > 0
                    fx = float(manifest["virtual_stereo_rig"][f"{camera_name}_camera"]["fx"])
                    baseline = float(manifest["virtual_stereo_rig"]["baseline_mm"])
                    np.testing.assert_allclose(
                        disparity[valid],
                        fx * baseline / depth[valid],
                        rtol=1e-6,
                        atol=1e-6,
                    )
                    self.assertTrue(np.any(correspondence))
                    self.assertTrue(np.any(np.isnan(depth)), "fixture must retain background")
                    self.assertTrue(np.any(instance_id > 0), "fixture must contain pipe pixels")
                    self.assertTrue(set(np.unique(instance_id)).issubset(set(range(10))))

            limitations = " ".join(generated["limitations"]).upper()
            self.assertIn("FULLY_OCCLUDED", limitations)
            self.assertIn("NOT_OBSERVED", limitations)
            self.assertIn("NEVER IMPLY NOT_INSTALLED", limitations)


if __name__ == "__main__":
    unittest.main()
