from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from pipe_twin.gui import build_dashboard_model
from pipe_twin.stereo_analyzer import StereoDepthResult, analyze_stereo_capture

try:
    import rhino3dm
except ImportError:  # pragma: no cover - exercised when the optional reader is absent.
    rhino3dm = None


@unittest.skipUnless(rhino3dm is not None, "rhino3dm is required for the 3DM integration test")
class Stereo3dmIntegrationTests(unittest.TestCase):
    """Exercise the complete native-3DM -> stereo report -> GUI binding path."""

    _GUID = uuid.UUID("12345678-1234-5678-9abc-1234567890ab")

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def _write_model(cls, path: Path) -> str:
        assert rhino3dm is not None
        model = rhino3dm.File3dm()
        model.Settings.ModelUnitSystem = rhino3dm.UnitSystem.Millimeters
        layer = rhino3dm.Layer()
        layer.Name = "Pipes"
        layer_index = model.Layers.Add(layer)

        # A 200 x 30 x 20 mm mesh centred at z=1000 mm.  The dimensions line
        # up with the synthetic 120 x 100 camera images below (fx=fy=300).
        mesh = rhino3dm.Mesh()
        for point in (
            (-100, -15, 990),
            (100, -15, 990),
            (100, 15, 990),
            (-100, 15, 990),
            (-100, -15, 1010),
            (100, -15, 1010),
            (100, 15, 1010),
            (-100, 15, 1010),
        ):
            mesh.Vertices.Add(*point)
        for face in (
            (0, 1, 2, 3),
            (4, 7, 6, 5),
            (0, 4, 5, 1),
            (1, 5, 6, 2),
            (2, 6, 7, 3),
            (3, 7, 4, 0),
        ):
            mesh.Faces.AddFace(*face)

        attributes = rhino3dm.ObjectAttributes()
        attributes.Id = cls._GUID
        attributes.Name = "Pipe A"
        attributes.LayerIndex = layer_index
        attributes.ColorSource = rhino3dm.ObjectColorSource.ColorFromObject
        attributes.ObjectColor = (255, 0, 0, 255)
        attributes.SetUserString("pipe_id", "PIPE-A")
        added = model.Objects.AddMesh(mesh, attributes)
        if str(added).lower() != str(cls._GUID).lower():
            raise AssertionError("rhino3dm did not preserve the test GUID")

        # This unbound Brep has no cached render mesh.  The analyzer must
        # ignore it because it is not in the manifest-bound required GUID set.
        auxiliary = rhino3dm.Brep.CreateFromBoundingBox(
            rhino3dm.BoundingBox(0, 0, 0, 1, 1, 1)
        )
        auxiliary_attributes = rhino3dm.ObjectAttributes()
        auxiliary_attributes.Id = uuid.UUID("abcdefab-cdef-abcd-efab-cdefabcdefab")
        auxiliary_attributes.Name = "Unbound construction solid"
        auxiliary_attributes.LayerIndex = layer_index
        model.Objects.AddBrep(auxiliary, auxiliary_attributes)

        if not model.Write(str(path), 8):
            raise AssertionError("rhino3dm could not write the test model")
        return str(cls._GUID)

    @classmethod
    def _write_png(cls, path: Path, image: np.ndarray) -> str:
        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise AssertionError("OpenCV could not encode the test image")
        path.write_bytes(encoded.tobytes())
        return cls._sha256(path)

    @staticmethod
    def _mock_depth(value: float = 1000.0) -> StereoDepthResult:
        shape = (100, 120)
        depth = np.full(shape, value, dtype=np.float32)
        valid = np.ones(shape, dtype=bool)
        return StereoDepthResult(
            left_depth_mm=depth,
            right_depth_mm=depth.copy(),
            left_valid=valid,
            right_valid=valid.copy(),
            audit={
                "status": "VALID",
                "reason_codes": [],
                "valid_left_fraction": 1.0,
                "valid_right_fraction": 1.0,
            },
        )

    @classmethod
    def _make_manifest(cls, root: Path, model_path: Path, guid: str) -> tuple[Path, dict]:
        left = np.full((100, 120, 3), 128, dtype=np.uint8)
        right = left.copy()
        cv2.line(left, (30, 50), (90, 50), (0, 0, 255), 9, cv2.LINE_8)
        cv2.line(right, (15, 50), (75, 50), (0, 0, 255), 9, cv2.LINE_8)
        left_path = root / "left.png"
        right_path = root / "right.png"
        left_hash = cls._write_png(left_path, left)
        right_hash = cls._write_png(right_path, right)

        def view(path: str, digest: str, camera_id: str) -> dict:
            return {
                "camera_id": camera_id,
                "path": path,
                "sha256": digest,
                "expected_width": 120,
                "expected_height": 100,
                "captured_at": "2026-09-03T00:00:00+08:00",
                "timestamp_source": "MANIFEST_OPERATOR_CONFIRMED",
                "orientation_policy": "RAW_PIXELS_NO_EXIF_TRANSFORM",
            }

        manifest = {
            "schema_version": "2.0",
            "dataset_id": "3dm-integration-test",
            "model_revision": "native-3dm-v1",
            "model": {
                "path": model_path.name,
                "sha256": cls._sha256(model_path),
                "unit": "millimeter",
                # Uppercase is intentional: GUID matching must be case-insensitive.
                "pipes": [
                    {
                        "instance_id": 1,
                        "pipe_id": "PIPE-A",
                        "cad_object_id": guid.upper(),
                        "cad_uuid": guid.upper(),
                        "layer_id": "Pipes",
                        "color_class": "red",
                        "color_srgb": "#FF0000",
                        "nominal_diameter_mm": 30.0,
                        "centerline_world_mm": [[-100, 0, 1000], [100, 0, 1000]],
                    }
                ],
            },
            "stereo_calibration": {
                "calibration_id": "native-3dm-cal-v1",
                "validated": True,
                "registration_validated": True,
                "rectified": True,
                "baseline_mm": 50.0,
                "max_sync_delta_ms": 5.0,
                "left_camera": {
                    "camera_id": "camera-left",
                    "width": 120,
                    "height": 100,
                    "fx": 300.0,
                    "fy": 300.0,
                    "cx": 60.0,
                    "cy": 50.0,
                    "rotation_world_to_camera": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "center_world_mm": [0.0, 0.0, 0.0],
                    "D": [0, 0, 0, 0, 0],
                },
                "right_camera": {
                    "camera_id": "camera-right",
                    "width": 120,
                    "height": 100,
                    "fx": 300.0,
                    "fy": 300.0,
                    "cx": 60.0,
                    "cy": 50.0,
                    "rotation_world_to_camera": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "center_world_mm": [50.0, 0.0, 0.0],
                    "D": [0, 0, 0, 0, 0],
                },
            },
            "capture": {
                "kind": "stereo_still_capture_set",
                "camera_layout": "stereo",
                "capture_group_id": "inspection-current",
                "interval_minutes": 30,
                "capture_groups": [
                    {
                        "capture_id": "pair-001",
                        "views": {
                            "left": view("left.png", left_hash, "camera-left"),
                            "right": view("right.png", right_hash, "camera-right"),
                        },
                    }
                ],
            },
            "analysis": {
                "minimum_focus_laplacian_variance": 0.0,
                "minimum_luminance_p05": 0.0,
                "maximum_luminance_p95": 255.0,
                "minimum_amodal_pixels": 10,
                "minimum_valid_depth_fraction": 0.8,
                "minimum_target_depth_fraction": 0.8,
                "minimum_color_support_fraction": 0.6,
                "maximum_width_relative_error": 0.4,
                "depth_tolerance_mm": 20.0,
                "occlusion_margin_mm": 20.0,
                "free_space_margin_mm": 50.0,
                "minimum_free_space_fraction": 0.8,
                "fully_occluded_fraction": 0.9,
                "color_delta_e76_tolerance": 30.0,
                "minimum_repeated_absence_captures": 2,
                "minimum_depth_mm": 100.0,
                "maximum_depth_mm": 2000.0,
                "stereo_matching": {"min_disparity": 0, "num_disparities": 32, "block_size": 5},
            },
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path, manifest

    def test_native_3dm_guid_filter_report_and_gui_binding(self) -> None:
        assert rhino3dm is not None
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / "pipes.3dm"
            guid = self._write_model(model_path)
            manifest_path, manifest = self._make_manifest(root, model_path, guid)

            with mock.patch(
                "pipe_twin.stereo_analyzer._compute_stereo_depth",
                return_value=self._mock_depth(),
            ):
                report = analyze_stereo_capture(manifest_path)

            self.assertEqual(report["model"]["format"], "3dm")
            self.assertEqual(report["model"]["object_binding_validation"], "3DM_GUID_AND_MESH_VALIDATED")
            self.assertEqual(report["model"]["object_count"], 1)
            self.assertEqual(report["counts"], {"INSTALLED": 1, "NOT_INSTALLED": 0, "UNKNOWN": 0})
            self.assertEqual(report["pipes"][0]["cad_object_id"], guid.upper())

            dashboard = build_dashboard_model(
                manifest,
                report,
                manifest_sha256=self._sha256(manifest_path),
                model_actual_sha256=self._sha256(model_path),
            )
            self.assertTrue(dashboard["binding_valid"])
            self.assertEqual(dashboard["counts"], {"INSTALLED": 1, "NOT_INSTALLED": 0, "UNKNOWN": 0})
            self.assertEqual(dashboard["pipes"][0]["installation_state"], "INSTALLED")


if __name__ == "__main__":
    unittest.main()
