from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree

import cv2
import numpy as np

from pipe_twin.qr_registration import (
    QrRegistrationError,
    detect_qr_pose,
    estimate_square_pose,
    printable_qr_png,
    printable_qr_svg,
    qr_payload,
    register_calibration_from_qr,
)
from pipe_twin.stereo_analyzer import _calibration_from_manifest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "test_model" / "field_stereo_demo_manifest.json"


class QrRegistrationTests(unittest.TestCase):
    def setUp(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.calibration = manifest["stereo_calibration"]
        self.camera = _calibration_from_manifest(self.calibration).left

    def test_printable_svg_has_a4_physical_size_and_reference_line(self):
        svg = printable_qr_svg(marker_id="PIPE-ROOM-A", marker_edge_mm=120.0)
        root = ElementTree.fromstring(svg)
        self.assertEqual(root.attrib["width"], "210mm")
        self.assertEqual(root.attrib["height"], "297mm")
        self.assertIn("edge_mm=120.000", svg)
        self.assertIn('x1="55" y1="270" x2="155" y2="270"', svg)
        self.assertGreater(svg.count("<rect"), 100)

    def test_printable_png_has_a4_300dpi_size_and_decodable_120mm_marker(self):
        png = printable_qr_png(marker_id="PIPE-ROOM-A", marker_edge_mm=120.0)
        image = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(image.shape, (3508, 2480))
        chunk = png.index(b"pHYs")
        x_ppm, y_ppm, unit = struct.unpack(">IIB", png[chunk + 4 : chunk + 13])
        self.assertEqual((x_ppm, y_ppm, unit), (11811, 11811, 1))
        self.assertIn(b"data_edge_px=1417", png)
        camera_sized = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        decoded, corners, _straight = cv2.QRCodeDetector().detectAndDecode(camera_sized)
        self.assertEqual(decoded, qr_payload("PIPE-ROOM-A", 120.0))
        self.assertIsNotNone(corners)

    def test_square_pose_recovers_metric_distance_and_camera_side(self):
        edge = 120.0
        half = edge / 2
        object_points = np.asarray(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
            dtype=np.float64,
        )
        rotation = np.diag([1.0, -1.0, -1.0])
        rvec, _ = cv2.Rodrigues(rotation)
        translation = np.asarray([20.0, -10.0, 1000.0])
        corners, _ = cv2.projectPoints(
            object_points,
            rvec,
            translation,
            self.camera.intrinsic,
            np.zeros(5),
        )
        estimate = estimate_square_pose(
            corners_px=corners,
            marker_edge_mm=edge,
            intrinsic=self.camera.intrinsic,
            decoded_payload=qr_payload("PIPE-ROOM-A", edge),
            source_image_sha256="a" * 64,
        )
        self.assertLess(estimate.reprojection_rms_px, 1e-5)
        self.assertAlmostEqual(estimate.camera_distance_mm, np.linalg.norm(translation), places=5)
        self.assertTrue(
            np.allclose(
                estimate.camera_center_marker_mm,
                -(rotation.T @ translation),
                atol=1e-5,
            )
        )

    def test_qr_pose_places_the_whole_stereo_rig_in_cad_coordinates(self):
        rotation = np.diag([1.0, -1.0, -1.0])
        rvec, _ = cv2.Rodrigues(rotation)
        half = 60.0
        points = np.asarray(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]],
            dtype=np.float64,
        )
        translation = np.asarray([0.0, 0.0, 1000.0])
        corners, _ = cv2.projectPoints(
            points, rvec, translation, self.camera.intrinsic, np.zeros(5)
        )
        estimate = estimate_square_pose(
            corners_px=corners,
            marker_edge_mm=120.0,
            intrinsic=self.camera.intrinsic,
            decoded_payload=qr_payload("PIPE-ROOM-A", 120.0),
            source_image_sha256="b" * 64,
        )
        result = register_calibration_from_qr(
            self.calibration,
            estimate,
            marker_center_world_mm=[100.0, 200.0, 0.0],
            print_right_world="+X",
            print_up_world="+Y",
            registration_validated=True,
        )
        parsed = _calibration_from_manifest(result)
        self.assertTrue(np.allclose(parsed.left.center_world_mm, [100, 200, 1000]))
        self.assertTrue(np.allclose(parsed.right.center_world_mm, [195, 200, 1000]))
        self.assertTrue(
            np.allclose(
                parsed.left.rotation_world_to_camera.T @ [0, 0, 1],
                [0, 0, -1],
            )
        )
        self.assertTrue(parsed.registration_validated)
        self.assertEqual(
            result["registration_adjustment"]["mode"],
            "qr_single_planar_control",
        )

    def test_detector_decodes_a_generated_marker(self):
        payload = qr_payload("PIPE-ROOM-A", 120.0)
        marker = cv2.QRCodeEncoder_create().encode(payload)
        marker = cv2.resize(
            marker,
            (marker.shape[1] * 16, marker.shape[0] * 16),
            interpolation=cv2.INTER_NEAREST,
        )
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "marker.png"
            self.assertTrue(cv2.imwrite(str(path), marker))
            intrinsic = np.asarray(
                [[900.0, 0.0, marker.shape[1] / 2], [0.0, 900.0, marker.shape[0] / 2], [0, 0, 1]]
            )
            estimate = detect_qr_pose(
                path,
                expected_payload=payload,
                marker_edge_mm=114.5,
                intrinsic=intrinsic,
                expected_size=(marker.shape[1], marker.shape[0]),
            )
        self.assertEqual(estimate.decoded_payload, payload)
        self.assertEqual(estimate.marker_edge_mm, 114.5)
        self.assertLess(estimate.reprojection_rms_px, 2.0)

    def test_invalid_orientation_or_reprojection_is_rejected(self):
        estimate = estimate_square_pose(
            corners_px=[[900, 480], [1020, 480], [1020, 600], [900, 600]],
            marker_edge_mm=120.0,
            intrinsic=self.camera.intrinsic,
            decoded_payload=qr_payload("PIPE-ROOM-A", 120.0),
            source_image_sha256="c" * 64,
        )
        with self.assertRaisesRegex(QrRegistrationError, "互相垂直"):
            register_calibration_from_qr(
                self.calibration,
                estimate,
                marker_center_world_mm=[0, 0, 0],
                print_right_world="+X",
                print_up_world="-X",
                registration_validated=True,
            )
        estimate = estimate.__class__(
            **{**estimate.__dict__, "reprojection_rms_px": 3.0}
        )
        with self.assertRaisesRegex(QrRegistrationError, "不能应用定位"):
            register_calibration_from_qr(
                self.calibration,
                estimate,
                marker_center_world_mm=[0, 0, 0],
                print_right_world="+X",
                print_up_world="+Y",
                registration_validated=True,
                max_reprojection_rms_px=2.0,
            )


if __name__ == "__main__":
    unittest.main()
