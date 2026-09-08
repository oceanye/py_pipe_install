from __future__ import annotations

import json
import struct
import tempfile
import unittest
import zlib
from pathlib import Path

import cv2
import numpy as np

from pipe_twin.calibration_wizard import (
    CHESS_ID_MARKER,
    BoardObservation,
    Rectifier,
    _canonicalize_corner_order,
    align_pair_orientation,
    build_rectification_recipe,
    detect_board_corners,
    pose_diversity_report,
    printable_chessboard_png,
    rectifier_for_calibration,
    solve_stereo_calibration,
    write_printable_chessboard_png,
)
from pipe_twin.capture_gui import field_calibration_problem
from pipe_twin.stereo_analyzer import _calibration_from_manifest


SIZE = (640, 480)
INTRINSIC = np.asarray([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
DISTORTION = np.asarray([-0.15, 0.03, 0.0, 0.0, 0.0])
BASELINE_MM = 95.0
SQUARE_MM = 40.0
PATTERN = (8, 6)


def _object_points() -> np.ndarray:
    grid = np.zeros((PATTERN[0] * PATTERN[1], 3), dtype=np.float64)
    grid[:, :2] = np.mgrid[0 : PATTERN[0], 0 : PATTERN[1]].T.reshape(-1, 2)
    return grid * SQUARE_MM


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    return np.asarray(cv2.Rodrigues(axis * angle)[0], dtype=float)


def _synthetic_pairs(count: int = 12) -> list[tuple[BoardObservation, BoardObservation]]:
    """Project a known raw rig (K/D, 95 mm baseline) across diverse poses.

    Odd pairs deliberately hand the right eye a 180-degree-relabeled grid so
    the solver's per-pair orientation alignment is exercised, mirroring what
    the real detector can do between the two eyes of one instant.
    """
    object_points = _object_points()
    left_to_right_rotation = _rotation(np.asarray([0.0, 1.0, 0.0]), np.deg2rad(0.4))
    left_to_right_translation = np.asarray([-BASELINE_MM, 0.0, 0.0])
    pairs = []
    for index in range(count):
        column, row = index % 3, index // 3
        board_to_left_rotation = (
            _rotation(np.asarray([0.0, 0.0, 1.0]), np.deg2rad((column - 1) * 6.0))
            @ _rotation(np.asarray([1.0, 0.0, 0.0]), np.deg2rad((row - 1) * 5.0))
        )
        board_to_left_translation = np.asarray(
            [(column - 1) * 300.0, (row - 1) * 180.0, 900.0 + 60.0 * row], dtype=float
        )
        board_to_right_rotation = left_to_right_rotation @ board_to_left_rotation
        board_to_right_translation = (
            left_to_right_rotation @ board_to_left_translation + left_to_right_translation
        )
        corners = {}
        for role, rotation, translation in (
            ("left", board_to_left_rotation, board_to_left_translation),
            ("right", board_to_right_rotation, board_to_right_translation),
        ):
            image_points, _ = cv2.projectPoints(
                object_points,
                cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))[0],
                np.asarray(translation, dtype=np.float64),
                INTRINSIC,
                DISTORTION,
            )
            ordered = _canonicalize_corner_order(image_points.reshape(-1, 2), PATTERN)
            zones = [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2)]
            flat_index = row * 3 + column
            corners[role] = BoardObservation(
                corners_px=ordered,
                sharpness=100.0,
                centroid_zone=zones[flat_index % len(zones)],
            )
        right = corners["right"]
        if index % 2 == 1:
            columns, rows = PATTERN
            flipped_grid = right.corners_px.reshape(rows, columns, 2)[::-1, ::-1]
            right = BoardObservation(flipped_grid.reshape(-1, 2).copy(), right.sharpness, right.centroid_zone)
        pairs.append((corners["left"], right))
    return pairs


def _solve(pairs=None, **overrides):
    arguments = dict(
        pairs=pairs if pairs is not None else _synthetic_pairs(),
        image_size=SIZE,
        square_mm=SQUARE_MM,
        pattern=PATTERN,
        max_reprojection_rms_px=0.5,
        min_pairs=10,
        operator="field",
        max_sync_delta_ms=1.0,
        layout="side_by_side_left_right",
    )
    arguments.update(overrides)
    return solve_stereo_calibration(**arguments)


class ChessboardWizardTests(unittest.TestCase):
    def test_opencv_chessboard_symbols_available(self):
        for name in (
            "findChessboardCornersSB",
            "cornerSubPix",
            "calibrateCamera",
            "stereoCalibrate",
            "stereoRectify",
            "initUndistortRectifyMap",
        ):
            self.assertTrue(hasattr(cv2, name), f"cv2.{name} is required by the wizard")

    def test_printable_chessboard_png_has_dpi_metadata_and_metric_labels(self):
        payload = printable_chessboard_png(square_mm=15.0, columns=9, rows=7, dpi=300)
        self.assertTrue(payload.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertIn(b"pHYs", payload[:200])
        pixels_per_metre = 300 / 0.0254
        self.assertIn(struct.pack(">IIB", round(pixels_per_metre), round(pixels_per_metre), 1), payload)
        # tEXt chunk carries the metric metadata for later verification.
        text = payload[payload.index(b"tEXt") : payload.index(b"tEXt") + 400]
        self.assertIn(b"square_mm=15.000", text)
        self.assertIn(b"columns=9", text)
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape[:2], (round(297 * 300 / 25.4), round(210 * 300 / 25.4)))

    def test_printable_chessboard_rejects_oversized_board(self):
        with self.assertRaisesRegex(Exception, "超出 A4"):
            printable_chessboard_png(square_mm=60.0, columns=20, rows=30, dpi=300)

    def test_write_printable_chessboard_png_writes_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = write_printable_chessboard_png(
                Path(temp) / "board.png", square_mm=15.0, columns=9, rows=7, dpi=150
            )
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 10000)

    def test_corner_order_canonicalization_is_stable_under_180_flip(self):
        grid = np.mgrid[0:6, 0:8].T.reshape(-1, 2).astype(float) * 10.0
        canonical = _canonicalize_corner_order(grid, PATTERN)
        flipped = _canonicalize_corner_order(grid[::-1].copy(), PATTERN)
        np.testing.assert_allclose(canonical, flipped)

    def test_detection_finds_board_and_matches_under_image_rotation(self):
        square = 40
        board = np.full(((7 + 4) * square, (9 + 4) * square), 255, np.uint8)
        for row in range(7):
            for column in range(9):
                if (row + column) % 2 == 0:
                    board[(row + 2) * square : (row + 3) * square, (column + 2) * square : (column + 3) * square] = 0
        scene = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
        direct = detect_board_corners(scene, pattern=PATTERN)
        self.assertIsNotNone(direct)
        rotated = detect_board_corners(cv2.rotate(scene, cv2.ROTATE_180), pattern=PATTERN)
        self.assertIsNotNone(rotated)
        self.assertEqual(direct.corners_px.shape, (PATTERN[0] * PATTERN[1], 2))
        # A 180-degree view may start its grid at the opposite physical
        # corner; pair orientation alignment must restore the labelling.
        realigned = align_pair_orientation(direct, rotated, pattern=PATTERN)
        np.testing.assert_allclose(direct.corners_px, realigned.corners_px, atol=1.5)
        self.assertEqual(pose_diversity_report([direct], image_size=scene.shape[:2][::-1])["unique_zones"], 1)

    def test_synthetic_raw_rig_recovers_known_intrinsics_baseline_and_distortion(self):
        result = _solve()
        self.assertTrue(result.validated, result.rejection_reasons)
        calibration = _calibration_from_manifest(result.calibration)
        self.assertAlmostEqual(calibration.left.fx, 800.0, delta=16.0)
        self.assertAlmostEqual(calibration.left.fy, 800.0, delta=16.0)
        self.assertAlmostEqual(calibration.baseline_mm, BASELINE_MM, delta=1.0)
        self.assertLess(result.stereo_rms_px, 0.5)
        self.assertTrue(result.audit["p1_p2_k_identical"])

    def test_wizard_calibration_satisfies_contract_and_field_gate(self):
        result = _solve()
        _calibration_from_manifest(result.calibration)
        self.assertIsNone(field_calibration_problem(result.calibration))
        self.assertIn(CHESS_ID_MARKER, result.calibration["calibration_id"])
        for marker in ("SYNTHETIC", "DEMO", "EXAMPLE", "REPLACE_WITH_REAL"):
            self.assertNotIn(marker.upper(), result.calibration["calibration_id"].upper())

    def test_recipe_and_rectifier_remap_matches_contract_image_size(self):
        result = _solve()
        recipe = build_rectification_recipe(
            calibration_id=result.calibration["calibration_id"],
            K1=np.eye(3),
            D1=np.zeros(5),
            K2=np.eye(3),
            D2=np.zeros(5),
            R1=np.eye(3),
            R2=np.eye(3),
            P1=np.hstack([np.eye(3), np.zeros((3, 1))]),
            P2=np.hstack([np.eye(3), [[-90.0], [0.0], [0.0]]]),
            image_size=SIZE,
        )
        self.assertEqual(
            json.loads(json.dumps(recipe))["image_width_px"],
            result.calibration["left_camera"]["width"],
        )
        rectifier = Rectifier(result.recipe)
        frame = np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8)
        for role in ("left", "right"):
            output = rectifier.rectify(role, frame)
            self.assertEqual(output.shape, (SIZE[1], SIZE[0], 3))
        with self.assertRaisesRegex(Exception, "尺寸"):
            rectifier.rectify("left", np.zeros((100, 120, 3), dtype=np.uint8))

    def test_rectifier_for_calibration_is_fail_closed(self):
        result = _solve()
        profile = {"rectification_recipe": result.recipe}
        self.assertIsInstance(rectifier_for_calibration(result.calibration, profile), Rectifier)
        # A wizard calibration without its matching recipe must never silently
        # analyse raw frames with the rectified K.
        with self.assertRaisesRegex(Exception, "极线矫正配方"):
            rectifier_for_calibration(result.calibration, {})
        # External calibrations keep today's already-rectified assumption.
        external = dict(result.calibration)
        external["calibration_id"] = "VENDOR-RIG-V2"
        self.assertIsNone(rectifier_for_calibration(external, profile))
        self.assertIsNone(rectifier_for_calibration(external, {}))

    def test_per_view_rms_gate_rejects_a_corrupted_pair(self):
        pairs = _synthetic_pairs()
        noisy = list(pairs)
        left, right = noisy[0]
        perturbed = BoardObservation(
            corners_px=right.corners_px + np.asarray([25.0, -18.0]),
            sharpness=right.sharpness,
            centroid_zone=right.centroid_zone,
        )
        noisy[0] = (left, perturbed)
        result = _solve(pairs=noisy)
        self.assertFalse(result.validated)
        self.assertTrue(any("重投影误差过大" in reason for reason in result.rejection_reasons))

    def test_wizard_rejects_insufficient_or_repeated_poses(self):
        few = _solve(pairs=_synthetic_pairs(5))
        self.assertFalse(few.validated)
        self.assertTrue(any("照片对不足" in reason for reason in few.rejection_reasons))
        pairs = _synthetic_pairs(12)
        flattened = [
            (BoardObservation(item[0].corners_px, item[0].sharpness, (1, 1)), item[1])
            for item in pairs
        ]
        result = _solve(pairs=flattened)
        self.assertFalse(result.validated)
        self.assertTrue(any("姿态太单一" in reason for reason in result.rejection_reasons))

    def test_detection_overlay_draws_canonicalized_corners_without_error(self):
        # Regression: canonicalized corners are float64 while
        # cv2.drawChessboardCorners requires CV_32FC2; the overlay must
        # degrade to the plain frame instead of killing the preview.
        from pipe_twin.calibration_wizard import draw_detection_overlay

        square = 40
        board = np.full(((7 + 4) * square, (9 + 4) * square), 255, np.uint8)
        for row in range(7):
            for column in range(9):
                if (row + column) % 2 == 0:
                    board[(row + 2) * square : (row + 3) * square, (column + 2) * square : (column + 3) * square] = 0
        scene = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
        observation = detect_board_corners(scene, pattern=PATTERN)
        self.assertIsNotNone(observation)
        self.assertEqual(observation.corners_px.dtype, np.float64)
        overlay = draw_detection_overlay(scene, observation, PATTERN)
        self.assertEqual(overlay.shape, scene.shape)
        self.assertTrue(np.any(overlay != scene))

    def test_save_wizard_result_persists_calibration_recipe_and_profile(self):
        from pipe_twin.workbench_profile import load_profile

        result = _solve()
        with tempfile.TemporaryDirectory() as temp:
            from pipe_twin import workbench_profile

            original_profile = workbench_profile.default_profile_path
            original_calibration = workbench_profile.default_calibration_path
            root = Path(temp)
            workbench_profile.default_profile_path = lambda: root / "workbench_profile.json"
            workbench_profile.default_calibration_path = lambda: root / "calibration_current.json"
            try:
                from pipe_twin.calibration_wizard import save_wizard_result

                path = save_wizard_result(result)
                self.assertEqual(path, root / "calibration_current.json")
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["stereo_calibration"]["validated"], True)
                profile, problem = load_profile()
                self.assertEqual(problem, "")
                self.assertEqual(profile["calibration_current"]["calibration_id"], result.calibration["calibration_id"])
                self.assertEqual(
                    profile["rectification_recipe"]["calibration_id"],
                    result.calibration["calibration_id"],
                )
            finally:
                workbench_profile.default_profile_path = original_profile
                workbench_profile.default_calibration_path = original_calibration


if __name__ == "__main__":
    unittest.main()
