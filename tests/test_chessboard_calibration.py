from __future__ import annotations

import json
import struct
import tempfile
import threading
import unittest
import zlib
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from pipe_twin.calibration_wizard import (
    CHESS_ID_MARKER,
    BoardObservation,
    ChessboardWizardDialog,
    Rectifier,
    _canonicalize_corner_order,
    align_pair_orientation,
    build_rectification_recipe,
    calibrate_stereo_from_folders,
    detect_board_corners,
    pose_diversity_report,
    preferred_stream_mode,
    printable_chessboard_png,
    rectifier_for_calibration,
    replay_calibration_diagnostic,
    solve_stereo_calibration,
    write_calibration,
    write_calibration_diagnostic,
    write_printable_chessboard_png,
)
from pipe_twin.capture_gui import field_calibration_problem
from pipe_twin.stereo_analyzer import _calibration_from_manifest
from pipe_twin.workbench_profile import load_camera_calibration_bundle


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
        # Keep the complete board inside both 640x480 eyes while still
        # spanning the image, depth and out-of-plane tilt.  Real detectors
        # cannot return corners outside the source frame.
        board_to_left_translation = np.asarray(
            [
                (column - 1) * 385.0 - 110.0,
                (row - 1.5) * 210.0 - 70.0,
                1400.0 + 120.0 * row,
            ],
            dtype=float,
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

    def test_pair_alignment_uses_grid_direction_when_disparity_exceeds_board_width(self):
        columns, rows = PATTERN
        xx, yy = np.meshgrid(np.arange(columns), np.arange(rows))
        grid = np.stack((xx, yy), axis=-1).astype(float)
        grid *= 20.0
        left = BoardObservation(grid.reshape(-1, 2), 100.0, (1, 1))
        correct_right = grid + np.asarray([-240.0, 7.0])
        detector_reversed = correct_right[::-1, ::-1].reshape(-1, 2)
        right = BoardObservation(detector_reversed, 100.0, (1, 1))

        aligned = align_pair_orientation(left, right, pattern=PATTERN)

        np.testing.assert_allclose(aligned.corners_px, correct_right.reshape(-1, 2))

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

    def test_sb_native_error_falls_back_without_stopping_preview(self):
        square = 40
        board = np.full(((7 + 4) * square, (9 + 4) * square), 255, np.uint8)
        for row in range(7):
            for column in range(9):
                if (row + column) % 2 == 0:
                    board[(row + 2) * square : (row + 3) * square, (column + 2) * square : (column + 3) * square] = 0
        scene = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
        failures: list[tuple[str, str]] = []
        native_error = cv2.error("Unknown C++ exception from OpenCV code")
        with mock.patch.object(cv2, "findChessboardCornersSB", side_effect=native_error):
            observation = detect_board_corners(
                scene,
                pattern=PATTERN,
                on_cv_error=lambda stage, error: failures.append((stage, str(error))),
            )
        self.assertIsNotNone(observation)
        self.assertEqual(failures, [("find_chessboard_corners_sb", str(native_error))])

    def test_sb_detector_does_not_enable_partial_board_mode(self):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        seen_flags: list[int] = []

        def sb(_gray, _pattern, flags):
            seen_flags.append(flags)
            return False, None

        with mock.patch.object(cv2, "findChessboardCornersSB", side_effect=sb):
            detect_board_corners(image, pattern=PATTERN)
        self.assertEqual(len(seen_flags), 1)
        self.assertFalse(seen_flags[0] & cv2.CALIB_CB_LARGER)

    def test_live_sb_mode_skips_expensive_accuracy_upsampling(self):
        image = np.zeros((120, 160, 3), dtype=np.uint8)
        seen_flags: list[int] = []

        def sb(_gray, _pattern, flags):
            seen_flags.append(flags)
            return False, None

        with mock.patch.object(cv2, "findChessboardCornersSB", side_effect=sb):
            self.assertIsNone(
                detect_board_corners(image, pattern=PATTERN, sb_accuracy=False)
            )
        self.assertEqual(len(seen_flags), 1)
        self.assertFalse(seen_flags[0] & cv2.CALIB_CB_ACCURACY)

    def test_open_preview_queues_and_cancels_an_active_probe(self):
        dialog = object.__new__(ChessboardWizardDialog)
        dialog._opening = False
        dialog._probing_modes = True
        dialog.open_after_id = "probe-poll"
        dialog._start_after_probe = False
        dialog._probe_cancelled = threading.Event()
        dialog.message = mock.Mock()

        self.assertTrue(dialog.start())
        self.assertTrue(dialog._start_after_probe)
        self.assertTrue(dialog._probe_cancelled.is_set())
        self.assertIn("自动打开预览", dialog.message.set.call_args.args[0])

    def test_synthetic_raw_rig_recovers_known_intrinsics_baseline_and_distortion(self):
        result = _solve()
        self.assertTrue(result.validated, result.rejection_reasons)
        calibration = _calibration_from_manifest(result.calibration)
        self.assertAlmostEqual(calibration.left.fx, 800.0, delta=16.0)
        self.assertAlmostEqual(calibration.left.fy, 800.0, delta=16.0)
        self.assertAlmostEqual(calibration.baseline_mm, BASELINE_MM, delta=1.0)
        self.assertLess(result.stereo_rms_px, 0.5)
        self.assertTrue(result.audit["p1_p2_k_identical"])
        self.assertTrue(
            next(
                candidate
                for candidate in result.audit["right_frame_transform_candidates"]
                if candidate["transform"] == "none"
            )["geometry_compatible"]
        )

    def test_solver_rejects_vertical_baseline_and_reversed_disparity(self):
        source = _synthetic_pairs()
        vertical = [
            (
                left,
                BoardObservation(
                    left.corners_px + np.asarray([0.0, 20.0]),
                    right.sharpness,
                    right.centroid_zone,
                ),
            )
            for left, right in source
        ]
        # Swapping the physical left/right cameras produces a negative
        # horizontal baseline and negative rectified disparity.
        reversed_disparity = [(right, left) for left, right in source]
        for label, pairs, expected_reason in (
            ("vertical", vertical, "水平双目"),
            ("reversed", reversed_disparity, "视差方向"),
        ):
            with self.subTest(label=label):
                result = _solve(pairs=pairs)
                self.assertFalse(result.validated)
                self.assertTrue(
                    any(
                        expected_reason in reason
                        for reason in result.rejection_reasons
                    ),
                    result.rejection_reasons,
                )

    def test_solver_rejects_mixed_corner_detector_conventions(self):
        pairs = _synthetic_pairs()
        first_left, first_right = pairs[0]
        second_left, second_right = pairs[1]
        pairs[0] = (
            BoardObservation(
                first_left.corners_px, first_left.sharpness, first_left.centroid_zone, "sb"
            ),
            BoardObservation(
                first_right.corners_px, first_right.sharpness, first_right.centroid_zone, "sb"
            ),
        )
        pairs[1] = (
            BoardObservation(
                second_left.corners_px,
                second_left.sharpness,
                second_left.centroid_zone,
                "classic",
            ),
            BoardObservation(
                second_right.corners_px,
                second_right.sharpness,
                second_right.centroid_zone,
                "classic",
            ),
        )
        with self.assertRaisesRegex(Exception, "混用了 SB 与 classic"):
            _solve(pairs=pairs)

    def test_solver_detects_and_corrects_a_horizontally_mirrored_right_sensor(self):
        mirrored = []
        for left, right in _synthetic_pairs():
            # Simulate detection after the camera driver mirrored the complete
            # right frame: pixel coordinates reflect and grid columns relabel.
            grid = right.corners_px.reshape(PATTERN[1], PATTERN[0], 2).copy()
            grid[..., 0] = SIZE[0] - 1 - grid[..., 0]
            grid = grid[:, ::-1]
            mirrored.append(
                (
                    left,
                    BoardObservation(
                        grid.reshape(-1, 2).copy(),
                        right.sharpness,
                        right.centroid_zone,
                    ),
                )
            )

        result = _solve(pairs=mirrored)

        self.assertTrue(result.validated, result.rejection_reasons)
        self.assertEqual(result.audit["right_frame_transform"], "flip_horizontal")
        self.assertEqual(result.recipe["right_frame_transform"], "flip_horizontal")
        self.assertAlmostEqual(result.calibration["baseline_mm"], BASELINE_MM, delta=1.0)
        self.assertLess(result.stereo_rms_px, 0.5)

    def test_failed_solve_diagnostic_round_trips_all_corner_points(self):
        pairs = _synthetic_pairs()
        result = _solve(pairs=pairs)
        with tempfile.TemporaryDirectory() as temp:
            path = write_calibration_diagnostic(
                pairs,
                result,
                image_size=SIZE,
                pattern=PATTERN,
                square_mm=SQUARE_MM,
                root=temp,
            )
            with np.load(path) as payload:
                self.assertEqual(payload["left_points"].shape, (12, 48, 2))
                self.assertEqual(payload["right_points"].shape, (12, 48, 2))
                metadata = json.loads(str(payload["metadata_json"]))
            self.assertEqual(metadata["pattern_inner_corners"], [8, 6])
            replayed = replay_calibration_diagnostic(path)
            self.assertTrue(replayed.validated, replayed.rejection_reasons)
            self.assertAlmostEqual(
                replayed.calibration["baseline_mm"], BASELINE_MM, delta=1.0
            )

    def test_offline_writer_keeps_rectification_recipe_in_portable_bundle(self):
        result = _solve()
        calibration = dict(result.calibration)
        calibration["source_audit"] = {
            "auto_calibration": {"rectification_recipe": result.recipe}
        }
        with tempfile.TemporaryDirectory() as temp:
            path = write_calibration(Path(temp) / "camera.json", calibration)
            loaded = load_camera_calibration_bundle(path)
        self.assertEqual(
            loaded["rectification_recipe"]["calibration_id"],
            result.calibration["calibration_id"],
        )

    def test_offline_left_pose_moves_both_cameras_as_one_rigid_rig(self):
        result = _solve()
        rotation = _rotation(np.asarray([0.0, 0.0, 1.0]), np.deg2rad(90.0))
        center = np.asarray([100.0, 200.0, 300.0])
        observation = BoardObservation(np.zeros((54, 2)), 100.0, (1, 1), "classic")
        with tempfile.TemporaryDirectory() as temp:
            left_dir = Path(temp) / "left"
            right_dir = Path(temp) / "right"
            left_dir.mkdir()
            right_dir.mkdir()
            for index in range(8):
                (left_dir / f"{index:02}.png").write_bytes(b"x")
                (right_dir / f"{index:02}.png").write_bytes(b"x")
            with (
                mock.patch(
                    "pipe_twin.calibration_wizard.cv2.imread",
                    return_value=np.zeros((SIZE[1], SIZE[0], 3), dtype=np.uint8),
                ),
                mock.patch(
                    "pipe_twin.calibration_wizard.detect_board_corners",
                    return_value=observation,
                ),
                mock.patch(
                    "pipe_twin.calibration_wizard.solve_stereo_calibration",
                    return_value=result,
                ),
            ):
                calibration = calibrate_stereo_from_folders(
                    left_dir,
                    right_dir,
                    left_camera_pose={
                        "rotation_world_to_camera": rotation.tolist(),
                        "center_world_mm": center.tolist(),
                    },
                )
        parsed = _calibration_from_manifest(calibration)
        np.testing.assert_allclose(parsed.left.center_world_mm, center)
        np.testing.assert_allclose(parsed.left.rotation_world_to_camera, rotation)
        np.testing.assert_allclose(parsed.right.rotation_world_to_camera, rotation)
        baseline_camera = rotation @ (
            parsed.right.center_world_mm - parsed.left.center_world_mm
        )
        np.testing.assert_allclose(baseline_camera, [BASELINE_MM, 0.0, 0.0], atol=1e-3)
        self.assertTrue(parsed.registration_validated)
        self.assertIn("-pose-", parsed.calibration_id)

    def test_solver_keeps_independent_intrinsics_when_stereo_gate_is_strict(self):
        rng = np.random.default_rng(42)
        noisy = []
        for left, right in _synthetic_pairs():
            noisy.append(
                (
                    BoardObservation(
                        left.corners_px + rng.normal(0.0, 0.5, left.corners_px.shape),
                        left.sharpness,
                        left.centroid_zone,
                    ),
                    BoardObservation(
                        right.corners_px + rng.normal(0.0, 0.5, right.corners_px.shape),
                        right.sharpness,
                        right.centroid_zone,
                    ),
                )
            )
        object_lists = [_object_points().astype(np.float32)] * len(noisy)
        left_points = [
            pair[0].corners_px.reshape(-1, 1, 2).astype(np.float32)
            for pair in noisy
        ]
        _rms, separate_k, _d, _rvecs, _tvecs = cv2.calibrateCamera(
            object_lists,
            left_points,
            SIZE,
            None,
            None,
            flags=cv2.CALIB_FIX_K3,
        )
        result = _solve(pairs=noisy, max_reprojection_rms_px=0.01)
        self.assertEqual(result.audit["solve_mode"], "fix_intrinsics")
        np.testing.assert_allclose(
            np.asarray(result.recipe["K1"]), separate_k, rtol=1e-4, atol=0.1
        )

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

    def test_rectifier_owns_the_persisted_right_frame_transform(self):
        size = (8, 6)
        intrinsic = np.asarray(
            [[5.0, 0.0, 3.5], [0.0, 5.0, 2.5], [0.0, 0.0, 1.0]]
        )
        projection = np.hstack((intrinsic, np.zeros((3, 1))))
        right_projection = projection.copy()
        right_projection[0, 3] = -intrinsic[0, 0] * 1.0
        recipe = build_rectification_recipe(
            calibration_id="FIELD-CHESS-MIRROR",
            K1=intrinsic,
            D1=np.zeros(5),
            K2=intrinsic,
            D2=np.zeros(5),
            R1=np.eye(3),
            R2=np.eye(3),
            P1=projection,
            P2=right_projection,
            image_size=size,
            right_frame_transform="flip_horizontal",
        )
        frame = np.arange(size[0] * size[1] * 3, dtype=np.uint8).reshape(
            size[1], size[0], 3
        )
        rectifier = Rectifier(recipe)
        np.testing.assert_array_equal(
            rectifier.rectify("right", frame), cv2.flip(frame, 1)
        )
        np.testing.assert_array_equal(rectifier.rectify("left", frame), frame)

    def test_expected_baseline_and_pose_diversity_are_guarded(self):
        wrong_baseline = _solve(expected_baseline_mm=140.0)
        self.assertFalse(wrong_baseline.validated)
        self.assertTrue(
            any("实测镜头中心距" in reason for reason in wrong_baseline.rejection_reasons)
        )

        repeated_pose = _synthetic_pairs()
        first_left, first_right = repeated_pose[0]
        flat = [
            (
                BoardObservation(
                    first_left.corners_px.copy(),
                    first_left.sharpness,
                    (index % 3, index // 4),
                ),
                BoardObservation(
                    first_right.corners_px.copy(),
                    first_right.sharpness,
                    (index % 3, index // 4),
                ),
            )
            for index in range(12)
        ]
        result = _solve(pairs=flat)
        self.assertFalse(result.validated)
        self.assertTrue(
            any("倾斜变化不足" in reason for reason in result.rejection_reasons)
        )

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

    def test_robust_solve_discards_one_corrupted_pair(self):
        pairs = _synthetic_pairs()
        noisy = list(pairs)
        left, right = noisy[0]
        perturbed = BoardObservation(
            corners_px=right.corners_px + np.asarray([15.0, 10.0]),
            sharpness=right.sharpness,
            centroid_zone=right.centroid_zone,
        )
        noisy[0] = (left, perturbed)
        result = _solve(pairs=noisy)
        self.assertTrue(result.validated, result.rejection_reasons)
        self.assertEqual(result.audit["input_pair_count"], 12)
        self.assertEqual(result.audit["discarded_pair_indices"], [1])
        self.assertEqual(result.pair_count, 11)

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

    def test_pose_novelty_gate_for_automatic_capture(self):
        from pipe_twin.calibration_wizard import pose_is_novel

        self.assertTrue(pose_is_novel(None, (100.0, 100.0), SIZE))
        self.assertFalse(pose_is_novel((320.0, 240.0), (326.0, 244.0), SIZE))
        self.assertTrue(pose_is_novel((320.0, 240.0), (420.0, 240.0), SIZE))
        # 6% of the diagonal is the threshold: 48 px on a 800 px diagonal.
        self.assertFalse(pose_is_novel((320.0, 240.0), (360.0, 240.0), SIZE))
        self.assertTrue(pose_is_novel((320.0, 240.0), (370.0, 240.0), SIZE))

    def test_native_high_resolution_stereo_stream_is_preferred(self):
        sizes = [(2560, 720), (1280, 480), (3840, 1080), (1920, 1080)]
        self.assertEqual(
            preferred_stream_mode(sizes, layout="side_by_side_left_right"),
            (3840, 1080),
        )
        self.assertEqual(
            preferred_stream_mode(sizes, layout="separate_devices"),
            (3840, 1080),
        )

    def test_zone_coverage_report_names_missing_areas(self):
        from pipe_twin.calibration_wizard import zone_coverage_report

        report = zone_coverage_report({(1, 1), (2, 1), (1, 2)})
        self.assertEqual(report["covered"], 3)
        self.assertEqual(report["required"], 6)
        self.assertIn("左上", report["missing"])
        self.assertNotIn("中心", report["missing"])
        self.assertEqual(zone_coverage_report(set())["covered"], 0)
        full = zone_coverage_report(
            {(col, row) for col in range(3) for row in range(3)}
        )
        self.assertEqual((full["covered"], full["missing"]), (9, []))

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
