from __future__ import annotations

import unittest

import cv2
import numpy as np

from pipe_twin.stereo_camera import (
    LAYOUT_SEPARATE,
    LAYOUT_SIDE_BY_SIDE_LR,
    LAYOUT_SIDE_BY_SIDE_RL,
    StereoCameraError,
    StereoCameraSession,
    probe_video_devices,
    probe_video_modes,
)


class FakeCapture:
    def __init__(self, frame: np.ndarray | None, *, opened: bool = True) -> None:
        self.frame = frame
        self.opened = opened
        self.released = False
        self.properties: dict[int, float] = {}

    def isOpened(self) -> bool:
        return self.opened

    def set(self, key: int, value: float) -> bool:
        self.properties[key] = value
        return True

    def get(self, key: int) -> float:
        if key == cv2.CAP_PROP_FRAME_WIDTH and self.frame is not None:
            return float(self.frame.shape[1])
        if key == cv2.CAP_PROP_FRAME_HEIGHT and self.frame is not None:
            return float(self.frame.shape[0])
        if key == cv2.CAP_PROP_FPS:
            return 30.0
        return self.properties.get(key, 0.0)

    def read(self):
        return self.frame is not None, None if self.frame is None else self.frame.copy()

    def grab(self) -> bool:
        return self.frame is not None

    def retrieve(self):
        return self.read()

    def release(self) -> None:
        self.released = True


class StereoCameraTests(unittest.TestCase):
    def test_side_by_side_frame_is_split_without_resizing(self):
        left = np.full((2, 3, 3), 17, dtype=np.uint8)
        right = np.full((2, 3, 3), 91, dtype=np.uint8)
        capture = FakeCapture(np.concatenate((left, right), axis=1))
        session = StereoCameraSession(
            layout=LAYOUT_SIDE_BY_SIDE_LR,
            left_index=0,
            right_index=None,
            eye_width=3,
            eye_height=2,
            backend=123,
            capture_factory=lambda *_args: capture,
            clock_ns=lambda: 1_000_000_000,
        )
        session.open()
        pair = session.read_pair()
        self.assertTrue(np.array_equal(pair.left, left))
        self.assertTrue(np.array_equal(pair.right, right))
        self.assertEqual(pair.left_captured_at, pair.right_captured_at)
        self.assertEqual(pair.sync_delta_ms, 0.0)
        self.assertEqual(pair.timestamp_source, "HOST_SYSTEM_CLOCK")
        self.assertEqual(pair.provenance["left"]["capture_sync_method"], "SAME_UVC_FRAME")
        self.assertEqual(capture.properties[cv2.CAP_PROP_FRAME_WIDTH], 6)
        self.assertEqual(capture.properties[cv2.CAP_PROP_FRAME_HEIGHT], 2)
        session.close()
        self.assertTrue(capture.released)

    def test_side_by_side_order_can_be_reversed(self):
        first = np.full((2, 2, 3), 1, dtype=np.uint8)
        second = np.full((2, 2, 3), 2, dtype=np.uint8)
        capture = FakeCapture(np.concatenate((first, second), axis=1))
        with StereoCameraSession(
            layout=LAYOUT_SIDE_BY_SIDE_RL,
            left_index=0,
            right_index=None,
            eye_width=2,
            eye_height=2,
            capture_factory=lambda *_args: capture,
            clock_ns=lambda: 1_000_000_000,
        ) as session:
            pair = session.read_pair()
        self.assertTrue(np.array_equal(pair.left, second))
        self.assertTrue(np.array_equal(pair.right, first))
        self.assertEqual(pair.provenance["left"]["side_by_side_order"], "RIGHT_THEN_LEFT")

    def test_two_devices_are_grabbed_as_a_pair_and_keep_device_identity(self):
        captures = {
            2: FakeCapture(np.full((2, 3, 3), 22, dtype=np.uint8)),
            5: FakeCapture(np.full((2, 3, 3), 55, dtype=np.uint8)),
        }
        session = StereoCameraSession(
            layout=LAYOUT_SEPARATE,
            left_index=2,
            right_index=5,
            eye_width=3,
            eye_height=2,
            capture_factory=lambda index, _backend: captures[index],
            clock_ns=lambda: 2_000_000_000,
        )
        session.open()
        pair = session.read_pair()
        self.assertEqual(int(pair.left[0, 0, 0]), 22)
        self.assertEqual(int(pair.right[0, 0, 0]), 55)
        self.assertEqual(pair.sync_delta_ms, 0.0)
        self.assertEqual(pair.provenance["left"]["capture_device_index"], 2)
        self.assertEqual(pair.provenance["right"]["capture_device_index"], 5)
        session.close()

    def test_invalid_indices_or_frame_dimensions_fail_closed(self):
        with self.assertRaisesRegex(StereoCameraError, "different indices"):
            StereoCameraSession(
                layout=LAYOUT_SEPARATE,
                left_index=0,
                right_index=0,
                eye_width=3,
                eye_height=2,
            )
        capture = FakeCapture(np.zeros((2, 3, 3), dtype=np.uint8))
        session = StereoCameraSession(
            layout=LAYOUT_SIDE_BY_SIDE_LR,
            left_index=0,
            right_index=None,
            eye_width=3,
            eye_height=2,
            capture_factory=lambda *_args: capture,
        )
        session.open()
        with self.assertRaisesRegex(StereoCameraError, "标定要求"):
            session.read_pair()
        session.close()

    def test_probe_releases_every_attempt_and_reports_open_devices(self):
        captures: list[FakeCapture] = []

        def factory(index: int, _backend: int) -> FakeCapture:
            capture = FakeCapture(
                np.zeros((480, 640, 3), dtype=np.uint8),
                opened=index in {0, 2},
            )
            captures.append(capture)
            return capture

        devices = probe_video_devices(
            maximum_index=3,
            backend=123,
            capture_factory=factory,
        )
        self.assertEqual([item["index"] for item in devices], [0, 2])
        self.assertTrue(all(capture.released for capture in captures))

    def test_probe_video_modes_reports_delivered_sizes_deduplicated(self):
        class ModeCapture:
            def __init__(self) -> None:
                self.size = (0, 0)
                self.released = True

            def isOpened(self) -> bool:
                return True

            def set(self, key: int, value: float) -> bool:
                if key == cv2.CAP_PROP_FRAME_WIDTH:
                    self.size = (int(value), self.size[1])
                if key == cv2.CAP_PROP_FRAME_HEIGHT:
                    self.size = (self.size[0], int(value))
                return True

            def read(self):
                width, height = self.size
                return True, np.zeros((height, width, 3), dtype=np.uint8)

            def release(self) -> None:
                self.released = True

        capture = ModeCapture()
        modes = probe_video_modes(
            index=0,
            candidates=[(2560, 720), (640, 480), (2560, 720)],
            capture_factory=lambda *_args: capture,
            per_mode_timeout_s=2.0,
        )
        self.assertEqual(modes, [(2560, 720), (640, 480)])

    def test_probe_video_modes_bounds_a_hung_driver(self):
        import time

        def hanging_factory(_index: int, _backend: int):
            time.sleep(1.0)
            raise AssertionError("the hung open should be abandoned, not joined")

        started = time.time()
        modes = probe_video_modes(
            index=0,
            candidates=[(2560, 720)],
            capture_factory=hanging_factory,
            per_mode_timeout_s=0.2,
        )
        self.assertEqual(modes, [])
        self.assertLess(time.time() - started, 5.0)


if __name__ == "__main__":
    unittest.main()
