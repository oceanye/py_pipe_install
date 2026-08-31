from __future__ import annotations

import unittest

from pipe_twin.state import confirmed_intervals, debounce_states


RED = ("pipe-red-d40",)
RED_BLUE = ("pipe-blue-d45", "pipe-red-d40")
ALL = ("pipe-blue-d45", "pipe-red-d40", "pipe-white-d20")


class DebounceStatesContractTests(unittest.TestCase):
    def test_confirms_initial_and_changed_state_causally_without_backfill(self) -> None:
        raw = [RED, RED, RED, RED_BLUE, RED_BLUE, RED_BLUE]

        stable = debounce_states(raw, min_stable_frames=3, fps=60.0)

        self.assertEqual(stable, [None, None, RED, RED, RED, RED_BLUE])
        self.assertEqual(len(stable), len(raw))

    def test_suppresses_a_short_state_spike(self) -> None:
        raw = [ALL, ALL, ALL, RED, RED, ALL, ALL]

        stable = debounce_states(raw, min_stable_frames=3, fps=60.0)

        self.assertEqual(stable, [None, None, ALL, ALL, ALL, ALL, ALL])

    def test_initial_empty_visibility_is_confirmed_and_kept_as_an_interval(self) -> None:
        raw = [(), (), (), ()]

        stable = debounce_states(raw, min_stable_frames=2, fps=60.0)
        intervals = confirmed_intervals(stable, min_stable_frames=2, fps=60.0)

        self.assertEqual(stable, [None, (), (), ()])
        self.assertEqual(
            intervals,
            [
                {
                    "start_frame": 0,
                    "confirmation_frame": 1,
                    "start_seconds": 0.0,
                    "confirmation_seconds": 1 / 60.0,
                    "visible_pipe_ids": [],
                    "end_frame": 3,
                    "end_seconds": 3 / 60.0,
                }
            ],
        )

    def test_causal_intervals_preserve_visible_to_empty_transition(self) -> None:
        raw = [RED, RED, (), (), ()]

        stable = debounce_states(raw, min_stable_frames=2, fps=60.0)
        intervals = confirmed_intervals(stable, min_stable_frames=2, fps=60.0)

        self.assertEqual(stable, [None, RED, RED, (), ()])
        self.assertEqual([item["visible_pipe_ids"] for item in intervals], [[*RED], []])
        self.assertEqual([item["start_frame"] for item in intervals], [0, 2])
        self.assertEqual([item["confirmation_frame"] for item in intervals], [1, 3])

    def test_minimum_one_frame_is_identity(self) -> None:
        raw = [ALL, RED_BLUE, RED, ()]
        self.assertEqual(
            debounce_states(raw, min_stable_frames=1, fps=25.0),
            raw,
        )

    def test_empty_input_returns_empty_output(self) -> None:
        self.assertEqual(
            debounce_states([], min_stable_frames=3, fps=60.0),
            [],
        )

    def test_invalid_timing_arguments_are_rejected(self) -> None:
        for minimum, fps in ((0, 60.0), (-1, 60.0), (3, 0.0), (3, -1.0)):
            with self.subTest(min_stable_frames=minimum, fps=fps):
                with self.assertRaises(ValueError):
                    debounce_states([RED], min_stable_frames=minimum, fps=fps)


if __name__ == "__main__":
    unittest.main()
