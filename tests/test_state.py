from __future__ import annotations

import unittest

from pipe_twin.state import (
    INSTALLATION_STATES,
    NegativeInstallationEvidence,
    classify_installation_state,
    confirmed_intervals,
    debounce_states,
    installation_state_label_zh,
)


RED = ("pipe-red-d40",)
RED_BLUE = ("pipe-blue-d45", "pipe-red-d40")
ALL = ("pipe-blue-d45", "pipe-red-d40", "pipe-white-d20")


def _qualified_negative_evidence(
    **overrides: object,
) -> NegativeInstallationEvidence:
    values: dict[str, object] = {
        "calibration_validated": True,
        "registration_validated": True,
        "expected_region_in_frame": True,
        "expected_region_unoccluded": True,
        "sensor_health_validated": True,
        "free_space_validated": True,
        "repeated_absence_observations": 2,
        "independent_evidence_sources": 2,
    }
    values.update(overrides)
    return NegativeInstallationEvidence(**values)


class InstallationStateContractTests(unittest.TestCase):
    def test_three_states_have_stable_chinese_labels(self) -> None:
        self.assertEqual(
            INSTALLATION_STATES,
            ("INSTALLED", "NOT_INSTALLED", "UNKNOWN"),
        )
        self.assertEqual(installation_state_label_zh("INSTALLED"), "安装")
        self.assertEqual(installation_state_label_zh("NOT_INSTALLED"), "未安装")
        self.assertEqual(installation_state_label_zh("UNKNOWN"), "不明")
        with self.assertRaises(ValueError):
            installation_state_label_zh("MISSING")

    def test_positive_negative_and_insufficient_evidence_resolve_safely(self) -> None:
        self.assertEqual(
            classify_installation_state(direct_instance_evidence=True),
            "INSTALLED",
        )
        negative = _qualified_negative_evidence()
        self.assertTrue(negative.is_qualified())
        self.assertEqual(
            classify_installation_state(negative_evidence=negative),
            "NOT_INSTALLED",
        )
        self.assertEqual(classify_installation_state(), "UNKNOWN")

    def test_every_negative_evidence_gate_is_mandatory(self) -> None:
        failures: dict[str, object] = {
            "calibration_validated": False,
            "registration_validated": False,
            "expected_region_in_frame": False,
            "expected_region_unoccluded": False,
            "sensor_health_validated": False,
            "free_space_validated": False,
            "repeated_absence_observations": 1,
            "independent_evidence_sources": 1,
        }
        for field, failing_value in failures.items():
            with self.subTest(field=field):
                evidence = _qualified_negative_evidence(**{field: failing_value})
                self.assertFalse(evidence.is_qualified())
                self.assertEqual(
                    classify_installation_state(negative_evidence=evidence),
                    "UNKNOWN",
                )

    def test_negative_gate_rejects_truthy_non_boolean_or_non_integer_values(self) -> None:
        for field, failing_value in (
            ("calibration_validated", "yes"),
            ("repeated_absence_observations", True),
            ("repeated_absence_observations", 2.0),
            ("independent_evidence_sources", True),
        ):
            with self.subTest(field=field, value=failing_value):
                self.assertFalse(
                    _qualified_negative_evidence(
                        **{field: failing_value}
                    ).is_qualified()
                )

    def test_resolver_rejects_truthy_values_that_are_not_typed_evidence(self) -> None:
        invalid_inputs = (
            {"direct_instance_evidence": "false"},
            {"evidence_healthy": "false"},
            {"conflict": 1},
            {"negative_evidence": True},
        )
        for values in invalid_inputs:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    classify_installation_state(**values)

    def test_unhealthy_conflicting_or_mixed_evidence_is_unknown(self) -> None:
        cases = (
            {
                "direct_instance_evidence": True,
                "evidence_healthy": False,
            },
            {
                "negative_evidence": _qualified_negative_evidence(),
                "evidence_healthy": False,
            },
            {
                "direct_instance_evidence": True,
                "conflict": True,
            },
            {
                "direct_instance_evidence": True,
                "negative_evidence": _qualified_negative_evidence(),
            },
        )
        for evidence in cases:
            with self.subTest(evidence=evidence):
                self.assertEqual(
                    classify_installation_state(**evidence),
                    "UNKNOWN",
                )


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
