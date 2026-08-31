"""Visibility-state debouncing and interval utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any


VisibilityState = tuple[str, ...]

INSTALLATION_STATES = ("INSTALLED", "NOT_INSTALLED", "UNKNOWN")
INSTALLATION_STATE_LABELS_ZH = {
    "INSTALLED": "安装",
    "NOT_INSTALLED": "未安装",
    "UNKNOWN": "不明",
}


@dataclass(frozen=True)
class NegativeInstallationEvidence:
    """Required gates for concluding that a designed pipe is not installed.

    This structure intentionally has no permissive defaults. A missing
    detection is not negative evidence; every spatial, sensor-health, and
    temporal gate must be supplied and pass.
    """

    calibration_validated: bool
    registration_validated: bool
    expected_region_in_frame: bool
    expected_region_unoccluded: bool
    sensor_health_validated: bool
    free_space_validated: bool
    repeated_absence_observations: int
    independent_evidence_sources: int

    def is_qualified(self) -> bool:
        """Return whether this bundle clears the conservative negative gate."""

        required_flags = (
            self.calibration_validated,
            self.registration_validated,
            self.expected_region_in_frame,
            self.expected_region_unoccluded,
            self.sensor_health_validated,
            self.free_space_validated,
        )
        return (
            all(value is True for value in required_flags)
            and type(self.repeated_absence_observations) is int
            and self.repeated_absence_observations >= 2
            and type(self.independent_evidence_sources) is int
            and self.independent_evidence_sources >= 2
        )


def classify_installation_state(
    *,
    direct_instance_evidence: bool = False,
    negative_evidence: NegativeInstallationEvidence | None = None,
    evidence_healthy: bool = True,
    conflict: bool = False,
) -> str:
    """Resolve one mutually exclusive installation state from qualified evidence.

    Negative evidence is deliberately stronger than absence of a detection.
    It must be supplied as a complete :class:`NegativeInstallationEvidence`
    bundle. Occlusion or no observation alone therefore falls through to
    ``UNKNOWN``.
    """

    boolean_inputs = {
        "direct_instance_evidence": direct_instance_evidence,
        "evidence_healthy": evidence_healthy,
        "conflict": conflict,
    }
    for name, value in boolean_inputs.items():
        if type(value) is not bool:
            raise ValueError(f"{name} must be a boolean")
    if negative_evidence is not None and not isinstance(
        negative_evidence, NegativeInstallationEvidence
    ):
        raise ValueError(
            "negative_evidence must be NegativeInstallationEvidence or None"
        )
    qualified_negative_evidence = (
        negative_evidence is not None and negative_evidence.is_qualified()
    )
    if not evidence_healthy or conflict:
        return "UNKNOWN"
    if direct_instance_evidence and qualified_negative_evidence:
        return "UNKNOWN"
    if direct_instance_evidence:
        return "INSTALLED"
    if qualified_negative_evidence:
        return "NOT_INSTALLED"
    return "UNKNOWN"


def installation_state_label_zh(state: str) -> str:
    """Return the required Chinese display label for an installation state."""

    try:
        return INSTALLATION_STATE_LABELS_ZH[state]
    except KeyError as error:
        raise ValueError(f"Unsupported installation state: {state}") from error


def normalize_state(state: Iterable[str]) -> VisibilityState:
    """Return a deterministic, duplicate-free pipe-id tuple."""

    return tuple(sorted(set(state)))


def debounce_states(
    raw_states: Sequence[Iterable[str]],
    min_stable_frames: int,
    fps: float = 1.0,
) -> list[VisibilityState | None]:
    """Causally confirm a state after ``min_stable_frames`` observations.

    The output has the same length as the input.  Before the first state is
    confirmed it contains ``None``; an empty tuple is therefore available to
    represent a confirmed frame with no visible pipes.  A transition is never backfilled;
    callers can separately expose the evidence start and confirmation time.
    ``fps`` is validated and retained in this frame-based MVP API so it can be
    upgraded to duration-based debounce without changing call sites.
    """

    if min_stable_frames < 1:
        raise ValueError("min_stable_frames must be at least 1")
    if fps <= 0:
        raise ValueError("fps must be positive")

    stable: VisibilityState | None = None
    candidate: VisibilityState | None = None
    candidate_count = 0
    output: list[VisibilityState | None] = []

    for raw_state in raw_states:
        current = normalize_state(raw_state)
        if stable is not None and current == stable:
            candidate = None
            candidate_count = 0
        else:
            if current == candidate:
                candidate_count += 1
            else:
                candidate = current
                candidate_count = 1
            if candidate_count >= min_stable_frames:
                stable = candidate
                candidate = None
                candidate_count = 0
        output.append(stable)

    return output


def confirmed_intervals(
    stable_states: Sequence[Iterable[str] | None],
    min_stable_frames: int,
    fps: float,
) -> list[dict[str, Any]]:
    """Compress causal states while preserving evidence and confirmation frames."""

    if min_stable_frames < 1:
        raise ValueError("min_stable_frames must be at least 1")
    if fps <= 0:
        raise ValueError("fps must be positive")

    intervals: list[dict[str, Any]] = []
    unset = object()
    previous: VisibilityState | object = unset
    for frame_index, value in enumerate(stable_states):
        if value is None:
            continue
        state = normalize_state(value)
        if state == previous:
            continue
        evidence_start = max(0, frame_index - min_stable_frames + 1)
        if intervals:
            intervals[-1]["end_frame"] = evidence_start - 1
            intervals[-1]["end_seconds"] = (evidence_start - 1) / fps
        intervals.append(
            {
                "start_frame": evidence_start,
                "confirmation_frame": frame_index,
                "start_seconds": evidence_start / fps,
                "confirmation_seconds": frame_index / fps,
                "visible_pipe_ids": list(state),
            }
        )
        previous = state

    if intervals:
        intervals[-1]["end_frame"] = len(stable_states) - 1
        intervals[-1]["end_seconds"] = (len(stable_states) - 1) / fps
    return intervals
