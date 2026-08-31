"""Visibility-state debouncing and interval utilities."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


VisibilityState = tuple[str, ...]


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
