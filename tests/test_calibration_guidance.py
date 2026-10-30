from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from pipe_twin.calibration_wizard import (
    BoardObservation, BoardStabilityGate, ChessboardWizardDialog,
    draw_zone_guidance, zone_capture_guidance,
)


def observations(zone=(0, 0)):
    xy = np.mgrid[:8, :6].reshape(2, -1).T.astype(float) * 25 + [200, 150]
    return {role: BoardObservation(xy + [offset, 0], 1000., zone, "sb")
            for role, offset in (("left", 0), ("right", -50))}


def settle(gate, obs, start=10.):
    for t in (start, start + 3, start + 6):
        gate.observe(obs, t)


def test_green_requires_a_retained_pair_not_detection_or_expired_timer():
    gate, obs = BoardStabilityGate(), observations()
    settle(gate, obs)
    cells = zone_capture_guidance([], obs, gate)
    assert cells[0]["state"] == "ready"
    assert not any(cell["count"] for cell in cells)
    assert [cell["name"] for cell in cells] == [
        "左上", "上中", "右上", "左中", "中心", "右中", "左下", "下中", "右下"]
    saved = [(obs["left"], obs["right"])]
    assert zone_capture_guidance(saved, obs, gate)[0]["state"] == "captured"
    # Deleting the last pair for a zone immediately removes its green state.
    assert zone_capture_guidance([], obs, gate)[0]["state"] != "captured"
    # Restored pairs supply coverage even with the camera stopped.
    gate.reset()
    assert zone_capture_guidance(saved, {}, gate)[0]["count"] == 1


def test_changing_zone_restarts_dwell_even_with_subpixel_motion():
    gate, obs = BoardStabilityGate(), observations()
    settle(gate, obs)
    changed = {role: replace(value, centroid_zone=(1, 0)) for role, value in obs.items()}
    assert not gate.observe(changed, 17.)
    assert gate.elapsed_seconds == 0
    cells = zone_capture_guidance([], changed, gate)
    assert cells[0]["state"] == "pending"
    assert cells[1]["state"] == "timing"


def test_time_alone_never_advances_confirmed_progress():
    gate, obs = BoardStabilityGate(), observations()
    gate.observe(obs, 10.)
    gate.observe(obs, 13.)
    with patch("pipe_twin.calibration_wizard.time.monotonic", return_value=100.):
        cells = zone_capture_guidance([], obs, gate)
    assert "3.0 / 6 秒" in cells[0]["text"]
    assert cells[0]["state"] == "timing"
    assert not gate.observe(obs, 100.)  # A stalled detector requires a new dwell.


def dialog_and_pair():
    dialog = object.__new__(ChessboardWizardDialog)
    dialog._capture_stability = BoardStabilityGate()
    dialog.continuous = Mock()
    dialog.continuous.get.return_value = True
    dialog.sharpness_gate = SimpleNamespace(get=lambda: "60")
    dialog.max_sync = SimpleNamespace(get=lambda: "1")
    dialog.message = Mock()
    dialog.pairs = []
    dialog._last_auto_time = 0.
    dialog._last_left_observation = None
    dialog._pattern = lambda: (8, 6)
    dialog._accept_pair = Mock()
    pair = SimpleNamespace(left=np.zeros((480, 640, 3), np.uint8),
                           right=np.ones((480, 640, 3), np.uint8), sync_delta_ms=0.)
    return dialog, pair


@pytest.mark.parametrize("problem", ["missing_eye", "blur", "detector", "identical", "sync"])
def test_invalid_pair_clears_dwell_without_turning_green(problem):
    dialog, pair = dialog_and_pair()
    obs = observations()
    settle(dialog._capture_stability, obs)
    if problem == "missing_eye":
        obs.pop("right")
    elif problem == "blur":
        obs["right"] = replace(obs["right"], sharpness=5.)
    elif problem == "detector":
        obs["right"] = replace(obs["right"], detector="classic")
    elif problem == "identical":
        pair.right = pair.left.copy()
    else:
        pair.sync_delta_ms = 2.
    with patch("pipe_twin.calibration_wizard.time.monotonic", return_value=17.):
        dialog._maybe_auto_capture(pair, obs)
    dialog._accept_pair.assert_not_called()
    assert dialog._capture_stability.elapsed_seconds == 0
    assert not any(cell["count"] for cell in zone_capture_guidance(
        dialog.pairs, obs, dialog._capture_stability))


def test_archive_failure_does_not_mark_a_zone_captured():
    dialog, pair = dialog_and_pair()
    obs = observations()
    settle(dialog._capture_stability, obs)
    # Exercise the production accept path, including its raw-image archive.
    del dialog._accept_pair
    dialog._ensure_capture_archive = Mock()
    dialog._ensure_capture_archive.return_value.add_pair.side_effect = OSError("disk full")
    with patch("pipe_twin.calibration_wizard.time.monotonic", return_value=17.):
        dialog._maybe_auto_capture(pair, obs)
    assert dialog.pairs == []
    dialog.continuous.set.assert_called_with(False)
    assert "disk full" in dialog._guidance_note
    assert zone_capture_guidance([], obs, dialog._capture_stability)[0]["state"] != "captured"


def test_manual_mode_counts_dwell_but_waits_for_capture_button():
    dialog, pair = dialog_and_pair()
    dialog.continuous.get.return_value = False
    obs = observations()
    for t in (10., 13., 16.):
        with patch("pipe_twin.calibration_wizard.time.monotonic", return_value=t):
            dialog._maybe_auto_capture(pair, obs)
    dialog._accept_pair.assert_not_called()
    assert dialog._capture_stability.ready
    assert "抓拍一组" in dialog._guidance_note
    assert zone_capture_guidance([], obs, dialog._capture_stability)[0]["state"] == "ready"


def test_preview_grid_does_not_change_detector_or_archived_pixels():
    raw = np.full((270, 480, 3), 90, np.uint8)
    original = raw.copy()
    shown = draw_zone_guidance(raw, zone_capture_guidance([], {}, BoardStabilityGate()))
    assert np.array_equal(raw, original)
    assert not np.array_equal(shown, raw)
