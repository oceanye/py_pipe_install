from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from pipe_twin.remote_capture import capture_stereo_pairs
from pipe_twin.stereo_camera import CapturedStereoPair


class _FakeSession:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.index = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read_pair(self):
        self.index += 1
        left = np.full((4, 6, 3), self.index, dtype=np.uint8)
        right = np.full((4, 6, 3), self.index + 10, dtype=np.uint8)
        return CapturedStereoPair(
            left=left,
            right=right,
            left_captured_at=f"2026-09-13T10:00:{self.index % 60:02d}.000+08:00",
            right_captured_at=f"2026-09-13T10:00:{self.index % 60:02d}.001+08:00",
            sync_delta_ms=1.0,
            timestamp_source="HOST_SYSTEM_CLOCK",
            provenance={"left": {"capture_layout": "SINGLE_FRAME_SIDE_BY_SIDE"}, "right": {}},
        )


def test_capture_writes_pair_pngs_and_hashed_metadata(tmp_path: Path):
    manifest_path = capture_stereo_pairs(
        tmp_path / "run",
        eye_width=6,
        eye_height=4,
        count=2,
        detect_chessboard=True,
        session_factory=_FakeSession,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["pair_count"] == 2
    assert payload["status"] == "COMPLETED"
    assert payload["stop_reason"] == "COUNT_LIMIT"
    assert payload["requested_pair_count"] == 2
    assert payload["rectified"] is False
    assert payload["calibration_validated"] is False
    assert payload["camera_opened_at"] is not None
    assert payload["first_frame_at"] == payload["captures"][0]["left"]["captured_at"]
    assert payload["requested_frame_size_px"] == [12, 4]
    assert payload["chessboard_detection"]["enabled"] is True
    first = payload["captures"][0]
    for role in ("left", "right"):
        path = manifest_path.parent / first[role]["path"]
        data = path.read_bytes()
        assert first[role]["width"] == 6
        assert first[role]["height"] == 4
        assert hashlib.sha256(data).hexdigest() == first[role]["sha256"]
    assert first["chessboard"]["left"]["found"] is False


def test_capture_can_run_for_duration_with_injected_clock(tmp_path: Path):
    ticks = iter([0.0, 0.0, 0.5, 1.1])
    manifest_path = capture_stereo_pairs(
        tmp_path / "run",
        eye_width=6,
        eye_height=4,
        duration_s=1.0,
        session_factory=_FakeSession,
        clock=lambda: next(ticks),
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["pair_count"] == 2


def test_capture_rejects_non_empty_output(tmp_path: Path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "old.txt").write_text("x")
    try:
        capture_stereo_pairs(output, session_factory=_FakeSession)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected non-empty output directory to be rejected")


@pytest.mark.parametrize("failure", [RuntimeError("camera disconnected"), KeyboardInterrupt()])
def test_capture_failure_preserves_completed_pair_and_report(tmp_path: Path, failure):
    output = tmp_path / "run"

    class FailingSession(_FakeSession):
        def read_pair(self):
            if self.index:
                running = json.loads((output / "capture.json").read_text(encoding="utf-8"))
                assert running["status"] == "RUNNING"
                assert running["pair_count"] == 1
                raise failure
            return super().read_pair()

    with pytest.raises(type(failure)):
        capture_stereo_pairs(output, count=2, session_factory=FailingSession)
    payload = json.loads((output / "capture.json").read_text(encoding="utf-8"))
    assert payload["status"] == ("INTERRUPTED" if isinstance(failure, KeyboardInterrupt) else "FAILED")
    assert payload["pair_count"] == 1
    assert payload["error"]["type"] == type(failure).__name__
    assert payload["error"]["stage"] == "read_pair"
    assert (output / payload["captures"][0]["left"]["path"]).is_file()
    assert (output / payload["captures"][0]["right"]["path"]).is_file()


def test_camera_open_failure_leaves_zero_pair_failed_report(tmp_path: Path):
    output = tmp_path / "run"

    class FailingOpen(_FakeSession):
        def __enter__(self):
            running = json.loads((output / "capture.json").read_text(encoding="utf-8"))
            assert running["status"] == "RUNNING"
            raise RuntimeError("cannot open camera")

    with pytest.raises(RuntimeError, match="cannot open"):
        capture_stereo_pairs(output, session_factory=FailingOpen)
    payload = json.loads((output / "capture.json").read_text(encoding="utf-8"))
    assert payload["status"] == "FAILED"
    assert payload["pair_count"] == 0
    assert payload["error"]["stage"] == "open_camera"


def test_duration_exhausted_before_first_frame_is_failure(tmp_path: Path):
    ticks = iter([0.0, 2.0])
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="no stereo pairs"):
        capture_stereo_pairs(output, duration_s=1.0, clock=lambda: next(ticks), session_factory=_FakeSession)
    payload = json.loads((output / "capture.json").read_text(encoding="utf-8"))
    assert payload["status"] == "FAILED"
    assert payload["pair_count"] == 0


def test_interval_continuously_drains_old_camera_frames(tmp_path: Path):
    now = [0.0]
    sessions = []

    def factory(**kwargs):
        session = _FakeSession(**kwargs)
        sessions.append(session)
        return session

    manifest = capture_stereo_pairs(
        tmp_path / "run", count=2, interval_s=0.025, session_factory=factory,
        clock=lambda: now[0], sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["pair_count"] == 2
    assert payload["discarded_pair_count"] > 0
    assert sessions[0].index == payload["discarded_pair_count"] + 2


def test_chessboard_detection_uses_wizard_and_keeps_detector_errors(monkeypatch):
    from types import SimpleNamespace
    from pipe_twin import remote_capture

    def detector(image, *, pattern, sb_accuracy, on_cv_error):
        assert pattern == (11, 7)
        assert sb_accuracy is False
        on_cv_error("sb", ValueError("SB failure, classic used"))
        return SimpleNamespace(detector="classic", corners_px=np.zeros((77, 2)))

    monkeypatch.setattr(remote_capture, "detect_board_corners", detector)
    result = remote_capture._chessboard_status(np.zeros((40, 60, 3), dtype=np.uint8), columns=11, rows=7)
    assert result["found"] is True
    assert result["detector"] == "classic"
    assert result["corner_count"] == 77
    assert result["errors"][0]["stage"] == "sb"


def test_duration_starts_after_camera_open(tmp_path: Path):
    now = [0.0]

    class SlowOpen(_FakeSession):
        def __enter__(self):
            now[0] = 100.0
            return self

        def read_pair(self):
            now[0] += 0.6
            return super().read_pair()

    manifest = capture_stereo_pairs(
        tmp_path / "run", duration_s=1.0, session_factory=SlowOpen, clock=lambda: now[0],
    )
    assert json.loads(manifest.read_text(encoding="utf-8"))["pair_count"] == 2


def test_cli_capture_failures_reach_existing_error_log(monkeypatch, tmp_path):
    from pipe_twin import cli, remote_capture

    logged = []

    def fail(*args, **kwargs):
        raise RuntimeError("camera failed")

    monkeypatch.setattr(remote_capture, "capture_stereo_pairs", fail)
    monkeypatch.setattr(cli._LOGGER, "exception", lambda *args, **kwargs: logged.append(args[0]))
    with pytest.raises(RuntimeError, match="camera failed"):
        cli.main(["capture-stereo", "--output-dir", str(tmp_path / "run")])
    assert logged == ["command_failed"]
