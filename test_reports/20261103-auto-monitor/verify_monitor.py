"""Two archived-pair replays through the real monitor and local analyzer.

This script NEVER opens a camera. Both cycles replay the same already-rectified
2026-11-03 09:28 pair, with explicitly simulated replay timestamps. Only the
test adapter skips raw-camera rectification. It does not claim two new captures.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import queue
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from pipe_twin.auto_monitor import SingleFrameMonitor
from pipe_twin.elevation_dataset import load_elevation_dataset
from pipe_twin.stereo_analyzer import analyze_stereo_capture
from pipe_twin.stereo_camera import CapturedStereoPair, LAYOUT_SIDE_BY_SIDE_LR


SOURCE = ROOT.parent.parent / "outputs/elevation_workbench/captures/elevation-20261103-092822-e46925ff/manifest.json"
LOG_ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    loaded = load_elevation_dataset(SOURCE)
    original = loaded["manifest"]
    views = original["capture"]["capture_groups"][-1]["views"]
    source_paths = {"manifest": SOURCE, "model": SOURCE.parent / original["model"]["path"],
                    **{role: SOURCE.parent / views[role]["path"] for role in ("left", "right")}}
    before = {key: sha(path) for key, path in source_paths.items()}
    images = {role: cv2.imdecode(np.frombuffer(source_paths[role].read_bytes(), np.uint8), cv2.IMREAD_COLOR)
              for role in ("left", "right")}
    original_times = {role: views[role]["captured_at"] for role in images}
    assert len(original["capture"]["capture_groups"]) == 1
    assert not original["capture"]["capture_groups"][0].get("temporal_frames")
    run_stamp = datetime.now().strftime("%H%M%S")
    output_root = ROOT / "outputs" / ("amv" + run_stamp)
    assert output_root.resolve().is_relative_to(ROOT)
    log_path = LOG_ROOT / ("replay-" + run_stamp + ".jsonl")
    summary_path = LOG_ROOT / ("replay-" + run_stamp + ".json")
    sessions = []
    timings = []

    class ArchivedPairSession:
        def __init__(self, **_options):
            self.closed = False
            self.read_count = 0
            sessions.append(self)

        def open(self):
            pass

        def read_pair(self):
            time.sleep(.1)
            self.read_count += 1
            replay_time = datetime.now(timezone.utc).isoformat(timespec="microseconds")
            provenance = {role: {
                "capture_backend": "ARCHIVED_REPLAY_VALIDATION_NO_CAMERA",
                "capture_layout": "ARCHIVED_RECTIFIED_PAIR_SIMULATED_TIMESTAMPS",
                "capture_sync_method": "ARCHIVED_SAME_UVC_FRAME_REPLAY",
            } for role in images}
            return CapturedStereoPair(images["left"], images["right"], replay_time, replay_time,
                                      0.0, "MANIFEST_OPERATOR_CONFIRMED", provenance)

        def close(self):
            self.closed = True

    def real_analysis(path, *, report_output_path, evidence_dir, progress):
        started = time.monotonic()
        report = analyze_stereo_capture(path, report_output_path=report_output_path,
                                        evidence_dir=evidence_dir, progress=progress)
        timings.append({"manifest_path": str(path), "analysis_seconds": round(time.monotonic() - started, 3)})
        return report

    settings = copy.deepcopy(original["analysis"])
    settings.pop("mode", None)
    settings.pop("elevation_auto", None)
    arguments = dict(output_root=output_root, calibration=copy.deepcopy(loaded["calibration"]),
                     pipe_specs=copy.deepcopy(loaded["pipe_specs"]), model_path=source_paths["model"],
                     stl_unit=original["model"].get("source_unit", "millimeter"),
                     registration_settings=copy.deepcopy(loaded["registration_settings"]),
                     scope_settings=copy.deepcopy(loaded["scope_settings"]), analysis_settings=settings,
                     rectification_recipe=copy.deepcopy(loaded["rectification_recipe"]))
    camera_options = dict(layout=LAYOUT_SIDE_BY_SIDE_LR, left_index=0, right_index=None,
                          eye_width=images["left"].shape[1], eye_height=images["left"].shape[0])
    monitor = SingleFrameMonitor(arguments, camera_options, interval_seconds=1.0, warmup_seconds=0,
                                 session_factory=ArchivedPairSession, analyzer=real_analysis)
    results, errors = [], []
    started = time.monotonic()
    deadline = started + 900
    with log_path.open("w", encoding="utf-8") as log:
        def record(value):
            line = json.dumps(value, ensure_ascii=False, allow_nan=False)
            log.write(line + "\n")
            log.flush()
            print(line, flush=True)

        record({"event": "ARCHIVED_REPLAY_NOT_NEW_CAPTURE", "source": str(SOURCE),
                "source_hashes": before, "original_capture_times": original_times,
                "rectification_adapter": "SKIP_SECOND_REMAP_FOR_ARCHIVED_RECTIFIED_PIXELS",
                "camera_hardware_opened": False, "output_root": str(output_root)})
        # A replay adapter must not remap pixels that have already been rectified.
        # Production monitor and analyzer implementations remain unchanged.
        with patch("pipe_twin.auto_monitor.rectifier_for_calibration", return_value=None):
            monitor.start()
            try:
                while time.monotonic() < deadline:
                    try:
                        event = monitor.events.get(timeout=.5)
                    except queue.Empty:
                        if not monitor.active:
                            break
                        continue
                    short = {key: value for key, value in event.items() if key != "report"}
                    if event["kind"] == "result":
                        path = Path(event["manifest_path"])
                        manifest = json.loads(path.read_bytes())
                        groups = manifest["capture"]["capture_groups"]
                        assert len(groups) == 1 and not groups[0].get("temporal_frames")
                        photo_hashes = {role: sha(path.parent / groups[0]["views"][role]["path"]) for role in images}
                        assert photo_hashes == {role: before[role] for role in images}, "Replay pixels were altered"
                        assert event["report"]["inputs"]["manifest"] == sha(path)
                        assert all(groups[0]["views"][role]["capture_backend"] == "ARCHIVED_REPLAY_VALIDATION_NO_CAMERA"
                                   for role in images)
                        for role in images:
                            assert event["report"]["inputs"][groups[0]["capture_id"] + "_" + role] == before[role]
                        summary = {"cycle_id": event["cycle_id"], "manifest_path": str(path),
                                   "manifest_sha256": sha(path), "report_path": event["report_path"],
                                   "groups": len(groups), "burst_frames": 0, "photo_hashes": photo_hashes,
                                   "replay_timestamp": event["captured_at"], "counts": event["report"].get("counts"),
                                   "analysis_seconds": timings[-1]["analysis_seconds"]}
                        results.append(summary)
                        short["verified"] = summary
                        if len(results) == 2:
                            monitor.stop()
                    elif event["kind"] == "error":
                        errors.append(short)
                        monitor.stop()
                    record(short)
                    if event["kind"] == "stopped":
                        break
            finally:
                monitor.stop()
                until = time.monotonic() + 30
                while monitor.active and time.monotonic() < until:
                    time.sleep(.05)
        after = {key: sha(path) for key, path in source_paths.items()}
        summary = {"kind": "ARCHIVED_REPLAY_VALIDATION", "new_physical_captures": 0,
                   "passed": len(results) == 2 and not errors and before == after and not monitor.active,
                   "elapsed_seconds": round(time.monotonic() - started, 3), "source": str(SOURCE),
                   "source_hashes_before": before, "source_hashes_after": after,
                   "source_files_unchanged": before == after, "cycles": results, "errors": errors,
                   "monitor_active_after_stop": monitor.active,
                   "fake_sessions": [{"read_count": s.read_count, "closed": s.closed} for s in sessions],
                   "rectification_adapter": "ARCHIVED_PIXELS_ALREADY_RECTIFIED_NO_SECOND_REMAP",
                   "log_path": str(log_path)}
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        record(summary)
        assert summary["passed"] and all(s.closed for s in sessions), "Monitor replay validation failed"
    print("Validation summary: " + str(summary_path), flush=True)


if __name__ == "__main__":
    main()
