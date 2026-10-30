"""Opt-in hardware smoke check: preview only, automatic capture disabled."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import hashlib
import json
import sys
import time
import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.calibration_wizard import ChessboardWizardDialog


def main():
    calibration = ROOT / "outputs/measurement_workbench/calibration_current.json"
    digest = hashlib.sha256(calibration.read_bytes()).hexdigest()
    root = tk.Tk()
    root.geometry("100x60+0+0")
    messages = Mock()
    dialog = ChessboardWizardDialog(SimpleNamespace(root=root, tk=tk, ttk=ttk, messagebox=messages))
    dialog.window.geometry("1080x780+20+20")
    dialog.continuous.set(False)
    dialog._capture_mode_changed()
    samples = []
    original = dialog._maybe_auto_capture

    def observed(pair, observations):
        original(pair, observations)
        samples.append({"at": round(time.monotonic(), 3),
                        "eyes": sorted(observations),
                        "zones": {role: value.centroid_zone for role, value in observations.items()},
                        "sharpness": {role: value.sharpness for role, value in observations.items()},
                        "stable_seconds": dialog._capture_stability.elapsed_seconds,
                        "ready": dialog._capture_stability.ready,
                        "status": dialog.guidance_status.get()})

    dialog._maybe_auto_capture = observed
    result = {"mode": "live preview only; no calibration capture"}
    try:
        dialog.start()
        deadline = time.monotonic() + 28
        while time.monotonic() < deadline:
            root.update()
            time.sleep(.05)
        result.update({"opened": dialog.session is not None,
                       "errors": [str(call) for call in messages.showerror.call_args_list],
                       "samples": samples, "pair_count": len(dialog.pairs)})
        assert dialog.pairs == [] and dialog.capture_archive is None
    finally:
        dialog.close()
        root.destroy()
    assert hashlib.sha256(calibration.read_bytes()).hexdigest() == digest
    result["calibration_sha256_unchanged"] = digest
    Path(__file__).with_name("live-validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(result, ensure_ascii=False), flush=True)
    assert result["opened"] and not result["errors"]
    assert any(item["eyes"] == ["left", "right"] for item in samples)


if __name__ == "__main__":
    main()
