"""Replay a saved raw pair through the real Tk guide; never touch active calibration."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import json
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.calibration_wizard import (
    CalibrationCaptureArchive, ChessboardWizardDialog, detect_board_corners,
    draw_detection_overlay, read_calibration_image, zone_capture_guidance,
)

OUTPUT = ROOT / "outputs/calibration-guidance-20261030"
SOURCE = ROOT / "outputs/measurement_workbench/calibration_captures/20261030_110841_956633_67311fd3"


def snapshot(dialog, name, pair, observations):
    window = dialog.window
    window.update()
    shown = [dialog._photo(draw_detection_overlay(frame, observations[role], (8, 6)),
                           guidance=role == "left")
             for role, frame in (("left", pair.left), ("right", pair.right))]
    dialog.preview_images = shown
    dialog.left_preview.configure(image=shown[0])
    dialog.right_preview.configure(image=shown[1])
    window.update()
    x, y, w, h = window.winfo_rootx(), window.winfo_rooty(), window.winfo_width(), window.winfo_height()
    target = str(OUTPUT / f"{name}.png").replace("'", "''")
    script = f"""Add-Type -AssemblyName System.Drawing
$shot = [System.Drawing.Bitmap]::new({w}, {h})
$canvas = [System.Drawing.Graphics]::FromImage($shot)
try {{
    $canvas.CopyFromScreen({x}, {y}, 0, 0, $shot.Size)
    $shot.Save('{target}', [System.Drawing.Imaging.ImageFormat]::Png)
}} finally {{ $canvas.Dispose(); $shot.Dispose() }}
"""
    subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script], check=True)
    widgets = {"left_preview": dialog.left_preview, "right_preview": dialog.right_preview,
               **{f"cell_{zone}": cell for zone, cell in dialog.zone_cells.items()}}
    bounds = {name: [widget.winfo_rootx() - x, widget.winfo_rooty() - y,
                     widget.winfo_width(), widget.winfo_height()]
              for name, widget in widgets.items()}
    assert all(0 <= bx and 0 <= by and bx + bw <= w and by + bh <= h
               for bx, by, bw, bh in bounds.values()), bounds
    return {"size": [w, h], "bounds": bounds, "status": dialog.guidance_status.get(),
            "cells": zone_capture_guidance(dialog.pairs, dialog._guidance_observations,
                                           dialog._capture_stability),
            "screenshot": str(Path(target).relative_to(ROOT))}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protected = [ROOT / "outputs/measurement_workbench/calibration_current.json",
                 SOURCE / "session.json", SOURCE / "left/000001.png", SOURCE / "right/000001.png"]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    pair = SimpleNamespace(left=read_calibration_image(protected[2]),
                           right=read_calibration_image(protected[3]), sync_delta_ms=0.)
    obs = {role: detect_board_corners(frame, pattern=(8, 6), sb_accuracy=False)
           for role, frame in (("left", pair.left), ("right", pair.right))}
    assert all(value is not None for value in obs.values())
    root = tk.Tk()
    root.geometry("100x60+0+0")
    app = SimpleNamespace(root=root, tk=tk, ttk=ttk, messagebox=messagebox)
    dialog = ChessboardWizardDialog(app)
    dialog.window.geometry("1080x780+20+20")
    dialog.window.lift()
    dialog.window.attributes("-topmost", True)
    dialog.message.set("界面验证：使用已保存照片回放，不是新的现场采集或标定。")
    dialog.left_status.set("左目：回放照片已检出完整棋盘")
    dialog.right_status.set("右目：回放照片已检出完整棋盘")
    dialog._save_pair_checkpoint = lambda: None
    dialog.capture_archive = CalibrationCaptureArchive.create(
        session_metadata={"purpose": "UI replay only"}, root=OUTPUT / "replay-captures")
    result = {"source": str(SOURCE.relative_to(ROOT)), "mode": "saved-pair replay; simulated clock"}
    try:
        for t in (10., 13.):
            with patch("pipe_twin.calibration_wizard.time.monotonic", return_value=t):
                dialog._maybe_auto_capture(pair, obs)
        assert not dialog.pairs
        result["timing"] = snapshot(dialog, "timing", pair, obs)
        with patch("pipe_twin.calibration_wizard.time.monotonic", return_value=16.):
            dialog._maybe_auto_capture(pair, obs)
        assert len(dialog.pairs) == 1
        result["captured"] = snapshot(dialog, "captured", pair, obs)
        dialog.remove_last()
        assert not dialog.pairs
        result["removed"] = snapshot(dialog, "removed", pair, obs)
        dialog.stop()
        dialog.window.geometry("960x700+20+20")
        result["minimum_window"] = snapshot(dialog, "minimum-window", pair, obs)
        assert min(result["minimum_window"]["bounds"][role][3]
                   for role in ("left_preview", "right_preview")) >= 200
        dialog.window.geometry("1080x940+20+20")
        dialog._toggle_capture_settings()
        assert dialog._settings_visible
        result["expanded_settings"] = snapshot(dialog, "expanded-settings", pair, obs)
    finally:
        dialog.close()
        root.destroy()
    for name, expected in hashes.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    result["protected_sha256"] = hashes
    Path(__file__).with_name("gui-validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"result": "passed", "states": list(result),
                      "screenshots": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
