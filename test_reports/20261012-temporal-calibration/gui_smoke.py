"""Isolated Tk layout check with images, without camera or user GUI changes."""
from pathlib import Path
import json
import sys
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"tests")]
from pipe_twin.elevation_gui import ElevationCaptureDialog
from pipe_twin.stereo_camera import CapturedStereoPair
from test_elevation_dataset import _calibration

root = tk.Tk()
root.withdraw()
app = SimpleNamespace(root=root, tk=tk, ttk=ttk, filedialog=Mock(), messagebox=Mock())
records = []

def walk(widget):
    for child in widget.winfo_children():
        yield child
        yield from walk(child)

def inspect(window, labels):
    root.update()
    row = {"window": window.title(), "size": [window.winfo_width(), window.winfo_height()], "controls": {}}
    for label in labels:
        widget = next(w for w in walk(window) if isinstance(w, ttk.Button) and w.cget("text") == label)
        x, y = widget.winfo_rootx()-window.winfo_rootx(), widget.winfo_rooty()-window.winfo_rooty()
        w, h = widget.winfo_width(), widget.winfo_height()
        visible = bool(widget.winfo_ismapped() and x>=0 and y>=0 and x+w<=window.winfo_width() and y+h<=window.winfo_height())
        row["controls"][label] = {"rect": [x, y, w, h], "fully_visible": visible}
    records.append(row)

try:
    with patch("pipe_twin.workbench_profile.load_profile", return_value=(None, None)):
        owner = ElevationCaptureDialog(app, output_root=ROOT/"outputs/temporal-calibration-20261012/gui", restore=False)
    owner.calibration_override = _calibration()
    for size in ("940x660", "1120x760", "1120x860"):
        owner.window.geometry(size+"+20+20")
        inspect(owner.window, ["④ 保存并评估", "识别诊断", "重新标定 / 质量检查"])
    owner.capture_camera()
    camera = owner.camera_dialog
    frame = np.full((1080, 1920, 3), 110, np.uint8)
    pair = CapturedStereoPair(frame, frame, "2026-10-12T00:00:00+00:00", "2026-10-12T00:00:00+00:00", 0, "HOST_SYSTEM_CLOCK", {})
    for size in ("850x600", "1020x700"):
        camera.window.geometry(size+"+20+20")
        root.update()
        camera._show_pair(pair)
        inspect(camera.window, ["连续采集 20 对并使用", "同步抓拍并使用"])
    owner.open_calibration()
    wizard = owner.calibration_dialog
    wizard.window.transient(owner.window)
    for size in ("960x700", "1080x780"):
        wizard.window.geometry(size+"+20+20")
        root.update()
        wizard.preview_images = [wizard._photo(frame), wizard._photo(frame)]
        wizard.left_preview.configure(image=wizard.preview_images[0])
        wizard.right_preview.configure(image=wizard.preview_images[1])
        inspect(wizard.window, ["完成标定并保存", "生成 A4 棋盘格打印 PNG"])
    output = {"layouts": records, "camera_opened": False, "all_controls_visible": all(c["fully_visible"] for r in records for c in r["controls"].values())}
    (Path(__file__).parent/"gui.json").write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    assert output["all_controls_visible"]
finally:
    owner.close()
    root.destroy()
