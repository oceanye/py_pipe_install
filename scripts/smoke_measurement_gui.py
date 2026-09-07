"""Exercise the real Tk widgets without adding samples to an operator workbook.

Run after producing outputs/gui_validation/stereo_report.json.  --show makes
the window visible briefly for desktop validation; the default stays hidden.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import tkinter as tk
    from pipe_twin.gui import _PipeTwinApplication
    from pipe_twin.measurement_book import empty_book, load_book
    from pipe_twin.camera_pose import POSE_MODE_LABELS
    from pipe_twin.capture_gui import CameraPoseDialog, CaptureInputDialog, StereoCameraDialog

    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    report = ROOT / "outputs/gui_validation/stereo_report.json"
    manifest = ROOT / "test_model/field_stereo_demo_manifest.json"
    errors = []
    root = tk.Tk()
    root.withdraw()
    root.report_callback_exception = lambda *error: errors.append("".join(traceback.format_exception(*error)))
    app = _PipeTwinApplication(root, None, None)
    app.messagebox = Mock()
    app._load_sources(manifest, report)
    if args.show:
        root.deiconify()
    root.update()
    layout = {"window": [root.winfo_width(), root.winfo_height()],
              "photo": [app.measurement_panel.viewport.canvas.winfo_width(), app.measurement_panel.viewport.canvas.winfo_height()],
              "pipe_table": [app.measurement_panel.pipe_tree.winfo_width(), app.measurement_panel.pipe_tree.winfo_height()]}
    if args.show:
        assert root.winfo_ismapped()
        assert min(layout["photo"]) >= 180, layout
        assert layout["pipe_table"][1] >= 120, layout
    assert app.dashboard["binding_valid"], app.dashboard["binding_issues"]
    panel = app.measurement_panel
    measured = [r for r in panel.summary["pipes"] if r["measurement_status"] == "MEASURED"]
    assert len(measured) >= 2, panel.summary
    app._select_pipe(measured[0]["pipe_id"])
    root.update()
    destination = ROOT / "outputs/gui_validation"
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination) as temporary:
        panel.book_path = Path(temporary) / "smoke_samples.json"
        panel.explicit_book = True
        panel.book = empty_book()
        panel.reference.set("40.0")
        panel.use.set("校正样本")
        panel.notes.set("自动界面测试样本，不是真实人工实测")
        panel.add_sample()
        root.update()
        assert len(panel.book["samples"]) == 1
        assert load_book(panel.book_path)["samples"][0]["raw_mm"] == measured[0]["raw_diameter_mm"]
        panel.role.set("right")
        panel.update_photo()
        panel.viewport._wheel(type("Event", (), {"x": 100, "y": 100, "delta": 120})())
        panel.set_roi([500, 200, 200, 160])
        assert panel.active_rois()[measured[0]["pipe_id"]]["right"] == [500, 200, 200, 160]
        panel.clear_roi()
        panel.book = empty_book()
        panel.refresh()
        folder = panel.export_to(destination)
        exported = json.loads((folder / "逐管状态.json").read_text(encoding="utf-8"))
        assert len(exported["pipes"]) == 9
        assert len(exported["pairs"]) == 36
        panel.tabs.select(1)
        root.update()
        panel.tabs.select(3)
        root.update()
        wizard = CaptureInputDialog(app)
        root.update()
        assert len(wizard.tree.get_children()) == 9
        assert wizard.model_toolbar.winfo_reqwidth() < 950
        assert wizard.action_toolbar.winfo_reqwidth() < 950
        calibration = json.loads(manifest.read_text(encoding="utf-8"))["stereo_calibration"]
        camera = StereoCameraDialog(wizard, calibration)
        root.update()
        assert "并排双目流" in camera.mode.get()
        assert "3840×1080" in camera.message.get()
        camera.close()
        pose = CameraPoseDialog(wizard, calibration)
        pose.mode.set(POSE_MODE_LABELS["positive_z"])
        for variable, value in zip(pose.center, (250.0, 150.0, 2000.0)):
            variable.set(str(value))
        pose.yaw.set("3")
        pose.pitch.set("-2")
        pose.roll.set("1.5")
        pose.validated.set(True)
        pose.save()
        root.update()
        assert wizard.pose_adjustment["mode"] == "positive_z"
        assert wizard.pose_adjustment["roll_deg"] == 1.5
        assert wizard.pose_adjustment["registration_validated"] is True
        wizard.window.destroy()
        app.main_tabs.select(1)
        root.update()
        app.main_tabs.select(0)
        root.update()
    assert not errors, errors
    assert not app.messagebox.showerror.called, app.messagebox.showerror.call_args_list
    root.destroy()
    print(json.dumps({"status": "PASS", "pipe_count": 9, "pair_count": 36,
                      "measured_count": len(measured), "export_directory": str(folder), "layout": layout,
                      "checks": ["empty_start", "bound_report", "photo_zoom", "roi", "reference_save_reload", "pair_table", "detail", "capture_dialog", "stereo_camera_dialog", "camera_pose_dialog", "export"]}, ensure_ascii=True))


if __name__ == "__main__":
    main()
