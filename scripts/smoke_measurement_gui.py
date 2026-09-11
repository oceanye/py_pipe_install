"""Exercise the real Tk widgets without adding samples to an operator workbook.

Run after producing outputs/gui_validation/stereo_report.json.  --show makes
the window visible briefly for desktop validation; the default stays hidden.
The workbench profile paths are monkeypatched into a temporary directory so a
smoke run never touches the operator's saved configuration.
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

    import cv2
    import numpy as np

    from pipe_twin import workbench_profile
    from pipe_twin.calibration_wizard import ChessboardWizardDialog
    from pipe_twin.capture_gui import (
        CameraPoseDialog,
        CaptureInputDialog,
        ColorPickDialog,
        ModelAnchorPickerDialog,
        QrRegistrationDialog,
        StereoCameraDialog,
    )
    from pipe_twin.camera_pose import POSE_MODE_LABELS
    from pipe_twin.gui import _PipeTwinApplication
    from pipe_twin.measurement_book import empty_book, load_book

    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    report = ROOT / "outputs/gui_validation/stereo_report.json"
    manifest = ROOT / "test_model/field_stereo_demo_manifest.json"
    errors = []
    destination = ROOT / "outputs/gui_validation"
    destination.mkdir(parents=True, exist_ok=True)
    sandbox = tempfile.TemporaryDirectory(dir=destination)
    temporary = Path(sandbox.name)
    # Keep the operator's real workbench profile out of this run.
    original_paths = (
        workbench_profile.default_profile_path,
        workbench_profile.default_calibration_path,
    )
    workbench_profile.default_profile_path = lambda: temporary / "workbench_profile.json"
    workbench_profile.default_calibration_path = lambda: temporary / "calibration_current.json"
    root = tk.Tk()
    try:
        root.withdraw()
        root.report_callback_exception = lambda *error: errors.append(
            "".join(traceback.format_exception(*error))
        )
        app = _PipeTwinApplication(root, None, None)
        app.messagebox = Mock()
        app._load_sources(manifest, report)
        if args.show:
            root.deiconify()
        root.update()
        layout = {
            "window": [root.winfo_width(), root.winfo_height()],
            "photo": [
                app.measurement_panel.viewport.canvas.winfo_width(),
                app.measurement_panel.viewport.canvas.winfo_height(),
            ],
            "pipe_table": [
                app.measurement_panel.pipe_tree.winfo_width(),
                app.measurement_panel.pipe_tree.winfo_height(),
            ],
        }
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
        with tempfile.TemporaryDirectory(dir=destination) as book_temp:
            panel.book_path = Path(book_temp) / "smoke_samples.json"
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
        assert not wizard.manual_capture_visible
        wizard.toggle_manual_capture()
        root.update()
        assert wizard.manual_capture_visible
        wizard.toggle_manual_capture()
        assert not wizard.manual_capture_visible
        assert "相机标定" in wizard.calibration_status.get()
        calibration = json.loads(manifest.read_text(encoding="utf-8"))["stereo_calibration"]
        camera = StereoCameraDialog(wizard, calibration)
        root.update()
        assert "并排双目流" in camera.mode.get()
        assert "3840×1080" in camera.message.get()
        camera.close()
        qr = QrRegistrationDialog(wizard)
        root.update()
        assert qr.marker_id.get() == "PIPE-TWIN-QR-001"
        assert qr.marker_edge.get() == "120.0"
        assert qr.measured_edge.get() == "120.0"
        assert not wizard.fields["left"].get()
        model_picker = ModelAnchorPickerDialog(qr)
        root.update()
        assert model_picker.projection is not None, [
            model_picker.canvas.itemcget(item, "text")
            for item in model_picker.canvas.find_all()
            if model_picker.canvas.type(item) == "text"
        ]
        triangle = model_picker.projection["triangles_canvas"][0]
        click = np.mean(triangle, axis=0)
        model_picker._pick(
            type("Event", (), {"x": float(click[0]), "y": float(click[1])})()
        )
        assert model_picker.selected_point is not None
        model_picker._apply()
        root.update()
        assert qr.cad_confirmed.get()
        assert qr.custom_right_world is not None and qr.custom_up_world is not None
        assert np.allclose(np.dot(qr.custom_right_world, qr.custom_up_world), 0.0)
        assert "RIGHT" in qr.orientation_text.get() and "UP" in qr.orientation_text.get()
        qr.window.destroy()
        pose = CameraPoseDialog(wizard, calibration)
        assert not pose.advanced_visible
        pose.toggle_advanced()
        assert pose.advanced_visible
        pose.distance.set("1000")
        pose.select_side_view("top_down")
        assert pose.mode.get() == POSE_MODE_LABELS["positive_y"]
        assert pose.roll.get() == "0"
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
        board = ChessboardWizardDialog(app, owner=wizard)
        root.update()
        assert board.columns.get() == "9" and board.rows.get() == "7"
        assert "3840×1080" in board.stream_mode.get()
        assert (board.eye_width.get(), board.eye_height.get()) == ("1920", "1080")
        assert "180×140" in board.fit_hint.get() and "可打印" in board.fit_hint.get()
        board.columns.set("12")
        root.update()
        assert "超出 A4" in board.fit_hint.get()
        board.columns.set("9")
        assert board.session is None
        board.close()
        wizard.window.destroy()
        # Color picking: click a pure-blue patch in a synthetic left photo.
        wizard = CaptureInputDialog(app)
        root.update()
        photo_path = temporary / "pick_source.png"
        photo = np.full((300, 400, 3), 255, dtype=np.uint8)
        photo[150:210, 200:260] = (255, 0, 0)
        cv2.imwrite(str(photo_path), photo)
        wizard.fields["left"].set(str(photo_path))
        picker = ColorPickDialog(wizard, 0)
        root.update()
        picker._pick(type("Event", (), {"x": 230, "y": 180})())
        assert picker.sampled == "#0000FF", picker.sampled
        picker._apply()
        root.update()
        assert wizard.pipes[0]["color_srgb"] == "#0000FF"
        assert workbench_profile.default_profile_path().is_file()
        saved = json.loads(workbench_profile.default_profile_path().read_text(encoding="utf-8"))
        assert saved["pipes"][0]["color_srgb"] == "#0000FF"
        # A new dialog must restore the persisted catalog instead of the demo.
        wizard.window.destroy()
        wizard = CaptureInputDialog(app)
        root.update()
        assert len(wizard.tree.get_children()) == 9
        assert wizard.pipes[0]["color_srgb"] == "#0000FF"
        assert wizard._profile_used
        wizard.reset_profile()
        root.update()
        reset = json.loads(workbench_profile.default_profile_path().read_text(encoding="utf-8"))
        assert reset["pipes"] == []
        wizard.window.destroy()
        # Startup restore: point the profile at the demo manifest and restore.
        # The profile stores the manifest path only; a fresh session has no
        # recognition report until the operator runs the analysis again.
        workbench_profile.update_profile({"last_manifest_path": str(manifest)})
        app._restore_workbench_session()
        root.update()
        assert str(app.manifest_path) == str(manifest), app.manifest_path
        assert "已恢复上次工作台会话" in app.banner_text.get()
        app.main_tabs.select(1)
        root.update()
        app.main_tabs.select(0)
        root.update()
    finally:
        (
            workbench_profile.default_profile_path,
            workbench_profile.default_calibration_path,
        ) = original_paths
        root.destroy()
        sandbox.cleanup()
    assert not errors, errors
    assert not app.messagebox.showerror.called, app.messagebox.showerror.call_args_list
    print(
        json.dumps(
            {
                "status": "PASS",
                "pipe_count": 9,
                "pair_count": 36,
                "export_directory": str(folder),
                "layout": layout,
                "checks": [
                    "empty_start",
                    "bound_report",
                    "photo_zoom",
                    "roi",
                    "reference_save_reload",
                    "pair_table",
                    "detail",
                    "capture_dialog",
                    "stereo_camera_dialog",
                    "qr_registration_dialog",
                    "qr_model_anchor_picker",
                    "side_view_pose",
                    "camera_pose_dialog",
                    "export",
                    "wizard_dialog",
                    "color_pick",
                    "profile_prefill",
                    "profile_reset",
                    "startup_restore",
                ],
            },
            ensure_ascii=True,
        )
    )


if __name__ == "__main__":
    main()
