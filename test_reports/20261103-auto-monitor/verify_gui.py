"""Capture isolated Tk fixtures; no camera, network, AI, or user preferences."""
from pathlib import Path
import json
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from test_auto_monitor_gui import AutoMonitorGuiTests

screenshot = runpy.run_path(str(ROOT / "test_reports/20261030-ground-monitor/gui-verify.py"))["screenshot"]


def main():
    output = Path(__file__).resolve().parent
    AutoMonitorGuiTests.setUpClass()
    case = AutoMonitorGuiTests("test_opening_is_idle_and_does_not_acquire_camera_or_configuration")
    try:
        case.setUp()
        dialog = case.dialog
        dialog.window.deiconify()
        dialog.start()
        case.complete_cycle()
        snapshots = []
        for size in ("1024x720", "1280x860"):
            dialog.window.geometry(size)
            case.root.update()
            name = f"result-{size}.png"
            screenshot(dialog.window, output / name)
            snapshots.append(name)
        dialog.window.geometry("1024x720")
        dialog.handle_event({"kind": "state", "cycle_id": 2, "state": "CAPTURING"})
        dialog.handle_event(case.capture(2))
        dialog.handle_event({"kind": "state", "cycle_id": 2, "state": "ANALYZING"})
        dialog.progress.set("正在分析本轮单帧照片；完成后显示新的断面结果。")
        case.now = 170
        dialog._countdown()
        case.root.update()
        screenshot(dialog.window, output / "analysis-cleared-1024x720.png")
        case.assert_results_empty()
        dialog.handle_event({"kind": "error", "cycle_id": 2, "message": "模拟：计算连接断开，本轮未完成。"})
        case.root.update()
        screenshot(dialog.window, output / "error-inline-1024x720.png")
        evidence = {"backend": "FakeMonitor", "real_camera_opened": False, "network_requests": 0,
                    "ai_requests": 0, "user_preferences_written": False,
                    "result_snapshots": snapshots, "analysis_has_report": dialog.report is not None,
                    "analysis_has_cad_pipe": bool(dialog.cad_canvas.find_withtag("pipe")),
                    "analysis_has_measured_section": bool(dialog.section_canvas.find_withtag("scene-pipe")),
                    "analysis_table_rows": len(dialog.table.get_children()),
                    "countdown": dialog.countdown.get(), "error": dialog.error.get()}
        (output / "gui-evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    finally:
        if hasattr(case, "dialog"):
            case.tearDown()
            case.doCleanups()
        AutoMonitorGuiTests.tearDownClass()


if __name__ == "__main__":
    main()
