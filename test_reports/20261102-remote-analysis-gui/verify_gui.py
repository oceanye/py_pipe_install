"""Isolated screenshot smoke: synthetic photos, no camera/AI/production service."""
from pathlib import Path
import json
import runpy
import sys
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from pipe_twin.analysis_server import AnalysisServer
from test_analysis_runtime_gui import RuntimeGuiTests


screenshot = runpy.run_path(str(ROOT / "test_reports/20261030-ground-monitor/gui-verify.py"))["screenshot"]


def main():
    destination = Path(__file__).resolve().parent
    RuntimeGuiTests.setUpClass()
    case = RuntimeGuiTests("test_busy_mode_change_does_not_save_or_change_execution")
    server = None
    address_patch = mock.patch("pipe_twin.analysis_capabilities.listening_addresses", return_value=["127.0.0.1"])
    address_patch.start()
    try:
        case.setUp()
        dialog, runtime = case.dialog, case.dialog.runtime
        dialog.window.deiconify()
        dialog.window.geometry("1100x830+30+30")
        dialog.open_settings("runtime")
        dialog.settings_window.geometry("860x510+140+120")
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline and not runtime.addresses:
            case.root.update()
            time.sleep(0.02)
        case.root.update()
        dialog.settings_window.lift()
        dialog.settings_window.focus_force()
        screenshot(dialog.settings_window, destination / "runtime-settings.png")
        for mode in ("client", "server"):
            case.mode(mode)
            case.root.update()
            screenshot(dialog.settings_window, destination / f"runtime-settings-{mode}.png")

        # Allocate a loopback-only temporary service; its archive root is the
        # fixture's temporary directory and it receives no computation jobs.
        server = AnalysisServer("127.0.0.1", 0, case.output / "service").start()
        runtime.port.set(str(server.port))
        case.mode("server")
        runtime.server = server
        runtime.handle(runtime.revision, "runtime_server_started", server.health())
        dialog.hide_settings()
        dialog.window.lift()
        dialog.window.focus_force()
        case.root.update()
        screenshot(dialog.window, destination / "server-status.png")
        evidence = {"modes": list(runtime.mode_box.cget("values")),
                    "status": runtime.status.get(), "service_url": runtime.endpoint.get(),
                    "local_capabilities": runtime.capabilities.get(),
                    "server_capabilities": runtime.server_capabilities.get(),
                    "camera_opened": dialog.camera_dialog is not None,
                    "address_candidates_isolated": True,
                    "address_candidates_note": "截图 IP 候选固定为 127.0.0.1，未展示当前主机 LAN/VPN 地址。",
                    "server_capture_disabled": "disabled" in dialog.run_button.state(),
                    "settings_path_isolated": str(Path(case.tmp.name) / "runtime.json"),
                    "services": "temporary loopback-only service, no jobs", "ai_requests": 0}
        (destination / "gui-evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(evidence, ensure_ascii=False))
    finally:
        if server is not None:
            server.stop()
        if hasattr(case, "dialog"):
            case.tearDown()
            case.doCleanups()
        RuntimeGuiTests.tearDownClass()
        address_patch.stop()


if __name__ == "__main__":
    main()
