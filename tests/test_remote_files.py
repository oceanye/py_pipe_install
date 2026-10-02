from __future__ import annotations

import http.client
from contextlib import closing
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.request import ProxyHandler, build_opener

from pipe_twin import remote_files


class RemoteFilesTests(unittest.TestCase):
    def config(self, root: Path, **changes: object) -> Path:
        data = {"enabled": True, "bind_host": "100.103.31.118", "port": 8765,
                "directory": str(root), "control_ip": "100.92.137.53", **changes}
        path = root / "config.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_missing_or_disabled_configuration_does_not_open_a_port(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(remote_files.subprocess, "Popen") as spawn:
            root = Path(temporary)
            self.assertEqual(remote_files.ensure_remote_files(root / "missing.json"), {"status": "DISABLED"})
            self.assertEqual(remote_files.ensure_remote_files(self.config(root, enabled=False)), {"status": "DISABLED"})
            spawn.assert_not_called()

    def test_public_and_wildcard_addresses_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(remote_files.subprocess, "Popen") as spawn:
            root = Path(temporary)
            for host in ("0.0.0.0", "192.168.1.2", "8.8.8.8"):
                with self.subTest(host=host):
                    result = remote_files.ensure_remote_files(self.config(root, bind_host=host))
                    self.assertEqual(result["status"], "FAILED")
            result = remote_files.ensure_remote_files(self.config(root, control_ip="8.8.8.8"))
            self.assertEqual(result["status"], "FAILED")
            spawn.assert_not_called()

    def test_auto_bind_uses_current_adapter_ip_without_admin_cli(self):
        with patch.object(remote_files.socket, "getaddrinfo", return_value=[
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.26.0.176", 0)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("100.103.31.118", 0)),
        ]), patch.object(remote_files.subprocess, "run") as run:
            self.assertEqual(remote_files.discover_tailscale_ip(), "100.103.31.118")
            run.assert_not_called()

    def test_matching_existing_service_is_reused_without_spawning(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.config(Path(temporary))
            with patch.object(remote_files, "_probe", return_value=(True, 123)), patch.object(remote_files.subprocess, "Popen") as spawn:
                result = remote_files.ensure_remote_files(path)
            self.assertEqual(result["status"], "REUSED")
            self.assertEqual(result["pid"], 123)
            self.assertEqual(result["url"], "http://100.103.31.118:8765/")
            spawn.assert_not_called()

    def test_unrelated_listener_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = self.config(Path(temporary))
            with patch.object(remote_files, "_probe", return_value=(False, None)), patch.object(remote_files, "_port_open", return_value=True), patch.object(remote_files.subprocess, "Popen") as spawn:
                result = remote_files.ensure_remote_files(path)
            self.assertEqual(result["status"], "FAILED")
            self.assertIn("occupied", result["error"])
            spawn.assert_not_called()

    def test_existing_html_response_does_not_crash_gui_startup(self):
        response = Mock()
        response.read.return_value = b"<html>different directory</html>"
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with patch.object(remote_files, "build_opener") as opener:
            opener.return_value.open.return_value = response
            self.assertEqual(remote_files._probe("http://127.0.0.1:8765/", "expected"), (False, None))


class RemoteFilesProcessTests(unittest.TestCase):
    config = RemoteFilesTests.config

    def test_detached_service_survives_launcher_exit_and_restricts_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            content = bytes(range(256)) * 64
            (root / "pair_0001_left.png").write_bytes(content)
            path = self.config(root, bind_host="127.0.0.1", port=port, control_ip="127.0.0.3")
            launcher = [sys.executable, "-c",
                        "import json,sys; from pipe_twin.remote_files import ensure_remote_files; "
                        "print(json.dumps(ensure_remote_files(sys.argv[1])))", str(path)]
            pid = None
            try:
                completed = subprocess.run(launcher, capture_output=True, text=True, timeout=15, cwd=remote_files._REPO)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                first = json.loads(completed.stdout)
                self.assertEqual(first["status"], "STARTED", first)
                pid = first["pid"]
                # The launcher has exited. Its detached child must still serve
                # exactly the saved bytes and directory listing.
                opener = build_opener(ProxyHandler({}))
                with opener.open(first["url"], timeout=3) as response:
                    self.assertIn(b"pair_0001_left.png", response.read())
                with opener.open(first["url"] + "pair_0001_left.png", timeout=3) as response:
                    self.assertEqual(response.read(), content)
                # The legacy service must retain the integrated service's
                # read-only and path confinement rules when reused.
                for method, target, expected in (
                    ("POST", "/pair_0001_left.png", 405),
                    ("PUT", "/pair_0001_left.png", 405),
                    ("DELETE", "/pair_0001_left.png", 405),
                    ("GET", "/%2e%2e/outside.txt", 403),
                ):
                    with self.subTest(method=method, target=target):
                        with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=3)) as connection:
                            connection.request(method, target)
                            response = connection.getresponse()
                            self.assertEqual(response.status, expected)
                            response.read()
                from pipe_twin.office_client import OfficeClient, OfficeClientConfig
                client = OfficeClient(OfficeClientConfig(
                    bind="127.0.0.1", capture_port=0, file_port=port, output_root=root,
                ))
                try:
                    details = client.start()
                    self.assertEqual(details["file_service"], "REUSED_VERIFIED_ROOT")
                finally:
                    client.stop()
                # Closing the integrated client must not close the external server.
                with opener.open(first["url"] + "pair_0001_left.png", timeout=3) as response:
                    self.assertEqual(response.read(), content)
                reused = subprocess.run(launcher, capture_output=True, text=True, timeout=15, cwd=remote_files._REPO)
                second = json.loads(reused.stdout)
                self.assertEqual(second["status"], "REUSED", second)
                self.assertEqual(second["pid"], pid)
                with closing(http.client.HTTPConnection("127.0.0.1", port, timeout=3, source_address=("127.0.0.2", 0))) as connection:
                    connection.request("GET", "/pair_0001_left.png")
                    response = connection.getresponse()
                    self.assertEqual(response.status, 403)
                    self.assertNotEqual(response.read(), content)
            finally:
                if pid is not None:
                    os.kill(pid, signal.SIGTERM)
                    deadline = time.monotonic() + 3
                    while remote_files._port_open("127.0.0.1", port) and time.monotonic() < deadline:
                        time.sleep(0.05)


class GuiFileServiceStartupTests(unittest.TestCase):
    def test_gui_leaves_service_lifecycle_to_office_client(self):
        from pipe_twin import gui

        root = Mock()
        with patch("tkinter.Tk", return_value=root), patch.object(gui, "_PipeTwinApplication") as application, patch.object(remote_files, "ensure_remote_files") as ensure:
            gui.launch_gui()
        ensure.assert_not_called()
        application.assert_called_once_with(root, None, None)
        root.mainloop.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
