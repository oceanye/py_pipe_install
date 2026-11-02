from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

import test_elevation_gui as fixtures
from pipe_twin.analysis_runtime_settings import AnalysisRuntimeSettings, load_runtime_settings, save_runtime_settings


def http_gui_analyzer(manifest_path, *, report_output_path, evidence_dir, progress):
    """Spawnable analyzer fixture: real HTTP/ZIP/process, no camera or provider."""
    result = RuntimeGuiTests.report(manifest_path, report_output_path=report_output_path,
                                    evidence_dir=evidence_dir, progress=progress)
    evidence = Path(evidence_dir) / "sections" / "candidate.txt"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("fixture section evidence", encoding="utf-8")
    result["evidence_files"] = [{"path": str(evidence)}]
    result["fixture_section"] = {"path": str(evidence)}
    Path(report_output_path).write_text(json.dumps(result), encoding="utf-8")
    return result


class RuntimeSettingsTests(unittest.TestCase):
    def test_defaults_round_trip_and_invalid_config_are_local_to_given_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "machine" / "runtime.json"
            self.assertEqual(load_runtime_settings(path=path).mode, "local")
            self.assertEqual(load_runtime_settings(path=path).port, 8771)
            settings = AnalysisRuntimeSettings(mode="client", address="100.80.20.1", port=8871)
            save_runtime_settings(settings, path=path)
            self.assertEqual(load_runtime_settings(path=path), settings)
            self.assertEqual(settings.base_url, "http://100.80.20.1:8871")
            path.write_text('{"schema_version":"wrong"}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "本机运行设置"):
                load_runtime_settings(path=path)

    def test_endpoint_validation_and_ipv6(self):
        for values in ({"mode": "invalid"}, {"port": True}, {"port": 0}, {"port": 65536},
                       {"address": "http://host:8771"}, {"address": "user@host"}, {"address": ""}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                AnalysisRuntimeSettings(**values)
        self.assertEqual(AnalysisRuntimeSettings(address="[::1]").base_url, "http://[::1]:8771")


class RuntimeGuiTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.ElevationGuiTests.setUpClass.__func__)
    tearDownClass = classmethod(fixtures.ElevationGuiTests.tearDownClass.__func__)
    setUp = fixtures.ElevationGuiTests.setUp
    tearDown = fixtures.ElevationGuiTests.tearDown
    finish_worker = fixtures.ElevationGuiTests.finish_worker

    def mode(self, mode):
        from pipe_twin.analysis_runtime_settings import MODE_LABELS
        runtime = self.dialog.runtime
        runtime.mode_label.set(MODE_LABELS[mode])
        self.assertTrue(runtime.apply(), runtime.status.get())

    def pump(self, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertTrue(predicate())

    @staticmethod
    def report(path, *, report_output_path, evidence_dir, progress=None, cancel_event=None):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        inputs = {"manifest": hashlib.sha256(path.read_bytes()).hexdigest(), "model": manifest["model"]["sha256"]}
        for group in manifest["capture"]["capture_groups"]:
            inputs.update({f"{group['capture_id']}_{role}": view["sha256"] for role, view in group["views"].items()})
        report = {"pipes": [], "counts": {"INSTALLED": 0, "NOT_INSTALLED": 0, "UNKNOWN": 1}, "inputs": inputs}
        Path(report_output_path).write_text(json.dumps(report), encoding="utf-8")
        if progress:
            progress("服务器正在分析当前批")
        return report

    def test_frontend_submits_only_current_batch_and_preserves_local_archive(self):
        self.dialog.submit(False)
        self.finish_worker()
        old_manifest = self.dialog.last_manifest
        old_bytes = old_manifest.read_bytes()
        self.mode("client")
        self.assertFalse(self.dialog.history.get())
        self.dialog.history.set(True)  # Frontend scope does not inherit old local history.
        with mock.patch("pipe_twin.analysis_transport.RemoteAnalysisClient.analyze", side_effect=self.report) as analyze, \
             mock.patch("pipe_twin.stereo_analyzer.analyze_stereo_capture", side_effect=AssertionError("local analyzer must not run")):
            self.dialog.submit(True)
            self.finish_worker()
        analyze.assert_called_once()
        self.app.messagebox.showerror.assert_not_called()
        self.assertIsNotNone(self.dialog.report)
        self.assertNotEqual(self.dialog.last_manifest, old_manifest)
        self.assertEqual(old_manifest.read_bytes(), old_bytes)
        saved = json.loads(self.dialog.last_manifest.read_text(encoding="utf-8"))
        self.assertEqual(len(saved["capture"]["capture_groups"]), 1)
        self.assertEqual(json.loads((self.output / "last_scene.json").read_text())["manifest_path"], str(self.dialog.last_manifest))

    def test_gui_current_batch_round_trip_through_real_http_and_child_process(self):
        from pipe_twin.analysis_server import AnalysisServer
        server = AnalysisServer("127.0.0.1", 0, self.output / "server-jobs",
                                analyzer=http_gui_analyzer, capabilities={"execution_backend": "CPU"}).start()
        try:
            runtime = self.dialog.runtime
            runtime.port.set(str(server.port))
            self.mode("client")
            runtime.check_connection()
            self.pump(lambda: not runtime.checking)
            self.assertIn("已连接", runtime.status.get())
            self.assertIn("服务器计算：CPU", runtime.server_capabilities.get())
            with mock.patch("pipe_twin.stereo_analyzer.analyze_stereo_capture", side_effect=AssertionError("frontend must not analyze")):
                self.dialog.submit(True)
                self.finish_worker()
            self.app.messagebox.showerror.assert_not_called()
            result = self.dialog.report
            self.assertIsNotNone(result)
            self.assertTrue(result["remote_analysis"]["request_id"])
            self.assertEqual(server.jobs()[0]["status"], "SUCCEEDED")
            evidence = Path(result["evidence_files"][0]["path"])
            self.assertTrue(evidence.is_relative_to(self.dialog.last_manifest.parent))
            self.assertEqual(evidence.read_text(encoding="utf-8"), "fixture section evidence")
            self.assertEqual(result["fixture_section"]["path"], str(evidence))
        finally:
            server.stop()

    def test_frontend_retains_local_kimi_credentials_and_independent_ai_workflow(self):
        from pipe_twin.ai_settings import AISettings
        self.mode("client")
        self.assertNotIn("disabled", self.dialog.ai_run_button.state())
        with mock.patch("pipe_twin.ai_settings.configured_status", return_value={"configured": True}), \
             mock.patch("pipe_twin.ai_settings.load_ai_settings", return_value=AISettings()), \
             mock.patch("pipe_twin.ai_provider.request_section_assessment", side_effect=fixtures.ElevationGuiTests._provider_fixture) as provider, \
             mock.patch("pipe_twin.stereo_analyzer.analyze_stereo_capture", side_effect=AssertionError("no local depth")), \
             mock.patch("pipe_twin.analysis_transport.RemoteAnalysisClient.analyze", side_effect=AssertionError("no AI credential upload")):
            self.dialog.start_ai_evaluation()
            self.finish_worker()
        provider.assert_called_once()
        self.app.messagebox.showerror.assert_not_called()
        self.assertTrue(self.dialog.last_manifest.with_name("ai_section_result.json").is_file())

    def test_old_generation_report_does_not_replace_new_scene_or_report(self):
        self.mode("client")
        entered, release = threading.Event(), threading.Event()

        def delayed(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(5))
            return self.report(*args, **kwargs)

        with mock.patch("pipe_twin.analysis_transport.RemoteAnalysisClient.analyze", side_effect=delayed):
            self.dialog.submit(True)
            self.pump(entered.is_set)
            self.dialog.fields["left_time"].set("2026-11-02T10:00:00+08:00")
            new_report = {"new_scene": True}
            self.dialog.report = new_report
            release.set()
            self.finish_worker()
        self.assertIs(self.dialog.report, new_report)
        self.assertIsNone(self.dialog.last_manifest)
        self.assertFalse((self.output / "last_scene.json").exists())
        self.assertTrue(list((self.output / "captures").glob("*/report.json")))

    def test_modified_original_photo_rejects_returned_report_without_overwriting_scene(self):
        self.mode("client")
        entered, release = threading.Event(), threading.Event()

        def delayed(*args, **kwargs):
            entered.set()
            self.assertTrue(release.wait(5))
            return self.report(*args, **kwargs)

        with mock.patch("pipe_twin.analysis_transport.RemoteAnalysisClient.analyze", side_effect=delayed):
            self.dialog.submit(True)
            self.pump(entered.is_set)
            photo = Path(self.dialog.fields["left"].get())
            photo.write_bytes(photo.read_bytes() + b"changed-after-upload")
            new_report = {"new_scene": True}
            self.dialog.report = new_report
            release.set()
            self.finish_worker()
        self.assertIs(self.dialog.report, new_report)
        self.assertIsNone(self.dialog.last_manifest)
        self.assertIn("现场文件在计算时发生改变", self.dialog.runtime.status.get())
        self.assertFalse((self.output / "last_scene.json").exists())

    def test_cancelled_old_completion_cannot_unlock_new_remote_task(self):
        runtime = self.dialog.runtime
        old_event, new_event = threading.Event(), threading.Event()
        generation = self.dialog.generation
        runtime.active = (generation, old_event)
        self.dialog.busy = True
        runtime.cancel_remote()
        self.assertTrue(old_event.is_set())
        runtime.active = (generation, new_event)
        self.dialog.busy = True
        runtime.accept_remote(generation, "remote_ok", (old_event, Path("stale.json"), {}, False, {}))
        self.assertTrue(self.dialog.busy)
        self.assertIs(runtime.active[1], new_event)
        runtime.cancel_remote()

    def test_busy_mode_change_does_not_save_or_change_execution(self):
        runtime = self.dialog.runtime
        self.dialog.busy = True
        runtime.mode_label.set("服务器端")
        self.assertFalse(runtime.apply())
        self.assertEqual(runtime.mode, "local")
        self.assertFalse((Path(self.tmp.name) / "runtime.json").exists())
        self.dialog.busy = False

    def test_server_mode_blocks_camera_local_analysis_and_ai(self):
        self.mode("server")
        with mock.patch("pipe_twin.capture_gui.StereoCameraDialog") as camera, \
             mock.patch.object(self.dialog, "_worker") as analyze, \
             mock.patch.object(self.dialog.ai_controller, "configured") as ai:
            self.dialog.capture_camera()
            self.dialog.open_calibration()
            self.dialog.open_ground_monitor()
            self.dialog.inspect_ground()
            self.dialog.submit(True)
            self.dialog.start_ai_evaluation()
        camera.assert_not_called()
        analyze.assert_not_called()
        ai.assert_not_called()
        self.assertIsNone(self.dialog.calibration_dialog)
        self.assertIsNone(self.dialog.ground_monitor)
        self.assertIn("disabled", self.dialog.run_button.state())
        self.assertNotIn("disabled", self.dialog.settings_button.state())
        self.assertIsNone(self.dialog.runtime.server)  # Explicit start is required.

    def test_connection_check_runs_off_tk_and_reports_failure(self):
        self.mode("client")
        runtime = self.dialog.runtime
        entered, release = threading.Event(), threading.Event()
        main_thread = threading.get_ident()

        def health():
            self.assertNotEqual(threading.get_ident(), main_thread)
            entered.set()
            self.assertTrue(release.wait(5))
            raise OSError("fixture connection refused")

        with mock.patch("pipe_twin.analysis_transport.RemoteAnalysisClient.health", side_effect=health):
            started = time.monotonic()
            runtime.check_connection()
            self.assertLess(time.monotonic() - started, 0.5)
            self.pump(entered.is_set)
            self.assertTrue(runtime.checking)
            self.root.update()
            release.set()
            self.pump(lambda: not runtime.checking)
        self.assertIn("连接失败", runtime.status.get())
        self.assertIn("fixture connection refused", runtime.status.get())

    def test_server_start_address_progress_busy_guard_and_shutdown(self):
        self.mode("server")
        runtime = self.dialog.runtime
        main_thread = threading.get_ident()
        stopped = threading.Event()
        server = mock.Mock(port=8771, is_busy=False)
        server.start.side_effect = lambda: self.assertNotEqual(threading.get_ident(), main_thread)
        server.stop.side_effect = stopped.set
        server.health.return_value = {"active_job": {"status": "RUNNING", "progress": "计算点云"}, "jobs": []}
        with mock.patch("pipe_twin.analysis_server.AnalysisServer", return_value=server):
            runtime.toggle_server()
            self.pump(lambda: not runtime.changing)
            self.assertEqual(runtime.endpoint.get(), "http://127.0.0.1:8771")
            self.assertIn("计算点云", runtime.status.get())
            server.is_busy = True
            runtime.mode_label.set("本地运算")
            self.assertFalse(runtime.apply())
            runtime.toggle_server()
            server.stop.assert_not_called()
            self.dialog.close()
            self.assertTrue(stopped.wait(3))
        # tearDown expects a live window; close is intentionally idempotent.

    def test_close_during_server_start_releases_late_service(self):
        self.mode("server")
        runtime = self.dialog.runtime
        entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
        server = mock.Mock(port=8771, is_busy=False)

        def start():
            entered.set()
            self.assertTrue(release.wait(5))

        server.start.side_effect = start
        server.stop.side_effect = stopped.set
        with mock.patch("pipe_twin.analysis_server.AnalysisServer", return_value=server):
            runtime.toggle_server()
            self.pump(entered.is_set)
            runtime.close()
            release.set()
            self.assertTrue(stopped.wait(3))
        self.assertIsNone(runtime.server)

    def test_frontend_capture_schedules_current_batch_submission(self):
        self.mode("client")
        with mock.patch("pipe_twin.capture_gui.StereoCameraDialog") as camera:
            self.dialog.capture_camera()
            camera.assert_called_once()
        self.assertTrue(self.dialog._analyze_after_camera_capture)
        self.dialog.camera_dialog = None
        with mock.patch.object(self.dialog, "submit") as submit, mock.patch.object(self.dialog, "load_images"):
            self.dialog._camera_capture_finished(None, None)
            self.pump(lambda: submit.called)
            submit.assert_called_once_with(True)


if __name__ == "__main__":
    unittest.main()
