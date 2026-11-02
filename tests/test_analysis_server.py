from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

from pipe_twin.analysis_server import AnalysisServer, _trusted_address
from pipe_twin.analysis_transport import build_request_bundle
from tests.test_stereo_analyzer import _manifest


def _fake_analyzer(manifest_path, report_output_path, evidence_dir, *, progress):
    mode = json.loads(Path(manifest_path).read_text(encoding="utf-8")).get("test_mode")
    if mode == "fail":
        raise ValueError("deliberate failure")
    if mode == "crash":
        os._exit(7)
    if mode == "block":
        progress("waiting for cancellation")
        time.sleep(60)
    progress({"stage": "evidence", "percent": 80})
    evidence = Path(evidence_dir) / "sections" / "evidence.txt"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("generated evidence", encoding="utf-8")
    report = {"inputs": {"manifest": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()},
              "evidence_files": [{"path": str(evidence)}], "pid": os.getpid()}
    return report


class AnalysisServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        source = self.root / "source"
        source.mkdir()
        self.manifest, self.payload = _manifest(source, count=1, draw_pipe=True)
        self.opener = build_opener(ProxyHandler({}))
        self.server = self._start()

    def _start(self, **kwargs):
        options = dict(analyzer=_fake_analyzer, capabilities={"execution_backend": "CPU"})
        options.update(kwargs)
        work_root = options.pop("work_root", self.root / "jobs")
        server = AnalysisServer("127.0.0.1", 0, work_root, **options).start()
        self.addCleanup(server.stop)
        return server

    def _request(self, path, data=None, headers=None):
        request = Request(self.server.base_url + path, data=data, headers=headers or {})
        try:
            with self.opener.open(request, timeout=10) as response:
                return response.status, response.read()
        except HTTPError as error:
            return error.code, error.read()

    def _submit(self, mode=None):
        payload = dict(self.payload)
        if mode:
            payload["test_mode"] = mode
        self.manifest.write_text(json.dumps(payload), encoding="utf-8")
        bundle = build_request_bundle(self.manifest)
        status, content = self._request("/jobs", bundle, {"Content-Type": "application/zip"})
        self.assertEqual(status, 202, content)
        return json.loads(content)["job_id"]

    def _wait(self, job_id, predicate=None, timeout=20):
        predicate = predicate or (lambda job: job["status"] in {"SUCCEEDED", "FAILED", "CANCELLED"})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            job = self.server.get_job(job_id)
            if predicate(job):
                return job
            time.sleep(0.025)
        self.fail(f"Timed out waiting for job: {self.server.get_job(job_id)}")

    def test_health_and_result_preserve_input_and_bind_report(self):
        status, response = self._request("/health")
        health = json.loads(response)
        self.assertEqual(status, 200)
        self.assertEqual(health["service"], "pipe-twin-analysis")
        self.assertEqual(health["protocol_version"], "1")
        self.assertGreater(health["port"], 0)
        self.assertEqual(health["capabilities"]["execution_backend"], "CPU")
        self.assertIsNone(health["active_job"])
        job_id = self._submit()
        original = self.manifest.read_bytes()
        result = self._wait(job_id)
        self.assertEqual(result["status"], "SUCCEEDED", result)
        self.assertIsNotNone(result["started_at"])
        self.assertIsNotNone(result["finished_at"])
        self.assertEqual(self.manifest.read_bytes(), original)
        saved = list(self.server.work_root.glob("j*/input/scene/manifest.json"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].read_bytes(), original)
        status, content = self._request(f"/jobs/{job_id}/result")
        self.assertEqual(status, 200)
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            report = json.loads(archive.read("report.json"))
            self.assertNotEqual(report["pid"], os.getpid())
            self.assertEqual(report["inputs"]["manifest"], hashlib.sha256(original).hexdigest())
            self.assertEqual(len(report["evidence_files"]), 1)
            self.assertEqual(archive.read(report["evidence_files"][0]["path"]), b"generated evidence")
        status, content = self._request(f"/jobs/{job_id}")
        self.assertEqual(json.loads(content)["status"], "SUCCEEDED")
        self.assertFalse(any(key.startswith("_") for key in json.loads(content)))

    def test_queue_limit_queued_cancel_running_cancel_and_recovery(self):
        self.server.stop()
        self.server = self._start(max_pending_jobs=1)
        running = self._submit("block")
        self._wait(running, lambda job: job["progress"] == "waiting for cancellation")
        queued = self._submit()
        self.assertEqual(self.server.get_job(queued)["status"], "QUEUED")
        health = self.server.health()
        self.assertEqual(health["active_job"]["job_id"], running)
        self.assertEqual(health["active_job"]["progress"], "waiting for cancellation")
        status, _ = self._request("/jobs", build_request_bundle(self.manifest), {"Content-Type": "application/zip"})
        self.assertEqual(status, 429)
        status, content = self._request(f"/jobs/{queued}/cancel", b"")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(content)["status"], "CANCELLED")
        replacement = self._submit()
        status, _ = self._request(f"/jobs/{running}/result")
        self.assertEqual(status, 409)
        start = time.monotonic()
        status, _ = self._request(f"/jobs/{running}/cancel", b"")
        self.assertEqual(status, 200)
        self.assertEqual(self._wait(running)["status"], "CANCELLED")
        self.assertLess(time.monotonic() - start, 5)
        result = self._wait(replacement)
        self.assertEqual(result["status"], "SUCCEEDED", result)
        self.assertEqual(self.server.cancel(running)["status"], "CANCELLED")

    def test_exception_and_process_crash_do_not_block_later_jobs(self):
        for mode, expected in (("fail", "deliberate failure"), ("crash", "code 7")):
            failed = self._wait(self._submit(mode))
            self.assertEqual(failed["status"], "FAILED", failed)
            self.assertIn(expected, failed["error"])
        recovered = self._wait(self._submit())
        self.assertEqual(recovered["status"], "SUCCEEDED", recovered)

    def test_stop_cancels_running_and_queued_jobs(self):
        running = self._submit("block")
        self._wait(running, lambda job: job["progress"] == "waiting for cancellation")
        queued = self._submit()
        start = time.monotonic()
        self.server.stop()
        self.assertLess(time.monotonic() - start, 5)
        self.assertEqual(self.server.get_job(running)["status"], "CANCELLED")
        self.assertEqual(self.server.get_job(queued)["status"], "CANCELLED")
        self.assertEqual(self.server.health()["status"], "STOPPED")
        self.server.stop()

    def test_request_validation_and_failed_upload_release_capacity(self):
        self.server.stop()
        self.server = self._start(max_pending_jobs=0, max_upload_bytes=65536)
        for body, content_type, expected in (
            (b"anything", "text/plain", 415),
            (b"invalid zip", "application/zip", 400),
            (b"x" * 65537, "application/zip", 413),
        ):
            code, _ = self._request("/jobs", body, {"Content-Type": content_type})
            self.assertEqual(code, expected)
        bad = io.BytesIO()
        with zipfile.ZipFile(bad, "w") as archive:
            archive.writestr("../outside.txt", "not permitted")
        code, _ = self._request("/jobs", bad.getvalue(), {"Content-Type": "application/zip"})
        self.assertEqual(code, 400)
        self.assertFalse((self.root / "outside.txt").exists())
        result = self._wait(self._submit())
        self.assertEqual(result["status"], "SUCCEEDED", result)
        self.assertEqual(self._request("/jobs/" + "0" * 32)[0], 404)
        self.assertEqual(self._request("/jobs/../anything/result")[0], 404)
        self.assertEqual(self._request("/health?extra=yes")[0], 404)

    def test_ambiguous_http_framing_is_rejected(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=5)
        self.addCleanup(connection.close)
        connection.putrequest("POST", "/jobs")
        connection.putheader("Content-Type", "application/zip")
        connection.putheader("Content-Length", "1")
        connection.putheader("Content-Length", "2")
        connection.endheaders()
        self.assertEqual(connection.getresponse().status, 400)
        connection.close()
        connection = http.client.HTTPConnection("127.0.0.1", self.server.port, timeout=5)
        self.addCleanup(connection.close)
        connection.request("POST", "/jobs", b"", {"Transfer-Encoding": "chunked"})
        self.assertEqual(connection.getresponse().status, 400)

    def test_real_analyzer_runs_in_child_on_synthetic_fixture(self):
        self.server.stop()
        self.server = self._start(analyzer=None)
        result = self._wait(self._submit(), timeout=30)
        self.assertEqual(result["status"], "SUCCEEDED", result)
        code, content = self._request(f'/jobs/{result["job_id"]}/result')
        self.assertEqual(code, 200)
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            report = json.loads(archive.read("report.json"))
            self.assertIn("pipes", report)
            self.assertEqual(len(report["pipes"]), 1)

    def test_address_filter_and_configuration_validation(self):
        for address in ("127.0.0.1", "10.3.2.1", "172.16.0.2", "192.168.1.2", "100.64.1.2", "::1", "::ffff:127.0.0.1"):
            self.assertTrue(_trusted_address(address), address)
        for address in ("0.0.0.0", "8.8.8.8", "169.254.1.2", "100.63.255.255", "172.32.0.1", "not-an-ip"):
            self.assertFalse(_trusted_address(address), address)
        for options in ({"port": -1}, {"port": True}, {"bind": "8.8.8.8"},
                        {"max_pending_jobs": -1}, {"max_upload_bytes": 0}):
            with self.assertRaises(ValueError):
                AnalysisServer(**options)

    def test_long_work_root_still_writes_real_evidence_with_windows_limit(self):
        self.server.stop()
        # Reproduce real capture identifiers and a realistically deep output
        # root; the former UUID directory layout exceeded 260 characters.
        capture_id = "elevation-20261102-115517-4e0034c9"
        self.payload["capture"]["capture_groups"][0]["capture_id"] = capture_id
        component = "long-session-" + "x" * max(1, 130 - len(str(self.root)) - 14)
        work_root = self.root / component
        old_temporary = (work_root / ("a" * 32) / "input/scene" /
                         ("_remote_result_" + "b" * 32) / "evidence" /
                         f".{capture_id}_left_overlay.png.12345678.tmp")
        self.assertGreater(len(str(old_temporary)), 260)
        self.server = self._start(analyzer=None, work_root=work_root)
        result = self._wait(self._submit(), timeout=30)
        self.assertEqual(result["status"], "SUCCEEDED", result)
        evidence = list(work_root.glob("j*/input/scene/r*/evidence/*_overlay.png"))
        self.assertEqual(len(evidence), 2)
        self.assertTrue(all(len(str(path)) + 14 < 260 for path in evidence))
        self.assertTrue(all(path.stat().st_size > 0 for path in evidence))


if __name__ == "__main__":
    unittest.main()
