from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

import pytest

from pipe_twin.capture_agent import (
    CaptureAgentConfig,
    CaptureAgentError,
    CaptureAgentHTTPServer,
    CaptureJobManager,
    _validate_request,
    get_remote_capture,
    submit_remote_capture,
    write_evidence_manifest,
)


def _fake_capture(output_dir: str | Path, **_kwargs) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "pair_0001_left.png").write_bytes(b"left-frame")
    (root / "pair_0001_right.png").write_bytes(b"right-frame")
    capture = {
        "schema_version": "1.0",
        "capture_kind": "remote_stereo_raw",
        "status": "COMPLETED",
        "pair_count": 1,
        "captures": [],
    }
    path = root / "capture.json"
    path.write_text(json.dumps(capture), encoding="utf-8")
    return path


def _server(tmp_path: Path, token: str | None = "test-token", capture_fn=_fake_capture):
    config = CaptureAgentConfig(
        output_root=tmp_path / "runs",
        eye_width=16,
        eye_height=16,
        file_base_url="http://127.0.0.1:8765",
        default_count=1,
        max_count=5,
    )
    manager = CaptureJobManager(config, capture_fn=capture_fn)
    server = CaptureAgentHTTPServer(("127.0.0.1", 0), manager, token)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return config, manager, server, thread, f"http://127.0.0.1:{server.server_port}", token


def _stop(manager, server, thread):
    server.shutdown()
    server.server_close()
    manager.shutdown()
    thread.join(timeout=2)


def _wait(agent_url: str, token: str | None, job_id: str) -> dict:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        result = get_remote_capture(agent_url, token, job_id)
        if result["state"] in {"COMPLETED", "FAILED"}:
            return result
        time.sleep(0.01)
    raise AssertionError("capture job did not finish")


def test_write_evidence_manifest_hashes_files_and_excludes_itself(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "capture.json").write_text("{}", encoding="utf-8")
    manifest = write_evidence_manifest(root)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["self_excluded"] is True
    assert payload["file_count"] == 1
    item = payload["files"][0]
    data = (root / item["path"]).read_bytes()
    assert item["path"] == "capture.json"
    assert item["size_bytes"] == len(data)
    assert item["sha256"] == hashlib.sha256(data).hexdigest()


def test_request_defaults_and_duration_only_requests_remain_bounded(tmp_path: Path):
    config = CaptureAgentConfig(output_root=tmp_path / "runs", max_count=5)
    assert _validate_request({}, config)["count"] == 1
    assert _validate_request({}, config)["duration_s"] is None
    assert _validate_request({"exposure_ms": 1000 / 30}, config)["exposure_ms"] == 1000 / 30
    assert _validate_request({"exposure_ms": None}, config)["exposure_ms"] is None
    assert _validate_request({}, config)["warmup_s"] == 20.0
    assert _validate_request({"warmup_s": 0}, config)["warmup_s"] == 0.0
    duration_request = _validate_request({"duration_s": 1.0}, config)
    assert duration_request["count"] == 5
    assert duration_request["duration_s"] == 1.0
    with pytest.raises(CaptureAgentError, match="count"):
        _validate_request({"count": 6}, config)
    for exposure in (True, 0, 250.1, float("nan")):
        with pytest.raises(CaptureAgentError, match="exposure_ms"):
            _validate_request({"exposure_ms": exposure}, config)
    for warmup in (-1, 120.1, float("nan")):
        with pytest.raises(CaptureAgentError, match="warmup_s"):
            _validate_request({"warmup_s": warmup}, config)


def test_authenticated_remote_job_creates_fetchable_run(tmp_path: Path):
    config, manager, server, thread, agent_url, token = _server(tmp_path)
    try:
        queued = submit_remote_capture(
            agent_url,
            token,
            {"count": 1, "interval_s": 0},
        )
        assert queued["state"] in {"QUEUED", "RUNNING", "COMPLETED"}
        result = _wait(agent_url, token, queued["job_id"])
        assert result["state"] == "COMPLETED"
        assert result["pair_count"] == 1
        assert result["run_url"].endswith(f"/{queued['job_id']}/")
        root = config.output_root / queued["job_id"]
        evidence = json.loads((root / "evidence_manifest.json").read_text(encoding="utf-8"))
        assert {item["path"] for item in evidence["files"]} == {
            "capture.json",
            "job.json",
            "pair_0001_left.png",
            "pair_0001_right.png",
        }
        assert (root / "job.json").is_file()
    finally:
        _stop(manager, server, thread)


def test_remote_job_forwards_requested_exposure_to_capture_function(tmp_path: Path):
    forwarded: dict[str, object] = {}

    def capture(output_dir: str | Path, **kwargs):
        forwarded.update(kwargs)
        return _fake_capture(output_dir, **kwargs)

    _config, manager, server, thread, agent_url, token = _server(tmp_path, capture_fn=capture)
    try:
        requested_ms = 1000 / 30
        queued = submit_remote_capture(
            agent_url,
            token,
            {"count": 1, "exposure_ms": requested_ms},
        )
        result = _wait(agent_url, token, queued["job_id"])
        assert result["state"] == "COMPLETED"
        assert forwarded["exposure_ms"] == requested_ms
        assert forwarded["warmup_s"] == 20.0
        assert result["request"]["exposure_ms"] == requested_ms
        assert result["request"]["warmup_s"] == 20.0
    finally:
        _stop(manager, server, thread)


def test_wrong_token_and_unknown_fields_are_rejected(tmp_path: Path):
    _config, manager, server, thread, agent_url, _token = _server(tmp_path)
    try:
        with pytest.raises(CaptureAgentError, match="401"):
            submit_remote_capture(agent_url, "wrong", {"count": 1})
        with pytest.raises(CaptureAgentError, match="400"):
            submit_remote_capture(agent_url, "test-token", {"shell": "whoami"})
    finally:
        _stop(manager, server, thread)


@pytest.mark.parametrize("token", [None, "test-token"])
def test_second_request_is_rejected_while_camera_is_busy(tmp_path: Path, token):
    started = threading.Event()
    release = threading.Event()

    def blocking_capture(output_dir: str | Path, **kwargs):
        started.set()
        release.wait(timeout=3)
        return _fake_capture(output_dir, **kwargs)

    _config, manager, server, thread, agent_url, token = _server(tmp_path, token=token, capture_fn=blocking_capture)
    try:
        first = submit_remote_capture(agent_url, token, {"count": 1})
        assert started.wait(timeout=2)
        with pytest.raises(CaptureAgentError, match="409"):
            submit_remote_capture(agent_url, token, {"count": 1})
        release.set()
        assert _wait(agent_url, token, first["job_id"])["state"] == "COMPLETED"
    finally:
        release.set()
        _stop(manager, server, thread)


def test_bearer_auth_is_required_for_health(tmp_path: Path):
    _config, manager, server, thread, agent_url, _token = _server(tmp_path)
    try:
        request = Request(agent_url + "/v1/health")
        with pytest.raises(HTTPError) as error:
            build_opener().open(request, timeout=3)
        assert error.value.code == 401
    finally:
        _stop(manager, server, thread)


def test_tailnet_mode_accepts_bounded_job_without_bearer_token(tmp_path: Path):
    _config, manager, server, thread, agent_url, token = _server(tmp_path, token=None)
    try:
        with build_opener(ProxyHandler({})).open(agent_url + "/v1/health", timeout=3) as response:
            health = json.load(response)
        assert health["status"] == "READY"
        assert health["auth_required"] is False
        with pytest.raises(CaptureAgentError, match="400"):
            submit_remote_capture(agent_url, None, {"count": 6})
        with pytest.raises(CaptureAgentError, match="400"):
            submit_remote_capture(agent_url, None, {"shell": "whoami"})
        queued = submit_remote_capture(agent_url, token, {"count": 1})
        result = _wait(agent_url, token, queued["job_id"])
        assert result["state"] == "COMPLETED"
        assert result["pair_count"] == 1
        assert result["run_url"].endswith(f"/{queued['job_id']}/")
    finally:
        _stop(manager, server, thread)


@pytest.mark.parametrize("bind", ["0.0.0.0", "", "192.168.1.10", "203.0.113.1", "localhost"])
def test_token_free_server_rejects_unrestricted_or_non_tailnet_bind(tmp_path: Path, bind):
    manager = CaptureJobManager(CaptureAgentConfig(output_root=tmp_path / "runs"))
    try:
        with pytest.raises(CaptureAgentError, match="loopback or Tailscale"):
            CaptureAgentHTTPServer((bind, 0), manager, None)
    finally:
        manager.shutdown()


def test_capture_endpoint_requires_json_content_type(tmp_path: Path):
    _config, manager, server, thread, agent_url, token = _server(tmp_path)
    try:
        request = Request(
            agent_url + "/v1/captures",
            data=b"{}",
            method="POST",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "text/plain"},
        )
        with pytest.raises(HTTPError) as error:
            build_opener().open(request, timeout=3)
        assert error.value.code == 415
    finally:
        _stop(manager, server, thread)



def test_camera_lease_serializes_same_index(tmp_path: Path, monkeypatch):
    from pipe_twin.camera_lock import CameraBusyError, CameraLease

    monkeypatch.setenv("PIPE_TWIN_CAMERA_LOCK_DIR", str(tmp_path / "locks"))
    first = CameraLease([31])
    second = CameraLease([31])
    first.acquire()
    try:
        with pytest.raises(CameraBusyError, match="CAMERA_BUSY"):
            second.acquire()
    finally:
        first.release()
        second.release()
