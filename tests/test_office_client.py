from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipe_twin.capture_agent import CaptureAgentError, get_remote_capture, submit_remote_capture
from pipe_twin.cli import main
from pipe_twin.office_client import (
    OfficeClient,
    OfficeClientConfig,
    OfficeClientError,
    load_office_config,
    save_office_config,
)
from pipe_twin.remote_fetch import fetch_stereo_run


def _fake_capture(output_dir: str | Path, **_kwargs) -> Path:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "pair_0001_left.png").write_bytes(b"left")
    (root / "pair_0001_right.png").write_bytes(b"right")
    path = root / "capture.json"
    path.write_text(json.dumps({"status": "COMPLETED", "pair_count": 1, "captures": []}), encoding="utf-8")
    return path


@pytest.mark.parametrize("existing_token", [False, True])
def test_office_client_starts_both_services_and_remote_job(tmp_path: Path, monkeypatch, capsys, existing_token):
    token_path = tmp_path / "secret" / "capture.token"
    if existing_token:
        token_path.parent.mkdir()
        token_path.write_text("unused legacy token", encoding="ascii")
    monkeypatch.setenv("PIPE_TWIN_TOKEN_FILE", str(token_path))
    client = OfficeClient(
        OfficeClientConfig(
            bind="127.0.0.1",
            capture_port=0,
            file_port=0,
            output_root=tmp_path / "runs",
        ),
        capture_fn=_fake_capture,
    )
    details = client.start()
    try:
        assert details["status"] == "LOCAL_ONLY"
        assert details["file_service"] == "STARTED"
        assert details["auth_mode"] == "TAILSCALE_ONLY"
        assert details["auth_required"] is False
        assert details["token_created"] is False
        assert details["token_file"] is None
        assert main([
            "remote-capture", "--agent-url", details["capture_url"],
            "--count", "1", "--wait", "--poll-s", "0.1",
        ]) == 0
        assert '"state": "COMPLETED"' in capsys.readouterr().out
        job_path, = (tmp_path / "runs").glob("*/job.json")
        job_id = job_path.parent.name
        result = get_remote_capture(details["capture_url"], None, job_id)
        assert result["state"] == "COMPLETED"
        assert details["file_base_url"].startswith("http://127.0.0.1:")
        report_path = fetch_stereo_run(
            result["run_url"], tmp_path / "download", workers=2, retries=0,
        )
        assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "PASS"
        assert token_path.exists() is existing_token
        if existing_token:
            assert token_path.read_text(encoding="ascii") == "unused legacy token"
    finally:
        client.stop()
    assert client.snapshot()["status"] == "STOPPED"


def test_office_client_can_opt_into_bearer_token(tmp_path: Path):
    token_path = tmp_path / "secret" / "capture.token"
    client = OfficeClient(
        OfficeClientConfig(
            bind="127.0.0.1", capture_port=0, file_port=0,
            require_token=True, token_file=token_path, output_root=tmp_path / "runs",
        ),
        capture_fn=_fake_capture,
    )
    details = client.start()
    try:
        assert details["auth_required"] is True
        assert details["auth_mode"] == "BEARER_TOKEN"
        assert details["token_created"] is True
        assert token_path.is_file()
        with pytest.raises(CaptureAgentError, match="401"):
            submit_remote_capture(details["capture_url"], None, {"count": 1})
        with pytest.raises(CaptureAgentError, match="401"):
            submit_remote_capture(details["capture_url"], "wrong-token", {"count": 1})
        token = token_path.read_text(encoding="ascii").strip()
        queued = submit_remote_capture(details["capture_url"], token, {"count": 1})
        assert queued["job_id"]
    finally:
        client.stop()


def test_office_client_config_round_trip(tmp_path: Path):
    path = tmp_path / "office-client.json"
    config = OfficeClientConfig(
        bind="127.0.0.1", capture_port=8770, file_port=8765,
        require_token=True, token_file=tmp_path / "token", output_root=tmp_path / "runs",
    )
    save_office_config(config, path)
    loaded = load_office_config(path)
    assert loaded.bind == config.bind
    assert loaded.require_token is True
    assert loaded.token_file == config.token_file
    assert loaded.output_root == config.output_root


def test_office_client_rejects_token_inside_evidence_root(tmp_path: Path):
    try:
        OfficeClientConfig(
            bind="127.0.0.1", require_token=True, token_file=tmp_path / "runs" / "token",
            output_root=tmp_path / "runs",
        ).resolved()
    except OfficeClientError as error:
        assert "照片下载目录之外" in str(error)
    else:
        raise AssertionError("expected token/evidence-root separation")
