from __future__ import annotations

import json
from pathlib import Path

from pipe_twin.capture_agent import get_remote_capture, submit_remote_capture
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


def test_office_client_starts_both_services_and_remote_job(tmp_path: Path):
    client = OfficeClient(
        OfficeClientConfig(
            bind="127.0.0.1",
            capture_port=0,
            file_port=0,
            token_file=tmp_path / "secret" / "capture.token",
            output_root=tmp_path / "runs",
        ),
        capture_fn=_fake_capture,
    )
    details = client.start()
    try:
        assert details["status"] == "LOCAL_ONLY"
        assert details["file_service"] == "STARTED"
        assert details["token_created"] is True
        assert Path(details["token_file"]).is_file()
        token = Path(details["token_file"]).read_text(encoding="ascii").strip()
        queued = submit_remote_capture(details["capture_url"], token, {"count": 1})
        result = get_remote_capture(details["capture_url"], token, queued["job_id"])
        for _ in range(100):
            if result["state"] in {"COMPLETED", "FAILED"}:
                break
            result = get_remote_capture(details["capture_url"], token, queued["job_id"])
        assert result["state"] == "COMPLETED"
        assert details["file_base_url"].startswith("http://127.0.0.1:")
        report_path = fetch_stereo_run(
            result["run_url"], tmp_path / "download", workers=2, retries=0,
        )
        assert json.loads(report_path.read_text(encoding="utf-8"))["status"] == "PASS"
    finally:
        client.stop()
    assert client.snapshot()["status"] == "STOPPED"


def test_office_client_config_round_trip(tmp_path: Path):
    path = tmp_path / "office-client.json"
    config = OfficeClientConfig(
        bind="127.0.0.1", capture_port=8770, file_port=8765,
        token_file=tmp_path / "token", output_root=tmp_path / "runs",
    )
    save_office_config(config, path)
    loaded = load_office_config(path)
    assert loaded.bind == config.bind
    assert loaded.token_file == config.token_file
    assert loaded.output_root == config.output_root


def test_office_client_rejects_token_inside_evidence_root(tmp_path: Path):
    try:
        OfficeClientConfig(
            bind="127.0.0.1", token_file=tmp_path / "runs" / "token",
            output_root=tmp_path / "runs",
        ).resolved()
    except OfficeClientError as error:
        assert "照片下载目录之外" in str(error)
    else:
        raise AssertionError("expected token/evidence-root separation")
