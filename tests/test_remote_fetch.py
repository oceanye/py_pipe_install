from __future__ import annotations

import hashlib
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from pipe_twin.remote_fetch import RemoteFetchError, fetch_stereo_run


def manifest(files):
    return {
        "files": [{"path": name, "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                  for name, data in files.items()],
        "file_count": len(files), "total_size_bytes": sum(map(len, files.values())),
    }


@contextmanager
def server(files, description=None, *, change_manifest=False):
    state = {"requests": [], "manifest_reads": 0}
    payload = manifest(files) if description is None else description

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            state["requests"].append(self.path)
            if self.path == "/run/evidence_manifest.json":
                state["manifest_reads"] += 1
                data = json.dumps(payload).encode()
                if change_manifest and state["manifest_reads"] > 1:
                    data += b"\n"
            else:
                name = self.path.removeprefix("/run/")
                if name not in files:
                    self.send_error(404)
                    return
                data = files[name]
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *_args):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/run/", state
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_download_verifies_files_and_reuses_matching_content(tmp_path, monkeypatch):
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:1")
    files = {"attempt/capture.json": b'{"pair_count": 0}', "logs/empty.txt": b""}
    output = tmp_path / "run"
    with server(files) as (url, state):
        report_path = fetch_stereo_run(url, output)
        report = json.loads(report_path.read_text())
        assert report["status"] == "PASS"
        assert report["manifest_unchanged"] is True
        assert report["verified_bytes"] == sum(map(len, files.values()))
        assert report_path.parent == output.parent
        for name, data in files.items():
            assert (output / name).read_bytes() == data
        state["requests"].clear()
        report_path = fetch_stereo_run(url, output)
        assert state["requests"] == ["/run/evidence_manifest.json"] * 2
        assert all(x["status"] == "REUSED" for x in json.loads(report_path.read_text())["files"])


def test_hash_mismatch_fails_without_installing_bad_file(tmp_path):
    declared = manifest({"left.png": b"good"})
    with server({"left.png": b"evil"}, declared) as (url, _):
        with pytest.raises(RemoteFetchError, match="failed verification"):
            fetch_stereo_run(url, tmp_path / "run", retries=0)
    assert not (tmp_path / "run/left.png").exists()
    assert not list((tmp_path / "run").glob(".fetch-*"))
    report = json.loads((tmp_path / "run.fetch.json").read_text())
    assert report["status"] == "FAILED"
    assert len(report["errors"]) == 1


@pytest.mark.parametrize("name", ["../outside.txt", "/outside.txt", "C:/outside.txt", "a\\b.txt", "a:b", "NUL", "dir/../x", "dir/", "x."])
def test_paths_cannot_leave_evidence_directory(tmp_path, name):
    with server({name: b"bad"}) as (url, state):
        with pytest.raises(RemoteFetchError, match="path"):
            fetch_stereo_run(url, tmp_path / "run", retries=0)
    assert state["requests"] == ["/run/evidence_manifest.json"]
    assert not (tmp_path / "outside.txt").exists()


def test_local_corruption_is_repaired_by_verified_download(tmp_path):
    files = {"left.png": b"original"}
    with server(files) as (url, _):
        fetch_stereo_run(url, tmp_path / "run")
        (tmp_path / "run/left.png").write_bytes(b"corrupt!")
        path = fetch_stereo_run(url, tmp_path / "run")
    assert (tmp_path / "run/left.png").read_bytes() == b"original"
    assert json.loads(path.read_text())["files"][0]["status"] == "DOWNLOADED"


def test_source_manifest_change_is_not_reported_as_success(tmp_path):
    with server({"left.png": b"frame"}, change_manifest=True) as (url, _):
        with pytest.raises(RemoteFetchError, match="changed"):
            fetch_stereo_run(url, tmp_path / "run")
    report = json.loads((tmp_path / "run.fetch.json").read_text())
    assert report["status"] == "FAILED"
    assert report["manifest_unchanged"] is False


def test_different_run_cannot_overwrite_existing_manifest(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "evidence_manifest.json").write_text('{"old": true}')
    with server({"left.png": b"frame"}) as (url, _):
        with pytest.raises(RemoteFetchError, match="different manifest"):
            fetch_stereo_run(url, root)
    assert (root / "evidence_manifest.json").read_text() == '{"old": true}'


def test_duplicate_windows_names_and_file_directory_collisions_are_rejected(tmp_path):
    for names in [("photo.PNG", "photo.png"), ("a", "a/b")]:
        with server({name: b"frame" for name in names}) as (url, _):
            with pytest.raises(RemoteFetchError):
                fetch_stereo_run(url, tmp_path / "run")


def test_size_budget_prevents_download(tmp_path):
    with server({"left.png": b"frame"}) as (url, state):
        with pytest.raises(RemoteFetchError, match="size limit"):
            fetch_stereo_run(url, tmp_path / "run", max_bytes=2)
    assert state["requests"] == ["/run/evidence_manifest.json"]


def test_cli_download_returns_success_and_report(tmp_path, capsys):
    from pipe_twin.cli import main

    with server({"report.md": b"field report"}) as (url, _):
        assert main(["fetch-stereo", "--url", url, "--output-dir", str(tmp_path / "run")]) == 0
    assert json.loads((tmp_path / "run.fetch.json").read_text())["status"] == "PASS"
    assert "1/1" in capsys.readouterr().out


def test_existing_link_cannot_redirect_download(tmp_path):
    root = tmp_path / "run"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"untouched")
    try:
        (root / "photo.png").symlink_to(outside)
    except OSError:
        pytest.skip("creating symlinks is unavailable on this host")
    with server({"photo.png": b"frame"}) as (url, _):
        with pytest.raises(RemoteFetchError, match="link"):
            fetch_stereo_run(url, root)
    assert outside.read_bytes() == b"untouched"
