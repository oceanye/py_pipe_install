from __future__ import annotations

import hashlib
import io
import json
import stat
import threading
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipe_twin.analysis_transport import (
    RemoteAnalysisCancelled, RemoteAnalysisClient, RemoteAnalysisError,
    accept_result_bundle, build_request_bundle, build_result_bundle,
    extract_request_bundle,
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def capture(tmp_path):
    root = tmp_path / "camera"
    root.mkdir()
    (root / "model.stl").write_bytes(b"test transport model")
    groups = []
    for index in range(3):
        views = {}
        for role, color in (("left", 30), ("right", 80)):
            name = f"photos/{index}/{role}.png"
            image = np.full((24, 32, 3), color + index, np.uint8)
            ok, encoded = cv2.imencode(".png", image)
            assert ok
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(encoded.tobytes())
            views[role] = {"path": name, "sha256": digest(encoded.tobytes()),
                           "camera_id": role, "expected_width": 32, "expected_height": 24,
                           "captured_at": f"2026-11-02T12:00:0{index}+08:00",
                           "timestamp_source": "HOST_SYSTEM_CLOCK",
                           "orientation_policy": "RAW_PIXELS_NO_EXIF_TRANSFORM"}
        groups.append({"capture_id": f"frame-{index}", "views": views})
    manifest = {"schema_version": "2.0", "dataset_id": "current-capture",
                "model": {"path": "model.stl", "sha256": digest((root / "model.stl").read_bytes())},
                "capture": {"capture_groups": [{**groups[-1], "temporal_frames": groups}]}}
    path = root / "capture.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def rewrite_archive(data, change):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    change(entries)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    return output.getvalue()


def result_for(request, tmp_path, *, structured=False, overlay_name="current_left_overlay.png"):
    manifest, binding = extract_request_bundle(request, tmp_path)
    output = manifest.parent / "_remote_result"
    evidence = output / "evidence"
    (evidence / "section-01").mkdir(parents=True)
    overlay = evidence / overlay_name
    overlay.write_bytes(b"overlay bytes")
    section = evidence / "section-01/section.svg"
    section.write_bytes(b"<svg/>")
    references = [str(overlay), str(section)]
    if structured:
        references = [{"path": name, "sha256": digest(Path(name).read_bytes())} for name in references]
    report = {"inputs": {"manifest": {"path": str(manifest), "sha256": binding["manifest_sha256"]} if structured else binding["manifest_sha256"]},
              "evidence_files": references, "nested": {"overlay": str(overlay)}}
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    return build_result_bundle(manifest.parent, report_path, binding), binding


def test_request_transfers_all_burst_frames_and_only_referenced_files(capture, tmp_path):
    for name in ("old-report.json", "ai-settings.json", ".env", "unused.png"):
        (capture.parent / name).write_text("must stay local", encoding="utf-8")
    bundle = build_request_bundle(capture)
    manifest, binding = extract_request_bundle(bundle, tmp_path / "server")
    assert manifest.read_bytes() == capture.read_bytes()
    assert len(binding["files"]) == 8  # manifest, model, 3 stereo pairs
    assert len(list(manifest.parent.rglob("*.png"))) == 6
    assert not (manifest.parent / ".env").exists()
    assert not any("ai-settings" in name or "old-report" in name for name in binding["files"])


@pytest.mark.parametrize("which", ["model.stl", "photos/0/left.png"])
def test_modified_model_and_burst_photo_cannot_be_uploaded(capture, which):
    (capture.parent / which).write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256"):
        build_request_bundle(capture)


def test_credentials_inside_manifest_are_rejected(capture):
    value = json.loads(capture.read_bytes())
    value["settings"] = {"api_key": "not-a-real-key"}
    capture.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="credentials"):
        build_request_bundle(capture)


@pytest.mark.parametrize("bad_name", ["../escape.txt", "/absolute.txt", "C:/escape.txt", "scene/../escape.txt", "scene/a:secret", "scene/NUL.png", "scene/foo.", "scene\\escape.txt"])
def test_archive_path_attacks_are_rejected_before_writing(capture, tmp_path, bad_name):
    bundle = rewrite_archive(build_request_bundle(capture), lambda files: files.update({bad_name: b"attack"}))
    target = tmp_path / "bad-request"
    with pytest.raises(ValueError):
        extract_request_bundle(bundle, target)
    assert not target.exists()


def test_symlink_zip_member_is_rejected(capture, tmp_path):
    output = io.BytesIO(build_request_bundle(capture))
    with zipfile.ZipFile(output, "a") as archive:
        info = zipfile.ZipInfo("scene/link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "../../outside")
    with pytest.raises(ValueError, match="non-regular"):
        extract_request_bundle(output.getvalue(), tmp_path / "server")


def test_archive_case_duplicates_are_rejected(capture, tmp_path):
    bundle = rewrite_archive(build_request_bundle(capture), lambda files: files.update({"SCENE/MODEL.STL": b"duplicate"}))
    with pytest.raises(ValueError, match="Duplicate"):
        extract_request_bundle(bundle, tmp_path / "server")


def test_archive_size_limit_applies_to_inflated_data(capture, tmp_path, monkeypatch):
    import pipe_twin.analysis_transport as transport
    bundle = build_request_bundle(capture)
    monkeypatch.setattr(transport, "MAX_UNPACKED_BYTES", 128)
    with pytest.raises(ValueError, match="size"):
        extract_request_bundle(bundle, tmp_path / "server")


def test_tampered_archive_rejected_before_any_extraction(capture, tmp_path):
    bundle = rewrite_archive(build_request_bundle(capture), lambda files: files.update({"scene/model.stl": b"tampered"}))
    target = tmp_path / "server"
    with pytest.raises(ValueError, match="SHA-256"):
        extract_request_bundle(bundle, target)
    assert not target.exists()


@pytest.mark.parametrize("structured", [False, True])
def test_result_roundtrip_relocates_evidence_and_preserves_nested_sections(capture, tmp_path, structured):
    result, binding = result_for(build_request_bundle(capture), tmp_path / "server", structured=structured)
    output, evidence = tmp_path / "front/report.json", tmp_path / "front/evidence"
    report = accept_result_bundle(result, capture, output, evidence)
    assert json.loads(output.read_bytes()) == report
    assert report["remote_analysis"] == binding
    paths = [item["path"] if structured else item for item in report["evidence_files"]]
    assert all(Path(name).is_relative_to(evidence) and Path(name).is_file() for name in paths)
    assert Path(paths[1]).parts[-2:] == ("section-01", "section.svg")
    assert report["nested"]["overlay"] == paths[0]
    if structured:
        assert report["inputs"]["manifest"]["path"] == str(capture)


@pytest.mark.parametrize("change", ["manifest", "model", "burst"])
def test_changed_local_capture_rejects_old_result_without_overwriting(capture, tmp_path, change):
    result, _ = result_for(build_request_bundle(capture), tmp_path / "server")
    target = {"manifest": capture, "model": capture.parent / "model.stl", "burst": capture.parent / "photos/0/left.png"}[change]
    target.write_bytes(target.read_bytes() + b" ")
    output = tmp_path / "report.json"
    output.write_bytes(b"previous report")
    with pytest.raises(ValueError):
        accept_result_bundle(result, capture, output, tmp_path / "evidence")
    assert output.read_bytes() == b"previous report"
    assert not (tmp_path / "evidence").exists()


def test_result_cannot_overwrite_an_input(capture, tmp_path):
    result, _ = result_for(build_request_bundle(capture), tmp_path / "server")
    original = capture.read_bytes()
    with pytest.raises(ValueError, match="collision"):
        accept_result_bundle(result, capture, capture, tmp_path / "evidence")
    assert capture.read_bytes() == original


def test_tampered_result_cannot_publish(capture, tmp_path):
    result, _ = result_for(build_request_bundle(capture), tmp_path / "server")
    result = rewrite_archive(result, lambda files: files.update({"evidence/section-01/section.svg": b"tampered"}))
    with pytest.raises(ValueError, match="SHA-256"):
        accept_result_bundle(result, capture, tmp_path / "report.json", tmp_path / "evidence")
    assert not (tmp_path / "report.json").exists()


def test_server_cannot_package_stale_report_or_external_evidence(capture, tmp_path):
    manifest, binding = extract_request_bundle(build_request_bundle(capture), tmp_path / "server")
    report_path = manifest.parent / "report.json"
    outside = tmp_path / "private.txt"
    outside.write_bytes(b"private bytes")
    report = {"inputs": {"manifest": "0" * 64}, "evidence_files": []}
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="Report manifest SHA-256"):
        build_result_bundle(manifest.parent, report_path, binding)
    report["inputs"]["manifest"] = binding["manifest_sha256"]
    report["evidence_files"] = [str(outside)]
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="inside the scene"):
        build_result_bundle(manifest.parent, report_path, binding)
    report["evidence_files"] = []
    report["inputs"]["model"] = "0" * 64
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="Report input SHA-256 mismatch: model"):
        build_result_bundle(manifest.parent, report_path, binding)


def test_client_polls_and_checks_current_request_identity(capture, tmp_path, monkeypatch):
    client = RemoteAnalysisClient("http://127.0.0.1:8766", poll_interval=0.001)
    state, messages = {}, []
    def request(method, route, data=None, *, binary=False):
        if route == "/jobs":
            state["result"], _ = result_for(data, tmp_path / "server")
            return {"job_id": "job-1"}
        if route.endswith("/result"):
            return state["result"]
        state["polls"] = state.get("polls", 0) + 1
        return {"status": "QUEUED" if state["polls"] == 1 else "SUCCEEDED"}
    monkeypatch.setattr(client, "_request", request)
    report = client.analyze(capture, tmp_path / "report.json", tmp_path / "evidence", progress=messages.append)
    assert report["remote_analysis"]["manifest_sha256"] == digest(capture.read_bytes())
    assert state["polls"] == 2
    assert any("排队" in message for message in messages)


def test_client_rejects_previous_job_for_identical_inputs(capture, tmp_path, monkeypatch):
    stale_result, _ = result_for(build_request_bundle(capture), tmp_path / "old-server")
    client = RemoteAnalysisClient("http://127.0.0.1:8766")
    calls = []
    def request(method, route, data=None, *, binary=False):
        calls.append(route)
        if route == "/jobs":
            return {"job_id": "new-job"}
        if route.endswith("/result"):
            return stale_result
        return {"status": "SUCCEEDED"}
    monkeypatch.setattr(client, "_request", request)
    with pytest.raises(RemoteAnalysisError, match="submitted job"):
        client.analyze(capture, tmp_path / "report.json", tmp_path / "evidence")
    assert not (tmp_path / "report.json").exists()
    assert calls[-1].endswith("/cancel")


def test_client_cancel_signals_server_and_does_not_download(capture, tmp_path, monkeypatch):
    client = RemoteAnalysisClient("http://127.0.0.1:8766", poll_interval=0.001)
    cancel, calls = threading.Event(), []
    def request(method, route, data=None, *, binary=False):
        calls.append(route)
        if route.endswith("/cancel"):
            assert data == b""
        if route == "/jobs":
            return {"job_id": "cancel-job"}
        cancel.set()
        return {"status": "RUNNING"}
    monkeypatch.setattr(client, "_request", request)
    with pytest.raises(RemoteAnalysisCancelled):
        client.analyze(capture, tmp_path / "report.json", tmp_path / "evidence", cancel_event=cancel)
    assert calls[-1] == "/jobs/cancel-job/cancel"
    assert not any(route.endswith("/result") for route in calls)
    assert not (tmp_path / "report.json").exists()


def test_client_timeout_requests_cancellation(capture, tmp_path, monkeypatch):
    client = RemoteAnalysisClient("http://127.0.0.1:8766", poll_interval=0.001, job_timeout=0.002)
    calls = []
    def request(method, route, data=None, *, binary=False):
        calls.append(route)
        return {"job_id": "slow-job"} if route == "/jobs" else {"status": "RUNNING"}
    monkeypatch.setattr(client, "_request", request)
    with pytest.raises(RemoteAnalysisError, match="timed out"):
        client.analyze(capture, tmp_path / "report.json", tmp_path / "evidence")
    assert calls[-1].endswith("/cancel")


def _http_fake_analyzer(manifest_path, report_output_path, evidence_dir, *, progress):
    progress("computing in server process")
    evidence = Path(evidence_dir) / "computed.txt"
    evidence.parent.mkdir(parents=True)
    evidence.write_text("remote computed evidence", encoding="utf-8")
    return {"inputs": {"manifest": digest(Path(manifest_path).read_bytes())},
            "evidence_files": [str(evidence)]}


def test_client_server_http_roundtrip(capture, tmp_path):
    from pipe_twin.analysis_server import AnalysisServer
    server = AnalysisServer("127.0.0.1", 0, tmp_path / "jobs", analyzer=_http_fake_analyzer).start()
    try:
        client = RemoteAnalysisClient(server.base_url, poll_interval=0.01, job_timeout=30)
        assert client.health()["service"] == "pipe-twin-analysis"
        output = tmp_path / "frontend/report.json"
        report = client.analyze(capture, output, tmp_path / "frontend/evidence")
        assert output.exists()
        evidence = Path(report["evidence_files"][0])
        assert evidence.read_text(encoding="utf-8") == "remote computed evidence"
        assert evidence.is_relative_to(tmp_path / "frontend")
    finally:
        server.stop()


def _http_waiting_analyzer(manifest_path, report_output_path, evidence_dir, *, progress):
    progress("waiting for remote cancellation")
    time.sleep(60)
    raise AssertionError("Server should have terminated this process")


def test_client_cancellation_reaches_http_server_and_stops_worker(capture, tmp_path):
    from pipe_twin.analysis_server import AnalysisServer
    server = AnalysisServer("127.0.0.1", 0, tmp_path / "jobs", analyzer=_http_waiting_analyzer).start()
    cancel = threading.Event()
    try:
        client = RemoteAnalysisClient(server.base_url, poll_interval=0.01, job_timeout=30)
        def progress(message):
            if "waiting for remote cancellation" in message:
                cancel.set()
        with pytest.raises(RemoteAnalysisCancelled):
            client.analyze(capture, tmp_path / "report.json", tmp_path / "evidence",
                           progress=progress, cancel_event=cancel)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and server.jobs()[0]["status"] != "CANCELLED":
            time.sleep(0.01)
        assert server.jobs()[0]["status"] == "CANCELLED"
        assert not (tmp_path / "report.json").exists()
    finally:
        server.stop()


@pytest.mark.parametrize("payload", [{"service": "office-client", "protocol_version": "1"},
                                    {"service": "pipe-twin-analysis", "protocol_version": "2"}])
def test_health_rejects_wrong_service_or_protocol(monkeypatch, payload):
    client = RemoteAnalysisClient("http://127.0.0.1:8771")
    monkeypatch.setattr(client, "_request", lambda *args: payload)
    with pytest.raises(RemoteAnalysisError, match="compatible analysis server"):
        client.health()


def test_report_write_failure_preserves_existing_report_and_its_evidence(capture, tmp_path, monkeypatch):
    import pipe_twin.analysis_transport as transport
    result, _ = result_for(build_request_bundle(capture), tmp_path / "server")
    output, evidence = tmp_path / "report.json", tmp_path / "evidence"
    old = accept_result_bundle(result, capture, output, evidence)
    original_report = output.read_bytes()
    original_evidence = {name: Path(name).read_bytes() for name in old["evidence_files"]}
    write = transport.atomic_write_bytes
    def fail_report(path, data):
        if path == output:
            raise OSError("simulated full disk")
        return write(path, data)
    monkeypatch.setattr(transport, "atomic_write_bytes", fail_report)
    with pytest.raises(OSError, match="full disk"):
        accept_result_bundle(result, capture, output, evidence)
    assert output.read_bytes() == original_report
    assert all(Path(name).read_bytes() == data for name, data in original_evidence.items())
    assert len(list(evidence.iterdir())) == 1


@pytest.mark.parametrize("interruption", ["cancel", "source_changed"])
def test_interruption_during_evidence_write_prevents_report_publication(capture, tmp_path, monkeypatch, interruption):
    import pipe_twin.analysis_transport as transport
    result, _ = result_for(build_request_bundle(capture), tmp_path / "server")
    output, evidence = tmp_path / "report.json", tmp_path / "evidence"
    output.write_bytes(b"previous report")
    cancel = threading.Event()
    write = transport.atomic_write_bytes
    first = True
    def interrupt_write(path, data):
        nonlocal first
        value = write(path, data)
        if first:
            first = False
            if interruption == "cancel":
                cancel.set()
            else:
                (capture.parent / "photos/0/left.png").write_bytes(b"changed during save")
        return value
    monkeypatch.setattr(transport, "atomic_write_bytes", interrupt_write)
    with pytest.raises(RemoteAnalysisCancelled if interruption == "cancel" else ValueError):
        accept_result_bundle(result, capture, output, evidence, cancel_event=cancel)
    assert output.read_bytes() == b"previous report"
    assert not list(evidence.iterdir())


def test_short_result_directory_retries_collision_without_reusing_old_evidence(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pipe_twin.analysis_transport as transport
    parent = tmp_path / "evidence"
    old = parent / ("r" + "a" * 12)
    old.mkdir(parents=True)
    (old / "keep.txt").write_bytes(b"old evidence")
    identities = iter(["a" * 32, "b" * 32])
    monkeypatch.setattr(transport.uuid, "uuid4", lambda: SimpleNamespace(hex=next(identities)))
    new = transport._reserve_result_directory(parent)
    assert new.name == "r" + "b" * 12
    assert new.is_dir()
    assert (old / "keep.txt").read_bytes() == b"old evidence"


def test_short_result_directory_writes_capture_filename_under_long_prefix(capture, tmp_path, monkeypatch):
    import pipe_twin.artifact_io as artifacts
    overlay_name = "elevation-20261102-115517-4e0034c9_left_overlay.png"
    result, binding = result_for(build_request_bundle(capture), tmp_path / "server",
                                 overlay_name=overlay_name)
    # With r+12hex the real NamedTemporaryFile path is 258 characters; the old
    # request-id + uuid directory would make the same operation exceed 300.
    padding = 178 - len(str(tmp_path.resolve())) - 1
    assert padding > 0, "This test requires a temporary root shorter than 178 characters"
    evidence = tmp_path / ("p" * padding)
    observed = []
    temporary_file = artifacts.tempfile.NamedTemporaryFile
    def record_temporary_file(*args, **kwargs):
        handle = temporary_file(*args, **kwargs)
        observed.append(str(handle.name))
        return handle
    monkeypatch.setattr(artifacts.tempfile, "NamedTemporaryFile", record_temporary_file)
    report = accept_result_bundle(result, capture, tmp_path / "report.json", evidence)
    overlay = Path(report["evidence_files"][0])
    assert overlay.name == overlay_name
    assert overlay.read_bytes() == b"overlay bytes"
    assert len(overlay.parent.name) == 13 and overlay.parent.name.startswith("r")
    temporary = next(name for name in observed if Path(name).name.startswith("." + overlay_name + "."))
    assert len(temporary) == 258
    assert Path(temporary).name.endswith(".tmp")
    assert report["remote_analysis"]["request_id"] == binding["request_id"]


@pytest.mark.parametrize("address", ["ftp://localhost", "http://user:secret@localhost", "http://localhost/a", "http://localhost:99999"])
def test_client_rejects_invalid_server_addresses(address):
    with pytest.raises(ValueError):
        RemoteAnalysisClient(address)
