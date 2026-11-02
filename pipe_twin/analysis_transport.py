"""Bounded, hash-bound transfer of analysis inputs and results.

Only files explicitly referenced by the capture manifest or result are sent.
Importing this module does not start a service or contact a remote host.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import time
import uuid
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .artifact_io import atomic_write_bytes, ensure_paths_distinct

MAX_BUNDLE_BYTES = 512 * 1024 * 1024
MAX_UNPACKED_BYTES = 1024 * 1024 * 1024
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_FILES = 4096
_REQUEST_SCHEMA = "pipe-twin-analysis-request-v1"
_RESULT_SCHEMA = "pipe-twin-analysis-result-v1"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_MODEL_SUFFIXES = {".stl", ".dxf", ".3dm", ".3mf", ".obj", ".ply"}
_SECRET_KEYS = {"api_key", "apikey", "api_secret", "secret_key", "access_token", "password", "authorization"}


class RemoteAnalysisError(RuntimeError):
    """A remote job or its transport failed; no result should be displayed."""


class RemoteAnalysisCancelled(RemoteAnalysisError):
    """The current remote analysis was cancelled."""


def _check_cancel(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RemoteAnalysisCancelled("Remote analysis cancelled")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")


def _object(data: bytes, label: str) -> dict:
    try:
        value = json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f"Invalid {label} JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _relative(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Archive paths must be portable relative paths")
    path, windows = PurePosixPath(value), PureWindowsPath(value)
    if path.is_absolute() or windows.drive or path.as_posix() != value:
        raise ValueError("Archive paths must be portable relative paths")
    for part in value.split("/"):
        if part in {"", ".", ".."} or any(c in part for c in ':<>"|?*') or part.endswith((".", " ")):
            raise ValueError("Unsafe relative path in analysis bundle")
        if any(ord(c) < 32 for c in part) or PureWindowsPath(part).is_reserved():
            raise ValueError("Unsafe Windows filename in analysis bundle")
    return value


def _inside(root: Path, relative: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(_relative(relative)).parts).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError("Analysis path escapes its root")
    return candidate


def _read_file(path: Path) -> bytes:
    if not path.is_file() or path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"Missing or oversized analysis file: {path.name}")
    with path.open("rb") as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("Analysis file exceeds size limit")
    return data


def _no_credentials(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in _SECRET_KEYS:
                raise ValueError("Capture manifests must not contain credentials")
            _no_credentials(item)
    elif isinstance(value, list):
        for item in value:
            _no_credentials(item)


def _asset_records(manifest: dict) -> list[tuple[dict, bool]]:
    """Explicit schema traversal prevents unrelated settings becoming file reads."""
    _no_credentials(manifest)
    model = manifest.get("model")
    if not isinstance(model, dict) or not model.get("path"):
        raise ValueError("Remote analysis requires a manifest-bound model")
    records = [(model, False)]
    elevation = manifest.get("elevation")
    if isinstance(elevation, dict) and elevation.get("path"):
        records.append((elevation, False))
    capture = manifest.get("capture")
    groups = capture.get("capture_groups") if isinstance(capture, dict) else None
    if not isinstance(groups, list) or not groups:
        raise ValueError("Remote analysis requires stereo capture groups")
    for group in groups:
        if not isinstance(group, dict):
            raise ValueError("Capture group must be an object")
        frames = group.get("temporal_frames", [])
        if not isinstance(frames, list):
            raise ValueError("Temporal frames must be a list")
        for frame in [group, *frames]:
            views = frame.get("views") if isinstance(frame, dict) else None
            if not isinstance(views, dict) or any(not isinstance(views.get(role), dict) for role in ("left", "right")):
                raise ValueError("Every capture and burst frame needs left and right views")
            records.extend((views[role], True) for role in ("left", "right"))
    return records


def _expected_files(manifest_data: bytes) -> tuple[dict, dict[str, str], list[dict]]:
    from .photo_capture import _validate_view
    manifest = _object(manifest_data, "manifest")
    hashes = {"manifest.json": _sha(manifest_data)}
    photos = []
    for record, photo in _asset_records(manifest):
        name = _relative(record.get("path"))
        digest = record.get("sha256")
        if not isinstance(digest, str) or not _HASH.fullmatch(digest.lower()):
            raise ValueError("Every model and photo requires a SHA-256 binding")
        suffix = PurePosixPath(name).suffix.lower()
        if suffix not in ({".png", ".jpg", ".jpeg"} if photo else _MODEL_SUFFIXES):
            raise ValueError("Unsupported model or photo file type")
        if photo:
            _validate_view(record, "view")
            photos.append(record)
        digest = digest.lower()
        if name in hashes and hashes[name] != digest:
            raise ValueError("Conflicting SHA-256 bindings for the same file")
        hashes[name] = digest
    if len({name.casefold() for name in hashes}) != len(hashes):
        raise ValueError("Case-colliding analysis input filenames")
    if len(hashes) + 1 > MAX_FILES:
        raise ValueError("Analysis bundle exceeds file count limit")
    return manifest, hashes, photos


def _zip(files: dict[str, bytes]) -> bytes:
    if len(files) > MAX_FILES or sum(map(len, files.values())) > MAX_UNPACKED_BYTES:
        raise ValueError("Analysis bundle exceeds size or file count limit")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for name, data in files.items():
            _relative(name)
            if len(data) > MAX_FILE_BYTES:
                raise ValueError("Analysis file exceeds size limit")
            archive.writestr(name, data)
    data = buffer.getvalue()
    if len(data) > MAX_BUNDLE_BYTES:
        raise ValueError("Analysis bundle exceeds upload size limit")
    return data


def _unzip(data: bytes) -> dict[str, bytes]:
    if len(data) > MAX_BUNDLE_BYTES:
        raise ValueError("Analysis bundle exceeds upload size limit")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            infos = archive.infolist()
            if len(infos) > MAX_FILES or sum(i.file_size for i in infos) > MAX_UNPACKED_BYTES:
                raise ValueError("Analysis bundle exceeds size or file count limit")
            names = set()
            for info in infos:
                name = _relative(info.filename)
                mode = info.external_attr >> 16
                if (name.casefold() in names or info.is_dir() or info.flag_bits & 1
                        or stat.S_IFMT(mode) not in {0, stat.S_IFREG}
                        or info.file_size > MAX_FILE_BYTES):
                    raise ValueError("Duplicate, non-regular or oversized archive member")
                names.add(name.casefold())
            return {info.filename: archive.read(info) for info in infos}
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError) as exc:
        raise ValueError("Invalid analysis ZIP bundle") from exc


def _binding(value: Any) -> dict:
    if not isinstance(value, dict) or value.get("schema_version") != _REQUEST_SCHEMA:
        raise ValueError("Unsupported analysis request binding")
    if not isinstance(value.get("request_id"), str) or not re.fullmatch(r"[0-9a-f]{32}", value["request_id"]):
        raise ValueError("Invalid request identity")
    hashes = value.get("files")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Missing input file bindings")
    for name, digest in hashes.items():
        _relative(name)
        if not isinstance(digest, str) or not _HASH.fullmatch(digest):
            raise ValueError("Invalid input SHA-256 binding")
    if "manifest.json" not in hashes or value.get("manifest_sha256") != hashes["manifest.json"]:
        raise ValueError("Manifest SHA-256 binding mismatch")
    return value


def _snapshot(manifest_path: str | Path, *, decode_photos: bool = False,
              include_data: bool = True, cancel_event=None) -> tuple[dict, dict[str, bytes]]:
    path = Path(manifest_path).resolve()
    raw = _read_file(path)
    _manifest, hashes, photos = _expected_files(raw)
    files = {}
    total = 0
    for name in hashes:
        _check_cancel(cancel_event)
        source = path if name == "manifest.json" else _inside(path.parent, name)
        total += source.stat().st_size
        if total > MAX_UNPACKED_BYTES:
            raise ValueError("Analysis bundle exceeds unpacked size limit")
        data = raw if name == "manifest.json" else _read_file(source)
        if _sha(data) != hashes[name]:
            raise ValueError("Input SHA-256 mismatch; capture files have changed")
        if include_data or name == "manifest.json":
            files[name] = data
    if decode_photos:
        from .photo_capture import load_photo_snapshot
        for view in photos:
            load_photo_snapshot(path, view)
    if path.read_bytes() != raw:
        raise ValueError("Manifest changed during upload preparation")
    return hashes, files


def build_request_bundle(manifest_path: str | Path) -> bytes:
    """Snapshot the model and all manifest-referenced capture/burst photos."""
    hashes, files = _snapshot(manifest_path, decode_photos=True)
    binding = {"schema_version": _REQUEST_SCHEMA, "request_id": uuid.uuid4().hex,
               "manifest_sha256": hashes["manifest.json"], "files": hashes}
    return _zip({"request.json": _json_bytes(binding), **{f"scene/{name}": data for name, data in files.items()}})


def extract_request_bundle(data: bytes, destination: str | Path) -> tuple[Path, dict]:
    """Validate the complete archive before writing into a fresh job directory."""
    files = _unzip(data)
    binding = _binding(_object(files.get("request.json", b""), "request"))
    expected = {"request.json", *(f"scene/{name}" for name in binding["files"])}
    if set(files) != expected:
        raise ValueError("Unexpected or missing request files")
    for name, digest in binding["files"].items():
        if _sha(files[f"scene/{name}"]) != digest:
            raise ValueError("Request file SHA-256 mismatch")
    _, manifest_hashes, _ = _expected_files(files["scene/manifest.json"])
    if manifest_hashes != binding["files"]:
        raise ValueError("Request files do not match the manifest references")
    root = Path(destination).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("Request extraction requires an empty directory")
    for name in sorted(expected - {"request.json"}):
        target = _inside(root, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(files[name])
    return root / "scene/manifest.json", binding


def _report_binding(report: dict, binding: dict, manifest: dict | None = None) -> None:
    inputs = report.get("inputs")
    manifest_hash = inputs.get("manifest") if isinstance(inputs, dict) else None
    if isinstance(manifest_hash, dict):
        manifest_hash = manifest_hash.get("sha256")
    if manifest_hash != binding["manifest_sha256"]:
        raise ValueError("Report manifest SHA-256 does not match this request")
    if manifest is None:
        return
    if report.get("dataset_id", manifest.get("dataset_id")) != manifest.get("dataset_id"):
        raise ValueError("Report dataset does not match this request")
    expected = {"model": manifest["model"]["sha256"]}
    for group in manifest["capture"]["capture_groups"]:
        for role in ("left", "right"):
            expected[f"{group.get('capture_id')}_{role}"] = group["views"][role]["sha256"]
        for index, frame in enumerate(group.get("temporal_frames", [])):
            for role in ("left", "right"):
                expected[f"{group.get('capture_id')}_burst_{index}_{role}"] = frame["views"][role]["sha256"]
    for name, digest in expected.items():
        if name not in inputs:
            continue
        actual = inputs[name]
        if isinstance(actual, dict):
            actual = actual.get("actual_sha256", actual.get("sha256"))
        if actual != digest:
            raise ValueError(f"Report input SHA-256 mismatch: {name}")


def _evidence_references(report: dict) -> list[str]:
    values = report.get("evidence_files", [])
    if not isinstance(values, list):
        raise ValueError("Report evidence_files must be a list")
    result = []
    for value in values:
        name = value.get("path") if isinstance(value, dict) else value
        if not isinstance(name, str) or not name:
            raise ValueError("Invalid evidence file reference")
        result.append(name)
    return result


def _replace_strings(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, str):
        return replacements.get(value, value)
    if isinstance(value, dict):
        return {key: _replace_strings(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace_strings(item, replacements) for item in value]
    return value


def build_result_bundle(scene_root: str | Path, report_path: str | Path, binding: dict) -> bytes:
    """Package only the report's evidence, preserving nested section artifacts."""
    binding = _binding(binding)
    root, path = Path(scene_root).resolve(), Path(report_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Report must be inside the extracted scene")
    for name, digest in binding["files"].items():
        if _sha(_read_file(_inside(root, name))) != digest:
            raise ValueError("Server input changed during analysis")
    report = _object(_read_file(path), "report")
    manifest = _object(_read_file(root / "manifest.json"), "manifest")
    _report_binding(report, binding, manifest)
    references = _evidence_references(report)
    if len(references) + 2 > MAX_FILES:
        raise ValueError("Analysis result exceeds file count limit")
    evidence_paths = []
    for reference in references:
        evidence = Path(reference)
        if not evidence.is_absolute():
            evidence = root / _relative(reference)
        evidence = evidence.resolve()
        if not evidence.is_relative_to(root) or evidence == path or evidence.relative_to(root).as_posix() in binding["files"]:
            raise ValueError("Evidence must be a generated file inside the scene")
        evidence_paths.append(evidence)
    common = Path(os.path.commonpath([str(p.parent) for p in evidence_paths])) if evidence_paths else root
    files, replacements = {}, {}
    total = 0
    for reference, evidence in zip(references, evidence_paths):
        name = "evidence/" + evidence.relative_to(common).as_posix()
        if name not in files:
            total += evidence.stat().st_size
            if total > MAX_UNPACKED_BYTES:
                raise ValueError("Analysis result exceeds unpacked size limit")
            files[name] = _read_file(evidence)
        replacements[reference] = name
    report = _replace_strings(report, replacements)
    for record in report.get("evidence_files", []):
        if isinstance(record, dict) and "sha256" in record:
            if record["sha256"] != _sha(files[record["path"]]):
                raise ValueError("Report evidence SHA-256 mismatch")
    files["report.json"] = _json_bytes(report)
    metadata = {"schema_version": _RESULT_SCHEMA, "binding": binding,
                "files": {name: _sha(data) for name, data in files.items()}}
    return _zip({"result.json": _json_bytes(metadata), **files})


def _read_result(data: bytes) -> tuple[dict, dict, dict[str, bytes]]:
    files = _unzip(data)
    metadata = _object(files.get("result.json", b""), "result")
    if metadata.get("schema_version") != _RESULT_SCHEMA:
        raise ValueError("Unsupported analysis result format")
    binding = _binding(metadata.get("binding"))
    hashes = metadata.get("files")
    if not isinstance(hashes, dict) or set(files) != {"result.json", *hashes} or "report.json" not in hashes:
        raise ValueError("Unexpected or missing result files")
    for name, digest in hashes.items():
        if name != "report.json" and not name.startswith("evidence/"):
            raise ValueError("Unexpected result archive file")
        if _sha(files[name]) != digest:
            raise ValueError("Result SHA-256 mismatch")
    report = _object(files["report.json"], "report")
    _report_binding(report, binding)
    references = _evidence_references(report)
    if set(references) != set(hashes) - {"report.json"}:
        raise ValueError("Result evidence does not match report references")
    for record in report.get("evidence_files", []):
        if isinstance(record, dict) and "sha256" in record and record["sha256"] != hashes[record["path"]]:
            raise ValueError("Report evidence SHA-256 mismatch")
    return binding, report, files


def _reserve_result_directory(parent: Path) -> Path:
    """Reserve a short output name atomically, including on Windows."""
    parent.mkdir(parents=True, exist_ok=True)
    for _ in range(32):
        candidate = parent / ("r" + uuid.uuid4().hex[:12])
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        return candidate
    raise FileExistsError("Cannot reserve a unique result evidence directory")


def accept_result_bundle(data: bytes, manifest_path: str | Path,
                         report_output_path: str | Path, evidence_dir: str | Path,
                         *, cancel_event=None) -> dict:
    """Check the current local inputs before publishing any remote output."""
    _check_cancel(cancel_event)
    binding, report, files = _read_result(data)
    hashes, source_files = _snapshot(manifest_path, include_data=False, cancel_event=cancel_event)
    if hashes != binding["files"]:
        raise ValueError("Remote result belongs to a different or changed capture")
    _report_binding(report, binding, _object(source_files["manifest.json"], "manifest"))
    path = Path(manifest_path).resolve()
    output, evidence_parent = Path(report_output_path).resolve(), Path(evidence_dir).resolve()
    # A fresh directory keeps existing reports and their evidence consistent if
    # a write, cancellation or source recheck interrupts this publication.
    # Keep the full request identity in the report, not in filesystem names:
    # real capture filenames plus atomic-write suffixes approach Windows' limit.
    _check_cancel(cancel_event)
    evidence_root = _reserve_result_directory(evidence_parent)
    try:
        targets = {name: _inside(evidence_root, name.removeprefix("evidence/")) for name in _evidence_references(report)}
        ensure_paths_distinct(manifest=path, **{f"input_{i}": _inside(path.parent, name) for i, name in enumerate(hashes) if name != "manifest.json"},
                              report=output, **{f"evidence_{i}": target for i, target in enumerate(targets.values())})
        for target in [output, *targets.values()]:
            if target.exists() and not target.is_file():
                raise ValueError("Result destination is not a regular file")
        if any(output in target.parents or target in output.parents for target in targets.values()):
            raise ValueError("Report and evidence destinations must not contain one another")
        report = _replace_strings(report, {name: str(target) for name, target in targets.items()})
        inputs = report.get("inputs", {})
        if isinstance(inputs.get("manifest"), dict):
            inputs["manifest"]["path"] = str(path)
        report["remote_analysis"] = copy.deepcopy(binding)
        for name, target in targets.items():
            _check_cancel(cancel_event)
            atomic_write_bytes(target, files[name])
        final_hashes, _ = _snapshot(manifest_path, include_data=False, cancel_event=cancel_event)
        if final_hashes != hashes:
            raise ValueError("Capture changed before remote result publication")
        _check_cancel(cancel_event)
        atomic_write_bytes(output, _json_bytes(report))
    except BaseException:
        # Remove only the uniquely named directory created by this acceptance.
        if (evidence_root.parent == evidence_parent and not evidence_root.is_symlink()
                and evidence_root.resolve().is_relative_to(evidence_parent)):
            shutil.rmtree(evidence_root, ignore_errors=True)
        raise
    return report


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class RemoteAnalysisClient:
    """Synchronous worker-thread client for a bounded HTTP analysis job."""

    def __init__(self, base_url: str, *, timeout: float = 30.0,
                 poll_interval: float = 0.5, job_timeout: float = 3600.0):
        parsed = urlsplit(base_url)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username
                or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
            raise ValueError("Server address must be http(s)://host:port")
        # Accessing .port checks malformed or out-of-range ports immediately.
        parsed.port
        for value in (timeout, poll_interval, job_timeout):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Remote timeouts must be positive finite numbers")
        self.base_url = base_url.rstrip("/")
        self.timeout, self.poll_interval, self.job_timeout = timeout, poll_interval, job_timeout
        self._opener = build_opener(ProxyHandler({}), _NoRedirect())

    def _request(self, method: str, route: str, data: bytes | None = None, *, binary: bool = False) -> Any:
        headers = {"Accept": "application/zip" if binary else "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/zip" if route == "/jobs" else "application/json"
        request = Request(self.base_url + route, data=data, headers=headers, method=method)
        limit = MAX_BUNDLE_BYTES if binary else 1024 * 1024
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                payload = response.read(limit + 1)
                if len(payload) > limit:
                    raise RemoteAnalysisError("Remote response exceeds size limit")
        except HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            raise RemoteAnalysisError(f"Remote server HTTP {exc.code}: {detail}") from exc
        except (URLError, OSError) as exc:
            raise RemoteAnalysisError(f"Cannot reach analysis server: {exc}") from exc
        if binary:
            return payload
        try:
            return _object(payload, "server response")
        except ValueError as exc:
            raise RemoteAnalysisError(str(exc)) from exc

    def health(self) -> dict:
        value = self._request("GET", "/health")
        if value.get("service") != "pipe-twin-analysis" or value.get("protocol_version") != "1":
            raise RemoteAnalysisError("This address is not a compatible analysis server")
        return value

    def analyze(self, manifest_path: str | Path, report_output_path: str | Path,
                evidence_dir: str | Path, progress: Callable[[str], None] | None = None,
                cancel_event=None) -> dict:
        def cancelled() -> bool:
            return cancel_event is not None and cancel_event.is_set()

        if cancelled():
            raise RemoteAnalysisCancelled("Remote analysis cancelled")
        if progress:
            progress("正在校验并打包模型和本次拍摄照片……")
        bundle = build_request_bundle(manifest_path)
        with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
            sent_binding = _object(archive.read("request.json"), "request")
        if cancelled():
            raise RemoteAnalysisCancelled("Remote analysis cancelled")
        if progress:
            progress("正在上传到计算服务器……")
        created = self._request("POST", "/jobs", bundle)
        job_id = created.get("job_id")
        if not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
            raise RemoteAnalysisError("Server returned an invalid job identity")
        route = "/jobs/" + job_id
        deadline = time.monotonic() + self.job_timeout
        last_progress = None
        finished = False
        try:
            while True:
                if cancelled():
                    raise RemoteAnalysisCancelled("Remote analysis cancelled")
                if time.monotonic() >= deadline:
                    raise RemoteAnalysisError("Remote analysis timed out")
                status = self._request("GET", route)
                state = status.get("status")
                message = status.get("progress") or {"QUEUED": "服务器任务排队中……", "RUNNING": "服务器正在计算……"}.get(state, state)
                if progress and message != last_progress:
                    progress(str(message))
                    last_progress = message
                if state == "SUCCEEDED":
                    if cancelled():
                        raise RemoteAnalysisCancelled("Remote analysis cancelled")
                    result = self._request("GET", route + "/result", binary=True)
                    binding = _read_result(result)[0]
                    if binding != sent_binding:
                        raise RemoteAnalysisError("Remote result does not belong to the submitted job")
                    if cancelled():
                        raise RemoteAnalysisCancelled("Remote analysis cancelled")
                    report = accept_result_bundle(result, manifest_path, report_output_path, evidence_dir,
                                                  cancel_event=cancel_event)
                    finished = True
                    return report
                if state == "CANCELLED":
                    finished = True
                    raise RemoteAnalysisCancelled("Remote analysis cancelled by server")
                if state == "FAILED":
                    finished = True
                    raise RemoteAnalysisError(str(status.get("error") or "Remote analysis failed"))
                if state not in {"QUEUED", "RUNNING"}:
                    raise RemoteAnalysisError("Server returned an invalid job status")
                wait = min(self.poll_interval, max(0.0, deadline - time.monotonic()))
                if cancel_event is not None:
                    cancel_event.wait(wait)
                else:
                    time.sleep(wait)
        finally:
            if not finished:
                try:
                    self._request("POST", route + "/cancel", b"")
                except RemoteAnalysisError:
                    pass
