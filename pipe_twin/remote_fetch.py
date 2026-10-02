"""Download a completed stereo evidence run directly from an HTTP file server.

This is a read-only transfer, not a network camera or a remote command API.
The source must expose an evidence_manifest.json with per-file SHA-256 hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, build_opener

from .pipeline import atomic_write_text

MANIFEST_NAME = "evidence_manifest.json"
MAX_MANIFEST_BYTES = 8 * 1024 * 1024


class RemoteFetchError(ValueError):
    """The remote run could not be downloaded and verified."""


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RemoteFetchError("remote evidence server redirected the request")


def _base_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RemoteFetchError("url must be an HTTP(S) run-directory URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise RemoteFetchError("run URL must not contain credentials, query or fragment")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/") + "/", "", ""))


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name:
        raise RemoteFetchError("invalid evidence path")
    parts = name.split("/")
    if any(part in {"", ".", ".."} or ":" in part or part.endswith((".", " "))
           or PureWindowsPath(part).is_reserved() or re.search(r'[<>"|?*\x00-\x1f]', part)
           for part in parts):
        raise RemoteFetchError(f"invalid evidence path: {name}")
    if PurePosixPath(name).is_absolute():
        raise RemoteFetchError("absolute evidence paths are not supported")
    root = root.resolve()
    target = root
    for part in parts:
        target = target / part
        if target.is_symlink() or (hasattr(target, "is_junction") and target.is_junction()):
            raise RemoteFetchError(f"evidence path crosses a link: {name}")
        # Resolve only components that already exist.  Resolving a path while
        # another worker is creating its parent can transiently produce a
        # false containment failure on Windows; lexical validation above and
        # this component-by-component check retain the link protection.
        if target.exists():
            try:
                resolved = target.resolve(strict=True)
            except OSError as error:
                raise RemoteFetchError(f"could not validate evidence path: {name}") from error
            if not resolved.is_relative_to(root):
                raise RemoteFetchError(f"evidence path leaves output directory: {name}")
    return target


def _entries(payload: Any, root: Path, max_bytes: int) -> list[dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("files"), list):
        raise RemoteFetchError("evidence manifest must contain a files list")
    files = payload["files"]
    if not 1 <= len(files) <= 10000:
        raise RemoteFetchError("evidence manifest must list 1 to 10000 files")
    seen: set[str] = set()
    total = 0
    for item in files:
        if not isinstance(item, dict):
            raise RemoteFetchError("invalid evidence file entry")
        name, size, digest = item.get("path"), item.get("size_bytes"), item.get("sha256")
        _target(root, name)
        if name.casefold().split("/")[0] == MANIFEST_NAME or name.casefold() in seen:
            raise RemoteFetchError(f"duplicate or self-referencing evidence path: {name}")
        seen.add(name.casefold())
        if type(size) is not int or size < 0 or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise RemoteFetchError(f"invalid size or SHA-256: {name}")
        total += size
    # File/directory name collisions are ambiguous on every supported OS.
    for item in files:
        parts = item["path"].casefold().split("/")
        if any("/".join(parts[:i]) in seen for i in range(1, len(parts))):
            raise RemoteFetchError("evidence manifest has a file/directory path collision")
    if total > max_bytes:
        raise RemoteFetchError("evidence run exceeds the configured download size limit")
    if payload.get("file_count", len(files)) != len(files) or payload.get("total_size_bytes", total) != total:
        raise RemoteFetchError("manifest file count or total size is inconsistent")
    return files


def fetch_stereo_run(
    url: str,
    output_dir: str | Path,
    *,
    timeout_s: float = 45.0,
    workers: int = 4,
    retries: int = 2,
    max_bytes: int = 2 * 1024**3,
    progress: Callable[[int, int, str], None] | None = None,
) -> Path:
    """Download and verify one run; return the sibling transfer-report path.

    Existing matching files are reused. A different existing manifest is
    refused so unrelated runs cannot be mixed. HTTP proxies are bypassed for
    direct VPN access. Completed files survive a failed transfer for retries.
    """
    base = _base_url(url)
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise RemoteFetchError("timeout_s must be positive and finite")
    if type(workers) is not int or not 1 <= workers <= 8 or type(retries) is not int or not 0 <= retries <= 5:
        raise RemoteFetchError("workers must be 1..8 and retries must be 0..5")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise RemoteFetchError("max_bytes must be a positive integer")
    root = Path(output_dir).absolute()
    if any(p.is_symlink() or (hasattr(p, "is_junction") and p.is_junction()) for p in [root, *root.parents]):
        raise RemoteFetchError("output directory must not cross a link")
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root.with_name(root.name + ".fetch.json")
    if report_path.is_symlink() or (hasattr(report_path, "is_junction") and report_path.is_junction()):
        raise RemoteFetchError("transfer report must not be a link")
    report: dict[str, Any] = {
        "status": "RUNNING", "source_url": base, "output_directory": str(root),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "transfer_kind": "HTTP_EXISTING_FILES_ONLY", "system_proxy_used": False,
        "files": [], "errors": [], "verified_bytes": 0,
    }

    def save_report() -> None:
        atomic_write_text(report_path, json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    def read_manifest() -> bytes:
        with build_opener(ProxyHandler({}), _NoRedirect()).open(base + MANIFEST_NAME, timeout=timeout_s) as response:
            data = response.read(MAX_MANIFEST_BYTES + 1)
        if len(data) > MAX_MANIFEST_BYTES:
            raise RemoteFetchError("evidence manifest is too large")
        return data

    def transfer(item: dict[str, Any]) -> dict[str, Any]:
        dest = _target(root, item["path"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        expected = item["sha256"].lower()
        if dest.is_file() and dest.stat().st_size == item["size_bytes"] and _hash(dest) == expected:
            return {"path": item["path"], "status": "REUSED", "size_bytes": item["size_bytes"], "sha256": expected}
        for attempt in range(retries + 1):
            temp_path: Path | None = None
            try:
                opener = build_opener(ProxyHandler({}), _NoRedirect())
                digest, size = hashlib.sha256(), 0
                with opener.open(base + quote(item["path"], safe="/"), timeout=timeout_s) as response:
                    with tempfile.NamedTemporaryFile(dir=dest.parent, prefix=".fetch-", delete=False) as target:
                        temp_path = Path(target.name)
                        while True:
                            chunk = response.read(256 * 1024)
                            if not chunk:
                                break
                            size += len(chunk)
                            if size > item["size_bytes"]:
                                raise RemoteFetchError("remote file exceeds declared size")
                            digest.update(chunk)
                            target.write(chunk)
                if size != item["size_bytes"] or digest.hexdigest() != expected:
                    raise RemoteFetchError("downloaded size or SHA-256 does not match manifest")
                _target(root, item["path"])
                os.replace(temp_path, dest)
                return {"path": item["path"], "status": "DOWNLOADED", "size_bytes": size, "sha256": expected}
            except Exception as error:
                if attempt == retries:
                    return {"path": item["path"], "status": "FAILED", "error": str(error)}
                time.sleep(0.25)
            finally:
                if temp_path is not None and temp_path.exists():
                    temp_path.unlink()
        raise AssertionError("unreachable")

    save_report()
    try:
        original = read_manifest()
        files = _entries(json.loads(original.decode("utf-8-sig")), root, max_bytes)
        local_manifest = _target(root, MANIFEST_NAME)
        if local_manifest.exists() and local_manifest.read_bytes() != original:
            raise RemoteFetchError("output contains a different manifest; choose a new output directory")
        local_manifest.write_bytes(original)
        report.update(manifest_sha256=hashlib.sha256(original).hexdigest(), expected_file_count=len(files),
                      manifest_self_hash_note="Source manifest is saved and compared before/after transfer; it does not hash itself.")
        # Make the directory tree before workers start.  Concurrent mkdir and
        # resolve calls are otherwise racy on Windows, where a field run can
        # report spurious "evidence path leaves output directory" failures.
        for item in files:
            _target(root, item["path"]).parent.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(transfer, item): item for item in files}
            for future in as_completed(futures):
                item = futures[future]
                try:
                    record = future.result()
                except Exception as error:
                    record = {"path": item["path"], "status": "FAILED", "error": str(error)}
                report["files"].append(record)
                if record["status"] == "FAILED":
                    report["errors"].append(record)
                else:
                    report["verified_bytes"] += record["size_bytes"]
                save_report()
                if progress is not None:
                    progress(len(report["files"]), len(files), record["path"])
        report["manifest_unchanged"] = read_manifest() == original
        if not report["manifest_unchanged"]:
            raise RemoteFetchError("source manifest changed during download; re-fetch to a new directory")
        if report["errors"]:
            raise RemoteFetchError(f"{len(report['errors'])} evidence files failed verification")
        report["status"] = "PASS"
    except (Exception, KeyboardInterrupt) as error:
        report["status"] = "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED"
        report["error"] = str(error)
        raise
    finally:
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        save_report()
    return report_path
