"""Bounded remote control for the local stereo camera.

The file server on port 8765 deliberately remains read-only.  This module
provides the separate control plane used by a home workstation to request a
finite capture on the office computer.  It never accepts a remote output
path or an arbitrary command; every job is written below the configured
output root and uses the existing :func:`capture_stereo_pairs` safeguards.

The integrated office client binds only to loopback or a Tailscale address.
That mode can use Tailscale as the trust boundary without a second token
exchange.  The lower-level ``serve-capture`` entry point still requires a
bearer token, and the integrated client can opt back into bearer auth with
``--require-token``.
"""

from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import math
import re
import secrets
import subprocess
import threading
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
from uuid import uuid4

from .camera_lock import CameraBusyError
from .camera_exposure import MAX_EXPOSURE_MS, MIN_EXPOSURE_MS
from .logging_config import get_logger, log_event
from .pipeline import atomic_write_text
from .remote_capture import capture_stereo_pairs
from .stereo_camera import (
    LAYOUT_SEPARATE,
    LAYOUT_SIDE_BY_SIDE_LR,
    LAYOUT_SIDE_BY_SIDE_RL,
)

_LOGGER = get_logger("capture_agent")

API_PREFIX = "/v1"
MAX_REQUEST_BYTES = 16 * 1024
JOB_ID_RE = re.compile(r"^remote-[0-9]{8}-[0-9]{6}-[0-9a-f]{8}$")
LAYOUTS = (LAYOUT_SIDE_BY_SIDE_LR, LAYOUT_SIDE_BY_SIDE_RL, LAYOUT_SEPARATE)
TERMINAL_STATES = frozenset(("COMPLETED", "FAILED"))
_JOB_FILE = "job.json"
_EVIDENCE_MANIFEST = "evidence_manifest.json"


@lru_cache(maxsize=1)
def _runtime_info() -> dict[str, Any]:
    """Snapshot deployment identity at service startup, not at each request."""
    import cv2
    root = Path(__file__).resolve().parents[1]
    identity: dict[str, Any] = {"capture_contract": "remote-auto-native-dshow-v1",
        "default_exposure_ms": None, "opencv_version": cv2.__version__,
        "camera_backends": {str(b): bool(cv2.videoio_registry.hasBackend(b)) for b in (cv2.CAP_DSHOW, cv2.CAP_MSMF)},
        "code_sha": None, "working_tree_dirty": None}
    try:
        options = dict(cwd=root, capture_output=True, text=True, encoding="utf-8", timeout=3,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        head = subprocess.run(["git", "rev-parse", "HEAD"], **options)
        changes = subprocess.run(["git", "status", "--porcelain", "--untracked-files=normal", "--", "pipe_twin"], **options)
        if head.returncode == 0 and changes.returncode == 0:
            identity.update(code_sha=head.stdout.strip(), working_tree_dirty=bool(changes.stdout.strip()))
    except (OSError, subprocess.SubprocessError):
        pass
    return identity
_TAILNET = ipaddress.ip_network("100.64.0.0/10")


class CaptureAgentError(ValueError):
    """A client supplied an invalid request or service configuration."""


class CaptureAgentBusy(CaptureAgentError):
    """The camera is already reserved by another capture job."""


def _trusted_control_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return False
    return address.version == 4 and (address.is_loopback or address in _TAILNET)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise CaptureAgentError("capture agent must not redirect requests")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _is_link(path: Path) -> bool:
    junction = getattr(path, "is_junction", None)
    return path.is_symlink() or bool(junction and junction())


def _safe_output_root(path: str | Path) -> Path:
    root = Path(path).expanduser().absolute()
    if any(_is_link(part) for part in (root, *root.parents)):
        raise CaptureAgentError("output root must not cross a symbolic link or junction")
    root.mkdir(parents=True, exist_ok=True)
    if _is_link(root):
        raise CaptureAgentError("output root became a symbolic link or junction")
    return root.resolve()


def _finite_number(value: Any, name: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CaptureAgentError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise CaptureAgentError(f"{name} must be between {minimum:g} and {maximum:g}")
    return result


@dataclass(frozen=True)
class CaptureAgentConfig:
    """Fixed camera and safety limits for one office capture agent."""

    output_root: Path
    left_index: int = 0
    right_index: int | None = None
    layout: str = LAYOUT_SIDE_BY_SIDE_LR
    eye_width: int = 1920
    eye_height: int = 1080
    backend: int | None = None
    default_count: int = 1
    max_count: int = 30
    default_interval_s: float = 0.0
    max_interval_s: float = 60.0
    default_warmup_s: float = 20.0
    max_warmup_s: float = 120.0
    max_duration_s: float = 300.0
    max_history: int = 100
    file_base_url: str | None = None

    def __post_init__(self) -> None:
        root = _safe_output_root(self.output_root)
        object.__setattr__(self, "output_root", root)
        if type(self.left_index) is not int or self.left_index < 0:
            raise CaptureAgentError("left_index must be a non-negative integer")
        if self.right_index is not None and (
            type(self.right_index) is not int or self.right_index < 0
        ):
            raise CaptureAgentError("right_index must be a non-negative integer or None")
        if self.layout not in LAYOUTS:
            raise CaptureAgentError(f"unsupported camera layout: {self.layout}")
        for name, value in (("eye_width", self.eye_width), ("eye_height", self.eye_height)):
            if type(value) is not int or not 16 <= value <= 4096:
                raise CaptureAgentError(f"{name} must be an integer between 16 and 4096")
        if self.layout == LAYOUT_SEPARATE and self.right_index is None:
            raise CaptureAgentError("separate_devices layout requires right_index")
        if self.layout == LAYOUT_SEPARATE and self.right_index == self.left_index:
            raise CaptureAgentError("left and right cameras must use different indices")
        if self.backend is not None and type(self.backend) is not int:
            raise CaptureAgentError("backend must be an integer or None")
        if type(self.max_count) is not int or not 1 <= self.max_count <= 120:
            raise CaptureAgentError("max_count must be between 1 and 120")
        if type(self.default_count) is not int or not 1 <= self.default_count <= self.max_count:
            raise CaptureAgentError("default_count must be between 1 and max_count")
        max_interval_s = _finite_number(
            self.max_interval_s,
            "max_interval_s",
            minimum=0.0,
            maximum=3600.0,
        )
        _finite_number(
            self.default_interval_s,
            "default_interval_s",
            minimum=0.0,
            maximum=max_interval_s,
        )
        max_warmup_s = _finite_number(
            self.max_warmup_s,
            "max_warmup_s",
            minimum=0.0,
            maximum=3600.0,
        )
        _finite_number(
            self.default_warmup_s,
            "default_warmup_s",
            minimum=0.0,
            maximum=max_warmup_s,
        )
        _finite_number(self.max_duration_s, "max_duration_s", minimum=0.1, maximum=3600.0)
        if type(self.max_history) is not int or not 1 <= self.max_history <= 1000:
            raise CaptureAgentError("max_history must be between 1 and 1000")
        if self.file_base_url is not None:
            object.__setattr__(self, "file_base_url", _base_url(self.file_base_url))


def _base_url(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CaptureAgentError("base URL must be a non-empty HTTP(S) URL")
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise CaptureAgentError("base URL must be an HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise CaptureAgentError("base URL must not contain credentials, query or fragment")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def _read_token_file(path: str | Path) -> str:
    token_path = Path(path).expanduser()
    if _is_link(token_path):
        raise CaptureAgentError("token file must not be a symbolic link or junction")
    try:
        token = token_path.read_text(encoding="utf-8-sig").strip()
    except OSError as error:
        raise CaptureAgentError(f"cannot read token file: {token_path}") from error
    _validate_token(token)
    return token


def _validate_token(token: str) -> None:
    if (
        not isinstance(token, str)
        or not 1 <= len(token) <= 512
        or any(not 33 <= ord(char) <= 126 for char in token)
    ):
        raise CaptureAgentError("token must contain 1 to 512 printable ASCII characters without spaces")


def generate_token() -> str:
    """Return a URL-safe token suitable for a local token file."""

    return secrets.token_urlsafe(32)


def _relative_file(path: Path, root: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise CaptureAgentError(f"evidence file leaves output root: {path}")
    relative = resolved.relative_to(root).as_posix()
    if not relative or relative == _EVIDENCE_MANIFEST:
        raise CaptureAgentError("invalid evidence file path")
    return relative


def write_evidence_manifest(root: str | Path) -> Path:
    """Hash every completed run file except the manifest itself."""

    directory = Path(root).absolute()
    if not directory.is_dir() or any(_is_link(part) for part in (directory, *directory.parents)):
        raise CaptureAgentError("capture output directory must be a real directory")
    directory = directory.resolve()
    entries: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if _is_link(path):
            raise CaptureAgentError("capture evidence must not contain symbolic links or junctions")
        if not path.is_file() or path == directory / _EVIDENCE_MANIFEST:
            continue
        relative = _relative_file(path, directory)
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(chunk)
                digest.update(chunk)
        entries.append({"path": relative, "size_bytes": size, "sha256": digest.hexdigest()})
    if not entries:
        raise CaptureAgentError("capture output contains no evidence files")
    payload = {
        "schema_version": "1.0",
        "run_id": directory.name,
        "generated_at": _now(),
        "hash_algorithm": "SHA-256",
        "self_excluded": True,
        "file_count": len(entries),
        "total_size_bytes": sum(item["size_bytes"] for item in entries),
        "files": entries,
    }
    return atomic_write_text(
        directory / _EVIDENCE_MANIFEST,
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    )


def _validate_request(payload: Any, config: CaptureAgentConfig) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise CaptureAgentError("request body must be a JSON object")
    allowed = {
        "count", "interval_s", "duration_s", "detect_chessboard",
        "board_columns", "board_rows", "exposure_ms", "warmup_s",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise CaptureAgentError(f"unsupported capture fields: {', '.join(map(str, unknown))}")
    # Neither null nor duration-only requests may remove the count/time caps.
    count = payload.get("count")
    if count is None:
        count = config.max_count if payload.get("duration_s") is not None else config.default_count
    if type(count) is not int or not 1 <= count <= config.max_count:
        raise CaptureAgentError(f"count must be an integer between 1 and {config.max_count}")
    interval_s = payload.get("interval_s", config.default_interval_s)
    interval_s = _finite_number(
        interval_s,
        "interval_s",
        minimum=0.0,
        maximum=config.max_interval_s,
    )
    duration_s = payload.get("duration_s")
    if duration_s is not None:
        duration_s = _finite_number(
            duration_s,
            "duration_s",
            minimum=0.1,
            maximum=config.max_duration_s,
        )
    detect = payload.get("detect_chessboard", False)
    if type(detect) is not bool:
        raise CaptureAgentError("detect_chessboard must be a boolean")
    columns = payload.get("board_columns", 8)
    rows = payload.get("board_rows", 6)
    if type(columns) is not int or not 3 <= columns <= 50:
        raise CaptureAgentError("board_columns must be an integer between 3 and 50")
    if type(rows) is not int or not 3 <= rows <= 50:
        raise CaptureAgentError("board_rows must be an integer between 3 and 50")
    exposure_ms = payload.get("exposure_ms", None)
    if exposure_ms is not None:
        exposure_ms = _finite_number(
            exposure_ms,
            "exposure_ms",
            minimum=MIN_EXPOSURE_MS,
            maximum=MAX_EXPOSURE_MS,
        )
    warmup_s = _finite_number(
        payload.get("warmup_s", config.default_warmup_s),
        "warmup_s",
        minimum=0.0,
        maximum=config.max_warmup_s,
    )
    return {
        "count": count,
        "interval_s": interval_s,
        "duration_s": duration_s,
        "detect_chessboard": detect,
        "board_columns": columns,
        "board_rows": rows,
        "exposure_ms": exposure_ms,
        "warmup_s": warmup_s,
    }


def _job_id() -> str:
    return f"remote-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}"


class CaptureJobManager:
    """Serialize camera access and retain bounded job status in memory."""

    def __init__(
        self,
        config: CaptureAgentConfig,
        *,
        capture_fn: Callable[..., Path] = capture_stereo_pairs,
    ) -> None:
        self.config = config
        self._capture_fn = capture_fn
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="capture-agent")
        self._jobs: dict[str, dict[str, Any]] = {}
        self._active_job: str | None = None
        self._accepting = True
        self._runtime = copy.deepcopy(_runtime_info())

    def _public(self, job: Mapping[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(dict(job))
        result.pop("_run_dir", None)
        return result

    def _prune(self) -> None:
        if len(self._jobs) <= self.config.max_history:
            return
        removable = [
            (item.get("finished_at") or item.get("created_at") or "", job_id)
            for job_id, item in self._jobs.items()
            if job_id != self._active_job and item.get("state") in TERMINAL_STATES
        ]
        for _timestamp, job_id in sorted(removable)[: max(0, len(self._jobs) - self.config.max_history)]:
            self._jobs.pop(job_id, None)

    def submit(self, request: Mapping[str, Any]) -> dict[str, Any]:
        request = _validate_request(request, self.config)
        with self._lock:
            if not self._accepting:
                raise CaptureAgentError("capture agent is shutting down")
            if self._active_job is not None:
                active = self._jobs.get(self._active_job, {})
                raise CaptureAgentBusy(
                    f"camera is busy with job {self._active_job}; state={active.get('state', 'UNKNOWN')}"
                )
            job_id = _job_id()
            run_dir = self.config.output_root / job_id
            if run_dir.exists():
                raise CaptureAgentError("generated job directory already exists; retry the request")
            created_at = _now()
            job = {
                "runtime": copy.deepcopy(self._runtime),
                "schema_version": "1.0",
                "kind": "pipe_twin_remote_capture_job",
                "job_id": job_id,
                "state": "QUEUED",
                "created_at": created_at,
                "started_at": None,
                "finished_at": None,
                "request": dict(request),
                "run_directory": job_id,
                "run_url": (
                    f"{self.config.file_base_url.rstrip('/')}/{quote(job_id)}/"
                    if self.config.file_base_url
                    else None
                ),
                "pair_count": 0,
                "usable_pair_count": 0,
                "unusable_pair_count": 0,
                "quality_status": "PENDING",
                "error": None,
                "_run_dir": run_dir,
            }
            self._jobs[job_id] = job
            self._active_job = job_id
            self._prune()
            self._executor.submit(self._run, job_id)
            snapshot = self._public(job)
        log_event(_LOGGER, "capture_job_queued", job_id=job_id, request=request)
        return snapshot

    def _update(self, job_id: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            job = self._jobs[job_id]
            job.update(fields)
            return self._public(job)

    def _snapshot_internal(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return copy.deepcopy(self._jobs[job_id])

    def _write_job_file(self, job_id: str, *, override: Mapping[str, Any] | None = None) -> None:
        job = self._snapshot_internal(job_id)
        if override:
            job.update(override)
        run_dir = Path(job["_run_dir"])
        run_dir.mkdir(parents=True, exist_ok=True)
        record = self._public(job)
        atomic_write_text(run_dir / _JOB_FILE, json.dumps(record, ensure_ascii=False, indent=2) + "\n")

    def _run(self, job_id: str) -> None:
        self._update(job_id, state="RUNNING", started_at=_now())
        log_event(_LOGGER, "capture_job_started", job_id=job_id)
        job = self._snapshot_internal(job_id)
        run_dir = Path(job["_run_dir"])
        request = job["request"]
        final_fields: dict[str, Any]
        try:
            manifest = self._capture_fn(
                run_dir,
                left_index=self.config.left_index,
                right_index=self.config.right_index,
                layout=self.config.layout,
                eye_width=self.config.eye_width,
                eye_height=self.config.eye_height,
                backend=self.config.backend,
                count=request["count"],
                interval_s=request["interval_s"],
                duration_s=request["duration_s"],
                detect_chessboard=request["detect_chessboard"],
                board_columns=request["board_columns"],
                board_rows=request["board_rows"],
                exposure_ms=request["exposure_ms"],
                warmup_s=request["warmup_s"],
            )
            manifest_path = Path(manifest).absolute()
            expected_manifest = run_dir / "capture.json"
            if manifest_path != expected_manifest or manifest_path.is_symlink() or not manifest_path.is_file():
                raise CaptureAgentError("capture returned an invalid manifest path")
            capture_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            pair_count = capture_payload.get("pair_count")
            if (
                capture_payload.get("status") != "COMPLETED"
                or type(pair_count) is not int
                or not 1 <= pair_count <= request["count"]
            ):
                raise CaptureAgentError("capture did not produce a completed, bounded stereo run")
            self._update(
                job_id,
                pair_count=pair_count,
                usable_pair_count=capture_payload.get("usable_pair_count", 0),
                unusable_pair_count=capture_payload.get("unusable_pair_count", 0),
                quality_status=capture_payload.get("quality_status", "UNKNOWN"),
                capture_json="capture.json",
            )
            final_fields = {"state": "COMPLETED", "finished_at": _now()}
        except BaseException as error:  # worker must always release the camera slot
            final_fields = {
                "state": "FAILED",
                "finished_at": _now(),
                "error": {
                    "code": "CAMERA_BUSY" if isinstance(error, CameraBusyError) else "CAPTURE_FAILED",
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
            # Retain the count from a partial run when the camera disconnects.
            try:
                partial = json.loads((run_dir / "capture.json").read_text(encoding="utf-8"))
                pair_count = partial.get("pair_count")
                if type(pair_count) is int and 0 <= pair_count <= request["count"]:
                    final_fields.update(
                        pair_count=pair_count,
                        usable_pair_count=partial.get("usable_pair_count", 0),
                        unusable_pair_count=partial.get("unusable_pair_count", 0),
                        quality_status=partial.get("quality_status", "UNKNOWN"),
                        capture_json="capture.json",
                    )
            except (OSError, ValueError, AttributeError):
                pass
            log_event(
                _LOGGER,
                "capture_job_failed",
                job_id=job_id,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        finally:
            self._update(job_id, state="FINALIZING")
            try:
                # Both success and failure are published only after the
                # downloadable diagnostic record and its hashes are ready.
                self._write_job_file(job_id, override=final_fields)
                write_evidence_manifest(run_dir)
            except BaseException as error:
                # A camera capture without a verifiable evidence manifest is
                # never reported as successful.
                final_fields.update(
                    state="FAILED", finished_at=_now(),
                    error={"code": "EVIDENCE_FAILED", "type": type(error).__name__, "message": str(error)},
                )
                try:
                    (run_dir / _EVIDENCE_MANIFEST).unlink(missing_ok=True)
                    self._write_job_file(job_id, override=final_fields)
                except BaseException:
                    _LOGGER.exception("capture_job_status_write_failed")
                log_event(
                    _LOGGER,
                    "capture_job_evidence_failed",
                    job_id=job_id,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )
            finally:
                with self._lock:
                    self._jobs[job_id].update(final_fields)
                    if self._active_job == job_id:
                        self._active_job = None
            if final_fields["state"] == "COMPLETED":
                log_event(_LOGGER, "capture_job_completed", job_id=job_id, pair_count=pair_count)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else self._public(job)

    def health(self) -> dict[str, Any]:
        with self._lock:
            active = self._jobs.get(self._active_job) if self._active_job else None
            return {
                "status": "READY" if self._accepting else "STOPPING",
                "runtime": copy.deepcopy(self._runtime),
                "active_job": None if active is None else self._public(active),
                "camera_profile": {
                    "left_index": self.config.left_index,
                    "right_index": self.config.right_index,
                    "layout": self.config.layout,
                    "eye_width": self.config.eye_width,
                    "eye_height": self.config.eye_height,
                },
            }

    def shutdown(self, *, wait: bool = True) -> None:
        with self._lock:
            self._accepting = False
        self._executor.shutdown(wait=wait, cancel_futures=False)


class CaptureAgentRequestHandler(BaseHTTPRequestHandler):
    """Small JSON API; arbitrary HTTP methods and paths are rejected."""

    server_version = "PipeTwinCaptureAgent/1.0"
    protocol_version = "HTTP/1.1"
    timeout = 10.0

    @property
    def agent_server(self) -> "CaptureAgentHTTPServer":
        return self.server  # type: ignore[return-value]

    def log_message(self, format: str, *args: Any) -> None:
        log_event(_LOGGER, "http_request", message=format % args, client=self.client_address[0])

    def _json(self, code: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # Rejected POST bodies must never be parsed as the next request.
        self.send_header("Connection", "close")
        self.close_connection = True
        if code == 401:
            self.send_header("WWW-Authenticate", "Bearer")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, error: str, message: str) -> None:
        self._json(code, {"error": error, "message": message})

    def _authorized(self) -> bool:
        if self.agent_server.token is None:
            if not _trusted_control_address(self.client_address[0]):
                self._error(403, "FORBIDDEN", "token-free control requires a loopback or Tailscale peer")
                return False
            return True
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {self.agent_server.token}"
        if not secrets.compare_digest(supplied.encode("utf-8"), expected.encode("ascii")):
            self._error(401, "UNAUTHORIZED", "valid bearer token required")
            return False
        return True

    def _path(self) -> str:
        parsed = urlsplit(self.path)
        if parsed.query or parsed.fragment:
            raise CaptureAgentError("query and fragment are not supported")
        return parsed.path

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        try:
            if not self._authorized():
                return
            path = self._path()
            if path == f"{API_PREFIX}/health":
                result = self.agent_server.manager.health()
                result["auth_required"] = self.agent_server.token is not None
                self._json(200, result)
                return
            prefix = f"{API_PREFIX}/captures/"
            if path.startswith(prefix):
                job_id = unquote(path[len(prefix) :])
                if not JOB_ID_RE.fullmatch(job_id):
                    self._error(400, "INVALID_JOB_ID", "invalid job id")
                    return
                result = self.agent_server.manager.get(job_id)
                if result is None:
                    self._error(404, "JOB_NOT_FOUND", "capture job was not found")
                else:
                    self._json(200, result)
                return
            self._error(404, "NOT_FOUND", "unknown capture-agent endpoint")
        except CaptureAgentError as error:
            self._error(400, "BAD_REQUEST", str(error))
        except Exception as error:  # native socket errors must not kill the server
            _LOGGER.exception("capture_agent_get_failed")
            self._error(500, "INTERNAL_ERROR", str(error))

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        try:
            if not self._authorized():
                return
            if self._path() != f"{API_PREFIX}/captures":
                self._error(404, "NOT_FOUND", "unknown capture-agent endpoint")
                return
            if self.headers.get("Transfer-Encoding") is not None:
                raise CaptureAgentError("Transfer-Encoding is not supported")
            if len(self.headers.get_all("Content-Length", [])) > 1:
                raise CaptureAgentError("duplicate Content-Length is not supported")
            if self.headers.get_content_type() != "application/json":
                self._error(415, "UNSUPPORTED_MEDIA_TYPE", "request body must be application/json")
                return
            raw_length = self.headers.get("Content-Length")
            if raw_length is None:
                self._error(411, "LENGTH_REQUIRED", "Content-Length is required")
                return
            try:
                length = int(raw_length)
            except ValueError as error:
                raise CaptureAgentError("Content-Length must be an integer") from error
            if length < 0 or length > MAX_REQUEST_BYTES:
                self._error(413, "REQUEST_TOO_LARGE", f"request must be <= {MAX_REQUEST_BYTES} bytes")
                return
            raw = self.rfile.read(length)
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CaptureAgentError("request body must be UTF-8 JSON") from error
            try:
                result = self.agent_server.manager.submit(payload)
            except CaptureAgentBusy as error:
                self._error(409, "CAMERA_BUSY", str(error))
                return
            self._json(202, result)
        except CaptureAgentError as error:
            self._error(400, "BAD_REQUEST", str(error))
        except Exception as error:
            _LOGGER.exception("capture_agent_post_failed")
            self._error(500, "INTERNAL_ERROR", str(error))

    def do_PUT(self) -> None:  # noqa: N802
        self._error(405, "METHOD_NOT_ALLOWED", "only POST /v1/captures is supported")

    def do_DELETE(self) -> None:  # noqa: N802
        self._error(405, "METHOD_NOT_ALLOWED", "remote deletion is disabled")


class CaptureAgentHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], manager: CaptureJobManager, token: str | None):
        if token is not None:
            _validate_token(token)
        elif not _trusted_control_address(server_address[0]):
            raise CaptureAgentError("token-free control must bind to a loopback or Tailscale IPv4 address")
        self.manager = manager
        self.token = token
        super().__init__(server_address, CaptureAgentRequestHandler)


def serve_capture_agent(
    *,
    bind: str,
    port: int,
    token_file: str | Path,
    config: CaptureAgentConfig,
) -> None:
    """Run the blocking office-side HTTP control service."""

    if type(port) is not int or not 1 <= port <= 65535:
        raise CaptureAgentError("port must be between 1 and 65535")
    if Path(token_file).expanduser().resolve().is_relative_to(config.output_root):
        raise CaptureAgentError("token file must be outside the output root served for downloads")
    token = _read_token_file(token_file)
    manager = CaptureJobManager(config)
    try:
        server = CaptureAgentHTTPServer((bind, port), manager, token)
    except OSError as error:
        manager.shutdown(wait=False)
        raise CaptureAgentError(f"cannot bind capture agent to {bind}:{port}: {error}") from error
    log_event(
        _LOGGER,
        "capture_agent_started",
        bind=bind,
        port=port,
        output_root=config.output_root,
        file_base_url=config.file_base_url,
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()
        manager.shutdown(wait=True)
        log_event(_LOGGER, "capture_agent_stopped")


def _agent_url(value: str) -> str:
    base = _base_url(value)
    return base.rstrip("/")


def _request_json(
    method: str,
    url: str,
    token: str | None,
    payload: Mapping[str, Any] | None = None,
    *,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    if token is not None:
        _validate_token(token)
    if isinstance(timeout_s, bool) or not math.isfinite(float(timeout_s)) or timeout_s <= 0:
        raise CaptureAgentError("timeout_s must be positive and finite")
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Accept": "application/json",
        **({"Authorization": f"Bearer {token}"} if token is not None else {}),
        **({"Content-Type": "application/json"} if data is not None else {}),
    }
    request = Request(
        url,
        data=data,
        method=method,
        headers=headers,
    )
    opener = build_opener(ProxyHandler({}), _NoRedirect())
    try:
        with opener.open(request, timeout=float(timeout_s)) as response:
            raw = response.read(MAX_REQUEST_BYTES)
    except HTTPError as error:
        try:
            detail = error.read(MAX_REQUEST_BYTES).decode("utf-8", errors="replace")
            parsed = json.loads(detail)
            message = str(parsed.get("message", detail)) if isinstance(parsed, dict) else detail
        except Exception:
            message = str(error)
        raise CaptureAgentError(f"remote capture request failed ({error.code}): {message}") from error
    except (OSError, URLError) as error:
        raise CaptureAgentError(f"remote capture request failed: {error}") from error
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CaptureAgentError("remote capture service returned invalid JSON") from error
    if not isinstance(parsed, dict):
        raise CaptureAgentError("remote capture service must return a JSON object")
    return parsed


def submit_remote_capture(
    agent_url: str,
    token: str | None,
    request: Mapping[str, Any],
    *,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    """Submit one bounded capture request from the home workstation."""

    return _request_json(
        "POST",
        _agent_url(agent_url) + f"{API_PREFIX}/captures",
        token,
        request,
        timeout_s=timeout_s,
    )


def get_remote_capture(
    agent_url: str,
    token: str | None,
    job_id: str,
    *,
    timeout_s: float = 15.0,
) -> dict[str, Any]:
    """Read one remote job status without exposing a local file path."""

    if not JOB_ID_RE.fullmatch(job_id):
        raise CaptureAgentError("invalid job id")
    return _request_json(
        "GET",
        _agent_url(agent_url) + f"{API_PREFIX}/captures/{quote(job_id)}",
        token,
        timeout_s=timeout_s,
    )


def read_token(path: str | Path) -> str:
    """Public token-file reader for the home-side CLI."""

    return _read_token_file(path)


__all__ = [
    "CaptureAgentBusy",
    "CaptureAgentConfig",
    "CaptureAgentError",
    "CaptureAgentHTTPServer",
    "CaptureJobManager",
    "generate_token",
    "get_remote_capture",
    "read_token",
    "serve_capture_agent",
    "submit_remote_capture",
    "write_evidence_manifest",
]
