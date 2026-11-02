"""Trusted-network HTTP service for queued stereo analysis.

This is a LAN/Tailscale service, without authentication or TLS. It accepts only
loopback, RFC1918, ULA and Tailscale source addresses. Bind defaults
to loopback; no firewall settings are changed. Every analysis runs in its own
spawned process so a cancelled or crashed computation cannot block later jobs.
"""
from __future__ import annotations

import ipaddress
import json
import multiprocessing
import queue
import re
import shutil
import socket
import threading
import time
from collections import deque
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit
from uuid import uuid4

from .analysis_transport import MAX_BUNDLE_BYTES, build_result_bundle, extract_request_bundle

SERVICE = "pipe-twin-analysis"
PROTOCOL_VERSION = "1"
_TERMINAL = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})
_JOB_ID = re.compile(r"[0-9a-f]{32}\Z")
_TRUSTED_NETWORKS = tuple(ipaddress.ip_network(value) for value in (
    "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
    "100.64.0.0/10", "::1/128", "fc00::/7",
))


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _trusted_address(value: str) -> bool:
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return any(address.version == net.version and address in net for net in _TRUSTED_NETWORKS)


def _analysis_process(manifest: str, result_dir: str, binding: dict,
                      analyzer: Callable | None, events: Any) -> None:
    """Top-level target is required by Windows multiprocessing spawn."""
    def progress(value: Any, *args: Any, **kwargs: Any) -> None:
        message = str(value) if args or kwargs else value
        try:
            # Reject non-JSON values from custom analyzers and cap event size.
            encoded = json.dumps(message, ensure_ascii=False, allow_nan=False)
            if len(encoded) > 8000:
                message = encoded[:8000]
        except (ValueError, TypeError):
            message = str(message)[:8000]
        try:
            events.put_nowait(("progress", message))
        except queue.Full:
            pass

    try:
        if analyzer is None:
            from .stereo_analyzer import analyze_stereo_capture
            analyzer = analyze_stereo_capture
        output = Path(result_dir)
        output.mkdir(parents=True, exist_ok=False)
        report_path = output / "report.json"
        progress("正在读取输入并运行 CPU 分析……")
        report = analyzer(Path(manifest), report_output_path=report_path,
                          evidence_dir=output / "evidence", progress=progress)
        if not report_path.exists():
            if not isinstance(report, dict):
                raise ValueError("Analyzer did not produce a report")
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        progress("正在校验并打包报告和证据文件……")
        result = build_result_bundle(Path(manifest).parent, report_path, binding)
        (output / "result.zip").write_bytes(result)
        events.put(("succeeded", str(output / "result.zip")))
    except BaseException as error:
        # Child failures remain job failures, including unexpected SystemExit.
        events.put(("failed", f"{type(error).__name__}: {error}"[:8000]))


class AnalysisServer:
    """Run one computation at a time, with a bounded pending queue.

    ``analyzer`` is optional and must be a spawn-pickleable callable accepting
    the same arguments as ``analyze_stereo_capture`` (including ``progress``).
    ``start()`` is non-blocking and returns this instance. ``port=0`` asks the
    OS for a free port. Call ``jobs()``/``health()`` from the GUI thread; their
    snapshots contain JSON-compatible values. Stopping cancels unfinished jobs.
    """
    def __init__(self, bind: str = "127.0.0.1", port: int = 8771,
                 work_root: str | Path = "outputs/analysis-server", *,
                 analyzer: Callable | None = None, max_pending_jobs: int = 4,
                 max_upload_bytes: int = MAX_BUNDLE_BYTES,
                 capabilities: dict[str, Any] | None = None,
                 request_timeout: float = 30.0) -> None:
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("port must be an integer between 0 and 65535")
        if type(max_pending_jobs) is not int or max_pending_jobs < 0:
            raise ValueError("max_pending_jobs must be a non-negative integer")
        if type(max_upload_bytes) is not int or not 0 < max_upload_bytes <= MAX_BUNDLE_BYTES:
            raise ValueError(f"max_upload_bytes must be between 1 and {MAX_BUNDLE_BYTES}")
        if not isinstance(bind, str) or not bind:
            raise ValueError("bind must be a local interface address or localhost")
        if request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        addresses = socket.getaddrinfo(bind, port, type=socket.SOCK_STREAM)
        family, _, _, _, endpoint = addresses[0]
        if any(not (_trusted_address(item[4][0]) or item[4][0] in ("0.0.0.0", "::"))
               for item in addresses):
            raise ValueError("Server may bind only to loopback, LAN or Tailscale addresses")
        self.bind, self.port = bind, port
        self.work_root = Path(work_root).resolve()
        self.max_pending_jobs = max_pending_jobs
        self.max_upload_bytes = max_upload_bytes
        self.request_timeout = float(request_timeout)
        self._family, self._endpoint = family, endpoint
        self._analyzer = analyzer
        self._capabilities = capabilities
        self._condition = threading.Condition(threading.RLock())
        self._lifecycle_lock = threading.Lock()
        self._pending: deque[str] = deque()
        self._jobs: dict[str, dict] = {}
        self._active: str | None = None
        self._admissions = 0
        self._stop_event = threading.Event()
        self._http: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self._worker: threading.Thread | None = None
        self._process: Any = None
        self._running = False
        self._generation = 0

    @property
    def is_busy(self) -> bool:
        """Include uploads being validated, pending jobs and active processes."""
        with self._condition:
            return self._admissions > 0

    @property
    def base_url(self) -> str:
        host = self.bind
        if host == "0.0.0.0":
            host = "127.0.0.1"
        elif host == "::":
            host = "::1"
        if ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self.port}"

    def start(self) -> AnalysisServer:
        with self._lifecycle_lock:
            if self._running:
                return self
            self.work_root.mkdir(parents=True, exist_ok=True)
            if self._capabilities is None:
                from .analysis_capabilities import probe_capabilities
                self._capabilities = probe_capabilities()
            owner = self

            class Handler(_Handler):
                service = owner

            class Server(ThreadingHTTPServer):
                address_family = owner._family
                daemon_threads = True
                block_on_close = False
                request_queue_size = 16
                _connections = threading.BoundedSemaphore(16)

                def verify_request(self, request: Any, client_address: Any) -> bool:
                    return _trusted_address(client_address[0])

                def process_request(self, request: Any, client_address: Any) -> None:
                    if not self._connections.acquire(blocking=False):
                        self.shutdown_request(request)
                        return
                    try:
                        super().process_request(request, client_address)
                    except BaseException:
                        self._connections.release()
                        raise

                def process_request_thread(self, request: Any, client_address: Any) -> None:
                    try:
                        super().process_request_thread(request, client_address)
                    finally:
                        self._connections.release()

            self._http = Server(self._endpoint, Handler)
            self.port = int(self._http.server_address[1])
            self._stop_event.clear()
            self._generation += 1
            self._running = True
            self._worker = threading.Thread(target=self._work, name="analysis-worker", daemon=True)
            self._http_thread = threading.Thread(target=self._http.serve_forever,
                                                 kwargs={"poll_interval": 0.1},
                                                 name="analysis-http", daemon=True)
            self._worker.start()
            self._http_thread.start()
        return self

    def stop(self) -> None:
        with self._lifecycle_lock:
            if not self._running:
                return
            with self._condition:
                self._running = False
                self._stop_event.set()
                for job_id in list(self._pending):
                    self._finish_locked(job_id, "CANCELLED", "服务器已停止")
                self._pending.clear()
                if self._active:
                    self._jobs[self._active]["_cancel"].set()
                self._condition.notify_all()
            if self._http is not None:
                self._http.shutdown()
                self._http.server_close()
            if self._worker is not None:
                self._worker.join(timeout=10)
            if self._http_thread is not None:
                self._http_thread.join(timeout=2)
            self._http = None

    def __enter__(self) -> AnalysisServer:
        return self.start()

    def __exit__(self, *args: Any) -> None:
        self.stop()

    @staticmethod
    def _public(job: dict) -> dict:
        # JSON round-trip prevents callers mutating nested progress snapshots.
        return json.loads(json.dumps({key: value for key, value in job.items()
                                      if not key.startswith("_")}, ensure_ascii=False))

    def jobs(self) -> list[dict]:
        with self._condition:
            return [self._public(job) for job in reversed(list(self._jobs.values()))]

    def get_job(self, job_id: str) -> dict:
        with self._condition:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._public(self._jobs[job_id])

    def health(self) -> dict:
        with self._condition:
            return {
                "service": SERVICE, "protocol_version": PROTOCOL_VERSION,
                "status": "RUNNING" if self._running else "STOPPED",
                "bind": self.bind, "port": self.port, "base_url": self.base_url,
                "capabilities": json.loads(json.dumps(self._capabilities or {})),
                "active_job": self._public(self._jobs[self._active]) if self._active else None,
                "queued_jobs": len(self._pending), "max_pending_jobs": self.max_pending_jobs,
                "is_busy": self.is_busy,
                "jobs": self.jobs(),
            }

    def cancel(self, job_id: str) -> dict:
        with self._condition:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job["status"] == "QUEUED":
                self._pending.remove(job_id)
                self._finish_locked(job_id, "CANCELLED", "任务已取消")
            elif job["status"] == "RUNNING":
                job["_cancel"].set()
                job["progress"] = "正在终止计算……"
            self._condition.notify_all()
            return self._public(job)

    def _reserve(self) -> int:
        with self._condition:
            if not self._running:
                raise RuntimeError("Server is stopping")
            if self._admissions >= self.max_pending_jobs + 1:
                raise queue.Full("Analysis queue is full")
            self._admissions += 1
            return self._generation

    def _submit_reserved(self, content: bytes, generation: int) -> dict:
        directory: Path | None = None
        try:
            # Windows tempfile names for evidence must fit below MAX_PATH on
            # ordinary installations. Keep the HTTP identity long, but use a
            # compact, exclusively created directory for the on-disk job.
            while True:
                job_id = uuid4().hex
                candidate = self.work_root / ("j" + job_id[:11])
                try:
                    candidate.mkdir(exist_ok=False)
                except FileExistsError:
                    continue
                directory = candidate
                break
            manifest, binding = extract_request_bundle(content, directory / "input")
            while True:
                output = manifest.parent / ("r" + uuid4().hex[:8])
                if not output.exists():
                    break
            job = {"job_id": job_id, "request_id": binding.get("request_id"),
                   "status": "QUEUED", "progress": "等待计算", "error": None,
                   "created_at": _timestamp(), "started_at": None, "finished_at": None,
                   "_manifest": manifest, "_binding": binding, "_output": output,
                   "_cancel": threading.Event(), "_result": None}
            with self._condition:
                if not self._running or generation != self._generation:
                    raise RuntimeError("Server is stopping")
                self._jobs[job_id] = job
                self._pending.append(job_id)
                self._condition.notify()
                return self._public(job)
        except BaseException:
            with self._condition:
                self._admissions -= 1
            # Only remove the newly generated direct child of this work root.
            if (directory is not None and directory.resolve().parent == self.work_root
                    and re.fullmatch(r"j[0-9a-f]{11}", directory.name)):
                shutil.rmtree(directory, ignore_errors=True)
            raise

    def _finish_locked(self, job_id: str, status: str, progress: str,
                       error: str | None = None) -> None:
        job = self._jobs[job_id]
        if job["status"] in _TERMINAL:
            return
        job.update(status=status, progress=progress, error=error, finished_at=_timestamp())
        self._admissions -= 1

    def _work(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(lambda: bool(self._pending) or self._stop_event.is_set())
                if self._stop_event.is_set():
                    return
                job_id = self._pending.popleft()
                job = self._jobs[job_id]
                self._active = job_id
                job.update(status="RUNNING", progress="正在启动 CPU 分析进程……", started_at=_timestamp())
            try:
                self._execute(job_id)
            except Exception as error:
                with self._condition:
                    self._finish_locked(job_id, "FAILED", "任务失败", f"{type(error).__name__}: {error}")
            finally:
                with self._condition:
                    self._active = None
                    self._process = None

    def _execute(self, job_id: str) -> None:
        job = self._jobs[job_id]
        context = multiprocessing.get_context("spawn")
        events = context.Queue(maxsize=64)
        process = context.Process(target=_analysis_process,
                                  args=(str(job["_manifest"]), str(job["_output"]),
                                        job["_binding"], self._analyzer, events), daemon=True)
        self._process = process
        try:
            process.start()
            dead_since: float | None = None
            while True:
                if job["_cancel"].is_set() or self._stop_event.is_set():
                    self._terminate(process)
                    with self._condition:
                        self._finish_locked(job_id, "CANCELLED", "任务已取消")
                    return
                try:
                    kind, payload = events.get(timeout=0.05)
                except queue.Empty:
                    if not process.is_alive():
                        dead_since = dead_since or time.monotonic()
                        if time.monotonic() - dead_since > 0.2:
                            with self._condition:
                                self._finish_locked(job_id, "FAILED", "分析进程异常退出",
                                                    f"Analysis process exited with code {process.exitcode}")
                            return
                    continue
                with self._condition:
                    if job["_cancel"].is_set() or self._stop_event.is_set():
                        continue
                    if kind == "progress":
                        job["progress"] = payload
                    elif kind == "succeeded":
                        job["_result"] = Path(payload)
                        self._finish_locked(job_id, "SUCCEEDED", "分析完成")
                        return
                    elif kind == "failed":
                        self._finish_locked(job_id, "FAILED", "任务失败", payload)
                        return
        finally:
            if process.pid is not None:
                process.join(timeout=0.5)
                if process.is_alive():
                    self._terminate(process)
                process.close()
            events.cancel_join_thread()
            events.close()

    @staticmethod
    def _terminate(process: Any) -> None:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
        if process.is_alive():
            process.kill()
            process.join(timeout=2)


class _Handler(BaseHTTPRequestHandler):
    service: AnalysisServer
    server_version = "PipeTwinAnalysis/1"
    protocol_version = "HTTP/1.0"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.service.request_timeout)

    def log_message(self, format: str, *args: Any) -> None:
        # GUI status is surfaced through health()/jobs(); don't print image paths.
        pass

    def _json(self, status: int, value: dict) -> None:
        content = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _route(self) -> list[str]:
        parsed = urlsplit(self.path)
        if parsed.query or parsed.fragment or parsed.scheme or parsed.netloc:
            return []
        return parsed.path.strip("/").split("/")

    def do_GET(self) -> None:
        try:
            route = self._route()
            if route == ["health"]:
                self._json(HTTPStatus.OK, self.service.health())
                return
            if len(route) in (2, 3) and route[0] == "jobs" and _JOB_ID.fullmatch(route[1]):
                job = self.service.get_job(route[1])
                if len(route) == 2:
                    self._json(HTTPStatus.OK, job)
                    return
                if route[2] == "result":
                    if job["status"] != "SUCCEEDED":
                        self._error(HTTPStatus.CONFLICT, "Result is available only for a succeeded job")
                        return
                    with self.service._condition:
                        path = self.service._jobs[route[1]]["_result"]
                    with path.open("rb") as stream:
                        self.send_response(HTTPStatus.OK)
                        self.send_header("Content-Type", "application/zip")
                        self.send_header("Content-Length", str(path.stat().st_size))
                        self.send_header("Content-Disposition", f'attachment; filename="{route[1]}.zip"')
                        self.end_headers()
                        shutil.copyfileobj(stream, self.wfile, length=1024 * 1024)
                    return
            self._error(HTTPStatus.NOT_FOUND, "Unknown API path")
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "Unknown job")
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        except OSError:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "Could not read job result")

    def do_POST(self) -> None:
        reserved = False
        try:
            lengths = self.headers.get_all("Content-Length") or []
            if self.headers.get("Transfer-Encoding"):
                self._error(HTTPStatus.BAD_REQUEST, "Transfer-Encoding is unsupported; use Content-Length")
                return
            if len(lengths) > 1 or (lengths and not re.fullmatch(r"[0-9]{1,12}", lengths[0])):
                self._error(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
                return
            length = int(lengths[0]) if lengths else 0
            route = self._route()
            if len(route) == 3 and route[0] == "jobs" and route[2] == "cancel" and _JOB_ID.fullmatch(route[1]):
                if length:
                    self._error(HTTPStatus.BAD_REQUEST, "Cancel request must have an empty body")
                    return
                self._json(HTTPStatus.OK, self.service.cancel(route[1]))
                return
            if route != ["jobs"]:
                self._error(HTTPStatus.NOT_FOUND, "Unknown API path")
                return
            if not lengths or length <= 0:
                self._error(HTTPStatus.LENGTH_REQUIRED, "A positive Content-Length is required")
                return
            if length > self.service.max_upload_bytes:
                self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Upload exceeds the configured size limit")
                return
            if self.headers.get_content_type() != "application/zip":
                self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/zip")
                return
            generation = self.service._reserve()
            reserved = True
            content = self.rfile.read(length)
            if len(content) != length:
                self._error(HTTPStatus.BAD_REQUEST, "Incomplete upload")
                return
            reserved = False  # _submit_reserved owns admission release, including errors.
            job = self.service._submit_reserved(content, generation)
            self._json(HTTPStatus.ACCEPTED, job)
        except queue.Full:
            self._error(HTTPStatus.TOO_MANY_REQUESTS, "Analysis queue is full; retry after a job completes")
        except KeyError:
            self._error(HTTPStatus.NOT_FOUND, "Unknown job")
        except (ValueError, OSError) as error:
            if isinstance(error, (BrokenPipeError, ConnectionResetError)):
                return
            self._error(HTTPStatus.BAD_REQUEST, str(error)[:1000])
        except RuntimeError as error:
            self._error(HTTPStatus.SERVICE_UNAVAILABLE, str(error)[:1000])
        finally:
            if reserved:
                with self.service._condition:
                    self.service._admissions -= 1


__all__ = ["AnalysisServer", "SERVICE", "PROTOCOL_VERSION"]
