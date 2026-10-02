"""Office services owned by the desktop client; no camera is opened at startup."""

from __future__ import annotations

import ipaddress
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
from dataclasses import asdict, dataclass, replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlsplit
from urllib.request import ProxyHandler, build_opener

from .capture_agent import (
    CaptureAgentConfig, CaptureAgentHTTPServer, CaptureJobManager,
    _is_link, generate_token, read_token,
)
from .logging_config import get_logger, log_event
from .pipeline import atomic_write_text
from .remote_fetch import _NoRedirect

_LOGGER = get_logger("office_client")
_TAILNET = ipaddress.ip_network("100.64.0.0/10")


class OfficeClientError(ValueError):
    """An actionable office service configuration or startup error."""


def settings_path() -> Path:
    return Path.home() / ".pipe-twin" / "office-client.json"


def _private_path(path: Path) -> Path:
    path = path.expanduser().absolute()
    if any(_is_link(part) for part in (path, *path.parents)):
        raise OfficeClientError("配置或密钥路径不能经过符号链接或目录联接")
    return path


def _tailscale_command() -> str | None:
    command = shutil.which("tailscale")
    if command:
        return command
    candidate = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Tailscale" / "tailscale.exe"
    return str(candidate) if candidate.is_file() else None


def detect_tailscale_address() -> str | None:
    """Prefer the installed Tailscale CLI, then inspect local IPv4 addresses."""
    addresses = []
    command = _tailscale_command()
    if command:
        try:
            result = subprocess.run(
                [command, "ip", "-4"], capture_output=True, text=True, timeout=4,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if result.returncode == 0:
                addresses.extend(result.stdout.split())
        except (OSError, subprocess.TimeoutExpired):
            pass
    try:
        addresses.extend(item[4][0] for item in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    for value in addresses:
        try:
            address = ipaddress.ip_address(value)
            if address in _TAILNET:
                return str(address)
        except ValueError:
            continue
    return None


def default_output_root() -> Path:
    configured = os.environ.get("PIPE_TWIN_OUTPUT_ROOT")
    if configured:
        return Path(configured).expanduser()
    if os.name == "nt" and Path("D:/").is_dir():
        return Path("D:/pipe_twin_runs")
    return Path.home() / "pipe_twin_runs"


@dataclass(frozen=True)
class OfficeClientConfig:
    bind: str = ""
    capture_port: int = 8770
    file_port: int = 8765
    require_token: bool = False
    token_file: Path | None = None
    output_root: Path | None = None
    left_index: int = 0
    right_index: int | None = None
    layout: str = "side_by_side_left_right"
    eye_width: int = 1920
    eye_height: int = 1080
    backend: int | None = None

    def resolved(self) -> "OfficeClientConfig":
        bind = self.bind or os.environ.get("PIPE_TWIN_BIND") or detect_tailscale_address() or "127.0.0.1"
        try:
            address = ipaddress.ip_address(bind)
        except ValueError as error:
            raise OfficeClientError("监听地址必须为本机 Tailscale IPv4 地址或 127.0.0.1") from error
        if address.version != 4 or not (address.is_loopback or address in _TAILNET):
            raise OfficeClientError("远程助手只监听 Tailscale 或本机回环地址")
        for port in (self.capture_port, self.file_port):
            if type(port) is not int or not 0 <= port <= 65535:
                raise OfficeClientError("端口必须为 0 到 65535 的整数")
        if self.capture_port and self.capture_port == self.file_port:
            raise OfficeClientError("拍照与文件服务必须使用不同端口")
        if type(self.require_token) is not bool:
            raise OfficeClientError("require_token 必须为布尔值")
        root = _private_path(Path(self.output_root or default_output_root()))
        token = None
        if self.require_token:
            token = self.token_file or Path(os.environ.get("PIPE_TWIN_TOKEN_FILE", str(Path.home() / ".pipe-twin/capture-agent.token")))
            token = _private_path(Path(token))
            if token.resolve().is_relative_to(root.resolve()):
                raise OfficeClientError("连接密钥必须放在照片下载目录之外")
        return replace(self, bind=str(address), output_root=root, token_file=token)


def load_office_config(path: Path | None = None) -> OfficeClientConfig:
    source = path or settings_path()
    if not source.exists():
        return OfficeClientConfig()
    try:
        value = json.loads(_private_path(source).read_text(encoding="utf-8"))
        for key in ("token_file", "output_root"):
            if value.get(key) is not None:
                value[key] = Path(value[key])
        return OfficeClientConfig(**value)
    except (OSError, ValueError, TypeError, AttributeError) as error:
        raise OfficeClientError(f"无法读取远程助手设置 {source}：{error}") from error


def save_office_config(config: OfficeClientConfig, path: Path | None = None) -> None:
    value = asdict(config)
    for key in ("token_file", "output_root"):
        value[key] = str(value[key]) if value[key] is not None else None
    destination = _private_path(path or settings_path())
    root = Path(config.output_root or default_output_root()).expanduser().resolve()
    if destination.resolve().is_relative_to(root):
        raise OfficeClientError("助手设置必须放在照片下载目录之外")
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(destination, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _ensure_token(path: Path) -> bool:
    path = _private_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        read_token(path)
        return False
    with os.fdopen(descriptor, "w", encoding="ascii") as handle:
        handle.write(generate_token() + "\n")
    return True


def _configure_windows_firewall(port: int) -> str:
    """Best-effort idempotent Tailnet-only rule for the control port."""
    if os.name != "nt":
        return "NOT_APPLICABLE"
    if os.environ.get("PIPE_TWIN_CONFIGURE_FIREWALL", "1").casefold() in {"0", "false", "no"}:
        return "SKIPPED_BY_ENVIRONMENT"
    display_name = "Pipe Twin Capture Agent (Tailscale)"
    command = (
        "$rule = Get-NetFirewallRule -DisplayName 'Pipe Twin Capture Agent (Tailscale)' "
        "-ErrorAction SilentlyContinue; "
        "if ($null -eq $rule) { "
        f"New-NetFirewallRule -DisplayName '{display_name}' -Direction Inbound "
        f"-Action Allow -Protocol TCP -LocalPort {int(port)} "
        "-RemoteAddress '100.64.0.0/10' -Profile Any -ErrorAction Stop | Out-Null; "
        "'CREATED' } else { 'EXISTING' }"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, timeout=12,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired):
        return "UNAVAILABLE"
    if result.returncode == 0:
        return result.stdout.strip() or "CONFIGURED"
    _LOGGER.warning("windows firewall rule was not configured: %s", result.stderr.strip())
    return "NEEDS_ADMINISTRATOR"


class _EvidenceHandler(SimpleHTTPRequestHandler):
    """Read-only files confined to the evidence root, including on Windows."""

    def _read_only(self) -> None:
        self.send_error(405, "evidence server is read-only")

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler hook
        self._read_only()

    def do_PUT(self) -> None:  # noqa: N802 - stdlib handler hook
        self._read_only()

    def do_PATCH(self) -> None:  # noqa: N802 - stdlib handler hook
        self._read_only()

    def do_DELETE(self) -> None:  # noqa: N802 - stdlib handler hook
        self._read_only()

    def send_head(self):
        try:
            name = unquote(urlsplit(self.path).path)
            parts = name.strip("/").split("/") if name.strip("/") else []
            if any(part in {".", ".."} or ":" in part or "\\" in part or "\x00" in part for part in parts):
                raise ValueError("invalid path")
            root = Path(self.directory).absolute()
            if any(_is_link(item) for item in (root, *root.parents)):
                raise ValueError("link in root")
            path = root
            for part in parts:
                path = path / part
                if _is_link(path):
                    raise ValueError("link in path")
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("path leaves root")
        except (ValueError, OSError):
            self.send_error(403, "file is outside the evidence directory")
            return None
        return super().send_head()

    def log_message(self, format: str, *args: Any) -> None:
        log_event(_LOGGER, "evidence_request", message=format % args, client=self.client_address[0])


class _EvidenceServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _existing_server_matches(root: Path, base_url: str) -> bool:
    with tempfile.NamedTemporaryFile(dir=root, prefix=".pipe-twin-probe-", suffix=".txt", delete=False) as handle:
        probe = Path(handle.name)
        data = os.urandom(32).hex().encode("ascii")
        handle.write(data)
    try:
        with build_opener(ProxyHandler({}), _NoRedirect()).open(base_url + "/" + probe.name, timeout=2) as response:
            return response.read(len(data) + 1) == data
    except (OSError, ValueError):
        return False
    finally:
        probe.unlink(missing_ok=True)


class OfficeClient:
    """Own services; never stop an existing external evidence server."""

    def __init__(self, config: OfficeClientConfig, *, capture_fn: Callable | None = None):
        self.config = config
        self.capture_fn = capture_fn
        self.capture_manager: CaptureJobManager | None = None
        self.capture_server: CaptureAgentHTTPServer | None = None
        self.file_server: _EvidenceServer | None = None
        self._threads: dict[Any, threading.Thread] = {}
        self.details: dict[str, Any] = {"status": "STOPPED"}

    def start(self) -> dict[str, Any]:
        if self.capture_server is not None:
            raise OfficeClientError("远程助手已经启动")
        cfg = self.config.resolved()
        self.details = {"status": "STARTING"}
        try:
            capture_config = CaptureAgentConfig(
                output_root=cfg.output_root, left_index=cfg.left_index, right_index=cfg.right_index,
                layout=cfg.layout, eye_width=cfg.eye_width, eye_height=cfg.eye_height,
                backend=cfg.backend,
            )
            created = False
            token = None
            if cfg.require_token:
                created = _ensure_token(cfg.token_file)
                token = read_token(cfg.token_file)
            file_url = f"http://{cfg.bind}:{cfg.file_port}"
            try:
                self.file_server = _EvidenceServer(
                    (cfg.bind, cfg.file_port), partial(_EvidenceHandler, directory=str(capture_config.output_root)),
                )
                file_url = f"http://{cfg.bind}:{self.file_server.server_port}"
            except OSError as error:
                if not cfg.file_port or not _existing_server_matches(capture_config.output_root, file_url):
                    raise OfficeClientError(f"文件端口 {cfg.file_port} 无法使用，或现有服务未共享相同照片目录：{error}") from error
            capture_config = replace(capture_config, file_base_url=file_url)
            options = {} if self.capture_fn is None else {"capture_fn": self.capture_fn}
            self.capture_manager = CaptureJobManager(capture_config, **options)
            try:
                self.capture_server = CaptureAgentHTTPServer((cfg.bind, cfg.capture_port), self.capture_manager, token)
            except OSError as error:
                raise OfficeClientError(f"拍照端口 {cfg.capture_port} 无法启动；请关闭重复的助手或检查网络地址：{error}") from error
            for server in (self.file_server, self.capture_server):
                if server is not None:
                    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
                    thread.start()
                    self._threads[server] = thread
            firewall = _configure_windows_firewall(cfg.capture_port) if not ipaddress.ip_address(cfg.bind).is_loopback else "LOCAL_ONLY"
            self.details = {
                "status": "LOCAL_ONLY" if ipaddress.ip_address(cfg.bind).is_loopback else "READY",
                "bind": cfg.bind,
                "capture_url": f"http://{cfg.bind}:{self.capture_server.server_port}",
                "capture_port": self.capture_server.server_port,
                "file_port": int(urlsplit(file_url).port),
                "file_base_url": file_url,
                "file_service": "STARTED" if self.file_server else "REUSED_VERIFIED_ROOT",
                "output_root": str(capture_config.output_root),
                "auth_required": cfg.require_token,
                "auth_mode": "BEARER_TOKEN" if cfg.require_token else "TAILSCALE_ONLY",
                "token_file": str(cfg.token_file) if cfg.require_token else None,
                "token_created": created,
                "firewall": firewall,
            }
            log_event(_LOGGER, "office_client_started", **self.details)
            return dict(self.details)
        except Exception:
            self.stop()
            self.details = {"status": "FAILED"}
            raise

    def stop(self) -> None:
        for server in (self.capture_server, self.file_server):
            if server is not None:
                thread = self._threads.get(server)
                if thread is not None and thread.is_alive():
                    server.shutdown()
                    thread.join(timeout=3)
                server.server_close()
        if self.capture_manager is not None:
            self.capture_manager.shutdown(wait=True)
        self._threads.clear()
        self.capture_server = self.file_server = None
        self.capture_manager = None
        self.details = {"status": "STOPPED"}
        log_event(_LOGGER, "office_client_stopped")

    def snapshot(self) -> dict[str, Any]:
        result = dict(self.details)
        if self.capture_manager is not None:
            result["active_job"] = self.capture_manager.health()["active_job"]
        return result

    def run(self, *, manifest: str | Path | None = None,
            report: str | Path | None = None, gui: bool = True) -> int:
        """Start services, then keep them alive for the desktop GUI."""
        details = self.start()
        print(json.dumps(details, ensure_ascii=False, indent=2), flush=True)
        if details.get("status") == "LOCAL_ONLY":
            print("未发现 Tailscale 地址；当前只允许本机访问 8770。", flush=True)
        try:
            if gui:
                from .gui import launch_gui
                launch_gui(manifest, report)
            else:
                threading.Event().wait()
        except KeyboardInterrupt:
            return 0
        finally:
            self.stop()
        return 0


def run_office_client(config: OfficeClientConfig | None = None, **kwargs: Any) -> int:
    """Launch the integrated office services and optional GUI."""
    config = config or OfficeClientConfig()
    manifest = kwargs.pop("manifest", None)
    report = kwargs.pop("report", None)
    gui = bool(kwargs.pop("gui", True))
    if kwargs:
        config = replace(config, **kwargs)
    return OfficeClient(config).run(manifest=manifest, report=report, gui=gui)


__all__ = [
    "OfficeClient",
    "OfficeClientConfig",
    "OfficeClientError",
    "detect_tailscale_address",
    "default_output_root",
    "load_office_config",
    "save_office_config",
    "run_office_client",
]
