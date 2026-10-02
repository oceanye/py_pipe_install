"""Start or reuse a detached, Tailscale-bound capture-file service.

Importing this module never opens a port. Explicitly enabled legacy launchers
can retain file access after closing the workbench. The integrated office
client reuses this service when it already serves the configured directory;
the GUI must not launch a second service.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import ipaddress
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.request import ProxyHandler, Request, build_opener

from .office_client import _EvidenceHandler


_REPO = Path(__file__).resolve().parents[1]
_TAILSCALE_NETWORK = ipaddress.ip_network("100.64.0.0/10")
_IDENTITY_FILE = ".pipe_twin_file_share.json"


def default_config_path() -> Path:
    return _REPO / "outputs/measurement_workbench/remote_files.json"


def _address(value: str, *, allow_loopback: bool = False) -> str:
    address = ipaddress.IPv4Address(value)
    if address not in _TAILSCALE_NETWORK and not (allow_loopback and address.is_loopback):
        raise ValueError("File access requires a Tailscale IPv4 address; public and wildcard binds are refused")
    return str(address)


def discover_tailscale_ip() -> str:
    """Read the adapter address without requiring an administrator session."""
    addresses = {
        record[4][0]
        for record in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
        if ipaddress.IPv4Address(record[4][0]) in _TAILSCALE_NETWORK
    }
    if len(addresses) == 1:
        return addresses.pop()
    executable = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Tailscale/tailscale.exe"
    command = str(executable) if executable.is_file() else "tailscale"
    result = subprocess.run(
        [command, "ip", "-4"], capture_output=True, text=True, timeout=4,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    if result.returncode == 0:
        candidates = result.stdout.strip().splitlines()
        if len(candidates) == 1:
            return _address(candidates[0])
    raise ValueError("A unique local Tailscale IPv4 address is not available")


def load_config(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(config, dict) or type(config.get("enabled")) is not bool:
        raise ValueError("remote_files.json must contain an enabled boolean")
    if not config["enabled"]:
        return None
    host = config.get("bind_host", "auto")
    host = discover_tailscale_ip() if host == "auto" else _address(host, allow_loopback=True)
    control_ip = _address(config["control_ip"], allow_loopback=ipaddress.ip_address(host).is_loopback)
    port = config.get("port", 8765)
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("The file service port must be an integer between 1 and 65535")
    directory = Path(config["directory"]).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Capture directory does not exist: {directory}")
    return {"host": host, "control_ip": control_ip, "port": port, "directory": directory}


@contextlib.contextmanager
def _startup_lock(path: Path) -> Iterator[None]:
    """Serialize GUI launches; the OS releases this lock if a launcher exits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        deadline = time.monotonic() + 8
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Another workbench is starting the file service")
                time.sleep(0.1)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _identity(directory: Path) -> str:
    path = directory / _IDENTITY_FILE
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"identity": uuid.uuid4().hex}, handle)
    value = json.loads(path.read_text(encoding="utf-8"))["identity"]
    if not isinstance(value, str) or not value:
        raise ValueError("Invalid file-service directory identity")
    return value


def _probe(url: str, identity: str) -> tuple[bool, int | None]:
    # Tailscale traffic must bypass a system Internet proxy.
    opener = build_opener(ProxyHandler({}))
    try:
        with opener.open(Request(url + _IDENTITY_FILE), timeout=0.7) as response:
            value = json.loads(response.read(2048))
            pid = response.headers.get("X-Pipe-Twin-Pid")
            return isinstance(value, dict) and value.get("identity") == identity, int(pid) if pid else None
    except (OSError, ValueError):
        return False, None


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False


def _ensure(config: dict[str, Any]) -> dict[str, Any]:
    host, port, directory = config["host"], config["port"], config["directory"]
    url = f"http://{host}:{port}/"
    identity = _identity(directory)
    common = {"url": url, "directory": str(directory), "control_ip": config["control_ip"]}
    ready, pid = _probe(url, identity)
    if ready:
        return {"status": "REUSED", "pid": pid, **common}
    if _port_open(host, port):
        raise ValueError(f"Port {port} is occupied by a service that does not serve {directory}")

    log_directory = directory / "service_logs"
    log_directory.mkdir(exist_ok=True)
    stdout_path = log_directory / "remote_files.stdout.log"
    stderr_path = log_directory / "remote_files.stderr.log"
    python = Path(sys.executable)
    if python.name.lower() == "pythonw.exe":
        python = python.with_name("python.exe")
    options: dict[str, Any] = {"close_fds": True}
    if os.name == "nt":
        options["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        startup = subprocess.STARTUPINFO()
        startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startup.wShowWindow = subprocess.SW_HIDE
        options["startupinfo"] = startup
    else:
        options["start_new_session"] = True
    with stdout_path.open("ab") as stdout, stderr_path.open("ab") as stderr:
        process = subprocess.Popen(
            [str(python), "-u", "-m", "pipe_twin.remote_files", "serve",
             "--bind", host, "--port", str(port), "--directory", str(directory),
             "--control-ip", config["control_ip"]],
            cwd=_REPO, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, **options,
        )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        ready, pid = _probe(url, identity)
        if ready:
            return {"status": "STARTED", "pid": pid or process.pid,
                    "stdout_path": str(stdout_path), "stderr_path": str(stderr_path), **common}
        if process.poll() is not None:
            raise OSError(f"File service exited with code {process.returncode}; see {stderr_path}")
        time.sleep(0.1)
    raise TimeoutError(f"File service did not respond in time; see {stderr_path}")


def ensure_remote_files(config_path: str | Path | None = None) -> dict[str, Any]:
    """Return a status without preventing the camera GUI from opening on failure."""
    path = Path(config_path) if config_path is not None else default_config_path()
    try:
        config = load_config(path)
        if config is None:
            return {"status": "DISABLED"}
        with _startup_lock(path.with_suffix(".lock")):
            result = _ensure(config)
            path.with_suffix(".state.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            return result
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        return {"status": "FAILED", "error": str(error)}


class _CaptureFilesHandler(_EvidenceHandler):
    def __init__(self, *args: Any, control_ip: str, bind_host: str, **kwargs: Any) -> None:
        self.allowed_sources = {control_ip, bind_host}
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:
        if self.client_address[0] not in self.allowed_sources:
            self.send_error(403, "This capture service is restricted to its configured controller")
            return
        super().do_GET()

    def do_HEAD(self) -> None:
        if self.client_address[0] not in self.allowed_sources:
            self.send_error(403, "This capture service is restricted to its configured controller")
            return
        super().do_HEAD()

    def end_headers(self) -> None:
        self.send_header("X-Pipe-Twin-Pid", str(os.getpid()))
        super().end_headers()


class _CaptureFilesServer(ThreadingHTTPServer):
    # Windows SO_REUSEADDR can let the integrated client bind the same port
    # again instead of probing/reusing this server. Own our listener exclusively.
    allow_reuse_address = os.name != "nt"

    def server_bind(self) -> None:
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("--bind", required=True)
    parser.add_argument("--port", required=True, type=int)
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--control-ip", required=True)
    args = parser.parse_args()
    host = _address(args.bind, allow_loopback=True)
    control_ip = _address(args.control_ip, allow_loopback=ipaddress.ip_address(host).is_loopback)
    directory = args.directory.resolve()
    if not directory.is_dir():
        parser.error("The capture directory does not exist")
    handler = functools.partial(
        _CaptureFilesHandler, directory=str(directory), control_ip=control_ip, bind_host=host,
    )
    with _CaptureFilesServer((host, args.port), handler) as server:
        print(json.dumps({"event": "file_service_started", "pid": os.getpid(),
                          "url": f"http://{host}:{args.port}/", "directory": str(directory),
                          "control_ip": control_ip}), flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
