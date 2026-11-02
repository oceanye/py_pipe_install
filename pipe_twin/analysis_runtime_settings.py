"""Per-machine execution preferences; never embedded in a scene package."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import ipaddress
import json
import os
from pathlib import Path
import re
import sys

from .artifact_io import atomic_write_text


MODE_LABELS = {"local": "本地运算", "server": "服务器端", "client": "前端"}
SCHEMA_VERSION = "pipe-twin-analysis-runtime-v1"


@dataclass(frozen=True)
class AnalysisRuntimeSettings:
    mode: str = "local"
    address: str = "127.0.0.1"
    port: int = 8771

    def __post_init__(self):
        if self.mode not in MODE_LABELS:
            raise ValueError("请选择服务器端、前端或本地运算。")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("计算端口须为 1–65535 的整数。")
        if not isinstance(self.address, str):
            raise ValueError("请填写服务器 IP 或主机名。")
        address = self.address.strip().strip("[]")
        try:
            ipaddress.ip_address(address)
        except ValueError:
            if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9.-]{0,251}[A-Za-z0-9])?", address):
                raise ValueError("地址只填写 IP 或主机名；计算端口单独填写。") from None
        object.__setattr__(self, "address", address)

    @property
    def base_url(self) -> str:
        host = f"[{self.address}]" if ":" in self.address else self.address
        return f"http://{host}:{self.port}"


def settings_path() -> Path:
    override = os.environ.get("PIPE_TWIN_RUNTIME_SETTINGS")
    if override:
        return Path(override)
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "PipeTwin"
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "pipe-twin"
    return root / "analysis-runtime.json"


def load_runtime_settings(*, path: Path | None = None) -> AnalysisRuntimeSettings:
    source = Path(path) if path is not None else settings_path()
    if not source.exists():
        return AnalysisRuntimeSettings()
    try:
        if source.stat().st_size > 65536:
            raise ValueError()
        document = json.loads(source.read_text(encoding="utf-8"))
        if document.get("schema_version") != SCHEMA_VERSION:
            raise ValueError()
        return AnalysisRuntimeSettings(**document["settings"])
    except (TypeError, KeyError, AttributeError, UnicodeError, ValueError) as error:
        raise ValueError("本机运行设置无效，请在设置中重新保存。") from error


def save_runtime_settings(settings: AnalysisRuntimeSettings, *, path: Path | None = None) -> Path:
    validated = AnalysisRuntimeSettings(**asdict(settings))
    destination = Path(path) if path is not None else settings_path()
    atomic_write_text(destination, json.dumps({"schema_version": SCHEMA_VERSION,
                                             "settings": asdict(validated)}, ensure_ascii=False, indent=2))
    return destination
