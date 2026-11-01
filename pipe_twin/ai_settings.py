"""Machine-local Kimi preferences and current-user Windows DPAPI credentials.

Only non-secret preferences belong in AISettings. Scene packages and reports
must never contain this file, its encrypted credential, or the resolved key.
MOONSHOT_API_KEY takes precedence over the optional Windows credential.
"""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import re
import sys

from .artifact_io import atomic_write_text


OFFICIAL_BASE_URLS = ("https://api.moonshot.cn/v1", "https://api.moonshot.ai/v1")
DEFAULT_MODEL = "kimi-k3.1-pro"
SCHEMA_VERSION = "pipe-twin-ai-settings-v1"


class AISettingsError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class AISettings:
    base_url: str = OFFICIAL_BASE_URLS[0]
    model: str = DEFAULT_MODEL
    timeout_seconds: float = 180.0
    max_tokens: int = 8192
    reasoning_effort: str = "low"

    def __post_init__(self):
        if not isinstance(self.base_url, str):
            raise AISettingsError("AI_SETTINGS_INVALID", "Kimi 服务地址无效。")
        base = self.base_url.rstrip("/")
        if base not in OFFICIAL_BASE_URLS:
            raise AISettingsError("AI_ENDPOINT_NOT_ALLOWED", "仅支持 Kimi 官方 HTTPS 服务地址。")
        object.__setattr__(self, "base_url", base)
        if not isinstance(self.model, str) or not re.fullmatch(r"kimi-[a-z0-9][a-z0-9.-]{0,79}", self.model):
            raise AISettingsError("AI_SETTINGS_INVALID", "请输入有效的 Kimi 模型 ID。")
        if (type(self.timeout_seconds) not in (int, float) or
                not math.isfinite(self.timeout_seconds) or not 5 <= self.timeout_seconds <= 600):
            raise AISettingsError("AI_SETTINGS_INVALID", "AI 超时需为 5～600 秒。")
        if type(self.max_tokens) is not int or not 256 <= self.max_tokens <= 32768:
            raise AISettingsError("AI_SETTINGS_INVALID", "AI 输出上限需为 256～32768 tokens。")
        if self.reasoning_effort not in ("low", "high", "max"):
            raise AISettingsError("AI_SETTINGS_INVALID", "AI 推理强度需为 low、high 或 max。")


def settings_path() -> Path:
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
        return root / "PipeTwin" / "ai-settings.json"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "pipe-twin" / "ai-settings.json"


def _read_document(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        if path.stat().st_size > 65536:
            raise ValueError()
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
            raise ValueError()
        if set(document) - {"schema_version", "settings", "credential_dpapi"}:
            raise ValueError()
        return document
    except (OSError, UnicodeError, ValueError):
        raise AISettingsError("AI_SETTINGS_UNREADABLE", "无法读取本机 AI 设置，请重新保存设置。") from None


def load_ai_settings(*, path: Path | None = None) -> AISettings:
    document = _read_document(Path(path) if path is not None else settings_path())
    values = document.get("settings", {})
    if not isinstance(values, dict):
        raise AISettingsError("AI_SETTINGS_INVALID", "本机 AI 设置格式无效。")
    try:
        return AISettings(**values)
    except TypeError:
        raise AISettingsError("AI_SETTINGS_INVALID", "本机 AI 设置包含不支持的参数。") from None


def _validated_key(value: str) -> str:
    if not isinstance(value, str):
        raise AISettingsError("AI_CREDENTIAL_INVALID", "AI 凭据格式无效。")
    key = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]{10,512}", key):
        raise AISettingsError("AI_CREDENTIAL_INVALID", "AI 凭据格式无效。")
    return key


def _dpapi(data: bytes, *, protect: bool) -> bytes:
    """DPAPI current-user scope, no UI and no machine-wide protection."""
    if sys.platform != "win32":
        raise AISettingsError("AI_CREDENTIAL_STORAGE_UNAVAILABLE", "此平台请使用 MOONSHOT_API_KEY 环境变量。")
    import ctypes
    from ctypes import wintypes

    class Blob(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]

    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    source, target = Blob(len(data), buffer), Blob()
    function = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    function.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.POINTER(Blob),
                         ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    function.restype = wintypes.BOOL
    try:
        if not function(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)):
            raise AISettingsError("AI_CREDENTIAL_STORAGE_ERROR", "无法访问当前 Windows 用户的 AI 凭据，请重新保存 key。")
        return ctypes.string_at(target.pbData, target.cbData)
    finally:
        ctypes.memset(buffer, 0, len(data))
        if target.pbData:
            ctypes.memset(target.pbData, 0, target.cbData)
            kernel32.LocalFree(target.pbData)


def save_ai_settings(settings: AISettings, api_key: str | None = None, *, path: Path | None = None) -> AISettings:
    """Save preferences; a blank key keeps the stored credential unchanged."""
    if not isinstance(settings, AISettings):
        raise AISettingsError("AI_SETTINGS_INVALID", "AI 设置类型无效。")
    if api_key is not None and not isinstance(api_key, str):
        raise AISettingsError("AI_CREDENTIAL_INVALID", "AI 凭据格式无效。")
    destination = Path(path) if path is not None else settings_path()
    document = _read_document(destination)
    if api_key is not None and api_key.strip():
        key = _validated_key(api_key)
        encrypted = _dpapi(key.encode("utf-8"), protect=True)
        document["credential_dpapi"] = base64.b64encode(encrypted).decode("ascii")
    document.update(schema_version=SCHEMA_VERSION, settings=asdict(settings))
    try:
        atomic_write_text(destination, json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    except OSError:
        raise AISettingsError("AI_SETTINGS_WRITE_FAILED", "无法保存本机 AI 设置。") from None
    return settings


def resolve_api_key(settings: AISettings | None = None, *, path: Path | None = None) -> str:
    """Resolve only at request time; callers must not log the returned value."""
    if settings is not None and not isinstance(settings, AISettings):
        raise AISettingsError("AI_SETTINGS_INVALID", "AI 设置类型无效。")
    if os.environ.get("MOONSHOT_API_KEY", "").strip():
        return _validated_key(os.environ["MOONSHOT_API_KEY"])
    document = _read_document(Path(path) if path is not None else settings_path())
    encrypted = document.get("credential_dpapi")
    if not encrypted:
        raise AISettingsError("AI_PROVIDER_NOT_CONFIGURED", "请在 AI 设置中配置 Kimi key，或设置 MOONSHOT_API_KEY。")
    try:
        protected = base64.b64decode(encrypted, validate=True)
        return _validated_key(_dpapi(protected, protect=False).decode("utf-8"))
    except AISettingsError:
        raise
    except (UnicodeError, ValueError, TypeError):
        raise AISettingsError("AI_CREDENTIAL_STORAGE_ERROR", "无法读取本机 AI 凭据，请在当前用户下重新保存 key。") from None


def clear_saved_api_key(*, path: Path | None = None) -> None:
    destination = Path(path) if path is not None else settings_path()
    document = _read_document(destination)
    if "credential_dpapi" in document:
        document.pop("credential_dpapi")
        try:
            atomic_write_text(destination, json.dumps(document, ensure_ascii=False, indent=2) + "\n")
        except OSError:
            raise AISettingsError("AI_SETTINGS_WRITE_FAILED", "无法删除本机 AI 凭据。") from None


def configured_status(*, path: Path | None = None) -> dict:
    """Safe UI metadata only. The key is never returned or cached."""
    settings = AISettings()
    try:
        settings = load_ai_settings(path=path)
        resolve_api_key(settings, path=path)
    except AISettingsError as error:
        return {"configured": False, "credential_source": "none", "model": settings.model,
                "base_url": settings.base_url, "error_code": error.code}
    return {"configured": True,
            "credential_source": "environment" if os.environ.get("MOONSHOT_API_KEY", "").strip() else "windows_dpapi",
            "model": settings.model, "base_url": settings.base_url, "error_code": None}
