"""Process-wide structured logging for the pipe-twin tools.

The log is local JSONL so a run can be inspected or ingested without parsing
human-readable messages.  Secrets and oversized values are deliberately
filtered before they reach disk.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Mapping

LOGGER_NAME = "pipe_twin"
_RUN_ID = uuid.uuid4().hex
_CONFIGURED = False
_LOG_PATH: Path | None = None
_SENSITIVE = ("token", "secret", "password", "passwd", "api_key", "connection_string")


def _safe_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _safe_value(v) for k, v in list(value.items())[:50]}
    if isinstance(value, (list, tuple, set)):
        return [_safe_value(item) for item in list(value)[:50]]
    if isinstance(value, Path):
        return str(value)
    text = str(value)
    return text if len(text) <= 500 else text[:497] + "..."


def _safe_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in fields.items():
        name = str(key)
        if any(part in name.casefold() for part in _SENSITIVE):
            result[name] = "[REDACTED]"
        else:
            result[name] = _safe_value(value)
    return result


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "run_id": _RUN_ID,
            "event": getattr(record, "event", "log"),
            "message": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if isinstance(fields, Mapping):
            payload["fields"] = _safe_fields(fields)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def configure_logging(log_dir: str | Path | None = None) -> Path:
    """Configure rotating JSONL file logging once and return its path."""

    global _CONFIGURED, _LOG_PATH
    if _CONFIGURED and _LOG_PATH is not None:
        return _LOG_PATH
    directory = Path(log_dir or os.environ.get("PIPE_TWIN_LOG_DIR", "logs")).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "pipe_twin.log.jsonl"
    root = logging.getLogger(LOGGER_NAME)
    root.setLevel(logging.INFO)
    root.propagate = False
    file_handler = RotatingFileHandler(
        path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(_JsonFormatter())
    root.addHandler(file_handler)
    console = logging.StreamHandler(sys.stderr)
    console.setLevel(logging.WARNING)
    console.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root.addHandler(console)
    _CONFIGURED = True
    _LOG_PATH = path
    return path


def get_logger(name: str | None = None) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"{LOGGER_NAME}.{name}" if name else LOGGER_NAME)


def log_event(logger: logging.Logger, event: str, message: str | None = None, **fields: Any) -> None:
    """Write a structured event with bounded, redacted context."""

    logger.info(message or event, extra={"event": event, "fields": _safe_fields(fields)})


__all__ = ["configure_logging", "get_logger", "log_event"]
