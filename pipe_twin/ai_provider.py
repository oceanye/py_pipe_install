"""Bounded Kimi Chat Completions transport for independent section assessment.

Documentation: https://platform.kimi.com/docs/api/chat and
https://platform.kimi.com/docs/guide/kimi-k3-1-pro-quickstart (2026-11-01).
No automatic retry: a timed-out paid request may already have been processed.
Cancellation is checked before/after blocking I/O; GUI generation checks must
still reject a cancelled or superseded result before saving it.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import http.client
import json
import re
import socket
import time
import urllib.error
import urllib.request
from uuid import uuid4

from .ai_settings import AISettings, AISettingsError, load_ai_settings, resolve_api_key


MAX_REQUEST_BYTES = 32 * 1024 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


class AIProviderError(ValueError):
    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise AIProviderError("AI_CANCELLED", "AI 评估已取消。")


def _json_object(text):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError()
            result[key] = value
        return result
    def invalid_constant(_value):
        raise ValueError()
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError()
    return value


def request_section_assessment(messages, *, settings: AISettings | None = None, cancel_event=None) -> dict:
    """Send one request; only final JSON content and whitelisted metadata return."""
    _cancelled(cancel_event)
    try:
        settings = settings or load_ai_settings()
        if not isinstance(settings, AISettings):
            raise AISettingsError("AI_SETTINGS_INVALID", "AI 设置类型无效。")
        key = resolve_api_key(settings)
    except AISettingsError as error:
        raise AIProviderError(error.code, str(error)) from None
    if (not isinstance(messages, Sequence) or isinstance(messages, (str, bytes)) or
            not messages or any(not isinstance(m, Mapping) for m in messages)):
        raise AIProviderError("AI_REQUEST_INVALID", "AI 输入消息无效。")
    payload = {"model": settings.model, "messages": list(messages), "stream": False,
               "response_format": {"type": "json_object"}}
    if settings.model.startswith("kimi-k3"):
        payload["max_completion_tokens"] = settings.max_tokens
        payload["reasoning_effort"] = settings.reasoning_effort
    else:
        payload["max_tokens"] = settings.max_tokens
    try:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        raise AIProviderError("AI_REQUEST_INVALID", "AI 输入无法序列化。") from None
    if len(body) > MAX_REQUEST_BYTES:
        raise AIProviderError("AI_REQUEST_TOO_LARGE", "AI 图片输入过大，请缩小评估范围或图片数量。")
    request_id = uuid4().hex
    request = urllib.request.Request(settings.base_url + "/chat/completions", data=body, method="POST",
                                     headers={"Authorization": "Bearer " + key,
                                              "Content-Type": "application/json", "Accept": "application/json"})
    opener = urllib.request.build_opener(_NoRedirect())
    deadline = time.monotonic() + settings.timeout_seconds
    try:
        _cancelled(cancel_event)
        with opener.open(request, timeout=settings.timeout_seconds) as response:
            raw = bytearray()
            read = getattr(response, "read1", response.read)
            while True:
                _cancelled(cancel_event)
                if time.monotonic() > deadline:
                    raise AIProviderError("AI_TIMEOUT", "Kimi 请求超时，请稍后重试。")
                chunk = read(min(65536, MAX_RESPONSE_BYTES + 1 - len(raw)))
                if not chunk:
                    break
                raw.extend(chunk)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise AIProviderError("AI_RESPONSE_TOO_LARGE", "Kimi 返回内容超出允许大小。")
        _cancelled(cancel_event)
    except urllib.error.HTTPError as error:
        status = error.code
        error.close()
        code, message = {
            400: ("AI_BAD_REQUEST", "Kimi 不接受当前请求，请核对模型 ID 和图片输入格式。"),
            401: ("AI_AUTHENTICATION_FAILED", "Kimi 鉴权失败，请检查 key 与服务地区。"),
            403: ("AI_PERMISSION_DENIED", "Kimi 拒绝访问，请检查模型权限、账户额度与地区。"),
            429: ("AI_RATE_LIMITED", "Kimi 请求限流或额度不足，请稍后重试并检查账户。"),
        }.get(status, ("AI_HTTP_ERROR", f"Kimi 服务请求失败（HTTP {status}）。"))
        raise AIProviderError(code, message) from None
    except (TimeoutError, socket.timeout):
        raise AIProviderError("AI_TIMEOUT", "Kimi 请求超时，请稍后重试。") from None
    except urllib.error.URLError as error:
        timeout = isinstance(error.reason, (TimeoutError, socket.timeout))
        raise AIProviderError("AI_TIMEOUT" if timeout else "AI_CONNECTION_FAILED",
                              "Kimi 请求超时，请稍后重试。" if timeout else "无法连接 Kimi 服务，请检查网络。") from None
    except (OSError, http.client.HTTPException):
        raise AIProviderError("AI_CONNECTION_FAILED", "无法读取 Kimi 响应，请检查网络。") from None
    finally:
        request.remove_header("Authorization")
    try:
        # Never persist an accidental credential echo, even inside a valid JSON response.
        text = raw.decode("utf-8")
        if key in text:
            raise ValueError()
        envelope = _json_object(text)
        choices = envelope.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise ValueError()
        choice = choices[0]
        if choice.get("finish_reason") == "length":
            raise AIProviderError("AI_OUTPUT_TRUNCATED", "Kimi 输出达到 token 上限，请增加输出上限后重试。")
        if choice.get("finish_reason") != "stop":
            raise ValueError()
        message = choice.get("message")
        if not isinstance(message, dict) or message.get("tool_calls") or message.get("refusal"):
            raise ValueError()
        assessment = _json_object(message.get("content"))
        if envelope.get("model") not in (None, settings.model):
            raise AIProviderError("AI_MODEL_MISMATCH", "Kimi 返回的模型与请求不一致，结果未接纳。")
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        if isinstance(error, AIProviderError):
            raise
        raise AIProviderError("AI_INVALID_RESPONSE", "Kimi 未返回完整有效的 JSON 结果，请重试。") from None
    finally:
        key = ""
    usage = envelope.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}
    result = {"response": assessment, "provider": "kimi", "model": settings.model, "request_id": request_id,
              "usage": {name: usage[name] for name in ("prompt_tokens", "completion_tokens", "total_tokens", "cached_tokens")
                        if type(usage.get(name)) is int and 0 <= usage[name] <= 10**9}}
    provider_id = envelope.get("id")
    if isinstance(provider_id, str) and re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", provider_id):
        result["provider_request_id"] = provider_id
    _cancelled(cancel_event)
    return result
