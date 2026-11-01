"""Transport acceptance, cancellation and failures with no external requests."""
from io import BytesIO
import http.client
import json
import socket
from threading import Event
import urllib.error

import pytest

from pipe_twin import ai_provider as provider
from pipe_twin.ai_settings import AISettings


MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "Return a JSON object with point_assessments."},
                                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}]}]


def envelope(content=None, **extra):
    return dict({"id": "cmpl-synthetic", "model": "kimi-k3.1-pro",
                 "choices": [{"finish_reason": "stop", "message": {
                     "content": content if content is not None else '{"point_assessments":[]}',
                     "reasoning_content": "Do not parse this as the assessment"}}],
                 "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
                           "unexpected": "discard"}}, **extra)


@pytest.fixture
def fake_transport(monkeypatch):
    captured = []
    response_data = [json.dumps(envelope()).encode()]
    errors = []

    class Opener:
        def open(self, request, timeout):
            captured.append({"request": request, "auth": request.get_header("Authorization"), "timeout": timeout,
                             "body": json.loads(request.data), "url": request.full_url})
            if errors:
                raise errors[0]
            return BytesIO(response_data[0])

    monkeypatch.setattr(provider, "resolve_api_key", lambda _settings: "synthetic-transport-key")
    monkeypatch.setattr(provider.urllib.request, "build_opener", lambda *handlers: Opener())
    return captured, response_data, errors


def test_json_image_request_and_whitelisted_response(fake_transport):
    captured, _, _ = fake_transport
    result = provider.request_section_assessment(MESSAGES, settings=AISettings())
    request = captured[0]
    assert request["url"] == "https://api.moonshot.cn/v1/chat/completions"
    assert request["auth"] == "Bearer synthetic-transport-key"
    assert request["request"].get_header("Authorization") is None
    assert request["body"]["max_completion_tokens"] == 8192
    assert request["body"]["reasoning_effort"] == "low"
    assert request["body"]["response_format"] == {"type": "json_object"}
    assert request["body"]["messages"] == MESSAGES
    assert result["response"] == {"point_assessments": []}
    assert result["provider_request_id"] == "cmpl-synthetic"
    assert result["usage"] == {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
    assert "reasoning_content" not in json.dumps(result)
    assert "synthetic-transport-key" not in json.dumps(result)


@pytest.mark.parametrize("status,code", [(400,"AI_BAD_REQUEST"), (401,"AI_AUTHENTICATION_FAILED"),
                                        (403,"AI_PERMISSION_DENIED"), (429,"AI_RATE_LIMITED"),
                                        (302,"AI_HTTP_ERROR"), (500,"AI_HTTP_ERROR")])
def test_http_errors_ignore_body_and_do_not_retry(fake_transport, status, code):
    captured, _, errors = fake_transport
    errors.append(urllib.error.HTTPError("https://api.moonshot.cn/v1/chat/completions",status,
                                        "synthetic-transport-key", {}, BytesIO(b"private content")))
    with pytest.raises(provider.AIProviderError) as error:
        provider.request_section_assessment(MESSAGES, settings=AISettings())
    assert error.value.code == code
    assert "synthetic-transport-key" not in str(error.value)
    assert "private content" not in str(error.value)
    assert len(captured) == 1
    assert captured[0]["request"].get_header("Authorization") is None


@pytest.mark.parametrize("content", ['not json', '[]', '{"x":NaN}', '{"x":1,"x":2}',
                                     '{"echo":"synthetic-transport-key"}'])
def test_invalid_final_json_not_accepted(fake_transport, content):
    _, data, _ = fake_transport
    data[0] = json.dumps(envelope(content)).encode()
    with pytest.raises(provider.AIProviderError) as error:
        provider.request_section_assessment(MESSAGES, settings=AISettings())
    assert error.value.code == "AI_INVALID_RESPONSE"
    assert "synthetic-transport-key" not in str(error.value)


def test_truncation_and_model_mismatch_are_not_partial_success(fake_transport):
    _, data, _ = fake_transport
    value = envelope()
    value["choices"][0]["finish_reason"] = "length"
    data[0] = json.dumps(value).encode()
    with pytest.raises(provider.AIProviderError) as error:
        provider.request_section_assessment(MESSAGES, settings=AISettings())
    assert error.value.code == "AI_OUTPUT_TRUNCATED"
    data[0] = json.dumps(envelope(model="kimi-k3")).encode()
    with pytest.raises(provider.AIProviderError) as error:
        provider.request_section_assessment(MESSAGES, settings=AISettings())
    assert error.value.code == "AI_MODEL_MISMATCH"


def test_cancel_before_send_and_after_response(fake_transport, monkeypatch):
    captured, data, _ = fake_transport
    cancel = Event()
    cancel.set()
    with pytest.raises(provider.AIProviderError) as error:
        provider.request_section_assessment(MESSAGES, settings=AISettings(), cancel_event=cancel)
    assert error.value.code == "AI_CANCELLED"
    assert not captured
    cancel.clear()

    class CancelAfterReading(BytesIO):
        def read1(self, size):
            chunk = super().read1(size)
            cancel.set()
            return chunk
    class Opener:
        def open(self, request, timeout):
            return CancelAfterReading(data[0])
    monkeypatch.setattr(provider.urllib.request, "build_opener", lambda *handlers: Opener())
    with pytest.raises(provider.AIProviderError) as error:
        provider.request_section_assessment(MESSAGES, settings=AISettings(), cancel_event=cancel)
    assert error.value.code == "AI_CANCELLED"


def test_transport_has_bounded_request_and_response(fake_transport, monkeypatch):
    captured, data, _ = fake_transport
    monkeypatch.setattr(provider, "MAX_REQUEST_BYTES", 10)
    with pytest.raises(provider.AIProviderError) as error:
        provider.request_section_assessment(MESSAGES, settings=AISettings())
    assert error.value.code == "AI_REQUEST_TOO_LARGE"
    assert not captured
    monkeypatch.setattr(provider, "MAX_REQUEST_BYTES", 4096)
    monkeypatch.setattr(provider, "MAX_RESPONSE_BYTES", 10)
    with pytest.raises(provider.AIProviderError) as error:
        provider.request_section_assessment(MESSAGES, settings=AISettings())
    assert error.value.code == "AI_RESPONSE_TOO_LARGE"


@pytest.mark.parametrize("failure", [socket.timeout("private message"),
                                     urllib.error.URLError(socket.timeout("private message")),
                                     urllib.error.URLError("private message"),
                                     http.client.IncompleteRead(b"private message", 32)])
def test_network_failure_is_sanitized(fake_transport, failure):
    _, _, errors = fake_transport
    errors.append(failure)
    with pytest.raises(provider.AIProviderError) as error:
        provider.request_section_assessment(MESSAGES, settings=AISettings())
    assert error.value.code in ("AI_TIMEOUT", "AI_CONNECTION_FAILED")
    assert "private message" not in str(error.value)


def test_redirect_handler_never_forwards_authorization():
    handler = provider._NoRedirect()
    assert handler.redirect_request(None,None,302,"",{},"https://elsewhere.invalid") is None
