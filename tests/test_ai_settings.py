"""Local preferences never persist plain credentials or require a paid API."""
import json
import os
import sys

import pytest

from pipe_twin import ai_settings as settings


@pytest.fixture(autouse=True)
def no_real_credential(monkeypatch):
    monkeypatch.delenv("MOONSHOT_API_KEY", raising=False)


def test_defaults_and_missing_key_status(tmp_path):
    path = tmp_path / "settings.json"
    assert settings.load_ai_settings(path=path).model == "kimi-k3.1-pro"
    status = settings.configured_status(path=path)
    assert status["configured"] is False
    assert status["error_code"] == "AI_PROVIDER_NOT_CONFIGURED"
    assert not path.exists()


def test_preferences_and_key_never_share_dataclass(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setattr(settings, "_dpapi", lambda value, *, protect: b"encrypted-for-test")
    preferences = settings.AISettings(timeout_seconds=60, max_tokens=4096)
    settings.save_ai_settings(preferences, "synthetic-test-key", path=path)
    assert settings.load_ai_settings(path=path) == preferences
    assert "synthetic-test-key" not in path.read_text()
    assert "api_key" not in repr(preferences)
    original = json.loads(path.read_text())["credential_dpapi"]
    settings.save_ai_settings(preferences, " ", path=path)
    assert json.loads(path.read_text())["credential_dpapi"] == original
    settings.clear_saved_api_key(path=path)
    assert "credential_dpapi" not in json.loads(path.read_text())
    assert settings.load_ai_settings(path=path) == preferences


def test_environment_has_precedence_and_is_not_persisted(tmp_path, monkeypatch):
    path = tmp_path / "settings.json"
    monkeypatch.setenv("MOONSHOT_API_KEY", "synthetic-environment-key")
    assert settings.resolve_api_key(path=path) == "synthetic-environment-key"
    assert settings.configured_status(path=path)["credential_source"] == "environment"
    settings.save_ai_settings(settings.AISettings(), path=path)
    assert "synthetic-environment-key" not in path.read_text()


@pytest.mark.skipif(sys.platform != "win32", reason="Current-user Windows DPAPI")
def test_real_dpapi_round_trip_in_isolated_temp_directory(tmp_path):
    path = tmp_path / "settings.json"
    secret = "synthetic-dpapi-test-only"
    settings.save_ai_settings(settings.AISettings(), secret, path=path)
    assert settings.resolve_api_key(path=path) == secret
    assert secret not in path.read_text()
    assert settings.configured_status(path=path)["credential_source"] == "windows_dpapi"


def test_non_windows_requires_environment_for_key_storage(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    path = tmp_path / "settings.json"
    with pytest.raises(settings.AISettingsError) as error:
        settings.save_ai_settings(settings.AISettings(), "synthetic-test-key", path=path)
    assert error.value.code == "AI_CREDENTIAL_STORAGE_UNAVAILABLE"
    assert not path.exists()
    settings.save_ai_settings(settings.AISettings(), path=path)
    assert path.exists()


@pytest.mark.parametrize("url", ["http://api.moonshot.cn/v1", "https://evil.invalid/v1",
                                "https://api.moonshot.cn.evil.invalid/v1",
                                "https://user:password@api.moonshot.cn/v1",
                                "https://api.moonshot.cn/v1?api_key=never-echo-this",
                                "https://["])
def test_endpoint_allowlist_never_echoes_input(url):
    with pytest.raises(settings.AISettingsError) as error:
        settings.AISettings(base_url=url)
    assert url not in str(error.value)
    assert error.value.code == "AI_ENDPOINT_NOT_ALLOWED"


@pytest.mark.parametrize("values", [{"timeout_seconds": float("nan")}, {"timeout_seconds": True},
                                   {"timeout_seconds": 601}, {"max_tokens": 1},
                                   {"max_tokens": True}, {"reasoning_effort": "medium"},
                                   {"model": "bad\r\nheader"}])
def test_invalid_settings(values):
    with pytest.raises(settings.AISettingsError):
        settings.AISettings(**values)


def test_malformed_config_error_does_not_reveal_content(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text('{"api_key":"synthetic-do-not-echo"}')
    status = settings.configured_status(path=path)
    assert status["error_code"] == "AI_SETTINGS_UNREADABLE"
    assert "synthetic-do-not-echo" not in json.dumps(status)


def test_encrypted_config_cannot_be_replaced_with_plaintext(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"schema_version":settings.SCHEMA_VERSION,
                               "credential_dpapi":"not-base64!", "settings":{}}))
    with pytest.raises(settings.AISettingsError) as error:
        settings.resolve_api_key(path=path)
    assert error.value.code == "AI_CREDENTIAL_STORAGE_ERROR"
