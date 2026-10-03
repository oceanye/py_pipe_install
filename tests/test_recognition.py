from __future__ import annotations

import pytest

from pipe_twin.elevation_dataset import normalize_registration_settings
from pipe_twin.recognition import (
    RecognitionBackendError,
    available_recognizers,
    recognize_local_pipes,
    register_recognizer,
    unregister_recognizer,
)


def test_builtin_recognizers_are_explicit_and_stable():
    names = available_recognizers()
    assert names == ("auto", "cylinder", "geometry_only", "parallel_strip")


def test_custom_recognizer_can_be_selected_without_changing_elevation_flow():
    def detector(_left, _right, _depth, _calibration, _specs):
        return {"observations": [], "audit": {"test_backend": True}}

    register_recognizer("test_backend", detector)
    try:
        settings = normalize_registration_settings({"local_observation_mode": "test_backend"})
        assert settings["local_observation_mode"] == "test_backend"
        surface = recognize_local_pipes("test_backend", None, None, None, None, [])
        assert surface["observations"] == []
        assert surface["audit"]["recognizer_backend"] == "test_backend"
        assert surface["audit"]["recognizer_backend_used"] == "test_backend"
    finally:
        unregister_recognizer("test_backend")


def test_unknown_recognizer_fails_closed_with_available_names():
    with pytest.raises(ValueError, match="local_observation_mode"):
        normalize_registration_settings({"local_observation_mode": "missing_backend"})
    with pytest.raises(RecognitionBackendError, match="未注册"):
        recognize_local_pipes("missing_backend", None, None, None, None, [])


def test_backend_contract_rejects_non_surface_result():
    register_recognizer("bad_backend", lambda *_args: [])
    try:
        with pytest.raises(RecognitionBackendError, match="未返回对象"):
            recognize_local_pipes("bad_backend", None, None, None, None, [])
    finally:
        unregister_recognizer("bad_backend")
