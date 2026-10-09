from __future__ import annotations

import pytest

from pipe_twin.matching_config import normalize_matching_settings


def test_diameter_is_mandatory_and_colour_is_off_by_default():
    settings = normalize_matching_settings()
    assert settings["diameter_filter_enabled"] is True
    assert settings["color_filter_enabled"] is False
    assert settings["color_filter_mode"] == "hint"


def test_colour_hint_and_diameter_tolerances_are_normalized():
    settings = normalize_matching_settings({
        "color_filter_enabled": True,
        "diameter_tolerance_mm": 4,
        "diameter_tolerance_ratio": 0.15,
        "color_delta_lab": 60,
    })
    assert settings["color_filter_enabled"] is True
    assert settings["diameter_tolerance_mm"] == 4.0
    assert settings["diameter_tolerance_ratio"] == 0.15
    assert settings["color_delta_lab"] == 60.0


@pytest.mark.parametrize("payload", [
    {"diameter_filter_enabled": False},
    {"color_filter_mode": "strict"},
    {"diameter_tolerance_ratio": 1.1},
    {"color_delta_lab": 151},
])
def test_matching_policy_rejects_unsupported_or_unsafe_values(payload):
    with pytest.raises(ValueError):
        normalize_matching_settings(payload)
