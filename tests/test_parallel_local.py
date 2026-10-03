from __future__ import annotations

import sys

import pytest

from pipe_twin.elevation_auto import analyze_elevation_auto_groups
from pipe_twin.elevation_dataset import normalize_registration_settings
from pipe_twin.elevation_registration import register_elevation
from pipe_twin.parallel_local import extract_parallel_local_pipes

sys.path.insert(0, "tests")
from test_elevation_auto import _groups  # noqa: E402


def test_parallel_local_strip_uses_partial_visible_width_and_keeps_diameter_primary():
    calibration, specs, groups = _groups()
    surface = extract_parallel_local_pipes(
        groups[0]["left"], groups[0]["right"], groups[0]["depth"], calibration, specs
    )
    assert surface["audit"]["observation_type"] == "PARALLEL_LOCAL_STRIP"
    assert len(surface["observations"]) == 3
    assert all(item["parallel_local"] for item in surface["observations"])
    assert all(item["diameter_source"].startswith("PROJECTED_LOCAL_WIDTH") for item in surface["observations"])
    measured = sorted(item["diameter_mm"] for item in surface["observations"])
    assert measured == pytest.approx([24, 36, 50], abs=5.0)
    result = register_elevation(specs, surface["observations"])
    assert result["status"] == "MATCHED"
    assert {row["pipe_id"] for row in result["matches"]} == {"P1", "P2", "P3"}


def test_parallel_local_mode_reports_same_installed_subset_when_a_region_is_occluded():
    calibration, specs, groups = _groups(occluded=True)
    result = analyze_elevation_auto_groups(
        groups, calibration=calibration, pipe_specs=specs,
        registration_settings={"local_observation_mode": "parallel_strip"},
    )
    assert result["registration"]["status"] == "MATCHED"
    assert result["counts"] == {"INSTALLED": 3, "NOT_INSTALLED": 0, "UNKNOWN": 1}
    assert result["scene_inventory"]["observed_local_strip_count"] == 3


def test_registration_settings_persist_parallel_local_mode_contract():
    assert normalize_registration_settings({"local_observation_mode": "parallel_strip"})["local_observation_mode"] == "parallel_strip"
    assert normalize_registration_settings({})["local_observation_mode"] == "auto"
    with pytest.raises(ValueError, match="local_observation_mode"):
        normalize_registration_settings({"local_observation_mode": "anything_else"})
