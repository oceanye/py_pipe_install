from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from pipe_twin.capture_gui import catalog_from_model
from pipe_twin.elevation_zones import ElevationZoneError, normalize_zone_settings
from pipe_twin.model_zones import (
    ALGORITHM,
    catalog_fingerprint,
    propose_model_zones,
    zone_model_specs,
)


def _spec(pipe_id: str, x: float, *, axis: tuple[float, float, float] = (0, 0, 1), diameter: float = 40.0,
          start: float = 0.0, length: float = 500.0) -> dict:
    direction = np.asarray(axis, dtype=float)
    direction /= np.linalg.norm(direction)
    origin = np.array([x, 0.0, start])
    return {"pipe_id": pipe_id, "nominal_diameter_mm": diameter,
            "centerline_world_mm": [origin.tolist(), (origin + direction * length).tolist()]}


def test_parallel_model_proposals_are_deterministic_and_keep_all_components():
    specs = [_spec("P2", 90), _spec("P1", 0), _spec("P3", 500, start=800)]
    first = propose_model_zones(specs, maximum_gap_mm=60)
    second = propose_model_zones(list(reversed(specs)), maximum_gap_mm=60)
    assert first == second
    assert [zone["model_pipe_ids"] for zone in first] == [["P1", "P2"], ["P3"]]
    assert all(zone["source"] == "stl_parallel" and not zone["confirmed"] and not zone["enabled"]
               for zone in first)
    assert all(zone["roi_rect_px"] is None and zone["roi_source"] == "unmapped" for zone in first)
    assert all(zone["proposal"]["algorithm"] == ALGORITHM for zone in first)


def test_different_directions_do_not_share_a_model_zone():
    specs = [_spec("P1", 0), _spec("P2", 100, axis=(0, 0, 1)), _spec("P3", 0, axis=(1, 0, 0))]
    zones = propose_model_zones(specs, maximum_gap_mm=100)
    assert sorted(sorted(zone["model_pipe_ids"]) for zone in zones) == [["P1", "P2"], ["P3"]]


def test_stl_fixture_produces_one_connected_candidate_with_model_fingerprint():
    model = Path(__file__).resolve().parents[1] / "test_model" / "管道布置.stl"
    specs, skipped = catalog_from_model(model, stl_unit="millimeter")
    assert not skipped and len(specs) == 12
    zones = propose_model_zones(specs)
    assert len(zones) == 1
    assert zones[0]["model_pipe_ids"] == sorted(spec["pipe_id"] for spec in specs)
    assert zones[0]["model_catalog_sha256"] == catalog_fingerprint(specs)


def test_model_draft_can_be_saved_unmapped_but_only_confirmed_roi_can_run():
    specs = [_spec("P1", 0), _spec("P2", 100), _spec("P3", 200)]
    draft = propose_model_zones(specs)[0]
    settings = normalize_zone_settings({"scope": "full_frame", "zones": [draft]})
    assert settings["zones"][0]["roi_rect_px"] is None
    with pytest.raises(ElevationZoneError):
        normalize_zone_settings({"scope": "zones", "zones": [{**draft, "enabled": True}]}, image_size=(640, 480))
    confirmed = copy.deepcopy(draft)
    confirmed.update({"enabled": True, "confirmed": True, "roi_source": "user", "roi_rect_px": [0, 0, 640, 480]})
    saved = normalize_zone_settings({"scope": "zones", "zones": [confirmed]}, image_size=(640, 480))
    assert saved["zones"][0]["model_pipe_ids"] == ["P1", "P2", "P3"]


def test_zone_model_specs_rejects_changed_catalog_and_returns_bound_subset():
    specs = [_spec("P1", 0), _spec("P2", 100), _spec("P3", 200)]
    zone = propose_model_zones(specs)[0]
    subset = zone_model_specs(zone, specs)
    assert [item["pipe_id"] for item in subset] == ["P1", "P2", "P3"]
    changed = copy.deepcopy(specs)
    changed[0]["nominal_diameter_mm"] = 41.0
    with pytest.raises(ValueError, match="模型已改变"):
        zone_model_specs(zone, changed)
