from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from pipe_twin.elevation_auto import _merge_zone_rows, analyze_elevation_auto_zones
from pipe_twin.elevation_dataset import elevation_history_compatible, load_elevation_dataset
from pipe_twin.elevation_zones import (
    ElevationZoneError,
    crop_calibration_for_zone,
    crop_stereo_group_for_zone,
    normalize_zone_settings,
)
from pipe_twin.model_zones import propose_model_zones
from pipe_twin.stereo_analyzer import _calibration_from_manifest
from test_elevation_auto import _groups, _package
from test_elevation_dataset import _calibration


def _zone(rect=(80, 60, 240, 180)):
    return {"zone_id": "Z01", "label": "上排", "enabled": True,
            "coordinate_space": "rectified_left", "roi_rect_px": list(rect)}


def test_zone_policy_is_bounded_and_allows_group_overlap():
    settings = normalize_zone_settings({"scope": "zones", "right_roi_padding_px": 96,
                                        "zones": [_zone(), {**_zone((160, 100, 240, 180)), "zone_id": "Z02"}]},
                                       image_size=(640, 480))
    assert settings["scope"] == "zones"
    assert settings["right_roi_padding_px"] == 96
    assert len(settings["zones"]) == 2
    with pytest.raises(ElevationZoneError):
        normalize_zone_settings({"scope": "zones", "zones": []}, image_size=(640, 480))
    with pytest.raises(ElevationZoneError):
        normalize_zone_settings({"scope": "zones", "zones": [{**_zone(), "roi_rect_px": [0, 0, 20, 20]}]},
                                image_size=(640, 480))


def test_zone_crop_shifts_intrinsics_and_pixel_boxes():
    calibration = _calibration_from_manifest(_calibration())
    depth = SimpleNamespace(
        left_depth_mm=np.full((480, 640), 1000, np.float32),
        right_depth_mm=np.full((480, 640), 1000, np.float32),
        left_valid=np.ones((480, 640), bool), right_valid=np.ones((480, 640), bool),
        audit={"status": "VALID"},
    )
    group = {"left": np.zeros((480, 640, 3), np.uint8), "right": np.zeros((480, 640, 3), np.uint8),
             "depth": depth, "capture_id": "CAP-1", "captured_at": "2026-10-04T10:00:00+08:00"}
    cropped, offsets = crop_stereo_group_for_zone(group, calibration, _zone((80, 60, 240, 180)), right_padding_px=64)
    cropped_calibration = crop_calibration_for_zone(calibration, _zone((80, 60, 240, 180)), right_padding_px=64)
    assert cropped["left"].shape[:2] == (180, 240)
    assert cropped["right"].shape[:2] == (180, 368)
    assert offsets == {"left_x": 80, "left_y": 60, "right_x": 16, "right_y": 60,
                      "width": 240, "height": 180, "right_width": 368}
    assert cropped_calibration.left.rectified_intrinsic[0, 2] == pytest.approx(240.0)
    assert cropped_calibration.right.rectified_intrinsic[0, 2] == pytest.approx(304.0)


def test_full_width_zone_reuses_automatic_geometry_and_qualifies_observations():
    calibration, specs, groups = _groups()
    result = analyze_elevation_auto_zones(
        groups, calibration=calibration, pipe_specs=specs,
        scope_settings={"scope": "zones", "zones": [{**_zone((0, 0, 640, 480)), "label": "全幅"}]},
        registration_settings={"camera_side_world": [1, 0, 0]},
    )
    assert result["scope"] == "zones"
    assert result["counts"] == {"INSTALLED": 3, "NOT_INSTALLED": 0, "UNKNOWN": 1}
    assert all(item["observation_id"].startswith("Z01:") for item in result["local_surface"]["observations"])
    assert result["capture_audit"]["groups"][0]["zone_ids"] == ["Z01"]
    assert result["zone_registration_audit"]["registrations"][0]["camera_view"]["camera_side_world"] == [1, 0, 0]


def test_ground_check_uses_full_image_once_even_when_ground_is_outside_pipe_zone(monkeypatch):
    from pipe_twin import ground_check
    calibration, specs, groups = _groups()
    region = {'image_size': [640, 480], 'rect_px': [100, 340, 400, 130]}
    calls = []
    def check(group, cal, selection, *, section_stages):
        calls.append((group, cal, selection, section_stages))
        return {'status': 'TEST_FULL_FRAME'}
    monkeypatch.setattr(ground_check, 'check_ground', check)
    result = analyze_elevation_auto_zones(groups, calibration=calibration, pipe_specs=specs,
        scope_settings={'scope': 'zones', 'zones': [_zone((100, 0, 440, 280))]},
        registration_settings={'ground_region': region})
    assert len(calls) == 1
    assert calls[0][0] is groups[-1] and calls[0][1] is calibration
    assert calls[0][2]['rect_px'] == region['rect_px']
    # Plane is solved before pipe sections, so the same full-frame result can
    # constrain each cropped section without refitting the ground.
    assert calls[0][3] == ()
    assert len(result['ground_check']['corrections']) == len(result['section_evaluations'])
    assert result['ground_check']['status'] == 'TEST_FULL_FRAME'


@pytest.mark.parametrize("backend", ["cylinder", "geometry_only", "parallel_strip", "image_guided", "auto"])
def test_partial_zone_uses_real_backends_with_unequal_crop_widths(backend):
    calibration, specs, groups = _groups()
    result = analyze_elevation_auto_zones(
        groups, calibration=calibration, pipe_specs=specs,
        scope_settings={"scope": "zones", "right_roi_padding_px": 128,
                        "zones": [_zone((120, 0, 440, 480))]},
        registration_settings={"local_observation_mode": backend},
    )
    assert len(result["local_surface"]["observations"]) == 3
    for observation in result["local_surface"]["observations"]:
        # Published image bounds return to full-frame coordinates.
        assert observation["left_region_px"][0] >= 120
        assert 0 <= observation["right_region_px"][0] < 640


def test_cropped_backprojection_preserves_metric_points_and_rejects_mismatched_arrays():
    from pipe_twin.local_surface import _points, _to_left, _validate_arrays, LocalSurfaceError
    from dataclasses import replace

    calibration, _, groups = _groups()
    zone = _zone((120, 40, 440, 350))
    cropped, offsets = crop_stereo_group_for_zone(groups[0], calibration, zone, right_padding_px=128)
    cropped_cal = crop_calibration_for_zone(calibration, zone, right_padding_px=128)
    _validate_arrays(cropped["left"], cropped["right"], cropped["depth"], cropped_cal)
    for role in ("left", "right"):
        camera = getattr(calibration, role)
        original_mask = np.zeros((camera.height, camera.width), bool)
        x, y = offsets[f"{role}_x"], offsets[f"{role}_y"]
        width = offsets["width" if role == "left" else "right_width"]
        original_mask[y:y+350, x:x+width] = True
        original_depth = groups[0]["depth"]
        truth = _to_left(_points(getattr(original_depth, f"{role}_depth_mm"),
                                getattr(original_depth, f"{role}_valid"), camera, original_mask), calibration, role)
        cropped_depth = cropped["depth"]
        actual = _to_left(_points(getattr(cropped_depth, f"{role}_depth_mm"),
                                 getattr(cropped_depth, f"{role}_valid"), getattr(cropped_cal, role),
                                 np.ones(cropped[role].shape[:2], bool)), cropped_cal, role)
        np.testing.assert_allclose(actual, truth, atol=1e-8)
    broken = copy.deepcopy(cropped["depth"])
    broken.right_valid = broken.left_valid
    with pytest.raises(LocalSurfaceError, match="dimensions"):
        _validate_arrays(cropped["left"], cropped["right"], broken, cropped_cal)
    for row, col, offset in [(0, 0, 5), (1, 2, 5)]:
        intrinsic = cropped_cal.right.intrinsic.copy()
        intrinsic[row, col] += offset
        projection = cropped_cal.right.projection_matrix.copy()
        projection[row, col] += offset
        broken_cal = replace(cropped_cal, right=replace(cropped_cal.right, intrinsic=intrinsic,
                                                       projection_matrix=projection))
        with pytest.raises(LocalSurfaceError, match="rectified"):
            _validate_arrays(cropped["left"], cropped["right"], cropped["depth"], broken_cal)


def test_confirmed_model_zone_limits_registration_catalog(monkeypatch):
    import pipe_twin.elevation_auto as elevation_auto

    calibration, specs, groups = _groups()
    proposal = propose_model_zones(specs)[0]
    proposal.update({"enabled": True, "confirmed": True,
                     "roi_source": "user", "roi_rect_px": [0, 0, 640, 480]})
    calls = []
    original = elevation_auto.analyze_elevation_auto_groups

    def wrapped(cropped_groups, **kwargs):
        calls.append([item["pipe_id"] for item in kwargs["pipe_specs"]])
        return original(cropped_groups, **kwargs)

    monkeypatch.setattr(elevation_auto, "analyze_elevation_auto_groups", wrapped)
    result = analyze_elevation_auto_zones(
        groups, calibration=calibration, pipe_specs=specs,
        scope_settings={"scope": "zones", "zones": [proposal]},
    )
    assert calls == [[spec["pipe_id"] for spec in specs]]
    assert result["local_surface"]["audit"]["zones"][0]["source"] == "stl_parallel"
    assert result["capture_audit"]["groups"][0]["zone_audits"][0]["source"] == "stl_parallel"


def test_zone_scope_manifest_roundtrip(tmp_path):
    path, _ = _package(tmp_path, scope_settings={
        "scope": "zones", "zones": [{**_zone((0, 0, 640, 480)), "label": "全幅"}],
    })
    loaded = load_elevation_dataset(path)
    assert loaded["scope_settings"]["scope"] == "zones"
    assert loaded["scope_settings"]["zones"][0]["zone_id"] == "Z01"


def test_scope_change_invalidates_history(tmp_path):
    path, settings = _package(tmp_path)
    assert not elevation_history_compatible(
        path, settings["calibration"], settings["pipe_specs"],
        model_path=settings["model_path"], stl_unit="millimeter",
        registration_settings=settings.get("registration_settings"),
        analysis_settings=settings.get("analysis_settings"),
        scope_settings={"scope": "zones", "zones": [{**_zone((0, 0, 640, 480)), "label": "全幅"}]},
    )


def test_zone_merge_marks_installed_negative_conflict_unknown():
    specs = [{"pipe_id": "P1", "nominal_diameter_mm": 40.0, "color_srgb": "#FF0000"}]
    base = {"pipe_id": "P1", "measurement": {}, "current_evidence": {}}
    installed = {**base, "installation_state": "INSTALLED", "reason_codes": ["OBSERVED"]}
    negative = {**base, "installation_state": "NOT_INSTALLED", "reason_codes": ["EMPTY"]}
    rows = _merge_zone_rows([{"zone_id": "Z01", "pipes": [installed]},
                             {"zone_id": "Z02", "pipes": [negative]}], specs)
    assert rows[0]["installation_state"] == "UNKNOWN"
    assert rows[0]["reason_codes"] == ["ZONE_RESULT_CONFLICT"]
