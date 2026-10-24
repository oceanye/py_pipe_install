import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from pipe_twin.elevation_dataset import create_elevation_dataset, load_elevation_dataset
from pipe_twin.dxf_elevation import catalog_from_dxf
from pipe_twin.stereo_analyzer import analyze_stereo_capture
from test_elevation_dataset import _calibration, MODEL_PATH


def _arguments(tmp_path, count=3):
    rng = np.random.default_rng(891)
    base = rng.integers(40, 210, (480, 640, 3), dtype=np.uint8)
    records = []
    for i in range(count):
        left = np.clip(base.astype(float)+rng.normal(0, .7, base.shape), 0, 255).astype(np.uint8)
        right = np.roll(left, -40, axis=1)
        record = {"sha256": {}, "timestamp_sources": {r: "HOST_SYSTEM_CLOCK" for r in ("left", "right")}}
        for role, image in (("left", left), ("right", right)):
            data = cv2.imencode(".png", image)[1].tobytes()
            path = tmp_path / f"{i}_{role}.png"
            path.write_bytes(data)
            record[f"{role}_path"] = path
            record[f"{role}_time"] = f"2026-10-12T10:00:{i:02d}+08:00"
            record["sha256"][role] = hashlib.sha256(data).hexdigest()
        records.append(record)
    specs, _ = catalog_from_dxf(MODEL_PATH, unitless_unit="millimeter")
    return dict(output_root=tmp_path/"out", calibration=_calibration(),
        **{key: records[-1][key] for key in ("left_path", "right_path", "left_time", "right_time")},
        timestamp_sources=records[-1]["timestamp_sources"], temporal_frames=records,
        pipe_specs=specs, pair_confirmed=True, model_path=MODEL_PATH, stl_unit="millimeter",
        analysis_settings={"stereo_matching": {"num_disparities": 64}})


def test_burst_package_reopens_and_survives_history_copy(tmp_path):
    args = _arguments(tmp_path)
    first = create_elevation_dataset(**args)
    loaded = load_elevation_dataset(first)
    group = loaded["manifest"]["capture"]["capture_groups"][-1]
    assert len(group["temporal_frames"]) == 3
    assert len(loaded["manifest"]["capture"]["capture_groups"]) == 1
    args.pop("temporal_frames")
    second = create_elevation_dataset(**args, previous_manifest=first)
    loaded_again = load_elevation_dataset(second)
    assert len(loaded_again["manifest"]["capture"]["capture_groups"]) == 1
    for frame in loaded_again["manifest"]["capture"]["capture_groups"][0]["temporal_frames"]:
        for role in ("left", "right"):
            assert (second.parent / frame["views"][role]["path"]).is_file()


def test_burst_manifest_runs_actual_sgbm_once_per_frame_and_one_state_capture(tmp_path):
    path = create_elevation_dataset(**_arguments(tmp_path))
    original = path.read_bytes()
    report = analyze_stereo_capture(path, report_output_path=tmp_path/"report.json")
    assert path.read_bytes() == original
    assert report["temporal_fusion"]["status"] == "VALID"
    assert report["temporal_fusion"]["accepted_frames"] == 3
    assert report["capture_audit"]["count"] == 1
    assert report["counts"]["NOT_INSTALLED"] == 0
    assert len(report["inputs"]) >= 8


@pytest.mark.parametrize("problem", ["hash", "timestamp", "reference", "settings", "count", "span", "identical_eyes"])
def test_burst_invalid_provenance_is_rejected(tmp_path, problem):
    args = _arguments(tmp_path)
    frames = args["temporal_frames"]
    if problem == "hash":
        frames[0]["sha256"]["left"] = "0"*64
    elif problem == "timestamp":
        frames[1]["left_time"] = frames[0]["left_time"]
        frames[1]["right_time"] = frames[0]["right_time"]
    elif problem == "reference":
        args["left_path"] = frames[0]["left_path"]
    elif problem == "settings":
        frames[0]["camera_capture_provenance"] = {"left": {"capture_exposure_status": "AUTO"}}
    elif problem == "span":
        frames[0]["left_time"] = frames[0]["right_time"] = "2026-10-12T09:59:00+08:00"
    elif problem == "identical_eyes":
        frames[0]["right_path"] = frames[0]["left_path"]
        frames[0]["sha256"]["right"] = frames[0]["sha256"]["left"]
    else:
        frames.pop(0)
    with pytest.raises(ValueError):
        create_elevation_dataset(**args)


def test_tampering_any_nested_frame_is_rejected_on_load(tmp_path):
    path = create_elevation_dataset(**_arguments(tmp_path))
    manifest = json.loads(path.read_text(encoding="utf-8"))
    view = manifest["capture"]["capture_groups"][0]["temporal_frames"][0]["views"]["left"]
    (path.parent/view["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError):
        load_elevation_dataset(path)
