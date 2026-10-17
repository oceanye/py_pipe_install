"""Offline real-photo replay and numerical controls; never opens a camera."""
from pathlib import Path
import hashlib
import json
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"tests")]
from pipe_twin.calibration_wizard import Rectifier, BoardObservation, read_calibration_diagnostic
from pipe_twin.calibration_quality import holdout_quality
from pipe_twin.temporal_stereo import fuse_stereo_burst, POLICY, POLICY_SHA256, REVISION
from pipe_twin.stereo_analyzer import _analysis_config, _calibration_from_manifest, _compute_stereo_depth, _quality
from pipe_twin.elevation_auto import analyze_elevation_auto_manifest
from pipe_twin.elevation_gui import recognition_diagnostic_lines
from test_temporal_stereo import _burst, _run
from test_temporal_cylinders import numerical_cylinder_comparison

OUT = ROOT/"outputs/temporal-calibration-20261012"
OUT.mkdir(parents=True, exist_ok=True)
tracked = Path(__file__).resolve().parent
hashes = {}
def read(path):
    data = path.read_bytes()
    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return data

source = ROOT/"outputs/elevation_workbench/captures/elevation-20261012-173724-476aa8d1"
for path in (source/"manifest.json", source/"report.json", source/"photos/left.png", source/"photos/right.png"):
    read(path)
started = time.perf_counter()
report = analyze_elevation_auto_manifest(source/"manifest.json", report_output_path=OUT/"single_report.json")
(OUT/"single_diagnostic.txt").write_text("\n".join(recognition_diagnostic_lines(report)), encoding="utf-8")
summary = {"latest_single_capture": {"counts": report["counts"], "observations": len(report["local_surface"]["observations"]),
           "temporal_fusion": report.get("temporal_fusion"), "elapsed_s": time.perf_counter()-started}}
print("LATEST_SINGLE", summary["latest_single_capture"], flush=True)

profile = json.loads(read(ROOT/"outputs/measurement_workbench/workbench_profile.json"))
calibration = profile["calibration_current"]
recipe = profile["rectification_recipe"]
rectifier = Rectifier(recipe)
parsed = _calibration_from_manifest(calibration)
old = ROOT/"outputs/quick-calibration-20261001/capture-070621"
capture = json.loads(read(old/"capture.json"))
frames = [{"capture_id": f"old-board-{i}", "record": pair} for i, pair in enumerate(capture["pairs"])]
for frame in frames:
    for role in ("left", "right"):
        read(Path(frame["record"]["eyes"][role]["path"]))

def images(frame):
    return {role: rectifier.rectify(role, cv2.imdecode(np.frombuffer(Path(frame["record"]["eyes"][role]["path"]).read_bytes(), np.uint8), cv2.IMREAD_COLOR))
            for role in ("left", "right")}

config = _analysis_config({"stereo_matching": {"num_disparities": 256, "preprocessing": "low_light"}})
started = time.perf_counter()
fused = fuse_stereo_burst(frames, load_images=images,
    compute_depth=lambda l, r: _compute_stereo_depth(l, r, parsed, config),
    quality_check=lambda im: _quality(im, config), calibration=parsed,
    progress=lambda i, n: print(f"OLD_REAL_SEQUENCE {i}/{n}", flush=True))
summary["old_real_sequence"] = {"date": "2026-10-01", "not_current_pipe_accuracy_acceptance": True,
    "elapsed_s": time.perf_counter()-started, "depth_audit": fused.audit}
np.savez_compressed(OUT/"old_sequence_depth.npz", left=fused.left_depth_mm, right=fused.right_depth_mm)

pairs = [tuple(BoardObservation(np.asarray(frame["record"]["eyes"][r]["corners"]), 100., (1, 1)) for r in ("left", "right")) for frame in frames]
check = holdout_quality(pairs, recipe=recipe, image_size=(1920, 1080), pattern=(8, 6), square_mm=17.)
check["validation_scope"] = "LEGACY_STATIONARY_BOARD_DIAGNOSTIC_NOT_NEW_MULTIPOSE_CALIBRATION"
summary["legacy_calibration_check"] = {"calibration_id": calibration["calibration_id"],
    "current_profile_chessboard": profile["chessboard"], "K1": recipe["K1"], "D1": recipe["D1"], "D2": recipe["D2"],
    "board_check": check}
diag = ROOT/"outputs/measurement_workbench/calibration_diagnostics/calibration_autosave.npz"
read(diag)
_, metadata = read_calibration_diagnostic(diag)
summary["legacy_calibration_check"]["old_restore_metadata"] = {k: metadata[k] for k in ("square_mm", "pattern_inner_corners", "image_size_px")}

simulated, cal = _burst(missing=True)
result = _run(simulated, cal)
mask = result.left_valid
summary["synthetic_known_depth"] = {"truth_mm": 1000., "single_median_absolute_error_mm": float(np.nanmedian(np.abs(simulated[-1]["depth"].left_depth_mm[mask]-1000))),
    "fused_median_absolute_error_mm": float(np.nanmedian(np.abs(result.left_depth_mm[mask]-1000))),
    "never_observed_patch_stays_invalid": bool(not mask[25:35, 30:40].any()),
    "audit": result.audit["temporal_fusion"]}
surfaces, temporal_audit = numerical_cylinder_comparison()
summary["synthetic_cylinders"] = {"truth_diameters_mm": [24, 36, 50],
    "single": sorted(o["diameter_mm"] for o in surfaces["single"]["observations"]),
    "fused": sorted(o["diameter_mm"] for o in surfaces["fused"]["observations"]),
    "accepted_frames": temporal_audit["accepted_frames"], "not_field_accuracy": True}
summary.update(revision=REVISION, policy=POLICY, policy_sha256=POLICY_SHA256,
    source_hashes=hashes, actual_camera_capture=False,
    source_unchanged=all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == value for p, value in hashes.items()))
assert summary["source_unchanged"]
files = ["temporal_stereo", "calibration_quality", "calibration_wizard", "elevation_auto", "elevation_dataset", "elevation_gui", "capture_gui", "stereo_analyzer"]
summary["code_sha256"] = {f"pipe_twin/{f}.py": hashlib.sha256((ROOT/f"pipe_twin/{f}.py").read_bytes()).hexdigest() for f in files}
for path in (OUT/"evidence.json", tracked/"evidence.json"):
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
print("OLD_BURST", fused.audit["temporal_fusion"]["status"], fused.audit["temporal_fusion"]["accepted_frames"], flush=True)
print("CALIBRATION_CHECK", check["status"], check["vertical_p95_px"], check["cross_reprojection_rms_px"], flush=True)
print("SYNTHETIC_DEPTH", {k:v for k,v in summary["synthetic_known_depth"].items() if k != "audit"}, flush=True)
