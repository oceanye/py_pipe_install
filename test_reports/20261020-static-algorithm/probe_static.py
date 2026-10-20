"""Replay fixed input images; cache measured disparities for algorithm comparisons."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.elevation_dataset import load_elevation_dataset
from pipe_twin.stereo_analyzer import _analysis_config, _calibration_from_manifest, _compute_stereo_depth
from pipe_twin.image_regions import filter_disparity

OUT = ROOT / "outputs/static-algorithm-20261020"
SOURCE = ROOT / "outputs/retry-20261020-131151/active-retry"


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    summary = json.loads((SOURCE / "burst-summary.json").read_text(encoding="utf-8"))
    loaded = load_elevation_dataset(ROOT / summary["manifest"])
    cal = _calibration_from_manifest(loaded["calibration"])
    config = _analysis_config(loaded["manifest"]["analysis"])
    metadata = {"calibration_sha256": hashlib.sha256(
        (ROOT / "outputs/measurement_workbench/calibration_current.json").read_bytes()).hexdigest(),
        "analysis": config, "images": {}}
    images, samples = {}, {}
    for role in ("left", "right"):
        paths = sorted((SOURCE / "rectified").glob(f"*_{role}.png"))
        assert len(paths) == 20
        metadata["images"][role] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        images[role] = [cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR) for p in paths]
        samples[role] = []
    cache = OUT / "cache"
    cache.mkdir(exist_ok=True)
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    for i in range(20):
        path = cache / f"{key[:16]}-{i:02d}.npz"
        if path.exists():
            arrays = np.load(path)
            for role in samples:
                samples[role].append(arrays[role])
        else:
            result = _compute_stereo_depth(images["left"][i], images["right"][i], cal, config)
            arrays = {r: (cal.left.fx * cal.baseline_mm / getattr(result, r+"_depth_mm"))
                      for r in samples}
            np.savez_compressed(path, **arrays)
            for role in samples:
                samples[role].append(arrays[role])
        print("cached", i, flush=True)
    results = {}
    for selection, indices in (("mean", list(range(20))), ("even", list(range(0, 20, 2))),
                               ("odd", list(range(1, 20, 2)))):
        averaged = {r: np.rint(np.mean([images[r][i] for i in indices], axis=0)).astype(np.uint8)
                    for r in images}
        depth = _compute_stereo_depth(averaged["left"], averaged["right"], cal, config)
        results[selection] = {r: cal.left.fx*cal.baseline_mm/getattr(depth, r+"_depth_mm") for r in images}
    results["last"] = {r: samples[r][-1] for r in images}
    # Pin the pre-change reducer so rerunning this probe after deployment still
    # measures the same baseline. It is trusted, committed repository code.
    old_source = subprocess.run(
        ["git", "show", "f6a7a28:pipe_twin/temporal_stereo.py"], cwd=ROOT,
        check=True, capture_output=True).stdout
    old_module = {"__name__": "pipe_twin.temporal_baseline", "__package__": "pipe_twin"}
    exec(compile(old_source, "f6a7a28:pipe_twin/temporal_stereo.py", "exec"), old_module)
    results["old_fusion_before_lr"] = {r: old_module["_fuse_disparities"](samples[r])[0] for r in images}
    metadata["baseline_reducer_commit"] = "f6a7a28"
    np.savez_compressed(cache / "aggregate-disparities.npz",
                        **{f"{kind}_{r}": value for kind, arrays in results.items() for r, value in arrays.items()})
    metrics = []
    # Fixed geometric boxes are only diagnostics, not selected pipe identities.
    boxes = {"left": [(1100, 120, 320, 100), (1100, 325, 320, 80)],
             "right": [(980, 130, 320, 100), (980, 325, 320, 80)]}
    for role in images:
        for x, y, w, h in boxes[role]:
            region = np.zeros(images[role][0].shape[:2], bool)
            region[y:y+h, x:x+w] = True
            for kind, arrays in results.items():
                d = arrays[role]
                support = region & np.isfinite(d)
                kept, audit = filter_disparity(cal.left.fx*cal.baseline_mm/d, support,
                                               getattr(cal, role), cal.baseline_mm, [1, 0])
                metrics.append({"role": role, "box": [x, y, w, h], "method": kind,
                                "valid_fraction": float(np.mean(support[region])),
                                "filter": audit})
    (OUT / "probe.json").write_text(json.dumps({"inputs": metadata, "metrics": metrics},
                                               indent=2, allow_nan=False), encoding="utf-8")
    for item in metrics:
        print(item["role"], item["box"], item["method"], round(item["valid_fraction"], 4),
              item["filter"]["reason"], flush=True)


if __name__ == "__main__":
    main()
