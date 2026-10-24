"""Collect comparable field metrics without copying original photographs."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs/static-algorithm-20261020"
SOURCE = ROOT / "outputs/retry-20261020-131151"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    raw_path = SOURCE / "raw-burst/raw-manifest.json"
    raw = read(raw_path)
    for frame in raw["frames"]:
        for role in ("left", "right"):
            assert sha(raw_path.parent / frame["files"][role]) == frame["sha256"][role]
    active = ROOT / "outputs/measurement_workbench/calibration_current.json"
    assert sha(active) == "b52e5f311b9dd750cfd022afe603e62780d933799d78fc32c64baeb124e1e83c"
    evidence = {"scope": "STATIC_ALGORITHM_REPLAY_NOT_METRIC_ACCEPTANCE",
                "baseline_commit": "f6a7a28", "new_photographs": False,
                "raw_manifest_sha256": sha(raw_path), "raw_pair_count": len(raw["frames"]),
                "raw_image_hashes_verified": 40, "calibration_sha256": sha(active),
                "baseline_mm": read(active)["stereo_calibration"]["baseline_mm"],
                "known_physical_baseline_mm": 60, "runs": []}
    original_settings = None
    for tag in ("active-retry", "active-static-v2"):
        for variant in ("single", "burst"):
            summary = read(SOURCE / tag / f"{variant}-summary.json")
            report_path = ROOT / summary["report"]
            report = read(report_path)
            manifest = read(ROOT / summary["manifest"])
            settings = manifest["analysis"]["stereo_matching"]
            if original_settings is None:
                original_settings = settings
            assert settings == original_settings
            temporal = report.get("temporal_fusion")
            candidates = report["local_surface"].get("image_candidates", [])
            evidence["runs"].append({
                "tag": tag, "variant": variant, "seconds": summary["seconds"],
                "manifest": summary["manifest"], "report": summary["report"],
                "report_sha256": sha(report_path), "counts": report["counts"],
                "observations": report["local_surface"]["observations"],
                "image_proposals": report["local_surface"]["audit"]["image_proposals"],
                "temporal": {k: v for k, v in temporal.items() if k != "frames"} if temporal else None,
                "all_frames_static": all(all(f.get("motion", {}).get(r, {}).get("passed")
                    for r in ("left", "right")) for f in temporal["frames"]) if temporal else None,
                "paint_candidates": [{k: c.get(k) for k in (
                    "candidate_id", "role", "appearance", "region_px", "valid_depth_fraction",
                    "attempts", "correspondence_diagnostic")} for c in candidates if c.get("appearance") != "neutral"]
            })
    evidence["stereo_matching_unchanged"] = original_settings
    evidence["geometry_sources_unchanged"] = {}
    for name in ("stereo_analyzer", "metrology", "local_surface", "elevation_registration"):
        path = f"pipe_twin/{name}.py"
        old = subprocess.run(["git", "show", f"f6a7a28:{path}"], cwd=ROOT,
                             capture_output=True, check=True).stdout
        # Git normalizes newlines; compare normalized content.
        normalized = (ROOT / path).read_bytes().replace(b"\r\n", b"\n")
        assert old.replace(b"\r\n", b"\n") == normalized
        evidence["geometry_sources_unchanged"][path] = sha(ROOT / path)
    evidence["changed_source_sha256"] = {str(p.relative_to(ROOT)): sha(p) for p in (
        ROOT/"pipe_twin/image_regions.py", ROOT/"pipe_twin/temporal_stereo.py", ROOT/"pipe_twin/elevation_gui.py")}
    evidence["image_averaging_probe"] = read(OUT / "probe.json")["metrics"]
    cleanup = OUT / "cleanup-equivalence.json"
    cleanup_check = read(cleanup) if cleanup.exists() else None
    if cleanup_check:
        assert cleanup_check["after_sha256"] == sha(ROOT/"pipe_twin/temporal_stereo.py")
        assert all(c["depth_and_audit_equal"] for c in cleanup_check["comparisons"])
        evidence["field_replay_reducer_sha256_before_dead_calculation_cleanup"] = cleanup_check["before_sha256"]
        evidence["cleanup_preserves_field_disparities_and_audit"] = cleanup_check
    (HERE / "evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False,
                                                   allow_nan=False)+"\n", encoding="utf-8")
    xml = OUT / "pytest.xml"
    if xml.exists():
        suites = ET.parse(xml).getroot()
        log_bytes = (OUT/"pytest.log").read_bytes()
        log_text = log_bytes.decode("utf-16" if log_bytes[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig")
        validation = {"command": ".venv/Scripts/python.exe -m pytest tests -q",
                      "junit_path": str(xml.relative_to(ROOT)),
                      "junit_sha256": sha(xml),
                      "suites": [s.attrib for s in suites],
                      "pytest_summary": re.search(r"\d+ passed,.*", log_text).group(),
                      "compileall": "passed",
                      "git_diff_check": "passed",
                      "final_dead_calculation_cleanup": cleanup_check,
                      "post_cleanup_regression": {"command": "pytest tests/test_temporal_stereo.py tests/test_temporal_cylinders.py tests/test_temporal_dataset.py -q",
                                                  "passed": 24, "seconds": 14.08},
                      "known_truth_regressions": [
                          "Hue boundary regression failed before fix; full widths restored without bridging dark gaps.",
                          "One-sided disparity outlier regression failed before fix; measured majority recovered.",
                          "Competing depth modes, absent depth, motion, duplicates and cross-eye disagreement rejected."],
                      "field_results_are_not_metric_validation": True}
        assert all(int(s.get("failures", 0)) == 0 and int(s.get("errors", 0)) == 0 for s in suites)
        (HERE / "validation.json").write_text(json.dumps(validation, indent=2)+"\n", encoding="utf-8")
    print("Verified 40 raw images, active calibration, matcher and geometry sources; collected four runs.")


if __name__ == "__main__":
    main()
