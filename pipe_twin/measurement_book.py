"""Reference samples, scoped diameter corrections, and reviewable GUI summaries."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Mapping
from uuid import uuid4

import numpy as np

from .metrology import measurement_settings
from .pipeline import atomic_write_text


KINDS = {"diameter": "外径", "center_distance": "中心距", "clear_gap": "净距"}
USES = {"reference": "登记对照", "calibration": "校正样本", "validation": "独立验证"}
STATE_LABELS = {"INSTALLED": "已安装", "NOT_INSTALLED": "未安装", "UNKNOWN": "待确认"}


def scope_id(manifest: Mapping) -> str:
    calibration = manifest.get("stereo_calibration", {})
    return hashlib.sha256(json.dumps(calibration, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def capture_signature(manifest: Mapping) -> str:
    groups = manifest.get("capture", {}).get("capture_groups", [])
    if not groups:
        return ""
    return ":".join(str(groups[-1].get("views", {}).get(role, {}).get("sha256", "")) for role in ("left", "right"))


def empty_book() -> dict:
    return {"schema_version": "1.0", "kind": "pipe-measurement-workbook", "samples": [], "settings": measurement_settings()}


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{label} 必须是{'正的' if positive else ''}有限数值")
    return float(value)


def validate_book(book: Any) -> dict:
    if not isinstance(book, dict) or book.get("kind") != "pipe-measurement-workbook" or book.get("schema_version") != "1.0":
        raise ValueError("请选择测量工作簿 JSON（版本 1.0）")
    if not isinstance(book.get("samples"), list):
        raise ValueError("工作簿缺少 samples 列表")
    ids = set()
    for sample in book["samples"]:
        if not isinstance(sample, dict):
            raise ValueError("样本必须是对象")
        for key in ("sample_id", "pipe_id", "scope_id", "section_id", "created_at"):
            if not isinstance(sample.get(key), str) or not sample[key].strip():
                raise ValueError(f"样本缺少 {key}")
        if sample["sample_id"] in ids:
            raise ValueError("样本编号重复")
        ids.add(sample["sample_id"])
        if sample.get("kind") not in KINDS or sample.get("use_for") not in USES:
            raise ValueError("样本类型或用途无效")
        if sample["use_for"] == "calibration" and sample["kind"] != "diameter":
            raise ValueError("二次校正样本仅支持外径")
        for key in ("notes", "capture_id", "capture_signature", "pipe_id_b", "calibration_id"):
            if not isinstance(sample.get(key, ""), str):
                raise ValueError(f"{key} 必须是文本")
        _number(sample.get("reference_mm"), "实测值", positive=sample["kind"] != "clear_gap")
        if sample["kind"] != "diameter" and (not sample.get("pipe_id_b") or sample["pipe_id_b"] == sample["pipe_id"]):
            raise ValueError("间距需要两根不同的管件")
        raw = sample.get("raw_mm")
        if raw is not None:
            _number(raw, "自动测量值", positive=sample["kind"] != "clear_gap")
            if sample.get("raw_source") != "stereo_local" or not sample.get("capture_signature"):
                raise ValueError("自动值必须带有双目照片来源")
    result = json.loads(json.dumps(book, allow_nan=False))
    for sample in result["samples"]:
        sample.setdefault("raw_mm", None)
        for key in ("notes", "capture_id", "capture_signature", "pipe_id_b", "calibration_id"):
            sample.setdefault(key, "")
    result["settings"] = measurement_settings(book.get("settings"))
    return result


def load_book(path: str | Path) -> dict:
    return validate_book(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def save_book(path: str | Path, book: Mapping) -> Path:
    valid = validate_book(dict(book))
    return atomic_write_text(path, json.dumps(valid, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def new_sample(manifest: Mapping, *, pipe_id: str, kind: str, reference_mm: float,
               section_id: str, use_for: str = "reference", pipe_id_b: str = "",
               raw_mm: float | None = None, notes: str = "", measurement_settings_id: str = "") -> dict:
    groups = manifest.get("capture", {}).get("capture_groups", [])
    result = {
        "sample_id": uuid4().hex[:12], "created_at": datetime.now(timezone.utc).isoformat(),
        "pipe_id": pipe_id, "pipe_id_b": pipe_id_b, "kind": kind, "reference_mm": reference_mm,
        "raw_mm": raw_mm, "raw_source": "stereo_local" if raw_mm is not None else None,
        "scope_id": scope_id(manifest), "calibration_id": manifest.get("stereo_calibration", {}).get("calibration_id", ""),
        "capture_id": groups[-1].get("capture_id", "") if groups else "",
        "capture_signature": capture_signature(manifest), "section_id": section_id,
        "measurement_settings_id": measurement_settings_id, "use_for": use_for, "notes": notes,
    }
    validate_book(empty_book() | {"samples": [result]})
    return result


def diameter_correction(samples: list[dict], scope: str, tolerance_mm: float) -> dict:
    eligible = [s for s in samples if s["scope_id"] == scope and s["kind"] == "diameter"
                and s.get("raw_mm") is not None and s["use_for"] == "calibration"]
    # Re-entering a photograph/pipe/section must not increase statistical weight.
    unique = {(s["capture_signature"], s["pipe_id"], s["section_id"]): s for s in eligible}
    train = list(unique.values())
    if not train:
        return {"status": "NO_SAMPLES", "label": "尚无可用校正样本", "offset_mm": None, "validation_count": 0}
    offset = median(s["reference_mm"] - s["raw_mm"] for s in train)
    residual = max(abs(s["raw_mm"] + offset - s["reference_mm"]) for s in train)
    low, high = min(s["raw_mm"] for s in train), max(s["raw_mm"] for s in train)
    signatures = {s["capture_signature"] for s in train}
    validation = [s for s in samples if s["scope_id"] == scope and s["kind"] == "diameter"
                  and s["use_for"] == "validation" and s.get("raw_mm") is not None
                  and s["capture_signature"] not in signatures and low - tolerance_mm <= s["raw_mm"] <= high + tolerance_mm]
    validation = list({(s["capture_signature"], s["pipe_id"], s["section_id"]): s for s in validation}.values())
    errors = [s["raw_mm"] + offset - s["reference_mm"] for s in validation]
    passed = bool(len({s["capture_signature"] for s in validation}) >= 3 and max(map(abs, errors)) <= tolerance_mm)
    return {
        "status": "INCONSISTENT" if residual > tolerance_mm else "VALIDATED_SAMPLES" if passed else "TRIAL",
        "label": "样本不一致，暂停校正" if residual > tolerance_mm else "独立样本对照通过" if passed else "试校正，待独立样本验证",
        "offset_mm": offset, "raw_range_mm": [low, high], "training_count": len(train),
        "training_max_residual_mm": residual, "validation_count": len(validation),
        "validation_max_error_mm": max(map(abs, errors)) if errors else None,
        "validation_mae_mm": sum(map(abs, errors)) / len(errors) if errors else None,
    }


def corrected_diameter(raw_mm: float | None, correction: Mapping, tolerance_mm: float) -> float | None:
    if raw_mm is None or correction["status"] not in {"TRIAL", "VALIDATED_SAMPLES"}:
        return None
    lo, hi = correction["raw_range_mm"]
    value = raw_mm + correction["offset_mm"]
    return value if lo - tolerance_mm <= raw_mm <= hi + tolerance_mm and value > 0 else None


def validated_metric(report: Mapping | None, manifest: Mapping, dashboard: Mapping) -> tuple[dict, str]:
    """Discard malformed/stale metric payloads without compromising the GUI."""
    if not dashboard.get("binding_valid") or not report or not report.get("local_measurements"):
        return {}, "NO_BOUND_METRIC_REPORT"
    metric = report["local_measurements"]
    groups = manifest.get("capture", {}).get("capture_groups", [])
    try:
        if (not isinstance(metric, dict) or metric.get("schema_version") != "1.0"
                or metric.get("coordinate_frame") != "CAD_WORLD_MM"
                or not groups or metric.get("capture_id") != groups[-1].get("capture_id")
                or metric.get("calibration_id") != manifest.get("stereo_calibration", {}).get("calibration_id")):
            raise ValueError("metric provenance")
        rows = metric["pipes"]
        measurement_settings(metric.get("settings"))
        expected = {row["pipe_id"] for row in dashboard["pipes"]}
        if len(rows) != len(expected) or {row["pipe_id"] for row in rows} != expected:
            raise ValueError("metric identities")
        measured = set()
        for row in rows:
            if row["status"] not in {"MEASURED", "UNKNOWN"}:
                raise ValueError("metric status")
            if not isinstance(row.get("reason_codes"), list) or not all(isinstance(c, str) for c in row["reason_codes"]):
                raise ValueError("metric reasons")
            if row["status"] == "MEASURED":
                measured.add(row["pipe_id"])
                _number(row["diameter_mm"], "diameter", positive=True)
                _number(row["camera_depth_mm"], "depth", positive=True)
                if _number(row.get("view_position_difference_mm", 0), "view position difference") < 0:
                    raise ValueError("negative view difference")
                for key, shape in (("center_world_mm", (3,)), ("observed_segment_world_mm", (2, 3)), ("axis_direction_world", (3,))):
                    value = np.asarray(row[key], dtype=float)
                    if value.shape != shape or not np.isfinite(value).all():
                        raise ValueError("metric coordinates")
                if set(row["views"]) != {"left", "right"}:
                    raise ValueError("metric views")
            for view in row.get("views", {}).values():
                for key in ("roi_xywh", "support_bbox_xywh"):
                    box = view.get(key)
                    if box is not None and (len(box) != 4 or any(type(v) not in (int, float) or not math.isfinite(v) for v in box)):
                        raise ValueError("metric box")
        seen = set()
        for pair in metric["pairs"]:
            identities = frozenset((pair["pipe_id_a"], pair["pipe_id_b"]))
            if len(identities) != 2 or not identities <= expected or identities in seen:
                raise ValueError("pair identities")
            seen.add(identities)
            if pair["status"] not in {"MEASURED", "UNKNOWN"}:
                raise ValueError("pair status")
            if pair["status"] == "MEASURED":
                if not identities <= measured:
                    raise ValueError("unmeasured pair")
                for key in ("center_distance_mm", "clear_gap_mm", "depth_delta_b_minus_a_mm", "depth_order_threshold_mm"):
                    _number(pair[key], key)
        if len(seen) != len(expected) * (len(expected) - 1) // 2:
            raise ValueError("incomplete pairs")
    except (ValueError, TypeError, KeyError, OverflowError, AttributeError):
        return {}, "INVALID_LOCAL_REPORT"
    return metric, ""


def build_measurement_summary(manifest: Mapping, report: Mapping | None, dashboard: Mapping,
                              book: Mapping, tolerance_mm: float = 1.0) -> dict:
    tolerance_mm = _number(tolerance_mm, "对照阈值", positive=True)
    valid = bool(dashboard.get("binding_valid") and report)
    metric, metric_issue = validated_metric(report, manifest, dashboard)
    correction = diameter_correction(book["samples"], scope_id(manifest), tolerance_mm)
    by_id = {row["pipe_id"]: row for row in metric.get("pipes", [])}
    priors = {row["pipe_id"]: row for row in manifest.get("model", {}).get("pipes", [])}
    rows = []
    for pipe in dashboard.get("pipes", []):
        identity = pipe["pipe_id"]
        measured = by_id.get(identity, {})
        available = measured.get("status") == "MEASURED"
        raw = measured.get("diameter_mm") if available else None
        corrected = corrected_diameter(raw, correction, tolerance_mm)
        nominal = pipe.get("nominal_diameter_mm")
        value = corrected if corrected is not None else raw
        error = value - nominal if value is not None and nominal is not None else None
        position_error = None
        if available and priors.get(identity, {}).get("centerline_world_mm"):
            line = np.asarray(priors[identity]["centerline_world_mm"], dtype=float)
            direction = line[1] - line[0]
            direction /= np.linalg.norm(direction)
            delta = np.asarray(measured["center_world_mm"]) - line[0]
            position_error = float(np.linalg.norm(delta - direction * np.dot(delta, direction)))
        status = STATE_LABELS.get(pipe.get("installation_state"), "待确认")
        comparison = "待测量" if error is None else "外径对照内" if abs(error) <= tolerance_mm else "外径偏差"
        if position_error is not None and position_error > tolerance_mm:
            comparison += " / 位置偏差"
        rows.append({"pipe_id": identity, "installation_state": pipe.get("installation_state", "UNKNOWN"),
                     "installation_label": status, "measurement_status": "MEASURED" if available else "UNKNOWN",
                     "nominal_diameter_mm": nominal, "raw_diameter_mm": raw, "corrected_diameter_mm": corrected,
                     "diameter_error_mm": error, "position_error_mm": position_error,
                     "center_world_mm": measured.get("center_world_mm") if available else None,
                     "camera_depth_mm": measured.get("camera_depth_mm") if available else None,
                     "comparison_label": comparison, "current_status": status + "；" + comparison,
                     "installation_reasons": pipe.get("reason_codes", []),
                     "measurement_reasons": measured.get("reason_codes", [metric_issue or "NO_BOUND_METRIC_REPORT"]),
                     "relations": [], "views": measured.get("views", {})})
    display_by_id = {r["pipe_id"]: r for r in rows}
    pairs = []
    for source in metric.get("pairs", []):
        pair = dict(source)
        a, b = pair["pipe_id_a"], pair["pipe_id_b"]
        if a not in display_by_id or b not in display_by_id:
            continue
        if pair.get("status") == "MEASURED":
            ra, rb = display_by_id[a], display_by_id[b]
            da = ra["corrected_diameter_mm"] if ra["corrected_diameter_mm"] is not None else ra["raw_diameter_mm"]
            db = rb["corrected_diameter_mm"] if rb["corrected_diameter_mm"] is not None else rb["raw_diameter_mm"]
            pair["corrected_clear_gap_mm"] = pair["center_distance_mm"] - (da + db) / 2
            threshold = max(tolerance_mm, by_id[a].get("view_position_difference_mm", 0), by_id[b].get("view_position_difference_mm", 0))
            delta = pair.get("depth_delta_b_minus_a_mm", 0)
            front = None if abs(delta) <= threshold else a if delta > 0 else b
            pair["front_pipe_id"] = front
            pair["depth_order_threshold_mm"] = threshold
            pair["depth_order"] = "DEPTH_TOO_CLOSE" if front is None else "RESOLVED"
            pair["order_label"] = "深度接近，前后待确认" if front is None else f"{front} 在前"
            for own, other in ((a, b), (b, a)):
                label = "深度接近" if front is None else "在其前方" if own == front else "在其后方"
                display_by_id[own]["relations"].append({"other_pipe_id": other, "order": label,
                    "center_distance_mm": pair["center_distance_mm"], "clear_gap_mm": pair["corrected_clear_gap_mm"]})
        else:
            pair["order_label"] = "待确认"
        pairs.append(pair)
    return {"generated_at": datetime.now(timezone.utc).isoformat(), "units": "mm", "tolerance_mm": tolerance_mm,
            "accuracy_validated": False, "dataset_id": manifest.get("dataset_id", ""),
            "capture_id": metric.get("capture_id"), "binding_valid": valid,
            "correction": correction, "pipes": rows, "pairs": pairs,
            "measurement_settings": metric.get("settings", {}),
            "notes": "安装状态、尺寸对照和精度验证为独立结论。前后顺序以左相机观察方向为准。"}


def csv_text(rows: list[dict], fields: list[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        values = {key: json.dumps(row.get(key), ensure_ascii=False) if isinstance(row.get(key), (list, dict)) else row.get(key) for key in fields}
        # Spreadsheet exports are data, including user-entered pipe IDs/notes.
        values = {k: "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v for k, v in values.items()}
        writer.writerow(values)
    return "\ufeff" + stream.getvalue()
