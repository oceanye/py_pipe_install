"""Propose parallel, spatially connected straight-pipe groups from an STL catalogue.

Model groups are independent of camera pixels. A current, healthy elevation
registration may project only its observed axial interval, never guessed STL
endpoints (axial translation is unobservable in that registration).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

ALGORITHM = "stl-parallel-groups-v1"


def _geometry(specs: Sequence[Mapping]) -> list[dict]:
    if not specs or len(specs) > 512:
        raise ValueError("自动分区需要 1–512 个独立直管组件")
    rows, seen = [], set()
    for spec in specs:
        pid = spec.get("pipe_id")
        line = np.asarray(spec.get("centerline_world_mm"), float)
        diameter = spec.get("nominal_diameter_mm")
        if (not isinstance(pid, str) or not pid or pid in seen or line.shape != (2, 3)
                or not np.isfinite(line).all() or type(diameter) not in (int, float)
                or not math.isfinite(diameter) or diameter <= 0):
            raise ValueError("自动分区需要唯一编号、有效中心线和正数外径")
        axis = line[1] - line[0]
        length = float(np.linalg.norm(axis))
        if not math.isfinite(length) or length <= 1e-6:
            raise ValueError("自动分区不能使用零长度中心线")
        axis /= length
        if axis[np.argmax(np.abs(axis))] < 0:
            axis = -axis
        rows.append(dict(pipe_id=pid, line=line, axis=axis, diameter=float(diameter)))
        seen.add(pid)
    return sorted(rows, key=lambda row: row["pipe_id"])


def catalog_fingerprint(specs: Sequence[Mapping]) -> str:
    """Geometry/IDs only; paint edits do not change model grouping."""
    rows = _geometry(specs)
    value = [{"pipe_id": row["pipe_id"], "diameter": row["diameter"],
              "line": sorted(row["line"].tolist())} for row in rows]
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _group_axis(rows: list[dict]) -> np.ndarray:
    axes = np.asarray([row["axis"] for row in rows])
    _, vectors = np.linalg.eigh(axes.T @ axes)
    axis = vectors[:, -1]
    return -axis if axis[np.argmax(np.abs(axis))] < 0 else axis


def propose_model_zones(specs: Sequence[Mapping], *, angle_tolerance_deg: float = 2.0,
                        maximum_gap_mm: float = 200.0,
                        minimum_common_length_mm: float = 40.0) -> list[dict[str, Any]]:
    """Deterministic complete-link directions, then connected transverse groups.

    Every component is retained, including singletons. Neighbours must share
    an axial interval; each final group also retains a common interval. Thus
    chained directions and disjoint collinear segments cannot become a fake
    parallel run.
    """
    for name, value, upper in (("平行角容差", angle_tolerance_deg, 2.0),
                               ("最大横向净间距", maximum_gap_mm, 100000.0),
                               ("最小共同管长", minimum_common_length_mm, 100000.0)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= upper:
            raise ValueError(f"{name}必须大于 0 且不超过 {upper:g}")
    rows = _geometry(specs)
    gate = math.cos(math.radians(angle_tolerance_deg))
    directions: list[list[dict]] = []
    for row in rows:
        target = next((group for group in directions
                       if all(abs(float(row["axis"] @ item["axis"])) >= gate - 1e-12 for item in group)), None)
        if target is None:
            directions.append([row])
        else:
            target.append(row)
    clusters = []
    for direction in directions:
        axis = _group_axis(direction)
        intervals = [np.sort(row["line"] @ axis) for row in direction]
        centers = [row["line"].mean(0) for row in direction]
        remaining = list(range(len(direction)))
        while remaining:
            members = [remaining.pop(0)]
            low, high = intervals[members[0]]
            growing = True
            while growing:
                growing = False
                for index in remaining[:]:
                    new_low, new_high = max(low, intervals[index][0]), min(high, intervals[index][1])
                    if new_high - new_low < minimum_common_length_mm:
                        continue
                    for other in members:
                        delta = centers[index] - centers[other]
                        distance = np.linalg.norm(delta - axis * (delta @ axis))
                        gap = distance - (direction[index]["diameter"] + direction[other]["diameter"]) / 2
                        if gap <= maximum_gap_mm:
                            members.append(index)
                            remaining.remove(index)
                            low, high = new_low, new_high
                            growing = True
                            break
            clusters.append(([direction[index] for index in members], axis, [float(low), float(high)]))
    fingerprint = catalog_fingerprint(specs)
    proposals = []
    for index, (members, axis, interval) in enumerate(sorted(clusters, key=lambda item: item[0][0]["pipe_id"]), 1):
        ids = sorted(row["pipe_id"] for row in members)
        proposals.append({
            "zone_id": f"M{index:02d}", "label": f"平行管组 {index}（{len(ids)} 根）",
            "source": "stl_parallel", "confirmed": False, "enabled": False,
            "coordinate_space": "rectified_left", "roi_rect_px": None,
            "roi_source": "unmapped", "model_pipe_ids": ids, "axis_world": axis.tolist(),
            "model_catalog_sha256": fingerprint,
            "proposal": {"algorithm": ALGORITHM, "angle_tolerance_deg": float(angle_tolerance_deg),
                         "maximum_gap_mm": float(maximum_gap_mm),
                         "minimum_common_length_mm": float(minimum_common_length_mm),
                         "common_interval_model_mm": interval,
                         "note": "INSUFFICIENT_LAYOUT_FOR_REGISTRATION" if len(ids) < 3 else "MODEL_GROUP_ONLY"},
        })
    return proposals


def zone_model_specs(zone: Mapping, specs: Sequence[Mapping]) -> list[dict]:
    """Validate the model group binding and return its candidate catalogue."""
    ids = zone.get("model_pipe_ids")
    if ids is None:
        return list(specs)
    by_id = {row["pipe_id"]: row for row in specs}
    if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids) or any(pid not in by_id for pid in ids):
        raise ValueError("分区引用了不存在或重复的模型管道，请重新自动分区")
    if zone.get("model_catalog_sha256") != catalog_fingerprint(specs):
        raise ValueError("分区所属模型已改变，请根据当前 STL 重新生成")
    selected = [by_id[pid] for pid in ids]
    axis = np.asarray(zone.get("axis_world"), float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis), 1):
        raise ValueError("分区缺少有效的模型管长方向")
    if any(abs(float(row["axis"] @ axis)) < math.cos(math.radians(2)) - 1e-12 for row in _geometry(selected)):
        raise ValueError("分区内的模型管段不平行，请重新分区")
    return selected


def project_model_zone(zone: Mapping, specs: Sequence[Mapping], calibration: Any,
                       report: Mapping | None) -> list[int] | None:
    """Project a group into the current observed section of a healthy report.

    No pose, ambiguous/unhealthy evidence, or a different model-axis family
    yields no pixel ROI. User can then map the model group to a single image
    rectangle.
    """
    selected = zone_model_specs(zone, specs)
    if not isinstance(report, Mapping) or report.get("scope") == "zones":
        return None  # independent zone poses cannot be treated as one rigid pose
    registration = report.get("registration") or {}
    captures = (report.get("capture_audit") or {}).get("groups") or []
    if registration.get("status") != "MATCHED" or not captures or not captures[-1].get("analysis_healthy"):
        return None
    rotation = np.asarray(registration.get("rotation_model_to_camera"), float)
    translation = np.asarray(registration.get("translation_model_to_camera_mm"), float)
    if (rotation.shape != (3, 3) or translation.shape != (3,) or not np.isfinite(rotation).all()
            or not np.isfinite(translation).all() or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(rotation), 1)):
        return None
    axis = rotation @ np.asarray(zone["axis_world"], float)
    reference_axis = np.asarray(registration.get("camera_axis"), float)
    if reference_axis.shape != (3,) or abs(float(axis @ reference_axis)) < math.cos(math.radians(2)):
        return None
    surface = report.get("local_surface")
    if not isinstance(surface, Mapping):
        return None
    observations = {row.get("observation_id"): row for row in surface.get("observations", [])
                    if isinstance(row, Mapping) and row.get("observation_id")}
    intervals = []
    for match in registration.get("matches", []):
        obs = observations.get(match.get("observation_id"), {})
        segment = np.asarray(obs.get("observed_segment_camera_mm"), float)
        if segment.shape == (2, 3) and np.isfinite(segment).all():
            intervals.append(np.sort(segment @ axis))
    if not intervals:
        return None
    low, high = max(item[0] for item in intervals), min(item[1] for item in intervals)
    if high - low < 40:
        return None
    # Bounding boxes around the two section centres conservatively include
    # the cylinder radius, without claiming knowledge of either pipe end.
    vertices = []
    for spec in selected:
        center = rotation @ np.asarray(spec["centerline_world_mm"]).mean(0) + translation
        radius = float(spec["nominal_diameter_mm"]) / 2
        for station in (low, high):
            section = center + axis * (station - float(center @ axis))
            vertices.extend(section + radius * np.asarray(signs) for signs in itertools.product((-1, 1), repeat=3))
    points = np.asarray(vertices)
    points = points[np.isfinite(points).all(axis=1) & (points[:, 2] > 1)]
    if len(points) == 0:
        return None
    projected = points @ calibration.left.rectified_intrinsic.T
    pixels = projected[:, :2] / projected[:, 2, None]
    minimum = np.floor(pixels.min(0) - 8).astype(int)
    maximum = np.ceil(pixels.max(0) + 8).astype(int)
    minimum = np.maximum(minimum, [0, 0])
    maximum = np.minimum(maximum, [calibration.left.width, calibration.left.height])
    size = maximum - minimum
    if (size < 32).any():
        return None
    return [*minimum.tolist(), *size.tolist()]
