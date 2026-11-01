"""Read-only AI candidates for one model cross-section.

The strict stereo/state path deliberately remains outside this module.  A
provider (for example a multimodal model) can return a small JSON response
for the fixed ``P01`` ... ``Pn`` model points created by
``model_section_reference``.  This module validates that response, applies
the section evidence gates that are already present in a report, and returns
an advisory payload suitable for storing under ``report["ai_section_assist"]``.

There is no image, camera-pose, axial-segment, installation-state, or count
mutation here.  The input report and evidence mappings are treated as
immutable; ``attach_ai_section_assist`` deep-copies a report before adding
the namespaced payload.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import math
import re
from typing import Any


SCHEMA_VERSION = "ai-section-assist-v1"
SECTION_SCOPE = "SECTION_ONLY"
OUT_OF_SCOPE_STATE = "OUT_OF_SCOPE_FOR_AI_SECTION_ASSIST"

# These are intentionally small, finite vocabularies.  In particular there
# is no ``INSTALLED``/``NOT_INSTALLED`` value in this contract.
POINT_STATUSES = (
    "SUPPORTED_VISIBLE",
    "AMBIGUOUS",
    "UNRESOLVED",
    "RECAPTURE_RECOMMENDED",
)
SECTION_DECISIONS = (
    "CANDIDATES_READY",
    "AMBIGUOUS",
    "UNRESOLVED",
    "RECAPTURE_RECOMMENDED",
    "NOT_RUN",
)
RUN_STATUSES = ("VALID", "PARTIAL", "REJECTED", "NOT_RUN")
EVIDENCE_QUALITIES = ("HIGH", "MEDIUM", "LOW")

_POINT_RE = re.compile(r"^P(\d{2})$")
_FORBIDDEN_TERMS = {
    "installation_state",
    "strict_state",
    "is_installed",
    "installed",
    "not_installed",
    "counts",
}
_SOURCE_KEYS = (
    "provider",
    "model",
    "request_id",
    "input_hash",
    "source_id",
    "generated_at",
)


def _is_forbidden_key(key: Any) -> bool:
    """Return whether a provider key could leak strict installation state."""

    return str(key).strip().lower() in _FORBIDDEN_TERMS


def _finite_number(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _finite_pair(value: Any, *, field: str, allow_none: bool = False) -> list[float] | None:
    if value is None and allow_none:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{field} must be a finite 2-vector")
    return [_finite_number(value[0], field=f"{field}[0]"), _finite_number(value[1], field=f"{field}[1]")]


def _point_rows(point_map: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate and copy the fixed model points without 3-D fields."""

    if not isinstance(point_map, Mapping):
        raise ValueError("point_map must be a mapping")
    raw_points = point_map.get("points")
    if isinstance(raw_points, (str, bytes)) or not isinstance(raw_points, Sequence) or not raw_points:
        raise ValueError("point_map.points must be a non-empty sequence")

    rows: list[dict[str, Any]] = []
    seen_points: set[str] = set()
    seen_pipes: set[str] = set()
    for index, raw in enumerate(raw_points):
        if not isinstance(raw, Mapping):
            raise ValueError(f"point_map.points[{index}] must be a mapping")
        point_id = raw.get("point_id")
        pipe_id = raw.get("pipe_id")
        if not isinstance(point_id, str) or not _POINT_RE.fullmatch(point_id):
            raise ValueError(f"point_map.points[{index}].point_id must use P%02d")
        if not isinstance(pipe_id, str) or not pipe_id.strip():
            raise ValueError(f"point_map.points[{index}].pipe_id must be a non-empty string")
        if point_id in seen_points or pipe_id in seen_pipes:
            raise ValueError("point_map contains duplicate point_id or pipe_id")
        seen_points.add(point_id)
        seen_pipes.add(pipe_id)
        point_number = int(point_id[1:])
        if point_number != index + 1:
            raise ValueError("point_map point IDs must be contiguous P01...Pn")
        row = {
            "point_id": point_id,
            "pipe_id": pipe_id,
            "section_uv_mm": _finite_pair(raw.get("section_uv_mm"), field=f"points[{index}].section_uv_mm"),
            "projected_center_px": _finite_pair(
                raw.get("projected_center_px"), field=f"points[{index}].projected_center_px", allow_none=True
            ),
            "nominal_diameter_mm": _finite_number(
                raw.get("nominal_diameter_mm"), field=f"points[{index}].nominal_diameter_mm"
            ),
            "projection_ambiguity": str(raw.get("projection_ambiguity", "NONE")),
        }
        if row["nominal_diameter_mm"] <= 0:
            raise ValueError(f"points[{index}].nominal_diameter_mm must be positive")
        rows.append(row)

    metadata = {
        "mapping_sha256": point_map.get("mapping_sha256"),
        "view_id": point_map.get("view_id"),
        "point_id_format": point_map.get("point_id_format", "P%02d"),
        "observation_direction_world": copy.deepcopy(point_map.get("observation_direction_world")),
    }
    return rows, metadata


def _extract_section_evidence(
    report: Mapping[str, Any] | None,
    section_evidence: Mapping[str, Any] | None,
) -> tuple[Mapping[str, Any] | None, str]:
    """Select only the section evidence branch of a report.

    Top-level ``report["pipes"]`` and ``report["counts"]`` are intentionally
    ignored because they contain strict installation state.  A merged report
    may provide ``section_evaluations``; the latest section is the current
    read-only evidence.
    """

    if section_evidence is not None:
        if not isinstance(section_evidence, Mapping):
            raise ValueError("section_evidence must be a mapping")
        return section_evidence, "EXPLICIT_SECTION_EVIDENCE"
    if not isinstance(report, Mapping):
        return None, "NO_SECTION_EVIDENCE"
    candidate = report.get("section_evaluation")
    if isinstance(candidate, Mapping):
        return candidate, "REPORT_SECTION_EVALUATION"
    many = report.get("section_evaluations")
    if isinstance(many, Sequence) and not isinstance(many, (str, bytes)):
        for candidate in reversed(many):
            if isinstance(candidate, Mapping):
                return candidate, "REPORT_LATEST_SECTION_EVALUATION"
    # Permit callers that already pass the section object as ``report`` while
    # still refusing to inspect the unrelated top-level installation rows.
    if any(key in report for key in ("position_assessment", "registration", "candidates")):
        return report, "DIRECT_SECTION_EVIDENCE"
    return None, "NO_SECTION_EVIDENCE"


def _evidence_rows(section: Mapping[str, Any] | None) -> dict[str, Mapping[str, Any]]:
    if not isinstance(section, Mapping):
        return {}
    position = section.get("position_assessment")
    rows = position.get("pipes") if isinstance(position, Mapping) else None
    if isinstance(rows, Mapping):
        result = {}
        for key, row in rows.items():
            if isinstance(row, Mapping):
                pipe_id = row.get("pipe_id", key)
                if isinstance(pipe_id, str):
                    result.setdefault(pipe_id, row)
        return result
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        rows = section.get("pipes")
    if isinstance(rows, Mapping):
        result = {}
        for key, row in rows.items():
            if isinstance(row, Mapping):
                pipe_id = row.get("pipe_id", key)
                if isinstance(pipe_id, str):
                    result.setdefault(pipe_id, row)
        return result
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if isinstance(row, Mapping) and isinstance(row.get("pipe_id"), str):
            result.setdefault(row["pipe_id"], row)
    return result


def _evidence_gate(row: Mapping[str, Any] | None) -> tuple[str | None, list[str], dict[str, Any]]:
    """Return a conservative status cap from section evidence.

    The cap is applied only for explicit evidence conflicts.  A normal
    ``UNKNOWN`` position row is preserved as an unresolved candidate; it is
    never converted into a negative installation decision.
    """

    if not isinstance(row, Mapping):
        return None, [], {"available": False}
    raw_reasons = row.get("reason_codes") or []
    if isinstance(raw_reasons, (str, bytes)) or not isinstance(raw_reasons, Sequence):
        raw_reasons = []
    reasons = [
        str(value)
        for value in raw_reasons
        if isinstance(value, (str, int, float)) and not _is_forbidden_key(value)
    ]
    lower = {value.upper() for value in reasons}
    gate: str | None = None
    if lower & {"SECTION_SEARCH_TRUNCATED", "CAPTURE_UNHEALTHY", "CALIBRATION_NOT_VALIDATED"}:
        gate = "RECAPTURE_RECOMMENDED"
    elif lower & {"NO_UNIQUE_VISIBLE_SECTION", "SECTION_MODEL_CORRESPONDENCE_UNRESOLVED", "POOLED_BOUNDARY_UNSTABLE"}:
        gate = "UNRESOLVED"
    source_state = row.get("state")
    if source_state not in {"IN_POSITION", "OUT_OF_POSITION", "UNKNOWN"}:
        source_state = "UNSPECIFIED"
    evidence = {
        "available": True,
        "source_state": source_state,
        "source_reason_codes": sorted(set(reasons)),
    }
    return gate, reasons, evidence


def _status_worst(first: str, second: str) -> str:
    rank = {"SUPPORTED_VISIBLE": 0, "UNRESOLVED": 1, "AMBIGUOUS": 2, "RECAPTURE_RECOMMENDED": 3}
    return second if rank[second] > rank[first] else first


def _section_decision(rows: Sequence[Mapping[str, Any]]) -> str:
    statuses = [str(row.get("section_status")) for row in rows]
    if not statuses:
        return "NOT_RUN"
    if "RECAPTURE_RECOMMENDED" in statuses:
        return "RECAPTURE_RECOMMENDED"
    if "AMBIGUOUS" in statuses:
        return "AMBIGUOUS"
    if all(status == "UNRESOLVED" for status in statuses):
        return "UNRESOLVED"
    return "CANDIDATES_READY"


def _source_metadata(source: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(source, Mapping):
        return {}
    result = {}
    for key in _SOURCE_KEYS:
        value = source.get(key)
        if value is not None and isinstance(value, (str, int, float, bool)):
            result[key] = copy.deepcopy(value)
    return result


def _base_payload(rows: Sequence[Mapping[str, Any]], metadata: Mapping[str, Any], *, evidence_source: str) -> dict[str, Any]:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scope": SECTION_SCOPE,
        "scope_detail": "CROSS_SECTION_CANDIDATE_ONLY",
        "advisory_only": True,
        "mainline_unchanged": True,
        "installation_state": OUT_OF_SCOPE_STATE,
        "run_status": "NOT_RUN",
        "status": "NOT_RUN",
        "section_decision": "NOT_RUN",
        "point_assessments": [],
        "candidates": [],
        "hard_conflicts": [],
        "reason_codes": ["AI_SECTION_NOT_RUN"],
        "evidence_source": evidence_source,
        "point_count": len(rows),
    }
    payload.update({key: copy.deepcopy(value) for key, value in metadata.items() if value is not None})
    return payload


def _rejected_payload(
    rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    *,
    evidence_source: str,
    reason: str,
) -> dict[str, Any]:
    payload = _base_payload(rows, metadata, evidence_source=evidence_source)
    payload.update(run_status="REJECTED", status="UNRESOLVED", section_decision="UNRESOLVED", reason_codes=[reason])
    return payload


def build_ai_section_assist(
    point_map: Mapping[str, Any],
    ai_response: Mapping[str, Any] | None = None,
    *,
    report: Mapping[str, Any] | None = None,
    section_evidence: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate one provider response against a fixed model section map.

    ``ai_response`` is expected to contain ``point_assessments``.  Each item
    has ``point_id``, ``section_status`` and ``evidence_quality``; optional
    ``reason_codes`` and ``observation_refs`` are copied after validation.
    Missing points become explicit ``UNRESOLVED`` rows and make the run
    ``PARTIAL``.  Unknown or duplicated point IDs, installation fields, and
    invalid states are rejected fail-closed.
    """

    rows, metadata = _point_rows(point_map)
    evidence, evidence_source = _extract_section_evidence(report, section_evidence)
    payload = _base_payload(rows, metadata, evidence_source=evidence_source)
    control_section = evidence.get("control_section") if isinstance(evidence, Mapping) else None
    if isinstance(control_section, Mapping):
        # The pooled-section midpoint is the only control-plane metadata
        # copied into the AI namespace.  It contains no installation state.
        payload["control_section"] = copy.deepcopy(dict(control_section))
    payload.update(_source_metadata(source))
    if ai_response is None:
        return payload
    if not isinstance(ai_response, Mapping):
        return _rejected_payload(rows, metadata, evidence_source=evidence_source, reason="AI_RESPONSE_NOT_OBJECT") | _source_metadata(source)
    if any(_is_forbidden_key(key) for key in ai_response):
        return _rejected_payload(rows, metadata, evidence_source=evidence_source, reason="FORBIDDEN_INSTALLATION_FIELD") | _source_metadata(source)

    # ``point_assessments`` is the canonical provider contract.  The two
    # aliases make it possible to ingest a small hand-authored/Kimi response
    # without weakening the fixed-point validation below.
    raw_assessments = ai_response.get("point_assessments")
    if raw_assessments is None:
        raw_assessments = ai_response.get("candidates", ai_response.get("observations"))
    if isinstance(raw_assessments, (str, bytes)) or not isinstance(raw_assessments, Sequence):
        return _rejected_payload(rows, metadata, evidence_source=evidence_source, reason="AI_POINT_ASSESSMENTS_REQUIRED") | _source_metadata(source)

    fixed_by_id = {row["point_id"]: row for row in rows}
    evidence_by_pipe = _evidence_rows(evidence)
    seen: set[str] = set()
    normalized: dict[str, dict[str, Any]] = {}
    hard_conflicts: list[str] = []
    for index, raw in enumerate(raw_assessments):
        if not isinstance(raw, Mapping):
            return _rejected_payload(rows, metadata, evidence_source=evidence_source, reason="AI_POINT_ASSESSMENT_NOT_OBJECT") | _source_metadata(source)
        if any(_is_forbidden_key(key) for key in raw):
            return _rejected_payload(rows, metadata, evidence_source=evidence_source, reason="FORBIDDEN_INSTALLATION_FIELD") | _source_metadata(source)
        point_id = raw.get("point_id")
        if point_id not in fixed_by_id:
            return _rejected_payload(rows, metadata, evidence_source=evidence_source, reason="AI_UNKNOWN_MODEL_POINT") | _source_metadata(source)
        if point_id in seen:
            return _rejected_payload(rows, metadata, evidence_source=evidence_source, reason="AI_DUPLICATE_MODEL_POINT") | _source_metadata(source)
        seen.add(point_id)
        fixed = fixed_by_id[point_id]
        if raw.get("pipe_id") is not None and raw.get("pipe_id") != fixed["pipe_id"]:
            return _rejected_payload(rows, metadata, evidence_source=evidence_source, reason="AI_PIPE_ID_MISMATCH") | _source_metadata(source)
        status = raw.get("section_status", raw.get("status"))
        quality = raw.get("evidence_quality", raw.get("quality"))
        if status not in POINT_STATUSES:
            return _rejected_payload(rows, metadata, evidence_source=evidence_source, reason="AI_INVALID_SECTION_STATUS") | _source_metadata(source)
        if quality not in EVIDENCE_QUALITIES:
            return _rejected_payload(rows, metadata, evidence_source=evidence_source, reason="AI_INVALID_EVIDENCE_QUALITY") | _source_metadata(source)
        reason_codes = raw.get("reason_codes", [])
        observation_refs = raw.get("observation_refs", [])
        if not isinstance(reason_codes, Sequence) or isinstance(reason_codes, (str, bytes)) or not all(isinstance(value, str) for value in reason_codes):
            return _rejected_payload(rows, metadata, evidence_source=evidence_source, reason="AI_INVALID_REASON_CODES") | _source_metadata(source)
        if not isinstance(observation_refs, Sequence) or isinstance(observation_refs, (str, bytes)) or not all(isinstance(value, str) for value in observation_refs):
            return _rejected_payload(rows, metadata, evidence_source=evidence_source, reason="AI_INVALID_OBSERVATION_REFS") | _source_metadata(source)

        gate, evidence_reasons, evidence_summary = _evidence_gate(evidence_by_pipe.get(fixed["pipe_id"]))
        final_status = status
        if fixed["projection_ambiguity"] != "NONE":
            final_status = _status_worst(final_status, "AMBIGUOUS")
            hard_conflicts.append(f"{point_id}:OVERLAPPING_MODEL_POINTS")
        if gate is not None:
            final_status = _status_worst(final_status, gate)
            hard_conflicts.append(f"{point_id}:SECTION_EVIDENCE_GATE")
        normalized[point_id] = {
            "point_id": point_id,
            "pipe_id": fixed["pipe_id"],
            "section_status": final_status,
            "evidence_quality": quality,
            "reason_codes": sorted(set(str(value) for value in reason_codes + evidence_reasons)),
            "observation_refs": list(observation_refs),
            "model_section": {
                "section_uv_mm": list(fixed["section_uv_mm"]),
                "projected_center_px": None if fixed["projected_center_px"] is None else list(fixed["projected_center_px"]),
                "nominal_diameter_mm": fixed["nominal_diameter_mm"],
                "projection_ambiguity": fixed["projection_ambiguity"],
            },
            "section_evidence": evidence_summary,
        }

    missing = [row for row in rows if row["point_id"] not in seen]
    for fixed in missing:
        point_id = fixed["point_id"]
        _gate, evidence_reasons, evidence_summary = _evidence_gate(evidence_by_pipe.get(fixed["pipe_id"]))
        normalized[point_id] = {
            "point_id": point_id,
            "pipe_id": fixed["pipe_id"],
            "section_status": "UNRESOLVED",
            "evidence_quality": "LOW",
            "reason_codes": sorted(set(["MISSING_AI_ASSESSMENT", *evidence_reasons])),
            "observation_refs": [],
            "model_section": {
                "section_uv_mm": list(fixed["section_uv_mm"]),
                "projected_center_px": None if fixed["projected_center_px"] is None else list(fixed["projected_center_px"]),
                "nominal_diameter_mm": fixed["nominal_diameter_mm"],
                "projection_ambiguity": fixed["projection_ambiguity"],
            },
            "section_evidence": evidence_summary,
        }

    ordered = [normalized[row["point_id"]] for row in rows]
    decision = _section_decision(ordered)
    candidates = []
    for item in ordered:
        location = item["model_section"]
        uv = location["section_uv_mm"]
        section_position = {
            "label": f"u={uv[0]:.1f}, v={uv[1]:.1f} mm",
            "section_uv_mm": list(uv),
            "projected_center_px": None
            if location["projected_center_px"] is None
            else list(location["projected_center_px"]),
        }
        candidates.append(item | {
            "status": item["section_status"],
            "quality": item["evidence_quality"],
            "section_position": section_position,
            "model_position": copy.deepcopy(section_position),
        })
    payload.update(
        run_status="VALID" if not missing else "PARTIAL",
        status=decision,
        section_decision=decision,
        point_assessments=ordered,
        candidates=candidates,
        hard_conflicts=sorted(set(hard_conflicts)),
        reason_codes=([] if not missing else ["AI_RESPONSE_MISSING_MODEL_POINTS"]),
    )
    return payload


def attach_ai_section_assist(
    report: Mapping[str, Any],
    point_map: Mapping[str, Any],
    ai_response: Mapping[str, Any] | None = None,
    *,
    section_evidence: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a copied report with an ``ai_section_assist`` namespace.

    The source report is never mutated.  In particular, existing ``pipes``,
    ``counts``, and strict ``installation_state`` values are byte-for-byte
    preserved by the caller-visible object.
    """

    if not isinstance(report, Mapping):
        raise ValueError("report must be a mapping")
    result = copy.deepcopy(dict(report))
    result["ai_section_assist"] = build_ai_section_assist(
        point_map,
        ai_response,
        report=report,
        section_evidence=section_evidence,
        source=source,
    )
    return result


__all__ = [
    "EVIDENCE_QUALITIES",
    "OUT_OF_SCOPE_STATE",
    "POINT_STATUSES",
    "RUN_STATUSES",
    "SCHEMA_VERSION",
    "SECTION_DECISIONS",
    "SECTION_SCOPE",
    "attach_ai_section_assist",
    "build_ai_section_assist",
]
