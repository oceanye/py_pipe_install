"""Contract tests for the independent AI section candidate channel."""
from __future__ import annotations

import copy

import pytest

from pipe_twin.ai_section_assist import (
    attach_ai_section_assist,
    build_ai_section_assist,
)


def _point_map(*, overlap: bool = False):
    return {
        "schema_version": "section-point-map-v1",
        "mapping_sha256": "map-hash",
        "view_id": "front",
        "points": [
            {
                "point_id": "P01",
                "pipe_id": "UP",
                "section_uv_mm": [-20, 30],
                "projected_center_px": [120, 80],
                "nominal_diameter_mm": 40,
                "projection_ambiguity": "OVERLAPPING_MODEL_POINTS" if overlap else "NONE",
            },
            {
                "point_id": "P02",
                "pipe_id": "DOWN",
                "section_uv_mm": [30, -20],
                "projected_center_px": [240, 180],
                "nominal_diameter_mm": 50,
                "projection_ambiguity": "NONE",
            },
        ],
    }


def _response():
    return {
        "point_assessments": [
            {
                "point_id": "P01",
                "section_status": "SUPPORTED_VISIBLE",
                "evidence_quality": "HIGH",
                "reason_codes": ["AI_SEES_SECTION_CANDIDATE"],
                "observation_refs": ["left:001", "right:001"],
            },
            {
                "point_id": "P02",
                "section_status": "AMBIGUOUS",
                "evidence_quality": "MEDIUM",
                "reason_codes": [],
            },
        ]
    }


def test_valid_response_is_namespaced_section_only_and_preserves_fixed_locations():
    point_map = _point_map()
    before = copy.deepcopy(point_map)
    result = build_ai_section_assist(point_map, _response(), source={"provider": "kimi", "model": "kimi-3.1-pro"})

    assert result["schema_version"] == "ai-section-assist-v1"
    assert result["scope"] == "SECTION_ONLY"
    assert result["scope_detail"] == "CROSS_SECTION_CANDIDATE_ONLY"
    assert result["run_status"] == "VALID"
    assert result["status"] == "AMBIGUOUS"
    assert [row["point_id"] for row in result["candidates"]] == ["P01", "P02"]
    assert result["candidates"][0]["pipe_id"] == "UP"
    assert result["candidates"][0]["section_position"]["section_uv_mm"] == [-20.0, 30.0]
    assert result["provider"] == "kimi"
    assert result["model"] == "kimi-3.1-pro"
    assert point_map == before
    assert "counts" not in result
    assert result["installation_state"] == "OUT_OF_SCOPE_FOR_AI_SECTION_ASSIST"


def test_model_overlap_caps_supported_candidate_to_ambiguous():
    result = build_ai_section_assist(_point_map(overlap=True), _response())
    first = result["candidates"][0]
    assert first["section_status"] == "AMBIGUOUS"
    assert "P01:OVERLAPPING_MODEL_POINTS" in result["hard_conflicts"]
    assert result["status"] == "AMBIGUOUS"


def test_report_section_evidence_gate_does_not_read_or_change_strict_pipes_and_counts():
    report = {
        "counts": {"INSTALLED": 1, "UNKNOWN": 1},
        "pipes": [{"pipe_id": "UP", "installation_state": "INSTALLED"}],
        "section_evaluation": {
            "control_section": {
                "schema_version": "pooled-control-section-v1",
                "status": "DEFINED", "station_mm": 321.5,
                "interval_mm": [300.0, 343.0],
                "source": "PIPE_SEGMENT_AGGREGATION_COMMON_SUPPORT_MIDPOINT",
            },
            "position_assessment": {
                "pipes": [
                    {"pipe_id": "UP", "state": "UNKNOWN", "reason_codes": ["NO_UNIQUE_VISIBLE_SECTION"]},
                    {"pipe_id": "DOWN", "state": "IN_POSITION", "reason_codes": []},
                ]
            }
        },
    }
    original = copy.deepcopy(report)
    result = attach_ai_section_assist(report, _point_map(), _response())

    assert report == original
    assert result["counts"] == original["counts"]
    assert result["pipes"] == original["pipes"]
    assert result["ai_section_assist"]["candidates"][0]["section_status"] == "UNRESOLVED"
    assert "NO_UNIQUE_VISIBLE_SECTION" in result["ai_section_assist"]["candidates"][0]["reason_codes"]
    assert result["ai_section_assist"]["candidates"][1]["section_status"] == "AMBIGUOUS"
    assert result["ai_section_assist"]["evidence_source"] == "REPORT_SECTION_EVALUATION"
    assert result["ai_section_assist"]["control_section"]["station_mm"] == 321.5
    assert result["ai_section_assist"]["control_section"]["source"] == "PIPE_SEGMENT_AGGREGATION_COMMON_SUPPORT_MIDPOINT"


def test_section_namespace_does_not_echo_strict_evidence_state():
    section = {"pipes": [{"pipe_id": "UP", "state": "INSTALLED", "reason_codes": ["installed"]}]}
    result = build_ai_section_assist(_point_map(), _response(), section_evidence=section)
    first = result["candidates"][0]
    assert first["section_evidence"]["source_state"] == "UNSPECIFIED"
    assert first["section_evidence"]["source_reason_codes"] == []
    assert all("installed" not in str(value).lower() for value in first["section_evidence"].values())


def test_missing_point_is_explicit_partial_unresolved_candidate():
    response = {"point_assessments": [_response()["point_assessments"][0]]}
    result = build_ai_section_assist(_point_map(), response)
    assert result["run_status"] == "PARTIAL"
    assert result["candidates"][1]["section_status"] == "UNRESOLVED"
    assert result["candidates"][1]["reason_codes"] == ["MISSING_AI_ASSESSMENT"]
    assert result["section_decision"] == "CANDIDATES_READY"


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        ({"point_assessments": [{"point_id": "P99", "section_status": "SUPPORTED_VISIBLE", "evidence_quality": "HIGH"}]}, "AI_UNKNOWN_MODEL_POINT"),
        ({"point_assessments": [{"point_id": "P01", "pipe_id": "WRONG", "section_status": "SUPPORTED_VISIBLE", "evidence_quality": "HIGH"}]}, "AI_PIPE_ID_MISMATCH"),
        ({"point_assessments": [{"point_id": "P01", "section_status": "INSTALLED", "evidence_quality": "HIGH"}]}, "AI_INVALID_SECTION_STATUS"),
        ({"installation_state": "INSTALLED", "point_assessments": []}, "FORBIDDEN_INSTALLATION_FIELD"),
    ],
)
def test_invalid_or_installation_leaking_provider_response_is_rejected(response, reason):
    result = build_ai_section_assist(_point_map(), response)
    assert result["run_status"] == "REJECTED"
    assert result["status"] == "UNRESOLVED"
    assert result["reason_codes"] == [reason]
    assert result["candidates"] == []


def test_no_response_is_explicit_not_run_and_has_no_candidates():
    result = build_ai_section_assist(_point_map())
    assert result["scope"] == "SECTION_ONLY"
    assert result["status"] == "NOT_RUN"
    assert result["run_status"] == "NOT_RUN"
    assert result["candidates"] == []
    assert result["reason_codes"] == ["AI_SECTION_NOT_RUN"]


def test_pooled_control_section_is_carried_into_ai_namespace_without_installation_fields():
    control = {
        "schema_version": "pooled-control-section-v1",
        "source": "PIPE_SEGMENT_AGGREGATION_COMMON_SUPPORT_MIDPOINT",
        "status": "DEFINED", "station_mm": 123.4,
        "interval_mm": [100.0, 146.8], "candidate_ids": ["A"],
        "reason_codes": [],
    }
    report = {"section_evaluation": {"control_section": control,
                                     "position_assessment": {"pipes": []}}}
    result = build_ai_section_assist(_point_map(), report=report)
    assert result["control_section"] == control
    assert "installation_state" in result
    assert result["installation_state"] == "OUT_OF_SCOPE_FOR_AI_SECTION_ASSIST"
