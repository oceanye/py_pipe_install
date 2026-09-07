"""Fail-closed local dashboard for CAD-bound pipe installation results.

The data binding and projection helpers in this module deliberately have no
Tk dependency.  ``tkinter`` is imported only while launching the desktop
application so the existing command-line tools and headless CI remain usable.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import os
import queue
import re
import threading
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .logging_config import get_logger, log_event
from .dxf_elevation import DxfElevation, arc_points, read_dxf_elevation

_LOGGER = get_logger("gui")


INSTALLATION_STATES = ("INSTALLED", "NOT_INSTALLED", "UNKNOWN")
STATE_PRESENTATION: dict[str, dict[str, str]] = {
    "INSTALLED": {
        "label_zh": "安装",
        "color": "#2E7D32",
        "symbol": "●",
        "dash": "",
    },
    "NOT_INSTALLED": {
        "label_zh": "未安装",
        "color": "#C62828",
        "symbol": "×",
        "dash": "8 4",
    },
    "UNKNOWN": {
        "label_zh": "不确定",
        "color": "#F9A825",
        "symbol": "?",
        "dash": "4 4",
    },
}

_SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
_UNASSESSABLE_VIEW_STATES = {"FULLY_OCCLUDED", "OUT_OF_FRUSTUM"}
_STEREO_VIEW_ROLES = ("left", "right")

# These are the values emitted by ``stereo_analyzer``.  A couple of legacy
# synthetic values are kept in the allow-list so an old, otherwise valid
# report is still readable; arbitrary strings are rejected in schema 2.0.
_STEREO_OCCLUSION_STATES = {
    "FULLY_VISIBLE",
    "PARTIALLY_VISIBLE",
    "PARTIALLY_OCCLUDED",
    "FULLY_OCCLUDED",
    "OUT_OF_FRUSTUM",
    "NOT_OBSERVED",
}
_STEREO_EVIDENCE_TYPES = {
    "DIRECT_STEREO_CAD_EVIDENCE",
    "NEGATIVE_FREE_SPACE_CANDIDATE",
    "INCONCLUSIVE",
    "DIRECT_SYNTHETIC_INSTANCE_EVIDENCE",
    "NEGATIVE_EVIDENCE_CANDIDATE",
    "QUALIFIED_NEGATIVE_EVIDENCE",
    "UNQUALIFIED_NEGATIVE_CANDIDATE",
}
_STEREO_VISIBILITY_VALUES = {
    "VISIBLE",
    "NOT_OBSERVED",
    "PARTIALLY_VISIBLE",
    "FULLY_VISIBLE",
    "PARTIALLY_OCCLUDED",
    "FULLY_OCCLUDED",
    "OUT_OF_FRUSTUM",
}


def _require_mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _require_nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _normal_sha256(value: object) -> str | None:
    if isinstance(value, str) and _SHA256_RE.fullmatch(value):
        return value.lower()
    return None


def _report_model_hashes(report: Mapping[str, Any]) -> tuple[list[str], bool]:
    """Return all declared model hashes and whether a malformed value was seen."""

    candidates: list[object] = [report.get("model_sha256")]
    model = report.get("model")
    if isinstance(model, Mapping):
        candidates.extend(
            model.get(key)
            for key in ("sha256", "source_sha256", "actual_sha256", "expected_sha256")
        )
    inputs = report.get("inputs")
    if isinstance(inputs, Mapping):
        input_model = inputs.get("model")
        if isinstance(input_model, Mapping):
            candidates.extend(
                input_model.get(key)
                for key in ("actual_sha256", "sha256", "expected_sha256")
            )

    values: list[str] = []
    malformed = False
    for candidate in candidates:
        if candidate is None:
            continue
        normalized = _normal_sha256(candidate)
        if normalized is None:
            malformed = True
        elif normalized not in values:
            values.append(normalized)
    return values, malformed


def _report_manifest_hash(report: Mapping[str, Any]) -> str | None:
    inputs = report.get("inputs")
    if not isinstance(inputs, Mapping):
        return None
    manifest = inputs.get("manifest")
    if not isinstance(manifest, Mapping):
        return None
    return _normal_sha256(manifest.get("sha256"))


def _report_latest_photo_hashes(
    report: Mapping[str, Any],
) -> tuple[dict[str, str], bool]:
    """Extract the latest left/right photo hashes from a stereo report.

    Returns ``(hashes, malformed)``.  A missing capture audit is represented
    by an empty mapping and ``True`` so schema-2 callers can fail closed rather
    than silently trusting a report whose image provenance is unknown.
    """

    audit = report.get("capture_audit")
    if not isinstance(audit, Mapping):
        return {}, True
    groups = audit.get("groups")
    if not isinstance(groups, list) or not groups:
        return {}, True
    latest = groups[-1]
    if not isinstance(latest, Mapping):
        return {}, True
    photos = latest.get("photos")
    if not isinstance(photos, Mapping):
        return {}, True
    hashes: dict[str, str] = {}
    malformed = False
    for role in _STEREO_VIEW_ROLES:
        record = photos.get(role)
        if not isinstance(record, Mapping):
            malformed = True
            continue
        # ``actual_sha256`` is emitted by the analyzer.  ``sha256`` is
        # accepted for reports produced by an older adapter, but still must be
        # a valid digest.
        value = record.get("actual_sha256", record.get("sha256"))
        normalized = _normal_sha256(value)
        if normalized is None:
            malformed = True
        else:
            hashes[role] = normalized
    return hashes, malformed


def _issue(code: str, message: str) -> dict[str, str]:
    return {"code": code, "message": message}


def _coerce_centerline(value: object) -> list[list[float]] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        return None
    result: list[list[float]] = []
    for point in value:
        if (
            not isinstance(point, Sequence)
            or isinstance(point, (str, bytes))
            or len(point) != 3
        ):
            return None
        coordinates: list[float] = []
        for coordinate in point:
            if type(coordinate) not in (int, float) or not math.isfinite(coordinate):
                return None
            coordinates.append(float(coordinate))
        result.append(coordinates)
    return result


def _coerce_positive_number(value: object) -> float | None:
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        return None
    return float(value)


def _safe_reason_codes(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _safe_visibility(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, dict[str, Any]] = {}
    for role, record in value.items():
        if isinstance(role, str) and isinstance(record, Mapping):
            result[role] = copy.deepcopy(dict(record))
    return result


def _schema_two_view_issues(
    result: Mapping[str, Any],
    *,
    pipe_label: str,
) -> list[dict[str, str]]:
    """Validate the per-view evidence contract used by schema 2.0 reports.

    The GUI is a presentation boundary, so it must not infer a missing view or
    turn an arbitrary string into a trustworthy observation.  The analyzer's
    canonical fields are ``visibility_by_view.{left,right}``,
    ``installation_evidence`` and ``occlusion_state``.  ``assessable`` and
    ``reason_codes`` are checked as well because they are needed to explain a
    state and to keep total occlusion fail-closed.  The helper returns issues
    instead of raising; callers can then invalidate the complete report and
    paint every pipe UNKNOWN in one place.
    """

    issues: list[dict[str, str]] = []
    visibility = result.get("visibility_by_view")
    if not isinstance(visibility, Mapping):
        return [
            _issue(
                "REPORT_VISIBILITY_INVALID",
                f"{pipe_label}: visibility_by_view must contain left and right view objects",
            )
        ]

    unexpected_roles = sorted(
        role for role in visibility if role not in _STEREO_VIEW_ROLES
    )
    if unexpected_roles:
        issues.append(
            _issue(
                "REPORT_VISIBILITY_INVALID",
                f"{pipe_label}: unsupported stereo view roles: {unexpected_roles}",
            )
        )

    for role in _STEREO_VIEW_ROLES:
        record = visibility.get(role)
        prefix = f"{pipe_label}.visibility_by_view.{role}"
        if not isinstance(record, Mapping):
            issues.append(
                _issue(
                    "REPORT_VISIBILITY_INVALID",
                    f"{prefix} is missing or is not an object",
                )
            )
            continue

        # These fields are the minimum provenance needed to trust a decisive
        # state in a schema-2 report.  In particular, the GUI must not accept
        # an ``INSTALLED``/``NOT_INSTALLED`` enum merely because a producer
        # supplied a plausible-looking occlusion label.  The analyzer emits
        # the values below for every view; missing values therefore invalidate
        # the complete report rather than being guessed here.
        geometry_source = record.get("projection_geometry_source")
        if geometry_source != "cad_triangle_mesh":
            issues.append(
                _issue(
                    "REPORT_GEOMETRY_SOURCE_INVALID",
                    f"{prefix}.projection_geometry_source must be cad_triangle_mesh",
                )
            )

        occlusion = record.get("occlusion_state")
        if not isinstance(occlusion, str) or occlusion not in _STEREO_OCCLUSION_STATES:
            issues.append(
                _issue(
                    "REPORT_OCCLUSION_INVALID",
                    f"{prefix}.occlusion_state is missing or unsupported",
                )
            )

        evidence = record.get("installation_evidence")
        if not isinstance(evidence, str) or evidence not in _STEREO_EVIDENCE_TYPES:
            issues.append(
                _issue(
                    "REPORT_EVIDENCE_INVALID",
                    f"{prefix}.installation_evidence is missing or unsupported",
                )
            )

        # ``assessable`` is optional for compatibility with early schema-2
        # adapters, but when present it must be a real boolean.  We still
        # enforce the safety relationship for total occlusion.
        assessable = record.get("assessable")
        if assessable is not None and type(assessable) is not bool:
            issues.append(
                _issue(
                    "REPORT_ASSESSABLE_INVALID",
                    f"{prefix}.assessable must be a boolean when present",
                )
            )
        elif (
            type(assessable) is bool
            and isinstance(occlusion, str)
            and occlusion in _UNASSESSABLE_VIEW_STATES
        ):
            if assessable:
                issues.append(
                    _issue(
                        "REPORT_ASSESSABLE_OCCLUSION_CONFLICT",
                        f"{prefix} cannot be assessable when occlusion_state={occlusion}",
                    )
                )

        reason_codes = record.get("reason_codes")
        if reason_codes is not None and (
            not isinstance(reason_codes, list)
            or any(not isinstance(code, str) or not code for code in reason_codes)
        ):
            issues.append(
                _issue(
                    "REPORT_VIEW_REASON_CODES_INVALID",
                    f"{prefix}.reason_codes must be a list of non-empty strings when present",
                )
            )

        for boolean_field in (
            "expected_region_in_frame",
            "expected_region_unoccluded",
            "direct_instance_evidence",
            "negative_candidate",
        ):
            value = record.get(boolean_field)
            if type(value) is not bool:
                issues.append(
                    _issue(
                        "REPORT_EVIDENCE_FLAG_INVALID",
                        f"{prefix}.{boolean_field} must be a boolean",
                    )
                )

        # The state-specific evidence enum and the typed flags must agree.  A
        # forged report otherwise could claim a direct installation while
        # carrying only an inconclusive/occluded observation.
        direct_flag = record.get("direct_instance_evidence")
        negative_flag = record.get("negative_candidate")
        if direct_flag is True and negative_flag is True:
            issues.append(
                _issue(
                    "REPORT_EVIDENCE_FLAG_CONFLICT",
                    f"{prefix}: direct and negative evidence flags cannot both be true",
                )
            )
        if isinstance(evidence, str):
            direct_evidence_values = {
                "DIRECT_STEREO_CAD_EVIDENCE",
                "DIRECT_SYNTHETIC_INSTANCE_EVIDENCE",
            }
            negative_evidence_values = {
                "NEGATIVE_FREE_SPACE_CANDIDATE",
                "NEGATIVE_EVIDENCE_CANDIDATE",
                "QUALIFIED_NEGATIVE_EVIDENCE",
            }
            if evidence in direct_evidence_values and direct_flag is not True:
                issues.append(
                    _issue(
                        "REPORT_EVIDENCE_FLAG_CONFLICT",
                        f"{prefix}: direct evidence requires direct_instance_evidence=true",
                    )
                )
            if evidence in direct_evidence_values and negative_flag is True:
                issues.append(
                    _issue(
                        "REPORT_EVIDENCE_FLAG_CONFLICT",
                        f"{prefix}: direct evidence cannot carry negative_candidate=true",
                    )
                )
            if evidence in negative_evidence_values and negative_flag is not True:
                issues.append(
                    _issue(
                        "REPORT_EVIDENCE_FLAG_CONFLICT",
                        f"{prefix}: negative evidence requires negative_candidate=true",
                    )
                )
            if evidence in negative_evidence_values and direct_flag is True:
                issues.append(
                    _issue(
                        "REPORT_EVIDENCE_FLAG_CONFLICT",
                        f"{prefix}: negative evidence cannot carry direct_instance_evidence=true",
                    )
                )
            if evidence == "INCONCLUSIVE" and (
                direct_flag is True or negative_flag is True
            ):
                issues.append(
                    _issue(
                        "REPORT_EVIDENCE_FLAG_CONFLICT",
                        f"{prefix}: INCONCLUSIVE evidence cannot carry a decisive flag",
                    )
                )

        if direct_flag is True and assessable is False:
            issues.append(
                _issue(
                    "REPORT_EVIDENCE_FLAG_CONFLICT",
                    f"{prefix}: direct evidence cannot be marked unassessable",
                )
            )
        if negative_flag is True and (
            record.get("expected_region_in_frame") is not True
            or record.get("expected_region_unoccluded") is not True
        ):
            issues.append(
                _issue(
                    "REPORT_EVIDENCE_FLAG_CONFLICT",
                    f"{prefix}: negative evidence requires an in-frame unoccluded region",
                )
            )

        # Newer producers may also expose a short ``visibility`` enum.  If it
        # is present, validate it; the canonical analyzer currently derives
        # visibility from occlusion_state, so this field remains optional for
        # backwards compatibility while the container itself is mandatory.
        explicit_visibility = record.get("visibility")
        if explicit_visibility is not None and (
            not isinstance(explicit_visibility, str)
            or explicit_visibility not in _STEREO_VISIBILITY_VALUES
        ):
            issues.append(
                _issue(
                    "REPORT_VISIBILITY_VALUE_INVALID",
                    f"{prefix}.visibility is unsupported",
                )
            )

    return issues


def _schema_two_pipe_state_issues(
    result: Mapping[str, Any],
    *,
    pipe_label: str,
) -> list[dict[str, str]]:
    """Check that a schema-2 pipe state is backed by its current view flags."""

    state = result.get("installation_state")
    visibility = result.get("visibility_by_view")
    if not isinstance(visibility, Mapping):
        return []
    records = [visibility.get(role) for role in _STEREO_VIEW_ROLES]
    if not all(isinstance(record, Mapping) for record in records):
        return []
    direct = any(record.get("direct_instance_evidence") is True for record in records)
    negative = all(record.get("negative_candidate") is True for record in records)
    if state == "INSTALLED" and not direct:
        return [
            _issue(
                "REPORT_STATE_EVIDENCE_CONFLICT",
                f"{pipe_label}: INSTALLED requires direct_instance_evidence in at least one view",
            )
        ]
    if state == "NOT_INSTALLED" and not negative:
        return [
            _issue(
                "REPORT_STATE_EVIDENCE_CONFLICT",
                f"{pipe_label}: NOT_INSTALLED requires negative_candidate in both views",
            )
        ]
    return []


def _explicitly_unassessable_in_both_stereo_views(
    visibility_by_view: Mapping[str, Mapping[str, Any]],
) -> bool:
    """Protect the UI from displaying a decisive state for total occlusion.

    The safeguard is intentionally narrow: both left and right must explicitly
    report full occlusion or out-of-frustum.  ``NOT_OBSERVED`` is not included,
    because it may accompany separately qualified free-space evidence.
    """

    states: list[str] = []
    for role in ("left", "right"):
        record = visibility_by_view.get(role)
        if not isinstance(record, Mapping):
            return False
        state = record.get("occlusion_state")
        if not isinstance(state, str):
            return False
        states.append(state)
    return all(state in _UNASSESSABLE_VIEW_STATES for state in states)


def build_dashboard_model(
    manifest: Mapping[str, Any],
    report: Mapping[str, Any] | None,
    *,
    manifest_sha256: str | None = None,
    model_actual_sha256: str | None = None,
    photo_actual_sha256: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Bind a report to a CAD manifest and return a GUI-ready view model.

    A report is trusted only when its model hash, exact pipe-id set, unique IDs,
    state enum, and (for schema 2.0) complete left/right evidence records all
    match the manifest.  When ``photo_actual_sha256`` is supplied, the latest
    manifest-bound image pair is also checked against both the manifest and the
    report provenance.  Any global binding failure sets every model pipe to
    ``UNKNOWN``.  This prevents a stale or foreign result from painting the
    correct CAD object with an incorrect installation state.
    """

    manifest = _require_mapping(manifest, "manifest")
    model = _require_mapping(manifest.get("model"), "manifest.model")
    raw_pipes = model.get("pipes")
    if not isinstance(raw_pipes, list) or not raw_pipes:
        raise ValueError("manifest.model.pipes must be a non-empty list")
    strict_stereo_manifest = manifest.get("schema_version") == "2.0"

    pipe_specs: list[dict[str, Any]] = []
    pipe_ids: list[str] = []
    geometry_issues: list[dict[str, str]] = []
    for index, raw_pipe in enumerate(raw_pipes):
        pipe = _require_mapping(raw_pipe, f"manifest.model.pipes[{index}]")
        pipe_id = _require_nonempty_string(
            pipe.get("pipe_id"), f"manifest.model.pipes[{index}].pipe_id"
        )
        pipe_ids.append(pipe_id)
        centerline = _coerce_centerline(pipe.get("centerline_world_mm"))
        diameter = _coerce_positive_number(pipe.get("nominal_diameter_mm"))
        if centerline is None:
            geometry_issues.append(
                _issue(
                    "PIPE_CENTERLINE_UNAVAILABLE",
                    f"{pipe_id}: no valid two-point centerline_world_mm; list-only display",
                )
            )
        if diameter is None:
            geometry_issues.append(
                _issue(
                    "PIPE_DIAMETER_UNAVAILABLE",
                    f"{pipe_id}: no valid nominal_diameter_mm; using a schematic width",
                )
            )
        raw_cad_object_id = pipe.get("cad_object_id", pipe.get("cad_uuid", ""))
        raw_cad_uuid = pipe.get("cad_uuid", "")
        cad_object_id = (
            raw_cad_object_id.strip() if isinstance(raw_cad_object_id, str) else ""
        )
        cad_uuid = raw_cad_uuid.strip() if isinstance(raw_cad_uuid, str) else ""
        pipe_specs.append(
            {
                "pipe_id": pipe_id,
                "cad_object_id": cad_object_id,
                "cad_uuid": cad_uuid,
                "cad_name": str(pipe.get("cad_name", "")),
                "layer_id": str(pipe.get("layer_id", pipe.get("layer_path", ""))),
                "appearance_color": str(
                    pipe.get("appearance_color", pipe.get("color_class", ""))
                ),
                "material_color_srgb": str(
                    pipe.get("nominal_color_srgb", pipe.get("color_srgb", ""))
                ),
                "nominal_diameter_mm": diameter,
                "centerline_world_mm": centerline,
                "drawable": centerline is not None,
            }
        )

    duplicate_manifest_ids = sorted(
        pipe_id for pipe_id, count in Counter(pipe_ids).items() if count > 1
    )
    if duplicate_manifest_ids:
        raise ValueError(
            f"manifest.model.pipes contains duplicate pipe_id values: {duplicate_manifest_ids}"
        )

    expected_model_hash = _normal_sha256(model.get("sha256"))
    binding_issues: list[dict[str, str]] = []
    if strict_stereo_manifest:
        # A schema-2 report is only meaningful when the GUI can draw every
        # manifest-bound CAD object.  The analyzer requires these fields too;
        # reject a forged/hand-edited manifest instead of showing a decisive
        # state for a pipe that has no corresponding geometry in the view.
        binding_issues.extend(geometry_issues)
        cad_object_ids = [spec["cad_object_id"] for spec in pipe_specs]
        duplicate_cad_ids = sorted(
            cad_id
            for cad_id, count in Counter(cad_object_ids).items()
            if cad_id and count > 1
        )
        if duplicate_cad_ids:
            binding_issues.append(
                _issue(
                    "MANIFEST_CAD_OBJECT_ID_DUPLICATE",
                    "schema-2 manifest reuses CAD object IDs: "
                    f"{duplicate_cad_ids}",
                )
            )
        if any(not cad_id for cad_id in cad_object_ids):
            binding_issues.append(
                _issue(
                    "MANIFEST_CAD_OBJECT_ID_INVALID",
                    "schema-2 manifest contains an empty CAD object identity",
                )
            )
    if expected_model_hash is None:
        binding_issues.append(
            _issue(
                "MANIFEST_MODEL_HASH_INVALID",
                "manifest.model.sha256 is missing or is not a 64-character SHA-256",
            )
        )

    if model_actual_sha256 is not None:
        normalized_actual = _normal_sha256(model_actual_sha256)
        if normalized_actual is None:
            binding_issues.append(
                _issue(
                    "MODEL_ASSET_UNAVAILABLE",
                    "The current manifest-bound CAD model could not be hashed",
                )
            )
        elif expected_model_hash is not None and normalized_actual != expected_model_hash:
            binding_issues.append(
                _issue(
                    "CURRENT_MODEL_HASH_MISMATCH",
                    "The current CAD model bytes do not match manifest.model.sha256",
                )
            )

    report_by_id: dict[str, Mapping[str, Any]] = {}
    if report is None:
        binding_issues.append(
            _issue("NO_REPORT", "No recognition report is loaded")
        )
    elif not isinstance(report, Mapping):
        binding_issues.append(
            _issue("REPORT_INVALID", "Recognition report must be an object")
        )
    else:
        strict_stereo_binding = strict_stereo_manifest
        if strict_stereo_binding and report.get("schema_version") != "2.0":
            binding_issues.append(
                _issue(
                    "REPORT_SCHEMA_MISMATCH",
                    "Schema-2 stereo manifests require a schema_version='2.0' report",
                )
            )
        if photo_actual_sha256 is not None and not isinstance(
            photo_actual_sha256, Mapping
        ):
            binding_issues.append(
                _issue(
                    "CURRENT_PHOTO_HASHES_INVALID",
                    "Current manifest-bound photo hashes must be an object",
                )
            )
        report_hashes, malformed_hash = _report_model_hashes(report)
        if malformed_hash:
            binding_issues.append(
                _issue(
                    "REPORT_MODEL_HASH_INVALID",
                    "Recognition report contains a malformed model SHA-256",
                )
            )
        if not report_hashes:
            binding_issues.append(
                _issue(
                    "REPORT_MODEL_HASH_MISSING",
                    "Recognition report does not bind its result to a model SHA-256",
                )
            )
        elif len(report_hashes) > 1:
            binding_issues.append(
                _issue(
                    "REPORT_MODEL_HASH_CONFLICT",
                    "Recognition report contains conflicting model SHA-256 values",
                )
            )
        elif expected_model_hash is not None and report_hashes[0] != expected_model_hash:
            binding_issues.append(
                _issue(
                    "MODEL_HASH_MISMATCH",
                    "Recognition report model SHA-256 does not match the manifest",
                )
            )

        if manifest_sha256 is not None:
            normalized_manifest_hash = _normal_sha256(manifest_sha256)
            report_manifest_hash = _report_manifest_hash(report)
            if normalized_manifest_hash is None:
                binding_issues.append(
                    _issue(
                        "CURRENT_MANIFEST_HASH_INVALID",
                        "The currently loaded manifest snapshot has no valid SHA-256",
                    )
                )
            elif report_manifest_hash is None:
                binding_issues.append(
                    _issue(
                        "REPORT_MANIFEST_HASH_MISSING",
                        "Recognition report does not bind its result to the current manifest",
                    )
                )
            elif report_manifest_hash != normalized_manifest_hash:
                binding_issues.append(
                    _issue(
                        "REPORT_MANIFEST_HASH_MISMATCH",
                        "Recognition report was generated from a different manifest snapshot",
                    )
                )

        expected_revision = manifest.get("model_revision")
        if strict_stereo_binding and (
            not isinstance(expected_revision, str)
            or report.get("model_revision") != expected_revision
        ):
            binding_issues.append(
                _issue(
                    "MODEL_REVISION_MISMATCH",
                    "Recognition report model_revision does not match the stereo manifest",
                )
            )

        capture = manifest.get("capture")
        expected_capture_group = (
            capture.get("capture_group_id") if isinstance(capture, Mapping) else None
        )
        if strict_stereo_binding and (
            not isinstance(expected_capture_group, str)
            or report.get("capture_group_id") != expected_capture_group
        ):
            binding_issues.append(
                _issue(
                    "CAPTURE_GROUP_MISMATCH",
                    "Recognition report does not belong to the current capture group",
                )
            )

        calibration = manifest.get("stereo_calibration")
        expected_calibration = (
            calibration.get("calibration_id")
            if isinstance(calibration, Mapping)
            else None
        )
        calibration_audit = report.get("calibration_audit")
        report_calibration = (
            calibration_audit.get("calibration_id")
            if isinstance(calibration_audit, Mapping)
            else None
        )
        if strict_stereo_binding and (
            not isinstance(expected_calibration, str)
            or report_calibration != expected_calibration
        ):
            binding_issues.append(
                _issue(
                    "CALIBRATION_ID_MISMATCH",
                    "Recognition report calibration_id does not match the stereo manifest",
                )
            )

        raw_report_pipes = report.get("pipes")
        if not isinstance(raw_report_pipes, list):
            binding_issues.append(
                _issue("REPORT_PIPES_INVALID", "Recognition report.pipes must be a list")
            )
        else:
            duplicate_report_ids: set[str] = set()
            malformed_pipe = False
            invalid_state = False
            for index, raw_result in enumerate(raw_report_pipes):
                if not isinstance(raw_result, Mapping):
                    malformed_pipe = True
                    continue
                result_id = raw_result.get("pipe_id")
                state = raw_result.get("installation_state")
                if not isinstance(result_id, str) or not result_id:
                    malformed_pipe = True
                    continue
                if strict_stereo_binding:
                    binding_issues.extend(
                        _schema_two_view_issues(
                            raw_result,
                            pipe_label=f"report.pipes[{index}]({result_id})",
                        )
                    )
                    binding_issues.extend(
                        _schema_two_pipe_state_issues(
                            raw_result,
                            pipe_label=f"report.pipes[{index}]({result_id})",
                        )
                    )
                if result_id in report_by_id:
                    duplicate_report_ids.add(result_id)
                    continue
                report_by_id[result_id] = raw_result
                if state not in INSTALLATION_STATES:
                    invalid_state = True
            if malformed_pipe:
                binding_issues.append(
                    _issue(
                        "REPORT_PIPE_INVALID",
                        "Recognition report contains a pipe entry without a valid pipe_id",
                    )
                )
            if duplicate_report_ids:
                binding_issues.append(
                    _issue(
                        "REPORT_PIPE_ID_DUPLICATE",
                        f"Recognition report contains duplicate pipe IDs: {sorted(duplicate_report_ids)}",
                    )
                )
            if invalid_state:
                binding_issues.append(
                    _issue(
                        "REPORT_STATE_INVALID",
                        "Recognition report contains an unsupported installation_state",
                    )
                )
            expected_ids = set(pipe_ids)
            actual_ids = set(report_by_id)
            if expected_ids != actual_ids:
                missing = sorted(expected_ids - actual_ids)
                extra = sorted(actual_ids - expected_ids)
                binding_issues.append(
                    _issue(
                        "REPORT_PIPE_ID_SET_MISMATCH",
                        f"Recognition report pipe IDs differ from the model; missing={missing}, extra={extra}",
                    )
                )
            if strict_stereo_binding and expected_ids == actual_ids:
                mismatched_bindings = sorted(
                    spec["pipe_id"]
                    for spec in pipe_specs
                    if str(report_by_id[spec["pipe_id"]].get("cad_object_id", ""))
                    != spec["cad_object_id"]
                )
                if mismatched_bindings:
                    binding_issues.append(
                        _issue(
                            "CAD_OBJECT_BINDING_MISMATCH",
                            "Recognition report CAD object bindings differ for "
                            f"pipe IDs: {mismatched_bindings}",
                        )
                    )
                mismatched_uuids = sorted(
                    spec["pipe_id"]
                    for spec in pipe_specs
                    if spec["cad_uuid"]
                    and str(report_by_id[spec["pipe_id"]].get("cad_uuid", ""))
                    != spec["cad_uuid"]
                )
                if mismatched_uuids:
                    binding_issues.append(
                        _issue(
                            "CAD_UUID_BINDING_MISMATCH",
                            "Recognition report CAD UUID bindings differ for "
                            f"pipe IDs: {mismatched_uuids}",
                        )
                    )

        report_model = report.get("model")
        if isinstance(report_model, Mapping) and report_model.get("verified") is False:
            binding_issues.append(
                _issue(
                    "REPORT_MODEL_NOT_VERIFIED",
                    "Recognition report explicitly marks the model asset as unverified",
                )
            )
        inputs = report.get("inputs")
        if isinstance(inputs, Mapping):
            input_model = inputs.get("model")
            if isinstance(input_model, Mapping) and input_model.get("verified") is False:
                binding_issues.append(
                    _issue(
                        "REPORT_MODEL_NOT_VERIFIED",
                        "Recognition report input model is not verified",
                    )
                )

        if strict_stereo_binding and photo_actual_sha256 is not None:
            # Verify the bytes currently on disk against the manifest's latest
            # pair and the report's recorded provenance.  A changed image must
            # never leave a previously decisive state visible in the GUI.
            capture = manifest.get("capture")
            groups = (
                capture.get(
                    "capture_groups",
                    capture.get("captures", capture.get("pairs")),
                )
                if isinstance(capture, Mapping)
                else None
            )
            latest_manifest_group = groups[-1] if isinstance(groups, list) and groups else None
            expected_latest_capture_id = (
                latest_manifest_group.get("capture_id")
                if isinstance(latest_manifest_group, Mapping)
                else None
            )
            latest_views = (
                latest_manifest_group.get("views")
                if isinstance(latest_manifest_group, Mapping)
                else None
            )
            if not isinstance(latest_views, Mapping):
                binding_issues.append(
                    _issue(
                        "MANIFEST_PHOTO_BINDING_INVALID",
                        "Stereo manifest has no latest left/right photo views",
                    )
                )
            current_hashes: Mapping[str, str] = (
                photo_actual_sha256
                if isinstance(photo_actual_sha256, Mapping)
                else {}
            )
            report_photo_hashes, report_photo_malformed = _report_latest_photo_hashes(
                report
            )
            report_audit = report.get("capture_audit")
            report_groups = (
                report_audit.get("groups")
                if isinstance(report_audit, Mapping)
                else None
            )
            report_latest_group = (
                report_groups[-1]
                if isinstance(report_groups, list) and report_groups
                else None
            )
            report_latest_capture_id = (
                report_latest_group.get("capture_id")
                if isinstance(report_latest_group, Mapping)
                else None
            )
            if (
                not isinstance(expected_latest_capture_id, str)
                or report_latest_capture_id != expected_latest_capture_id
            ):
                binding_issues.append(
                    _issue(
                        "CAPTURE_ID_MISMATCH",
                        "Recognition report latest capture_id does not match the manifest",
                    )
                )
            if report_photo_malformed:
                binding_issues.append(
                    _issue(
                        "REPORT_PHOTO_HASH_INVALID",
                        "Recognition report latest capture is missing valid left/right photo hashes",
                    )
                )
            for role in _STEREO_VIEW_ROLES:
                view = latest_views.get(role) if isinstance(latest_views, Mapping) else None
                expected = (
                    _normal_sha256(view.get("sha256"))
                    if isinstance(view, Mapping)
                    else None
                )
                actual = _normal_sha256(current_hashes.get(role))
                if expected is None:
                    binding_issues.append(
                        _issue(
                            "MANIFEST_PHOTO_HASH_INVALID",
                            f"manifest capture {role} view has no valid SHA-256",
                        )
                    )
                if actual is None:
                    binding_issues.append(
                        _issue(
                            "CURRENT_PHOTO_UNAVAILABLE",
                            f"Current manifest-bound {role} photo is unavailable or unhashed",
                        )
                    )
                elif expected is not None and actual != expected:
                    binding_issues.append(
                        _issue(
                            "CURRENT_PHOTO_HASH_MISMATCH",
                            f"Current manifest-bound {role} photo bytes differ from the manifest",
                        )
                    )
                recorded = report_photo_hashes.get(role)
                if recorded is None:
                    # The malformed flag above gives the report-level reason;
                    # retain a role-specific code for diagnostics as well.
                    binding_issues.append(
                        _issue(
                            "REPORT_PHOTO_HASH_MISSING",
                            f"Recognition report has no valid {role} photo hash",
                        )
                    )
                elif actual is not None and recorded != actual:
                    binding_issues.append(
                        _issue(
                            "REPORT_CURRENT_PHOTO_HASH_MISMATCH",
                            f"Recognition report {role} photo hash differs from the current bytes",
                        )
                    )

    binding_valid = not binding_issues
    dashboard_pipes: list[dict[str, Any]] = []
    per_pipe_safety_issues: list[dict[str, str]] = []
    for spec in pipe_specs:
        pipe_id = spec["pipe_id"]
        result = report_by_id.get(pipe_id) if binding_valid else None
        if result is None:
            state = "UNKNOWN"
            basis = "GUI_FAIL_CLOSED" if report is not None else "NO_REPORT"
            reason_codes = [issue["code"] for issue in binding_issues] or ["NO_REPORT"]
            visibility: dict[str, dict[str, Any]] = {}
        else:
            state = str(result["installation_state"])
            basis = str(result.get("state_basis", "REPORT_RESULT"))
            reason_codes = _safe_reason_codes(result.get("reason_codes"))
            visibility = _safe_visibility(result.get("visibility_by_view"))
            if state != "UNKNOWN" and _explicitly_unassessable_in_both_stereo_views(
                visibility
            ):
                state = "UNKNOWN"
                basis = "GUI_TOTAL_OCCLUSION_FAIL_CLOSED"
                reason_codes = [
                    "FULLY_UNASSESSABLE_IN_BOTH_STEREO_VIEWS",
                    *reason_codes,
                ]
                per_pipe_safety_issues.append(
                    _issue(
                        "DECISIVE_STATE_WITH_TOTAL_OCCLUSION",
                        f"{pipe_id}: report state was suppressed because both stereo views are unassessable",
                    )
                )

        presentation = STATE_PRESENTATION[state]
        dashboard_pipes.append(
            {
                **spec,
                "installation_state": state,
                "installation_state_zh": presentation["label_zh"],
                "state_color": presentation["color"],
                "state_symbol": presentation["symbol"],
                "state_dash": presentation["dash"],
                "state_basis": basis,
                "reason_codes": reason_codes,
                "visibility_by_view": visibility,
            }
        )

    counts = Counter(pipe["installation_state"] for pipe in dashboard_pipes)
    model_path = model.get("path")
    return {
        "binding_valid": binding_valid,
        "binding_issues": binding_issues,
        "display_issues": geometry_issues + per_pipe_safety_issues,
        "model": {
            "path": str(model_path) if model_path is not None else "",
            "sha256": expected_model_hash,
            "format": Path(str(model_path)).suffix.lower().lstrip(".")
            if model_path
            else "",
        },
        "counts": {state: counts[state] for state in INSTALLATION_STATES},
        "pipes": dashboard_pipes,
    }


def _project_dashboard_pipes(
    dashboard: Mapping[str, Any],
    width: int,
    height: int,
    mode: str = "elevation",
) -> list[dict[str, Any]]:
    """Project CAD centerlines into deterministic canvas coordinates."""

    if width <= 0 or height <= 0:
        raise ValueError("Projection dimensions must be positive")
    if mode not in {"elevation", "isometric"}:
        raise ValueError("Projection mode must be 'elevation' or 'isometric'")

    projected: list[tuple[dict[str, Any], list[tuple[float, float]], float]] = []
    for raw_pipe in dashboard.get("pipes", []):
        if not isinstance(raw_pipe, Mapping):
            continue
        centerline = _coerce_centerline(raw_pipe.get("centerline_world_mm"))
        if centerline is None:
            continue
        points: list[tuple[float, float]] = []
        average_z = sum(point[2] for point in centerline) / 2.0
        for x, y, z in centerline:
            if mode == "elevation":
                u, v = x, -y
            else:
                u = 0.8660254037844386 * (x - z)
                v = -(y + 0.5 * (x + z))
            points.append((u, v))
        projected.append((dict(raw_pipe), points, average_z))

    if not projected:
        return []

    all_u = [value for _, points, _ in projected for value, _ in points]
    all_v = [value for _, points, _ in projected for _, value in points]
    max_radius = max(
        (
            float(pipe.get("nominal_diameter_mm") or 2.0) / 2.0
            for pipe, _, _ in projected
        ),
        default=1.0,
    )
    min_u, max_u = min(all_u) - max_radius, max(all_u) + max_radius
    min_v, max_v = min(all_v) - max_radius, max(all_v) + max_radius
    margin = 48.0
    available_width = max(1.0, width - 2.0 * margin)
    available_height = max(1.0, height - 2.0 * margin)
    span_u = max(max_u - min_u, 1.0)
    span_v = max(max_v - min_v, 1.0)
    scale = min(available_width / span_u, available_height / span_v)
    offset_u = (width - span_u * scale) / 2.0 - min_u * scale
    offset_v = (height - span_v * scale) / 2.0 - min_v * scale

    result: list[dict[str, Any]] = []
    # More-negative Z is the rear layer in the current world convention.
    for pipe, points, average_z in sorted(projected, key=lambda item: item[2]):
        diameter = float(pipe.get("nominal_diameter_mm") or 2.0)
        canvas_points = [
            [u * scale + offset_u, v * scale + offset_v] for u, v in points
        ]
        result.append(
            {
                **pipe,
                "canvas_centerline": canvas_points,
                "canvas_width": max(3.0, min(72.0, diameter * scale)),
            }
        )
    return result


def _read_json_object(path: str | Path, field: str) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{field} must contain a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _loaded_asset_hashes(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> tuple[str, str]:
    manifest_hash = _sha256_file(manifest_path)
    model = manifest.get("model")
    model_path = (
        _safe_manifest_asset_path(manifest_path, model.get("path"))
        if isinstance(model, Mapping)
        else None
    )
    model_hash = _sha256_file(model_path) if model_path is not None and model_path.is_file() else ""
    return manifest_hash, model_hash


def _loaded_photo_hashes(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> dict[str, str]:
    """Hash the latest manifest-bound left/right pair for GUI provenance checks.

    Missing or unreadable files are represented by an empty string.  The
    dashboard validator turns that into a binding failure; it must not fall
    back to a manually selected preview image.
    """

    paths = _first_manifest_stereo_paths(manifest_path, manifest)
    result: dict[str, str] = {}
    for role in _STEREO_VIEW_ROLES:
        path = paths.get(role)
        if path is None or not path.is_file():
            result[role] = ""
            continue
        try:
            result[role] = _sha256_file(path)
        except OSError:
            result[role] = ""
    return result


def _empty_dashboard(
    *,
    reason_code: str = "NO_ACTIVE_SOURCES",
    message: str = "No manifest/report is loaded",
) -> dict[str, Any]:
    """Return a structurally complete, all-unknown dashboard placeholder."""

    return {
        "binding_valid": False,
        "binding_issues": [_issue(reason_code, message)],
        "display_issues": [],
        "model": {"path": "", "sha256": None, "format": ""},
        "counts": {state: 0 for state in INSTALLATION_STATES},
        "pipes": [],
    }


def _safe_manifest_asset_path(manifest_path: Path, relative_path: object) -> Path | None:
    """Resolve a manifest-relative asset while preserving its user-facing spelling.

    ``Path.resolve()`` is useful for the containment check below, but on
    Windows it may return an 8.3 short path (for example ``RUNNER~1``) even
    when the caller supplied the long path.  Returning that canonical path
    makes otherwise equal ``Path`` values compare unequal and is especially
    surprising for the GUI preview API.  Keep a lexical absolute path for the
    returned value, and use a separately canonicalized path only to reject
    symlink/junction escapes.
    """
    if not isinstance(relative_path, str) or not relative_path:
        return None
    candidate = Path(relative_path)
    if candidate.is_absolute():
        return None
    # ``abspath`` normalizes ``.``/``..`` without asking the filesystem for a
    # final path, so it retains the long path spelling on Windows.  This is
    # intentionally distinct from ``resolve(strict=False)`` used below.
    root = Path(os.path.abspath(os.fspath(manifest_path))).parent
    lexical = Path(
        os.path.normpath(os.path.join(os.fspath(root), os.fspath(candidate)))
    )
    try:
        canonical_root = root.resolve(strict=False)
        canonical = lexical.resolve(strict=False)
        canonical.relative_to(canonical_root)
    except (OSError, RuntimeError, ValueError):
        return None
    return lexical


def _first_manifest_stereo_paths(
    manifest_path: Path, manifest: Mapping[str, Any]
) -> dict[str, Path]:
    capture = manifest.get("capture")
    if not isinstance(capture, Mapping):
        return {}
    groups: object = capture.get(
        "capture_groups", capture.get("captures", capture.get("pairs"))
    )
    if not isinstance(groups, list) or not groups:
        return {}
    # Final pipe state and visibility_by_view refer to the most recent group.
    first = groups[-1]
    if not isinstance(first, Mapping):
        return {}
    views = first.get("views")
    if not isinstance(views, Mapping):
        pairs = first.get("pairs")
        if isinstance(pairs, list) and pairs and isinstance(pairs[0], Mapping):
            views = pairs[0].get("views")
    if not isinstance(views, Mapping):
        return {}
    result: dict[str, Path] = {}
    for role in ("left", "right"):
        view = views.get(role)
        if not isinstance(view, Mapping):
            continue
        path = _safe_manifest_asset_path(manifest_path, view.get("path"))
        if path is not None and path.is_file():
            result[role] = path
    return result


class _PipeTwinApplication:
    """Tk adapter.  It is instantiated only by :func:`launch_gui`."""

    def __init__(
        self,
        root: Any,
        manifest_path: str | Path | None,
        report_path: str | Path | None,
    ) -> None:
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        self.tk = tk
        self.ttk = ttk
        self.filedialog = filedialog
        self.messagebox = messagebox
        self.root = root
        self.manifest_path = Path(manifest_path).resolve() if manifest_path else None
        self.report_path = Path(report_path).resolve() if report_path else None
        self.manifest: dict[str, Any] = {}
        self.report: dict[str, Any] | None = None
        self.dashboard: dict[str, Any] = _empty_dashboard(message="载入现场清单，或打开合成示例开始使用。")
        self.selected_pipe_id: str | None = None
        self.projection_mode = tk.StringVar(value="elevation")
        self.banner_text = tk.StringVar(value="正在载入……")
        self.summary_text = tk.StringVar(value="")
        self.detail_text = tk.StringVar(value="")
        self._worker_messages: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._analysis_running = False
        self._photo_paths: dict[str, Path] = {}
        self._photo_images: dict[str, Any] = {}
        self._photo_geometry: dict[str, tuple[float, float, int, int]] = {}
        self.dxf_elevation: DxfElevation | None = None
        self.dxf_layer_colors: dict[str, str] = {}
        self.dxf_bindings: dict[str, str] = {}
        self.selected_dxf_entity_id: str | None = None
        self._manifest_sha256: str | None = None
        self._model_actual_sha256: str | None = None
        self._photo_actual_sha256: dict[str, str] = {}

        self.root.title("管件测量工作台 · 外径 / 中心距 / 净距 / 前后状态")
        self.root.geometry("1480x900")
        self.root.minsize(1180, 780)
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", font=("Microsoft YaHei UI", 10))
        style.configure("Treeview", rowheight=28)
        self._build_widgets()
        if self.manifest_path:
            self._load_sources(self.manifest_path, self.report_path)
        else:
            self._refresh_dashboard()
        self.root.after(80, self._poll_worker)

    def _build_widgets(self) -> None:
        tk, ttk = self.tk, self.ttk
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="载入清单", command=self._choose_manifest).pack(
            side="left", padx=3
        )
        ttk.Button(toolbar, text="载入识别结果", command=self._choose_report).pack(
            side="left", padx=3
        )
        ttk.Button(toolbar, text="现场数据录入", command=self._input_capture).pack(side="left", padx=3)
        ttk.Button(toolbar, text="打开合成示例", command=self._open_demo).pack(side="left", padx=3)
        ttk.Button(toolbar, text="导入DXF侧立面", command=self._import_dxf).pack(side="left", padx=3)
        ttk.Button(toolbar, text="DXF自动建档", command=self._automate_dxf_setup).pack(side="left", padx=3)
        ttk.Button(toolbar, text="从DXF生成manifest草稿", command=self._create_dxf_manifest_draft).pack(side="left", padx=3)
        ttk.Button(toolbar, text="指定图层颜色", command=self._choose_dxf_layer_color).pack(side="left", padx=3)
        ttk.Button(toolbar, text="保存DXF颜色配置", command=self._save_dxf_colors).pack(side="left", padx=3)
        ttk.Button(toolbar, text="绑定DXF图元", command=self._bind_dxf_entity).pack(side="left", padx=3)
        ttk.Button(toolbar, text="保存DXF映射到manifest", command=self._save_dxf_bindings).pack(side="left", padx=3)
        self.run_button = ttk.Button(toolbar, text="运行双目识别与测量", command=self._run_analysis)
        self.run_button.pack(side="left", padx=(14, 3))
        ttk.Label(toolbar, text="模型视图：").pack(side="left", padx=(18, 2))
        projection = ttk.Combobox(
            toolbar,
            textvariable=self.projection_mode,
            values=("elevation", "isometric", "dxf"),
            state="readonly",
            width=11,
        )
        projection.pack(side="left")
        projection.bind("<<ComboboxSelected>>", lambda _event: self._draw_model())

        self.banner = tk.Label(
            self.root,
            textvariable=self.banner_text,
            anchor="w",
            padx=10,
            pady=7,
            bg="#FFF3CD",
            fg="#5F4500",
        )
        self.banner.pack(fill="x")

        self.main_tabs = ttk.Notebook(self.root)
        self.main_tabs.pack(fill="both", expand=True)
        measurement_tab = ttk.Frame(self.main_tabs)
        cad_tab = ttk.Frame(self.main_tabs)
        self.main_tabs.add(measurement_tab, text="测量工作台")
        self.main_tabs.add(cad_tab, text="CAD 与识别证据")
        from .measurement_gui import MeasurementPanel

        self.measurement_panel = MeasurementPanel(measurement_tab, self)
        panes = ttk.Panedwindow(cad_tab, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=8, pady=8)
        left = ttk.Frame(panes)
        right = ttk.Frame(panes, width=560)
        panes.add(left, weight=3)
        panes.add(right, weight=2)

        notebook = ttk.Notebook(left)
        notebook.pack(fill="both", expand=True)
        model_tab = ttk.Frame(notebook)
        stereo_tab = ttk.Frame(notebook)
        notebook.add(model_tab, text="CAD 状态视图")
        notebook.add(stereo_tab, text="左右照片")
        preview_toolbar = ttk.Frame(stereo_tab)
        preview_toolbar.pack(fill="x")
        for role, label in (("left", "左"), ("right", "右")):
            ttk.Button(preview_toolbar, text=f"选择{label}图（仅预览）", command=lambda r=role: self._choose_photo(r)).pack(side="left", padx=3)

        self.model_canvas = tk.Canvas(model_tab, bg="#F4F6F8", highlightthickness=0)
        self.model_canvas.pack(fill="both", expand=True)
        self.model_canvas.bind("<Configure>", lambda _event: self._draw_model())
        self.model_canvas.bind("<Button-1>", self._select_canvas_pipe)

        stereo_panes = ttk.Panedwindow(stereo_tab, orient="horizontal")
        stereo_panes.pack(fill="both", expand=True)
        self.photo_canvases: dict[str, Any] = {}
        for role, title in (("left", "左相机"), ("right", "右相机")):
            frame = ttk.LabelFrame(stereo_panes, text=title, padding=4)
            canvas = tk.Canvas(frame, bg="#202124", highlightthickness=0)
            canvas.pack(fill="both", expand=True)
            canvas.bind("<Configure>", lambda _event, r=role: self._render_photo(r))
            self.photo_canvases[role] = canvas
            stereo_panes.add(frame, weight=1)

        ttk.Label(right, textvariable=self.summary_text, anchor="w").pack(
            fill="x", pady=(0, 6)
        )
        columns = (
            "state",
            "diameter",
            "layer",
            "left",
            "right",
            "reason",
        )
        self.tree = ttk.Treeview(right, columns=columns, show="tree headings", selectmode="browse")
        self.tree.heading("#0", text="pipe_id")
        self.tree.heading("state", text="状态")
        self.tree.heading("diameter", text="设计外径/mm")
        self.tree.heading("layer", text="层")
        self.tree.heading("left", text="左目")
        self.tree.heading("right", text="右目")
        self.tree.heading("reason", text="依据")
        self.tree.column("#0", width=220, stretch=True)
        self.tree.column("state", width=115, anchor="center")
        self.tree.column("diameter", width=72, anchor="e")
        self.tree.column("layer", width=70, anchor="center")
        self.tree.column("left", width=125, anchor="center")
        self.tree.column("right", width=125, anchor="center")
        self.tree.column("reason", width=230, stretch=True)
        for state, presentation in STATE_PRESENTATION.items():
            self.tree.tag_configure(state, foreground=presentation["color"])
        scroll_y = ttk.Scrollbar(right, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll_y.set)
        self.tree.pack(side="left", fill="both", expand=True)
        scroll_y.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._select_tree_pipe)

        detail = ttk.Label(
            cad_tab,
            textvariable=self.detail_text,
            anchor="w",
            justify="left",
            padding=(10, 5),
        )
        detail.pack(fill="x")

    def _current_photo_hashes(self) -> dict[str, str]:
        """Refresh hashes for the manifest-bound pair, never preview overrides."""

        if not self.manifest:
            return {}
        try:
            return _loaded_photo_hashes(self.manifest_path, self.manifest)
        except (OSError, ValueError):
            return {role: "" for role in _STEREO_VIEW_ROLES}

    def _clear_loaded_state(
        self,
        *,
        manifest: object = None,
        manifest_path: Path | None = None,
        reason_code: str = "NO_ACTIVE_SOURCES",
        message: str = "No manifest/report is loaded",
        preserve_manifest: bool = False,
    ) -> None:
        """Drop stale report state and repaint a safe dashboard after an error."""

        candidate_manifest = manifest if isinstance(manifest, dict) else None
        if preserve_manifest and candidate_manifest is not None:
            candidate_path = (
                Path(manifest_path).resolve()
                if manifest_path is not None
                else self.manifest_path
            )
            try:
                # A parsed manifest can still be displayed as an all-unknown
                # CAD list even when its file was removed immediately after
                # parsing.  Hash each source independently and let the normal
                # binding checks record unavailable assets.
                try:
                    manifest_hash = _sha256_file(candidate_path)
                except OSError:
                    manifest_hash = None
                model_spec = candidate_manifest.get("model")
                model_path = (
                    _safe_manifest_asset_path(candidate_path, model_spec.get("path"))
                    if isinstance(model_spec, Mapping)
                    else None
                )
                try:
                    model_hash = (
                        _sha256_file(model_path)
                        if model_path is not None and model_path.is_file()
                        else ""
                    )
                except OSError:
                    model_hash = ""
                photo_hashes = _loaded_photo_hashes(candidate_path, candidate_manifest)
                safe_dashboard = build_dashboard_model(
                    candidate_manifest,
                    None,
                    manifest_sha256=manifest_hash,
                    model_actual_sha256=model_hash or None,
                    photo_actual_sha256=photo_hashes,
                )
            except Exception:
                safe_dashboard = _empty_dashboard(
                    reason_code=reason_code,
                    message=message,
                )
            self.manifest = candidate_manifest
            self.manifest_path = candidate_path
            self._manifest_sha256 = (
                manifest_hash if "manifest_hash" in locals() else None
            )
            self._model_actual_sha256 = (
                model_hash if "model_hash" in locals() else None
            )
            self._photo_actual_sha256 = (
                photo_hashes if "photo_hashes" in locals() else {}
            )
        else:
            self.manifest = {}
            if manifest_path is not None:
                self.manifest_path = Path(manifest_path).resolve()
            self._manifest_sha256 = None
            self._model_actual_sha256 = None
            self._photo_actual_sha256 = {}
            safe_dashboard = _empty_dashboard(
                reason_code=reason_code,
                message=message,
            )
        self.report = None
        self.report_path = None
        self.dashboard = safe_dashboard
        self.selected_pipe_id = None
        self._photo_paths.clear()
        self._photo_images.clear()
        self._photo_geometry.clear()
        if preserve_manifest and self.manifest:
            self._photo_paths.update(
                _first_manifest_stereo_paths(self.manifest_path, self.manifest)
            )
        # During construction or headless tests the Tk widgets may not yet
        # exist.  State is still cleared; repaint only when the adapter is
        # fully initialized.
        if hasattr(self, "tree"):
            self._refresh_dashboard()
            for role in _STEREO_VIEW_ROLES:
                self._render_photo(role)

    def _load_sources(self, manifest_path: Path, report_path: Path | None) -> None:
        manifest_path = Path(manifest_path).resolve()
        report_path = Path(report_path).resolve() if report_path is not None else None
        log_event(_LOGGER, "gui_sources_load_start", manifest=manifest_path, report=report_path)
        try:
            manifest = _read_json_object(manifest_path, "manifest")
            report = _read_json_object(report_path, "report") if report_path else None
            manifest_sha256, model_actual_sha256 = _loaded_asset_hashes(
                manifest_path, manifest
            )
            photo_actual_sha256 = _loaded_photo_hashes(manifest_path, manifest)
            dashboard = build_dashboard_model(
                manifest,
                report,
                manifest_sha256=manifest_sha256,
                model_actual_sha256=model_actual_sha256,
                photo_actual_sha256=photo_actual_sha256,
            )
        except Exception as error:
            # Never leave a previously loaded decisive result on screen after
            # a failed source switch.  If the new manifest parsed, retain its
            # CAD list but rebuild it without a report (all pipes UNKNOWN);
            # otherwise replace the dashboard with an empty fail-closed model.
            self._clear_loaded_state(
                manifest=locals().get("manifest"),
                manifest_path=manifest_path,
                reason_code="LOAD_FAILED",
                message=f"载入失败：{error}",
                preserve_manifest=isinstance(locals().get("manifest"), dict),
            )
            self.messagebox.showerror("载入失败", str(error))
            return
        self.manifest_path = manifest_path
        self.report_path = report_path
        self.manifest = manifest
        self.report = report
        self.dashboard = dashboard
        self._restore_dxf_from_manifest(manifest_path, manifest)
        self._manifest_sha256 = manifest_sha256
        self._model_actual_sha256 = model_actual_sha256
        self._photo_actual_sha256 = photo_actual_sha256
        self.selected_pipe_id = None
        self._photo_paths = _first_manifest_stereo_paths(self.manifest_path, self.manifest)
        self._photo_images.clear()
        self._photo_geometry.clear()
        self._refresh_dashboard()
        log_event(
            _LOGGER,
            "gui_sources_loaded",
            manifest=manifest_path,
            report=report_path,
            pipe_count=len(self.dashboard.get("pipes", [])),
            binding_valid=self.dashboard.get("binding_valid"),
        )
        for role in ("left", "right"):
            self._render_photo(role)

    def _restore_dxf_from_manifest(self, manifest_path: Path, manifest: Mapping[str, Any]) -> None:
        elevation = manifest.get("elevation")
        if not isinstance(elevation, Mapping) or elevation.get("format") != "dxf":
            return
        path = _safe_manifest_asset_path(manifest_path, elevation.get("path"))
        try:
            document = read_dxf_elevation(path) if path is not None else None
            if document is None or str(elevation.get("sha256", "")).lower() != document.source_sha256:
                raise ValueError("DXF 文件哈希与 manifest 不一致")
            self.dxf_elevation = document
            self.dxf_layer_colors = dict(document.layers)
            if isinstance(elevation.get("layer_colors"), Mapping):
                self.dxf_layer_colors.update({str(k): str(v).upper() for k, v in elevation["layer_colors"].items()})
            self.dxf_bindings = {str(k): str(v) for k, v in (elevation.get("entity_bindings") or {}).items()} if isinstance(elevation.get("entity_bindings"), Mapping) else {}
        except (OSError, ValueError):
            self.dxf_elevation = None
            self.dxf_layer_colors = {}
            self.dxf_bindings = {}

    def _choose_manifest(self) -> None:
        selected = self.filedialog.askopenfilename(
            title="选择双目采集清单",
            filetypes=(("JSON", "*.json"), ("All files", "*.*")),
        )
        if selected:
            self._load_sources(Path(selected), None)

    def _choose_report(self) -> None:
        if not self.manifest:
            self.messagebox.showinfo("先载入清单", "请先载入现场清单或打开示例，再载入对应识别结果。")
            return
        selected = self.filedialog.askopenfilename(
            title="选择识别结果",
            filetypes=(("JSON", "*.json"), ("All files", "*.*")),
        )
        if not selected:
            return
        current_photo_hashes = self._current_photo_hashes()
        try:
            report = _read_json_object(selected, "report")
            dashboard = build_dashboard_model(
                self.manifest,
                report,
                manifest_sha256=self._manifest_sha256,
                model_actual_sha256=self._model_actual_sha256,
                photo_actual_sha256=current_photo_hashes,
            )
        except Exception as error:
            # Keep the CAD manifest useful, but discard the old report and its
            # dashboard immediately.  Rebuilding with ``None`` guarantees no
            # old INSTALLED/NOT_INSTALLED state survives a bad report load.
            self._clear_loaded_state(
                manifest=self.manifest,
                manifest_path=self.manifest_path,
                reason_code="REPORT_LOAD_FAILED",
                message=f"结果载入失败：{error}",
                preserve_manifest=True,
            )
            self.messagebox.showerror("结果载入失败", str(error))
            return
        self.report_path = Path(selected).resolve()
        self.report = report
        self.dashboard = dashboard
        self._photo_actual_sha256 = current_photo_hashes
        self.selected_pipe_id = None
        self._refresh_dashboard()

    def _choose_photo(self, role: str) -> None:
        selected = self.filedialog.askopenfilename(
            title=f"选择{'左' if role == 'left' else '右'}相机照片（仅预览）",
            filetypes=(
                ("Images", "*.png *.jpg *.jpeg"),
                ("All files", "*.*"),
            ),
        )
        if selected:
            self._photo_paths[role] = Path(selected).resolve()
            self._render_photo(role)

    def _refresh_dashboard(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        counts = self.dashboard["counts"]
        self.summary_text.set(
            f"安装 {counts['INSTALLED']}  |  未安装 {counts['NOT_INSTALLED']}  |  "
            f"不确定 {counts['UNKNOWN']}"
        )
        for pipe in self.dashboard["pipes"]:
            visibility = pipe["visibility_by_view"]
            left = self._view_label(visibility.get("left"))
            right = self._view_label(visibility.get("right"))
            reasons = ", ".join(pipe["reason_codes"]) or pipe["state_basis"]
            diameter = pipe["nominal_diameter_mm"]
            self.tree.insert(
                "",
                "end",
                iid=pipe["pipe_id"],
                text=pipe["pipe_id"],
                values=(
                    f"{pipe['state_symbol']} {pipe['installation_state_zh']}",
                    f"{diameter:g}" if diameter is not None else "—",
                    pipe["layer_id"] or "—",
                    left,
                    right,
                    reasons,
                ),
                tags=(pipe["installation_state"],),
            )
        if self.dashboard["binding_valid"]:
            self.banner.configure(bg="#D7F3E3", fg="#174D2D")
            self.banner_text.set(
                f"结果与 CAD 模型绑定有效：{self.dashboard['model']['path']}"
            )
        else:
            self.banner.configure(bg="#FFF3CD", fg="#5F4500")
            messages = "；".join(
                issue["message"] for issue in self.dashboard["binding_issues"]
            )
            self.banner_text.set(f"安全降级：全部管道显示为不确定。{messages}")
        self._draw_model()
        self._update_detail()
        if hasattr(self, "measurement_panel"):
            self.measurement_panel.refresh()
        if not self.manifest:
            self.banner_text.set("开始：载入现场清单 / 现场数据录入；也可打开合成示例熟悉操作。")
        elif "SYNTHETIC" in self.manifest.get("validation_scope", ""):
            self.banner_text.set("合成示例数据 · 可用于软件操作与算法回归；现场精度需要真实双目照片与实测样本验证。")

    @staticmethod
    def _view_label(record: object) -> str:
        if not isinstance(record, Mapping):
            return "未报告"
        value = record.get("occlusion_state", record.get("visibility", "未报告"))
        return str(value)

    def _draw_model(self) -> None:
        if not self.dashboard:
            return
        canvas = self.model_canvas
        width = max(canvas.winfo_width(), 400)
        height = max(canvas.winfo_height(), 300)
        canvas.delete("all")
        mode = self.projection_mode.get()
        if mode == "dxf":
            self._draw_dxf()
            return
        projected = _project_dashboard_pipes(self.dashboard, width, height, mode)
        canvas.create_text(
            16,
            14,
            anchor="nw",
            text=(
                "CAD 状态立面（X/Y）" if mode == "elevation" else "CAD 状态等轴示意"
            ),
            fill="#263238",
            font=("Segoe UI", 12, "bold"),
        )
        legend_x = 18
        for state in INSTALLATION_STATES:
            style = STATE_PRESENTATION[state]
            canvas.create_text(
                legend_x,
                42,
                anchor="nw",
                text=f"{style['symbol']} {style['label_zh']} ({state})",
                fill=style["color"],
                font=("Segoe UI", 10, "bold"),
            )
            legend_x += 165

        for pipe in projected:
            (x1, y1), (x2, y2) = pipe["canvas_centerline"]
            selected = pipe["pipe_id"] == self.selected_pipe_id
            if selected:
                canvas.create_line(
                    x1,
                    y1,
                    x2,
                    y2,
                    width=pipe["canvas_width"] + 7,
                    fill="#1565C0",
                    capstyle="round",
                )
            canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                width=pipe["canvas_width"] + 2,
                fill="#263238",
                capstyle="round",
            )
            line_id = canvas.create_line(
                x1,
                y1,
                x2,
                y2,
                width=pipe["canvas_width"],
                fill=pipe["state_color"],
                dash=pipe["state_dash"] or None,
                capstyle="round",
                tags=("cad_pipe", f"pipe::{pipe['pipe_id']}"),
            )
            canvas.tag_bind(
                line_id,
                "<Button-1>",
                lambda _event, pipe_id=pipe["pipe_id"]: self._select_pipe(pipe_id),
            )
            canvas.create_text(
                (x1 + x2) / 2.0,
                (y1 + y2) / 2.0,
                text=f"{pipe['state_symbol']} {pipe['pipe_id']}",
                fill="#111111",
                font=("Segoe UI", 9, "bold"),
                tags=("cad_pipe_label", f"pipe::{pipe['pipe_id']}"),
            )
        if not projected:
            canvas.create_text(
                width / 2,
                height / 2,
                text="清单中没有可绘制的 centerline_world_mm\n状态仍可在右侧列表查看",
                justify="center",
                fill="#616161",
                font=("Segoe UI", 12),
            )

    def _draw_dxf(self) -> None:
        document = self.dxf_elevation
        canvas = self.model_canvas
        width = max(canvas.winfo_width(), 400)
        height = max(canvas.winfo_height(), 300)
        if document is None:
            canvas.create_text(width / 2, height / 2, text="请先导入 DXF 侧立面", fill="#616161")
            return
        drawable = [point for entity in document.entities for point in (arc_points(entity) if entity.kind in {"ARC", "CIRCLE"} else entity.points)]
        min_x = min(point[0] for point in drawable); max_x = max(point[0] for point in drawable)
        min_y = min(point[1] for point in drawable); max_y = max(point[1] for point in drawable)
        margin = 48.0
        scale = min((width - 2 * margin) / max(max_x - min_x, 1.0), (height - 2 * margin) / max(max_y - min_y, 1.0))
        def xy(point: tuple[float, float]) -> tuple[float, float]:
            return (margin + (point[0] - min_x) * scale, height - margin - (point[1] - min_y) * scale)
        canvas.create_text(16, 14, anchor="nw", text="DXF 侧立面（图层颜色）", fill="#263238", font=("Segoe UI", 12, "bold"))
        for entity in document.entities:
            color = self.dxf_layer_colors.get(entity.layer, entity.color)
            if entity.kind == "CIRCLE":
                cx, cy = xy(entity.points[0]); radius = float(entity.radius or 0) * scale
                item = canvas.create_oval(cx - radius, cy - radius, cx + radius, cy + radius, outline=color, width=2, tags=("dxf_entity", f"dxf::{entity.entity_id}"))
            else:
                vertices = arc_points(entity) if entity.kind == "ARC" else entity.points
                item = canvas.create_line([coordinate for point in vertices for coordinate in xy(point)], fill=color, width=2, smooth=entity.kind == "ARC", tags=("dxf_entity", f"dxf::{entity.entity_id}"))
            canvas.tag_bind(item, "<Button-1>", lambda _event, entity_id=entity.entity_id: self._select_dxf_entity(entity_id))
            if entity.points:
                middle = entity.points[len(entity.points) // 2]
                label_x, label_y = xy(middle)
                binding = self.dxf_bindings.get(entity.entity_id)
                label = f"{entity.entity_id} → {binding}" if binding else entity.entity_id
                canvas.create_text(label_x + 4, label_y - 4, anchor="sw", text=label, fill=color, font=("Segoe UI", 8, "bold"), tags=("dxf_entity", f"dxf::{entity.entity_id}"))
        canvas.create_text(16, 38, anchor="nw", text=f"文件：{document.source_path.name} · 图层 {len(document.layers)} · 实体 {len(document.entities)}", fill="#455A64")

    def _import_dxf(self) -> None:
        selected = self.filedialog.askopenfilename(title="导入 DXF 侧立面", filetypes=(("DXF", "*.dxf"), ("All files", "*.*")))
        if not selected:
            return
        try:
            document = read_dxf_elevation(selected)
        except Exception as error:
            log_event(_LOGGER, "dxf_import_failed", path=selected, error=str(error))
            self.messagebox.showerror("DXF 导入失败", str(error))
            return
        self.dxf_elevation = document
        self.dxf_layer_colors = dict(document.layers)
        color_config = document.source_path.with_suffix(".colors.json")
        if color_config.is_file():
            try:
                saved = _read_json_object(color_config, "DXF color config")
                if saved.get("dxf_sha256") == document.source_sha256 and isinstance(saved.get("layers"), Mapping):
                    self.dxf_layer_colors.update({str(key): str(value).upper() for key, value in saved["layers"].items()})
            except (OSError, ValueError, TypeError):
                log_event(_LOGGER, "dxf_color_config_ignored", path=color_config)
        self.projection_mode.set("dxf")
        self._draw_model()
        log_event(_LOGGER, "dxf_import_finished", path=document.source_path, entity_count=len(document.entities), layer_count=len(document.layers))

    def _automate_dxf_setup(self) -> None:
        """Run the common DXF import → numbering → manifest-draft workflow."""

        if self.dxf_elevation is None:
            self._import_dxf()
        if self.dxf_elevation is None:
            return
        self._create_dxf_manifest_draft()
        log_event(_LOGGER, "dxf_automated_setup_finished", path=self.dxf_elevation.source_path)

    def _choose_dxf_layer_color(self) -> None:
        if self.dxf_elevation is None:
            self.messagebox.showinfo("未导入 DXF", "请先导入 DXF 侧立面。")
            return
        from tkinter import colorchooser, simpledialog
        layers = sorted(self.dxf_layer_colors)
        layer = simpledialog.askstring("指定图层颜色", "输入图层名：\n" + ", ".join(layers), parent=self.root)
        if not layer or layer not in self.dxf_layer_colors:
            return
        chosen = colorchooser.askcolor(color=self.dxf_layer_colors[layer], title=f"选择图层 {layer} 颜色", parent=self.root)[1]
        if chosen:
            self.dxf_layer_colors[layer] = chosen.upper()
            self.projection_mode.set("dxf")
            self._draw_model()
            log_event(_LOGGER, "dxf_layer_color_changed", layer=layer, color=chosen.upper())

    def _create_dxf_manifest_draft(self) -> None:
        if self.dxf_elevation is None:
            self.messagebox.showinfo("未导入 DXF", "请先导入 DXF 侧立面。")
            return
        pipes: list[dict[str, Any]] = []
        for index, entity in enumerate(self.dxf_elevation.entities, 1):
            points = arc_points(entity) if entity.kind == "ARC" else entity.points
            if entity.kind == "CIRCLE" and entity.radius is not None:
                cx, cy = entity.points[0]
                centerline = [[cx - entity.radius, cy, 0.0], [cx + entity.radius, cy, 0.0]]
                diameter = entity.radius * 2.0
            elif len(points) >= 2:
                centerline = [[points[0][0], points[0][1], 0.0], [points[-1][0], points[-1][1], 0.0]]
                diameter = 1.0
            else:
                continue
            entity_id = entity.entity_id or f"P{index:03d}"
            pipes.append({"instance_id": index, "pipe_id": entity_id, "cad_object_id": entity_id, "layer_id": entity.layer, "color_class": entity.layer, "color_srgb": self.dxf_layer_colors.get(entity.layer, entity.color), "nominal_diameter_mm": round(float(diameter), 6), "centerline_world_mm": centerline})
        if not pipes:
            self.messagebox.showerror("无法生成草稿", "DXF 中没有可转换为管道目录的图元。")
            return
        selected = self.filedialog.asksaveasfilename(title="保存 DXF manifest 草稿", initialfile=f"{self.dxf_elevation.source_path.stem}.manifest.json", initialdir=str(self.dxf_elevation.source_path.parent), defaultextension=".json", filetypes=(("JSON", "*.json"),))
        if not selected:
            return
        selected_path = Path(selected).resolve()
        relative = os.path.relpath(self.dxf_elevation.source_path, selected_path.parent).replace("\\", "/")
        if relative.startswith("../") or relative == "..":
            self.messagebox.showerror("保存失败", "DXF 文件必须位于 manifest 草稿目录或其子目录内。")
            return
        payload = {"schema_version": "dxf-elevation-draft-v1", "dataset_id": f"dxf-draft-{self.dxf_elevation.source_sha256[:12]}", "model_revision": self.dxf_elevation.source_sha256[:16], "validation_scope": "DXF_ELEVATION_DRAFT", "model": {"path": relative, "sha256": self.dxf_elevation.source_sha256, "unit": "millimeter", "pipes": pipes}, "elevation": {"format": "dxf", "path": relative, "sha256": self.dxf_elevation.source_sha256, "layer_colors": dict(self.dxf_layer_colors), "entity_bindings": {entity["pipe_id"]: entity["pipe_id"] for entity in pipes}}}
        selected_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        self._load_sources(selected_path, None)
        self.projection_mode.set("dxf")
        self._draw_model()
        log_event(_LOGGER, "dxf_manifest_draft_created", manifest=selected_path, pipe_count=len(pipes))

    def _select_dxf_entity(self, entity_id: str) -> None:
        self.selected_dxf_entity_id = entity_id
        self.projection_mode.set("dxf")
        self._draw_model()
        log_event(_LOGGER, "dxf_entity_selected", entity_id=entity_id)

    def _bind_dxf_entity(self) -> None:
        if self.dxf_elevation is None or not self.selected_dxf_entity_id:
            self.messagebox.showinfo("未选择DXF图元", "请先在 DXF 侧立面中点击一条线、圆弧或外轮廓圆。")
            return
        selected = self.tree.selection()
        if not selected:
            self.messagebox.showinfo("未选择管道", "请先在右侧管道列表中选择目标管道。")
            return
        pipe_id = str(selected[0])
        self.dxf_bindings[self.selected_dxf_entity_id] = pipe_id
        self._draw_model()
        log_event(_LOGGER, "dxf_entity_bound", entity_id=self.selected_dxf_entity_id, pipe_id=pipe_id)

    def _save_dxf_bindings(self) -> None:
        if self.dxf_elevation is None or not self.dxf_bindings:
            self.messagebox.showinfo("没有DXF映射", "请先导入 DXF 并至少绑定一个图元。")
            return
        if not self.manifest_path or not self.manifest:
            self.messagebox.showinfo("没有manifest", "请先载入或创建 manifest。")
            return
        try:
            updated = copy.deepcopy(self.manifest)
            selected = self.filedialog.asksaveasfilename(title="保存带DXF映射的manifest", initialfile=self.manifest_path.name, initialdir=str(self.manifest_path.parent), defaultextension=".json", filetypes=(("JSON", "*.json"),))
            if not selected:
                return
            selected_path = Path(selected).resolve()
            relative = os.path.relpath(self.dxf_elevation.source_path, selected_path.parent).replace("\\", "/")
            if relative.startswith("../") or relative == "..":
                raise ValueError("DXF 文件必须位于要保存的 manifest 目录或其子目录内，才能写入相对路径。")
            updated["elevation"] = {
                "format": "dxf",
                "path": relative,
                "sha256": self.dxf_elevation.source_sha256,
                "layer_colors": dict(self.dxf_layer_colors),
                "entity_bindings": dict(self.dxf_bindings),
            }
            selected_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            log_event(_LOGGER, "dxf_bindings_saved", manifest=selected, binding_count=len(self.dxf_bindings))
            self.messagebox.showinfo("已保存", f"DXF 映射已写入：\n{selected}\n\n报告需重新分析以匹配新的 manifest 哈希。")
        except (OSError, ValueError) as error:
            self.messagebox.showerror("保存映射失败", str(error))

    def _save_dxf_colors(self) -> None:
        if self.dxf_elevation is None:
            self.messagebox.showinfo("未导入 DXF", "请先导入 DXF 侧立面。")
            return
        selected = self.filedialog.asksaveasfilename(
            title="保存 DXF 图层颜色配置",
            initialfile=f"{self.dxf_elevation.source_path.stem}.colors.json",
            initialdir=str(self.dxf_elevation.source_path.parent),
            defaultextension=".json",
            filetypes=(("JSON", "*.json"), ("All files", "*.*")),
        )
        if not selected:
            return
        payload = {
            "format": "pipe_twin_dxf_layer_colors_v1",
            "dxf_path": str(self.dxf_elevation.source_path),
            "dxf_sha256": self.dxf_elevation.source_sha256,
            "layers": dict(self.dxf_layer_colors),
        }
        Path(selected).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log_event(_LOGGER, "dxf_layer_colors_saved", path=selected, layer_count=len(self.dxf_layer_colors))
        self.messagebox.showinfo("已保存", f"图层颜色配置已保存到：\n{selected}")

    def _select_canvas_pipe(self, event: Any) -> None:
        items = self.model_canvas.find_overlapping(
            event.x - 3, event.y - 3, event.x + 3, event.y + 3
        )
        for item in reversed(items):
            for tag in self.model_canvas.gettags(item):
                if tag.startswith("pipe::"):
                    self._select_pipe(tag.split("::", 1)[1])
                    return

    def _select_tree_pipe(self, _event: Any) -> None:
        selection = self.tree.selection()
        if selection:
            self._select_pipe(selection[0], update_tree=False)

    def _select_pipe(self, pipe_id: str, *, update_tree: bool = True) -> None:
        self.selected_pipe_id = pipe_id
        if update_tree and self.tree.exists(pipe_id):
            self.tree.selection_set(pipe_id)
            self.tree.see(pipe_id)
        self._draw_model()
        for role in ("left", "right"):
            self._render_photo(role)
        self._update_detail()
        if hasattr(self, "measurement_panel"):
            self.measurement_panel.pipe_id.set(pipe_id)
            self.measurement_panel.selection_changed()

    def _selected_pipe(self) -> Mapping[str, Any] | None:
        return next(
            (
                pipe
                for pipe in self.dashboard.get("pipes", [])
                if pipe.get("pipe_id") == self.selected_pipe_id
            ),
            None,
        )

    def _update_detail(self) -> None:
        pipe = self._selected_pipe()
        if pipe is None:
            self.detail_text.set("选择一根管道可查看 CAD GUID、物理颜色和判定依据。")
            return
        reasons = ", ".join(pipe["reason_codes"]) or "无原因码"
        self.detail_text.set(
            f"{pipe['pipe_id']}  |  CAD对象 {pipe['cad_object_id'] or pipe['cad_uuid'] or '—'}  |  "
            f"材料颜色 {pipe['material_color_srgb'] or pipe['appearance_color'] or '—'}  |  "
            f"状态 {pipe['installation_state_zh']} ({pipe['installation_state']})  |  "
            f"依据 {pipe['state_basis']} / {reasons}"
        )

    def _render_photo(self, role: str) -> None:
        canvas = self.photo_canvases.get(role)
        if canvas is None:
            return
        canvas.delete("all")
        path = self._photo_paths.get(role)
        if path is None:
            canvas.create_text(
                max(canvas.winfo_width(), 300) / 2,
                max(canvas.winfo_height(), 250) / 2,
                text=f"未选择{'左' if role == 'left' else '右'}相机照片",
                fill="#DADCE0",
                font=("Segoe UI", 12),
            )
            return
        try:
            import cv2
            import numpy as np

            raw = path.read_bytes()
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("OpenCV cannot decode this image")
            source_height, source_width = image.shape[:2]
            available_width = max(canvas.winfo_width() - 20, 200)
            available_height = max(canvas.winfo_height() - 42, 160)
            scale = min(
                available_width / source_width,
                available_height / source_height,
                1.0,
            )
            display_width = max(1, round(source_width * scale))
            display_height = max(1, round(source_height * scale))
            if (display_width, display_height) != (source_width, source_height):
                image = cv2.resize(
                    image,
                    (display_width, display_height),
                    interpolation=cv2.INTER_AREA,
                )
            ok, encoded = cv2.imencode(".png", image)
            if not ok:
                raise ValueError("OpenCV cannot encode the preview")
            photo = self.tk.PhotoImage(
                data=base64.b64encode(encoded.tobytes()).decode("ascii"),
                format="png",
            )
            center_x = max(canvas.winfo_width(), 300) / 2
            center_y = max(canvas.winfo_height(), 250) / 2 + 10
            canvas.create_image(center_x, center_y, image=photo)
            self._photo_images[role] = photo
            offset_x = center_x - display_width / 2
            offset_y = center_y - display_height / 2
            self._photo_geometry[role] = (offset_x, offset_y, source_width, source_height)
            canvas.create_text(
                8,
                7,
                anchor="nw",
                text=path.name,
                fill="#FFFFFF",
                font=("Segoe UI", 10, "bold"),
            )
            self._draw_photo_selection(role, scale)
        except Exception as error:
            canvas.create_text(
                max(canvas.winfo_width(), 300) / 2,
                max(canvas.winfo_height(), 250) / 2,
                text=f"照片预览失败\n{error}",
                justify="center",
                fill="#FFCDD2",
                font=("Segoe UI", 11),
            )

    def _draw_photo_selection(self, role: str, scale: float) -> None:
        pipe = self._selected_pipe()
        if pipe is None:
            return
        record = pipe.get("visibility_by_view", {}).get(role)
        if not isinstance(record, Mapping):
            return
        bbox = record.get("amodal_bbox_xywh")
        if (
            not isinstance(bbox, Sequence)
            or isinstance(bbox, (str, bytes))
            or len(bbox) != 4
            or any(type(value) not in (int, float) for value in bbox)
        ):
            return
        x, y, width, height = (float(value) for value in bbox)
        if width <= 0 or height <= 0:
            return
        offset_x, offset_y, _, _ = self._photo_geometry[role]
        canvas = self.photo_canvases[role]
        canvas.create_rectangle(
            offset_x + x * scale,
            offset_y + y * scale,
            offset_x + (x + width) * scale,
            offset_y + (y + height) * scale,
            outline=pipe["state_color"],
            width=3,
            dash=(5, 3) if pipe["installation_state"] == "UNKNOWN" else None,
        )
        canvas.create_text(
            offset_x + x * scale,
            offset_y + y * scale - 4,
            anchor="sw",
            text=f"{pipe['pipe_id']} · {pipe['installation_state_zh']}",
            fill=pipe["state_color"],
            font=("Segoe UI", 9, "bold"),
        )

    def _run_analysis(self) -> None:
        if self._analysis_running:
            return
        if not self.manifest_path:
            self.messagebox.showinfo("尚未载入数据", "请先载入现场清单或打开合成示例。")
            return
        try:
            options = self.measurement_panel.analysis_options()
        except ValueError as error:
            self.messagebox.showerror("测量设置无效", str(error))
            return
        self._analysis_request = (self.manifest_path, copy.deepcopy(self.manifest), options, self._manifest_sha256)
        self._analysis_running = True
        self.run_button.configure(state="disabled")
        self.banner.configure(bg="#DCEBFA", fg="#164B75")
        self.banner_text.set("正在分析双目照片、局部管径、中心距、净距与前后关系……")
        log_event(_LOGGER, "gui_analysis_start", manifest=self.manifest_path)
        thread = threading.Thread(target=self._analysis_worker, daemon=True)
        thread.start()

    def _analysis_worker(self) -> None:
        try:
            manifest_path, manifest, options, _digest = self._analysis_request
            capture = manifest.get("capture")
            capture_kind = capture.get("kind") if isinstance(capture, Mapping) else None
            if (
                manifest.get("schema_version") == "2.0"
                or capture_kind == "stereo_still_capture_set"
            ):
                from .stereo_analyzer import analyze_stereo_capture

                report = analyze_stereo_capture(manifest_path, measurement_options=options)
            else:
                from .pipeline import analyze_manifest

                report = analyze_manifest(manifest_path)
            self._worker_messages.put(("ok", report))
        except Exception as error:
            self._worker_messages.put(("error", error))

    def _poll_worker(self) -> None:
        try:
            kind, payload = self._worker_messages.get_nowait()
        except queue.Empty:
            self.root.after(80, self._poll_worker)
            return
        self._analysis_running = False
        self.run_button.configure(state="normal")
        if (self._analysis_request[0] != self.manifest_path
                or self._analysis_request[3] != self._manifest_sha256):
            self.measurement_panel.notice.set("上一次分析完成；当前数据已切换，请分析当前照片。")
            self.root.after(80, self._poll_worker)
            return
        if kind == "error":
            log_event(_LOGGER, "gui_analysis_failed", error=str(payload))
            # A failed rerun must not leave the previous decisive report on
            # screen.  The manifest remains available for diagnostics, but
            # all pipe states are rebuilt as UNKNOWN before showing the error.
            self._clear_loaded_state(
                manifest=self.manifest,
                manifest_path=self.manifest_path,
                reason_code="ANALYSIS_FAILED",
                message=f"识别失败：{payload}",
                preserve_manifest=True,
            )
            self.banner.configure(bg="#F8D7DA", fg="#721C24")
            self.banner_text.set(f"识别失败：{payload}")
            self.messagebox.showerror("识别失败", str(payload))
        else:
            try:
                dashboard = build_dashboard_model(
                    self.manifest,
                    payload,
                    manifest_sha256=self._manifest_sha256,
                    model_actual_sha256=self._model_actual_sha256,
                    photo_actual_sha256=self._current_photo_hashes(),
                )
            except Exception as error:
                # Treat an invalid worker result exactly like an analysis
                # failure: discard any prior report instead of leaving stale
                # INSTALLED/NOT_INSTALLED colours visible.
                self._clear_loaded_state(
                    manifest=self.manifest,
                    manifest_path=self.manifest_path,
                    reason_code="REPORT_INVALID",
                    message=f"识别结果无效：{error}",
                    preserve_manifest=True,
                )
                self.messagebox.showerror("识别结果无效", str(error))
            else:
                self.report = payload
                self.dashboard = dashboard
                self.selected_pipe_id = None
                self._refresh_dashboard()
                self.measurement_panel.notice.set("分析完成。查看逐管状态和间距/前后关系，可录入实测样本或导出报告。")
                log_event(_LOGGER, "gui_analysis_finished", pipe_count=len(self.dashboard.get("pipes", [])), counts=self.dashboard.get("counts"))
        self.root.after(80, self._poll_worker)

    def _open_demo(self) -> None:
        path = Path(__file__).resolve().parents[1] / "test_model" / "field_stereo_demo_manifest.json"
        self._load_sources(path, None)
        self.main_tabs.select(0)

    def _input_capture(self) -> None:
        from .capture_gui import CaptureInputDialog

        CaptureInputDialog(self)


def launch_gui(
    manifest_path: str | Path | None = None,
    report_path: str | Path | None = None,
) -> None:
    """Launch the Windows-friendly local desktop dashboard.

    The manifest is authoritative for CAD identity and geometry.  A report is
    optional; without one every pipe is shown as ``UNKNOWN`` until recognition
    is run.  File-dialog-selected photos are previews only and never silently
    replace the manifest-bound stereo pair used by the analyzer.
    """

    log_event(_LOGGER, "gui_launch", manifest=manifest_path, report=report_path)
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as error:
        raise RuntimeError(
            "The pipe dashboard requires an interactive desktop session with Tk 8.6"
        ) from error
    _PipeTwinApplication(root, manifest_path, report_path)
    root.mainloop()


__all__ = [
    "INSTALLATION_STATES",
    "STATE_PRESENTATION",
    "build_dashboard_model",
    "launch_gui",
]
