"""Pure dashboard contracts and projections for the local GUI.

This module has no Tk dependency. It validates report provenance, binds
recognition results to CAD identities, and produces deterministic view models
for the desktop adapter.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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

_FIELD_CALIBRATION_ID_MARKERS = (
    "SYNTHETIC",
    "DEMO",
    "EXAMPLE",
    "REPLACE_WITH_REAL",
)

# Mirrors measurement_gui.OUTPUT_ROOT and workbench_profile._ROOT.  It is
# duplicated here so the one-click capture can write its camera intake folders
# without importing a Tk panel module (see the module import rules below).
_WORKBENCH_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "measurement_workbench"


def summarize_stereo_diagnostics(
    manifest: Mapping[str, Any], report: Mapping[str, Any]
) -> str | None:
    """Explain an all-UNKNOWN stereo result in terms an operator can act on."""
    pipes = report.get("pipes")
    counts = report.get("counts")
    if not isinstance(pipes, list) or not pipes or not isinstance(counts, Mapping):
        return None
    if counts.get("UNKNOWN") != len(pipes):
        return None

    reasons: list[str] = []
    calibration = manifest.get("stereo_calibration")
    calibration = calibration if isinstance(calibration, Mapping) else {}
    calibration_id = str(calibration.get("calibration_id", ""))
    if any(marker in calibration_id.upper() for marker in _FIELD_CALIBRATION_ID_MARKERS):
        reasons.append(f"当前使用合成演示标定“{calibration_id}”，不是现场相机标定")

    calibration_audit = report.get("calibration_audit")
    if (
        isinstance(calibration_audit, Mapping)
        and calibration_audit.get("registration_validated") is not True
    ):
        reasons.append("相机—CAD 位姿尚未验证")

    reason_codes: set[str] = set()
    valid_fractions: list[float] = []
    capture_audit = report.get("capture_audit")
    groups = capture_audit.get("groups", []) if isinstance(capture_audit, Mapping) else []
    for group in groups if isinstance(groups, list) else []:
        if not isinstance(group, Mapping):
            continue
        depth_audit = group.get("depth_audit")
        if isinstance(depth_audit, Mapping):
            for key in ("valid_left_fraction", "valid_right_fraction"):
                value = depth_audit.get(key)
                if type(value) in (int, float) and math.isfinite(value):
                    valid_fractions.append(float(value))
        evidence = group.get("pipe_evidence")
        if not isinstance(evidence, Mapping):
            continue
        for views in evidence.values():
            if not isinstance(views, Mapping):
                continue
            for view in views.values():
                if not isinstance(view, Mapping):
                    continue
                codes = view.get("reason_codes")
                if isinstance(codes, list):
                    reason_codes.update(str(code) for code in codes)

    if "INSUFFICIENT_VALID_DEPTH" in reason_codes:
        if valid_fractions:
            fraction = 100.0 * sum(valid_fractions) / len(valid_fractions)
            reasons.append(f"有效双目视差约 {fraction:.1f}%，CAD 目标区域深度不足")
        else:
            reasons.append("CAD 目标区域有效双目深度不足")
    if "TARGET_COLOR_INTERSECTION_GATE_FAILED" in reason_codes:
        reasons.append("深度与管件颜色没有在 CAD 投影区域内重合")
    if "WIDTH_GATE_FAILED" in reason_codes:
        reasons.append("观测管径与 CAD 投影管径不一致")

    if not reasons:
        reasons.append("当前照片没有形成满足阈值的管件证据")
    return f"识别完成，但 {len(pipes)} 根管件全部不确定。" + "；".join(reasons) + "。"

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


__all__ = [
    "INSTALLATION_STATES",
    "STATE_PRESENTATION",
    "build_dashboard_model",
    "summarize_stereo_diagnostics",
    "_empty_dashboard",
    "_first_manifest_stereo_paths",
    "_loaded_asset_hashes",
    "_loaded_photo_hashes",
    "_project_dashboard_pipes",
    "_read_json_object",
    "_report_latest_photo_hashes",
    "_report_manifest_hash",
    "_report_model_hashes",
    "_safe_manifest_asset_path",
    "_sha256_file",
]
