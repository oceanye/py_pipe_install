"""Run, validate and persist independent AI section candidates.

Network work never writes a scene. A GUI accepts the returned envelope only
after its generation/cancellation checks, then calls accept_ai_result. The
sidecar binds every source asset and never replaces the strict report.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from .ai_inputs import AIInputError, digest, prepare_ai_request
from .ai_section_assist import build_ai_section_assist
from .artifact_io import atomic_write_text

RUN_VERSION = 'ai-section-run-v1'
SIDECAR_NAME = 'ai_section_result.json'
MAX_SIDECAR_BYTES = 2 * 1024 * 1024


class AIWorkflowError(ValueError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def _check_cancelled(event):
    if event is not None and event.is_set():
        raise AIWorkflowError('AI_CANCELLED', 'AI 任务已取消，结果未保存。')


def _path(manifest_path):
    path = Path(manifest_path).resolve()
    return path / 'manifest.json' if path.is_dir() else path


def _build_checked_namespace(prepared, response, source):
    if not isinstance(response, dict):
        raise AIWorkflowError('AI_RESPONSE_INVALID', 'AI 返回内容不是结构化对象。')
    if response.get('input_hash') != prepared['input_hash']:
        raise AIWorkflowError('AI_RESPONSE_INPUT_MISMATCH', 'AI 返回的输入标识与当前请求不一致，结果未接纳。')
    if response.get('mapping_sha256') != prepared['point_map']['mapping_sha256']:
        raise AIWorkflowError('AI_RESPONSE_MAP_MISMATCH', 'AI 返回的模型点位标识不一致，结果未接纳。')
    rows = response.get('point_assessments')
    if not isinstance(rows, list) or len(rows) > 512:
        raise AIWorkflowError('AI_RESPONSE_INVALID', 'AI 返回内容缺少有效的点位列表。')
    refs = {asset['asset_ref']: asset for asset in prepared['asset_refs']}
    normalized = copy.deepcopy(response)
    qualification = prepared['qualification_reason_codes']
    for row in normalized['point_assessments']:
        if not isinstance(row, dict):
            raise AIWorkflowError('AI_RESPONSE_INVALID', 'AI 点位条目格式无效。')
        references = row.get('observation_refs', [])
        if (not isinstance(references, list) or len(references) > len(refs) or
                any(not isinstance(ref, str) or ref not in refs for ref in references)):
            raise AIWorkflowError('AI_UNKNOWN_OBSERVATION_REF', 'AI 引用了不属于本次请求的图像，结果未接纳。')
        reasons = row.get('reason_codes', [])
        if (not isinstance(reasons, list) or len(reasons) > 30 or
                any(not isinstance(reason, str) or len(reason) > 1000 for reason in reasons)):
            raise AIWorkflowError('AI_RESPONSE_INVALID', 'AI 证据说明格式无效。')
        extra = list(qualification)
        if row.get('section_status') == 'SUPPORTED_VISIBLE' and not any(
                refs[ref]['kind'] == 'FIELD_IMAGE' for ref in references):
            extra.append('AI_SUPPORT_WITHOUT_FIELD_IMAGE')
        if extra and row.get('section_status') in ('SUPPORTED_VISIBLE', 'AMBIGUOUS'):
            row['section_status'] = 'UNRESOLVED'
            row['evidence_quality'] = 'LOW'
            row['reason_codes'] = list(dict.fromkeys(reasons + extra))
    namespace = build_ai_section_assist(
        prepared['point_map'], normalized,
        section_evidence={'control_section': prepared['control_section']}, source=source)
    if namespace.get('run_status') == 'REJECTED':
        raise AIWorkflowError('AI_RESPONSE_CONTRACT_REJECTED', 'AI 返回内容未通过点位契约校验，结果未接纳。')
    namespace['input_qualification'] = {
        'status': 'DEFINED' if not qualification else 'UNRESOLVED',
        'reason_codes': list(qualification),
        'scope': 'VISUAL_SECTION_ONLY_NOT_METRIC_REGISTRATION'}
    namespace['asset_refs'] = copy.deepcopy(prepared['asset_refs'])
    namespace['observation_scope'] = 'LATEST_REFERENCE_PAIR_ONLY_NOT_TEMPORAL_FUSION'
    return namespace


def evaluate_ai_scene(manifest_path, report=None, *, settings=None, cancel_event=None, progress=None):
    """Return an unpersisted result envelope; does not use A's matches."""
    _check_cancelled(cancel_event)
    if progress:
        progress('正在校验原图、模型与断面范围……')
    prepared = prepare_ai_request(manifest_path, report)
    _check_cancelled(cancel_event)
    if progress:
        progress('正在请求 Kimi 独立分析当前左右图与模型断面……')
    from .ai_provider import request_section_assessment
    provider = request_section_assessment(prepared['messages'], settings=settings, cancel_event=cancel_event)
    _check_cancelled(cancel_event)
    generated = datetime.now(timezone.utc).isoformat()
    source = {key: provider.get(key) for key in ('provider', 'model', 'request_id')}
    source.update(input_hash=prepared['input_hash'],
                  source_id=prepared['source_binding']['capture_id'], generated_at=generated)
    namespace = _build_checked_namespace(prepared, provider.get('response'), source)
    result = {
        'schema_version': RUN_VERSION, 'run_id': str(uuid.uuid4()),
        'input_hash': prepared['input_hash'], 'source_binding': prepared['source_binding'],
        'point_map': prepared['point_map'], 'control_section': prepared['control_section'],
        'qualification_reason_codes': prepared['qualification_reason_codes'],
        'response': provider['response'], 'ai_section_assist': namespace,
        **source, 'usage': provider.get('usage') or {},
    }
    result['envelope_sha256'] = digest(result)
    if progress:
        progress('AI 返回已校验，等待当前界面接纳并保存。')
    return result


def _validated_result(manifest_path, result, report=None):
    if not isinstance(result, dict) or result.get('schema_version') != RUN_VERSION:
        raise AIWorkflowError('AI_RESULT_SCHEMA_INVALID', 'AI 记录版本无效，不能恢复。')
    try:
        uuid.UUID(result['run_id'])
        content = {key: value for key, value in result.items() if key != 'envelope_sha256'}
        if digest(content) != result.get('envelope_sha256'):
            raise ValueError('checksum')
    except (KeyError, TypeError, ValueError):
        raise AIWorkflowError('AI_RESULT_DAMAGED', 'AI 结果记录不完整或已改变，请重新评估。') from None
    try:
        prepared = prepare_ai_request(manifest_path, report)
    except (AIInputError, OSError, ValueError) as error:
        raise AIWorkflowError('AI_INPUT_CHANGED', 'AI 来源文件已改变或不可用，旧结果不再适用。') from error
    if (result.get('input_hash') != prepared['input_hash'] or
            result.get('source_binding') != prepared['source_binding'] or
            result.get('point_map') != prepared['point_map'] or
            result.get('control_section') != prepared['control_section']):
        raise AIWorkflowError('AI_INPUT_CHANGED', '照片、模型、观察方向或范围已改变，旧 AI 结果不再适用。')
    source = {key: result.get(key) for key in
              ('provider', 'model', 'request_id', 'input_hash', 'source_id', 'generated_at')}
    namespace = _build_checked_namespace(prepared, result.get('response'), source)
    # Do not trust caller-authored or modified display rows in a saved envelope.
    if namespace != result.get('ai_section_assist'):
        raise AIWorkflowError('AI_RESULT_DAMAGED', 'AI 显示结果与已校验的响应不一致。')
    return namespace


def _merge(report, namespace):
    result = copy.deepcopy(report) if report is not None else {
        'pipes': [], 'counts': {'INSTALLED': 0, 'NOT_INSTALLED': 0, 'UNKNOWN': 0},
        'strict_evaluation': 'NOT_RUN', 'analysis_mode': 'AI_SECTION_ONLY'}
    if not isinstance(result, dict):
        raise AIWorkflowError('AI_REPORT_INVALID', '无法将 AI 结果合入当前报告。')
    result['ai_section_assist'] = namespace
    return result


def accept_ai_result(manifest_path, result, report=None, *, save=True):
    """Revalidate current sources, then atomically save only the AI sidecar.

    Callers must check GUI generation and cancellation before invoking. Files
    are rehashed here, so edits during a provider request also fail closed.
    """
    namespace = _validated_result(manifest_path, result, report)
    merged = _merge(report, namespace)
    if save:
        serialized = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
        if len(serialized.encode('utf-8')) > MAX_SIDECAR_BYTES:
            raise AIWorkflowError('AI_RESULT_TOO_LARGE', 'AI 结果过大，未保存；请缩小评估范围后重试。')
        atomic_write_text(_path(manifest_path).with_name(SIDECAR_NAME), serialized)
    return merged


def load_ai_result(manifest_path, report=None):
    """Restore only a currently bound AI sidecar; stale/damaged records raise."""
    path = _path(manifest_path).with_name(SIDECAR_NAME)
    if not path.exists():
        result = copy.deepcopy(report)
        if isinstance(result, dict):
            result.pop('ai_section_assist', None)
        return result
    try:
        with path.open('rb') as stream:
            data = stream.read(MAX_SIDECAR_BYTES+1)
        if len(data) > MAX_SIDECAR_BYTES:
            raise AIWorkflowError('AI_RESULT_DAMAGED', 'AI 结果文件过大或无效，不能恢复。')
        result = json.loads(data)
    except (OSError, ValueError):
        raise AIWorkflowError('AI_RESULT_DAMAGED', 'AI 结果文件无法读取，请重新评估。') from None
    return accept_ai_result(manifest_path, result, report, save=False)


__all__ = ['AIWorkflowError', 'prepare_ai_request', 'evaluate_ai_scene', 'accept_ai_result', 'load_ai_result']
