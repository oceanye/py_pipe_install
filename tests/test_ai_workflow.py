"""Scene-bound independent AI route, with no external calls or credentials."""
import copy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import threading
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from pipe_twin.ai_inputs import AIInputError, prepare_ai_request, _section_reference, _image_control, digest
from pipe_twin.ai_workflow import (
    AIWorkflowError, evaluate_ai_scene, accept_ai_result, load_ai_result,
)
from pipe_twin.dxf_elevation import catalog_from_dxf
from pipe_twin.elevation_dataset import create_elevation_dataset
import pipe_twin.ai_inputs as ai_inputs
from test_elevation_dataset import _calibration

MODEL = Path(__file__).resolve().parents[1] / 'test_model' / '管道布置.dxf'


@pytest.fixture
def scene(tmp_path):
    left = np.full((480, 640, 3), 160, np.uint8)
    left[150:200, :, 2] = 250
    left[:, 200:230, 0] = 30
    cv2.imwrite(str(tmp_path/'left.png'), left)
    cv2.imwrite(str(tmp_path/'right.png'), np.roll(left, -10, axis=1))
    specs, _ = catalog_from_dxf(MODEL, axis_world=(0., 0., 1.), unitless_unit='millimeter')
    return create_elevation_dataset(
        output_root=tmp_path/'scenes', calibration=_calibration(),
        left_path=tmp_path/'left.png', right_path=tmp_path/'right.png',
        left_time='2026-11-01T10:00:00+08:00', right_time='2026-11-01T10:00:00.001+08:00',
        pipe_specs=specs, pair_confirmed=True, model_path=MODEL, stl_unit='millimeter',
        registration_settings={'camera_side_world': [0., 1., 0.],
                               'aggregation_band': {'image_size': [640, 480],
                                                    'axis_segment_px': [[100, 220], [500, 250]]}})


def fake_provider(messages, **kwargs):
    context = json.loads(messages[1]['content'][0]['text'])
    return {'provider': 'kimi', 'model': 'fixture-only', 'request_id': 'local-fixture-no-network',
            'usage': {}, 'response': {
                'input_hash': context['input_hash'], 'mapping_sha256': context['mapping_sha256'],
                'point_assessments': [{'point_id': p['point_id'], 'section_status': 'SUPPORTED_VISIBLE',
                                       'evidence_quality': 'HIGH', 'reason_codes': ['FIXTURE_ONLY'],
                                       'observation_refs': ['scene_left', 'scene_right']}
                                      for p in context['model_points']]}}


@pytest.fixture
def provider(monkeypatch):
    mock = Mock(side_effect=fake_provider)
    monkeypatch.setattr('pipe_twin.ai_provider.request_section_assessment', mock)
    return mock


def alter_manifest(scene, mutate):
    manifest = json.loads(scene.read_text(encoding='utf-8'))
    mutate(manifest)
    scene.write_text(json.dumps(manifest), encoding='utf-8')


def test_section_normal_is_pipe_axis_and_retains_near_far_dimension():
    pipes = [{'pipe_id': f'PIPE{i}', 'nominal_diameter_mm': 30,
              'centerline_world_mm': [[x, y, 0], [x, y, 200]]}
             for i, (x, y) in enumerate(((0, 0), (0, 70), (70, 0)))]
    points, reasons = _section_reference(pipes, {'camera_side_world': [0, 1, 0]})
    assert not reasons
    assert np.allclose(points['basis_rows']['normal_world'], [0, 0, 1])
    assert np.allclose(points['basis_rows']['up_world'], [0, 1, 0])
    assert np.linalg.matrix_rank(np.array([p['section_uv_mm'] for p in points['points']])) == 2
    reverse, _ = _section_reference(pipes, {'camera_side_world': [0, -1, 0]})
    assert reverse['mapping_sha256'] != points['mapping_sha256']


def test_request_is_blind_to_local_identity_and_binds_pixels_and_control(scene):
    request = prepare_ai_request(scene, {'installation_assist': 'FORBIDDEN_LOCAL_MATCH_SENTINEL',
                                        'pipes': [{'pipe_id': 'USER_PREVIOUS_ANSWER'}]})
    prompt = json.dumps(request['messages'], ensure_ascii=False)
    assert 'FORBIDDEN_LOCAL_MATCH_SENTINEL' not in prompt
    assert 'USER_PREVIOUS_ANSWER' not in prompt
    assert 'color_srgb' not in prompt
    assert request['control_section']['status'] == 'DEFINED'
    assert request['control_section']['coordinate_space'] == 'rectified_left_pixel'
    assert 'station_mm' not in request['control_section']
    assert len(request['control_section']['line_px']) == 2
    assert request['source_binding']['manifest_sha256'] == hashlib.sha256(scene.read_bytes()).hexdigest()
    assert {a['asset_ref'] for a in request['asset_refs']} == {'scene_left', 'scene_right', 'model_section'}
    assert all(a.get('captured_at') for a in request['asset_refs'] if a['kind'] == 'FIELD_IMAGE')
    assert 'base64' not in json.dumps({k: v for k, v in request.items() if k != 'messages'})


def test_request_adds_identity_blind_replay_and_pooled_control_context(scene):
    report = {
        'pipes': [{'pipe_id': 'LEAKED_PIPE', 'installation_state': 'INSTALLED'}],
        'counts': {'INSTALLED': 1},
        'temporal_fusion': {
            'status': 'VALID', 'requested_frames': 10, 'accepted_frames': 10,
            'algorithm': 'static-burst-photometric-consensus-v3',
        },
        'section_evaluation': {
            'status': 'ANALYZED', 'conditional_section_count': 2,
            'candidates': [{'candidate_id': 'HIDDEN_PIPE_ID'}],
            'control_section': {
                'status': 'DEFINED', 'source': 'PIPE_SEGMENT_AGGREGATION_COMMON_SUPPORT_MIDPOINT',
                'station_mm': -141.676, 'interval_mm': [-296.06, 12.71],
                'candidate_ids': ['HIDDEN_PIPE_ID'],
            },
        },
        'local_surface': {
            'observations': [{'pipe_id': 'HIDDEN_PIPE_ID'}],
            'image_candidates': [{}, {}],
            'audit': {'metric_cylinder_count': 0, 'observation_count': 0,
                      'source_rejection_counts': {
                          'left:image_strip': {'reasons': {'INSUFFICIENT_VALID_DEPTH': 30}},
                      }},
        },
    }
    alter_manifest(scene, lambda manifest: manifest['analysis']['elevation_auto']['registration'].pop('aggregation_band'))
    request = prepare_ai_request(scene, report)
    context = json.loads(request['messages'][1]['content'][0]['text'])
    assert request['control_section']['coordinate_space'] == 'model_axis_mm'
    assert request['image_control_section']['status'] == 'UNRESOLVED'
    assert 'AI_AGGREGATION_BAND_MISSING' not in request['qualification_reason_codes']
    assert context['pooled_control_section']['status'] == 'DEFINED'
    assert context['pooled_control_section']['station_mm'] == -141.676
    assert context['evidence_summary']['temporal_fusion']['accepted_frames'] == 10
    assert context['evidence_summary']['local_surface']['rejection_reason_counts'] == {
        'INSUFFICIENT_VALID_DEPTH': 30}
    prompt = json.dumps(request['messages'], ensure_ascii=False)
    assert 'LEAKED_PIPE' not in prompt and 'HIDDEN_PIPE_ID' not in prompt
    assert '"installation_state"' not in prompt and '"counts"' not in prompt
    changed = json.loads(json.dumps(report))
    changed['section_evaluation']['control_section']['station_mm'] = -140.0
    assert prepare_ai_request(scene, changed)['input_hash'] != request['input_hash']


def test_request_includes_full_temporal_burst_as_bound_visual_evidence(tmp_path):
    left = tmp_path / 'left.png'
    right = tmp_path / 'right.png'
    image = np.full((480, 640, 3), 160, np.uint8)
    image[150:200, :, 2] = 250
    image[:, 200:230, 0] = 30
    cv2.imwrite(str(left), image)
    cv2.imwrite(str(right), np.roll(image, -10, axis=1))
    specs, _ = catalog_from_dxf(MODEL, axis_world=(0., 0., 1.), unitless_unit='millimeter')
    start = datetime.fromisoformat('2026-11-01T10:00:00+08:00')
    frames = []
    for index in range(3):
        frame_left, frame_right = left, right
        if index < 2:
            frame = image.copy()
            frame[10 + index, 10 + index, 1] = 200 + index
            frame_left = tmp_path / f'burst_left_{index}.png'
            frame_right = tmp_path / f'burst_right_{index}.png'
            cv2.imwrite(str(frame_left), frame)
            cv2.imwrite(str(frame_right), np.roll(frame, -10, axis=1))
        frames.append({
            'left_path': frame_left, 'right_path': frame_right,
            'left_time': (start + timedelta(milliseconds=index)).isoformat(),
            'right_time': (start + timedelta(milliseconds=index + 1)).isoformat(),
            'sha256': {
                'left': hashlib.sha256(frame_left.read_bytes()).hexdigest(),
                'right': hashlib.sha256(frame_right.read_bytes()).hexdigest(),
            },
        })
    left_hash = hashlib.sha256(left.read_bytes()).hexdigest()
    right_hash = hashlib.sha256(right.read_bytes()).hexdigest()
    manifest = create_elevation_dataset(
        output_root=tmp_path / 'scenes', calibration=_calibration(),
        left_path=left, right_path=right,
        left_time=(start + timedelta(milliseconds=2)).isoformat(),
        right_time=(start + timedelta(milliseconds=3)).isoformat(),
        pipe_specs=specs, pair_confirmed=True, model_path=MODEL, stl_unit='millimeter',
        registration_settings={
            'camera_side_world': [0., 1., 0.],
            'aggregation_band': {'image_size': [640, 480], 'axis_segment_px': [[100, 220], [500, 250]]},
        }, temporal_frames=frames,
    )
    request = prepare_ai_request(manifest)
    context = json.loads(request['messages'][1]['content'][0]['text'])
    assert context['image_scope'] == 'TEMPORAL_BURST_REFERENCE_SET'
    assert context['temporal_frame_count'] == 3
    assert len(context['temporal_asset_refs']) == 4
    assert all(ref in {asset['asset_ref'] for asset in request['asset_refs']}
               for ref in context['temporal_asset_refs'])
    burst_hashes = {frame['sha256'][role] for frame in frames for role in ('left', 'right')}
    assert all(asset['source_sha256'] in burst_hashes
               for asset in request['asset_refs'] if asset['asset_ref'].startswith('burst_'))
    assert '相机前后位移' in request['messages'][0]['content']


def test_roi_is_sent_with_original_pixel_coordinates(scene):
    def mutate(m):
        setting = m['analysis']['elevation_auto']
        setting['scope'] = 'zones'
        setting['zones'] = [{'zone_id': 'CURRENT', 'label': 'current', 'enabled': True,
                             'coordinate_space': 'rectified_left', 'roi_rect_px': [20, 30, 400, 300]}]
    alter_manifest(scene, mutate)
    request = prepare_ai_request(scene)
    crop = next(a for a in request['asset_refs'] if a['asset_ref'] == 'scene_left_roi')
    assert crop['crop_rect_px'] == [20, 30, 400, 300]
    assert crop['encoded_size_px'] == [400, 300]


def test_control_line_uses_band_roi_intersection_and_empty_intersection_rejected():
    registration = {'aggregation_band': {'image_size': [640, 480],
                                         'axis_segment_px': [[100, 200], [500, 200]]}}
    control = _image_control(registration, (640, 480), [400, 100, 200, 300])
    assert control['interval_px'] == [300., 400.]
    assert control['midpoint_px'] == [450., 200.]
    assert all(400 <= p[0] <= 599 and 100 <= p[1] <= 399 for p in control['line_px'])
    with pytest.raises(AIInputError, match='没有交集'):
        _image_control(registration, (640, 480), [550, 100, 80, 300])


def test_independent_run_saves_reopens_without_mutating_strict_report(scene, provider):
    strict = {'pipes': [{'pipe_id': 'P001', 'installation_state': 'UNKNOWN'}], 'counts': {'UNKNOWN': 12},
              'temporal_fusion': {'status': 'INVALID', 'accepted_frames': 4, 'requested_frames': 10}}
    before = copy.deepcopy(strict)
    strict_path = scene.with_name('report.json')
    strict_path.write_text(json.dumps(strict), encoding='utf-8')
    strict_bytes = strict_path.read_bytes()
    manifest_bytes = scene.read_bytes()
    result = evaluate_ai_scene(scene, strict)
    assert not scene.with_name('ai_section_result.json').exists()
    merged = accept_ai_result(scene, result, strict)
    assert merged['ai_section_assist']['status'] == 'CANDIDATES_READY'
    assert strict == before
    assert merged['pipes'] == strict['pipes'] and merged['counts'] == strict['counts']
    assert load_ai_result(scene, strict) == merged
    assert strict_path.read_bytes() == strict_bytes and scene.read_bytes() == manifest_bytes
    assert provider.call_count == 1


def test_no_local_report_has_explicit_not_run_strict_state(scene, provider):
    merged = accept_ai_result(scene, evaluate_ai_scene(scene))
    assert merged['strict_evaluation'] == 'NOT_RUN'
    assert merged['pipes'] == []
    assert load_ai_result(scene) == merged
    assert not scene.with_name('report.json').exists()


@pytest.mark.parametrize('field', ['camera_side_world', 'aggregation_band'])
def test_missing_orientation_or_control_downgrades_positive_claims(scene, provider, field):
    alter_manifest(scene, lambda m: m['analysis']['elevation_auto']['registration'].pop(field))
    result = evaluate_ai_scene(scene)
    assert result['ai_section_assist']['status'] == 'UNRESOLVED'
    assert all(p['section_status'] == 'UNRESOLVED' for p in result['ai_section_assist']['candidates'])


@pytest.mark.parametrize('key', ['input_hash', 'mapping_sha256'])
def test_wrong_or_missing_response_binding_is_rejected(scene, monkeypatch, key):
    def wrong(messages, **kwargs):
        response = fake_provider(messages)
        response['response'].pop(key)
        return response
    monkeypatch.setattr('pipe_twin.ai_provider.request_section_assessment', wrong)
    with pytest.raises(AIWorkflowError, match='标识'):
        evaluate_ai_scene(scene)
    assert not scene.with_name('ai_section_result.json').exists()


def test_unknown_image_reference_rejected_and_design_only_support_capped(scene, monkeypatch):
    def respond(reference):
        def inner(messages, **kwargs):
            result = fake_provider(messages)
            for row in result['response']['point_assessments']:
                row['observation_refs'] = [reference]
            return result
        return inner
    monkeypatch.setattr('pipe_twin.ai_provider.request_section_assessment', respond('another_scene'))
    with pytest.raises(AIWorkflowError, match='本次请求'):
        evaluate_ai_scene(scene)
    monkeypatch.setattr('pipe_twin.ai_provider.request_section_assessment', respond('model_section'))
    result = evaluate_ai_scene(scene)
    assert result['ai_section_assist']['status'] == 'UNRESOLVED'
    assert all('AI_SUPPORT_WITHOUT_FIELD_IMAGE' in p['reason_codes']
               for p in result['ai_section_assist']['candidates'])


@pytest.mark.parametrize('target', ['direction', 'photo', 'manifest_whitespace'])
def test_source_changes_during_request_prevent_acceptance(scene, provider, target):
    result = evaluate_ai_scene(scene)
    if target == 'direction':
        alter_manifest(scene, lambda m: m['analysis']['elevation_auto']['registration'].update(camera_side_world=[0, -1, 0]))
    elif target == 'photo':
        m = json.loads(scene.read_text(encoding='utf-8'))
        image = scene.parent / m['capture']['capture_groups'][-1]['views']['left']['path']
        image.write_bytes(image.read_bytes()+b'changed')
    else:
        scene.write_bytes(scene.read_bytes()+b'\n')
    with pytest.raises(AIWorkflowError):
        accept_ai_result(scene, result)
    assert not scene.with_name('ai_section_result.json').exists()


def test_modified_sidecar_display_is_not_trusted_even_with_updated_checksum(scene, provider):
    result = evaluate_ai_scene(scene)
    result['ai_section_assist']['candidates'][0]['pipe_id'] = 'FAKE_PIPE'
    result['envelope_sha256'] = digest({k: v for k, v in result.items() if k != 'envelope_sha256'})
    with pytest.raises(AIWorkflowError, match='显示结果'):
        accept_ai_result(scene, result)


def test_cancelled_task_never_calls_provider_or_saves(scene, provider):
    cancel = threading.Event(); cancel.set()
    with pytest.raises(AIWorkflowError, match='取消'):
        evaluate_ai_scene(scene, cancel_event=cancel)
    provider.assert_not_called()
    assert not scene.with_name('ai_section_result.json').exists()


def test_no_sidecar_preserves_report_and_damaged_record_is_not_loaded(scene):
    assert load_ai_result(scene) is None
    report = {'counts': {'UNKNOWN': 2}}
    assert load_ai_result(scene, report) == report
    assert load_ai_result(scene, report) is not report
    unverified = {**report, 'ai_section_assist': {'status': 'CANDIDATES_READY', 'scope': 'SECTION_ONLY'}}
    assert load_ai_result(scene, unverified) == report
    assert 'ai_section_assist' in unverified
    scene.with_name('ai_section_result.json').write_text('{broken', encoding='utf-8')
    with pytest.raises(AIWorkflowError, match='无法读取'):
        load_ai_result(scene, report)


def test_result_too_large_is_rejected_before_sidecar_write(scene, provider, monkeypatch):
    result = evaluate_ai_scene(scene)
    monkeypatch.setattr('pipe_twin.ai_workflow.MAX_SIDECAR_BYTES', 200)
    with pytest.raises(AIWorkflowError, match='过大'):
        accept_ai_result(scene, result)
    assert not scene.with_name('ai_section_result.json').exists()


def test_verified_scene_cache_reuses_geometry_only_and_returns_isolated_copies(scene, monkeypatch):
    loader = Mock(wraps=ai_inputs.load_elevation_dataset)
    monkeypatch.setattr(ai_inputs, 'load_elevation_dataset', loader)
    raw = scene.read_bytes()
    first = ai_inputs._load_verified_scene(scene, raw)
    first['pipe_specs'][0]['nominal_diameter_mm'] = 999999
    second = ai_inputs._load_verified_scene(scene, raw)
    assert second['pipe_specs'][0]['nominal_diameter_mm'] != 999999
    loader.assert_called_once()
    prepared1 = prepare_ai_request(scene)
    prepared2 = prepare_ai_request(scene)
    assert prepared1 == prepared2
    loader.assert_called_once()


@pytest.mark.parametrize('target', ['model', 'photo'])
def test_cached_scene_rehashes_changed_bytes_even_when_mtime_is_unchanged(scene, monkeypatch, target):
    loader = Mock(wraps=ai_inputs.load_elevation_dataset)
    monkeypatch.setattr(ai_inputs, 'load_elevation_dataset', loader)
    prepare_ai_request(scene)
    manifest = json.loads(scene.read_text(encoding='utf-8'))
    relative = (manifest['model']['path'] if target == 'model' else
                manifest['capture']['capture_groups'][-1]['views']['right']['path'])
    path = scene.parent / relative
    original_stat = path.stat()
    data = bytearray(path.read_bytes())
    data[len(data)//2] ^= 1
    path.write_bytes(data)
    os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert path.stat().st_mtime_ns == original_stat.st_mtime_ns
    assert path.stat().st_size == original_stat.st_size
    with pytest.raises(AIInputError) as caught:
        prepare_ai_request(scene)
    assert caught.value.code == 'AI_INPUT_CHANGED'
    loader.assert_called_once()


def test_manifest_changes_invalidate_geometry_cache(scene, monkeypatch):
    loader = Mock(wraps=ai_inputs.load_elevation_dataset)
    monkeypatch.setattr(ai_inputs, 'load_elevation_dataset', loader)
    first = prepare_ai_request(scene)
    alter_manifest(scene, lambda m: m['analysis']['elevation_auto']['registration'].update(camera_side_world=[0, -1, 0]))
    second = prepare_ai_request(scene)
    assert first['input_hash'] != second['input_hash']
    assert loader.call_count == 2
