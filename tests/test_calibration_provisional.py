"""Operator-authorized use must not change the failed quality evidence."""
import copy
import json
from dataclasses import replace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from pipe_twin.calibration_quality import provisional_calibration_allowed, calibration_quality_text
from pipe_twin.capture_gui import field_calibration_problem
from pipe_twin.elevation_dataset import _calibration_ok, create_elevation_dataset, load_elevation_dataset
from pipe_twin.elevation_gui import ElevationCaptureDialog
from pipe_twin.workbench_profile import save_camera_calibration_bundle, load_camera_calibration_bundle
from test_elevation_dataset import _calibration, _recipe, MODEL_PATH
from pipe_twin.dxf_elevation import catalog_from_dxf


def provisional():
    calibration = _calibration()
    calibration['validated'] = False
    calibration['quality_audit'] = {
        'square_mm': 17., 'holdout': {'status': 'FAILED', 'cross_reprojection_rms_px': 1.996},
        'operator_acceptance': {
            'status': 'PROVISIONAL', 'authorized': True, 'scope': 'relative_elevation',
            'calibration_id': calibration['calibration_id'],
            'accepted_at': '2026-10-14T09:30:00+08:00', 'reason': 'User requested temporary use'},
    }
    return calibration


def test_acceptance_allows_relative_use_without_claiming_validated():
    calibration = provisional()
    assert provisional_calibration_allowed(calibration)
    assert field_calibration_problem(calibration) is None
    assert _calibration_ok(calibration).validated is False
    text = calibration_quality_text(calibration)
    assert '暂用标定' in text and '精度未验收' in text and '1.996' in text
    owner = object.__new__(ElevationCaptureDialog)
    owner.current_calibration = lambda: calibration
    owner.calibration_status = Mock()
    owner.calibration_details = Mock()
    owner.refresh_calibration_status()
    assert '暂用标定' in owner.calibration_status.set.call_args.args[0]
    assert '1.996' in owner.calibration_details.set.call_args.args[0]


@pytest.mark.parametrize('key,value', [
    ('authorized', 'true'), ('authorized', False), ('status', 'PASSED'),
    ('scope', 'absolute_cad'), ('calibration_id', 'another-camera'),
    ('reason', ''), ('accepted_at', '2026-10-14T09:30:00'), ('accepted_at', None),
])
def test_incomplete_or_unbound_acceptance_is_not_usable(key, value):
    calibration = provisional()
    calibration['quality_audit']['operator_acceptance'][key] = value
    assert not provisional_calibration_allowed(calibration)
    assert field_calibration_problem(calibration)
    with pytest.raises(ValueError, match='validated=true'):
        _calibration_ok(calibration)


def test_provisional_record_does_not_override_geometry_or_demo_checks():
    calibration = provisional()
    calibration['baseline_mm'] *= 2
    with pytest.raises(ValueError, match='baseline'):
        _calibration_ok(calibration)
    calibration = provisional()
    calibration['calibration_id'] = 'SYNTHETIC-CAMERA'
    calibration['quality_audit']['operator_acceptance']['calibration_id'] = calibration['calibration_id']
    assert field_calibration_problem(calibration)
    with pytest.raises(ValueError, match='合成'):
        _calibration_ok(calibration)


def test_bundle_and_elevation_session_preserve_failed_quality(tmp_path):
    calibration = provisional()
    original = copy.deepcopy(calibration)
    recipe = _recipe(calibration)
    bundle_path = save_camera_calibration_bundle(tmp_path/'calibration.json', calibration, rectification_recipe=recipe)
    assert load_camera_calibration_bundle(bundle_path)['stereo_calibration'] == original
    photos = []
    for role in ('left', 'right'):
        path = tmp_path/f'{role}.png'
        cv2.imwrite(str(path), np.full((480, 640, 3), 100 if role == 'left' else 110, np.uint8))
        photos.append(path)
    specs, _ = catalog_from_dxf(MODEL_PATH, axis_world=(0., 0., 1.), unitless_unit='millimeter')
    path = create_elevation_dataset(output_root=tmp_path/'session', calibration=calibration,
        rectification_recipe=recipe, left_path=photos[0], right_path=photos[1],
        left_time='2026-10-14T09:30:00+08:00', right_time='2026-10-14T09:30:00+08:00',
        pipe_specs=specs, pair_confirmed=True, model_path=MODEL_PATH, stl_unit='millimeter')
    loaded = load_elevation_dataset(path)
    assert loaded['calibration'] == original
    assert loaded['calibration']['quality_audit']['holdout']['status'] == 'FAILED'
    assert loaded['calibration']['validated'] is False


def test_provisional_manifest_runs_real_depth_recognition_and_report(tmp_path):
    from test_elevation_auto import _package
    from pipe_twin.stereo_analyzer import analyze_stereo_capture
    calibration = provisional()
    path, _ = _package(tmp_path, calibration=calibration)
    before = path.read_bytes()
    report_path = tmp_path/'report.json'
    result = analyze_stereo_capture(path, report_output_path=report_path,
                                    evidence_dir=tmp_path/'evidence')
    assert result['counts'] == {'INSTALLED': 0, 'NOT_INSTALLED': 0, 'UNKNOWN': 12}
    audit = json.loads(report_path.read_text(encoding='utf-8'))['calibration_audit']
    assert audit['validated'] is False
    assert audit['quality_audit'] == calibration['quality_audit']
    assert path.read_bytes() == before


@pytest.mark.parametrize('backend', ['cylinder', 'geometry_only', 'parallel_strip', 'image_guided', 'auto'])
def test_provisional_calibration_reaches_cropped_geometry_backends(backend):
    from test_elevation_auto import _groups
    from test_elevation_zones import _zone
    from pipe_twin.elevation_auto import analyze_elevation_auto_zones
    from pipe_twin.stereo_analyzer import _calibration_from_manifest
    calibration = _calibration_from_manifest(provisional())
    _, specs, groups = _groups()
    result = analyze_elevation_auto_zones(groups, calibration=calibration, pipe_specs=specs,
        scope_settings={'scope': 'zones', 'right_roi_padding_px': 128,
                        'zones': [_zone((120, 0, 440, 480))]},
        registration_settings={'local_observation_mode': backend})
    assert len(result['local_surface']['observations']) == 3
    assert calibration.validated is False


def test_backend_rejects_missing_acceptance_and_invalid_rectification():
    from test_elevation_auto import _groups
    from pipe_twin.local_surface import _validate_arrays, LocalSurfaceError
    from pipe_twin.stereo_analyzer import _calibration_from_manifest
    _, _, groups = _groups()
    group = groups[0]
    data = provisional()
    data['quality_audit']['operator_acceptance']['authorized'] = False
    # A raw input boolean must never substitute for the bound acceptance record.
    data['relative_elevation_provisional'] = True
    rejected = _calibration_from_manifest(data)
    accepted = _calibration_from_manifest(provisional())
    for calibration in (rejected, replace(accepted, rectified=False),
                        replace(accepted, baseline_mm=120)):
        with pytest.raises(LocalSurfaceError):
            _validate_arrays(group['left'], group['right'], group['depth'], calibration)
