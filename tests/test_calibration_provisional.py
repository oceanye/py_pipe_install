"""Operator-authorized use must not change the failed quality evidence."""
import copy
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
    owner.refresh_calibration_status()
    assert '暂用标定' in owner.calibration_status.set.call_args.args[0]


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
