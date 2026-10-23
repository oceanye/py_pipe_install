import copy
import hashlib
import json

import cv2
import numpy as np
import pytest

from pipe_twin.ground_inspection import inspect_ground_pair
from pipe_twin.ground_visualization import decode_support
from test_elevation_dataset import _calibration
from test_ground_check import scene


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    group, _, region, _ = scene()
    group['depth'].audit = {'status': 'VALID'}
    monkeypatch.setattr('pipe_twin.ground_inspection._compute_stereo_depth', lambda *args: group['depth'])
    monkeypatch.setattr('pipe_twin.ground_inspection._quality', lambda *args: {'passed': True})
    paths, hashes = {}, {}
    for role in ('left', 'right'):
        path = tmp_path/(role+'.png')
        cv2.imwrite(str(path), np.full((480, 640, 3), 120, np.uint8))
        paths[role] = path
        hashes[role] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(paths=paths, expected_hashes=hashes, capture_times={r: '2026-10-23T09:00:00+08:00' for r in paths},
                calibration=_calibration(), region=region, analysis_settings={}, output_root=tmp_path/'output', pair_confirmed=True)


def test_standalone_requires_no_cad_and_saves_support_angles_and_exact_input_identity(inputs):
    original = copy.deepcopy(inputs['calibration'])
    path, check = inspect_ground_pair(**inputs)
    assert check['status'] == 'CONSISTENT_GROUND_PLANE'
    saved = json.loads(path.read_text(encoding='utf-8'))
    assert saved['scope'] == 'CURRENT_STEREO_PAIR_ONLY'
    assert not saved['new_camera_capture'] and not saved['pipe_dimensions_recomputed']
    assert decode_support(saved['ground_check'], (640, 480)) is not None
    assert (path.parent/'left_ground_overlay.png').is_file()
    assert inputs['calibration'] == original
    for role, source in saved['sources'].items():
        assert source['sha256'] == inputs['expected_hashes'][role]
        assert source['sha256'] == hashlib.sha256(inputs['paths'][role].read_bytes()).hexdigest()


@pytest.mark.parametrize('failure', ['changed_photo', 'wrong_size', 'sync', 'unconfirmed', 'no_ground', 'quality'])
def test_standalone_does_not_accept_incompatible_or_unhealthy_inputs(inputs, monkeypatch, failure):
    if failure == 'quality':
        monkeypatch.setattr('pipe_twin.ground_inspection._quality', lambda *args: {'passed': False})
        _, check = inspect_ground_pair(**inputs)
        assert check['status'] == 'CAPTURE_UNHEALTHY'
        assert 'observed_support' not in check
        return
    if failure in ('changed_photo', 'wrong_size'):
        path = inputs['paths']['left']
        cv2.imwrite(str(path), np.full((200, 300, 3), 20, np.uint8))
        if failure == 'wrong_size':
            inputs['expected_hashes']['left'] = hashlib.sha256(path.read_bytes()).hexdigest()
    elif failure == 'sync':
        inputs['capture_times']['right'] = '2026-10-23T09:00:02+08:00'
    elif failure == 'unconfirmed':
        inputs['pair_confirmed'] = False
    else:
        inputs['region'] = None
    with pytest.raises(ValueError):
        inspect_ground_pair(**inputs)
    assert not inputs['output_root'].exists()
