import base64
import copy
import json
import zlib

import cv2
import numpy as np
import pytest

from pipe_twin.ground_check import check_ground
from pipe_twin.ground_visualization import (checked_ground, decode_support, paint_support,
                                          tilt_angles, tilt_label_lines, ground_evidence_image)
from test_ground_check import scene


def test_support_preserves_missing_depth_obstacle_holes_and_original_pixels():
    group, calibration, region, _ = scene()
    group['depth'].left_valid[390:400, 160:560] = False
    group['depth'].left_depth_mm[325:340, 160:560] -= 250
    result = check_ground(group, calibration, region)
    assert checked_ground(result)
    rect, mask = decode_support(json.loads(json.dumps(result)), (640, 480))
    assert rect == region['rect_px']
    assert not mask[90:100].any() and not mask[25:40].any()
    assert mask.sum() == 400*(160-10-15)
    original = np.full((480, 640, 3), 100, np.uint8)
    colored = paint_support(original, result)
    changed = (colored != original).any(axis=2)
    assert changed.sum() == mask.sum()
    assert not changed[:300].any() and not changed[:, :160].any()
    assert original.min() == original.max() == 100
    assert result['observed_support']['inferred_pixels'] == 0


def test_viewport_mask_uses_the_same_zoom_pan_and_does_not_color_roi_background():
    group, calibration, region, _ = scene()
    group['depth'].left_valid[390:400, 160:560] = False
    result = check_ground(group, calibration, region)
    (_, _, _, _), mask = decode_support(result, (640, 480))
    full = np.zeros((480, 640), np.uint8); full[300:460, 160:560] = mask
    transform = (1.7, -120., -480.)
    expected = cv2.warpAffine(full, np.array([[1.7, 0, -120], [0, 1.7, -480]]), (400, 250), flags=cv2.INTER_NEAREST)
    original = np.full((250, 400, 3), 100, np.uint8)
    colored = paint_support(original, result, transform=transform, image_size=(640, 480))
    np.testing.assert_array_equal((colored != original).any(axis=2), expected.astype(bool))


@pytest.mark.parametrize('failure', ['left', 'right', 'unconfirmed'])
def test_rejected_and_unconfirmed_planes_have_amber_candidate_angles_not_correction(failure):
    group, calibration, region, _ = scene()
    if failure == 'left':
        group['depth'].left_depth_mm[300:380, 160:560] += 200
    elif failure == 'right':
        group['depth'].right_depth_mm *= 1.3
    else:
        region['selection_source'] = 'ASSISTANT_DIAGNOSTIC'
    result = check_ground(group, calibration, region)
    assert not checked_ground(result)
    assert tilt_angles(result) is not None
    assert '候选 / 不用于修正' in '\n'.join(tilt_label_lines(result))
    assert not result['pipe_axis_constrained_by_ground']
    out = paint_support(np.zeros((480, 640, 3), np.uint8), result)
    colored = out[out.any(axis=2)]
    assert len(colored) > 0 and np.all(colored[:, 2] > colored[:, 1])
    if failure != 'unconfirmed':
        assert 'optical_axis_to_ground_deg' not in result


def test_unobservable_and_missing_angles_are_not_published_as_zero_roll():
    group, calibration, region, _ = scene((0, 0, -1))
    result = check_ground(group, calibration, region)
    assert tilt_angles(result) == (90., None)
    assert '横滚不可可靠观测' in tilt_label_lines(result)
    group['depth'].left_valid[:] = False
    empty = check_ground(group, calibration, region)
    assert tilt_angles(empty) is None and decode_support(empty, (640, 480)) is None
    assert '倾角暂无有效解' in tilt_label_lines(empty)


def test_tilt_from_ground_normal_is_distinct_from_optical_axis_to_plane():
    pitch = np.deg2rad(75.45)
    result = {'status': 'GROUND_PLANE_INCONSISTENT',
        'left_plane': {'candidate_plane': {'normal_camera': [0, -np.cos(pitch), -np.sin(pitch)]}}}
    full = '\n'.join(tilt_label_lines(result))
    assert '光轴偏离地面法线 14.55°（0° 为垂直地面）' in full
    assert '光轴与地面夹角 75.45°（0° 为平行）' in full
    assert '光轴偏离地面法线 14.55°' in '\n'.join(tilt_label_lines(result, compact=True))


@pytest.mark.parametrize('damage', ['size', 'bounds', 'data', 'bomb', 'count'])
def test_bad_or_stale_serialized_support_is_not_painted(damage):
    group, calibration, region, _ = scene()
    result = copy.deepcopy(check_ground(group, calibration, region))
    support = result['observed_support']
    if damage == 'size':
        support['image_size'] = [1920, 1080]
    elif damage == 'bounds':
        support['rect_px'][2] = 100000
    elif damage == 'data':
        support['data'] = '?not base64'
    elif damage == 'bomb':
        support['data'] = base64.b64encode(zlib.compress(b'\0'*1_000_000)).decode('ascii')
    else:
        support['pixel_count'] += 1
    assert decode_support(result, (640, 480)) is None
    image = np.full((480, 640, 3), 100, np.uint8)
    np.testing.assert_array_equal(paint_support(image, result), image)


@pytest.mark.parametrize('trial', [False, True])
def test_current_pair_source_is_visible_in_gui_and_evidence_even_when_ground_passes(monkeypatch, trial):
    group, calibration, region, _ = scene()
    result = check_ground(group, calibration, region)
    result.update(source_scope='REFERENCE_STEREO_PAIR_ONLY', source_burst_status='INVALID')
    if trial:
        result['status'] = 'GROUND_PLANE_INCONSISTENT'
        result['corrections'] = [{'trial_mode': True, 'applied': True}]
    full = '\n'.join(tilt_label_lines(result))
    compact = tilt_label_lines(result, compact=True)
    assert '地面仅使用当前帧；连拍仍未通过' in full
    assert len(compact) == 2
    assert '仅当前帧' in compact[0] and '连拍未通过' in compact[0]
    assert ('候选试验未验证' if trial else '地面通过') in compact[0]
    drawn = []
    monkeypatch.setattr(cv2, 'putText', lambda image, text, *a, **k: drawn.append(text))
    ground_evidence_image(np.zeros((480, 640, 3), np.uint8), result)
    assert 'GROUND: CURRENT PAIR ONLY / BURST STILL REJECTED' in drawn
    assert ('TRIAL GROUND CORRECTION APPLIED - NOT VALIDATED' if trial else 'GROUND CHECK PASSED') in drawn
