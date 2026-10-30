import copy
from datetime import datetime, timedelta, timezone
import hashlib
import math

import cv2
import numpy as np
import pytest

import pipe_twin.ground_monitor as module
from pipe_twin.ground_monitor import GroundMonitorState, ground_monitor_pose, inspect_live_ground
from pipe_twin.stereo_camera import CapturedStereoPair
from test_elevation_dataset import _calibration, _recipe
from test_ground_check import add_ground_images, scene


START = datetime(2026, 10, 30, 12, tzinfo=timezone.utc)


def rotation(degrees, axis=(0, 1, 0)):
    return cv2.Rodrigues(np.asarray(axis, float)*math.radians(degrees))[0]


def check(normal=(0, 0, -1), right_normal=None):
    return {'status': 'CONSISTENT_GROUND_PLANE',
            'left_plane': {'normal_camera': list(normal)},
            'right_plane': {'normal_camera': list(normal if right_normal is None else right_normal)},
            'accepted_measurement': False}


def measurement(sequence, seconds, *, degrees=0, right_degrees=None, recipe=True):
    normal = rotation(degrees)@np.array([0., 0., -1.])
    right_normal = rotation(degrees if right_degrees is None else right_degrees)@np.array([0., 0., -1.])
    ground = check(normal, right_normal)
    pose = ground_monitor_pose(ground, rectification_recipe=_recipe(_calibration()) if recipe else None)
    return {'sequence': sequence, 'captured_at': (START+timedelta(seconds=seconds)).isoformat(),
            'config_signature': 'one-config', 'ground_check': ground, 'pose': pose,
            'status': ground['status'], 'valid': True}


def at(seconds):
    return START+timedelta(seconds=seconds)


def images(normal=(.01, -.015, -1)):
    group, calibration, region, normal = scene(normal, distance=800.)
    add_ground_images(group, calibration, normal, distance=800.)
    group['depth'].audit = {'status': 'VALID'}
    pair = CapturedStereoPair(group['left'], group['right'], START.isoformat(),
                              (START+timedelta(milliseconds=1)).isoformat(), 1., 'test_actual_capture', {})
    return group, region, normal, pair


def test_real_rendered_stereo_geometry_uses_strict_depth_and_subpixel_checks_without_writes(tmp_path, monkeypatch):
    group, region, normal, pair = images()
    calibration = _calibration()
    recipe = _recipe(calibration)
    original = copy.deepcopy((calibration, recipe, region))
    monkeypatch.chdir(tmp_path)
    result = inspect_live_ground(pair=pair, sequence=7, calibration=calibration, region=region,
                                 analysis_settings={}, rectification_recipe=recipe)
    assert result['valid'] and result['status'] == 'CONSISTENT_GROUND_PLANE'
    expected = math.degrees(math.acos(abs(normal[2])))
    assert result['pose']['off_normal_deg'] == pytest.approx(expected, abs=.15)
    assert result['pose']['tilt_right_deg'] < 0 and result['pose']['tilt_down_deg'] > 0
    assert result['pose']['coordinate_frame'] == 'RAW_LEFT_OPTICAL'
    assert result['pose']['physical_axes_known'] and not result['pose']['yaw_observable']
    assert result['captured_at'] == START.isoformat() and result['sequence'] == 7
    assert result['sync_delta_ms'] == pytest.approx(1.)
    assert result['sources']['left']['pixel_sha256'] == hashlib.sha256(group['left'].tobytes()).hexdigest()
    assert result['ground_check']['subpixel_refinement']['left']['status'] == 'REFINED_MATCHES'
    assert result['ground_check']['observed_support']['sampling'] == 'SPARSE_REFINED_STEREO_MATCHES'
    assert (calibration, recipe, region) == original
    assert list(tmp_path.iterdir()) == []


def test_raw_camera_normals_use_inverse_rotations_and_independent_eye_planes():
    recipe = _recipe(_calibration())
    r1, r2 = rotation(5.), rotation(-3., (1, 0, 0))
    recipe['R1'], recipe['R2'] = r1.tolist(), r2.tolist()
    raw_left = np.array([.05, -.08, -1.]); raw_left /= np.linalg.norm(raw_left)
    raw_right = np.array([-.03, .11, -1.]); raw_right /= np.linalg.norm(raw_right)
    ground = check(r1@raw_left, r2@raw_right)
    value = ground_monitor_pose(ground, rectification_recipe=recipe)
    assert value['off_normal_deg'] == pytest.approx(math.degrees(math.acos(abs(raw_left[2]))))
    assert value['right_off_normal_deg'] == pytest.approx(math.degrees(math.acos(abs(raw_right[2]))))
    np.testing.assert_allclose(value['axis_target_camera'], -raw_left)
    assert value['rectified_off_normal_deg'] != pytest.approx(value['off_normal_deg'])
    flipped = ground_monitor_pose(check(-(r1@raw_left), -(r2@raw_right)), rectification_recipe=recipe)
    for key in ('off_normal_deg', 'right_off_normal_deg', 'tilt_right_deg', 'tilt_down_deg'):
        assert flipped[key] == pytest.approx(value[key])


@pytest.mark.parametrize('failure', ['bad_id', 'nonrotation', 'nan', 'missing_r2'])
def test_invalid_recipe_is_rejected_before_depth(failure, monkeypatch):
    _, region, _, pair = images()
    calibration = _calibration()
    recipe = _recipe(calibration)
    if failure == 'bad_id':
        recipe['calibration_id'] = 'OTHER-CAMERA'
    elif failure == 'missing_r2':
        del recipe['R2']
    elif failure == 'nan':
        recipe['R1'][0][0] = float('nan')
    else:
        recipe['R1'][0][0] = 2.
    monkeypatch.setattr(module, '_compute_stereo_depth', lambda *a: pytest.fail('Bad recipe must not compute'))
    with pytest.raises(ValueError):
        inspect_live_ground(pair=pair, sequence=0, calibration=calibration, region=region,
                            analysis_settings={}, rectification_recipe=recipe)


@pytest.mark.parametrize('failure', ['dark', 'sync', 'size'])
def test_bad_current_images_never_inherit_previous_pose_or_compute_depth(failure, monkeypatch):
    group, region, _, _ = images()
    times = {'left': START.isoformat(), 'right': START.isoformat()}
    if failure == 'dark':
        group['left'][:] = 0
    elif failure == 'sync':
        times['right'] = (START+timedelta(seconds=1)).isoformat()
    else:
        group['right'] = group['right'][:-1]
    monkeypatch.setattr(module, '_compute_stereo_depth', lambda *a: pytest.fail('Unhealthy source must not compute'))
    result = inspect_live_ground(left=group['left'], right=group['right'], capture_times=times,
        sequence=4, calibration=_calibration(), region=region,
        analysis_settings={'minimum_luminance_p05': 5.})
    assert not result['valid'] and result['pose'] is None
    assert result['sequence'] == 4 and result['captured_at'] == START.isoformat()
    assert result['status'] == {'dark': 'NEAR_BLACK_FRAME', 'sync': 'STEREO_UNSYNCHRONIZED',
                                 'size': 'GROUND_IMAGE_PAIR_INVALID'}[failure]
    if failure == 'dark':
        assert 'DARK_EXPOSURE' in result['quality']['left']['reason_codes']


def test_near_black_monitor_guard_works_even_when_analysis_allows_dark_images(monkeypatch):
    group, region, _, pair = images()
    pair.left[:] = 1
    pair.right[:] = 1
    monkeypatch.setattr(module, '_compute_stereo_depth', lambda *a: pytest.fail('Near-black images must not compute'))
    result = inspect_live_ground(pair=pair, sequence=1, calibration=_calibration(), region=region, analysis_settings={})
    assert all(q['passed'] for q in result['quality'].values())
    assert result['status'] == 'NEAR_BLACK_FRAME' and result['pose'] is None
    assert result['monitor_quality']['left']['near_black']


def test_small_bright_ground_roi_is_not_mistaken_for_black_frame(monkeypatch):
    group, region, _, pair = images()
    region['rect_px'] = [320, 320, 32, 32]
    pair.left[:] = 1
    pair.right[:] = 1
    pair.left[320:352, 320:352] = 100
    pair.right[320:352, 280:312] = 100
    calls = []
    def compute(*args):
        calls.append(True)
        return group['depth']
    monkeypatch.setattr(module, '_compute_stereo_depth', compute)
    result = inspect_live_ground(pair=pair, sequence=1, calibration=_calibration(), region=region, analysis_settings={})
    assert result['quality']['left']['dark_pixel_fraction'] > .995
    assert not result['monitor_quality']['left']['near_black']
    assert not result['monitor_quality']['right']['near_black']
    assert calls and result['status'] != 'NEAR_BLACK_FRAME'


def test_perfect_fake_depth_cannot_make_textureless_current_images_pass(monkeypatch):
    group, region, _, pair = images()
    pair.left[:] = 120
    pair.right[:] = 120
    monkeypatch.setattr(module, '_compute_stereo_depth', lambda *a: group['depth'])
    result = inspect_live_ground(pair=pair, sequence=1, calibration=_calibration(), region=region, analysis_settings={})
    assert result['status'] == 'INSUFFICIENT_GROUND_SUBPIXEL_SUPPORT'
    assert not result['valid'] and result['pose'] is None


def test_normal_without_recipe_is_virtual_reference_only_and_never_green():
    state = GroundMonitorState()
    for seq in range(4):
        result = state.update(measurement(seq, seq, recipe=False), now=at(seq+.1))
        assert result['valid'] and not result['green'] and result['stable_count'] == 0
        assert result['status'] == 'RECTIFIED_REFERENCE_ONLY'
        assert result['pose']['coordinate_frame'] == 'RECTIFIED_OPTICAL'
        assert not result['pose']['yaw_observable']


def test_three_fresh_stable_samples_green_both_physical_axes_only():
    state = GroundMonitorState()
    for seq, degree in enumerate([.2, .3, .1]):
        value = state.update(measurement(seq, seq, degrees=degree), now=at(seq+.2), expected_sequence=seq)
        assert value['green'] == (seq == 2)
        assert value['stable_count'] == seq+1
    assert value['status'] == 'ALIGNED_STABLE' and not value['accepted_measurement']
    value['pose']['off_normal_deg'] = 90
    assert state.snapshot(now=at(3))['pose']['off_normal_deg'] < 1
    value = state.update(measurement(3, 3, degrees=.1, right_degrees=2.1), now=at(3.1))
    assert value['status'] == 'ADJUST_TILT' and value['valid'] and not value['green']
    assert value['stable_count'] == 0


def test_whole_window_normal_span_rejects_slow_drift_chain():
    state = GroundMonitorState(max_step_deg=.5)
    counts = [state.update(measurement(i, i, degrees=d), now=at(i+.1))['stable_count']
              for i, d in enumerate([0, .4, .8])]
    assert counts == [1, 2, 1]
    assert not state.snapshot(now=at(3))['green']


def test_right_eye_independent_jitter_cannot_hide_behind_stable_left_eye():
    state = GroundMonitorState(max_step_deg=.5)
    for i, right in enumerate([.1, 1.9, .1]):
        result = state.update(measurement(i, i, degrees=0, right_degrees=right), now=at(i+.1))
        assert result['stable_count'] == 1 and not result['green']


@pytest.mark.parametrize('failure', ['rejected_ground', 'old_sequence', 'wrong_sequence', 'old_time', 'future', 'expired'])
def test_invalid_late_or_stale_result_clears_green_and_pose_immediately(failure):
    state = GroundMonitorState()
    for i in range(3):
        state.update(measurement(i, i), now=at(i+.1))
    value = measurement(3, 3)
    expected = 3
    now = at(3.1)
    if failure == 'rejected_ground':
        value.update(valid=False, status='RIGHT_GROUND_PLANE_INCONSISTENT', pose=None)
    elif failure == 'old_sequence':
        value['sequence'] = 1; expected = 1
    elif failure == 'wrong_sequence':
        expected = 4
    elif failure == 'old_time':
        value['captured_at'] = at(1).isoformat()
    elif failure == 'future':
        value['captured_at'] = at(4).isoformat()
    else:
        now = at(12)
    status = state.update(value, now=now, expected_sequence=expected)
    assert not status['valid'] and not status['green'] and status['pose'] is None
    assert status['stable_count'] == 0


def test_roi_or_calibration_change_reset_breaks_stability_and_expiry_needs_new_samples():
    state = GroundMonitorState()
    state.update(measurement(0, 0), now=at(.1))
    state.update(measurement(1, 1), now=at(1.1))
    result = measurement(2, 2)
    result['config_signature'] = 'changed-region-or-calibration'
    assert state.update(result, now=at(2.1))['stable_count'] == 1
    assert state.snapshot(now=at(10.1))['status'] == 'EXPIRED'
    assert state.snapshot(now=at(10.1))['pose'] is None
    state.reset()
    assert state.update(measurement(0, 11), now=at(11.1))['stable_count'] == 1
    assert state.snapshot(now=at(11.2), expected_sequence=1)['status'] == 'STALE_SEQUENCE'


def test_slow_fresh_capture_results_use_longer_history_without_stale_display():
    state = GroundMonitorState(max_age_s=8)
    state.update(measurement(0, 0), now=at(4.6))
    assert state.snapshot(now=at(8.1))['pose'] is None
    state.update(measurement(1, 4.6), now=at(9.2))
    assert state.snapshot(now=at(12.7))['pose'] is None
    result = state.update(measurement(2, 9.2), now=at(13.8))
    assert result['stable_count'] == 3 and result['green']
    assert result['age_s'] == pytest.approx(4.6)
    assert state.snapshot(now=at(17.3))['status'] == 'EXPIRED'
    # A capture gap exceeds 8s: previously fresh history must not reconnect.
    result = state.update(measurement(3, 18), now=at(18.1))
    assert result['stable_count'] == 1 and not result['green']


def test_old_queued_results_never_count_toward_stable_and_history_is_bounded():
    state = GroundMonitorState(max_age_s=8)
    for i in range(3):
        result = state.update(measurement(i, i), now=at(15+i))
        assert result['status'] == 'EXPIRED' and result['stable_count'] == 0
    # A configurable short historical window also invalidates formerly green.
    state = GroundMonitorState(max_age_s=8, stability_window_s=8)
    state.reset()
    for i in range(3):
        state.update(measurement(i, i), now=at(i+.1))
    assert state.snapshot(now=at(8.1))['status'] == 'EXPIRED_STABILITY_WINDOW'


@pytest.mark.parametrize('option,value', [('tolerance_deg', 0), ('tolerance_deg', float('nan')),
                                        ('required_samples', 1), ('max_age_s', -1), ('max_step_deg', 0)])
def test_state_rejects_invalid_configuration(option, value):
    with pytest.raises(ValueError):
        GroundMonitorState(**{option: value})
