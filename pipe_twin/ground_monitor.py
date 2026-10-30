"""Read-only, low-rate ground measurements for adjusting a live stereo camera.

Images must already be rectified. Every result belongs to its captured pair,
not the newest preview frame. Alignment is guidance, never an accuracy claim.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math

import cv2
import numpy as np

from .elevation_dataset import _calibration_ok
from .ground_check import check_ground, normalize_ground_region
from .stereo_analyzer import _analysis_config, _compute_stereo_depth, _quality
from .workbench_profile import validate_rectification_recipe


NEAR_BLACK_POLICY = {'maximum_luminance_p95': 8., 'minimum_dark_fraction': .995,
                     'dark_pixel_maximum': 16, 'scope': 'WHOLE_FRAME_AND_GROUND_SEARCH_REGION'}


def _near_black_check(images, quality, selection, config):
    """Skip expensive matching only for effectively black input, ROI included.

This monitor-only guard does not change analysis quality or ground thresholds.
A small lit selected patch on an otherwise black image still gets checked.
"""
    x, y, w, h = selection['rect_px']
    matcher = config['stereo_matching']
    audits = {}
    for role, image in images.items():
        xmin, xmax = x, x+w
        if role == 'right':
            xmin = max(0, x-(matcher['min_disparity']+matcher['num_disparities']-1))
            xmax = min(image.shape[1], x+w-matcher['min_disparity'])
        patch = cv2.cvtColor(image[y:y+h, xmin:xmax], cv2.COLOR_BGR2GRAY) if xmax > xmin else np.empty(0)
        patch_p95 = float(np.percentile(patch, 95)) if patch.size else 0.
        patch_dark = float(np.mean(patch <= NEAR_BLACK_POLICY['dark_pixel_maximum'])) if patch.size else 1.
        whole_near_black = (quality[role]['luminance_p95'] <= NEAR_BLACK_POLICY['maximum_luminance_p95']
                            and quality[role]['dark_pixel_fraction'] >= NEAR_BLACK_POLICY['minimum_dark_fraction'])
        patch_near_black = (patch_p95 <= NEAR_BLACK_POLICY['maximum_luminance_p95']
                            and patch_dark >= NEAR_BLACK_POLICY['minimum_dark_fraction'])
        audits[role] = {'near_black': bool(whole_near_black and patch_near_black),
                        'ground_search_rect_px': [xmin, y, max(0, xmax-xmin), h],
                        'ground_search_luminance_p95': patch_p95, 'ground_search_dark_fraction': patch_dark}
    return {'policy': dict(NEAR_BLACK_POLICY), **audits}


def _time(value=None):
    value = datetime.now(timezone.utc) if value is None else value
    value = datetime.fromisoformat(value) if isinstance(value, str) else value
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError('Ground monitor timestamps must include a timezone')
    return value.astimezone(timezone.utc)


def _angles(normal):
    normal = np.asarray(normal, dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-12:
        raise ValueError('Invalid ground normal')
    normal = normal/np.linalg.norm(normal)
    # Choose the plane-normal direction facing into the scene. Camera +X is
    # image right, +Y is image down; do not infer any world heading from it.
    if normal[2] < 0:
        normal = -normal
    return {
        'axis_target_camera': normal.tolist(),
        'off_normal_deg': math.degrees(math.acos(float(np.clip(normal[2], 0, 1)))),
        'tilt_right_deg': math.degrees(math.atan2(float(normal[0]), float(normal[2]))),
        'tilt_down_deg': math.degrees(math.atan2(float(normal[1]), float(normal[2]))),
    }


def ground_monitor_pose(check, *, rectification_recipe=None):
    """Extract pointing guidance only from an accepted, two-eye ground check.

The caller must validate a recipe against the active calibration first. Without
one, only the virtual rectified optical axis is known. A plane never fixes yaw.
"""
    if check.get('status') != 'CONSISTENT_GROUND_PLANE':
        return None
    try:
        normal = np.asarray(check['left_plane']['normal_camera'], dtype=float)
        rectified = _angles(normal)
        left, right = rectified, None
        if rectification_recipe is not None:
            left = _angles(np.asarray(rectification_recipe['R1'], float).T@normal)
            right_normal = np.asarray(check['right_plane']['normal_camera'], dtype=float)
            right = _angles(np.asarray(rectification_recipe['R2'], float).T@right_normal)
    except (KeyError, TypeError, ValueError):
        return None
    return {
        **left,
        'normal_camera': normal.tolist(),
        'normal_coordinate_frame': 'RECTIFIED_LEFT_CAMERA',
        'coordinate_frame': 'RAW_LEFT_OPTICAL' if right is not None else 'RECTIFIED_OPTICAL',
        'physical_axes_known': right is not None,
        'right_off_normal_deg': right['off_normal_deg'] if right is not None else None,
        'right_axis_target_camera': right['axis_target_camera'] if right is not None else None,
        'rectified_off_normal_deg': rectified['off_normal_deg'],
        'rectified_tilt_right_deg': rectified['tilt_right_deg'],
        'rectified_tilt_down_deg': rectified['tilt_down_deg'],
        'yaw_observable': False,
        'accepted_measurement': False,
        'direction_definition': 'Positive right/down means aim toward camera image right/down; yaw is unobservable',
    }


def inspect_live_ground(*, sequence, calibration, region, analysis_settings,
                        pair=None, left=None, right=None, capture_times=None,
                        rectification_recipe=None):
    """Measure a captured rectified pair without writing any files.

Supply either a CapturedStereoPair or left/right arrays plus ISO capture times.
Returned pixel hashes identify the copied images actually used for this result.
Use GroundMonitorState on the UI thread to reject late/expired measurements.
"""
    if type(sequence) is not int or sequence < 0:
        raise ValueError('Ground monitor sequence must be a non-negative integer')
    if pair is not None:
        if left is not None or right is not None or capture_times is not None:
            raise ValueError('Supply a pair or separate images, not both')
        left, right = pair.left, pair.right
        capture_times = {'left': pair.left_captured_at, 'right': pair.right_captured_at}
    if capture_times is None:
        raise ValueError('Actual stereo capture times are required')
    times = {role: _time(capture_times[role]) for role in ('left', 'right')}
    parsed = _calibration_ok(calibration)
    if not parsed.rectified:
        raise ValueError('Ground monitor requires rectified images')
    selection = normalize_ground_region(region, image_size=(parsed.left.width, parsed.left.height))
    if selection is None:
        raise ValueError('Select visible ground in the rectified left image first')
    config = _analysis_config(analysis_settings)
    if config['stereo_matching']['num_disparities']+config['stereo_matching']['min_disparity'] >= parsed.left.width:
        raise ValueError('Disparity search range must be smaller than the image width')
    recipe = (validate_rectification_recipe(rectification_recipe, calibration=calibration)
              if rectification_recipe is not None else None)
    signature_payload = {'calibration': calibration, 'region': selection, 'analysis': config, 'rectification_recipe': recipe}
    signature = hashlib.sha256(json.dumps(signature_payload, sort_keys=True, allow_nan=False,
                                          separators=(',', ':')).encode('utf-8')).hexdigest()
    result = {
        'revision': 'live-ground-monitor-v1', 'sequence': sequence,
        'captured_at': min(times.values()).isoformat(),
        'capture_times': {r: t.isoformat() for r, t in times.items()},
        'config_signature': signature, 'sources': {}, 'quality': {}, 'monitor_quality': {}, 'depth_audit': None,
        'sync_delta_ms': abs((times['left']-times['right']).total_seconds()*1000),
        'scope': 'CURRENT_RECTIFIED_PAIR_ONLY', 'accepted_measurement': False,
    }
    images = {}
    image_valid = True
    for role, image in (('left', left), ('right', right)):
        camera = getattr(parsed, role)
        if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                or image.shape != (camera.height, camera.width, 3)):
            image_valid = False
            continue
        images[role] = np.ascontiguousarray(image).copy()
        result['sources'][role] = {
            'pixel_sha256': hashlib.sha256(images[role].tobytes()).hexdigest(),
            'captured_at': times[role].isoformat(), 'shape': list(images[role].shape),
        }
    if not image_valid:
        check = {'status': 'GROUND_IMAGE_PAIR_INVALID', 'accepted_measurement': False}
    elif result['sync_delta_ms'] > parsed.max_sync_delta_ms+1e-6:
        check = {'status': 'STEREO_UNSYNCHRONIZED', 'accepted_measurement': False}
    else:
        result['quality'] = {r: _quality(image, config) for r, image in images.items()}
        result['monitor_quality'] = _near_black_check(images, result['quality'], selection, config)
        if any(result['monitor_quality'][r]['near_black'] for r in images):
            check = {'status': 'NEAR_BLACK_FRAME', 'reason_codes': ['NEAR_BLACK_FRAME'], 'accepted_measurement': False}
        elif not all(q['passed'] for q in result['quality'].values()):
            check = {'status': 'CAPTURE_UNHEALTHY', 'accepted_measurement': False}
        else:
            depth = _compute_stereo_depth(images['left'], images['right'], parsed, config)
            result['depth_audit'] = depth.audit
            check = check_ground(dict(images, depth=depth, pair_healthy=depth.audit['status'] == 'VALID'),
                                 parsed, selection)
    pose = ground_monitor_pose(check, rectification_recipe=recipe)
    result.update(ground_check=check, status=check['status'], pose=pose, valid=pose is not None,
                  generated_at=datetime.now(timezone.utc).isoformat())
    return result


class GroundMonitorState:
    """Fresh, consecutive pointing guidance; no old angle is carried forward.

    Each sample must arrive fresh, inside the auxiliary target tolerance and
    within max_step_deg of every other sample (not just its neighbors). The
    current measurement expires independently of the longer stability window.
Only the physical optical axes of both cameras can earn a green indication.
"""

    def __init__(self, *, tolerance_deg=2., required_samples=3, max_age_s=8., max_step_deg=.5,
                 stability_window_s=None):
        for name, value, upper in [('tolerance_deg', tolerance_deg, 45.), ('max_age_s', max_age_s, 60.),
                                   ('max_step_deg', max_step_deg, 10.)]:
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= upper:
                raise ValueError(f'Invalid ground monitor {name}')
        if type(required_samples) is not int or not 2 <= required_samples <= 20:
            raise ValueError('required_samples must be between 2 and 20')
        self.tolerance_deg = float(tolerance_deg)
        self.required_samples = required_samples
        self.max_age_s = float(max_age_s)
        self.max_step_deg = float(max_step_deg)
        self.stability_window_s = self.max_age_s*self.required_samples if stability_window_s is None else stability_window_s
        if (type(self.stability_window_s) not in (int, float) or not math.isfinite(self.stability_window_s)
                or not self.max_age_s <= self.stability_window_s <= 1200):
            raise ValueError('stability_window_s must be finite and at least max_age_s')
        self.reset()

    def reset(self):
        self._history = []
        self._latest_sequence = -1
        self._latest_capture = None
        self._signature = None
        self._state = self._blank('WAITING_FOR_MEASUREMENT')
        return copy.deepcopy(self._state)

    def _blank(self, status, **metadata):
        return {'status': status, 'valid': False, 'green': False, 'pose': None,
                'stable_count': 0, 'required_samples': self.required_samples,
                'tolerance_deg': self.tolerance_deg, 'age_s': None,
                'max_age_s': self.max_age_s, 'stability_window_s': self.stability_window_s,
                'accepted_measurement': False, **metadata}

    def _invalidate(self, status, *, clear_history=True, **metadata):
        if clear_history:
            self._history.clear()
        self._state = self._blank(status, **metadata)
        return copy.deepcopy(self._state)

    def update(self, result, *, now=None, expected_sequence=None):
        now = _time(now)
        sequence = result.get('sequence')
        capture = _time(result['captured_at'])
        age = (now-capture).total_seconds()
        metadata = {'sequence': sequence, 'captured_at': capture.isoformat(), 'age_s': age}
        if (type(sequence) is not int or sequence < 0
                or expected_sequence is not None and sequence != expected_sequence
                or sequence <= self._latest_sequence):
            return self._invalidate('STALE_SEQUENCE', **metadata)
        self._latest_sequence = sequence
        if self._latest_capture is not None and capture <= self._latest_capture:
            return self._invalidate('NON_INCREASING_CAPTURE_TIME', **metadata)
        if self._latest_capture is not None and (capture-self._latest_capture).total_seconds() > self.max_age_s:
            self._history.clear()
        self._latest_capture = capture
        if age < 0 or age > self.max_age_s:
            return self._invalidate('CAPTURE_TIME_IN_FUTURE' if age < 0 else 'EXPIRED', **metadata)
        if result.get('config_signature') != self._signature:
            self._history.clear()
            self._signature = result.get('config_signature')
        pose = result.get('pose')
        if (not result.get('valid') or result.get('status') != 'CONSISTENT_GROUND_PLANE'
                or not isinstance(pose, dict)):
            return self._invalidate(result.get('status', 'INVALID_MEASUREMENT'), **metadata)
        try:
            normal = np.asarray(pose['axis_target_camera'], float)
            if normal.shape != (3,) or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1):
                raise ValueError('Invalid normal')
            off_normal = float(pose['off_normal_deg'])
            right = float(pose['right_off_normal_deg']) if pose.get('physical_axes_known') else None
            right_normal = np.asarray(pose['right_axis_target_camera'], float) if right is not None else None
            if right_normal is not None and (right_normal.shape != (3,) or not np.isfinite(right_normal).all()
                                            or not np.isclose(np.linalg.norm(right_normal), 1)):
                raise ValueError('Invalid right normal')
            if not math.isfinite(off_normal) or right is not None and not math.isfinite(right):
                raise ValueError('Invalid angle')
        except (KeyError, TypeError, ValueError):
            return self._invalidate('INVALID_MEASUREMENT', **metadata)
        base = dict(metadata, valid=True, pose=copy.deepcopy(pose))
        if not pose.get('physical_axes_known'):
            self._history.clear()
            self._state = self._blank('RECTIFIED_REFERENCE_ONLY', **base)
        elif max(off_normal, right) > self.tolerance_deg:
            self._history.clear()
            self._state = self._blank('ADJUST_TILT', **base)
        else:
            self._history = [h for h in self._history if (now-h['time']).total_seconds() <= self.stability_window_s]
            if any(math.degrees(math.acos(float(np.clip(current@h[key], -1, 1)))) > self.max_step_deg
                   for h in self._history for key, current in (('normal', normal), ('right_normal', right_normal))):
                self._history.clear()
            self._history.append({'normal': normal.copy(), 'right_normal': right_normal.copy(), 'time': capture})
            self._history = self._history[-self.required_samples:]
            count = len(self._history)
            stable = count == self.required_samples
            self._state = self._blank('ALIGNED_STABLE' if stable else 'TARGET_STABILIZING',
                                      **base, green=stable, stable_count=count)
        return copy.deepcopy(self._state)

    def snapshot(self, *, now=None, expected_sequence=None):
        """Call during preview repaint, even while no measurement has completed."""
        state = self._state
        if expected_sequence is not None and state.get('sequence') != expected_sequence:
            return self._invalidate('STALE_SEQUENCE', sequence=state.get('sequence'))
        if state.get('captured_at') is not None:
            age = (_time(now)-_time(state['captured_at'])).total_seconds()
            if age < 0 or age > self.max_age_s:
                return self._invalidate('CAPTURE_TIME_IN_FUTURE' if age < 0 else 'EXPIRED',
                                        clear_history=age < 0,
                                        sequence=state.get('sequence'), captured_at=state['captured_at'], age_s=age)
            state['age_s'] = age
            if state['green'] and self._history and any(
                    (_time(now)-h['time']).total_seconds() > self.stability_window_s for h in self._history):
                return self._invalidate('EXPIRED_STABILITY_WINDOW', sequence=state.get('sequence'),
                                        captured_at=state['captured_at'], age_s=age)
        return copy.deepcopy(state)
