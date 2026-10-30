"""Observed ground support and orientation labels shared by GUI and evidence.

Only measured, valid depth pixels close to the fitted plane are painted. The
search rectangle is not a segmentation, and rejected hypotheses stay amber.
"""
from __future__ import annotations

import base64
from functools import lru_cache
import math
import zlib

import cv2
import numpy as np

ENCODING = 'packbits-zlib-base64-v1'
MAX_PIXELS = 32_000_000


def observed_support(z, valid, intrinsic, region, plane, tolerance):
    x, y, w, h = region['rect_px']
    normal = np.asarray(plane['normal_camera'], float)
    # n dot (K^-1 [u,v,1] * Z) + d; no synthetic depths or hole filling.
    coefficients = np.linalg.inv(intrinsic).T @ normal
    yy, xx = np.ogrid[y:y+h, x:x+w]
    residual = z * (xx*coefficients[0]+yy*coefficients[1]+coefficients[2]) + plane['offset_mm']
    mask = valid & np.isfinite(residual) & (np.abs(residual) <= tolerance)
    packed = np.packbits(mask.ravel()).tobytes()
    return {'encoding': ENCODING, 'coordinate_space': 'rectified_left',
            'image_size': list(region['image_size']), 'rect_px': [x, y, w, h],
            'pixel_count': int(mask.sum()), 'valid_pixel_count': int(valid.sum()),
            'plane_tolerance_mm': tolerance, 'inferred_pixels': 0,
            'data': base64.b64encode(zlib.compress(packed)).decode('ascii')}


@lru_cache(maxsize=4)
def _unpack(data, w, h):
    expected = (w*h+7)//8
    if not isinstance(data, str) or len(data) > (expected+1024)*2:
        raise ValueError('Invalid ground mask data')
    stream = zlib.decompressobj()
    raw = stream.decompress(base64.b64decode(data, validate=True), expected+1)
    if len(raw) != expected or not stream.eof or stream.unused_data or stream.unconsumed_tail:
        raise ValueError('Invalid ground mask size')
    mask = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), count=w*h).reshape(h, w)
    mask.setflags(write=False)
    return mask


def decode_support(check, image_size):
    """Return a small ROI mask; malformed or differently sized evidence is hidden."""
    support = (check or {}).get('observed_support') or {}
    try:
        if (support.get('encoding') != ENCODING or support.get('coordinate_space') != 'rectified_left'
                or tuple(support.get('image_size', ())) != tuple(image_size)):
            return None
        rect = support['rect_px']
        if len(rect) != 4 or any(type(v) is not int for v in rect):
            return None
        x, y, w, h = rect
        if (min(x, y) < 0 or min(w, h) < 1 or w*h > MAX_PIXELS
                or x+w > image_size[0] or y+h > image_size[1]):
            return None
        mask = _unpack(support['data'], w, h)
        if int(mask.sum()) != support['pixel_count']:
            return None
        return rect, mask
    except (KeyError, TypeError, ValueError, zlib.error):
        return None


def checked_ground(check):
    return bool(check and check.get('status') == 'CONSISTENT_GROUND_PLANE'
                and check.get('surface_identity_source') == 'OPERATOR_MARKED_GROUND')


def tilt_angles(check):
    """Angle to plane is unsigned; roll follows x-right/y-down camera axes."""
    left = (check or {}).get('left_plane') or {}
    plane = left if left.get('status') == 'PLANE_FIT' else left.get('candidate_plane', {})
    try:
        normal = np.asarray(plane['normal_camera'], float)
        if normal.shape != (3,) or not np.isfinite(normal).all() or np.linalg.norm(normal) < 1e-9:
            return None
        normal /= np.linalg.norm(normal)
        pitch = math.degrees(math.asin(float(np.clip(abs(normal[2]), 0, 1))))
        roll = math.degrees(math.atan2(normal[0], -normal[1])) if np.hypot(*normal[:2]) >= math.sin(math.radians(5)) else None
        return pitch, roll
    except (KeyError, TypeError, ValueError):
        return None


def tilt_label_lines(check, *, compact=False):
    if not check:
        return []
    passed = checked_ground(check)
    trial = any(c.get('trial_mode') and c.get('applied') for c in check.get('corrections', []))
    lines = ['地面检查通过 · 绿色为平面支持像素' if passed else '地面未通过或未确认 · 橙色仅为候选支持']
    angles = tilt_angles(check)
    if angles:
        pitch, roll = angles
        prefix = '试验采用 / 未验证：' if trial else '' if passed else '候选 / 不用于修正：'
        lines.append(f'{prefix}光轴偏离地面法线 {90-pitch:.2f}°（0° 为垂直地面）')
        lines.append(f'{prefix}光轴与地面夹角 {pitch:.2f}°（0° 为平行）')
        lines.append(f'画面横滚 {roll:+.2f}°' if roll is not None else '横滚不可可靠观测')
    else:
        lines.append('倾角暂无有效解')
    support = check.get('observed_support')
    if support:
        count, valid = support['pixel_count'], support['valid_pixel_count']
        lines.append(f'涂色支持 {count:,} / {valid:,} 有效像素（{count/max(1, valid):.1%}）；空洞不填充')
    else:
        lines[0] = ('候选倾角已用于试验 · 无可涂色支持' if trial else
                    '地面检查通过 · 无涂色记录' if passed else '地面未通过或未确认 · 无可涂色支持')
    if not check.get('calibration_validated'):
        lines.append('暂用标定：角度精度未验证')
    source = ''
    if check.get('source_scope') == 'REFERENCE_STEREO_PAIR_ONLY':
        source = '地面仅使用当前帧'
        if check.get('source_burst_status') == 'INVALID':
            source += '；连拍仍未通过'
        lines.insert(0, source)
    if compact:
        state = ('候选倾角已用于试验 · 未验证' if trial else
                 '地面检查通过 · 绿色支持' if passed else '候选 / 不用于修正 · 橙色支持')
        if support:
            state += f" {support['pixel_count']/max(1, support['valid_pixel_count']):.1%}"
        else:
            state = ('候选倾角已用于试验 · 无可涂色支持' if trial else
                     '地面检查通过 · 无涂色记录' if passed else '地面未通过或未确认 · 无可涂色支持')
        if source:
            state = '仅当前帧：'+('候选试验未验证' if trial else '地面通过' if passed else '地面未通过/未确认')
            if check.get('source_burst_status') == 'INVALID':
                state += ' · 连拍未通过'
            if support:
                state += f" · 支持 {support['pixel_count']/max(1, support['valid_pixel_count']):.1%}"
        angle_text = '倾角暂无有效解'
        if angles:
            pitch, roll = angles
            angle_text = f'光轴偏离地面法线 {90-pitch:.2f}° · '+(f'横滚 {roll:+.2f}°' if roll is not None else '横滚不可观测')
        if not check.get('calibration_validated'):
            angle_text += ' · 暂用标定'
        return [state, angle_text]
    return lines


def paint_support(image, check, *, transform=None, image_size=None):
    """Paint into a copy, also supporting a zoomed/panned viewport."""
    out = image.copy()
    decoded = decode_support(check, image_size or image.shape[1::-1])
    if decoded is None:
        return out
    (x, y, w, h), mask = decoded
    color = np.array((115, 210, 70) if checked_ground(check) else (40, 170, 255))
    if transform is None:
        target = out[y:y+h, x:x+w]
        selected = mask.astype(bool)
    else:
        scale, ox, oy = transform
        matrix = np.array([[scale, 0, ox+x*scale], [0, scale, oy+y*scale]], dtype=float)
        selected = cv2.warpAffine(mask, matrix, out.shape[1::-1], flags=cv2.INTER_NEAREST).astype(bool)
        target = out
    target[selected] = np.rint(.58*target[selected]+.42*color).astype(np.uint8)
    return out


def ground_evidence_image(image, check):
    return ground_evidence_labels(paint_support(image, check), check)


def ground_evidence_labels(image, check):
    """Draw final labels above other annotations without painting samples twice."""
    out = image.copy()
    if not check or not check.get('selection'):
        return out
    x, y, w, h = check['selection']['rect_px']
    cv2.rectangle(out, (x, y), (x+w, y+h), (255, 169, 231), 2)
    color = (115, 210, 70) if checked_ground(check) else (40, 170, 255)
    status = 'GROUND CHECK PASSED' if checked_ground(check) else 'CANDIDATE ONLY - NOT VALIDATED'
    if any(c.get('trial_mode') and c.get('applied') for c in check.get('corrections', [])):
        status = 'TRIAL GROUND CORRECTION APPLIED - NOT VALIDATED'
    angles = tilt_angles(check)
    lines = [status]
    if check.get('source_scope') == 'REFERENCE_STEREO_PAIR_ONLY':
        scope = 'GROUND: CURRENT PAIR ONLY'
        if check.get('source_burst_status') == 'INVALID':
            scope += ' / BURST STILL REJECTED'
        lines.insert(0, scope)
    if angles:
        pitch, roll = angles
        lines += [f'Off ground normal: {90-pitch:.2f} deg (0 = perpendicular)',
                  f'Optical axis / plane: {pitch:.2f} deg (0 = parallel)',
                  f'Image roll: {roll:+.2f} deg' if roll is not None else 'Image roll: unobservable']
    else:
        lines += ['Tilt: unavailable']
    support = check.get('observed_support')
    if support:
        lines += [f"Measured support: {support['pixel_count']} / {support['valid_pixel_count']} valid pixels"]
    if not check.get('calibration_validated'):
        lines += ['PROVISIONAL CALIBRATION - accuracy unverified']
    for i, line in enumerate(lines):
        position = (12, 26+i*25)
        cv2.putText(out, line, position, cv2.FONT_HERSHEY_SIMPLEX, .62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, line, position, cv2.FONT_HERSHEY_SIMPLEX, .62, color, 1, cv2.LINE_AA)
    return out
