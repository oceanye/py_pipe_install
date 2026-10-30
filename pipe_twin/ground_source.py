"""Bind independent ground diagnostics to the current stereo reference pair.

A rejected burst never becomes valid here. Its current synchronized pair can
still explain a ground plane for conditional section diagnostics, just as the
standalone tilt inspector can. No historical images or planes are substituted.
"""
from __future__ import annotations

from . import ground_check


def reference_ground_source(group, capture, frames, *, compute_depth):
    burst = group['depth'].audit.get('temporal_fusion') or {}
    if (not frames or burst.get('status') != 'INVALID' or group.get('pair_healthy')
            or not capture.get('sync_valid')
            or not all(group.get('quality', {}).get(role, {}).get('passed') for role in ('left', 'right'))):
        return None
    reference = frames[-1]
    if reference['capture_id'] != burst.get('reference_capture_id'):
        raise ValueError('GROUND_REFERENCE_CAPTURE_MISMATCH')
    for role in ('left', 'right'):
        view = reference['views'][role]
        if (view['sha256'] != group['photos'][role]['actual_sha256']
                or any(view[key] != capture['views'][role][key] for key in ('sha256', 'captured_at'))):
            raise ValueError('GROUND_REFERENCE_PHOTO_MISMATCH')
    depth = compute_depth(group['left'], group['right'])
    if depth.audit.get('status') != 'VALID':
        return None
    return {
        'left': group['left'], 'right': group['right'], 'depth': depth, 'pair_healthy': True,
        'source_scope': 'REFERENCE_STEREO_PAIR_ONLY',
        'source_capture_id': reference['capture_id'],
        'source_photo_sha256': {r: reference['views'][r]['sha256'] for r in ('left', 'right')},
        'source_captured_at': {r: reference['views'][r]['captured_at'] for r in ('left', 'right')},
        'source_burst_status': burst['status'],
        'source_burst_reason_codes': list(burst.get('reason_codes', [])),
    }


def check_current_ground(group, calibration, region, *, section_stages=()):
    reference = group.get('reference_ground_pair')
    if reference is None or region is None:
        return ground_check.check_ground(group, calibration, region, section_stages=section_stages)
    result = ground_check.check_ground(reference, calibration, region, section_stages=section_stages)
    result.update({key: value for key, value in reference.items() if key.startswith('source_')})
    return result
