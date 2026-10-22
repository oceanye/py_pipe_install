"""A left-image direction segment and two perpendicular pooling limits.

The segment is in rectified image pixels, not a known metric length or a 3-D
axis. Right-eye limits are an epipolar search envelope, not correspondences.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

REVISION = 'left-image-axial-band-v1'
MINIMUM_SPAN_PX = 32.


def normalize_axial_band(payload, *, image_size=None):
    if payload is None:
        return None
    allowed = {'revision', 'coordinate_space', 'image_size', 'axis_segment_px'}
    if not isinstance(payload, Mapping) or set(payload)-allowed:
        raise ValueError('管段汇聚范围配置无效')
    if payload.get('revision', REVISION) != REVISION or payload.get('coordinate_space', 'rectified_left') != 'rectified_left':
        raise ValueError('管段汇聚范围必须使用左目矫正图坐标')
    size = payload.get('image_size')
    if not isinstance(size, (list, tuple)) or len(size) != 2 or any(type(v) is not int or v < 2 for v in size):
        raise ValueError('管段汇聚范围缺少有效的原图尺寸')
    if image_size is not None and tuple(size) != tuple(image_size):
        raise ValueError('照片尺寸已改变，请重新指定管长方向和汇聚范围')
    try:
        points = np.asarray(payload.get('axis_segment_px'), dtype=float)
    except (ValueError, TypeError) as error:
        raise ValueError('管长方向线必须包含两个有效端点') from error
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError('管长方向线必须包含两个有效端点')
    if np.any(points < 0) or np.any(points > np.asarray(size)-1):
        raise ValueError('管长方向线端点不能超出原图')
    if np.linalg.norm(points[1]-points[0]) < MINIMUM_SPAN_PX-1e-8:
        raise ValueError('两条汇聚边界间距至少需要 32 个原图像素')
    return {'revision': REVISION, 'coordinate_space': 'rectified_left',
            'image_size': list(size), 'axis_segment_px': points.tolist()}


def band_geometry(band):
    points = np.asarray(band['axis_segment_px'], float)
    length = float(np.linalg.norm(points[1]-points[0]))
    return points[0], (points[1]-points[0])/length, length


def move_band_limit(band, index, point):
    """Move one limit along the unchanged direction, clamped to the image."""
    band = normalize_axial_band(band)
    origin, axis, length = band_geometry(band)
    if index not in (0, 1):
        raise ValueError('无效的汇聚边界')
    low, high = -float('inf'), float('inf')
    for i in range(2):
        if abs(axis[i]) > 1e-10:
            bounds = sorted([(0-origin[i])/axis[i], (band['image_size'][i]-1-origin[i])/axis[i]])
            low, high = max(low, bounds[0]), min(high, bounds[1])
    if index == 0:
        high = min(high, length-MINIMUM_SPAN_PX)
    else:
        low = max(low, MINIMUM_SPAN_PX)
    station = float(np.clip((np.asarray(point)-origin)@axis, low, high))
    points = np.asarray(band['axis_segment_px'], float)
    points[index] = np.clip(origin+station*axis, 0, np.asarray(band['image_size'])-1)
    return normalize_axial_band({**band, 'axis_segment_px': points.tolist()})


def band_overlay(band):
    """Exact line/rectangle intersections and the clipped strip polygon."""
    origin, axis, length = band_geometry(band)
    width, height = band['image_size']
    rectangle = np.array([[0., 0.], [width-1., 0.], [width-1., height-1.], [0., height-1.]])
    lines = []
    for station in (0., length):
        intersections = []
        for first, second in zip(rectangle, np.roll(rectangle, -1, axis=0)):
            delta = second-first; denominator = float(delta@axis)
            if abs(denominator) < 1e-10:
                continue
            ratio = float((station-(first-origin)@axis)/denominator)
            if -1e-9 <= ratio <= 1+1e-9:
                point = first+np.clip(ratio, 0, 1)*delta
                if not any(np.linalg.norm(point-p) < 1e-6 for p in intersections):
                    intersections.append(point)
        lines.append([p.tolist() for p in intersections[:2]])
    polygon = list(rectangle)
    for normal, intercept in ((axis, float(origin@axis)), (-axis, float(-origin@axis-length))):
        clipped = []
        for first, second in zip(polygon, polygon[1:]+polygon[:1]):
            a, b = float(first@normal-intercept), float(second@normal-intercept)
            if a >= -1e-9:
                clipped.append(first)
            if (a < 0) != (b < 0):
                clipped.append(first+(second-first)*a/(a-b))
        polygon = clipped
    return {'boundary_lines_px': lines, 'polygon_px': [p.tolist() for p in polygon],
            'axis_image': axis.tolist(), 'span_px': length}


def band_sampling_masks(band, left_shape, right_shape, *, offsets=None, disparity_range=(0., 256.), right_vertical_offset=0.):
    """Left samples stay inside the band; right search is the disparity union.

    Offsets restore each zone's full-image coordinates. For a right pixel q,
    some possible left mate q+(d, cy_left-cy_right) must lie in the band.
    No depth, hidden pixels or precise mate is synthesized by this envelope.
    """
    band = normalize_axial_band(band)
    offsets = offsets or {}
    origin, axis, length = band_geometry(band)
    limits = np.asarray(disparity_range, float)
    if limits.shape != (2,) or not np.isfinite(limits).all() or limits[1] < limits[0]:
        raise ValueError('双目对应搜索视差范围无效')
    masks, audit = {}, {}
    for role, shape in [('left', left_shape), ('right', right_shape)]:
        yy, xx = np.ogrid[:shape[0], :shape[1]]
        station = ((xx+offsets.get(role+'_x', 0)-origin[0])*axis[0]
                   +(yy+offsets.get(role+'_y', 0)-origin[1]+(right_vertical_offset if role == 'right' else 0))*axis[1])
        low, high = 0., length
        if role == 'right':
            shifts = limits*axis[0]
            low -= float(shifts.max()); high -= float(shifts.min())
        masks[role] = (station >= low-1e-8) & (station <= high+1e-8)
        audit[role] = {'station_interval_px': [low, high], 'selected_pixels': int(masks[role].sum()),
                       'image_origin_px': [offsets.get(role+'_x', 0), offsets.get(role+'_y', 0)]}
    return masks, {'selection': band, 'axis_image': axis.tolist(), 'span_px': length,
        'left_scope': 'STRICT_USER_BAND', 'right_scope': 'EPIPOLAR_SEARCH_ENVELOPE',
        'disparity_range_px': limits.tolist(), 'views': audit, 'supplies_metric_length_or_3d_axis': False}
