"""Direction-independent image line coordinates, without rotating image pixels.

The sampling axis is unoriented. Its largest absolute component is positive
(X wins ties), keeping nearly vertical axes stable when their X component
changes sign. The normal is (-axis_y, axis_x); top/bottom mean its low/high side.
"""
from __future__ import annotations

import numpy as np


def canonical_image_axis(axis):
    value = np.array(axis, dtype=float, copy=True)
    if value.shape != (2,) or not np.isfinite(value).all() or np.linalg.norm(value) < 1e-12:
        raise ValueError('INVALID_IMAGE_AXIS')
    value /= np.linalg.norm(value)
    if value[int(np.argmax(np.abs(value)))] < 0:
        value = -value
    return value


def normal_from_axis(axis):
    axis = canonical_image_axis(axis)
    return np.array([-axis[1], axis[0]])


def sampling_coordinates(points, image_axis, origin=None):
    points = np.asarray(points, dtype=float)
    origin = np.zeros(2) if origin is None else np.asarray(origin, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all()
            or origin.shape != (2,) or not np.isfinite(origin).all()):
        raise ValueError('INVALID_IMAGE_POINTS')
    axis = canonical_image_axis(image_axis)
    return (points-origin) @ np.column_stack((axis, normal_from_axis(axis)))


def fit_image_line(points, image_axis=None):
    """Total least squares line a*u+b*v+c=0, with unit image normal."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2 or not np.isfinite(points).all():
        raise ValueError('INVALID_EDGE_POINTS')
    center = points.mean(axis=0)
    _, singular, vectors = np.linalg.svd(points-center, full_matrices=False)
    if singular[0] < 1e-10:
        raise ValueError('DEGENERATE_IMAGE_LINE')
    normal = vectors[-1]
    if image_axis is not None:
        direction = normal_from_axis(image_axis)
        if normal @ direction < 0:
            normal = -normal
    elif normal[int(np.argmax(np.abs(normal)))] < 0:
        normal = -normal
    return np.r_[normal, -normal@center]


def point_at_station(line, image_axis, station):
    """Intersect a homogeneous line with image_axis·pixel=station."""
    line = np.asarray(line, dtype=float)
    axis = canonical_image_axis(image_axis)
    if line.shape != (3,) or not np.isfinite(line).all() or not np.isfinite(station):
        raise ValueError('INVALID_IMAGE_LINE')
    matrix = np.stack((axis, line[:2]))
    if abs(np.linalg.det(matrix)) < 1e-10:
        raise ValueError('IMAGE_REFERENCE_STATION_UNOBSERVABLE')
    return np.linalg.solve(matrix, np.array([station, -line[2]]))
