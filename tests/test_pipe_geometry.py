"""Analytic checks for centre, near surface, silhouette and slant distance."""
import math

import numpy as np
import pytest

from pipe_twin.pipe_geometry import cylinder_section_geometry, pipe_measurement_record
from pipe_twin.elevation_gui import measurement_summary_lines


def geometry(center=(0, 0, 1290), axis=(1, 0, 0), diameter=51):
    return cylinder_section_geometry(center, axis, diameter, section_definition="TEST_COMMON_SECTION",
                                     intrinsic=[[1000, 0, 640], [0, 1000, 360], [0, 0, 1]])


def test_center_near_surface_and_silhouette_are_three_different_depths():
    result = geometry()
    assert result["centerline"]["depth_z_mm"] == 1290
    assert result["minimum_surface_depth_z_mm"] == pytest.approx(1264.5)
    assert result["nearest_surface_range_mm"] == pytest.approx(1264.5)
    for point in result["silhouette_tangent_points"]:
        assert point["depth_z_mm"] == pytest.approx(1290 - 25.5**2 / 1290)
    assert result["silhouette_tangent_points"][0]["pixel_xy"][1] < result["silhouette_tangent_points"][1]["pixel_xy"][1]


@pytest.mark.parametrize("axis", [(1, 0, 0), (1, .2, .4), (-1, -.2, -.4)])
def test_off_axis_distances_match_exhaustively_sampled_cross_section(axis):
    center = np.array([240., 160., 1290.])
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    first = np.cross(axis, np.eye(3)[np.argmin(np.abs(axis))]); first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    angles = np.linspace(0, 2*math.pi, 50000)
    circle = center + 25.5*(np.cos(angles)[:, None]*first + np.sin(angles)[:, None]*second)
    result = geometry(center, axis)
    assert result["minimum_surface_depth_z_mm"] == pytest.approx(float(circle[:, 2].min()), abs=1e-5)
    assert result["nearest_surface_range_mm"] == pytest.approx(float(np.linalg.norm(circle, axis=1).min()), abs=1e-5)
    assert result["centerline"]["range_mm"] > result["centerline"]["depth_z_mm"]
    for tangent in result["silhouette_tangent_points"]:
        point = np.array(tangent["point_camera_mm"])
        radial = point - center
        assert np.linalg.norm(radial) == pytest.approx(25.5)
        assert radial @ axis == pytest.approx(0, abs=1e-8)
        assert radial @ point == pytest.approx(0, abs=1e-8)


def test_axis_along_camera_has_no_unique_lateral_silhouette():
    result = geometry(axis=(0, 0, 1))
    assert result["silhouette_tangent_points"] == []
    assert result["minimum_surface_depth_z_mm"] == 1290
    assert result["nearest_surface_range_mm"] == pytest.approx(math.hypot(1290, 25.5))


@pytest.mark.parametrize("kwargs", [{"diameter": -1}, {"diameter": True}, {"diameter": float("nan")},
                                   {"center": (0, 0, 10)}, {"axis": (0, 0, 0)}])
def test_invalid_or_behind_camera_geometry_is_rejected(kwargs):
    with pytest.raises(ValueError):
        geometry(**kwargs)


def test_unknown_surface_median_never_becomes_axis_or_measured_diameter():
    spec = {"pipe_id": "BLUE", "nominal_diameter_mm": 51}
    record = pipe_measurement_record(spec, surface_samples={"left": {"depth_z_median_mm": 1290}})
    assert record["measured_section"] is None and record["measured_diameter_mm"] is None
    text = "\n".join(measurement_summary_lines(spec, {"measurement": record}))
    assert "未获得可靠实测" in text
    assert "不是中心线或顶点" in text
    assert "1.2900 m" in text
