import numpy as np
import pytest

from pipe_twin.section_diagnostics import axis_from_image_lines, circle_trial, section_basis


def test_axis_from_independently_projected_parallel_edges():
    k = np.array([[800., 0, 320], [0, 810, 240], [0, 0, 1]])
    direction = np.array([1., -.12, .25])
    direction /= np.linalg.norm(direction)
    lines = []
    for camera_x in (0, 60):
        for y, z in ((-60, 900), (30, 850), (80, 1200)):
            a = np.array([-100-camera_x, y, z])
            b = a+200*direction
            lines.append(np.cross(k@a, k@b))
    result = axis_from_image_lines(lines, k)
    np.testing.assert_allclose(result['axis'], direction, atol=1e-10)
    assert max(result['line_residual_degrees']) < 1e-8


def test_parallel_image_planes_cannot_determine_axis():
    with pytest.raises(ValueError, match='UNOBSERVABLE'):
        axis_from_image_lines([[0, 1, 0]]*4, np.eye(3))


@pytest.mark.parametrize('radius', [13., 20.5, 25.5, 35.])
def test_section_recovers_known_circle_without_nominal_radius(radius):
    axis = np.array([1., -.12, .25]); axis /= np.linalg.norm(axis)
    basis = section_basis(axis)
    theta = np.linspace(-1.1, 1.1, 300)
    section = np.column_stack((radius*np.sin(theta), -radius*np.cos(theta)))
    axial = np.random.default_rng(8).uniform(-8, 8, len(theta))
    points = section @ basis.T + axial[:, None]*axis + [20, 80, 900]
    fit = circle_trial(points@basis)
    assert fit['geometry_passed']
    assert fit['trial_diameter_mm'] == pytest.approx(2*radius, abs=1e-5)
    assert fit['measured_diameter_mm'] is None


def test_noisy_circle_with_outliers_uses_original_denominator():
    rng = np.random.default_rng(3)
    theta = np.linspace(-1.2, 1.2, 400)
    points = np.column_stack((20*np.sin(theta), -20*np.cos(theta)))
    points += rng.normal(0, .12, points.shape)
    points[:35] = rng.uniform(-80, 80, (35, 2))
    fit = circle_trial(points)
    assert fit['geometry_passed']
    assert fit['inlier_fraction'] < .93
    assert fit['trial_diameter_mm'] == pytest.approx(40, abs=.3)


@pytest.mark.parametrize('kind', ['flat', 'small_arc', 'two_circles', 'random', 'few'])
def test_unsupported_sections_do_not_become_measurements(kind):
    theta = np.linspace(-1.1, 1.1, 200)
    points = np.column_stack((20*np.sin(theta), -20*np.cos(theta)))
    if kind == 'flat':
        points[:, 1] = 0
    elif kind == 'small_arc':
        theta *= .1
        points = np.column_stack((20*np.sin(theta), -20*np.cos(theta)))
    elif kind == 'two_circles':
        points = np.vstack((points, points+[120, 40]))
    elif kind == 'random':
        points = np.random.default_rng(15).uniform(-50, 50, (300, 2))
    else:
        points = points[:5]
    fit = circle_trial(points)
    assert not fit['geometry_passed']
    assert fit['measured_diameter_mm'] is None
