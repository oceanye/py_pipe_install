"""Stable display geometry; these tests do not assert measurement accuracy."""
import numpy as np

from pipe_twin.elevation_viewer import model_section_basis


def test_auto_section_is_stable_across_pipe_order_and_endpoint_signs():
    axis = np.array([1., 2., 3.])
    axis /= np.linalg.norm(axis)
    centers = np.array([[0, 0, 0], [100, 0, 0], [0, 50, 0]])
    lines = centers[:, None, :] + np.array([-100., 100.])[None, :, None] * axis
    original = model_section_basis(lines)
    reversed_lines = lines[::-1].copy()
    reversed_lines[1] = reversed_lines[1, ::-1]
    assert np.allclose(original, model_section_basis(reversed_lines))
    assert np.allclose(original @ axis, [0, 0, 1])
    assert np.allclose(original @ original.T, np.eye(3))
    # Changing stations along a pipe must not change its cross-section point.
    shifted = centers + np.array([70., -90., 30.])[:, None]*axis
    assert np.allclose((centers @ original.T)[:, :2], (shifted @ original.T)[:, :2])


def test_reversing_directed_axis_mirrors_section_with_explicit_convention():
    lines = np.array([[[0, 0, -10], [0, 0, 10]], [[100, 20, -10], [100, 20, 10]]])
    forward = model_section_basis(lines, np.array([0., 0., 1.]))
    reverse = model_section_basis(lines, np.array([0., 0., -1.]))
    delta = lines[1].mean(axis=0)-lines[0].mean(axis=0)
    assert np.allclose(forward @ delta, [100, 20, 0])
    assert np.allclose(reverse @ delta, [-100, 20, 0])
