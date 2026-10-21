import json

import numpy as np
import pytest

from pipe_twin.projection_diagnostics import fit_silhouettes, pixel_error_scenarios
from pipe_twin.section_diagnostics import section_basis
from pipe_twin.section_model import build_section_model, fit_surface_arc, write_section_artifacts
from test_projection_diagnostics import independent_contours


def scene():
    axis = np.array([1., -.4, .2]); axis /= np.linalg.norm(axis)
    # Independent circular surfaces, rendered by OpenCV in the helper.
    centers = [np.array([100., 80., 1800.]), np.array([140., 160., 1870.])]
    items = []
    for index, (center, radius) in enumerate(zip(centers, [25., 20.])):
        obs, k = independent_contours(axis, center, radius)
        fit = fit_silhouettes(obs, k, 60., axis, 960.)
        items.append({'id': str(index), 'label': f'pipe {index}', 'color': '#333',
                      'reference_diameter_mm': 999., 'fit': fit,
                      'pixel_error': pixel_error_scenarios(obs, k, 60., axis, 960.)})
    return axis, centers, items


def test_common_plane_dimensions_do_not_depend_on_pipe_lengths_or_section_station():
    axis, centers, items = scene()
    a = build_section_model(items, axis, origin_id='0', station_mm=100.)
    b = build_section_model(items, axis, origin_id='1', station_mm=1300.)
    delta = centers[1]-centers[0]
    expected = np.linalg.norm(delta-axis*(delta@axis))
    assert a['pairs'][0]['center_distance_mm'] == pytest.approx(expected, abs=.02)
    assert a['pairs'][0]['surface_clearance_mm'] == pytest.approx(expected-45., abs=.02)
    assert b['pairs'][0]['center_distance_mm'] == pytest.approx(a['pairs'][0]['center_distance_mm'])
    for ca, cb in zip(a['candidates'], b['candidates']):
        np.testing.assert_allclose(np.array(cb['center_in_common_plane_camera_mm'])-ca['center_in_common_plane_camera_mm'],
                                   1200*axis, atol=1e-8)
    assert a['candidates'][0]['diameter_mm'] == pytest.approx(50., abs=.02)  # Not reference 999.
    assert a['candidates'][0]['reference_difference_mm'] == pytest.approx(-949., abs=.02)
    assert a['pairs'][0]['accepted_measurement'] is False


def test_missing_or_occluded_candidate_never_gains_a_reference_radius_or_fake_pair():
    axis, _, items = scene()
    items.append({'id': 'hidden', 'label': 'hidden', 'reference_diameter_mm': 40.,
                  'fit': {'status': 'INSUFFICIENT_INDEPENDENT_SILHOUETTES'}})
    model = build_section_model(items, axis, origin_id='0')
    assert len(model['pairs']) == 1
    hidden = model['candidates'][-1]
    assert hidden['radius_mm'] is None
    assert hidden['estimated_center_uv_mm'] is None


def test_opposite_axis_representation_and_covariance_are_transformed_together():
    axis, _, items = scene()
    before = build_section_model(items, axis, origin_id='0')
    fit = items[1]['fit']
    rotation = section_basis(-axis).T@section_basis(axis)
    fit['axis'] = (-axis).tolist()
    fit['section_center_parameters_mm'] = (rotation@fit['section_center_parameters_mm']).tolist()
    transform = np.eye(3); transform[:2, :2] = rotation
    for scenario in items[1]['pixel_error']['scenarios'].values():
        scenario['section_parameter_covariance_mm2'] = (transform@scenario['section_parameter_covariance_mm2']@transform.T).tolist()
    after = build_section_model(items, axis, origin_id='0')
    assert after['pairs'][0]['center_distance_mm'] == pytest.approx(before['pairs'][0]['center_distance_mm'])
    for key in before['pairs'][0]['pixel_error']['scenarios']:
        assert after['pairs'][0]['pixel_error']['scenarios'][key] == pytest.approx(before['pairs'][0]['pixel_error']['scenarios'][key])


def test_different_axes_rejected_and_intersecting_sections_not_clamped():
    axis, _, items = scene()
    items[1]['fit']['axis'] = [1., 0., 0.]
    with pytest.raises(ValueError, match='COMMON_AXIS'):
        build_section_model(items, axis)
    items[1]['fit']['axis'] = axis.tolist()
    items[1]['fit']['section_center_parameters_mm'] = (np.array(items[0]['fit']['section_center_parameters_mm'])+[10., 0.]).tolist()
    model = build_section_model(items, axis)
    assert model['pairs'][0]['surface_clearance_mm'] == pytest.approx(-35., abs=.02)
    assert model['pairs'][0]['status'] == 'OVERLAPPING_SECTION_INCONSISTENCY'


def test_section_exports_keep_units_and_provenance(tmp_path):
    import ezdxf
    axis, _, items = scene()
    model = build_section_model(items, axis, origin_id='0')
    write_section_artifacts(model, tmp_path)
    saved = json.loads((tmp_path/'section-model.json').read_text(encoding='utf-8'))
    assert saved['units'] == 'mm'
    assert saved['candidates'][0]['diameter_source'] == 'IMAGE_FIT'
    assert '<svg' in (tmp_path/'section.svg').read_text(encoding='utf-8')
    assert 'surface_clearance_mm' in (tmp_path/'pair-dimensions.csv').read_text(encoding='utf-8-sig')
    assert '暂用标定' in (tmp_path/'index.html').read_text(encoding='utf-8')
    assert '采购外径仅作参照' in (tmp_path/'index.html').read_text(encoding='utf-8')
    drawing = ezdxf.readfile(tmp_path/'section.dxf')
    assert drawing.units == 4
    assert sorted(c.dxf.radius for c in drawing.modelspace().query('CIRCLE')) == pytest.approx([20., 25.], abs=.02)


@pytest.mark.parametrize('span,expected', [(90., 'CONDITIONAL_FIT'), (25., 'INSUFFICIENT_SURFACE_ARC')])
def test_partial_arc_aggregates_unknown_axial_lengths_without_filling_occlusion(span, expected):
    axis = np.array([1., -.4, .2]); axis /= np.linalg.norm(axis)
    # Construct points using an independent normal-plane basis.
    u = np.cross(axis, [0, 1, 0]); u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    rng = np.random.default_rng(2801)
    theta = np.linspace(.2, .2+np.radians(span), 300)
    points = (np.array([100., 80., 1800.]) + rng.uniform(-100, 100, (300, 1))*axis
              + 25*(np.cos(theta)[:, None]*u+np.sin(theta)[:, None]*v)
              + rng.normal(0, .02, (300, 3)))
    result = fit_surface_arc(points, axis)
    assert result['status'] == expected
    assert result['arc_diagnostic']['point_count'] == 300
    assert result['accepted_diameter_mm'] is None
    if expected == 'CONDITIONAL_FIT':
        assert result['trial_diameter_mm'] == pytest.approx(50., abs=.15)
        assert result['arc_diagnostic']['arc_degrees'] == pytest.approx(90., abs=1.)
    else:
        assert 'trial_diameter_mm' not in result
