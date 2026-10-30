import copy
import math

import numpy as np
import pytest

from pipe_twin.view_geometry import section_view_geometry, report_view_geometry, view_geometry_summary_lines


@pytest.mark.parametrize('angle', [0, 5, 30, 60, 90])
def test_effective_baseline_uses_axis_normal_component_and_does_not_change_physical_baseline(angle):
    axis = [math.cos(math.radians(angle)), math.sin(math.radians(angle)), 0.]
    result = section_view_geometry(axis, 60.)
    assert result['physical_baseline_mm'] == 60.
    assert result['baseline_perpendicular_to_axis_mm'] == pytest.approx(60*math.sin(math.radians(angle)))
    assert result['baseline_axis_acute_angle_deg'] == pytest.approx(angle)
    assert not result['accepted_measurement']
    reversed_axis = section_view_geometry(-np.array(axis)*3, 60.)
    assert reversed_axis['baseline_perpendicular_to_axis_mm'] == pytest.approx(result['baseline_perpendicular_to_axis_mm'])


def test_latest_field_geometry_is_specific_to_section_method():
    value = section_view_geometry([.995467678698662, -.087032952265368, -.038332308647819], 59.82261029553183)
    assert value['baseline_perpendicular_to_axis_mm'] == pytest.approx(5.689158755829515)
    assert value['scope'] == 'SILHOUETTE_SECTION_GEOMETRY_ONLY'


def test_geometry_does_not_normalize_the_callers_array_in_place():
    axis = np.array([2., 3., 4.])
    before = axis.copy()
    section_view_geometry(axis, 60.)
    np.testing.assert_array_equal(axis, before)


@pytest.mark.parametrize('axis,baseline', [([0, 0, 0], 60), ([1, 0], 60), ([1, 0, float('nan')], 60),
                                           ([1, 0, 0], None), ([1, 0, 0], -1)])
def test_unresolved_geometry_never_fabricates_a_value(axis, baseline):
    result = section_view_geometry(axis, baseline)
    assert result['status'] == 'GEOMETRY_UNRESOLVED'
    assert 'baseline_perpendicular_to_axis_mm' not in result


def report():
    return {'generated_at': '2026-10-30T10:35:00+00:00',
            'calibration_audit': {'calibration_id': 'bound-report', 'quality_audit': {
                'rectification_geometry': {'horizontal_encoded_baseline_mm': 60.}}},
            'section_evaluation': {'axis': {'status': 'STABLE', 'axis': [0, 1, 0]},
                                   'ground_correction': {'trial_mode': True}, 'candidates': []}}


def test_report_uses_own_calibration_and_does_not_mutate_or_accept_it():
    source = report(); before = copy.deepcopy(source)
    value = report_view_geometry(source)[0]
    assert value['physical_baseline_mm'] == 60. and value['calibration_id'] == 'bound-report'
    text = '\n'.join(view_geometry_summary_lines(source))
    assert '不是实时角度' in text and '100.0%' in text and '条件诊断' in text
    assert source == before and not value['accepted_measurement']
    del source['calibration_audit']
    assert report_view_geometry(source)[0]['status'] == 'GEOMETRY_UNRESOLVED'


def test_new_fit_baseline_is_self_contained_and_conflicts_are_not_averaged():
    source = report(); del source['calibration_audit']
    source['section_evaluation']['candidates'] = [{'fit': {'physical_baseline_mm': 59.8}}]
    assert report_view_geometry(source)[0]['physical_baseline_mm'] == 59.8
    source['section_evaluation']['candidates'].append({'fit': {'physical_baseline_mm': 61.}})
    assert report_view_geometry(source)[0]['status'] == 'REPORT_BASELINE_CONFLICT'


def test_ground_boundary_failure_remains_visible_even_when_trial_produces_an_axis():
    source = report()
    source['section_evaluation']['ground_correction']['boundary_check_passed'] = False
    text = '\n'.join(view_geometry_summary_lines(source))
    assert '管轴轮廓检查未通过' in text and '60.00 mm' in text


def test_no_report_displays_next_action_and_no_old_measurement():
    assert report_view_geometry(None) == []
    assert '请先抓拍并评估' in '\n'.join(view_geometry_summary_lines(None))
