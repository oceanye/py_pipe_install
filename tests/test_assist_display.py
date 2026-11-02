import copy

from pipe_twin.assist_display import assist_rows, assist_summary_lines, assist_preview_states, hypothesis_assignments
from pipe_twin.relative_scene import model_states
from pipe_twin.workflow_status import workflow_summary


def report_fixture():
    return {'pipes': [{'pipe_id': f'P00{i}', 'installation_state': 'UNKNOWN'} for i in range(1, 5)],
        'installation_assist': {'advisory_only': True, 'accepted_measurement': False, 'status': 'AMBIGUOUS',
            'known_diameter_used_for_localization': True,
            'counts': {'LIKELY': 0, 'AMBIGUOUS': 4, 'NO_OBSERVATION': 0},
            'observations': [
                {'observation_id': 'a', 'label': 'A1', 'color_class': 'RED', 'relative_size_class': '中',
                 'image_width_px': 40., 'status': 'AMBIGUOUS', 'candidate_pipe_ids': ['P001', 'P003'],
                 'left_region_px': [100, 120, 280, 40], 'right_region_px': [90, 120, 280, 40]},
                {'observation_id': 'b', 'label': 'A2', 'color_class': 'BLUE', 'relative_size_class': '粗',
                 'status': 'AMBIGUOUS', 'candidate_pipe_ids': ['P002', 'P004'],
                 'left_region_px': [100, 220, 280, 60], 'right_region_px': [90, 220, 280, 60]}],
            'ranked_hypotheses': [
                {'rank': 1, 'difference_score': .2, 'assignments': [{'observation_id': 'a', 'pipe_id': 'P001'},
                                                                  {'observation_id': 'b', 'pipe_id': 'P002'}]},
                {'rank': 2, 'difference_score': .3, 'assignments': [{'observation_id': 'a', 'pipe_id': 'P003'},
                                                                  {'observation_id': 'b', 'pipe_id': 'P004'}]}]}}


def test_ambiguous_plan_does_not_mark_the_union_or_change_strict_states():
    report = report_fixture(); before = copy.deepcopy(report)
    strict = model_states(report)
    assert set(strict.values()) == {'UNKNOWN'}
    first = assist_preview_states(report, strict, 0)
    second = assist_preview_states(report, strict, 1)
    assert [p for p, s in first.items() if s == 'ASSIST_CANDIDATE'] == ['P001', 'P002']
    assert [p for p, s in second.items() if s == 'ASSIST_CANDIDATE'] == ['P003', 'P004']
    assert report == before and set(strict.values()) == {'UNKNOWN'}
    assert all(r['preview_pipe_id'] in ('P003', 'P004') for r in assist_rows(report, 1))


def test_qualitative_results_are_visible_without_a_metric_section_and_never_claim_probability():
    report = report_fixture()
    rows = assist_rows(report)
    assert len(rows) == 2 and rows[0]['values'][2] == 'P001'
    text = '\n'.join(assist_summary_lines(report))
    assert '采用已知管径辅助定位' in text and '不是本次实测' in text
    assert '不是概率' in text and '未观察到不等于未安装' in text
    assert '辅助识别' in workflow_summary(report, {})
    assert '精确尺寸见求解页' in workflow_summary(report, {})


def test_unique_advisory_never_overwrites_a_strict_state():
    report = report_fixture()
    report['installation_assist']['observations'][0].update(status='LIKELY', suggested_pipe_id='P001')
    assert assist_preview_states(report, {'P001': 'UNKNOWN'})['P001'] == 'SUSPECTED_INSTALLED'
    assert assist_preview_states(report, {'P001': 'OUT_OF_POSITION'})['P001'] == 'OUT_OF_POSITION'
    alternative = assist_preview_states(report, {'P001': 'UNKNOWN'}, 1)
    assert alternative['P001'] == 'UNKNOWN' and alternative['P003'] == 'ASSIST_CANDIDATE'
    report['installation_assist']['advisory_only'] = False
    assert assist_preview_states(report, {}) == {}


def test_malformed_nonunique_plan_cannot_light_model_pipes():
    report = report_fixture()
    plan = report['installation_assist']['ranked_hypotheses'][0]
    plan['assignments'][1]['pipe_id'] = 'P001'
    assert hypothesis_assignments(report) == {}
    assert assist_preview_states(report, {'P001': 'UNKNOWN'}) == {'P001': 'UNKNOWN'}


def test_coarse_tolerances_and_measured_size_remain_distinct_from_purchase_hint():
    report = report_fixture()
    data = report['installation_assist']
    data['policy'] = {'measured_diameter_tolerance_ratio': .10,
                      'relative_distance_tolerance_ratio': .10}
    data['observations'][0].update(estimated_diameter_mm=47.08, prior_diameter_mm=50.)
    before = copy.deepcopy(report)
    text = '\n'.join(assist_summary_lines(report))
    assert '估计管径 10%' in text and '相对距离 10%' in text
    assert '轮廓估计外径 47.08 mm' in text
    assert '采购外径提示 50 mm' in text
    assert set(assist_preview_states(report, model_states(report)).values()) == {
        'ASSIST_CANDIDATE', 'UNKNOWN'}
    assert report == before


def test_installed_set_consensus_lights_positions_without_claiming_observation_identity():
    report = report_fixture()
    data = report['installation_assist']
    data['ranked_hypotheses'][1]['assignments'] = [
        {'observation_id': 'a', 'pipe_id': 'P002'},
        {'observation_id': 'b', 'pipe_id': 'P001'}]
    data['pipes'] = [{'pipe_id': pid, 'status': 'LIKELY', 'presence_consensus': True,
                     'identity_basis': 'COMMON_INSTALLED_SET'} for pid in ('P001', 'P002')]
    before = copy.deepcopy(report)
    for index in (0, 1):
        states = assist_preview_states(report, model_states(report), index)
        assert states == {'P001': 'SUSPECTED_INSTALLED', 'P002': 'SUSPECTED_INSTALLED',
                          'P003': 'UNKNOWN', 'P004': 'UNKNOWN'}
        assert all(row['status'] == 'AMBIGUOUS' for row in assist_rows(report, index))
    assert '逐根对应仍可能互换' in '\n'.join(assist_summary_lines(report))
    assert report == before


def test_diagnostic_plan_outside_recognition_conditions_cannot_light_positions():
    report = report_fixture()
    data = report['installation_assist']
    data['ranked_hypotheses'][1]['near_best'] = False
    data['pipes'] = [{'pipe_id': 'P003', 'status': 'LIKELY', 'presence_consensus': True}]
    strict = model_states(report)
    assert assist_preview_states(report, strict, 1) == strict
    assert '不点亮安装管位' in '\n'.join(assist_summary_lines(report, 1))
    assert all(row['values'][-1] == '未满足识别条件，仅供诊断' for row in assist_rows(report, 1))
