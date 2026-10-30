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
