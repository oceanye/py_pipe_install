import copy

import pytest

from pipe_twin.ai_section_assist import build_ai_section_assist
from pipe_twin.ai_section_plot import ai_section_plot
from pipe_twin.model_section_reference import build_model_section_reference


def _pipes():
    return [{'pipe_id': f'P{i:03}', 'nominal_diameter_mm': 26 + 10 * i,
             'centerline_world_mm': [[i * 100, 0, 0], [i * 100, 0, 500]]}
            for i in range(1, 5)]


def _report():
    point_map = build_model_section_reference(_pipes(), axis_world=[0, 0, 1])
    states = ['SUPPORTED_VISIBLE', 'AMBIGUOUS', 'UNRESOLVED', 'RECAPTURE_RECOMMENDED']
    result = build_ai_section_assist(point_map, {'point_assessments': [
        {'point_id': point['point_id'], 'section_status': state, 'evidence_quality': 'HIGH',
         'reason_codes': ['observed evidence'], 'observation_refs': ['scene_left']}
        for point, state in zip(point_map['points'], states)]})
    return {'ai_section_assist': result, 'pipes': [{'pipe_id': 'P002', 'installation_state': 'INSTALLED'}],
            'installation_assist': {'hypotheses': [{'pipe_ids': ['P003']}]}}


def test_preserves_valid_point_states_and_never_returns_geometry_or_installation_claims():
    report = _report()
    before = copy.deepcopy(report)
    pipes = _pipes()
    view = ai_section_plot(report, pipes)
    assert view['available'] is True
    for candidate in report['ai_section_assist']['candidates']:
        row = view['by_pipe'][candidate['pipe_id']]
        assert row['status'] == candidate['section_status']
        assert row['point_id'] == candidate['point_id']
        assert row['reason'] == 'observed evidence'
        assert not {'center_uv_mm', 'nominal_diameter_mm', 'estimated_center_uv_mm', 'installation_state'} & row.keys()
    assert report == before
    assert pipes == _pipes()
    assert all(value == 1 for value in view['counts'].values())
    assert len({row['dash'] for row in view['by_pipe'].values()}) == 4
    assert all(isinstance(view[key], str) for key in ('legend', 'summary', 'caption'))
    assert 'CAD' in view['caption']


@pytest.mark.parametrize('run_status,decision,expected', [
    ('NOT_RUN', 'SUPPORTED_VISIBLE', 'NOT_RUN'),
    ('REJECTED', 'UNRESOLVED', 'REJECTED'),
    ('VALID', 'NOT_RUN', 'NOT_RUN'),
    ('VALID', 'STALE', 'STALE'),
    ('VALID', 'REJECTED', 'REJECTED'),
    ('INVALID', 'SUPPORTED_VISIBLE', 'REJECTED'),
])
def test_failed_or_not_run_envelopes_suppress_leftover_supported_rows(run_status, decision, expected):
    report = _report()
    report['ai_section_assist'].update(run_status=run_status, status=decision)
    view = ai_section_plot(report, _pipes())
    assert not view['available']
    assert {row['status'] for row in view['by_pipe'].values()} == {expected}
    assert not any(view['counts'].values())


def test_stale_current_scene_and_missing_ai_never_use_local_installed_states():
    report = _report()
    stale = ai_section_plot(report, _pipes(), current=False)
    assert stale['status'] == 'STALE'
    assert {row['status'] for row in stale['by_pipe'].values()} == {'STALE'}
    del report['ai_section_assist']
    absent = ai_section_plot(report, _pipes())
    assert absent['status'] == 'NOT_RUN'
    assert {row['status'] for row in absent['by_pipe'].values()} == {'NOT_RUN'}


def test_partial_results_keep_supported_points_but_missing_points_stay_unresolved():
    report = _report()
    data = report['ai_section_assist']
    data['run_status'] = 'PARTIAL'
    supported = next(row for row in data['candidates'] if row['section_status'] == 'SUPPORTED_VISIBLE')
    data['candidates'] = [supported]
    view = ai_section_plot(report, _pipes())
    assert view['by_pipe'][supported['pipe_id']]['status'] == 'SUPPORTED_VISIBLE'
    assert view['counts']['SUPPORTED_VISIBLE'] == 1
    assert view['counts']['UNRESOLVED'] == 3


@pytest.mark.parametrize('duplicate', ['pipe', 'point'])
def test_duplicate_pipe_or_point_cannot_light_supported_position(duplicate):
    report = _report()
    rows = report['ai_section_assist']['candidates']
    supported = next(row for row in rows if row['section_status'] == 'SUPPORTED_VISIBLE')
    if duplicate == 'pipe':
        rows.append(copy.deepcopy(supported))
    else:
        rows[1]['point_id'] = supported['point_id']
    view = ai_section_plot(report, _pipes())
    assert view['by_pipe'][supported['pipe_id']]['status'] == 'UNRESOLVED'
    assert view['counts']['SUPPORTED_VISIBLE'] == 0


def test_unknown_pipe_is_not_drawn_and_does_not_generate_cad_geometry():
    report = _report()
    report['ai_section_assist']['candidates'].append(
        {'pipe_id': 'FOREIGN_MODEL_PIPE', 'point_id': 'P99', 'status': 'SUPPORTED_VISIBLE'})
    view = ai_section_plot(report, _pipes())
    assert set(view['by_pipe']) == {pipe['pipe_id'] for pipe in _pipes()}


def test_empty_canonical_rows_do_not_resurrect_legacy_aliases():
    report = _report()
    report['ai_section_assist']['candidates'] = []
    view = ai_section_plot(report, _pipes())
    assert view['counts']['UNRESOLVED'] == 4
    assert view['counts']['SUPPORTED_VISIBLE'] == 0


@pytest.mark.parametrize('bad_rows', ['wrong', {'pipe_id': 'P001'}, 10])
def test_invalid_candidate_container_is_rejected(bad_rows):
    report = _report()
    report['ai_section_assist']['candidates'] = bad_rows
    view = ai_section_plot(report, _pipes())
    assert view['status'] == 'REJECTED'
    assert not view['available']


def test_legacy_section_scope_is_supported_but_installation_scope_is_not():
    report = {'ai_section_assist': {'scope': 'SECTION_ONLY', 'status': 'CANDIDATES_READY',
              'candidates': [{'pipe_id': 'P001', 'point_id': 'P01', 'status': 'SUPPORTED_VISIBLE'}]}}
    assert ai_section_plot(report, _pipes())['counts']['SUPPORTED_VISIBLE'] == 1
    report['ai_section_assist']['scope'] = 'INSTALLATION'
    assert ai_section_plot(report, _pipes())['status'] == 'NOT_RUN'
