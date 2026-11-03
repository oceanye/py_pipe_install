import copy

from pipe_twin.monitor_history import MonitorHistory, history_rows


def report(pipes, *, status='LIKELY', offset=0):
    return {'installation_assist': {'advisory_only': True, 'status': status,
            'observations': [{'observation_id': str(i + offset), 'status': status} for i, _ in enumerate(pipes)],
            'ranked_hypotheses': [{'near_best': True, 'assignments': [
                {'observation_id': str(i + offset), 'pipe_id': pipe} for i, pipe in enumerate(pipes)]}]}}


def ledger():
    return MonitorHistory([{'pipe_id': p} for p in ('P001', 'P002', 'P003')])


def test_first_result_is_baseline_then_pipe_ids_not_observation_ids_are_compared():
    history = ledger()
    first = report(['P001', 'P002'])
    original = copy.deepcopy(first)
    baseline = history.record(1, report=first, captured_at='2026-11-03T11:00:00+08:00')
    assert baseline['kind'] == 'BASELINE' and baseline['added'] == baseline['removed'] == []
    assert history_rows(baseline) == [('基线', 'P001'), ('基线', 'P002')]
    unchanged = history.record(2, report=report(['P002', 'P001'], offset=10))
    assert unchanged['kind'] == 'UNCHANGED'
    changed = history.record(3, report=report(['P002', 'P003']))
    assert changed['added'] == ['P003'] and changed['removed'] == ['P001']
    assert history_rows(changed) == [('新增提示', 'P003'), ('减少提示', 'P001')]
    assert first == original


def test_failure_and_insufficient_frames_never_remove_pipes_or_replace_baseline():
    history = ledger()
    history.record(1, report=report(['P001', 'P002']))
    error = history.record(2, error='server disconnected')
    missing = history.record(3, report=report([], status='INSUFFICIENT'))
    assert error['kind'] == 'ERROR' and missing['kind'] == 'UNRESOLVED'
    assert error['removed'] == missing['removed'] == []
    assert history.record(4, report=report(['P001', 'P002']))['kind'] == 'UNCHANGED'


def test_first_usable_result_after_failure_still_establishes_baseline_and_duplicate_is_ignored():
    history = ledger()
    history.record(1, error='failed')
    assert history.record(1, report=report(['P001'])) is None
    assert history.record(2, report=report(['P001']))['kind'] == 'BASELINE'
    assert history.record(2, report=report(['P002'])) is None
    assert history.record(3, report=report(['P001']))['kind'] == 'UNCHANGED'


def test_new_monitoring_session_does_not_compare_different_model_settings():
    first, second = ledger(), ledger()
    first.record(1, report=report(['P001']))
    baseline = second.record(1, report=report(['P003']))
    assert baseline['kind'] == 'BASELINE' and not baseline['removed']
    assert first.session_id != second.session_id


def test_out_of_order_old_cycle_cannot_change_the_comparison_baseline():
    history = ledger()
    history.record(3, report=report(['P003']))
    assert history.record(2, report=report(['P001'])) is None
    assert history.record(4, report=report(['P003']))['kind'] == 'UNCHANGED'


def test_ambiguous_displayed_hints_are_advisory_and_diagnostic_only_choices_are_skipped():
    history = ledger()
    ambiguous = report(['P001'], status='AMBIGUOUS')
    assert history.record(1, report=ambiguous)['advisory_only']
    ambiguous['installation_assist']['ranked_hypotheses'][0]['near_best'] = False
    assert history.record(2, report=ambiguous)['kind'] == 'UNRESOLVED'
    assert history.record(3, report=report(['UNKNOWN-ID']))['kind'] == 'UNRESOLVED'
    assert history.record(4, report=report(['P001']))['kind'] == 'UNCHANGED'
