import copy

import numpy as np
import pytest

from pipe_twin.axial_aggregation import aggregate_boundaries
from pipe_twin.section_display import boundary_overlay_segments, section_summary_lines


def gapped_report():
    columns = np.arange(384, dtype=float)
    columns = columns[((columns % 128) < 45) | ((columns % 128) >= 75)]
    audit, _ = aggregate_boundaries([
        {'role': 'left', 'edge': 'top', 'points': np.c_[columns, 100+.02*columns]}])
    return {'section_evaluation': {'views': {'left': [
        {'candidate_id': 'POOL-L01', 'boundary_stable': True, 'aggregation': audit}]}}}


def test_stable_overlay_does_not_bridge_missing_columns_inside_accepted_windows():
    report = gapped_report()
    edge = report['section_evaluation']['views']['left'][0]['aggregation']['boundaries'][0]
    assert edge['status'] == 'STABLE' and all(w['accepted'] for w in edge['windows'])
    segments = boundary_overlay_segments(report, 'left')
    assert [(a, c) for a, _, c, _ in segments] == [(0, 44), (75, 172), (203, 300), (331, 383)]
    for start, y1, end, y2 in segments:
        assert y1 == pytest.approx(100+.02*start)
        assert y2 == pytest.approx(100+.02*end)


def test_old_stable_edge_without_column_support_is_not_invented_from_window_extents():
    report = gapped_report()
    edge = report['section_evaluation']['views']['left'][0]['aggregation']['boundaries'][0]
    del edge['retained_column_runs_px']
    assert edge['windows'] and edge['observed_axial_interval_px']
    assert boundary_overlay_segments(report, 'left') == []
    assert '旧记录缺少逐列观测范围' in '\n'.join(section_summary_lines(report))


def test_rejected_edges_remain_undrawn_even_with_observed_runs():
    report = gapped_report()
    edge = report['section_evaluation']['views']['left'][0]['aggregation']['boundaries'][0]
    edge['status'] = 'UNSTABLE'
    assert boundary_overlay_segments(report, 'left') == []


def test_gaps_survive_right_refinement_and_zone_offsets():
    report = gapped_report()
    stage = report['section_evaluation']
    stage['views']['right'] = copy.deepcopy(stage['views']['left'])
    stage['zone_offsets'] = {'right_x': 100, 'right_y': 30}
    stage['epipolar_refinement'] = {'applied': True,
        'right_refined_to_original': [[1, 0, 2], [0, 1, -3], [0, 0, 1]]}
    segments = boundary_overlay_segments(report, 'right')
    assert [(a, c) for a, _, c, _ in segments] == [(102, 146), (177, 274), (305, 402), (433, 485)]
    assert segments[0][1] == pytest.approx(127)
