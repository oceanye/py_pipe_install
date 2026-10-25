import copy
import json

import numpy as np
import pytest

from pipe_twin.temporal_sections import associate, check_sections, sample_indices
from pipe_twin.section_status import solver_status_rows


def stage(diameters=(40., 50.), shift=0., prefix=''):
    candidates, models, views = [], [], []
    for i, diameter in enumerate(diameters):
        identity = prefix+str(i)
        box = [100, 40+i*60, 400, 30]
        candidates.append({'candidate_id': identity, 'left_candidate_id': identity, 'right_candidate_id': identity,
            'left_region_px': box, 'right_region_px': box, 'boundary_status': 'STABLE', 'reason_codes': [],
            'fit': {'status': 'CONDITIONAL_FIT', 'trial_diameter_mm': diameter}, 'included_in_section_model': True,
            'window_sensitivity': {'stable': True}})
        models.append({'id': identity, 'estimated_center_uv_mm': [i*100+shift, shift], 'diameter_mm': diameter})
        views.append({'candidate_id': identity, 'appearance': 'hue-0', 'color_srgb': '#AA2222',
                      'aggregation': {'stable_boundary_count': 2, 'boundaries': []}})
    return {'candidates': candidates, 'model': {'candidates': models}, 'views': {'left': views, 'right': views}}


def check(stages):
    return check_sections(stages[-1], stages, [{'capture_id': str(i), 'sha256': {}} for i in range(len(stages))], requested_frames=len(stages))


def test_frame_ids_are_not_identity_and_translation_is_not_relative_motion():
    stages = [stage(shift=i*500, prefix=str(i)+'-') for i in range(5)]
    before = copy.deepcopy(stages[-1]['model'])
    output = check(stages)
    for row in output['candidates']:
        audit = row['temporal_consistency']
        assert audit['solved_frames'] == 5
        assert audit['diameter_repeatable'] and audit['layout_repeatable']
        assert audit['accuracy_validated'] is False
    assert output['model'] == before
    assert output['temporal_sections']['pairwise'][0]['distance']['span_mm'] == 0
    json.dumps(output, allow_nan=False)


def test_bimodal_radius_is_reported_not_averaged_or_called_accurate():
    stages = [stage((40., diameter)) for diameter in (53, 52, 52, 43, 43)]
    output = check(stages)
    row = output['candidates'][1]
    assert row['fit']['trial_diameter_mm'] == 43
    assert row['temporal_consistency']['diameter']['span_mm'] == 10
    assert 'TEMPORAL_SECTION_DIAMETER_UNSTABLE' in row['reason_codes']
    assert not output['candidates'][0]['reason_codes']
    rows = solver_status_rows({'section_evaluation': output})
    assert rows[1]['solved'] and rows[1]['pipe_id'] is None
    assert '43.0～53.0' in rows[1]['values'][3]
    assert '重复性待改善' in rows[1]['values'][-1]


def test_ambiguous_duplicates_are_not_counted_and_historical_pipes_not_resurrected():
    old, current = stage(), stage((40.,))
    duplicated = stage()
    for role in ('left', 'right'):
        duplicated['candidates'][1][role+'_region_px'] = duplicated['candidates'][0][role+'_region_px']
    assert not associate(current, duplicated)
    result = check([old, duplicated, current])
    assert len(result['candidates']) == 1
    assert result['candidates'][0]['temporal_consistency']['solved_frames'] == 2
    assert 'TEMPORAL_SECTION_SUPPORT_INSUFFICIENT' in result['candidates'][0]['reason_codes']


def test_pair_distance_changes_are_not_hidden_by_individually_stable_diameters():
    stages = [stage() for _ in range(3)]
    stages[0]['model']['candidates'][1]['estimated_center_uv_mm'][0] += 10
    result = check(stages)
    assert all(c['temporal_consistency']['diameter_repeatable'] for c in result['candidates'])
    assert all(not c['temporal_consistency']['layout_repeatable'] for c in result['candidates'])


def test_sampling_is_bounded_and_retains_current_reference():
    assert sample_indices(0) == []
    assert sample_indices(5) == list(range(5))
    selected = sample_indices(24)
    assert len(set(selected)) == 7 and selected[0] == 0 and selected[-1] == 23
    with pytest.raises(ValueError):
        check_sections(stage(), [stage(), stage(), stage()], [{}, {}, {}], requested_frames=3)


def test_status_overlay_restores_crop_and_inverse_warp_without_assigning_cad():
    value = stage()
    value['zone_offsets'] = {'left_x': 50, 'left_y': 10, 'right_x': 20, 'right_y': 10}
    value['epipolar_refinement'] = {'applied': True, 'right_refined_to_original': [[1,0,2], [0,1,-3], [0,0,1]]}
    value['registration'] = {'status': 'AMBIGUOUS', 'matches': [{'observation_id': '0', 'pipe_id': 'P1'}]}
    rows = solver_status_rows({'section_evaluation': value})
    assert rows[0]['left_region_px'] == [150, 50, 400, 30]
    assert rows[0]['right_region_px'] == [122, 47, 400, 30]
    assert rows[0]['pipe_id'] is None
    assert rows[0]['values'][3] == '未复核'


def test_useful_unpaired_image_evidence_remains_visible_but_not_solved():
    value = stage()
    value['candidates'] = value['candidates'][:1]
    value['model']['candidates'] = value['model']['candidates'][:1]
    rows = solver_status_rows({'section_evaluation': value})
    assert len(rows) == 2 and not rows[1]['solved'] and rows[1]['diameter_mm'] is None
    assert rows[1]['values'][-1] == '断面未解'


def test_production_groups_check_only_current_burst_and_reject_wrong_reference():
    from test_elevation_auto import _groups
    from pipe_twin.elevation_auto import analyze_elevation_auto_groups
    cal, specs, groups = _groups(3)
    latest = groups[-1]
    latest['depth'].audit['temporal_fusion'] = {'status': 'VALID', 'reference_capture_id': latest['capture_id'], 'requested_frames': 3}
    latest['section_frames'] = [dict(left=g['left'], right=g['right'], capture_id=g['capture_id'], sha256={}) for g in groups]
    result = analyze_elevation_auto_groups([latest], calibration=cal, pipe_specs=specs)
    audit = result['section_evaluation']['temporal_sections']
    assert audit['status'] == 'CHECKED' and audit['sampled_frames'] == 3
    assert not audit['historical_frames_used']
    latest['section_frames'][-1]['left'] = latest['left'].copy()
    latest['section_frames'][-1]['left'][0, 0] += 1
    with pytest.raises(ValueError, match='REFERENCE_IMAGE_MISMATCH'):
        analyze_elevation_auto_groups([latest], calibration=cal, pipe_specs=specs)
    latest['section_frames'] = latest['section_frames'][:-1]
    result = analyze_elevation_auto_groups([latest], calibration=cal, pipe_specs=specs)
    assert result['section_evaluation']['temporal_sections']['status'] == 'BURST_REJECTED'
