from pipe_twin.workflow_status import workflow_summary
from pipe_twin.section_status import solver_status_rows
from test_temporal_sections import stage, check


def test_trial_application_is_separate_from_plane_and_position_acceptance():
    report = {'section_evaluation': stage(), 'ground_check': {
        'status': 'GROUND_PLANE_INCONSISTENT', 'correction_status': 'TRIAL_APPLIED'},
        'position_assessment': {'counts': {'IN_POSITION': 0, 'UNKNOWN': 12}}}
    text = workflow_summary(report, {}, loaded_pairs=1)
    assert '单帧' in text and '未做重复性复核' in text
    assert '平面未通过' in text and '试验已应用，未验证' in text
    assert 'CAD 已对应 0' in text and '待定 12' in text
    assert '观察侧：未指定' in text


def test_twenty_loaded_pairs_are_not_twenty_verified_pairs():
    text = workflow_summary(None, {}, loaded_pairs=20)
    assert '20 对，重复性待复核' in text
    rejected = {'section_evaluation': {'temporal_sections': {'status': 'BURST_REJECTED', 'requested_frames': 20}}}
    assert '20 对，多帧检查未通过' in workflow_summary(rejected, {})
    verified = {'section_evaluation': check([stage(), stage(), stage()])}
    assert '已复核（各管结果见下表）' in workflow_summary(verified, {})


def test_standalone_plane_check_does_not_claim_pipe_correction():
    text = workflow_summary(None, {'ground_region': {}}, loaded_pairs=1, ground_inspection={
        'status': 'CONSISTENT_GROUND_PLANE', 'surface_identity_source': 'OPERATOR_MARKED_GROUND'})
    assert '平面检查通过' in text and '未应用修正' in text


def test_single_frame_details_do_not_claim_a_burst():
    text = solver_status_rows({'section_evaluation': stage()})[0]['detail']
    assert '单帧断面结果' in text and '仅使用当前连拍' not in text


def test_unpaired_candidate_explains_missing_independent_boundaries():
    value = stage()
    value['candidates'] = []
    value['pairing_audit'] = {'left': [{'candidate_id': '0', 'stable_boundary_count': 1,
        'status': 'INSUFFICIENT_INDEPENDENT_SILHOUETTES', 'nearby_candidates': [
            {'candidate_id': 'R1', 'stable_boundary_count': 1, 'combined_stable_boundary_count': 2,
             'reason_codes': ['INSUFFICIENT_INDEPENDENT_SILHOUETTES', 'PROJECTED_WIDTH_MISMATCH']}]}]}
    row = solver_status_rows({'section_evaluation': value})[0]
    assert not row['solved']
    assert '稳定轮廓不足' in row['values'][-1]
    assert '合计 2 条' in row['detail'] and '至少需要 3 条' in row['detail']
    assert '左右可见宽度不一致' in row['detail']
