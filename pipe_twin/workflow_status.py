"""Compact, read-only workflow feedback; never promotes an estimate to acceptance."""
from .camera_view import camera_side_label
from .section_display import section_evaluations
from .section_status import solver_status_rows


def workflow_summary(report, registration, *, loaded_pairs=0, ground_inspection=None):
    registration = registration or {}
    stages = section_evaluations(report)
    fusion = (report or {}).get('temporal_fusion') or {}
    checks = [s.get('temporal_sections') or {} for s in stages]
    count = max([int(fusion.get('requested_frames') or 0),
                 *[int(c.get('requested_frames') or 0) for c in checks], loaded_pairs])
    if not count:
        sampling = '照片：待抓拍'
    elif count == 1:
        sampling = '照片：单帧，未做重复性复核'
    elif fusion.get('status') not in (None, 'VALID') or any(c.get('status') == 'BURST_REJECTED' for c in checks):
        sampling = f'照片：{count} 对，多帧检查未通过'
    elif checks and all(c.get('status') == 'CHECKED' for c in checks):
        sampling = f'照片：{count} 对，已复核（各管结果见下表）'
    else:
        sampling = f'照片：{count} 对，重复性待复核'
    side = registration.get('camera_side_world')
    viewing = '观察侧：未指定' if side is None else '观察侧：'+camera_side_label(side)
    if report:
        rows = solver_status_rows(report)
        solved = sum(bool(r['solved']) for r in rows)
        mapped = len({r['pipe_id'] for r in rows if r.get('pipe_id')})
        sections = f'条件断面 {solved}/{len(rows)} · CAD 已对应 {mapped}' if rows else '断面：未解出'
    else:
        sections = '断面与 CAD：待评估'
    ground = (report or {}).get('ground_check') or ground_inspection or {}
    status = ground.get('status', 'NOT_CONFIGURED')
    if status == 'CONSISTENT_GROUND_PLANE' and ground.get('surface_identity_source') == 'OPERATOR_MARKED_GROUND':
        plane = '平面检查通过'
    elif status == 'NOT_CONFIGURED':
        plane = '待检查' if registration.get('ground_region') else '未设置'
    else:
        plane = '平面未通过或未确认'
    correction = ground.get('correction_status')
    if correction in ('TRIAL_APPLIED', 'TRIAL_PARTIALLY_APPLIED'):
        applied = '试验已应用，未验证'
    elif correction == 'TRIAL_NOT_APPLIED':
        applied = '试验未应用'
    elif correction in ('APPLIED', 'PARTIALLY_APPLIED'):
        applied = '已用于轴向修正' if correction == 'APPLIED' else '部分轴向已修正'
    elif registration.get('ground_candidate_trial'):
        applied = '试验开启，待评估'
    else:
        applied = '未应用修正'
    counts = ((report or {}).get('position_assessment') or {}).get('counts')
    position = (f"相对就位：符合 {counts.get('IN_POSITION', 0)} / 不符 {counts.get('OUT_OF_POSITION', 0)} / 待定 {counts.get('UNKNOWN', 0)}"
                if counts else '相对就位：待评估')
    return f'{sampling} · {viewing} · {sections}\n地面：{plane} / {applied} · {position}'
