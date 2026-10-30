"""Present measured common sections independently of CAD registration.

Coordinates are relative to the first solved section in each zone. CAD sizes
never replace fitted sizes, and different zones are never joined implicitly.
"""
from __future__ import annotations

import numpy as np

from .section_display import section_evaluations

STATE_COLORS = {'IN_POSITION': '#26AD72', 'INSTALLED': '#31A9D4',
                'OUT_OF_POSITION': '#E28232', 'UNKNOWN': '#B9C1CA'}
STATE_LABELS = {'IN_POSITION': '断面相对就位', 'INSTALLED': '已识别安装，尺寸另核',
                'OUT_OF_POSITION': '尺寸或位置不符', 'UNKNOWN': '尚未确认'}


def model_states(report):
    """Read accepted report states; a best-fit candidate alone cannot light CAD."""
    positions = {r['pipe_id']: r.get('state') for r in (report or {}).get('position_assessment', {}).get('pipes', [])}
    states = {}
    for row in (report or {}).get('pipes', []):
        pid = row['pipe_id']
        state = positions.get(pid, (row.get('section_position') or {}).get('state'))
        states[pid] = (state if state in ('IN_POSITION', 'OUT_OF_POSITION')
                       else 'INSTALLED' if row.get('installation_state') == 'INSTALLED'
                       and row.get('installation_evidence_qualified') is True else 'UNKNOWN')
    for pid, state in positions.items():
        if state in ('IN_POSITION', 'OUT_OF_POSITION'):
            states[pid] = state
    return states


def scene_sections(report):
    scenes = []
    for stage in section_evaluations(report):
        model = stage.get('model') or {}
        solved = [row for row in model.get('candidates', []) if row.get('estimated_center_uv_mm') is not None]
        quality = {r['candidate_id']: r for r in stage.get('candidates', [])}
        registration = stage.get('registration') or {}
        matches = registration.get('matches', []) if registration.get('status') == 'MATCHED' else []
        # A malformed/nonunique assignment is never drawn as a linked pipe.
        if (len({m['observation_id'] for m in matches}) != len(matches)
                or len({m['pipe_id'] for m in matches}) != len(matches)):
            matches = []
        by_id = {m['observation_id']: m['pipe_id'] for m in matches}
        origin = np.asarray(solved[0]['estimated_center_uv_mm']) if solved else np.zeros(2)
        origin_covariance = ((solved[0].get('pixel_error') or {}).get('scenarios', {}).get('COHERENT_EDGE_OFFSETS', {})
                             .get('section_parameter_covariance_mm2')) if solved else None
        items = []
        for index, row in enumerate(solved):
            evidence = quality.get(row['id'], {})
            covariance = (row.get('pixel_error') or {}).get('scenarios', {}).get('COHERENT_EDGE_OFFSETS', {}).get('section_parameter_covariance_mm2')
            covariance = np.asarray(covariance, float)[:2, :2] if covariance is not None else None
            if covariance is not None and origin_covariance is not None:
                covariance = covariance + np.asarray(origin_covariance)[:2, :2] if index else np.zeros((2, 2))
            else:
                covariance = None
            stable = (evidence.get('boundary_status') == 'STABLE'
                      and evidence.get('window_sensitivity', {}).get('stable')
                      and not evidence.get('reason_codes'))
            items.append({'id': row['id'], 'label': f'S{index+1}', 'color': row.get('color') or '#8B9BAB',
                          'center_uv_mm': (np.asarray(row['estimated_center_uv_mm'])-origin).tolist(),
                          'diameter_mm': row['diameter_mm'], 'diameter_source': row.get('diameter_source'),
                          'center_covariance_mm2': covariance.tolist() if covariance is not None else None,
                          'subset_stable': bool(stable), 'pipe_id': by_id.get(row['id']),
                          'reason_codes': evidence.get('reason_codes', [])})
        scenes.append({'zone_id': stage.get('zone_id', '全幅'), 'items': items,
                       'pairs': model.get('pairs', []), 'axis_camera': model.get('section', {}).get('normal_camera'),
                       'registration_status': registration.get('status', 'NOT_EVALUATED'),
                       'calibration_validated': stage.get('position_assessment', {}).get('calibration_validated', False),
                       'source_scope': model.get('scope'), 'independent_of_ground': not stage.get('ground_correction', {}).get('applied', False)})
    return scenes


def draw_scene(canvas, scene, *, selected_pipe=None):
    """Equal-scale normal section; dashed circles denote conditional fits."""
    canvas.delete('all')
    w, h = max(canvas.winfo_width(), 150), max(canvas.winfo_height(), 150)
    items = scene.get('items', []) if scene else []
    if not items:
        canvas.create_text(w/2, h/2, text='尚未生成现场断面\n请先点击⑤保存并评估', justify='center', fill='#355371')
        return []
    centers = np.asarray([r['center_uv_mm'][::-1] for r in items], float)
    radii = np.asarray([r['diameter_mm']/2 for r in items])
    low = (centers-radii[:, None]).min(axis=0); high = (centers+radii[:, None]).max(axis=0)
    scale = min(max(40, w-160)/max(high[0]-low[0], 1), max(40, h-220)/max(high[1]-low[1], 1))
    positions = (centers-(low+high)/2)*scale+[w/2, (h+20)/2]
    hits = []
    for row, (x, y), radius in zip(items, positions, radii*scale):
        covariance = row['center_covariance_mm2']
        if covariance is not None:
            eigenvalues, eigenvectors = np.linalg.eigh(np.asarray(covariance)[::-1, ::-1])
            if np.all(eigenvalues >= 0):
                # A conditional 95% joint centre ellipse, not total accuracy.
                angle = np.linspace(0, 2*np.pi, 49)
                ellipse = (np.c_[np.cos(angle), np.sin(angle)]*np.sqrt(5.991*eigenvalues))@eigenvectors.T*scale+[x, y]
                canvas.create_line(*ellipse.ravel(), fill='#CAD3DF', dash=(3, 4), tags='center-uncertainty')
        selected = row['pipe_id'] is not None and row['pipe_id'] == selected_pipe
        radius = max(3., radius)
        canvas.create_oval(x-radius, y-radius, x+radius, y+radius, fill=row['color'],
                           outline='#C17A00' if selected else '#344A60', width=3 if selected else 1,
                           dash=(4, 3), tags=('scene-pipe', row['id']))
        suffix = f" → {row['pipe_id']}" if row['pipe_id'] else ' · 待对应'
        canvas.create_text(x, y-radius-8, text=f"{row['label']}  Ø{row['diameter_mm']:.1f} mm{suffix}",
                           anchor='s', fill='#20374B', tags='scene-label')
        hits.append((row, float(x), float(y), max(radius, 12.)))
    unstable = sum(not r['subset_stable'] for r in items)
    canvas.create_rectangle(0, 0, w, 78, fill='#F2F5F8', outline='')
    canvas.create_text(12, 12, anchor='nw', text=f"现场相对断面 · {scene['zone_id']} · {len(items)} 根候选", fill='#20374B', tags='scene-title')
    canvas.create_text(12, 36, anchor='nw', text='横轴 V（纵深分量），纵轴 U；以 S1 为原点，单位 mm。\n三维管轴从图像解算；无需假定相机与地面垂直。', fill='#355371', width=w-24)
    canvas.create_rectangle(0, h-83, w, h, fill='#F2F5F8', outline='')
    note = f'条件估计；{unstable} 根尚未通过完整尺寸/边界检查。'
    note += '\n虚线椭圆仅表示像素误差假设，不含标定和轴向误差。'
    note += '\n原始估计保留；未用 CAD 管径或间距替换。'
    canvas.create_text(12, h-76, text=note, anchor='nw', fill='#8A5815', width=w-24, tags='scene-limitations')
    return hits
