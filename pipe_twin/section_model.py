"""Common normal-plane section model, with explicit dimension provenance.

This consumes conditional fits; it does not turn them into accepted field
measurements. Relative centres and clearances do not depend on pipe length,
camera distance in CAD, or which axial station represents the common plane.
"""
from __future__ import annotations

import csv
from html import escape
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .section_diagnostics import circle_trial, section_basis


def fit_surface_arc(points_camera_mm, axis):
    """Collapse one already-separated cylinder's actual 3D samples to a section.

    No known length or reference diameter is required. Missing arc samples
    stay missing; curvature and coverage checks control the free circle fit.
    """
    points = np.asarray(points_camera_mm, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('INVALID_SURFACE_POINTS')
    result = circle_trial(points@section_basis(axis))
    fit = {'status': 'INSUFFICIENT_SURFACE_ARC', 'diameter_source': 'SURFACE_ARC_FIT',
           'axis': (np.asarray(axis)/np.linalg.norm(axis)).tolist(),
           'accepted_diameter_mm': None, 'arc_diagnostic': result}
    if result['geometry_passed']:
        fit.update(status='CONDITIONAL_FIT', trial_diameter_mm=result['trial_diameter_mm'],
                   section_center_parameters_mm=result['trial_center_mm'])
    return fit


def _pair_uncertainty(first, second, delta):
    distance = np.linalg.norm(delta)
    if distance < 1e-9:
        return {'status': 'COINCIDENT_CENTRES_NONDIFFERENTIABLE', 'scenarios': {}}
    direction = delta/distance
    scenarios = {}
    left = first.get('pixel_error', {}).get('scenarios', {})
    right = second.get('pixel_error', {}).get('scenarios', {})
    for label in sorted(left.keys() & right.keys()):
        covariances = [np.asarray(row[label]['section_parameter_covariance_mm2'], float)
                       for row in (left, right)]
        for covariance in covariances:
            if (covariance.shape != (3, 3) or not np.isfinite(covariance).all()
                    or not np.allclose(covariance, covariance.T, atol=1e-6)
                    or np.linalg.eigvalsh(covariance).min() < -1e-6):
                raise ValueError('INVALID_SECTION_COVARIANCE')
        values = {}
        for name, radius_derivative in [('center_distance', 0.), ('surface_clearance', -1.)]:
            gradients = [np.r_[-direction, radius_derivative], np.r_[direction, radius_derivative]]
            variance = sum(float(g@c@g) for g, c in zip(gradients, covariances))
            sigma = float(np.sqrt(max(variance, 0.)))
            values[name+'_sigma_mm'] = sigma
            values[name+'_normal_95_halfwidth_mm'] = 1.96*sigma
        scenarios[label] = values
    return {'status': 'CONDITIONAL_LINEAR_PROPAGATION' if scenarios else 'NOT_AVAILABLE',
            'total_accuracy_known': False, 'assumes_independent_candidate_edge_errors': True,
            'excluded': ['SHARED_CALIBRATION_AND_AXIS_ERRORS', 'EDGE_MISIDENTIFICATION',
                         'CROSS_CANDIDATE_CORRELATION', 'SUPPLIED_DIAMETER_TOLERANCE'],
            'scenarios': scenarios}


def build_section_model(candidates, axis, *, origin_id=None, station_mm=0., metadata=None):
    """Place any number of parallel-axis candidates in one common section.

    Each item has id, label, color, reference_diameter_mm and fit. An unresolved
    fit stays unresolved. All fits must use the exact same unoriented axis;
    opposite axis signs are transformed back to this model's basis. Negative
    clearance is preserved as an inconsistency, never clamped or hidden.
    """
    basis = section_basis(axis)
    axis = np.asarray(axis, float)/np.linalg.norm(axis)
    if not np.isfinite(station_mm):
        raise ValueError('INVALID_SECTION_STATION')
    ids = [c['id'] for c in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError('DUPLICATE_CANDIDATE_ID')
    if origin_id is not None and origin_id not in ids:
        raise ValueError('UNKNOWN_ORIGIN_CANDIDATE')
    items, original = [], {}
    for candidate in candidates:
        fit = candidate['fit']
        item = {k: candidate.get(k) for k in ('id', 'label', 'color', 'reference_diameter_mm')}
        item.update(status=fit['status'], accepted_measurement=False,
                    estimated_center_uv_mm=None, diameter_mm=None, radius_mm=None,
                    diameter_source=fit.get('diameter_source'), pixel_error=candidate.get('pixel_error'))
        if fit['status'] == 'CONDITIONAL_FIT':
            fit_axis = np.array(fit['axis'], dtype=float, copy=True)
            fit_basis = section_basis(fit_axis)
            fit_axis /= np.linalg.norm(fit_axis)
            if np.linalg.norm(np.cross(fit_axis, axis)) > 1e-8:
                raise ValueError('CANDIDATES_REQUIRE_COMMON_AXIS')
            transform = basis.T@fit_basis
            center = transform@np.asarray(fit['section_center_parameters_mm'], float)
            diameter = float(fit['trial_diameter_mm'])
            if center.shape != (2,) or not np.isfinite(center).all() or not np.isfinite(diameter) or diameter <= 0:
                raise ValueError('INVALID_FITTED_SECTION')
            item.update(section_parameters_camera_basis_mm=center.tolist(), diameter_mm=diameter,
                        radius_mm=diameter/2, reference_difference_mm=(diameter-item['reference_diameter_mm'])
                        if item['reference_diameter_mm'] is not None and fit.get('diameter_source') in ('IMAGE_FIT', 'SURFACE_ARC_FIT') else None)
            # Covariance belongs to the fit basis, including centre-radius
            # correlation. Rotate it with the centre, not just its diagonal.
            uncertainty = candidate.get('pixel_error') or {}
            rotation = np.eye(3); rotation[:2, :2] = transform
            item['pixel_error'] = {**uncertainty, 'scenarios': {
                key: {**value, 'section_parameter_covariance_mm2':
                      (rotation@np.asarray(value['section_parameter_covariance_mm2'])@rotation.T).tolist()}
                for key, value in uncertainty.get('scenarios', {}).items()}}
            original[item['id']] = center
        items.append(item)
    if origin_id is not None and origin_id not in original:
        raise ValueError('ORIGIN_CANDIDATE_UNRESOLVED')
    origin = original[origin_id] if origin_id is not None else np.zeros(2)
    for item in items:
        if item['id'] in original:
            center = original[item['id']]
            item['estimated_center_uv_mm'] = (center-origin).tolist()
            item['center_in_common_plane_camera_mm'] = (basis@center+station_mm*axis).tolist()
    pairs = []
    solved = [row for row in items if row['id'] in original]
    for first, second in combinations(solved, 2):
        delta = original[second['id']]-original[first['id']]
        distance = float(np.linalg.norm(delta))
        clearance = distance-first['radius_mm']-second['radius_mm']
        pairs.append({'ids': [first['id'], second['id']], 'delta_uv_mm': delta.tolist(),
                      'center_distance_mm': distance, 'surface_clearance_mm': clearance,
                      'status': 'OVERLAPPING_SECTION_INCONSISTENCY' if clearance < 0 else 'CONDITIONAL_ESTIMATE',
                      'accepted_measurement': False, 'pixel_error': _pair_uncertainty(first, second, delta)})
    return {'schema_version': 'conditional-common-section-v1', 'units': 'mm',
            'scope': 'CONDITIONAL_MODEL_NOT_ACCEPTED_MEASUREMENT', 'total_accuracy_known': False,
            'section': {'normal_camera': axis.tolist(), 'basis_camera_columns_uv': basis.tolist(),
                        'station_mm_along_axis_from_left_origin': float(station_mm), 'origin_id': origin_id,
                        'origin_camera_mm': (basis@origin+station_mm*axis).tolist(),
                        'definition': 'ONE_PLANE_NORMAL_TO_COMMON_PIPE_AXES',
                        'clearance_definition': 'CENTRE_DISTANCE_MINUS_BOTH_OUTER_RADII'},
            'candidates': items, 'pairs': pairs, 'metadata': metadata or {}}


def section_svg(model):
    """Equal-scale, portable vector drawing; no optional plotting dependency."""
    solved = [c for c in model['candidates'] if c['estimated_center_uv_mm'] is not None]
    result = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="590" viewBox="0 0 1000 590">',
              '<rect width="1000" height="590" fill="white"/>',
              '<g font-family="Microsoft YaHei,Arial,sans-serif" fill="#182b40">',
              '<text x="30" y="32" font-size="22">共同法向断面 · 条件模型（mm）</text>',
              '<text x="30" y="58" font-size="14">管径来源见圆旁标注；位置为图像估计。坐标横向 V、纵向 U，等比例。</text>']
    if solved:
        bounds = np.asarray([[c['estimated_center_uv_mm'][1]-c['radius_mm'],
                              c['estimated_center_uv_mm'][0]-c['radius_mm'],
                              c['estimated_center_uv_mm'][1]+c['radius_mm'],
                              c['estimated_center_uv_mm'][0]+c['radius_mm']] for c in solved])
        low = bounds[:, :2].min(axis=0); high = bounds[:, 2:].max(axis=0)
        scale = min(760/max(high[0]-low[0], 1), 300/max(high[1]-low[1], 1))
        def xy(c):
            return (np.asarray(c)[::-1]-low)*scale+np.array([110., 145.])
        by_id = {c['id']: c for c in solved}
        for pair in model['pairs']:
            p, q = [xy(by_id[i]['estimated_center_uv_mm']) for i in pair['ids']]
            delta = q-p; norm = max(float(np.linalg.norm(delta)), 1.)
            text_pos = (p+q)/2+20*np.array([-delta[1], delta[0]])/norm
            result += [f'<line x1="{p[0]:.2f}" y1="{p[1]:.2f}" x2="{q[0]:.2f}" y2="{q[1]:.2f}" stroke="#8996a3" stroke-dasharray="5 5"/>',
                       f'<text x="{text_pos[0]:.2f}" y="{text_pos[1]:.2f}" font-size="13" text-anchor="middle" paint-order="stroke" stroke="white" stroke-width="4" stroke-linejoin="round">心距 {pair["center_distance_mm"]:.1f} / 净距 {pair["surface_clearance_mm"]:.1f}</text>']
        for c in solved:
            p = xy(c['estimated_center_uv_mm']); radius = c['radius_mm']*scale
            color = escape(c.get('color') or '#666', quote=True)
            source = '给定' if c['diameter_source'] == 'USER_KNOWN_CONSTRAINT' else '估算'
            result += [f'<circle cx="{p[0]:.2f}" cy="{p[1]:.2f}" r="{radius:.2f}" fill="{color}" fill-opacity=".14" stroke="{color}" stroke-width="2" stroke-dasharray="6 3"/>',
                       f'<path d="M {p[0]-5:.2f},{p[1]:.2f} h 10 M {p[0]:.2f},{p[1]-5:.2f} v 10" stroke="{color}"/>',
                       f'<text x="{p[0]:.2f}" y="{p[1]-radius-24:.2f}" font-size="15" text-anchor="middle">{escape(c["label"] or c["id"])}</text>',
                       f'<text x="{p[0]:.2f}" y="{p[1]-radius-6:.2f}" font-size="13" text-anchor="middle">{source} D {c["diameter_mm"]:.1f} / R {c["radius_mm"]:.1f}</text>']
        bar = 50*scale
        result += [f'<line x1="{900-bar:.2f}" y1="490" x2="900" y2="490" stroke="#182b40" stroke-width="3"/>',
                   f'<text x="{900-bar/2:.2f}" y="513" text-anchor="middle" font-size="14">50 mm</text>']
    unresolved = [c['label'] or c['id'] for c in model['candidates'] if c['estimated_center_uv_mm'] is None]
    result += [f'<text x="30" y="545" font-size="14">未定位：{escape("、".join(unresolved) or "无")}</text>',
               '<text x="30" y="571" font-size="13">圆为完整圆柱假设的模型轮廓，并非已观测完整圆弧；标定、遮挡与边缘误差仍影响位置。</text>', '</g></svg>']
    return '\n'.join(result)


def write_section_artifacts(model, directory):
    """Export inspectable JSON, per-pipe/pair CSVs, SVG, DXF and local HTML."""
    import ezdxf

    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    (directory/'section-model.json').write_text(json.dumps(model, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    with (directory/'pipe-dimensions.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['id', 'label', 'status', 'diameter_source', 'diameter_mm', 'radius_mm', 'relative_u_mm', 'relative_v_mm', 'reference_diameter_mm'])
        for c in model['candidates']:
            writer.writerow([c['id'], c['label'], c['status'], c['diameter_source'], c['diameter_mm'], c['radius_mm'],
                             *(c['estimated_center_uv_mm'] or [None, None]), c['reference_diameter_mm']])
    with (directory/'pair-dimensions.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.writer(stream)
        writer.writerow(['first', 'second', 'center_distance_mm', 'surface_clearance_mm', 'status'])
        for p in model['pairs']:
            writer.writerow([*p['ids'], p['center_distance_mm'], p['surface_clearance_mm'], p['status']])
    svg = section_svg(model)
    (directory/'section.svg').write_text(svg, encoding='utf-8')
    drawing = ezdxf.new('R2010'); drawing.units = 4  # mm, CAD X=V and Y=-U.
    space = drawing.modelspace()
    solved = {c['id']: c for c in model['candidates'] if c['estimated_center_uv_mm'] is not None}
    for c in solved.values():
        u, v = c['estimated_center_uv_mm']
        space.add_circle((v, -u), c['radius_mm'])
        space.add_text(f'{c["id"]}: D {c["diameter_mm"]:.2f}, R {c["radius_mm"]:.2f}',
                       dxfattribs={'height': 3., 'insert': (v-c['radius_mm'], -u+c['radius_mm']+6)})
    for p in model['pairs']:
        centers = [(solved[i]['estimated_center_uv_mm'][1], -solved[i]['estimated_center_uv_mm'][0]) for i in p['ids']]
        space.add_aligned_dim(centers[0], centers[1], distance=12.,
                              override={'dimtxt': 3., 'dimasz': 3., 'dimdec': 1}).render()
        midpoint = np.mean(centers, axis=0)
        space.add_text(f'clearance {p["surface_clearance_mm"]:.2f} mm',
                       dxfattribs={'height': 3., 'insert': tuple(midpoint+[0, -8])})
    space.add_text('CONDITIONAL SECTION; mm; X=V, Y=-U; NOT ACCEPTED MEASUREMENT',
                   dxfattribs={'height': 3., 'insert': (0., -60.)})
    if len(solved) != len(model['candidates']):
        space.add_text('Unresolved candidates omitted: '+', '.join(c['id'] for c in model['candidates'] if c['id'] not in solved),
                       dxfattribs={'height': 3., 'insert': (0., -65.)})
    drawing.saveas(directory/'section.dxf')
    def number(value):
        return '—' if value is None else f'{value:.1f}'
    rows = []
    for c in model['candidates']:
        center = c['estimated_center_uv_mm'] or [None, None]
        source = {'USER_KNOWN_CONSTRAINT': '用户给定，参与定位', 'IMAGE_FIT': '图像自由估算', 'SURFACE_ARC_FIT': '局部三维圆弧拟合'}.get(c['diameter_source'], '未独立解出')
        rows.append('<tr>'+''.join(f'<td>{escape(str(v))}</td>' for v in [c['label'] or c['id'], number(c['reference_diameter_mm']),
                    number(c['diameter_mm']), number(c['radius_mm']), source, number(center[0]), number(center[1])])+'</tr>')
    pair_rows = []
    for p in model['pairs']:
        uncertainty = p['pixel_error']['scenarios'].get('COHERENT_EDGE_OFFSETS', {})
        pair_rows.append('<tr>'+''.join(f'<td>{escape(str(v))}</td>' for v in [' / '.join(p['ids']), number(p['center_distance_mm']),
                         number(p['surface_clearance_mm']), number(uncertainty.get('center_distance_sigma_mm'))])+'</tr>')
    metadata = escape(json.dumps(model['metadata'], ensure_ascii=False, indent=2))
    html = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>管道断面模型与尺寸</title><style>body{font:16px/1.7 Arial,"Microsoft YaHei",sans-serif;max-width:1100px;margin:25px auto;padding:0 20px;color:#182b40}svg{width:100%;height:auto}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{padding:8px;border:1px solid #ccd3dd;text-align:left}.notice{background:#fff4d9;padding:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>'
    uses_reference = any(c['diameter_source'] == 'USER_KNOWN_CONSTRAINT' for c in solved.values())
    size_note = '部分外径作为约束输入，见尺寸来源列。' if uses_reference else '采购外径仅作参照，没有用于拟合半径或位置。'
    html += '<h1>管道断面模型与尺寸</h1><p class="notice">这是图像解算得到的条件模型。'+size_note+'图像估计的尺寸尚未通过独立量测核验；暂用标定或图像微校正不能直接作为验收依据。仅绘制已解出的候选；完整圆周为拟合轮廓，不代表遮挡部分已被观测。</p>'+svg
    html += '<h2>管道尺寸（mm）</h2><p>圆心坐标相对选定原点，均在同一个垂直于共同管轴的平面上。改变截面沿轴的位置不会改变这里的管间尺寸。</p><table><tr><th>候选</th><th>参照外径</th><th>模型外径</th><th>半径</th><th>外径来源</th><th>圆心 U</th><th>圆心 V</th></tr>'+''.join(rows)+'</table>'
    html += '<h2>断面间距（mm）</h2><p>净距 = 轴心距 − 两个外半径。误差列假设每条边缘有独立的 0.5 px 整体偏移，给出 1σ；不包含标定、轴向、遮挡判断和给定直径公差，不能作为总精度。</p><table><tr><th>候选对</th><th>轴心距</th><th>表面净距</th><th>像素假设下心距 σ</th></tr>'+''.join(pair_rows)+'</table>'
    html += '<p><a href="section-model.json">完整模型 JSON</a> · <a href="pipe-dimensions.csv">管尺寸 CSV</a> · <a href="pair-dimensions.csv">间距 CSV</a> · <a href="section.svg">矢量断面 SVG</a> · <a href="section.dxf">CAD 断面 DXF</a></p><details><summary>输入与限制</summary><pre>'+metadata+'</pre></details></html>'
    (directory/'index.html').write_text(html, encoding='utf-8')
