"""Extended free-radius partial-contour search over all twenty saved pairs."""
import json
import re
from build_field_section import *


def save_result(frames, source_hashes):
    values = [d for frame in frames for d in frame['visibility_admissible_diameters_mm']]
    summary = {'frame_count': len(frames), 'initialization_count': sum(len(f['trial']['trials']) for f in frames),
               'visibility_admissible_hypotheses': len(values),
               'admissible_diameter_range_mm': [min(values), max(values)] if values else None,
               'frames_with_admissible_diameter_span_over_5mm': sum(
                   max(v)-min(v) > 5 for frame in frames if (v := frame['visibility_admissible_diameters_mm']))}
    (OUT/'occluded-rim-20-frames.json').write_text(json.dumps({'frames': frames, 'summary': summary,
        'reference_sizes_used': False, 'source_image_hashes': source_hashes, 'code_hash': sha(Path(__file__)),
        'fit_code_hash': sha(ROOT/'test_reports/20261021-section-model/build_field_section.py'),
        'accepted_diameter_mm': None}, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    html_path = OUT/'index.html'
    if html_path.exists():
        html = html_path.read_text(encoding='utf-8')
        html = re.sub(r'<section id="rim-search-summary">.*?</section>', '', html, flags=re.S)
        section = '<section id="rim-search-summary"><h2>后红局部轮廓的延长分析</h2><p>'
        section += f'已完成 {len(frames)} 对、{summary["initialization_count"]} 次初值搜索，每次最多 480 步。'
        if values:
            section += f'通过简单遮挡检查的外径假设仍跨 {min(values):.1f}～{max(values):.1f} mm；其中 {summary["frames_with_admissible_diameter_span_over_5mm"]} 帧内部存在相差超过 5 mm 的解。'
        section += '因此后红管保留局部可见轮廓，未选一个半径补成确定圆。</p><p><a href="occluded-rim-20-frames.json">全部搜索数值记录</a></p></section>'
        html_path.write_text(html.replace('</html>', section+'</html>'), encoding='utf-8')


def main():
    previous = read(ROOT/'outputs/section-check-20261021/evidence.json')
    for hashes in previous['source_image_hashes'].values():
        for name, digest in hashes.items():
            assert sha(SOURCE/name) == digest
    k = np.asarray(previous['rectified_intrinsic']); axis = np.asarray(previous['axis']['axis'])
    axis /= np.linalg.norm(axis); baseline = previous['active_baseline_mm']
    frames = []
    for index in range(20):
        images = {role: cv2.imdecode(np.fromfile(SOURCE/f'{index:03d}_{role}.png', np.uint8), 1) for role in ['left', 'right']}
        observations, _ = observations_for(images)
        rims = [partial_end_rim(images[role], role) for role in ['left', 'right']]
        trial = partial_rim_trials(rims, observations['red_rear'], observations['red_front'], k, baseline, axis)
        admissible = [t for t in trial['trials'] if all(v['hidden_inside_foreground_envelope'] for v in t['visibility_checks'])]
        frames.append({'frame': index, 'trial': trial, 'visibility_admissible_diameters_mm': [t['hypothesis_diameter_mm'] for t in admissible]})
        print(index, [round(t['hypothesis_diameter_mm'], 2) for t in admissible], flush=True)
    save_result(frames, previous['source_image_hashes'])


if __name__ == '__main__':
    main()
