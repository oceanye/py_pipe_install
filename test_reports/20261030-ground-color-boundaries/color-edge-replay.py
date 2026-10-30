"""Frozen last-frame colour-edge comparison; never changes the active scene."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from pipe_twin.elevation_dataset import _calibration_ok
from pipe_twin.pooled_sections import analyze_pooled_sections

SCENE = ROOT / 'outputs/elevation_workbench/captures/elevation-20261030-183635-b6b7b875'
OUT = ROOT / 'outputs/function-review-20261030/color-edges'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(stage):
    return {'conditional_section_count': stage['conditional_section_count'],
        'axis': stage['axis'], 'ground_correction': stage['ground_correction'],
        'epipolar_refinement': stage['epipolar_refinement'],
        'views': {role: [{k: c.get(k) for k in ['candidate_id', 'region_px', 'appearance',
            'width_px', 'boundary_status', 'edge_extraction', 'aggregation', 'full_span_audit', 'span_attempts']}
            for c in items if c['region_px'][1] > 250 and c['region_px'][1] < 600
            and c['region_px'][2] > 600] for role, items in stage['views'].items()},
        'candidates': [{k: c.get(k) for k in ['candidate_id', 'fit', 'window_sensitivity',
            'reason_codes', 'pixel_error']} for c in stage['candidates']],
        'pairs': (stage.get('model') or {}).get('pairs', [])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('label')
    args = parser.parse_args()
    cv2.setNumThreads(2)
    protected = [SCENE/'manifest.json', SCENE/'report.json', SCENE/'photos/left.png', SCENE/'photos/right.png']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    manifest = json.loads((SCENE/'manifest.json').read_text(encoding='utf8'))
    original = json.loads((SCENE/'report.json').read_text(encoding='utf8'))
    images = {role: cv2.imdecode(np.fromfile(SCENE/'photos'/f'{role}.png', np.uint8), cv2.IMREAD_COLOR)
        for role in ['left', 'right']}
    settings = manifest['analysis']['elevation_auto']['registration']
    stage = analyze_pooled_sections(images['left'], images['right'], _calibration_ok(manifest['stereo_calibration']),
        aggregation_band=settings['aggregation_band'], disparity_range=(0., 255.),
        ground_result=original['ground_check'], refine_epipolar=True)
    assert hashes == {str(p.relative_to(ROOT)): digest(p) for p in protected}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/f'{args.label}-stage.json').write_text(json.dumps(stage, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    result = {'source': str(SCENE.relative_to(ROOT)), 'source_hashes': hashes,
        'source_unchanged': True, 'scope': 'CONDITIONAL_LAST_FRAME_NOT_BURST_FUSION_OR_ACCEPTANCE',
        **summary(stage)}
    target = Path(__file__).with_name(f'color-edge-{args.label}.json')
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps({'output': str(target), 'count': stage['conditional_section_count'],
        'candidates': [{k: c.get(k) for k in ['candidate_id', 'window_sensitivity', 'reason_codes']} |
            {'diameter': c['fit'].get('trial_diameter_mm')} for c in stage['candidates']]}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
