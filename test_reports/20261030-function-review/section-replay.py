"""Replay only sections with frozen input images, calibration and ground hypothesis."""
from pathlib import Path
import hashlib
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.elevation_dataset import _calibration_ok
from pipe_twin.elevation_zones import crop_calibration_for_zone
from pipe_twin.pooled_sections import analyze_pooled_sections

SCENE = ROOT / 'outputs/elevation_workbench/captures/elevation-20261030-161444-3ab7011e'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dimensions(stage):
    return {row['candidate_id']: {'diameter_mm': row['fit'].get('trial_diameter_mm'),
        'section_center_parameters_mm': row['fit'].get('section_center_parameters_mm'),
        'boundary_count': row.get('observed_boundary_count'),
        'reason_codes': row['reason_codes'], 'window_sensitivity': row['window_sensitivity']}
        for row in stage['candidates']}


def main():
    protected = [SCENE/'manifest.json', SCENE/'report.json', SCENE/'photos/left.png', SCENE/'photos/right.png']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    manifest = json.loads((SCENE/'manifest.json').read_text(encoding='utf8'))
    original = json.loads((SCENE/'report.json').read_text(encoding='utf8'))
    settings = manifest['analysis']['elevation_auto']
    zone = settings['zones'][0]
    calibration = crop_calibration_for_zone(_calibration_ok(manifest['stereo_calibration']), zone,
        right_padding_px=settings['right_roi_padding_px'])
    old = original['section_evaluations'][0]
    offsets = old['zone_offsets']
    images = {}
    for role in ['left', 'right']:
        image = cv2.imdecode(np.fromfile(SCENE/'photos'/f'{role}.png', dtype=np.uint8), cv2.IMREAD_COLOR)
        x, y = offsets[role+'_x'], offsets[role+'_y']
        w, h = offsets.get(role+'_width', offsets['width']), offsets['height']
        images[role] = image[y:y+h, x:x+w].copy()
    # Freeze the old trial plane so changes by the parallel ground audit do
    # not get confused with the section-specific fixes evaluated here.
    ground = dict(original['ground_check'], candidate_trial_enabled=True)
    result = analyze_pooled_sections(images['left'], images['right'], calibration,
        aggregation_band=settings['registration'].get('aggregation_band'), image_offsets=offsets,
        disparity_range=(0., 255.), ground_result=ground, refine_epipolar=True)
    before, after = dimensions(old), dimensions(result)
    assert before.keys() == after.keys()
    for key in before:
        np.testing.assert_allclose(before[key]['diameter_mm'], after[key]['diameter_mm'], rtol=0, atol=1e-8)
        np.testing.assert_allclose(before[key]['section_center_parameters_mm'], after[key]['section_center_parameters_mm'], rtol=0, atol=1e-8)
        assert before[key]['reason_codes'] == after[key]['reason_codes']
    old_pairs = {tuple(row['ids']): row['center_distance_mm'] for row in old['model']['pairs']}
    new_pairs = {tuple(row['ids']): row['center_distance_mm'] for row in result['model']['pairs']}
    assert old_pairs.keys() == new_pairs.keys()
    np.testing.assert_allclose(list(old_pairs.values()), list(new_pairs.values()), rtol=0, atol=1e-8)
    red = next(row for row in result['pairing_audit']['left'] if row['candidate_id'] == 'POOL-L01')
    assert red['status'] == 'INSUFFICIENT_INDEPENDENT_SILHOUETTES'
    assert red['nearby_candidates'][0]['candidate_id'] == 'POOL-R16'
    assert red['nearby_candidates'][0]['combined_stable_boundary_count'] == 2
    assert hashes == {str(p.relative_to(ROOT)): digest(p) for p in protected}
    out = ROOT / 'outputs/function-review-20261030/sections'
    out.mkdir(parents=True, exist_ok=True)
    (out/'replayed-section.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    evidence = {'scope': 'FROZEN_GROUND_SECTION_ONLY_REPLAY_NOT_NEW_CAPTURE_OR_ACCEPTANCE',
        'dataset_id': manifest['dataset_id'], 'source_hashes': hashes,
        'before': before, 'after': after,
        'relative_center_distances_mm': [{'ids': list(key), 'before': old_pairs[key], 'after': new_pairs[key]} for key in old_pairs],
        'pairing_audit': result['pairing_audit'],
        'original_and_replayed_geometry_identical_at_1e_minus_8_mm': True,
        'source_unchanged': True,
        'synthetic_regressions': ['partial disjoint outlines do not crash or merge from one shared edge',
            'two tangents cannot create free circle radius', 'two compatible candidates remain ambiguous',
            'near-parallel baseline uncertainty failure is contained to candidate',
            'independently projected tilted cylinders recover free radius and same-station centre',
            'asymmetric crops preserve camera coordinate offsets']}
    target = Path(__file__).with_name('section-evidence.json')
    target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps({'evidence': str(target), 'upper_red_pairing': red,
                      'geometry_changed': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
