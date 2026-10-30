"""Compare the exact pre-fix LK inputs with brightness-compensated tracking.

Read-only diagnostics: cropped checks are not substituted into production
fusion. No images, coordinates, depths, thresholds or active scene are changed.
"""
from pathlib import Path
import hashlib
import inspect
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pipe_twin.temporal_stereo as temporal

MANIFEST = ROOT/'outputs/function-review-20261030/fresh/burst/elevation-20261030-171527-05c914bf/manifest.json'
COLUMNS = ['passed', 'median_motion_px', 'p95_motion_px', 'brightness_shift',
           'changed_fraction', 'maximum_tile_difference']


def legacy_checker():
    # Controlled ablation: the sole change is restoring the two original LK
    # input arrays. Every threshold, feature detector and other check is shared.
    source = inspect.getsource(temporal.static_pair_check)
    assert source.count('calcOpticalFlowPyrLK(a, tracking_b,') == 1
    assert source.count('calcOpticalFlowPyrLK(tracking_b, a,') == 1
    source = source.replace('calcOpticalFlowPyrLK(a, tracking_b,', 'calcOpticalFlowPyrLK(a, b,')
    source = source.replace('calcOpticalFlowPyrLK(tracking_b, a,', 'calcOpticalFlowPyrLK(b, a,')
    namespace = {'cv2': cv2, 'np': np, 'POLICY': temporal.POLICY}
    exec(compile(source, '<pre-fix LK inputs>', 'exec'), namespace)
    return namespace['static_pair_check']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cv2.setNumThreads(2)
    before = legacy_checker()
    manifest = json.loads(MANIFEST.read_text(encoding='utf8'))
    frames = manifest['capture']['capture_groups'][-1]['temporal_frames']
    files = [MANIFEST] + [MANIFEST.parent/frame['views'][role]['path']
                         for frame in frames for role in ['left', 'right']]
    original_hashes = {str(p.relative_to(ROOT)): digest(p) for p in files}
    regions = {
        'left': {'full': None, 'pipe_roi': [1112, 263, 547, 758], 'ground_roi': [172, 109, 596, 338]},
        'right': {'full': None, 'pipe_roi': [984, 263, 803, 758], 'ground_roi': [172, 109, 596, 338]},
    }
    checks, peaks, phase_checks = [], [], []
    paired_pass = {name: {'before': [], 'after': []} for name in regions['left']}
    for role in ['left', 'right']:
        images = [cv2.imdecode(np.fromfile(MANIFEST.parent/f['views'][role]['path'], np.uint8), 1) for f in frames]
        ref = images[-1]
        for index, image in enumerate(images):
            for name, rect in regions[role].items():
                a, b = ref, image
                if rect:
                    x, y, w, h = rect
                    a, b = a[y:y+h, x:x+w], b[y:y+h, x:x+w]
                old, new = before(a, b), temporal.static_pair_check(a, b)
                checks.append({'role': role, 'frame': index+1, 'region': name,
                               'before': [old.get(k) for k in COLUMNS],
                               'after': [new.get(k) for k in COLUMNS]})
                if name == 'ground_roi':
                    ag, bg = [cv2.cvtColor(v, cv2.COLOR_BGR2GRAY).astype(np.float32) for v in [a, b]]
                    shift, response = cv2.phaseCorrelate(ag-ag.mean(), bg-bg.mean(), cv2.createHanningWindow((w, h), cv2.CV_32F))
                    phase_checks.append({'role': role, 'frame': index+1, 'shift_px': shift, 'response': response})
            ag, bg = [cv2.cvtColor(v, cv2.COLOR_BGR2GRAY).astype(np.float32) for v in [ref, image]]
            brightness = float(np.median(bg-ag))
            delta = abs(cv2.GaussianBlur(bg, (5, 5), 0)-cv2.GaussianBlur(ag, (5, 5), 0)-brightness)
            tiles = [{'row': r, 'column': c, 'mean_abs_difference': float(tile.mean()),
                      'bounds_px': [c*240, r*180, 240, 180]}
                     for r, row in enumerate(np.array_split(delta, 6)) for c, tile in enumerate(np.array_split(row, 8, axis=1))]
            peaks.append({'role': role, 'frame': index+1, 'tile': max(tiles, key=lambda t: t['mean_abs_difference'])})
    for region in paired_pass:
        for variant in ['before', 'after']:
            paired_pass[region][variant] = [index for index in range(1, len(frames)+1)
                if all(row[variant][0] for row in checks if row['region'] == region and row['frame'] == index)]
    # Independent controlled image perturbations: known zero/1/2px motion,
    # with a known uniform -9-level offset, without using CAD or field labels.
    rng = np.random.default_rng(3010)
    gray = cv2.GaussianBlur(rng.normal(120, 10, (240, 320)).astype(np.float32), (5, 5), .8)
    reference = cv2.cvtColor(np.rint(gray).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    controlled = []
    for shift in [0., 1., 2.]:
        image = cv2.warpAffine(reference, np.array([[1., 0., shift], [0., 1., 0.]]),
                              (320, 240), borderMode=cv2.BORDER_REFLECT)
        image = np.clip(image.astype(int)-9, 0, 255).astype(np.uint8)
        controlled.append({'known_translation_px': shift, 'brightness_offset': -9,
                           'before': before(reference, image), 'after': temporal.static_pair_check(reference, image)})
    assert controlled[0]['after']['passed']
    assert all(not row['after']['passed'] for row in controlled[1:])
    assert original_hashes == {str(p.relative_to(ROOT)): digest(p) for p in files}
    evidence = {
        'scope': 'STATIC_CHECK_ABLATION_ONLY_NOT_DEPTH_FUSION_OR_ACCURACY_ACCEPTANCE',
        'source_manifest': str(MANIFEST.relative_to(ROOT)), 'source_hashes': original_hashes,
        'source_unchanged': True, 'revision_after': temporal.REVISION, 'policy_after': temporal.POLICY,
        'metric_columns': COLUMNS, 'regions': regions,
        'right_ground_region_scope': 'Same-coordinate independent carpet patch, not stereo-matched ground pixels',
        'frame_pair_pass_by_region': paired_pass, 'checks': checks, 'worst_global_tiles': peaks,
        'ground_zero_mean_hann_phase_correlation': phase_checks,
        'controlled_perturbations': controlled,
        'limitations': ['Full-frame production criteria remain enabled; no ROI substituted for acceptance',
                        'Pipe ROI still fails its original motion checks; cannot claim the burst is valid',
                        'Exposure, reflections, weak texture and background activity can affect optical flow',
                        'Phase correlation is supporting evidence of a dominant stable patch, not proof all pipes are stationary']}
    target = Path(__file__).with_name('temporal-motion-evidence.json')
    target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps({'evidence': str(target), 'paired_pass': paired_pass}, ensure_ascii=False))


if __name__ == '__main__':
    main()
