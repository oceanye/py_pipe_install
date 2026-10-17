"""Replay the 17 mm field data without changing the active calibration.

Default: the production checked solve. --joint: the diagnostic joint solve.
--redetect uniformly re-extracts every image with accurate SB, including the
fixed validation images. Validation is never used to choose initial values.
Raw images/checkpoints remain local and are not distributed with this report.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.calibration_wizard import (
    _intrinsic_problems, _object_points, align_pair_orientation,
    detect_board_corners, read_calibration_diagnostic,
    rectification_quality_report, solve_stereo_calibration_checked,
)
from pipe_twin.calibration_quality import holdout_quality


def json_default(value):
    return value.tolist() if isinstance(value, np.ndarray) else value.item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        default=ROOT/'outputs/calibration-20261013/accepted-31-before-final.npz')
    parser.add_argument('--output', type=Path,
                        default=ROOT/'outputs/calibration-20261014/readonly-replay.json')
    parser.add_argument('--joint', action='store_true')
    parser.add_argument('--redetect', action='store_true')
    args = parser.parse_args()
    cv2.setNumThreads(2)
    pairs, metadata = read_calibration_diagnostic(args.checkpoint)
    if len(pairs) != 31 or metadata['square_mm'] != 17 or metadata['pattern_inner_corners'] != [8, 6]:
        raise ValueError('This field replay requires the frozen 31-pair, 17 mm, 8x6 checkpoint')
    if args.redetect:
        archive = metadata['capture_archive']
        directory = Path(archive['session_path'])
        manifest = json.loads((directory/'session.json').read_text(encoding='utf-8'))
        entries = {p['capture_id']: p for p in manifest['pairs']}
        refined = []
        for capture_id, original in zip(archive['pair_capture_ids'], pairs):
            pair = []
            for role, prior in zip(('left', 'right'), original):
                path = directory/entries[capture_id][f'{role}_file']
                frame = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
                if frame is None or frame.shape[:2] != (1080, 1920):
                    raise ValueError(f'Invalid source image: {path}')
                obs = detect_board_corners(frame, pattern=(8, 6), sb_accuracy=True,
                                           fallback_to_classic=False)
                if obs is None:
                    raise ValueError(f'SB failed; do not silently drop image: {path}')
                pair.append(align_pair_orientation(prior, obs, pattern=(8, 6)))
            refined.append(tuple(pair))
        pairs = refined
    if not args.joint:
        result = solve_stereo_calibration_checked(
            pairs=pairs, image_size=(1920, 1080), pattern=(8, 6), square_mm=17.,
            min_pairs=10, max_reprojection_rms_px=1.5, operator='field',
            max_sync_delta_ms=1., layout='side_by_side_left_right', expected_baseline_mm=60.)
        record = asdict(result)
    else:
        reserved = [4, 8, 12, 17, 21, 25]
        training = [i for i in range(31) if i not in reserved]
        objects = [_object_points((8, 6), 17.) for _ in training]
        left = [pairs[i][0].corners_px.astype(np.float32).reshape(-1, 1, 2) for i in training]
        right = [align_pair_orientation(*pairs[i], pattern=(8, 6)).corners_px.astype(np.float32)
                 .reshape(-1, 1, 2) for i in training]
        flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3 | cv2.CALIB_USE_LU

        def fit(k1, d1, k2, d2, warm=None):
            common = (objects, left, right, k1, d1, k2, d2, (1920, 1080))
            if warm is None:
                values = cv2.stereoCalibrate(*common, flags=flags,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-9))
            else:
                values = cv2.stereoCalibrateExtended(*common, np.asarray(warm['R']), np.asarray(warm['T']),
                    flags=flags | cv2.CALIB_USE_EXTRINSIC_GUESS,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-10))
            rms, k1, d1, k2, d2, r, t = values[:7]
            r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(k1, d1, k2, d2, (1920, 1080), r, t,
                                                       flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.)
            problems = _intrinsic_problems(k1, d1, (1920, 1080), 'left') + _intrinsic_problems(k2, d2, (1920, 1080), 'right')
            baseline = float(np.linalg.norm(t))
            if abs(baseline - 60.) > 9.:
                problems.append('Baseline differs from expected 60 mm by >15%')
            return {'rms': float(rms), 'baseline_mm': baseline, 'problems': problems,
                **{k: v.tolist() for k, v in zip(('K1', 'D1', 'K2', 'D2', 'R', 'T', 'R1', 'R2', 'P1', 'P2'),
                                               (k1, d1, k2, d2, r, t, r1, r2, p1, p2))},
                'training_geometry': rectification_quality_report(left, right, k1, d1, k2, d2, r1, r2, p1, p2)}

        candidates = []
        for seed in (1000., 1600., 2400.):
            k = np.array([[seed, 0., 959.5], [0., seed, 539.5], [0., 0., 1.]])
            candidate = fit(k.copy(), np.zeros(5), k.copy(), np.zeros(5))
            candidate['seed'] = seed
            candidates.append(candidate)
            print('TRAIN', seed, candidate['rms'], flush=True)
        chosen = min(candidates, key=lambda c: (bool(c['problems']), c['rms']))
        trace = []
        converged = False
        for step in range(1, 9):
            previous = chosen
            chosen = fit(*(np.asarray(previous[k]) for k in ('K1', 'D1', 'K2', 'D2')), warm=previous)
            improvement = previous['rms'] - chosen['rms']
            trace.append({'step': step, 'rms': chosen['rms'], 'improvement_px': improvement,
                          'focal_px': chosen['K1'][0][0], 'baseline_mm': chosen['baseline_mm']})
            print('CONTINUE', trace[-1], flush=True)
            if abs(improvement) < 1e-7 and abs(chosen['K1'][0][0] - previous['K1'][0][0]) < 1e-3:
                converged = True
                break
            if chosen['problems']:
                break
        recipe = {k: chosen[k] for k in ('K1', 'D1', 'K2', 'D2', 'R1', 'R2', 'P1', 'P2')}
        check = holdout_quality([pairs[i] for i in reserved], recipe=recipe,
                                image_size=(1920, 1080), pattern=(8, 6), square_mm=17.)
        record = {'scope': 'diagnostic_only_not_saved', 'validated': False,
                  'training_indices': [i + 1 for i in training], 'holdout_indices': [i + 1 for i in reserved],
                  'initial_candidates': candidates, 'trace': trace, 'converged': converged,
                  'selected': chosen, 'holdout': check}
    record['source_sha256'] = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=json_default) + '\n', encoding='utf-8')
    print('Saved diagnostic:', args.output, flush=True)


if __name__ == '__main__':
    main()
