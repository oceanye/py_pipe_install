"""Replay the frozen field solve without saving/replacing active calibration.

Run from the repository root. Field images/checkpoints are local-only.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.calibration_wizard import read_calibration_diagnostic, solve_stereo_calibration_checked

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, default=ROOT/'outputs/calibration-20261013/accepted-31-before-final.npz')
parser.add_argument('--output', type=Path, default=ROOT/'outputs/calibration-20261013/readonly-replay.json')
args = parser.parse_args()
pairs, metadata = read_calibration_diagnostic(args.checkpoint)
result = solve_stereo_calibration_checked(pairs=pairs,
    image_size=tuple(metadata['image_size_px']), pattern=tuple(metadata['pattern_inner_corners']),
    square_mm=metadata['square_mm'], max_reprojection_rms_px=1.5, min_pairs=10,
    operator='field', max_sync_delta_ms=1., layout='side_by_side_left_right', expected_baseline_mm=60.)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2,
    default=lambda value: value.tolist() if isinstance(value,np.ndarray) else value.item())+'\n', encoding='utf-8')
print(json.dumps({'validated': result.validated, 'stereo_rms_px':result.stereo_rms_px,
    'holdout_cross_rms_px':result.audit['holdout']['cross_reprojection_rms_px'],
    'rejection_reasons':result.rejection_reasons},ensure_ascii=False))
