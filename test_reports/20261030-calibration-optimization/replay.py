"""Replay the frozen 23 pairs with the unchanged four-view holdout; never activate."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.calibration_wizard import read_calibration_diagnostic, solve_stereo_calibration_checked


def main():
    source = ROOT / "outputs/measurement_workbench/calibration_diagnostics/calibration_20261030_120105_953411.npz"
    output = ROOT / "outputs/calibration-optimization-20261030"
    output.mkdir(parents=True, exist_ok=True)
    active = ROOT / "outputs/measurement_workbench/calibration_current.json"
    protected = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (source, active)}
    pairs, metadata = read_calibration_diagnostic(source)
    original = metadata['result_audit']
    archive = metadata['capture_archive']
    session = Path(archive['session_path'])
    manifest = json.loads((session / 'session.json').read_text(encoding='utf8'))
    records = {entry['capture_id']: entry for entry in manifest['pairs']}
    for cid in archive['pair_capture_ids']:
        for role in ('left', 'right'):
            path = session / records[cid][f'{role}_file']
            protected[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    start = time.monotonic()
    result = solve_stereo_calibration_checked(
        pairs=pairs, image_size=(1920, 1080), pattern=(8, 6), square_mm=17.,
        min_pairs=10, max_reprojection_rms_px=1.5, operator='field', max_sync_delta_ms=1.,
        layout='side_by_side_left_right', expected_baseline_mm=60.)
    seconds = time.monotonic() - start
    assert result.audit['holdout']['input_pair_indices'] == original['holdout']['input_pair_indices']
    assert result.audit['training_input_indices'] == original['training_input_indices']
    assert result.audit['holdout']['policy_sha256'] == original['holdout']['policy_sha256']
    (output / 'production-result.json').write_text(json.dumps(asdict(result), ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    for name, digest in protected.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
    evidence = {'source': str(source.relative_to(ROOT)), 'capture_ids': archive['pair_capture_ids'],
                'training_indices': result.audit['training_input_indices'],
                'holdout_indices': result.audit['holdout']['input_pair_indices'],
                'policy_sha256': result.audit['holdout']['policy_sha256'],
                'seconds': seconds, 'validated': result.validated,
                'reasons': result.rejection_reasons, 'joint_refinement': result.audit['joint_refinement'],
                'before': {'stereo_rms_px': original['stereo_rms_px'],
                           'baseline_mm': original['baseline_mm'], 'holdout': original['holdout']},
                'after': {'stereo_rms_px': result.stereo_rms_px,
                          'baseline_mm': result.calibration['baseline_mm'], 'holdout': result.audit['holdout']},
                'protected_sha256': protected}
    Path(__file__).with_name('evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({k:v for k,v in evidence.items() if k not in ('protected_sha256','before','after')}, ensure_ascii=False), flush=True)
    print(json.dumps(evidence['after'], ensure_ascii=False), flush=True)
    assert result.validated, result.rejection_reasons


if __name__ == '__main__':
    main()
