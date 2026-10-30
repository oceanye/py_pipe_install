"""Same-pair ground audit; no camera, source mutation or tolerance changes."""
from pathlib import Path
from types import ModuleType
import hashlib
import json
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.ground_check import check_ground, ground_summary_lines, POLICY_SHA256
from pipe_twin.stereo_analyzer import _analysis_config, _calibration_from_manifest, _compute_stereo_depth


def compact(check):
    return {k: v for k, v in check.items() if k not in ['observed_support', 'policy']}


def main():
    source = ROOT/'outputs/elevation_workbench/captures/elevation-20261030-161444-3ab7011e'
    out = ROOT/'outputs/function-review-20261030/ground'
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf8'))
    old = ModuleType('pipe_twin._ground_before')
    old.__package__ = 'pipe_twin'
    baseline = subprocess.run(['git', 'show', 'e663f6e:pipe_twin/ground_check.py'], cwd=ROOT,
                              check=True, capture_output=True).stdout
    exec(compile(baseline, 'ground_check_at_e663f6e.py', 'exec'), old.__dict__)
    protected = [source/'manifest.json', source/'report.json', source/'photos/left.png', source/'photos/right.png',
                 ROOT/'outputs/measurement_workbench/calibration_current.json',
                 ROOT/'outputs/elevation_workbench/last_scene.json']
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    calibration = _calibration_from_manifest(manifest['stereo_calibration'])
    region = manifest['analysis']['elevation_auto']['registration']['ground_region']
    images = [cv2.imdecode(np.frombuffer((source/'photos'/f'{r}.png').read_bytes(), np.uint8), 1)
              for r in ['left', 'right']]
    rows = []
    for preprocessing in ['low_light', 'none']:
        settings = _analysis_config(manifest['analysis'])
        settings['stereo_matching']['preprocessing'] = preprocessing
        depth = _compute_stereo_depth(*images, calibration, settings)
        group = {'depth': depth, 'pair_healthy': True}
        before = old.check_ground(group, calibration, region)
        after = check_ground(group, calibration, region)
        assert before['policy_sha256'] == after['policy_sha256'] == POLICY_SHA256
        rows.append({'preprocessing': preprocessing, 'depth_audit': depth.audit,
                     'before': compact(before), 'after': compact(after),
                     'gui_detail_lines': ground_summary_lines(after)})
        print(preprocessing, after['status'], after['left_plane']['holdout_inlier_fraction'], flush=True)
    assert hashes == {str(p.relative_to(ROOT)): sha(p) for p in protected}
    result = {'scope': 'SAME_CAPTURE_CONFIGURATION_DIAGNOSTIC_NOT_METRIC_VALIDATION',
              'new_capture': False, 'source_sha256': hashes,
              'baseline_commit': 'e663f6e', 'source_inputs_unchanged': True,
              'threshold_policy_unchanged': True, 'rows': rows,
              'interpretation': 'None improves depth and plane support in this normally lit pair; both configurations still fail. No source settings changed.'}
    Path(__file__).with_name('ground-evidence.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
