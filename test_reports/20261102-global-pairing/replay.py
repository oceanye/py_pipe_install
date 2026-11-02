"""Replay saved stereo images into a separate, GUI-openable scene; never capture."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipe_twin.elevation_auto import analyze_elevation_auto_manifest


def hashes(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob('*') if p.is_file()}


def summary(report):
    stage = report.get('section_evaluation') or {}
    assist = report.get('installation_assist') or {}
    return {
        'generated_at': report.get('generated_at'),
        'capture_frames': {k: (report.get('temporal_fusion') or {}).get(k)
                           for k in ('status', 'requested_frames', 'accepted_frames')},
        'ground': {k: (report.get('ground_check') or {}).get(k)
                   for k in ('status', 'correction_status')},
        'pairing': (stage.get('pairing_audit') or {}).get('global_assignment'),
        'sections': [{k: r.get(k) for k in ('candidate_id', 'label', 'solved', 'diameter_mm',
                                           'temporal_consistency', 'reason_codes')}
                     for r in report.get('solver_status', [])],
        'pairwise': (stage.get('model') or {}).get('pairs', []),
        'assist': {k: assist.get(k) for k in ('revision', 'status', 'method', 'position_source',
                  'policy', 'counts', 'observations', 'plausible_hypothesis_count',
                  'ranked_hypotheses', 'reason_codes')},
        'strict_counts': report.get('counts'),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'outputs/elevation_workbench/captures/elevation-20261102-093519-f1d6650e')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, destination = args.source.resolve(), args.output.resolve()
    if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Replay destination must be separate from source')
    # Refuse to overwrite an earlier replay so comparison evidence stays reviewable.
    destination.mkdir(parents=True, exist_ok=False)
    before = hashes(source)
    original_report = json.loads((source / 'report.json').read_text(encoding='utf-8'))
    for name in ('photos', 'burst', 'model'):
        shutil.copytree(source / name, destination / name)
    shutil.copy2(source / 'manifest.json', destination / 'manifest.json')
    # The historical AI answer is bound to its old evidence. Do not carry it into
    # a replay with changed geometry or silently request a replacement answer.
    start = time.monotonic()
    result = analyze_elevation_auto_manifest(destination / 'manifest.json',
        report_output_path=destination / 'report.json', evidence_dir=destination / 'evidence',
        progress=lambda message: print(message, flush=True))
    elapsed = time.monotonic() - start
    if hashes(source) != before:
        raise AssertionError('Original scene changed during replay')
    evidence = {'scope': 'SAME_SAVED_STEREO_BURST_REPLAY_NOT_NEW_CAPTURE',
                'source': str(source), 'output': str(destination), 'source_sha256': before,
                'source_unchanged': True, 'elapsed_seconds': elapsed,
                'before': summary(original_report), 'after': summary(result)}
    target = destination.parent / (destination.name + '-comparison.json')
    target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'comparison': str(target), 'manifest': str(destination / 'manifest.json'),
                     'elapsed_seconds': elapsed, 'assist_status': result['installation_assist']['status'],
                     'counts': result['installation_assist']['counts']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
