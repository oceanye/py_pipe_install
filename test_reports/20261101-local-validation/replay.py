"""Replay an archived stereo manifest through production without editing it.

Default invocation only validates inputs and prints a plan. Pass --execute
after other CPU-heavy checks have finished. No camera or external AI is used.
All new artifacts stay below outputs/local-validation-20261101/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DEFAULT_SOURCE = ROOT / 'outputs/installation-assist-20261031/scenes/elevation-20261031-071031-189194c4/manifest.json'
OUT = ROOT / 'outputs/local-validation-20261101'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.resolve().relative_to(ROOT).as_posix()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def prepare(source):
    from pipe_twin.elevation_dataset import load_elevation_dataset
    loaded = load_elevation_dataset(source)
    manifest = loaded['manifest']
    protected = {source}
    for path in (source.with_name('report.json'),
                 ROOT / 'outputs/measurement_workbench/calibration_current.json',
                 ROOT / 'outputs/measurement_workbench/workbench_profile.json',
                 ROOT / 'outputs/elevation_workbench/last_scene.json'):
        if path.is_file():
            protected.add(path)
    model = source.parent / manifest['model']['path']
    assert sha(model) == manifest['model']['sha256']
    protected.add(model)
    for group in manifest['capture']['capture_groups']:
        for record in (group, *group.get('temporal_frames', [])):
            for view in record['views'].values():
                path = source.parent / view['path']
                assert sha(path) == view['sha256'], str(path)
                protected.add(path)
    # Also protect the original 18:32 archive and original active configuration
    # recorded by the earlier replay; do not depend on its inference outputs.
    earlier = ROOT / 'test_reports/20261031-installation-assist/evidence.json'
    if earlier.is_file():
        for name in read(earlier).get('source_sha256', {}):
            path = ROOT / name
            if path.is_file():
                protected.add(path)
    snapshot = {relative(path): sha(path) for path in sorted(protected)}
    latest = manifest['capture']['capture_groups'][-1]
    plan = {
        'source_manifest': relative(source),
        'source_capture_id': latest['capture_id'],
        'photo_sha256': {role: latest['views'][role]['sha256'] for role in ('left', 'right')},
        'captured_at': {role: latest['views'][role]['captured_at'] for role in ('left', 'right')},
        'requested_temporal_pairs': len(latest.get('temporal_frames', [])),
        'calibration_id': loaded['calibration']['calibration_id'],
        'protected_file_count': len(snapshot),
        'new_capture': False,
        'external_ai_called': False,
    }
    return plan, snapshot


def summary(report, old_report):
    section = report.get('section_evaluation') or {}
    old_section = (old_report or {}).get('section_evaluation') or {}
    def sizes(value):
        return {item['id']: item.get('diameter_mm')
                for item in (value.get('model') or {}).get('candidates', [])}
    previous, current = sizes(old_section), sizes(section)
    fusion = report.get('temporal_fusion') or {}
    assist = report.get('installation_assist') or {}
    return {
        'strict_counts': report.get('counts'),
        'strict_position_counts': (report.get('position_assessment') or {}).get('counts'),
        'temporal_fusion': {key: fusion.get(key) for key in
                            ('status', 'requested_frames', 'accepted_frames', 'reason_codes', 'frames')},
        'installation_assist': {key: assist.get(key) for key in
                                ('status', 'counts', 'reason_codes', 'ranked_hypotheses')},
        'historical_assist_counts_for_comparison_only': (old_report.get('installation_assist') or {}).get('counts'),
        'control_section': report.get('control_section'),
        'conditional_section_count': section.get('conditional_section_count'),
        'free_diameter_mm_by_candidate': current,
        'diameter_delta_from_archived_report_mm': {
            key: value - previous[key] for key, value in current.items()
            if isinstance(value, (int, float)) and isinstance(previous.get(key), (int, float))},
        'ground_check': report.get('ground_check'),
        'ai_section_namespace_present': 'ai_section_assist' in report,
        'ai_section_status': (report.get('ai_section_assist') or {}).get('status', 'NOT_RUN'),
        'production_reference_pair_diagnostics_present': bool(report.get('reference_pair_diagnostics')),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--execute', action='store_true', help='Run the CPU-heavy production replay.')
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.is_relative_to(ROOT):
        raise ValueError('Use an archived manifest inside this workspace.')
    plan, before = prepare(source)
    if not args.execute:
        print(json.dumps({'mode': 'PREPARED_ONLY', **plan}, ensure_ascii=False, indent=2))
        return
    import cv2
    from pipe_twin.stereo_analyzer import analyze_stereo_capture
    cv2.setNumThreads(2)
    run = OUT / f'replay-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}'
    run.mkdir(parents=True, exist_ok=False)
    write(run / 'input-protection.json', before)
    old_report = read(source.with_name('report.json')) if source.with_name('report.json').is_file() else {}
    evidence = {
        **plan, 'started_at': datetime.now(timezone.utc).isoformat(),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'run_directory': relative(run), 'scope': 'ARCHIVED_INPUT_PRODUCTION_REPLAY',
        'original_manifest_used_without_rewriting': True,
    }
    start = time.monotonic()
    error = None
    try:
        report = analyze_stereo_capture(source, report_output_path=run / 'report.json',
                                       evidence_dir=run / 'evidence',
                                       progress=lambda message: print(message, flush=True))
        evidence.update(status='COMPLETED', report=relative(run / 'report.json'),
                        report_sha256=sha(run / 'report.json'), result=summary(report, old_report))
    except Exception as exc:
        error = exc
        evidence.update(status='FAILED', exception_type=type(exc).__name__, exception=str(exc),
                        traceback=traceback.format_exc())
    finally:
        changes = [name for name, digest in before.items()
                   if not (ROOT / name).is_file() or sha(ROOT / name) != digest]
        evidence.update(seconds=time.monotonic() - start, sources_unchanged=not changes,
                        changed_sources=changes, finished_at=datetime.now(timezone.utc).isoformat())
        write(run / 'replay-evidence.json', evidence)
        # This pointer belongs only to the isolated validation output directory.
        write(OUT / 'latest-replay.json', {'evidence': relative(run / 'replay-evidence.json'),
                                        'report': evidence.get('report')})
        print('EVIDENCE', run / 'replay-evidence.json', flush=True)
        print(json.dumps({key: value for key, value in evidence.items() if key != 'result'},
                         ensure_ascii=False, indent=2), flush=True)
        assert not changes, f'Protected inputs changed: {changes}'
    if error is not None:
        raise error


if __name__ == '__main__':
    main()
