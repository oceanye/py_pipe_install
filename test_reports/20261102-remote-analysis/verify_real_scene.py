"""Compare local and HTTP worker results on one frozen real stereo burst.

This verifies transport/process separation on loopback, not the speed or reachability
of a second physical computer. It never opens a camera or calls an AI provider.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipe_twin.ai_gui import load_bound_local_report
from pipe_twin.analysis_server import AnalysisServer
from pipe_twin.analysis_transport import RemoteAnalysisClient
from pipe_twin.stereo_analyzer import analyze_stereo_capture


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity(report):
    assist = report['installation_assist']
    return {'status': assist['status'], 'counts': assist['counts'],
            'plausible_hypotheses': assist.get('plausible_hypothesis_count'),
            'observations': {r['observation_id']: {
                'pipe_id': r.get('suggested_pipe_id'), 'status': r['status'],
                'diameter_mm': r.get('estimated_diameter_mm')}
                for r in assist['observations']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if source == output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Source and verification output must be separate')
    output.mkdir(parents=True, exist_ok=False)
    original = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    manifest = copy.deepcopy(original)
    manifest['capture']['capture_groups'] = [manifest['capture']['capture_groups'][-1]]
    group = manifest['capture']['capture_groups'][0]
    files = {manifest['model']['path']}
    for frame in [group, *group.get('temporal_frames', [])]:
        files.update(view['path'] for view in frame['views'].values())
    protected = {name: sha(source/name) for name in [*files, 'manifest.json', 'report.json']}
    scene = output/'client-scene'
    scene.mkdir()
    for name in files:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Expected portable source paths')
        (scene/relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/relative, scene/relative)
    path = scene/'manifest.json'
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('LOCAL ANALYSIS', flush=True)
    start = time.monotonic()
    local = analyze_stereo_capture(path, report_output_path=output/'local-report.json',
        evidence_dir=output/'local-evidence', progress=lambda value: print(value, flush=True))
    local_seconds = time.monotonic()-start
    print('HTTP WORKER ANALYSIS', flush=True)
    progress = []
    with AnalysisServer(bind='127.0.0.1', port=0, work_root=output/'server') as server:
        client = RemoteAnalysisClient(server.base_url, job_timeout=900)
        health = client.health()
        start = time.monotonic()
        def show(message):
            progress.append(message)
            print(message, flush=True)
        remote = client.analyze(path, report_output_path=scene/'report.json',
                                evidence_dir=scene/'evidence', progress=show)
        remote_seconds = time.monotonic()-start
        jobs = server.jobs()
    first, second = identity(local), identity(remote)
    assert first == second, (first, second)
    assert local['counts'] == remote['counts']
    assert load_bound_local_report(path) is not None
    assert all(sha(source/name) == digest for name, digest in protected.items())
    evidence_count = 0
    for evidence in remote.get('evidence_files', []):
        name = evidence['path'] if isinstance(evidence, dict) else evidence
        evidence_path = Path(name)
        if not evidence_path.is_absolute():
            evidence_path = scene / evidence_path
        assert evidence_path.is_file() and evidence_path.resolve().is_relative_to(scene), name
        if isinstance(evidence, dict) and evidence.get('sha256'):
            assert sha(evidence_path) == evidence['sha256'], name
        evidence_count += 1
    result = {'scope': 'LOOPBACK_HTTP_PROCESS_SEPARATION_NOT_REMOTE_HARDWARE_BENCHMARK',
        'source_dataset': manifest['dataset_id'], 'source_hashes': protected,
        'source_unchanged': True, 'current_groups': 1,
        'stereo_pairs': len(group.get('temporal_frames', [])) or 1,
        'local_seconds': local_seconds, 'http_seconds': remote_seconds,
        'local_remote_identity_equal': True, 'strict_counts_equal': True,
        'gui_bound_report_restore_passed': True, 'verified_evidence_files': evidence_count, 'identity': second,
        'server_health': health, 'server_jobs': jobs, 'progress_messages': progress,
        'new_camera_capture': False, 'ai_request': False}
    (output/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2,
        allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('local_seconds', 'http_seconds', 'identity')},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
