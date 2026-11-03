"""Replay the failed 11:07 scene; preserve originals and never open a camera.

Use --remote-gray to additionally submit an independent raw-gray variant to the
configured server. This is an explicit diagnostic variation, not an automatic
change to the user's scene, profile or preprocessing selection.
"""
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
from pipe_twin.analysis_runtime_settings import load_runtime_settings
from pipe_twin.analysis_transport import RemoteAnalysisClient
from pipe_twin.section_status import solver_status_rows
from pipe_twin.stereo_analyzer import analyze_stereo_capture


def hashes(path):
    return {str(item.relative_to(path)): hashlib.sha256(item.read_bytes()).hexdigest()
            for item in path.rglob('*') if item.is_file()}


def preprocessing_backends(value):
    if isinstance(value, dict):
        found = {value['preprocessing_backend']} if 'preprocessing_backend' in value else set()
        return found.union(*(preprocessing_backends(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(preprocessing_backends(item) for item in value))
    return set()


def main():
    cv2.setNumThreads(2)
    source = ROOT / 'outputs/elevation_workbench/captures/elevation-20261103-110710-5b8316da'
    before = hashes(source)
    output = ROOT / 'outputs' / ('oc' + datetime.now().strftime('%H%M%S'))
    evidence = {'source': str(source), 'camera_opened': False, 'cases': [], 'original_sha256': before}
    cases = ['remote-gray'] if '--remote-gray' in sys.argv else []
    cases += ['local-nlm', 'local-missing-photo']
    for name in cases:
        folder = output / name
        shutil.copytree(source, folder)
        path = folder / 'manifest.json'
        if name == 'remote-gray':
            manifest = json.loads(path.read_text(encoding='utf-8'))
            manifest['analysis']['stereo_matching']['preprocessing'] = 'none'
            path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
            client = RemoteAnalysisClient(load_runtime_settings().base_url, timeout=30)
            evidence['server_health_before'] = client.health()
            analyze = client.analyze
        else:
            analyze = analyze_stereo_capture
        start = time.monotonic()
        print('START ' + name, flush=True)
        with mock.patch.object(cv2, 'fastNlMeansDenoising', None) if name == 'local-missing-photo' else nullcontext():
            report = analyze(path, report_output_path=folder / 'report.json', evidence_dir=folder / 'evidence',
                             progress=lambda message: print(name + ': ' + str(message), flush=True))
        observations = report.get('installation_assist', {}).get('observations', [])
        item = {'name': name, 'manifest': str(path), 'seconds': round(time.monotonic() - start, 3),
                'preprocessing_backends': sorted(preprocessing_backends(report)),
                'strict_counts': report.get('counts'),
                'observations': [{key: obs.get(key) for key in ('label', 'suggested_pipe_id', 'status')}
                                 for obs in observations],
                'sections': [{key: row.get(key) for key in ('label', 'diameter_mm', 'solved')}
                             for row in solver_status_rows(report)]}
        evidence['cases'].append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        evidence['source_unchanged'] = before == hashes(source)
        assert evidence['source_unchanged']
        (Path(__file__).parent / 'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
