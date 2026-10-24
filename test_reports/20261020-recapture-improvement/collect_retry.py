"""Summarize the 13:15 retry using unchanged production code and parameters."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/retry-20261020-131151'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw_path = OUT / 'raw-burst/raw-manifest.json'
    raw = json.loads(raw_path.read_text(encoding='utf-8'))
    for frame in raw['frames']:
        for role in ('left', 'right'):
            assert sha(raw_path.parent / frame['files'][role]) == frame['sha256'][role]
    calibration = ROOT / 'outputs/measurement_workbench/calibration_current.json'
    previous = json.loads((Path(__file__).parent / 'evidence.json').read_text(encoding='utf-8'))
    assert sha(calibration) == previous['calibration']['sha256']
    for name, digest in previous['source_sha256'].items():
        assert sha(ROOT / name) == digest
    evidence = {'scope': 'REAL_RETRY_NOT_METRIC_VALIDATION', 'production_commit': '5afe13d',
                'production_sources_unchanged': previous['source_sha256'],
                'calibration_unchanged': previous['calibration'],
                'raw_manifest_sha256': sha(raw_path), 'raw_capture': raw, 'runs': []}
    for variant in ('single', 'burst'):
        summary = json.loads((OUT / 'active-retry' / (variant + '-summary.json')).read_text(encoding='utf-8'))
        report_path = ROOT / summary['report']
        report = json.loads(report_path.read_text(encoding='utf-8'))
        manifest = json.loads((ROOT / summary['manifest']).read_text(encoding='utf-8'))
        assert manifest['analysis']['stereo_matching'] == previous['runs'][-1]['stereo_matching']
        temporal = report.get('temporal_fusion')
        candidates = []
        for c in report['local_surface']['image_candidates']:
            candidates.append({key: c.get(key) for key in (
                'candidate_id', 'role', 'appearance', 'region_px', 'valid_depth_fraction',
                'axis_to_epipolar_deg', 'correspondence_diagnostic', 'reason_codes')})
        evidence['runs'].append({'variant': variant, 'manifest': summary['manifest'],
            'report': summary['report'], 'report_sha256': sha(report_path),
            'counts': report['counts'], 'observations': report['local_surface']['observations'],
            'stereo_matching': manifest['analysis']['stereo_matching'],
            'image_proposals': report['local_surface']['audit']['image_proposals'],
            'candidates': candidates,
            'temporal_fusion': {k: v for k, v in temporal.items() if k != 'frames'} if temporal else None})
    evidence['right_candidate_fill_probe'] = json.loads((OUT / 'proposal-probe.json').read_text(encoding='utf-8'))
    path = Path(__file__).parent / 'retry-evidence.json'
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('raw hashes, active calibration, production source and matcher settings verified unchanged')
    for run in evidence['runs']:
        print(run['variant'], run['counts'], 'observations', len(run['observations']))


if __name__ == '__main__':
    main()
