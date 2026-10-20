"""Summarize the saved Oct 20 comparisons without modifying capture packages."""
from pathlib import Path
import hashlib
import json
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/improvement-20261020'
sys.path.insert(0, str(ROOT))
from pipe_twin.elevation_gui import recognition_diagnostic_lines


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw_path = OUT / 'raw-burst/raw-manifest.json'
    raw = json.loads(raw_path.read_text(encoding='utf-8'))
    for frame in raw['frames']:
        for role in ('left', 'right'):
            assert sha(raw_path.parent / frame['files'][role]) == frame['sha256'][role]
    calibration_path = ROOT / 'outputs/measurement_workbench/calibration_current.json'
    calibration = json.loads(calibration_path.read_text(encoding='utf-8'))['stereo_calibration']
    evidence = {
        'scope': 'REAL_RECAPTURE_AND_IMAGE_PROPOSAL_FIX_NOT_METRIC_VALIDATION',
        'raw_manifest': str(raw_path.relative_to(ROOT)), 'raw_manifest_sha256': sha(raw_path),
        'raw_capture': raw, 'known_physical_baseline_mm': 60.,
        'calibration': {'id': calibration['calibration_id'], 'baseline_mm': calibration['baseline_mm'],
                        'validated': calibration['validated'], 'sha256': sha(calibration_path)},
        'source_sha256': {name: sha(ROOT / name) for name in (
            'pipe_twin/image_regions.py', 'pipe_twin/stereo_correspondence.py', 'pipe_twin/elevation_gui.py')},
        'runs': []}
    for tag, variants in (('baseline', ['single', 'burst']), ('none256', ['single']),
                          ('none384', ['single']), ('fixed', ['single']), ('final', ['single', 'burst'])):
        out = OUT / ('active-' + tag)
        summary = json.loads((out / 'summary.json').read_text(encoding='utf-8'))
        assert summary['active_calibration_sha256_unchanged'] == sha(calibration_path)
        for variant in variants:
            saved = json.loads((out / (variant + '-summary.json')).read_text(encoding='utf-8'))
            report_path = ROOT / saved['report']
            manifest_path = ROOT / saved['manifest']
            report = json.loads(report_path.read_text(encoding='utf-8'))
            manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
            surface = report['local_surface']
            candidates = surface['image_candidates']
            item = {'tag': tag, 'variant': variant, 'manifest': saved['manifest'], 'report': saved['report'],
                    'report_sha256': sha(report_path), 'manifest_sha256': sha(manifest_path),
                    'stereo_matching': manifest['analysis']['stereo_matching'],
                    'model': {k: v for k, v in manifest['model'].items() if k != 'pipes'},
                    'counts': report['counts'], 'observation_count': len(surface['observations']),
                    'candidate_counts': dict(Counter(c['role'] for c in candidates)),
                    'image_proposals': surface['audit']['image_proposals'],
                    'blue_candidates': [c for c in candidates if c['appearance'] == 'hue-120'],
                    'temporal_fusion': report.get('temporal_fusion')}
            evidence['runs'].append(item)
            if tag == 'final' and variant == 'burst':
                (OUT / 'gui-diagnostics-final.txt').write_text(
                    '\n'.join(recognition_diagnostic_lines(report)), encoding='utf-8')
    (Path(__file__).parent / 'evidence.json').write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for item in evidence['runs']:
        print(item['tag'], item['variant'], item['candidate_counts'], item['counts'])


if __name__ == '__main__':
    main()
