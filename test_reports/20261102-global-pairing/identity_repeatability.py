"""Replay identity matching independently on the production section samples.

Uses saved rectified photographs only. The production ground result is frozen;
no camera, network, AI request, calibration solve or source overwrite occurs.
The reference result supplies comparison identities, not fit/matching inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipe_twin.elevation_dataset import _calibration_ok
from pipe_twin.installation_assist import analyze_installation_assist
from pipe_twin.pooled_sections import analyze_pooled_sections
from pipe_twin.temporal_sections import sample_indices


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=ROOT/'outputs/global-pairing-20261102/scene-verified')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/global-pairing-20261102/identity-repeatability.json')
    args = parser.parse_args()
    scene = args.scene.resolve()
    manifest_path, report_path = scene/'manifest.json', scene/'report.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    report = json.loads(report_path.read_text(encoding='utf8'))
    protected = {str(p): digest(p) for p in (manifest_path, report_path)}
    calibration = _calibration_ok(manifest['stereo_calibration'])
    settings = manifest['analysis']['elevation_auto']['registration']
    specs = manifest['analysis']['elevation_auto']['pipes']
    matcher = manifest['analysis']['stereo_matching']
    disparity_min = float(matcher.get('min_disparity', 0))
    disparity_max = disparity_min+float(matcher.get('num_disparities', 256))-1
    ground = dict(report['ground_check'])
    if settings.get('ground_candidate_trial'):
        ground['candidate_trial_enabled'] = True
    assert ground['status'] == 'INSUFFICIENT_GROUND_SUBPIXEL_SUPPORT'
    frames = manifest['capture']['capture_groups'][-1]['temporal_frames']
    selected = [frames[i] for i in sample_indices(len(frames))]
    expected = report['section_evaluation']['temporal_sections']['sources']
    assert [f['capture_id'] for f in selected] == [f['capture_id'] for f in expected]
    reference_likely = sorted(p['pipe_id'] for p in report['installation_assist']['pipes'] if p['status'] == 'LIKELY')
    reference_mapping = {o['observation_id']: o['suggested_pipe_id']
                         for o in report['installation_assist']['observations'] if o.get('suggested_pipe_id')}
    rows = []
    for index, (frame, source) in enumerate(zip(selected, expected), 1):
        images, hashes = {}, {}
        for eye in ('left', 'right'):
            path = scene/frame['views'][eye]['path']
            hashes[eye] = digest(path)
            assert hashes[eye] == frame['views'][eye]['sha256'] == source['sha256'][eye]
            protected[str(path)] = hashes[eye]
            images[eye] = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
            assert images[eye] is not None and images[eye].shape[:2] == (1080, 1920)
        stage = analyze_pooled_sections(images['left'], images['right'], calibration,
            aggregation_band=settings.get('aggregation_band'), disparity_range=(disparity_min, disparity_max),
            ground_result=ground, refine_epipolar=True)
        assert not stage['ground_correction']['applied']
        # The frames belong to the same production-validated 10/10 static
        # sequence. Each section and identity is nevertheless solved anew.
        assistance = analyze_installation_assist({'section_evaluation': stage,
            'temporal_fusion': report['temporal_fusion']}, specs, settings, calibration=calibration)
        likely = sorted(p['pipe_id'] for p in assistance['pipes'] if p['status'] == 'LIKELY')
        best = (assistance.get('ranked_hypotheses') or [{}])[0]
        mapping = {o['observation_id']: o['suggested_pipe_id'] for o in assistance['observations']
                   if o.get('suggested_pipe_id')}
        row = {'capture_id': frame['capture_id'], 'sha256': hashes,
            'captured_at': frame['views']['left']['captured_at'],
            'conditional_section_count': stage.get('conditional_section_count'),
            'diameters': [{'candidate_id': c['candidate_id'],
                'diameter_mm': c['fit'].get('trial_diameter_mm'), 'fit_status': c['fit']['status'],
                'reason_codes': c['reason_codes']} for c in stage['candidates']],
            'status': assistance['status'], 'plausible_hypothesis_count': assistance.get('plausible_hypothesis_count', 0),
            'likely_pipe_ids': likely, 'matches_reference_identity_set': likely == reference_likely,
            'observation_to_pipe_id': mapping, 'matches_reference_correspondences': mapping == reference_mapping,
            'best_maximum_pairwise_relative_difference': best.get('maximum_pairwise_relative_difference'),
            'best_pairwise_distances': best.get('pairwise_distances', []),
            'section_pairs': (stage.get('model') or {}).get('pairs', []),
            'assistance': assistance}
        rows.append(row)
        print(json.dumps({'frame': index, 'capture_id': frame['capture_id'], 'likely': likely,
            'plausible_hypothesis_count': row['plausible_hypothesis_count'],
            'maximum_pairwise_relative_difference': row['best_maximum_pairwise_relative_difference']}, ensure_ascii=False), flush=True)
    by_pair = defaultdict(list)
    for row in rows:
        for pair in row['best_pairwise_distances']:
            by_pair[tuple(pair['observation_ids'])].append(pair)
    summaries = []
    for identities, pairs in sorted(by_pair.items()):
        distances = [p['estimated_center_distance_mm'] for p in pairs]
        design = pairs[0]['design_center_distance_mm']
        summaries.append({'observation_ids': list(identities), 'frame_count': len(pairs),
            'design_center_distance_mm': design, 'minimum_mm': min(distances), 'maximum_mm': max(distances),
            'median_mm': float(np.median(distances)), 'span_mm': max(distances)-min(distances),
            'span_over_design_distance': (max(distances)-min(distances))/design,
            'maximum_frame_relative_difference_from_design': max(p['relative_difference'] for p in pairs)})
    assert all(digest(Path(path)) == value for path, value in protected.items())
    identities = Counter(tuple(r['likely_pipe_ids']) for r in rows)
    result = {'scope': 'SAME_BURST_INDEPENDENT_IDENTITY_REPEATABILITY_NOT_ABSOLUTE_ACCURACY',
        'generated_at': datetime.now(timezone.utc).isoformat(), 'scene': str(scene),
        'dataset_id': manifest['dataset_id'], 'source_hashes': protected,
        'source_unchanged': True, 'photographs_captured_this_run': False, 'ai_requested': False,
        'ground_frozen_status': ground['status'], 'ground_correction_applied': False,
        'available_stereo_pairs': len(frames), 'independently_evaluated_pairs': len(rows),
        'reference_likely_pipe_ids': reference_likely,
        'reference_observation_to_pipe_id': reference_mapping,
        'matching_reference_identity_frames': sum(r['matches_reference_identity_set'] for r in rows),
        'matching_reference_correspondence_frames': sum(r['matches_reference_correspondences'] for r in rows),
        'identity_sets': [{'pipe_ids': list(k), 'frame_count': v} for k, v in sorted(identities.items())],
        'pairwise_repeatability': summaries, 'frames': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print(json.dumps({'output': str(args.output), 'identity_sets': result['identity_sets'],
        'pairwise_repeatability': summaries}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
