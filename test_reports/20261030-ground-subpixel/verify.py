"""Replay independent per-pair ground checks; never promote a rejected burst."""
from pathlib import Path
import hashlib
import json
import sys
import time
from datetime import datetime

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.ground_check import check_ground, POLICY_SHA256
from pipe_twin.stereo_analyzer import _analysis_config, _calibration_from_manifest, _compute_stereo_depth, _quality


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(check):
    fields = ['status', 'valid_depth_fraction', 'estimated_refined_depth_fraction', 'right_confirmed_fraction',
              'left_right_normal_angle_deg', 'left_right_plane_offset_difference_mm',
              'camera_to_ground_perpendicular_mm', 'optical_axis_to_ground_deg', 'image_roll_relative_to_ground_deg',
              'subpixel_refinement']
    result = {key: check.get(key) for key in fields}
    for eye in ['raw_depth_plane', 'left_plane', 'right_plane']:
        plane = check.get(eye) or {}
        result[eye] = {key: plane.get(key) for key in ['status', 'sample_count', 'fit_scope',
            'training_sample_count', 'holdout_sample_count', 'inlier_fraction', 'holdout_inlier_fraction',
            'holdout_rms_mm', 'normal_camera', 'offset_mm', 'failure_reasons']}
    result['support'] = {key: check.get('observed_support', {}).get(key)
                         for key in ['sampling', 'pixel_count', 'valid_pixel_count', 'inferred_pixels']}
    return result


def main():
    started = time.monotonic()
    protected = [ROOT/'outputs/measurement_workbench/calibration_current.json',
                 ROOT/'outputs/elevation_workbench/last_scene.json']
    scenes = ['elevation-20261030-183635-b6b7b875', 'elevation-20261030-161444-3ab7011e']
    for name in scenes:
        protected.extend(ROOT/'outputs/elevation_workbench/captures'/name/file for file in ['manifest.json', 'report.json'])
    original = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    rows = []
    for name in scenes:
        source = ROOT/'outputs/elevation_workbench/captures'/name
        manifest = json.loads((source/'manifest.json').read_text(encoding='utf8'))
        calibration = _calibration_from_manifest(manifest['stereo_calibration'])
        config = _analysis_config(manifest['analysis'])
        region = manifest['analysis']['elevation_auto']['registration']['ground_region']
        group = manifest['capture']['capture_groups'][-1]
        frames = group.get('temporal_frames') or [group]
        for index, frame in enumerate(frames):
            images, hashes = {}, {}
            for role in ['left', 'right']:
                path = source/frame['views'][role]['path']
                hashes[role] = sha(path)
                assert hashes[role] == frame['views'][role]['sha256']
                images[role] = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), 1)
            depth = _compute_stereo_depth(images['left'], images['right'], calibration, config)
            # The individual images are checked separately here. This does NOT
            # change the original temporal rejection or make an installed claim.
            delta = abs((datetime.fromisoformat(frame['views']['left']['captured_at'])-
                         datetime.fromisoformat(frame['views']['right']['captured_at'])).total_seconds()*1000)
            healthy = (depth.audit['status'] == 'VALID' and delta <= calibration.max_sync_delta_ms
                       and all(_quality(image, config)['passed'] for image in images.values()))
            check = check_ground({'depth': depth, 'pair_healthy': healthy, **images}, calibration, region)
            assert check['policy_sha256'] == POLICY_SHA256
            rows.append({'scene': name, 'frame': index+1,
                         'captured_at': frame['views']['left']['captured_at'], 'photo_sha256': hashes,
                         'individual_ground': summary(check)})
            print(name, index+1, check['status'], flush=True)
    assert original == {str(p.relative_to(ROOT)): sha(p) for p in protected}
    result = {'scope': 'INDEPENDENT_SINGLE_PAIR_GROUND_REPLAY_NOT_BURST_ACCEPTANCE',
              'new_capture': False, 'source_files_unchanged': True, 'source_sha256': original,
              'policy_sha256': POLICY_SHA256, 'plane_thresholds_unchanged': True,
              'elapsed_seconds': time.monotonic()-started, 'rows': rows}
    Path(__file__).with_name('evidence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2,
                                                               allow_nan=False)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
