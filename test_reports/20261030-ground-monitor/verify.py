"""Check physical-axis geometry and rejection using real archived image pairs."""
from pathlib import Path
from datetime import datetime, timedelta, timezone
import hashlib
import json
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.ground_monitor import inspect_live_ground, GroundMonitorState


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact(result):
    check = result['ground_check']
    return {k: result[k] for k in ('sequence', 'captured_at', 'generated_at', 'config_signature',
                                  'sources', 'sync_delta_ms', 'status', 'pose', 'valid', 'quality')} | {
        'monitor_quality': result.get('monitor_quality'),
        'ground': {k: check.get(k) for k in ('status', 'left_right_normal_angle_deg',
             'left_right_plane_offset_difference_mm', 'calibration_validated')},
        'support': {k: check.get('observed_support', {}).get(k)
                    for k in ('pixel_count', 'valid_pixel_count', 'inferred_pixels')},
    }


def main():
    source = ROOT/'outputs/elevation_workbench/captures/elevation-20261030-183635-b6b7b875'
    capture_path = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT/'outputs/ground-monitor-20261030/live-232451/capture.json')
    captured = json.loads(capture_path.read_text(encoding='utf-8'))
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    profile_path = ROOT/'outputs/measurement_workbench/workbench_profile.json'
    profile = json.loads(profile_path.read_text(encoding='utf-8-sig'))
    protected = [source/'manifest.json', source/'report.json', profile_path,
                 ROOT/'outputs/measurement_workbench/calibration_current.json',
                 ROOT/'outputs/elevation_workbench/last_scene.json']
    protected += list((source/'photos').rglob('*.png'))
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    group = manifest['capture']['capture_groups'][-1]
    views = group['views']
    images = {r: cv2.imread(str(source/views[r]['path'])) for r in ('left', 'right')}
    times = {r: views[r]['captured_at'] for r in ('left', 'right')}
    start = time.monotonic()
    result = inspect_live_ground(sequence=1, left=images['left'], right=images['right'], capture_times=times,
        calibration=manifest['stereo_calibration'], analysis_settings=manifest['analysis'],
        region=manifest['analysis']['elevation_auto']['registration']['ground_region'],
        rectification_recipe=profile['rectification_recipe'])
    seconds = time.monotonic()-start
    assert result['valid'], result['status']
    replay_path = ROOT/'outputs/ground-monitor-20261030/historical-monitor.json'
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    assert result['pose']['coordinate_frame'] == 'RAW_LEFT_OPTICAL'
    recipe = profile['rectification_recipe']
    for eye, matrix, key in [('left', 'R1', 'off_normal_deg'), ('right', 'R2', 'right_off_normal_deg')]:
        normal = np.array(result['ground_check'][eye+'_plane']['normal_camera'])
        physical = np.array(recipe[matrix]).T@normal
        expected = float(np.degrees(np.arccos(abs(physical[2])/np.linalg.norm(physical))))
        assert abs(result['pose'][key]-expected) < 1e-8
    state = GroundMonitorState()
    replay_now = datetime.fromisoformat(result['captured_at'])+timedelta(seconds=1)
    historical = state.update(result, now=replay_now)
    assert historical['status'] == 'ADJUST_TILT' and not historical['green']
    expired = state.snapshot(now=datetime.now(timezone.utc))
    assert expired['status'] == 'EXPIRED' and expired['pose'] is None and not expired['green']
    print('historical', result['pose'], 'seconds', seconds, flush=True)
    failures = []
    for f in captured['frames']:
        loaded = {}
        for role, item in f['images'].items():
            path = ROOT/item['path']
            assert sha(path) == item['sha256']
            loaded[role] = cv2.imread(str(path))
        start = time.monotonic()
        dark = inspect_live_ground(sequence=f['sequence'], left=loaded['left'], right=loaded['right'],
            capture_times=f['capture_times'], calibration=captured['calibration'], region=captured['region'],
            analysis_settings=captured['analysis_settings'], rectification_recipe=captured['rectification_recipe'])
        assert dark['status'] == 'NEAR_BLACK_FRAME' and dark['pose'] is None and not dark['valid']
        assert dark['depth_audit'] is None  # Quality failure must skip expensive matching.
        failures.append(compact(dark) | {'seconds': time.monotonic()-start})
    assert before == {str(p.relative_to(ROOT)): sha(p) for p in protected}
    evidence = {'scope': 'ARCHIVED_GEOMETRY_REPLAY_AND_NEW_DARK_CAPTURE_REJECTION',
        'historical': compact(result), 'historical_analysis_seconds': seconds,
        'historical_state_at_capture_plus_one_second': historical,
        'historical_state_at_present_time': expired,
        'live_capture_metadata': str(capture_path.relative_to(ROOT)), 'dark_pairs': failures,
        'source_unchanged': True, 'protected_source_sha256': before,
        'field_alignment_success_demonstrated': False,
        'limitations': ['New live images are nearly black; cause not established.',
                        'Historical replay is not a current live angle or independent accuracy validation.']}
    Path(__file__).with_name('evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2,
                                                                allow_nan=False)+'\n', encoding='utf-8')
    print('dark pairs rejected', len(failures), 'sources unchanged', flush=True)


if __name__ == '__main__':
    main()
