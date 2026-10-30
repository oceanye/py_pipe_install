"""Capture isolated smoke-test evidence; never change the operator scene/profile."""
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import sys
import time

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.capture_gui import load_calibration_json
from pipe_twin.calibration_wizard import rectifier_for_calibration
from pipe_twin.stereo_camera import StereoCameraSession


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    profile_path = ROOT/'outputs/measurement_workbench/workbench_profile.json'
    calibration_path = ROOT/'outputs/measurement_workbench/calibration_current.json'
    scene = ROOT/'outputs/elevation_workbench/captures/elevation-20261030-183635-b6b7b875'
    protected = [profile_path, calibration_path, scene/'manifest.json', scene/'report.json',
                 ROOT/'outputs/elevation_workbench/last_scene.json']
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    profile = json.loads(profile_path.read_text(encoding='utf-8-sig'))
    calibration = load_calibration_json(calibration_path)
    rectifier = rectifier_for_calibration(calibration, profile)
    manifest = json.loads((scene/'manifest.json').read_text(encoding='utf-8'))
    camera = profile['camera']
    output = ROOT/'outputs/ground-monitor-20261030'/('live-'+time.strftime('%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    started = time.monotonic()
    from pipe_twin.stereo_analyzer import _calibration_from_manifest
    parsed = _calibration_from_manifest(calibration)
    with StereoCameraSession(layout=camera['layout'], left_index=camera['left_index'],
                             right_index=camera.get('right_index'), eye_width=parsed.left.width,
                             eye_height=parsed.left.height, exposure_ms=camera.get('exposure_ms')) as session:
        for _ in range(10):
            session.read_pair()
        for sequence in range(1, 5):
            pair = session.read_pair()
            if rectifier:
                pair = replace(pair, left=rectifier.rectify('left', pair.left),
                               right=rectifier.rectify('right', pair.right))
            row = {'sequence': sequence, 'sync_delta_ms': pair.sync_delta_ms,
                   'capture_times': {'left': pair.left_captured_at, 'right': pair.right_captured_at},
                   'timestamp_source': pair.timestamp_source, 'provenance': pair.provenance, 'images': {}}
            for role in ('left', 'right'):
                image = getattr(pair, role)
                path = output/f'{sequence:03d}_{role}.png'
                assert cv2.imwrite(str(path), image)
                row['images'][role] = {'path': str(path.relative_to(ROOT)), 'sha256': sha(path),
                                       'shape': list(image.shape), 'mean': float(image.mean()),
                                       'std': float(image.std())}
            rows.append(row)
            print('pair', sequence, row['capture_times'], flush=True)
    assert before == {str(p.relative_to(ROOT)): sha(p) for p in protected}
    result = {'scope': 'ISOLATED_NEW_CAPTURE_NOT_OPERATOR_ASSESSMENT', 'camera': camera,
              'calibration': calibration, 'rectification_recipe': rectifier.recipe if rectifier else None,
              'analysis_settings': manifest['analysis'],
              'region': manifest['analysis']['elevation_auto']['registration']['ground_region'],
              'region_origin': 'PREVIOUS_SCENE_PIXEL_ROI_RECHECK_REQUIRED',
              'frames': rows, 'seconds': time.monotonic()-started,
              'source_unchanged': True, 'protected_source_sha256': before, 'camera_released': True}
    path = output/'capture.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(path, flush=True)


if __name__ == '__main__':
    main()
