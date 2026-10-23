"""Standalone ground tilt inspection of the currently displayed stereo pair."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .elevation_dataset import _calibration_ok, _sync_delta_ms
from .ground_check import check_ground, normalize_ground_region
from .ground_visualization import ground_evidence_image
from .pipeline import atomic_write_bytes
from .stereo_analyzer import _analysis_config, _compute_stereo_depth, _quality


def inspect_ground_pair(*, paths, expected_hashes, capture_times, calibration,
                        region, analysis_settings, output_root, pair_confirmed):
    if not pair_confirmed:
        raise ValueError('请先确认左右图来自同一次同步拍摄')
    parsed = _calibration_ok(calibration)
    if not parsed.rectified:
        raise ValueError('倾角检查需要已矫正的双目照片')
    selection = normalize_ground_region(region, image_size=(parsed.left.width, parsed.left.height))
    if selection is None:
        raise ValueError('请先在左图框选真实地面')
    delta = _sync_delta_ms(capture_times['left'], capture_times['right'])
    if delta > parsed.max_sync_delta_ms+1e-6:
        raise ValueError('左右照片时间差超过标定允许值')
    config = _analysis_config(analysis_settings)
    images, quality, sources = {}, {}, {}
    for role in ('left', 'right'):
        path = Path(paths[role]).resolve()
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected_hashes.get(role):
            raise ValueError('照片文件在预览后已改变，请重新载入')
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        camera = getattr(parsed, role)
        if image is None or image.shape[:2] != (camera.height, camera.width):
            raise ValueError('照片尺寸与标定不一致')
        images[role] = image
        quality[role] = _quality(image, config)
        sources[role] = {'path': str(path), 'sha256': digest, 'captured_at': capture_times[role]}
    if config['stereo_matching']['num_disparities']+config['stereo_matching']['min_disparity'] >= parsed.left.width:
        raise ValueError('视差搜索范围必须小于每目图像宽度')
    depth = _compute_stereo_depth(images['left'], images['right'], parsed, config)
    healthy = depth.audit['status'] == 'VALID' and all(q['passed'] for q in quality.values())
    check = check_ground({'depth': depth, 'pair_healthy': healthy}, parsed, selection)
    result = {'revision': 'standalone-ground-inspection-v1', 'generated_at': datetime.now(timezone.utc).isoformat(),
              'scope': 'CURRENT_STEREO_PAIR_ONLY', 'ground_check': check,
              'sources': sources, 'calibration': calibration, 'analysis_settings': config,
              'quality': quality, 'depth_audit': depth.audit, 'sync_delta_ms': delta,
              'new_camera_capture': False, 'pipe_dimensions_recomputed': False}
    directory = Path(output_root)/('tilt-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:8])
    overlay = ground_evidence_image(images['left'], check)
    ok, encoded = cv2.imencode('.png', overlay)
    if not ok:
        raise ValueError('无法保存倾角检查叠图')
    atomic_write_bytes(directory/'left_ground_overlay.png', encoded.tobytes())
    path = directory/'inspection.json'
    atomic_write_bytes(path, (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8'))
    return path, check
