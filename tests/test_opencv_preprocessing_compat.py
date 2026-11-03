import json
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest

from pipe_twin import stereo_analyzer as analyzer
from pipe_twin.analysis_capabilities import probe_capabilities
from pipe_twin.analysis_runtime_gui import AnalysisRuntimeController
from pipe_twin.analysis_transport import RemoteAnalysisClient, RemoteAnalysisError


def test_standard_build_preserves_existing_low_light_pixels():
    image = np.random.default_rng(14).integers(0, 80, (48, 96, 3), dtype=np.uint8)
    expected = cv2.fastNlMeansDenoising(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), None, 5.0, 7, 21)
    expected = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(expected)
    np.testing.assert_array_equal(analyzer._stereo_gray(image, 'low_light'), expected)
    assert analyzer._stereo_preprocessing_backend('low_light') == 'NLM_CLAHE'


def test_missing_photo_module_uses_finite_edge_preserving_compatibility(monkeypatch):
    monkeypatch.delattr(cv2, 'fastNlMeansDenoising')
    image = np.full((80, 160, 3), 30, np.uint8)
    image[:, 80:] = 180
    original = image.copy()
    gray = analyzer._stereo_gray(image, 'low_light')
    assert gray.shape == image.shape[:2] and gray.dtype == np.uint8
    assert abs(int(gray[40, 81]) - int(gray[40, 78])) > 100
    np.testing.assert_array_equal(image, original)
    assert analyzer._stereo_preprocessing_backend('low_light') == 'BILATERAL_CLAHE'
    capabilities = probe_capabilities()
    assert not capabilities['preprocessing']['fast_nl_means_available']
    assert '双边兼容' in AnalysisRuntimeController._capability_text('服务器', capabilities)


def test_missing_photo_build_records_actual_backend_in_depth_report(monkeypatch):
    monkeypatch.delattr(cv2, 'fastNlMeansDenoising')
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'test_model/field_stereo_demo_manifest.json').read_text(encoding='utf-8'))
    calibration = analyzer._calibration_from_manifest(manifest['stereo_calibration'])
    config = analyzer._analysis_config({'stereo_matching': {'preprocessing': 'low_light', 'num_disparities': 16}})
    image = np.zeros((64, 128, 3), np.uint8)
    depth = analyzer._compute_stereo_depth(image, image.copy(), calibration, config)
    assert depth.audit['preprocessing_backend'] == 'BILATERAL_CLAHE'
    assert depth.audit['status'] == 'INVALID'
    assert not np.any(depth.left_valid)


def test_no_supported_denoiser_reports_clear_error_but_raw_gray_still_works(monkeypatch):
    monkeypatch.delattr(cv2, 'fastNlMeansDenoising')
    monkeypatch.delattr(cv2, 'bilateralFilter')
    image = np.zeros((5, 8, 3), np.uint8)
    with pytest.raises(analyzer.StereoAnalysisError, match='原始灰度'):
        analyzer._stereo_gray(image, 'low_light')
    assert analyzer._stereo_gray(image, 'none').shape == (5, 8)


def test_missing_contrast_api_is_visible_in_capabilities_and_error(monkeypatch):
    monkeypatch.delattr(cv2, 'createCLAHE')
    assert '补齐 OpenCV' in AnalysisRuntimeController._capability_text('服务器', probe_capabilities())
    with pytest.raises(analyzer.StereoAnalysisError, match='createCLAHE'):
        analyzer._stereo_gray(np.zeros((5, 8, 3), np.uint8), 'low_light')


def test_remote_compute_failure_is_logged_with_source_and_server():
    client = RemoteAnalysisClient('http://127.0.0.1:8771')
    error = RemoteAnalysisError("AttributeError: module 'cv2' has no attribute 'fastNlMeansDenoising'")
    with mock.patch.object(client, '_analyze', side_effect=error), \
         mock.patch('pipe_twin.logging_config.log_event') as log:
        with pytest.raises(RemoteAnalysisError):
            client.analyze('scene/manifest.json', 'scene/report.json', 'scene/evidence')
    failed = [call for call in log.call_args_list if call.args[1] == 'remote_analysis_failed']
    assert len(failed) == 1
    assert failed[0].kwargs['server'] == client.base_url
    assert failed[0].kwargs['manifest'] == 'scene/manifest.json'
    assert 'fastNlMeansDenoising' in failed[0].kwargs['error']
