import copy

import cv2
import numpy as np
import pytest

from pipe_twin.calibration_quality import holdout_quality, calibration_quality_text
from pipe_twin.calibration_wizard import solve_stereo_calibration_checked, BoardObservation
from test_chessboard_calibration import _synthetic_pairs, _solve, SIZE, PATTERN, SQUARE_MM


def _validation_pairs():
    from test_chessboard_calibration import _object_points, INTRINSIC, DISTORTION, BASELINE_MM
    rig = cv2.Rodrigues(np.array([0., np.deg2rad(.4), 0.]))[0]
    pairs = []
    for i in range(3):
        rotation = cv2.Rodrigues(np.array([-.12+.12*i, .14-.1*i, .05]))[0]
        translation = np.array([-240.+150*i, -180.+70*i, 1500.+80*i])
        observations = []
        for r, t in ((rotation, translation), (rig@rotation, rig@translation+[-BASELINE_MM, 0, 0])):
            points = cv2.projectPoints(_object_points(), cv2.Rodrigues(r)[0], t, INTRINSIC, DISTORTION)[0].reshape(-1, 2)
            observations.append(BoardObservation(points, 100., (i, i)))
        pairs.append(tuple(observations))
    return pairs


@pytest.fixture(scope="module")
def known_calibration():
    return _solve()


def test_unseen_projected_views_pass_without_refitting(known_calibration):
    pairs = _validation_pairs()
    report = holdout_quality(pairs, recipe=known_calibration.recipe,
        image_size=SIZE, pattern=PATTERN, square_mm=SQUARE_MM)
    assert report["status"] == "PASSED", report
    assert report["cross_reprojection_rms_px"] < .01
    assert report["vertical_p95_px"] < .01
    assert report["scale_verified_independently"] is False


def test_holdout_rejects_vertical_error_and_wrong_stereo_baseline(known_calibration):
    pairs = _validation_pairs()
    bad = [(left, BoardObservation(right.corners_px+np.array([0., 2.]), right.sharpness, right.centroid_zone))
           for left, right in pairs]
    report = holdout_quality(bad, recipe=known_calibration.recipe,
        image_size=SIZE, pattern=PATTERN, square_mm=SQUARE_MM)
    assert report["status"] == "FAILED"
    assert report["vertical_p95_px"] > 1.5
    recipe = copy.deepcopy(known_calibration.recipe)
    recipe["P2"][0][3] *= 1.15
    report = holdout_quality(pairs, recipe=recipe, image_size=SIZE, pattern=PATTERN, square_mm=SQUARE_MM)
    assert report["status"] == "FAILED"
    assert report["cross_reprojection_rms_px"] > 3


def test_changing_all_metric_units_cannot_be_detected_by_pixel_rms(known_calibration):
    recipe = copy.deepcopy(known_calibration.recipe)
    factor = 19/17
    recipe["P2"][0][3] *= factor
    report = holdout_quality(_validation_pairs(), recipe=recipe,
        image_size=SIZE, pattern=PATTERN, square_mm=SQUARE_MM*factor)
    assert report["status"] == "PASSED"
    assert report["scale_verified_independently"] is False


def test_checked_solver_requires_training_plus_holdout():
    with pytest.raises(ValueError, match="留出"):
        solve_stereo_calibration_checked(pairs=_synthetic_pairs(12), min_pairs=10)


def test_checked_solver_keeps_validation_out_of_training():
    result = solve_stereo_calibration_checked(pairs=_synthetic_pairs(12), image_size=SIZE,
        pattern=PATTERN, square_mm=SQUARE_MM, min_pairs=9, max_reprojection_rms_px=.5,
        operator="field", max_sync_delta_ms=1., layout="side_by_side_left_right")
    assert result.validated, result.rejection_reasons
    audit = result.audit
    assert len(audit["holdout"]["input_pair_indices"]) == 3
    assert not set(audit["training_input_indices"]) & set(audit["holdout"]["input_pair_indices"])
    assert result.calibration["quality_audit"]["holdout"]["status"] == "PASSED"
    assert result.recipe["calibration_id"] == result.calibration["calibration_id"]
    assert "毫米尺度待" in calibration_quality_text(result.calibration)


def test_legacy_calibration_does_not_claim_precision():
    assert "建议重新标定" in calibration_quality_text({"validated": True})


def test_checked_solver_rejects_duplicated_validation_pose_before_fit():
    pairs = _synthetic_pairs()
    pairs[2] = copy.deepcopy(pairs[0])
    with pytest.raises(ValueError, match="姿态重复"):
        solve_stereo_calibration_checked(pairs=pairs, min_pairs=9)


def test_heldout_error_does_not_get_pruned_or_saved():
    pairs = _synthetic_pairs()
    left, right = pairs[2]  # fixed split reserves index 3
    pairs[2] = (left, BoardObservation(right.corners_px+np.array([0., 2.]), right.sharpness, right.centroid_zone))
    result = solve_stereo_calibration_checked(pairs=pairs, image_size=SIZE,
        pattern=PATTERN, square_mm=SQUARE_MM, min_pairs=9, max_reprojection_rms_px=5.,
        operator="field", max_sync_delta_ms=1., layout="side_by_side_left_right")
    assert not result.validated  # relaxed training RMS cannot override holdout
    assert result.audit["holdout"]["status"] == "FAILED"
    assert 3 not in result.audit["discarded_pair_indices"]


def test_save_preserves_old_calibration_and_portable_recipe(tmp_path, known_calibration, monkeypatch):
    from pipe_twin import workbench_profile
    from pipe_twin.calibration_wizard import save_wizard_result
    import json
    destination = tmp_path/"calibration_current.json"
    original = b'{"old": "calibration"}'
    destination.write_bytes(original)
    monkeypatch.setattr(workbench_profile, "default_calibration_path", lambda: destination)
    monkeypatch.setattr(workbench_profile, "default_profile_path", lambda: tmp_path/"profile.json")
    saved = save_wizard_result(known_calibration)
    bundle = json.loads(saved.read_text(encoding="utf-8"))
    assert bundle["rectification_recipe"] == known_calibration.recipe
    assert bundle["stereo_calibration"]["quality_audit"]["scale_verified_independently"] is False
    backups = [p for p in tmp_path.glob("calibration_*.json") if p != destination]
    assert len(backups) == 1 and backups[0].read_bytes() == original
