"""Viewing side must constrain real poses without manufacturing evidence."""
import numpy as np
import pytest

from pipe_twin.camera_view import CAMERA_SIDE_PRESETS, preview_rotation
from pipe_twin.elevation_dataset import normalize_registration_settings
from pipe_twin.elevation_registration import register_elevation
from test_elevation_registration import _specs


def side_observations(specs):
    # Model Z pipe axis appears horizontal; model Y is camera depth.
    rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    return [{"observation_id": f"O{i+1}",
             "center_camera_mm": (rotation @ np.mean(spec["centerline_world_mm"], axis=0)
                                   + [0., 0., 1400.]).tolist(),
             "axis_camera": [1., 0., 0.], "diameter_mm": spec["nominal_diameter_mm"]}
            for i, spec in enumerate(specs)]


def test_correct_oblique_side_preserves_identity_and_opposite_side_rejects():
    specs = _specs()
    obs = side_observations(specs)
    result = register_elevation(specs, obs, camera_side_world=[1, -2, 0])
    assert result["status"] == "MATCHED"
    assert {(m["observation_id"], m["pipe_id"]) for m in result["matches"]} == {
        (f"O{i}", f"P{i}") for i in range(1, 5)}
    assert np.allclose(result["estimated_camera_side_world"], [0, -1, 0])
    wrong = register_elevation(specs, obs, camera_side_world=[0, 1, 0])
    assert wrong["status"] == "AMBIGUOUS"
    assert wrong["reason_codes"] == ["CAMERA_SIDE_CONFLICT"]
    assert wrong["matches"] == [] and wrong["rotation_model_to_camera"] is None
    assert wrong["camera_view"]["rejected_pose_count"] > 0


def test_side_can_remove_two_anchor_mirror_but_is_not_a_third_observation():
    specs = _specs()
    obs = side_observations(specs)[:2]
    anchors = {"O1": "P1", "O2": "P2"}
    assert register_elevation(specs, obs, anchors=anchors)["status"] == "AMBIGUOUS"
    assert register_elevation(specs, obs, anchors=anchors, camera_side_world=[0, -1, 0])["status"] == "MATCHED"
    assert register_elevation(specs, obs, camera_side_world=[0, -1, 0])["status"] == "INSUFFICIENT_OBSERVATIONS"
    assert register_elevation(specs, [], camera_side_world=[0, -1, 0])["matches"] == []


def test_symmetric_layout_and_search_limit_still_refuse_unique_match():
    specs = _specs(equal=True)
    result = register_elevation(specs, side_observations(specs), camera_side_world=[0, -1, 0])
    assert result["status"] == "AMBIGUOUS"
    assert result["alternatives"]
    limited = register_elevation(specs, side_observations(specs), camera_side_world=[0, -1, 0],
                                 config={"max_hypotheses": 1})
    assert limited["status"] == "SEARCH_LIMIT"


def test_edge_on_hint_does_not_resolve_hemisphere():
    specs = _specs()
    result = register_elevation(specs, side_observations(specs), camera_side_world=[1, 0, 0])
    assert result["reason_codes"] == ["CAMERA_SIDE_CONFLICT"]


@pytest.mark.parametrize("value", [[0, 0, 0], [1, 2], [True, 0, 0], ["1", 0, 0],
                                  [float("nan"), 0, 0], [float("inf"), 0, 0], "-Y"])
def test_invalid_side_is_rejected_at_manifest_boundary(value):
    with pytest.raises(ValueError):
        normalize_registration_settings({"camera_side_world": value})


def test_legacy_automatic_settings_and_signed_model_previews():
    assert normalize_registration_settings() == normalize_registration_settings({"camera_side_world": None})
    assert normalize_registration_settings({"camera_side_world": [0, -7, 0]})["camera_side_world"] == [0, -1, 0]
    for side in CAMERA_SIDE_PRESETS.values():
        if side is None:
            continue
        rotation = preview_rotation(side)
        assert np.allclose(rotation @ side, [0, 0, 1])
        assert np.linalg.det(rotation) == pytest.approx(1)
    # Looking from opposite Y faces reverses the displayed model-X order.
    assert (preview_rotation([0, -1, 0]) @ [1, 0, 0])[0] == 1
    assert (preview_rotation([0, 1, 0]) @ [1, 0, 0])[0] == -1
