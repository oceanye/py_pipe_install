import numpy as np
import pytest

from pipe_twin.model_section_reference import build_model_section_reference


def _pipes():
    # All pipes run along Z; x is section-right and y is section-up under the
    # viewer convention used by the elevation GUI.
    return [
        {"pipe_id": "LOW_LEFT", "instance_id": 1, "layer_id": "front", "nominal_diameter_mm": 40,
         "centerline_world_mm": [[-100, -100, -300], [-100, -100, 300]]},
        {"pipe_id": "MIDDLE", "instance_id": 2, "layer_id": "front", "nominal_diameter_mm": 50,
         "centerline_world_mm": [[0, 0, -300], [0, 0, 300]], "installation_state": "INSTALLED"},
        {"pipe_id": "UP_RIGHT", "instance_id": 3, "layer_id": "back", "nominal_diameter_mm": 60,
         "centerline_world_mm": [[100, 100, -300], [100, 100, 300]]},
    ]


def test_reference_contains_basis_coordinates_and_coarse_position_labels():
    result = build_model_section_reference(_pipes())
    assert result["schema_version"] == "model-section-reference-v1"
    assert result["axis_source"] == "DERIVED_FROM_CENTERLINES"
    assert result["basis_rows"]["up_world"] == [0.0, 1.0, 0.0]
    by_id = {row["pipe_id"]: row for row in result["pipes"]}
    assert by_id["LOW_LEFT"]["vertical_band"] == "LOWER"
    assert by_id["LOW_LEFT"]["horizontal_band"] == "LEFT"
    assert by_id["UP_RIGHT"]["vertical_band"] == "MIDDLE"  # singleton layer has no relative upper/lower
    assert by_id["UP_RIGHT"]["vertical_band_global"] == "UPPER"
    assert by_id["UP_RIGHT"]["horizontal_band"] == "RIGHT"
    assert by_id["MIDDLE"]["vertical_band"] == "UPPER"
    assert by_id["MIDDLE"]["horizontal_band"] == "CENTER"
    assert by_id["MIDDLE"]["deterministic_installation_state"] == "INSTALLED"
    assert by_id["LOW_LEFT"]["section_uv_mm"] == pytest.approx([-100.0, -100.0])


def test_explicit_axis_and_camera_side_are_normalized_and_reversing_axis_changes_basis():
    forward = build_model_section_reference(_pipes(), [0, 0, 1], camera_side_world=[0, -2, 0])
    reverse = build_model_section_reference(_pipes(), [0, 0, -1])
    assert forward["axis_source"] == "EXPLICIT"
    assert forward["orientation"]["camera_side_world"] == [0.0, -1.0, 0.0]
    assert np.allclose(forward["axis_world"], [0, 0, 1])
    assert np.allclose(reverse["axis_world"], [0, 0, -1])
    # The model viewer intentionally mirrors the section when the directed
    # axis is reversed; this is why the axis is part of the AI context.
    assert not np.allclose(forward["basis_rows"]["right_world"], reverse["basis_rows"]["right_world"])


def test_explicit_right_and_reference_basis_control_labels_and_camera_depth():
    # +X is the pipe axis in the PG2-style catalogue.  World Y is section
    # right and the derived section-up direction is world Z; camera +Z also
    # exposes front/back depth.
    pipes = [
        {"pipe_id": "FRONT", "nominal_diameter_mm": 40,
         "centerline_world_mm": [[0, 0, 100], [500, 0, 100]]},
        {"pipe_id": "BACK", "nominal_diameter_mm": 40,
         "centerline_world_mm": [[0, 0, -100], [500, 0, -100]]},
        {"pipe_id": "UP", "nominal_diameter_mm": 40,
         "centerline_world_mm": [[0, 100, 0], [500, 100, 0]]},
    ]
    reference = build_model_section_reference(
        pipes, [1, 0, 0], section_right_world=[0, 1, 0], camera_side_world=[0, 0, 1]
    )
    assert reference["basis_source"] == "EXPLICIT_SECTION_RIGHT"
    by_id = {row["pipe_id"]: row for row in reference["pipes"]}
    assert by_id["FRONT"]["camera_depth_rank_near_to_far"] == 1
    assert by_id["BACK"]["camera_depth_rank_near_to_far"] == 3
    assert by_id["UP"]["horizontal_band"] == "RIGHT"

    supplied = {
        "right_world": [0, 1, 0],
        "up_world": [0, 0, 1],
        "pipe_axis_world": [1, 0, 0],
    }
    same = build_model_section_reference(_pipes(), reference_basis=supplied)
    assert same["basis_source"] == "REFERENCE_BASIS"
    assert same["axis_source"] == "REFERENCE_BASIS"
    assert np.allclose(same["basis_rows"]["right_world"], [0, 1, 0])


def test_explicit_section_up_controls_upper_middle_lower_without_using_axis_as_depth():
    result = build_model_section_reference(_pipes(), [0, 0, 1], section_up_world=[0, 1, 0])
    assert result["basis_source"] == "EXPLICIT_SECTION_UP"
    assert result["basis_rows"]["up_world"] == [0.0, 1.0, 0.0]
    assert result["basis_rows"]["pipe_axis_world"] == [0.0, 0.0, 1.0]
    by_id = {row["pipe_id"]: row for row in result["pipes"]}
    assert by_id["LOW_LEFT"]["vertical_band"] == "LOWER"
    assert by_id["UP_RIGHT"]["vertical_band"] == "MIDDLE"
    assert by_id["UP_RIGHT"]["vertical_band_global"] == "UPPER"
    assert by_id["LOW_LEFT"]["axial_offset_mm"] == pytest.approx(0.0)


@pytest.mark.parametrize("bad", [[], [{"pipe_id": "P", "centerline_world_mm": [[0, 0, 0], [0, 0, 0]], "nominal_diameter_mm": 10}]])
def test_invalid_catalog_is_rejected(bad):
    with pytest.raises(ValueError):
        build_model_section_reference(bad)


def test_reference_is_independent_of_pipe_order_for_derived_axis_and_coordinates():
    first = build_model_section_reference(_pipes())
    second = build_model_section_reference(list(reversed(_pipes())))
    assert np.allclose(first["axis_world"], second["axis_world"])
    left = {row["pipe_id"]: row for row in first["pipes"]}
    right = {row["pipe_id"]: row for row in second["pipes"]}
    for pipe_id in left:
        assert left[pipe_id]["section_uv_mm"] == pytest.approx(right[pipe_id]["section_uv_mm"])
        assert left[pipe_id]["vertical_band"] == right[pipe_id]["vertical_band"]
        assert left[pipe_id]["horizontal_band"] == right[pipe_id]["horizontal_band"]
