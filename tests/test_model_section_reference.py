import numpy as np
import pytest

from pipe_twin.model_section_reference import build_model_section_reference


def _pipes():
    return [
        {"pipe_id": "LOW_LEFT", "instance_id": 1, "nominal_diameter_mm": 40,
         "centerline_world_mm": [[-100, -100, -300], [-100, -100, 300]]},
        {"pipe_id": "MIDDLE", "instance_id": 2, "nominal_diameter_mm": 50,
         "centerline_world_mm": [[0, 0, -300], [0, 0, 300]]},
        {"pipe_id": "UP_RIGHT", "instance_id": 3, "nominal_diameter_mm": 60,
         "centerline_world_mm": [[100, 100, -300], [100, 100, 300]]},
    ]


def test_reference_assigns_fixed_points_from_section_order_only():
    result = build_model_section_reference(_pipes(), [0, 0, 1], section_up_world=[0, 1, 0])
    assert result["schema_version"] == "section-point-map-v1"
    assert result["scope"] == "CROSS_SECTION_EVALUATION_ONLY"
    assert len(result["mapping_sha256"]) == 64
    assert [p["point_id"] for p in result["points"]] == ["P01", "P02", "P03"]
    by_id = {p["pipe_id"]: p for p in result["points"]}
    assert by_id["UP_RIGHT"]["point_id"] == "P01"
    assert by_id["MIDDLE"]["point_id"] == "P02"
    assert by_id["LOW_LEFT"]["point_id"] == "P03"
    assert all("layer_id" not in p and "camera_depth_mm" not in p and "axial_offset_mm" not in p for p in result["points"])
    assert all(pair["scope"] == "SAME_2D_SECTION_ONLY" for pair in result["pairs"])
    assert result["installation_state"] == "OUT_OF_SCOPE_FOR_SECTION_REFERENCE"


def test_observation_view_projection_controls_point_order_and_records_view():
    view = {
        "view_id": "reference_elevation",
        "rotation_world_to_view": np.eye(3).tolist(),
        "center_world_mm": [0, 0, 0],
        "principal_point_px": [800, 500],
        "scale_px_per_mm": 1.5,
    }
    result = build_model_section_reference(_pipes(), [0, 0, 1], section_up_world=[0, 1, 0], observation_view=view)
    assert result["observation_view"]["view_id"] == "reference_elevation"
    assert result["view_id"] == "reference_elevation"
    assert result["ordering_rule"] == "PROJECTED_PIXEL_V_ASC_THEN_U_ASC_THEN_PIPE_ID"
    assert all(p["projected_center_px"] is not None for p in result["points"])


def test_reference_basis_accepts_normal_alias_and_reversing_direction_changes_order():
    supplied = {"right_world": [1, 0, 0], "up_world": [0, 1, 0], "normal_world": [0, 0, 1]}
    forward = build_model_section_reference(_pipes(), reference_basis=supplied)
    reverse = build_model_section_reference(_pipes(), [0, 0, -1], section_up_world=[0, 1, 0])
    assert forward["normal_source"] == "REFERENCE_BASIS"
    assert not np.allclose(forward["basis_rows"]["right_world"], reverse["basis_rows"]["right_world"])


def test_projection_overlap_is_explicitly_marked():
    pipes = [
        {"pipe_id": "A", "nominal_diameter_mm": 20, "centerline_world_mm": [[0, 0, 0], [100, 0, 0]]},
        {"pipe_id": "B", "nominal_diameter_mm": 20, "centerline_world_mm": [[0, 0, 50], [100, 0, 50]]},
    ]
    view = {"view_id": "front", "rotation_world_to_view": np.eye(3).tolist(), "center_world_mm": [0, 0, 0], "principal_point_px": [0, 0], "scale_px_per_mm": 1}
    result = build_model_section_reference(pipes, [1, 0, 0], section_up_world=[0, 1, 0], observation_view=view)
    assert all(p["projection_ambiguity"] == "OVERLAPPING_MODEL_POINTS" for p in result["points"])


@pytest.mark.parametrize("bad", [[], [{"pipe_id": "P", "centerline_world_mm": [[0, 0, 0], [0, 0, 0]], "nominal_diameter_mm": 10}]])
def test_invalid_catalog_is_rejected(bad):
    with pytest.raises(ValueError):
        build_model_section_reference(bad)
