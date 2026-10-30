"""Physical cylinder -> raster images -> complete conditional section pipeline.

These tests exercise image localization as well as geometry. Rasterization and
colour thresholds have systematic edge bias; their tolerance is deliberately
separate from the exact analytic-point tests and is not a field accuracy claim.
"""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pipe_twin.pooled_sections import analyze_pooled_sections
from test_directional_projection import analytic_samples


@pytest.mark.parametrize('degrees,cropped_ground', [
    (60., False), (90., False), (-48.6, False), (90., True),
])
def test_rendered_physical_cylinder_reaches_the_section_model_in_original_coordinates(degrees, cropped_ground):
    observations, intrinsic, true_axis, true_center, _, _ = analytic_samples(degrees)
    images = {}
    for role in ('left', 'right'):
        edges = [o['points'] for o in observations if o['role'] == role]
        # Independently generated real tangent points bound the visible body.
        polygon = np.vstack((edges[0][[0, -1]], edges[1][[-1, 0]]))
        image = np.full((1080, 1920, 3), 60, np.uint8)
        cv2.fillConvexPoly(image, np.rint(polygon*256).astype(np.int32),
                           (235, 105, 15), cv2.LINE_AA, 8)
        images[role] = image
    right_intrinsic = intrinsic.copy()
    ground = None
    if cropped_ground:
        images['right'] = images['right'][23:, 57:].copy()
        right_intrinsic[:2, 2] -= [57, 23]
        normal = np.cross(true_axis, [.2, -.3, 1.])
        normal /= np.linalg.norm(normal)
        ground = {'status': 'CONSISTENT_GROUND_PLANE',
                  'surface_identity_source': 'OPERATOR_MARKED_GROUND',
                  'left_plane': {'normal_camera': normal.tolist()}}
    calibration = SimpleNamespace(
        left=SimpleNamespace(rectified_intrinsic=intrinsic),
        right=SimpleNamespace(rectified_intrinsic=right_intrinsic), baseline_mm=60.)
    originals = {role: image.copy() for role, image in images.items()}
    original_intrinsics = [intrinsic.copy(), right_intrinsic.copy()]

    result = analyze_pooled_sections(images['left'], images['right'], calibration, ground_result=ground)

    assert result['status'] == 'ANALYZED'
    assert result['boundary_pair_count'] == result['conditional_section_count'] == 1
    assert not result['nominal_dimensions_used_in_fitting']
    assert not result['occlusion_recovered']
    candidate, = result['candidates']
    fitted = candidate['fit']
    assert fitted['status'] == 'CONDITIONAL_FIT'
    assert fitted['accepted_diameter_mm'] is None
    # Up to 1.5 mm here allows the measured raster/threshold edge bias. Exact
    # analytic contour tests independently require 1e-5 mm recovery.
    assert fitted['trial_diameter_mm'] == pytest.approx(40., abs=1.5)
    assert fitted['reference_definition'] == 'IMAGE_TRANSVERSE_LINE'
    assert fitted['physical_baseline_mm'] == 60.
    assert candidate['window_sensitivity']['stable']
    assert not candidate['reason_codes']
    assert abs(np.dot(fitted['axis'], true_axis)) > .9999
    # The reference station need not be at the generating centre. Compare
    # distance to the known 3-D axis, not its arbitrary axial coordinate.
    center = np.asarray(fitted['center_at_reference_pixel_mm'])
    assert np.linalg.norm(np.cross(center-true_center, true_axis)) < 10.
    assert result['model'] is not None
    if degrees == -48.6:
        left_axis = result['views']['left'][0]['image_axis']
        right_axis = result['views']['right'][0]['image_axis']
        # Perspective puts the two estimated axes across the dominant-sign
        # convention boundary. Edge labels must be paired after this reversal.
        assert np.dot(left_axis, right_axis) < -.99
    if cropped_ground:
        assert result['ground_correction']['applied']
        assert not result['ground_correction']['calibration_modified']
        assert abs(np.dot(fitted['axis'], ground['left_plane']['normal_camera'])) < 1e-12
    for role in images:
        np.testing.assert_array_equal(images[role], originals[role])
    np.testing.assert_array_equal(intrinsic, original_intrinsics[0])
    np.testing.assert_array_equal(right_intrinsic, original_intrinsics[1])
