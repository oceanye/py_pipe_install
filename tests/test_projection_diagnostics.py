import cv2
import numpy as np
import pytest

from pipe_twin.projection_diagnostics import fit_silhouettes, pool_strip_mask, pixel_error_scenarios


def independent_contours(axis, center, radius, baseline=60.):
    """Project a dense known surface with OpenCV; obtain its convex outline.

    No use of the fitter's tangent-ray forward model or its section basis.
    """
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    u = np.cross(a, [0, 0, 1]); u /= np.linalg.norm(u)
    v = np.cross(a, u)
    theta = np.linspace(0, 2*np.pi, 12000, endpoint=False)
    ring = radius*(np.cos(theta)[:, None]*u+np.sin(theta)[:, None]*v)
    cloud = np.vstack((ring-450*a, ring+450*a))+center
    k = np.array([[1200., 0, 960], [0, 1200, 540], [0, 0, 1]])
    observations = []
    for role, camera in [('left', np.zeros(3)), ('right', np.array([baseline, 0., 0.]))]:
        image, _ = cv2.projectPoints(cloud, np.zeros(3), -camera, k, None)
        hull = cv2.convexHull(image.reshape(-1, 2).astype(np.float32)).reshape(-1, 2)
        nxt = np.roll(hull, -1, axis=0)
        lengths = np.linalg.norm(nxt-hull, axis=1)
        indices = np.argsort(lengths)[-2:]
        pairs = [(hull[i].astype(float), nxt[i].astype(float)) for i in indices]
        pairs.sort(key=lambda pair: np.mean([p[1] for p in pair]))
        for edge, (p, q) in zip(['top', 'bottom'], pairs):
            points = p[None, :]+np.linspace(.2, .8, 31)[:, None]*(q-p)
            observations.append({'role': role, 'edge': edge, 'points': points})
    return observations, k


@pytest.mark.parametrize('axis,radius', [([1., .35, .1], 13.), ([1., -.4, .2], 25.5), ([1., -.07, .12], 20.5)])
def test_independent_projected_surface_recovers_diameter_and_station(axis, radius):
    center = np.array([100., 80., 1800.])
    observations, k = independent_contours(axis, center, radius)
    reference = k[0, 0]*center[0]/center[2]+k[0, 2]
    result = fit_silhouettes(observations, k, 60., axis, reference)
    assert result['status'] == 'CONDITIONAL_FIT'
    assert result['trial_diameter_mm'] == pytest.approx(2*radius, abs=.02)
    np.testing.assert_allclose(result['center_at_reference_column_mm'], center, atol=.3)
    assert result['reprojection_rms_px'] < .001
    assert result['accepted_diameter_mm'] is None


def test_parallel_axis_and_baseline_do_not_create_a_distance():
    observations, k = independent_contours([1, 0, 0], [0, 50, 1800], 20)
    result = fit_silhouettes(observations, k, 60., [1, 0, 0], 960.)
    assert result['status'] == 'UNOBSERVABLE_PARALLEL_BASELINE'
    assert 'trial_diameter_mm' not in result


def test_pixel_edge_shift_is_more_damaging_near_parallel_baseline():
    errors = []
    for axis in ([1., .5, 0.], [1., .03, 0.]):
        observations, k = independent_contours(axis, [0, 50, 1800], 20)
        for obs in observations:
            if obs['role'] == 'right':
                obs['points'][:, 1] += .25
        result = fit_silhouettes(observations, k, 60., axis, 960.)
        assert result['status'] == 'CONDITIONAL_FIT'
        errors.append(abs(result['center_at_reference_column_mm'][2]-1800))
    assert errors[1] > 5*errors[0]


def test_pooling_preserves_width_and_absolute_offset_despite_damaged_columns():
    rows, cols = np.indices((180, 320))
    mask = (rows >= 40+.1*cols) & (rows < 90+.1*cols)
    mask[:, 100:120] = False
    a = pool_strip_mask(mask, 30, 290, 0, 170, .1)
    shifted = np.roll(mask, 8, axis=0)
    b = pool_strip_mask(shifted, 30, 290, 0, 170, .1)
    assert len(a['intervals']) == len(b['intervals']) == 1
    assert a['intervals'][0]['width_px'] == pytest.approx(50/np.sqrt(1.01), abs=.4)
    assert b['intervals'][0]['low_q_px']-a['intervals'][0]['low_q_px'] == pytest.approx(8/np.sqrt(1.01), abs=.1)


def test_pooling_does_not_merge_multiple_separate_segments():
    mask = np.zeros((140, 200), bool)
    mask[20:40] = True
    mask[70:100] = True
    result = pool_strip_mask(mask, 0, 200, 0, 139, 0.)
    assert len(result['intervals']) == 2
    assert [r['width_px'] for r in result['intervals']] == pytest.approx([20, 30])


def test_one_occluding_edge_is_excluded_and_three_true_edges_can_suffice():
    axis = [1., -.4, .2]
    observations, k = independent_contours(axis, [0, 50, 1800], 20)
    observations[0]['boundary_kind'] = 'occlusion'
    observations[0]['points'][:, 1] += 35  # Another object, not this pipe's tangent.
    result = fit_silhouettes(observations, k, 60., axis, 960.)
    assert result['status'] == 'CONDITIONAL_FIT'
    assert result['observed_boundary_count'] == 3
    assert result['ignored_occlusion_boundaries'] == [['left', 'top']]
    assert result['trial_diameter_mm'] == pytest.approx(40, abs=.02)


def test_only_two_visible_tangents_do_not_determine_free_radius_and_position():
    observations, k = independent_contours([1, -.4, .2], [0, 50, 1800], 20)
    visible = [o for o in observations if o['edge'] == 'top']
    result = fit_silhouettes(visible, k, 60., [1, -.4, .2], 960.)
    assert result['status'] == 'INSUFFICIENT_INDEPENDENT_SILHOUETTES'
    assert 'trial_diameter_mm' not in result


def test_pixel_uncertainty_increases_for_weaker_baseline_geometry():
    estimates = []
    for axis in ([1., .5, 0.], [1., .03, 0.]):
        observations, k = independent_contours(axis, [0, 50, 1800], 20)
        estimate = pixel_error_scenarios(observations, k, 60., axis, 960., sigma_px=.25)
        assert estimate['total_accuracy_known'] is False
        iid = estimate['scenarios']['IID_PIXELS_OPTIMISTIC']
        coherent = estimate['scenarios']['COHERENT_EDGE_OFFSETS']
        assert iid['diameter_sigma_mm'] < coherent['diameter_sigma_mm']
        estimates.append(coherent['diameter_sigma_mm'])
    assert estimates[1] > 5*estimates[0]


def test_more_pixel_copies_do_not_reduce_coherent_edge_error():
    axis = [1., -.4, .2]
    observations, k = independent_contours(axis, [0, 50, 1800], 20)
    a = pixel_error_scenarios(observations, k, 60., axis, 960.)
    repeated = [{**o, 'points': np.tile(o['points'], (4, 1))} for o in observations]
    b = pixel_error_scenarios(repeated, k, 60., axis, 960.)
    assert b['scenarios']['COHERENT_EDGE_OFFSETS']['diameter_sigma_mm'] == pytest.approx(
        a['scenarios']['COHERENT_EDGE_OFFSETS']['diameter_sigma_mm'], rel=.01)
    # The deliberately optimistic IID hypothesis would reduce the estimate;
    # it must remain labelled as an assumption, not an observed accuracy.
    assert b['scenarios']['IID_PIXELS_OPTIMISTIC']['diameter_sigma_mm'] == pytest.approx(
        a['scenarios']['IID_PIXELS_OPTIMISTIC']['diameter_sigma_mm']/2, rel=.01)
