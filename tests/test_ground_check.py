from types import SimpleNamespace
import math
import cv2
import numpy as np
import pytest

from pipe_twin.ground_check import check_ground, normalize_ground_region, ground_summary_lines, _fit_plane


def scene(normal=(.1, -.8, -.4), distance=500.):
    normal = np.asarray(normal, float); normal /= np.linalg.norm(normal)
    k = np.array([[500., 0, 320], [0, 500., 240], [0, 0, 1.]])
    camera = SimpleNamespace(width=640, height=480, rectified_intrinsic=k)
    calibration = SimpleNamespace(left=camera, right=camera, baseline_mm=60., validated=False)
    yy, xx = np.indices((480, 640)); rays = np.stack(((xx-320)/500, (yy-240)/500, np.ones_like(xx)), axis=-1)
    values = {}
    for role, origin in [('left', np.zeros(3)), ('right', np.array([60., 0., 0.]))]:
        denominator = rays@normal
        depth = np.divide(-(distance+origin@normal), denominator, out=np.zeros_like(denominator), where=abs(denominator)>1e-9)
        values[role+'_depth_mm'] = depth
        values[role+'_valid'] = depth > 0
    region = normalize_ground_region({'image_size': [640, 480], 'rect_px': [160, 300, 400, 160]})
    return {'depth': SimpleNamespace(**values), 'pair_healthy': True}, calibration, region, normal


def add_ground_images(group, calibration, normal, distance=500.):
    """Independently render matching texture using the analytic plane homography."""
    k = calibration.left.rectified_intrinsic
    yy, xx = np.indices((calibration.left.height, calibration.left.width), dtype=np.float32)
    rng = np.random.default_rng(613)
    left = cv2.GaussianBlur(rng.integers(0, 256, xx.shape, dtype=np.uint8), (3, 3), .7)
    a = -calibration.baseline_mm*normal[0]/distance
    b = -k[0, 0]*calibration.baseline_mm*normal[1]/(k[1, 1]*distance)
    c = -k[0, 0]*calibration.baseline_mm/distance*(normal[2]-normal[0]*k[0, 2]/k[0, 0]-normal[1]*k[1, 2]/k[1, 1])
    right = cv2.remap(left, ((xx+b*yy+c)/(1-a)).astype(np.float32), yy, cv2.INTER_LINEAR)
    group.update(left=cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), right=cv2.cvtColor(right, cv2.COLOR_GRAY2BGR))
    return group


@pytest.mark.parametrize('normal', [(0, -1, 0), (.1, -.8, -.4), (-.2, -.7, -.5)])
def test_ground_distance_and_pitch_use_camera_coordinates_not_vertical_image_assumption(normal):
    group, calibration, region, normal = scene(normal)
    pipe = np.cross(normal, [0., 0., 1.]); pipe /= np.linalg.norm(pipe)
    result = check_ground(group, calibration, region, section_stages=[{'axis': {'axis': pipe, 'status': 'STABLE'}}])
    assert result['status'] == 'CONSISTENT_GROUND_PLANE'
    assert result['camera_to_ground_perpendicular_mm'] == pytest.approx(500., abs=.1)
    assert result['optical_axis_to_ground_deg'] == pytest.approx(math.degrees(math.asin(abs(normal[2]))), abs=.01)
    assert result['pipe_axis_checks'][0]['angle_to_ground_deg'] < .01
    assert result['accepted_measurement'] is False
    assert result['pipe_axis_constrained_by_ground'] is False
    assert '暂用标定' in '\n'.join(ground_summary_lines(result))


def test_obstacles_and_missing_depth_do_not_get_filled_into_a_ground_plane():
    group, calibration, region, _ = scene()
    rng = np.random.default_rng(22)
    for role in ('left', 'right'):
        z = getattr(group['depth'], role+'_depth_mm')
        z += rng.normal(0, .5, z.shape)
        z[325:340, 160:560] -= 250  # An obstacle occupying less than a quarter of the ROI.
        getattr(group['depth'], role+'_valid')[390:400, 160:560] = False
    originals = {r: getattr(group['depth'], r+'_depth_mm').copy() for r in ('left', 'right')}
    result = check_ground(group, calibration, region)
    assert result['status'] == 'CONSISTENT_GROUND_PLANE'
    assert result['camera_to_ground_perpendicular_mm'] == pytest.approx(500., abs=2.)
    for role in originals:
        np.testing.assert_array_equal(originals[role], getattr(group['depth'], role+'_depth_mm'))


@pytest.mark.parametrize('failure', ['missing', 'random', 'right_plane', 'sliver'])
def test_bad_ground_evidence_does_not_publish_a_distance(failure):
    group, calibration, region, _ = scene()
    if failure == 'missing':
        group['depth'].left_valid[:] = False
    elif failure == 'random':
        group['depth'].left_depth_mm[:] = np.random.default_rng(2).uniform(500, 3000, (480, 640))
    elif failure == 'right_plane':
        group['depth'].right_depth_mm *= 1.3
    else:
        group['depth'].left_valid[:] = False
        group['depth'].left_valid[379:382] = True
    result = check_ground(group, calibration, region)
    assert result['status'] != 'CONSISTENT_GROUND_PLANE'
    assert 'camera_to_ground_perpendicular_mm' not in result


def test_ground_selection_is_optional_and_resolution_bound():
    group, calibration, region, _ = scene()
    assert check_ground(group, calibration, None)['status'] == 'NOT_CONFIGURED'
    with pytest.raises(ValueError):
        normalize_ground_region(region, image_size=(1920, 1080))
    with pytest.raises(ValueError):
        normalize_ground_region(region | {'rect_px': [0, 0, 900, 480]})


def test_unconfirmed_ground_and_unstable_pipe_axis_are_not_promoted():
    group, calibration, region, _ = scene()
    region['selection_source'] = 'ASSISTANT_DIAGNOSTIC'
    result = check_ground(group, calibration, region, section_stages=[
        {'axis': {'axis': [1, 0, 0], 'status': 'COMMON_AXIS_UNSTABLE'}}])
    assert result['status'] == 'CONSISTENT_GROUND_PLANE'
    assert result['surface_identity_source'] == 'ASSISTANT_SUGGESTED_GROUND_NOT_CONFIRMED'
    assert result['pipe_axis_checks'] == []
    assert '尚未经你确认' in '\n'.join(ground_summary_lines(result))


def test_top_down_ground_can_give_height_but_not_image_roll():
    group, calibration, region, _ = scene((0, 0, -1))
    result = check_ground(group, calibration, region)
    assert result['status'] == 'CONSISTENT_GROUND_PLANE'
    assert result['camera_to_ground_perpendicular_mm'] == pytest.approx(500., abs=.1)
    assert result['optical_axis_to_ground_deg'] == pytest.approx(90.)
    assert result['image_roll_relative_to_ground_deg'] is None
    assert not result['roll_observable']
    assert '横滚不可可靠观测' in '\n'.join(ground_summary_lines(result))


def test_held_out_depths_never_change_the_plane_used_for_ground_correction():
    yy, xx = np.mgrid[0:64, 0:64]
    cells = (xx//16+4*(yy//16)).ravel()
    held_out = (cells % 4+cells//4) % 2 != 0
    points = np.c_[xx.ravel()*4., yy.ravel()*4., np.full(xx.size, 1000.)]
    points[:, 2] += np.random.default_rng(36).normal(0, .15, len(points))
    first, _ = _fit_plane(points, cells)
    # Simulate an independently observed, slightly biased validation region.
    # It is within the existing checks but must not pull the published normal.
    changed = points.copy()
    changed[held_out, 2] += 2.0 + (changed[held_out, 0]-128.)*.004
    second, mask = _fit_plane(changed, cells)
    assert first['status'] == second['status'] == 'PLANE_FIT'
    assert first['normal_camera'] == second['normal_camera']
    assert first['offset_mm'] == second['offset_mm']
    assert second['holdout_rms_mm'] > first['holdout_rms_mm']+1.5
    assert second['fit_scope'] == 'TRAINING_CELLS_ONLY'
    residual = changed@second['normal_camera']+second['offset_mm']
    np.testing.assert_array_equal(mask, abs(residual) <= 5.)
    expected = residual[held_out & mask]
    assert second['holdout_rms_mm'] == pytest.approx(np.sqrt(np.mean(expected**2)))
    assert second['inlier_fraction'] == pytest.approx(mask.mean())


def test_ground_rejection_reports_failed_metrics_without_hiding_empty_holdout():
    yy, xx = np.mgrid[0:64, 0:64]
    cells = (xx//16+4*(yy//16)).ravel()
    held_out = (cells % 4+cells//4) % 2 != 0
    points = np.c_[xx.ravel()*4., yy.ravel()*4., 1000.+held_out*8.]
    result, mask = _fit_plane(points, cells)
    assert result['status'] == 'GROUND_PLANE_INCONSISTENT' and mask is None
    assert result['failure_reasons'] == ['GROUND_PLANE_SUPPORT_FRACTION_LOW',
        'GROUND_HOLDOUT_SUPPORT_FRACTION_LOW', 'GROUND_HOLDOUT_RMS_HIGH']
    assert result['holdout_rms_mm'] is None
    assert result['inlier_fraction'] == .5 and result['holdout_inlier_fraction'] == 0.
    checks = {item['reason_code']: item for item in result['quality_checks']}
    assert checks['GROUND_SPATIAL_COVERAGE_LOW']['passed']
    assert checks['GROUND_HOLDOUT_RMS_HIGH']['value'] is None
    text = '\n'.join(ground_summary_lines({'status': result['status'], 'left_plane': result,
                                          'valid_depth_fraction': .8}))
    assert '80.0%' in text and '50.0%' in text and '0.0%' in text and '≥70%' in text
    assert '无可用值' in text and '留出一致点不足' in text


def test_patch_refinement_recovers_tilted_plane_from_noisy_depth_without_filling_pixels():
    group, calibration, region, normal = scene((.05, -.08, -1.), distance=800.)
    add_ground_images(group, calibration, normal, distance=800.)
    rng = np.random.default_rng(810)
    for role in ('left', 'right'):
        getattr(group['depth'], role+'_depth_mm')[:] += rng.normal(0, 8., (480, 640))
    original = group['depth'].left_depth_mm.copy()
    check = check_ground(group, calibration, region)
    assert check['raw_depth_plane']['status'] == 'GROUND_PLANE_INCONSISTENT'
    assert check['status'] == 'CONSISTENT_GROUND_PLANE'
    assert check['camera_to_ground_perpendicular_mm'] == pytest.approx(800., abs=1.)
    assert np.dot(normal, check['left_plane']['normal_camera']) > .99999
    for role in ('left', 'right'):
        assert check['subpixel_refinement'][role]['retained_fraction'] > .8
        assert check[role+'_plane']['holdout_inlier_fraction'] >= .7
    assert check['observed_support']['sampling'] == 'SPARSE_REFINED_STEREO_MATCHES'
    assert check['observed_support']['pixel_count'] <= 4096
    assert check['observed_support']['inferred_pixels'] == 0
    np.testing.assert_array_equal(group['depth'].left_depth_mm, original)


@pytest.mark.parametrize('fault', ['textureless', 'epipolar', 'unrelated', 'size', 'missing_eye', 'intrinsic', 'camera_size'])
def test_subpixel_ground_rejects_bad_image_evidence_even_with_perfect_seed_plane(fault):
    group, calibration, region, normal = scene((.05, -.08, -1.), distance=800.)
    add_ground_images(group, calibration, normal, distance=800.)
    if fault == 'textureless':
        group['left'][:] = 100; group['right'][:] = 100
    elif fault == 'epipolar':
        group['right'] = np.roll(group['right'], 3, axis=0)
    elif fault == 'unrelated':
        group['right'] = np.random.default_rng(25).integers(0, 256, group['right'].shape, np.uint8)
    elif fault == 'size':
        group['right'] = group['right'][:200]
    elif fault == 'missing_eye':
        del group['right']
    elif fault == 'intrinsic':
        calibration.right = SimpleNamespace(**vars(calibration.right))
        calibration.right.rectified_intrinsic = calibration.right.rectified_intrinsic.copy()
        calibration.right.rectified_intrinsic[0, 2] += 10.
    else:
        calibration.right = SimpleNamespace(**vars(calibration.right))
        calibration.right.width -= 20
    check = check_ground(group, calibration, region)
    assert check['status'] in {'INSUFFICIENT_GROUND_SUBPIXEL_SUPPORT', 'GROUND_IMAGE_PAIR_INVALID',
                              'GROUND_RECTIFIED_GEOMETRY_UNSUPPORTED'}
    assert 'camera_to_ground_perpendicular_mm' not in check
