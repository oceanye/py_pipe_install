from types import SimpleNamespace
import math
import numpy as np
import pytest

from pipe_twin.ground_check import check_ground, normalize_ground_region, ground_summary_lines


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
