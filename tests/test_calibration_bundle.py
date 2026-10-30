import cv2
import numpy as np

from pipe_twin.calibration_bundle import StereoBundle, camera_parameters, refine_stereo_bundle


def synthetic_bundle(noise=0.):
    obj = np.zeros((48, 3), float)
    obj[:, :2] = np.mgrid[:8, :6].T.reshape(-1, 2) * 17
    k = np.array([[1100., 0, 960], [0, 1080., 540], [0, 0, 1.]])
    d = np.array([.03, -.1, .001, -.002, 0.])
    rng = np.random.default_rng(456)
    rs, ts = np.array([.01, .02, -.005]), np.array([-60., .1, .9])
    poses = [np.r_[rng.uniform(-.4, .4, 3), rng.uniform(-150, 40, 2), rng.uniform(500, 1100)]
             for _ in range(12)]
    observations = []
    for pose in poses:
        rr, tr, *_ = cv2.composeRT(pose[:3], pose[3:], rs, ts)
        pair = [cv2.projectPoints(obj, pose[:3], pose[3:], k, d)[0].reshape(-1, 2),
                cv2.projectPoints(obj, rr, tr, k, d)[0].reshape(-1, 2)]
        observations.append(np.asarray(pair) + rng.normal(0, noise, (2, 48, 2)))
    truth = np.r_[camera_parameters(k, d), camera_parameters(k, d), rs, ts, np.ravel(poses)]
    return StereoBundle(obj, observations), truth, k, d


def test_analytic_jacobian_matches_independent_finite_differences():
    bundle, truth, _k, _d = synthetic_bundle()
    jac = bundle.jac(truth).copy()
    for column in range(len(truth)):
        step = 1e-5 if column in (2, 3, 10, 11, 19, 20, 21) or column >= 22 and (column-22) % 6 >= 3 else 1e-6
        delta = np.zeros(len(truth)); delta[column] = step
        numeric = (bundle.fun(truth + delta) - bundle.fun(truth - delta)) / (2 * step)
        assert np.max(np.abs(jac[:, column] - numeric)) < 1e-4, column


def test_scaled_solve_recovers_unconstrained_metric_baseline_and_intrinsics():
    bundle, truth, k, d = synthetic_bundle(.08)
    seed = k.copy(); seed[0, 0] *= 1.04; seed[1, 1] *= .97
    result = refine_stereo_bundle(
        objects=bundle.objects, left_points=bundle.observations[:, 0],
        right_points=bundle.observations[:, 1], K1=seed, D1=d, K2=seed.copy(), D2=d,
        R=cv2.Rodrigues(truth[16:19] + .003)[0], T=truth[19:22] * 1.04)
    assert result['audit']['converged']
    assert result['audit']['positive_depth']
    assert result['audit']['baseline_constrained'] is False
    assert result['stereo_rms'] < .15
    assert abs(np.linalg.norm(result['T']) - np.linalg.norm(truth[19:22])) < .4
    assert abs(result['K1'][0, 0] - k[0, 0]) < 10


def test_exhausting_optimizer_budget_is_not_a_converged_result():
    bundle, truth, k, d = synthetic_bundle(.08)
    result = refine_stereo_bundle(
        objects=bundle.objects, left_points=bundle.observations[:, 0],
        right_points=bundle.observations[:, 1], K1=k, D1=d, K2=k, D2=d,
        R=cv2.Rodrigues(truth[16:19] + .1)[0], T=truth[19:22] * 1.1, max_nfev=1)
    assert not result['audit']['converged']
