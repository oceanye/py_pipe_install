"""Scaled stereo bundle adjustment using training corners only.

Both cameras use Brown k1/k2/p1/p2 (k3 stays zero). The board supplies metric
scale; the baseline remains free and is checked by the caller, not imposed.
Analytic projection/composition derivatives avoid slow mixed-unit finite steps.
"""
from __future__ import annotations

import cv2
import numpy as np


def camera_parameters(k: np.ndarray, d: np.ndarray) -> np.ndarray:
    return np.r_[np.log([k[0, 0], k[1, 1]]), k[0, 2], k[1, 2], np.ravel(d)[:4]]


def decode_camera(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (np.array([[np.exp(p[0]), 0., p[2]], [0., np.exp(p[1]), p[3]], [0., 0., 1.]]),
            np.r_[p[4:8], 0.])


class StereoBundle:
    """22 global camera/rig parameters, then six parameters per board pose."""

    def __init__(self, objects: np.ndarray, observations: np.ndarray) -> None:
        self.objects = np.asarray(objects, np.float64).reshape(-1, 3)
        self.observations = np.asarray(observations, np.float64)
        self.last_x: np.ndarray | None = None

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.last_x is not None and np.array_equal(x, self.last_x):
            return self.residual, self.jacobian
        k1, d1 = decode_camera(x[:8])
        k2, d2 = decode_camera(x[8:16])
        rs, ts = x[16:19], x[19:22]
        block = len(self.objects) * 2
        residual = np.empty(len(self.observations) * 2 * block)
        jacobian = np.zeros((len(residual), len(x)))
        for i, (obs_l, obs_r) in enumerate(self.observations):
            start = 22 + i * 6
            rb, tb = x[start:start + 3], x[start + 3:start + 6]
            comp = cv2.composeRT(rb, tb, rs, ts)
            rr, tr = comp[:2]
            pl, jl = cv2.projectPoints(self.objects, rb, tb, k1, d1)
            pr, jr = cv2.projectPoints(self.objects, rr, tr, k2, d2)
            a = slice(i * 2 * block, (i * 2 + 1) * block)
            b = slice((i * 2 + 1) * block, (i * 2 + 2) * block)
            residual[a] = (pl.reshape(-1, 2) - obs_l).ravel()
            residual[b] = (pr.reshape(-1, 2) - obs_r).ravel()
            jacobian[a, :8] = jl[:, 6:14]
            jacobian[a, 0] *= k1[0, 0]
            jacobian[a, 1] *= k1[1, 1]
            jacobian[a, start:start + 6] = jl[:, :6]
            jacobian[b, 8:16] = jr[:, 6:14]
            jacobian[b, 8] *= k2[0, 0]
            jacobian[b, 9] *= k2[1, 1]
            jacobian[b, 16:19] = jr[:, :3] @ comp[4] + jr[:, 3:6] @ comp[8]
            jacobian[b, 19:22] = jr[:, 3:6]
            jacobian[b, start:start + 3] = jr[:, :3] @ comp[2] + jr[:, 3:6] @ comp[6]
            jacobian[b, start + 3:start + 6] = jr[:, :3] @ comp[3] + jr[:, 3:6] @ comp[7]
        self.last_x = x.copy()
        self.residual, self.jacobian = residual, jacobian
        return residual, jacobian

    def fun(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)[0]

    def jac(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x)[1]


def refine_stereo_bundle(*, objects, left_points, right_points, K1, D1, K2, D2, R, T,
                         max_nfev: int = 300) -> dict:
    from scipy.optimize import least_squares

    observations = np.asarray([[np.asarray(left).reshape(-1, 2),
                                np.asarray(right).reshape(-1, 2)]
                               for left, right in zip(left_points, right_points)], dtype=float)
    bundle = StereoBundle(objects, observations)
    poses = []
    for left, _right in observations:
        ok, rv, tv = cv2.solvePnP(bundle.objects, left, K1, D1)
        if not ok or np.any((bundle.objects @ cv2.Rodrigues(rv)[0].T + tv.ravel())[:, 2] <= 0):
            raise ValueError("联合求解初始化未得到有效棋盘姿态")
        poses.extend(np.r_[rv.ravel(), tv.ravel()])
    initial = np.r_[camera_parameters(K1, D1), camera_parameters(K2, D2),
                    cv2.Rodrigues(R)[0].ravel(), np.ravel(T), poses]
    options = dict(jac=bundle.jac, method="lm", x_scale="jac", max_nfev=max_nfev,
                   ftol=1e-10, xtol=1e-10, gtol=1e-10)
    fitted = least_squares(bundle.fun, initial, **options)
    # A success flag alone can mean a small cost change along a flat direction.
    # Restart from the solution and check that camera parameters also settle.
    checked = least_squares(bundle.fun, fitted.x.copy(), **options)
    k1, d1 = decode_camera(checked.x[:8])
    k2, d2 = decode_camera(checked.x[8:16])
    pk1, pd1 = decode_camera(fitted.x[:8])
    pk2, pd2 = decode_camera(fitted.x[8:16])
    steps = {
        "intrinsic_px": float(max(np.max(np.abs(k1 - pk1)), np.max(np.abs(k2 - pk2)))),
        "distortion": float(max(np.max(np.abs(d1 - pd1)), np.max(np.abs(d2 - pd2)))),
        "rotation_rad": float(np.max(np.abs(checked.x[16:19] - fitted.x[16:19]))),
        "translation_mm": float(np.max(np.abs(checked.x[19:22] - fitted.x[19:22]))),
    }
    limits = {"intrinsic_px": .01, "distortion": 1e-5,
              "rotation_rad": 1e-6, "translation_mm": .001}
    rms = float(np.sqrt(np.mean(checked.fun ** 2) * 2))
    previous_rms = float(np.sqrt(np.mean(fitted.fun ** 2) * 2))
    converged = bool(fitted.success and checked.success and abs(rms - previous_rms) < 1e-7
                     and all(steps[key] <= limit for key, limit in limits.items()))
    per_view = np.sqrt(np.mean(checked.fun.reshape(len(observations), 2, -1, 2) ** 2,
                               axis=(2, 3)) * 2)
    positive = True
    for i in range(len(observations)):
        pose = checked.x[22 + 6*i:28 + 6*i]
        rb = cv2.Rodrigues(pose[:3])[0]
        left_3d = bundle.objects @ rb.T + pose[3:]
        right_3d = left_3d @ cv2.Rodrigues(checked.x[16:19])[0].T + checked.x[19:22]
        positive = positive and bool(np.all(left_3d[:, 2] > 0) and np.all(right_3d[:, 2] > 0))
    return {
        "K1": k1, "D1": d1, "K2": k2, "D2": d2,
        "R": cv2.Rodrigues(checked.x[16:19])[0], "T": checked.x[19:22].reshape(3, 1),
        "per_view": per_view, "stereo_rms": rms,
        "left_rms": float(np.sqrt(np.mean(per_view[:, 0] ** 2))),
        "right_rms": float(np.sqrt(np.mean(per_view[:, 1] ** 2))),
        "audit": {"algorithm": "scaled-stereo-bundle-v1", "converged": converged,
                  "positive_depth": positive, "training_pair_count": len(observations),
                  "baseline_constrained": False, "validation_used_for_selection": False,
                  "evaluations": [int(fitted.nfev), int(checked.nfev)],
                  "termination": [str(fitted.message), str(checked.message)],
                  "restart_parameter_steps": steps, "restart_step_limits": limits,
                  "restart_rms_change_px": abs(rms - previous_rms)},
    }
