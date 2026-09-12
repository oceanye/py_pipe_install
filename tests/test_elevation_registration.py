from __future__ import annotations

import unittest

import numpy as np

from pipe_twin.elevation_registration import ElevationRegistrationError, register_elevation


def _specs(*, equal: bool = False) -> list[dict]:
    points = [(0, 0), (120, 0), (0, 100), (120, 100)]
    specs = []
    for index, (x, y) in enumerate(points):
        diameter = 25.0 if equal else 20.0 + index * 7.0
        specs.append({
            "pipe_id": f"P{index + 1}",
            "centerline_world_mm": [[x, y, -300], [x, y, 300]],
            "nominal_diameter_mm": diameter,
            "color_srgb": "#B0B0B0" if equal else ["#FF0000", "#00FF00", "#0000FF", "#FFFF00"][index],
        })
    return specs


def _observations(specs: list[dict], indices: list[int] | None = None, angle_deg: float = 31.0) -> list[dict]:
    indices = list(range(len(specs))) if indices is None else indices
    angle = np.deg2rad(angle_deg)
    r = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    result = []
    for i in indices:
        center = np.asarray(specs[i]["centerline_world_mm"], dtype=float).mean(axis=0)
        xy = r @ center[:2] + np.asarray([420.0, -180.0])
        result.append({
            "observation_id": f"O{i + 1}",
            "center_camera_mm": [xy[0], xy[1], 1400.0],
            "axis_camera": [0, 0, 1],
            "diameter_mm": specs[i]["nominal_diameter_mm"],
            "color_srgb": specs[i]["color_srgb"],
            "observed_segment_camera_mm": [[xy[0], xy[1], 1100], [xy[0], xy[1], 1700]],
        })
    return result


class ElevationRegistrationTests(unittest.TestCase):
    def test_known_oblique_metric_registration(self) -> None:
        specs = _specs()
        result = register_elevation(specs, _observations(specs), axis_world=[0, 0, 1])
        self.assertEqual(result["status"], "MATCHED")
        self.assertLess(result["rms_mm"], 1e-6)
        self.assertEqual({(m["pipe_id"], m["observation_id"]) for m in result["matches"]}, {(f"P{i}", f"O{i}") for i in range(1, 5)})
        self.assertFalse(result["axial_translation_observable"])
        self.assertEqual(result["scale"], 1.0)

    def test_partial_visible_subset_registers_without_fabricating_missing(self) -> None:
        specs = _specs()
        result = register_elevation(specs, _observations(specs, [0, 1, 2]), axis_world=[0, 0, 1])
        self.assertEqual(result["status"], "MATCHED")
        self.assertEqual(len(result["matches"]), 3)

    def test_observation_order_does_not_change_identity(self) -> None:
        specs = _specs()
        obs = list(reversed(_observations(specs, [0, 1, 2], angle_deg=17)))
        result = register_elevation(specs, obs, axis_world=[0, 0, 1])
        self.assertEqual(result["status"], "MATCHED")
        self.assertEqual({m["pipe_id"] for m in result["matches"]}, {"P1", "P2", "P3"})

    def test_symmetric_scene_is_ambiguous(self) -> None:
        specs = _specs(equal=True)
        result = register_elevation(specs, _observations(specs, angle_deg=0))
        self.assertEqual(result["status"], "AMBIGUOUS")
        self.assertTrue(any("ALTERNATIVE_" in reason for reason in result["reason_codes"]))

    def test_collinear_without_anchors_is_insufficient(self) -> None:
        specs = _specs()
        obs = _observations(specs, [0, 1, 2])
        for item in obs:
            item["center_camera_mm"][1] = 0.0
        result = register_elevation(specs, obs)
        self.assertEqual(result["status"], "INSUFFICIENT_OBSERVATIONS")

    def test_two_identity_anchors_can_resolve_collinear_subset(self) -> None:
        specs = _specs()
        obs = _observations(specs, [0, 1])
        result = register_elevation(specs, obs, anchors={"O1": "P1", "O2": "P2"})
        self.assertEqual(result["status"], "AMBIGUOUS")
        self.assertEqual(len(result["matches"]), 2)

    def test_nonparallel_axes_are_rejected(self) -> None:
        specs = _specs()
        obs = _observations(specs)
        obs[-1]["axis_camera"] = [0, 1, 1]
        result = register_elevation(specs, obs)
        self.assertEqual(result["status"], "MATCHED")
        self.assertIn("O4", result["rejected_observation_ids"])

    def test_empty_surface_is_insufficient(self) -> None:
        result = register_elevation(_specs(), [])
        self.assertEqual(result["status"], "INSUFFICIENT_OBSERVATIONS")
        self.assertEqual(result["camera_axis"], None)

    def test_anchor_cannot_override_diameter_mismatch(self) -> None:
        specs = _specs()
        obs = _observations(specs, [0, 1, 2], angle_deg=5)
        obs[0]["diameter_mm"] = 999
        result = register_elevation(specs, obs, anchors={"O1": "P1", "O2": "P2", "O3": "P3"})
        self.assertEqual(result["status"], "AMBIGUOUS")
        self.assertIn("ANCHOR_DIAMETER_OR_COLOR_MISMATCH", result["reason_codes"])

    def test_angle_search_limit_fails_closed(self) -> None:
        result = register_elevation(_specs(), _observations(_specs()), axis_world=[0, 0, 1], config={"max_hypotheses": 1})
        self.assertEqual(result["status"], "SEARCH_LIMIT")

    def test_arbitrary_camera_axis_reports_common_direction(self) -> None:
        specs = _specs()
        model_axis = np.asarray([0.0, 0.0, 1.0])
        camera_axis = np.asarray([0.3, -0.4, 0.8660254]); camera_axis /= np.linalg.norm(camera_axis)
        helper = np.asarray([1.0, 0.0, 0.0]); u = np.cross(camera_axis, helper); u /= np.linalg.norm(u); v = np.cross(camera_axis, u)
        B = np.column_stack((u, v, camera_axis)); angle = np.deg2rad(23); R2 = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        obs = []
        for index, spec in enumerate(specs):
            point = np.asarray(spec["centerline_world_mm"], dtype=float).mean(axis=0)
            q = B @ np.r_[R2 @ point[:2] + np.asarray([250.0, 80.0]), 1200.0]
            obs.append({"observation_id": f"O{index+1}", "center_camera_mm": q.tolist(), "axis_camera": camera_axis.tolist(), "diameter_mm": spec["nominal_diameter_mm"]})
        result = register_elevation(specs, obs, axis_world=model_axis)
        self.assertIn(result["status"], {"MATCHED", "AMBIGUOUS"})
        self.assertTrue(np.allclose(np.asarray(result["rotation_model_to_camera"]) @ model_axis, np.asarray(result["camera_axis"]), atol=1e-5))

    def test_nan_or_infinite_config_is_rejected(self) -> None:
        with self.assertRaises(ElevationRegistrationError):
            register_elevation(_specs(), _observations(_specs()), config={"max_residual_mm": float("nan")})
        with self.assertRaises(ElevationRegistrationError):
            register_elevation(_specs(), _observations(_specs()), config={"max_residual_mm": float("inf")})

    def test_compatible_diameter_but_wrong_position_is_rejected(self) -> None:
        specs = _specs()
        obs = _observations(specs, [0, 1, 2], angle_deg=17)
        obs[2]["center_camera_mm"][0] += 100.0
        result = register_elevation(specs, obs, axis_world=[0, 0, 1])
        self.assertIn(result["status"], {"AMBIGUOUS", "INSUFFICIENT_OBSERVATIONS"})

    def test_same_diameter_far_clutter_is_rejected_by_consensus(self) -> None:
        specs = _specs()
        obs = _observations(specs, [0, 1, 2], angle_deg=17)
        clutter = dict(obs[0], observation_id="CLUTTER")
        clutter["center_camera_mm"] = [9999.0, -8888.0, 1400.0]
        obs.append(clutter)
        result = register_elevation(specs, obs, axis_world=[0, 0, 1])
        self.assertEqual(result["status"], "MATCHED")
        self.assertEqual(result["matched_observation_count"], 3)
        self.assertIn("CLUTTER", result["rejected_observation_ids"])

    def test_reversed_observation_axis_and_true_3d_rotation(self) -> None:
        specs = _specs()
        rvec = np.asarray([0.3, -0.8, 0.4], dtype=float)
        theta = np.linalg.norm(rvec); k = rvec / theta
        K = np.asarray([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) * np.cos(theta) + (1 - np.cos(theta)) * np.outer(k, k) + np.sin(theta) * K
        obs = []
        for index, spec in enumerate(specs):
            center = np.asarray(spec["centerline_world_mm"], dtype=float).mean(axis=0)
            q = R @ center + np.asarray([300.0, -200.0, 1200.0])
            obs.append({"observation_id": f"O{index+1}", "center_camera_mm": q.tolist(), "axis_camera": (-R @ np.asarray([0., 0., 1.])).tolist(), "diameter_mm": spec["nominal_diameter_mm"]})
        result = register_elevation(specs, obs, axis_world=[0, 0, 1], config={"ambiguity_delta_mm": 0.1})
        self.assertIn(result["status"], {"MATCHED", "AMBIGUOUS"})
        self.assertTrue(np.isfinite(result["rms_mm"]))

    def test_random_metric_poses_with_axis_signs_stations_and_clutter(self) -> None:
        rng = np.random.default_rng(20260912)
        specs = _specs()
        model_axis = np.asarray([0., 0., 1.])
        for trial in range(20):
            rvec = rng.normal(size=3); rvec = rvec / np.linalg.norm(rvec) * rng.uniform(.2, 1.4)
            theta = np.linalg.norm(rvec); k = rvec / theta
            K = np.asarray([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            true_r = np.eye(3) * np.cos(theta) + (1 - np.cos(theta)) * np.outer(k, k) + np.sin(theta) * K
            true_t = np.asarray([400., -250., 1300.]); camera_axis = true_r @ model_axis
            obs = []
            for index, spec in enumerate(specs):
                center = np.asarray(spec["centerline_world_mm"], dtype=float).mean(axis=0)
                q = true_r @ center + true_t + camera_axis * rng.uniform(-200, 200) + rng.normal(0, .3, 3)
                obs.append({"observation_id": f"O{index+1}", "center_camera_mm": q.tolist(), "axis_camera": (camera_axis * (1 if rng.random() > .5 else -1)).tolist(), "diameter_mm": spec["nominal_diameter_mm"]})
            obs.append({"observation_id": "CLUTTER", "center_camera_mm": [9999., 8888., 7777.], "axis_camera": camera_axis.tolist(), "diameter_mm": 27.})
            result = register_elevation(specs, obs, axis_world=model_axis, config={"max_residual_mm": 2.0})
            self.assertEqual(result["status"], "MATCHED", f"trial {trial}: {result}")
            self.assertEqual(result["matched_observation_count"], 4)
            self.assertIn("CLUTTER", result["rejected_observation_ids"])
            self.assertTrue(all(float(match["residual_mm"]) < 2.0 for match in result["matches"]))
            recovered_r = np.asarray(result["rotation_model_to_camera"])
            self.assertAlmostEqual(float(np.linalg.det(recovered_r)), 1.0, places=5)
            self.assertEqual(result["scale"], 1.0)


if __name__ == "__main__":
    unittest.main()
