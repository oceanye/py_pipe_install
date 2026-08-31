from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import numpy as np

from pipe_twin.detector import ColorDiameterDetector


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "test_model" / "manifest.json"

BLUE_ID = "pipe-blue-d45"
RED_ID = "pipe-red-d40"
WHITE_ID = "pipe-white-d20"
ALL_PIPE_IDS = {BLUE_ID, RED_ID, WHITE_ID}

SYNTHETIC_PIPES = {
    BLUE_ID: {"bgr": (250, 54, 0), "diameter": 45, "y": 300},
    RED_ID: {"bgr": (0, 0, 255), "diameter": 40, "y": 450},
    WHITE_ID: {"bgr": (255, 254, 254), "diameter": 20, "y": 600},
}


def _synthetic_frame(*visible_pipe_ids: str) -> np.ndarray:
    frame = np.full((1080, 1920, 3), 128, dtype=np.uint8)
    for pipe_id in visible_pipe_ids:
        spec = SYNTHETIC_PIPES[pipe_id]
        y0 = spec["y"]
        # A slightly-over-500 px long side clears the manifest's minimum after
        # robust percentile trimming while remaining inside its diameter gate.
        frame[y0 : y0 + spec["diameter"], 400:950] = spec["bgr"]
    return frame


def _observations_by_id(result: dict) -> dict[str, dict]:
    observations = result["observations"]
    if isinstance(observations, dict):
        return observations
    return {item["pipe_id"]: item for item in observations}


def _reported_diameter(observation: dict) -> float:
    value = observation.get("diameter_mm", observation.get("nominal_diameter_mm"))
    if value is None:
        raise AssertionError("Observation must expose its model-prior diameter")
    return float(value)


class ColorDiameterDetectorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        cls.detector = ColorDiameterDetector(cls.manifest)
        cls.nominal_diameters = {
            item["pipe_id"]: float(item["nominal_diameter_mm"])
            for item in cls.manifest["model"]["pipes"]
        }

    def test_detects_all_three_manifest_pipes_in_a_controlled_bgr_frame(self) -> None:
        result = self.detector.detect(_synthetic_frame(*ALL_PIPE_IDS))
        observations = _observations_by_id(result)

        self.assertEqual(set(result["visible_pipe_ids"]), ALL_PIPE_IDS)
        self.assertEqual(set(observations), ALL_PIPE_IDS)
        for pipe_id in ALL_PIPE_IDS:
            with self.subTest(pipe_id=pipe_id):
                observation = observations[pipe_id]
                self.assertEqual(observation["visibility"], "VISIBLE")
                self.assertEqual(observation["installation_state"], "UNKNOWN")
                self.assertEqual(observation["diameter_source"], "model_prior")
                self.assertIs(observation["metric_calibrated"], False)
                self.assertAlmostEqual(
                    _reported_diameter(observation),
                    self.nominal_diameters[pipe_id],
                )
                self.assertGreater(
                    float(observation["projected_diameter_estimate_mm"]), 0.0
                )

    def test_absent_pipes_are_explicitly_not_observed_and_never_missing(self) -> None:
        result = self.detector.detect(_synthetic_frame(RED_ID))
        observations = _observations_by_id(result)

        self.assertEqual(set(result["visible_pipe_ids"]), {RED_ID})
        self.assertEqual(set(observations), ALL_PIPE_IDS)
        self.assertEqual(observations[RED_ID]["visibility"], "VISIBLE")
        for pipe_id in (BLUE_ID, WHITE_ID):
            with self.subTest(pipe_id=pipe_id):
                self.assertEqual(observations[pipe_id]["visibility"], "NOT_OBSERVED")
                self.assertEqual(observations[pipe_id]["installation_state"], "UNKNOWN")

    def test_blank_frame_safely_returns_unknown_installation_state(self) -> None:
        result = self.detector.detect(_synthetic_frame())
        observations = _observations_by_id(result)

        self.assertEqual(list(result["visible_pipe_ids"]), [])
        self.assertEqual(set(observations), ALL_PIPE_IDS)
        for observation in observations.values():
            self.assertEqual(observation["visibility"], "NOT_OBSERVED")
            self.assertEqual(observation["installation_state"], "UNKNOWN")
            self.assertNotEqual(observation["installation_state"], "MISSING")
            self.assertEqual(observation["diameter_source"], "model_prior")
            self.assertIs(observation["metric_calibrated"], False)

    def test_small_or_outside_roi_color_distractors_are_rejected(self) -> None:
        frame = _synthetic_frame()
        frame[300:320, 400:420] = (0, 0, 255)  # Below minimum component area.
        frame[20:100, 20:900] = (250, 54, 0)  # Outside the manifest analysis ROI.
        frame[650:750, 500:600] = (255, 254, 254)  # Aspect ratio below threshold.

        result = self.detector.detect(frame)

        self.assertEqual(list(result["visible_pipe_ids"]), [])
        for observation in _observations_by_id(result).values():
            self.assertEqual(observation["installation_state"], "UNKNOWN")

    def test_requires_a_three_channel_bgr_frame(self) -> None:
        invalid = np.zeros((1080, 1920), dtype=np.uint8)
        with self.assertRaises(ValueError):
            self.detector.detect(invalid)

    def test_rejects_manifests_that_claim_capabilities_beyond_m0(self) -> None:
        mutations = (
            ("metric_calibrated", True),
            ("layer_model", "multi"),
            ("supports_stereo", True),
            ("supports_occlusion_reasoning", True),
            ("supports_3dgs_training", True),
        )

        for key, value in mutations:
            with self.subTest(scope_key=key, value=value):
                manifest = copy.deepcopy(self.manifest)
                manifest["scope"][key] = value
                with self.assertRaises(ValueError):
                    ColorDiameterDetector(manifest)


if __name__ == "__main__":
    unittest.main()
