from __future__ import annotations

import unittest

import numpy as np

from pipe_twin.elevation_depth import ElevationDepthError, analyze_elevation_groups


class _Depth:
    def __init__(self, shape: tuple[int, int], valid: bool = True) -> None:
        self.left_depth_mm = np.full(shape, 1000.0, dtype=np.float32)
        self.right_depth_mm = np.full(shape, 1000.0, dtype=np.float32)
        self.left_valid = np.full(shape, valid, dtype=bool)
        self.right_valid = np.full(shape, valid, dtype=bool)


def _group(image: np.ndarray, name: str, *, valid: bool = True, signature: str | None = None) -> dict:
    return {
        "left": image.copy(),
        "right": image.copy(),
        "depth": _Depth(image.shape[:2], valid),
        "pair_healthy": True,
        "capture_id": name,
        "pair_signature": signature or name,
    }


class ElevationDepthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = [{
            "pipe_id": "P001", "color_srgb": "#FF0000", "nominal_diameter_mm": 20,
            "left_region_px": [5, 5, 10, 10], "right_region_px": [5, 5, 10, 10],
        }]
        self.image = np.zeros((24, 32, 3), dtype=np.uint8)

    def test_installed_uses_colour_and_stereo_depth_without_registration(self) -> None:
        self.image[5:15, 5:15] = (0, 0, 255)
        report = analyze_elevation_groups([_group(self.image, "capture-1")], calibration=None, pipe_specs=self.spec)
        self.assertFalse(report["registration_required"])
        self.assertEqual(report["pipes"][0]["installation_state"], "INSTALLED")

    def test_two_independent_empty_captures_are_not_installed(self) -> None:
        report = analyze_elevation_groups(
            [_group(self.image, "capture-1"), _group(self.image, "capture-2")],
            calibration=None, pipe_specs=self.spec,
        )
        self.assertEqual(report["pipes"][0]["installation_state"], "NOT_INSTALLED")

    def test_repeated_same_image_does_not_count_twice(self) -> None:
        report = analyze_elevation_groups(
            [_group(self.image, "capture-1", signature="same"), _group(self.image, "capture-2", signature="same")],
            calibration=None, pipe_specs=self.spec,
        )
        self.assertEqual(report["pipes"][0]["installation_state"], "UNKNOWN")

    def test_occluded_or_invalid_depth_is_unknown(self) -> None:
        image = self.image.copy()
        image[5:15, 5:15] = (0, 0, 255)
        report = analyze_elevation_groups([_group(image, "capture-1", valid=False)], calibration=None, pipe_specs=self.spec)
        self.assertEqual(report["pipes"][0]["installation_state"], "UNKNOWN")

    def test_duplicate_regions_are_rejected_by_the_contract(self) -> None:
        bad = [dict(self.spec[0], pipe_id="P001")]
        with self.assertRaises(ElevationDepthError):
            analyze_elevation_groups([_group(self.image, "capture-1")], calibration=None, pipe_specs=self.spec + bad)

    def test_overlapping_same_colour_regions_are_unknown(self) -> None:
        image = self.image.copy()
        image[5:15, 5:15] = (0, 0, 255)
        second = dict(self.spec[0], pipe_id="P002", left_region_px=[6, 5, 10, 10], right_region_px=[6, 5, 10, 10])
        report = analyze_elevation_groups([_group(image, "capture-1")], calibration=None, pipe_specs=[self.spec[0], second])
        self.assertTrue(all(row["identity_ambiguous"] for row in report["pipes"]))
        self.assertTrue(all(row["installation_state"] == "UNKNOWN" for row in report["pipes"]))


if __name__ == "__main__":
    unittest.main()
