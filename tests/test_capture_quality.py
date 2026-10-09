"""Diagnostics describe pixel failures without overriding measurement gates."""
import unittest
import cv2
import numpy as np

from pipe_twin.capture_quality import measure_patch, patch_mask, summarize_pair, diagnostic_record
from pipe_twin.local_surface import color_candidate_mask


class CaptureQualityTests(unittest.TestCase):
    def test_independent_roi_exposes_blue_candidate_miss(self):
        image = np.full((10, 10, 3), [255, 169, 95], np.uint8)
        mask = np.ones((10, 10), bool)
        original = image.copy()
        result = measure_patch(image, "#0000FF", mask)
        self.assertEqual(result["color_coverage_percent"], 0)
        self.assertEqual(result["color_level"], "RED")
        self.assertEqual(result["highlight_risk_percent"], 0)
        self.assertEqual(result["sampled_color_srgb"], "#5FA9FF")
        sampled = measure_patch(image, "#5FA9FF", mask)
        self.assertEqual(sampled["color_coverage_percent"], 100)
        np.testing.assert_array_equal(image, original)

    def test_highlight_requires_two_channels_and_reports_actual_area(self):
        image = np.full((10, 10, 3), [0, 0, 255], np.uint8)
        image[0, :5] = [249, 250, 250]
        result = measure_patch(image, "#FF0000", np.ones((10, 10), bool))
        self.assertEqual(result["highlight_risk_percent"], 5)
        self.assertEqual(result["highlight_level"], "AMBER")
        image[0, 5] = [250, 250, 250]
        self.assertEqual(measure_patch(image, "#FF0000", np.ones((10, 10), bool))["highlight_level"], "RED")

    def test_missing_eye_never_becomes_a_passing_pair(self):
        good = measure_patch(np.full((8, 8, 3), [160, 40, 20], np.uint8), "#1428A0", np.ones((8, 8), bool))
        summary = summarize_pair({"left": good})
        self.assertEqual(summary["status"], "INCOMPLETE")
        self.assertIsNone(summary["color_coverage_percent"])
        self.assertIsNone(summary["highlight_risk_percent"])
        self.assertEqual(summary["color_level"], "PENDING")

    def test_color_warning_boundaries(self):
        for matching, level in ((19, "RED"), (20, "AMBER"), (59, "AMBER"), (60, "GREEN")):
            with self.subTest(matching=matching):
                image = np.full((10, 10, 3), 100, np.uint8)
                image.reshape(-1, 3)[:matching] = [0, 0, 255]
                value = measure_patch(image, "#FF0000", np.ones((10, 10), bool))
                self.assertAlmostEqual(value["color_coverage_percent"], matching)
                self.assertEqual(value["color_level"], level)

    def test_pair_uses_worse_eye_not_mean(self):
        left = {"color_coverage_percent": 90, "highlight_risk_percent": 0, "underexposed": False}
        right = {"color_coverage_percent": 10, "highlight_risk_percent": 30, "underexposed": False}
        summary = summarize_pair({"left": left, "right": right})
        self.assertEqual(summary["color_coverage_percent"], 10)
        self.assertEqual(summary["highlight_risk_percent"], 30)
        self.assertEqual(summary["color_level"], "RED")
        self.assertIn("先调灯光", summary["advice"][0])

    def test_dark_patch_cannot_masquerade_as_good_low_highlight(self):
        dark = measure_patch(np.zeros((8, 8, 3), np.uint8), "#000000", np.ones((8, 8), bool))
        self.assertEqual(dark["highlight_risk_percent"], 0)
        self.assertIsNone(dark["sampled_color_srgb"])
        summary = summarize_pair({"left": dark, "right": dark})
        self.assertEqual(summary["status"], "UNDEREXPOSED")
        self.assertIn("偏暗", summary["advice"][0])
        self.assertNotIn("两项取样指标较好", "".join(summary["advice"]))

    def test_mask_excludes_background_and_matches_actual_candidate_rule(self):
        rng = np.random.default_rng(10)
        image = rng.integers(0, 256, (40, 50, 3), dtype=np.uint8)
        mask = patch_mask(image, [10, 12, 20, 20])
        expected = np.mean(color_candidate_mask(cv2.cvtColor(image, cv2.COLOR_BGR2LAB), "#30A0C0")[mask]) * 100
        self.assertAlmostEqual(measure_patch(image, "#30A0C0", mask)["color_coverage_percent"], expected)
        self.assertEqual(int(mask.sum()), 400)
        for region in ([0, 0, 7, 8], [-1, 0, 8, 8], [42, 0, 9, 8], [0, 0, 8.0, 8]):
            with self.subTest(region=region), self.assertRaises(ValueError):
                patch_mask(image, region)

    def test_saved_record_is_advisory_and_detached(self):
        hashes = {"left": "a" * 64, "right": "b" * 64}
        samples = [{"name": "blue", "left_region_px": [0, 0, 8, 8]}]
        record = diagnostic_record(hashes, samples)
        samples[0]["left_region_px"][0] = 100
        hashes["left"] = "changed"
        self.assertEqual(record["samples"][0]["left_region_px"][0], 0)
        self.assertEqual(record["photo_sha256"]["left"], "a" * 64)
        self.assertIn("NOT_IDENTITY_OR_MEASUREMENT", record["scope"])
