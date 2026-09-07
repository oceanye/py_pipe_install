from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from pipe_twin.measurement_book import (build_measurement_summary, corrected_diameter, diameter_correction,
    empty_book, load_book, new_sample, save_book, scope_id)
from pipe_twin.measurement_gui import image_rectangle


def manifest(capture="a", calibration="camera-1"):
    return {"stereo_calibration": {"calibration_id": calibration, "baseline_mm": 50},
            "capture": {"capture_groups": [{"capture_id": capture, "views": {"left": {"sha256": capture * 64}, "right": {"sha256": "r" * 64}}}]}}


def sample(capture="a", use="calibration", raw=39.5, reference=40.):
    return new_sample(manifest(capture), pipe_id="P1", kind="diameter", reference_mm=reference,
                      section_id="S01", use_for=use, raw_mm=raw)


class MeasurementBookTests(unittest.TestCase):
    def test_round_trip_preserves_actual_and_raw_values_and_sources(self):
        book = empty_book() | {"samples": [sample()]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "中文路径" / "实测.json"
            save_book(path, book)
            loaded = load_book(path)
        self.assertEqual(loaded, book)
        self.assertEqual(loaded["samples"][0]["raw_mm"], 39.5)
        self.assertEqual(loaded["samples"][0]["reference_mm"], 40.)

    def test_scope_uses_full_calibration_instead_of_only_label(self):
        first, changed = manifest(), manifest()
        changed["stereo_calibration"]["baseline_mm"] = 60
        self.assertNotEqual(scope_id(first), scope_id(changed))
        result = diameter_correction([sample()], scope_id(changed), 1)
        self.assertIsNone(result["offset_mm"])

    def test_reference_without_valid_raw_value_cannot_calibrate(self):
        entry = new_sample(manifest(), pipe_id="P1", kind="diameter", reference_mm=40., section_id="S1", use_for="calibration")
        self.assertEqual(diameter_correction([entry], scope_id(manifest()), 1)["status"], "NO_SAMPLES")

    def test_correction_does_not_validate_against_its_training_capture(self):
        entries = [sample(), sample(use="validation")]
        result = diameter_correction(entries, scope_id(manifest()), 1)
        self.assertEqual(result["validation_count"], 0)
        self.assertEqual(result["status"], "TRIAL")
        self.assertEqual(corrected_diameter(39.5, result, 1), 40.)
        self.assertIsNone(corrected_diameter(20., result, 1))

    def test_three_independent_capture_validations_and_duplicate_weight(self):
        entries = [sample(), sample()] + [sample(capture, "validation") for capture in ("b", "c", "d")]
        result = diameter_correction(entries, scope_id(manifest()), 1)
        self.assertEqual(result["training_count"], 1)
        self.assertEqual(result["status"], "VALIDATED_SAMPLES")
        entries[-1]["reference_mm"] = 44
        result = diameter_correction(entries, scope_id(manifest()), 1)
        self.assertEqual(result["status"], "TRIAL")
        self.assertEqual(result["validation_max_error_mm"], 4)

    def test_inconsistent_calibration_samples_suspend_correction(self):
        entries = [sample(), sample("b", raw=49, reference=55)]
        result = diameter_correction(entries, scope_id(manifest()), 1)
        self.assertEqual(result["status"], "INCONSISTENT")
        self.assertIsNone(corrected_diameter(39.5, result, 1))

    def test_invalid_or_stale_report_never_provides_dimensions_or_state(self):
        dashboard = {"binding_valid": False, "pipes": [{"pipe_id": "P1", "nominal_diameter_mm": 40., "installation_state": "UNKNOWN"}]}
        report = {"local_measurements": {"pipes": [{"pipe_id": "P1", "status": "MEASURED", "diameter_mm": 40}]}}
        result = build_measurement_summary(manifest(), report, dashboard, empty_book())
        self.assertEqual(result["pipes"][0]["installation_state"], "UNKNOWN")
        self.assertIsNone(result["pipes"][0]["raw_diameter_mm"])
        self.assertEqual(result["pairs"], [])

    def test_bad_numeric_reference_and_self_pair_are_rejected(self):
        for value in (float("nan"), -4, 0, True):
            with self.assertRaises(ValueError):
                new_sample(manifest(), pipe_id="P1", kind="diameter", reference_mm=value, section_id="S01")
        with self.assertRaises(ValueError):
            new_sample(manifest(), pipe_id="P1", pipe_id_b="P1", kind="clear_gap", reference_mm=10, section_id="S01")

    def test_malformed_metric_payload_is_discarded_without_a_gui_exception(self):
        dashboard = {"binding_valid": True, "pipes": [{"pipe_id": "P1", "nominal_diameter_mm": 40., "installation_state": "INSTALLED"}]}
        metric = {"schema_version": "1.0", "coordinate_frame": "CAD_WORLD_MM", "capture_id": "a", "calibration_id": "camera-1",
                  "pipes": [{"pipe_id": "P1", "status": "MEASURED", "reason_codes": [], "diameter_mm": "not-a-number"}], "pairs": []}
        for value in (metric, "malformed", metric | {"capture_id": "different"}):
            result = build_measurement_summary(manifest(), {"local_measurements": value}, dashboard, empty_book())
            self.assertIsNone(result["pipes"][0]["raw_diameter_mm"])
            self.assertIn("INVALID_LOCAL_REPORT", result["pipes"][0]["measurement_reasons"])

    def test_zoomed_roi_stays_in_original_image_coordinates(self):
        self.assertEqual(image_rectangle((100, 80), (300, 220), (2, 20, 40), (480, 240)), [40, 20, 100, 70])
        self.assertEqual(image_rectangle((-100, -100), (300, 300), (1, 0, 0), (200, 100)), [0, 0, 200, 100])
        self.assertIsNone(image_rectangle((0, 0), (2, 2), (1, 0, 0), (200, 100)))


if __name__ == "__main__":
    unittest.main()
