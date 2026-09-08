from __future__ import annotations

import copy
import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock


MODEL_HASH = "a" * 64


class AnalysisDiagnosticTests(unittest.TestCase):
    def test_all_unknown_field_run_explains_calibration_and_depth(self):
        from pipe_twin.gui import summarize_stereo_diagnostics

        manifest = {
            "stereo_calibration": {
                "calibration_id": "PG2-SYNTHETIC-EXACT-V1",
            }
        }
        report = {
            "counts": {"INSTALLED": 0, "NOT_INSTALLED": 0, "UNKNOWN": 1},
            "pipes": [{"pipe_id": "P001", "installation_state": "UNKNOWN"}],
            "calibration_audit": {"registration_validated": False},
            "capture_audit": {
                "groups": [
                    {
                        "depth_audit": {
                            "valid_left_fraction": 0.09,
                            "valid_right_fraction": 0.11,
                        },
                        "pipe_evidence": {
                            "P001": {
                                "left": {
                                    "reason_codes": [
                                        "INSUFFICIENT_VALID_DEPTH",
                                        "TARGET_COLOR_INTERSECTION_GATE_FAILED",
                                    ]
                                }
                            }
                        },
                    }
                ]
            },
        }
        message = summarize_stereo_diagnostics(manifest, report)
        self.assertIn("全部不确定", message)
        self.assertIn("合成演示标定", message)
        self.assertIn("10.0%", message)


def _manifest() -> dict:
    return {
        "model": {
            "path": "pipes.3dm",
            "sha256": MODEL_HASH,
            "pipes": [
                {
                    "pipe_id": "P-RED-22",
                    "cad_object_id": "guid-red",
                    "layer_id": "front",
                    "appearance_color": "red",
                    "nominal_color_srgb": "#FF0000",
                    "nominal_diameter_mm": 22.0,
                    "centerline_world_mm": [[0.0, 50.0, 0.0], [500.0, 50.0, 0.0]],
                },
                {
                    "pipe_id": "P-BLUE-50",
                    "cad_object_id": "guid-blue",
                    "layer_id": "back",
                    "appearance_color": "blue",
                    "nominal_color_srgb": "#0000FF",
                    "nominal_diameter_mm": 50.0,
                    "centerline_world_mm": [[0.0, 10.0, -100.0], [500.0, 10.0, -100.0]],
                },
            ],
        }
    }


def _view(state: str) -> dict:
    direct = state in {"FULLY_VISIBLE", "PARTIALLY_OCCLUDED"}
    negative = state == "NOT_OBSERVED"
    return {
        "projection_geometry_source": "cad_triangle_mesh",
        "occlusion_state": state,
        "assessable": state not in {"FULLY_OCCLUDED", "OUT_OF_FRUSTUM"},
        "installation_evidence": (
            "DIRECT_STEREO_CAD_EVIDENCE"
            if direct
            else "NEGATIVE_FREE_SPACE_CANDIDATE"
            if negative
            else "INCONCLUSIVE"
        ),
        "reason_codes": [],
        "expected_region_in_frame": state != "OUT_OF_FRUSTUM",
        "expected_region_unoccluded": state not in {"FULLY_OCCLUDED", "OUT_OF_FRUSTUM"},
        "direct_instance_evidence": direct,
        "negative_candidate": negative,
        "amodal_bbox_xywh": [10, 20, 100, 30],
    }


def _report() -> dict:
    return {
        "model": {"sha256": MODEL_HASH, "verified": True},
        "pipes": [
            {
                "pipe_id": "P-RED-22",
                "installation_state": "INSTALLED",
                "state_basis": "DIRECT_STEREO_EVIDENCE",
                "reason_codes": ["VISIBLE_IN_LEFT"],
                "visibility_by_view": {
                    "left": _view("FULLY_VISIBLE"),
                    "right": _view("PARTIALLY_OCCLUDED"),
                },
            },
            {
                "pipe_id": "P-BLUE-50",
                "installation_state": "NOT_INSTALLED",
                "state_basis": "QUALIFIED_NEGATIVE_EVIDENCE",
                "reason_codes": ["FREE_SPACE_VALIDATED"],
                "visibility_by_view": {
                    "left": _view("NOT_OBSERVED"),
                    "right": _view("NOT_OBSERVED"),
                },
            },
        ],
    }


def _schema_two_manifest_report() -> tuple[dict, dict]:
    manifest = _manifest()
    manifest.update(
        {
            "schema_version": "2.0",
            "model_revision": "model-r1",
            "capture": {"capture_group_id": "capture-run-1"},
            "stereo_calibration": {"calibration_id": "calibration-1"},
        }
    )
    report = _report()
    report.update(
        {
            "schema_version": "2.0",
            "model_revision": "model-r1",
            "capture_group_id": "capture-run-1",
            "calibration_audit": {"calibration_id": "calibration-1"},
            "inputs": {"manifest": {"sha256": "b" * 64}},
        }
    )
    for result, pipe in zip(report["pipes"], manifest["model"]["pipes"], strict=True):
        result["cad_object_id"] = pipe["cad_object_id"]
    return manifest, report


class DashboardModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.gui = importlib.import_module("pipe_twin.gui")

    def test_valid_binding_preserves_report_states_and_manifest_order(self) -> None:
        dashboard = self.gui.build_dashboard_model(_manifest(), _report())

        self.assertTrue(dashboard["binding_valid"])
        self.assertEqual(
            [item["pipe_id"] for item in dashboard["pipes"]],
            ["P-RED-22", "P-BLUE-50"],
        )
        self.assertEqual(
            [item["installation_state"] for item in dashboard["pipes"]],
            ["INSTALLED", "NOT_INSTALLED"],
        )
        self.assertEqual(
            dashboard["counts"],
            {"INSTALLED": 1, "NOT_INSTALLED": 1, "UNKNOWN": 0},
        )
        self.assertEqual(dashboard["model"]["format"], "3dm")

    def test_missing_report_fails_closed_and_uses_requested_unknown_label(self) -> None:
        dashboard = self.gui.build_dashboard_model(_manifest(), None)

        self.assertFalse(dashboard["binding_valid"])
        self.assertEqual(
            {pipe["installation_state"] for pipe in dashboard["pipes"]}, {"UNKNOWN"}
        )
        self.assertTrue(
            all(pipe["installation_state_zh"] == "不确定" for pipe in dashboard["pipes"])
        )
        self.assertIn("NO_REPORT", {item["code"] for item in dashboard["binding_issues"]})

    def test_model_hash_mismatch_fails_closed_for_every_pipe(self) -> None:
        report = _report()
        report["model"]["sha256"] = "b" * 64

        dashboard = self.gui.build_dashboard_model(_manifest(), report)

        self.assertFalse(dashboard["binding_valid"])
        self.assertEqual(dashboard["counts"]["UNKNOWN"], 2)
        self.assertIn(
            "MODEL_HASH_MISMATCH",
            {item["code"] for item in dashboard["binding_issues"]},
        )

    def test_pipe_id_set_mismatch_fails_closed_for_every_pipe(self) -> None:
        report = _report()
        report["pipes"][1]["pipe_id"] = "FOREIGN-PIPE"

        dashboard = self.gui.build_dashboard_model(_manifest(), report)

        self.assertFalse(dashboard["binding_valid"])
        self.assertEqual(dashboard["counts"]["UNKNOWN"], 2)
        self.assertIn(
            "REPORT_PIPE_ID_SET_MISMATCH",
            {item["code"] for item in dashboard["binding_issues"]},
        )

    def test_unknown_state_enum_fails_closed_for_every_pipe(self) -> None:
        report = _report()
        report["pipes"][0]["installation_state"] = "MISSING"

        dashboard = self.gui.build_dashboard_model(_manifest(), report)

        self.assertFalse(dashboard["binding_valid"])
        self.assertEqual(dashboard["counts"]["UNKNOWN"], 2)
        self.assertIn(
            "REPORT_STATE_INVALID",
            {item["code"] for item in dashboard["binding_issues"]},
        )

    def test_total_stereo_occlusion_suppresses_decisive_state(self) -> None:
        report = _report()
        report["pipes"][0]["visibility_by_view"] = {
            "left": _view("FULLY_OCCLUDED"),
            "right": _view("FULLY_OCCLUDED"),
        }

        dashboard = self.gui.build_dashboard_model(_manifest(), report)
        pipe = dashboard["pipes"][0]

        self.assertTrue(dashboard["binding_valid"])
        self.assertEqual(pipe["installation_state"], "UNKNOWN")
        self.assertEqual(pipe["state_basis"], "GUI_TOTAL_OCCLUSION_FAIL_CLOSED")
        self.assertIn(
            "DECISIVE_STATE_WITH_TOTAL_OCCLUSION",
            {item["code"] for item in dashboard["display_issues"]},
        )

    def test_state_style_is_independent_from_physical_pipe_color(self) -> None:
        dashboard = self.gui.build_dashboard_model(_manifest(), _report())
        red_pipe = dashboard["pipes"][0]

        self.assertEqual(red_pipe["material_color_srgb"], "#FF0000")
        self.assertEqual(
            red_pipe["state_color"], self.gui.STATE_PRESENTATION["INSTALLED"]["color"]
        )
        self.assertNotEqual(red_pipe["material_color_srgb"], red_pipe["state_color"])

    def test_projection_is_deterministic_and_stays_inside_canvas(self) -> None:
        dashboard = self.gui.build_dashboard_model(_manifest(), _report())

        first = self.gui._project_dashboard_pipes(dashboard, 900, 600, "isometric")
        second = self.gui._project_dashboard_pipes(dashboard, 900, 600, "isometric")

        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        for pipe in first:
            for x, y in pipe["canvas_centerline"]:
                self.assertGreaterEqual(x, 0)
                self.assertLessEqual(x, 900)
                self.assertGreaterEqual(y, 0)
                self.assertLessEqual(y, 600)

    def test_inputs_model_actual_hash_is_accepted(self) -> None:
        report = _report()
        report.pop("model")
        report["inputs"] = {
            "model": {
                "actual_sha256": MODEL_HASH,
                "expected_sha256": MODEL_HASH,
                "verified": True,
            }
        }

        dashboard = self.gui.build_dashboard_model(_manifest(), report)

        self.assertTrue(dashboard["binding_valid"])

    def test_build_does_not_mutate_inputs(self) -> None:
        manifest = _manifest()
        report = _report()
        original_manifest = copy.deepcopy(manifest)
        original_report = copy.deepcopy(report)

        self.gui.build_dashboard_model(manifest, report)

        self.assertEqual(manifest, original_manifest)
        self.assertEqual(report, original_report)

    def test_capture_groups_populate_manifest_bound_stereo_previews(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "capture.json"
            left = root / "left.png"
            right = root / "right.png"
            manifest_path.write_text("{}", encoding="utf-8")
            left.write_bytes(b"left")
            right.write_bytes(b"right")
            manifest = {
                "capture": {
                    "capture_groups": [
                        {
                            "views": {
                                "left": {"path": "left.png"},
                                "right": {"path": "right.png"},
                            }
                        }
                    ]
                }
            }

            paths = self.gui._first_manifest_stereo_paths(manifest_path, manifest)

            self.assertEqual(paths, {"left": left, "right": right})

    def test_stereo_preview_uses_the_latest_capture_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "capture.json"
            manifest_path.write_text("{}", encoding="utf-8")
            for name in ("old-left.png", "old-right.png", "new-left.png", "new-right.png"):
                (root / name).write_bytes(name.encode("ascii"))
            manifest = {
                "capture": {
                    "capture_groups": [
                        {
                            "views": {
                                "left": {"path": "old-left.png"},
                                "right": {"path": "old-right.png"},
                            }
                        },
                        {
                            "views": {
                                "left": {"path": "new-left.png"},
                                "right": {"path": "new-right.png"},
                            }
                        },
                    ]
                }
            }

            paths = self.gui._first_manifest_stereo_paths(manifest_path, manifest)

            self.assertEqual(
                paths,
                {"left": root / "new-left.png", "right": root / "new-right.png"},
            )

    def test_schema_two_report_requires_full_capture_and_cad_binding(self) -> None:
        manifest = _manifest()
        manifest.update(
            {
                "schema_version": "2.0",
                "model_revision": "model-r1",
                "capture": {"capture_group_id": "capture-run-1"},
                "stereo_calibration": {"calibration_id": "calibration-1"},
            }
        )
        report = _report()
        report.update(
            {
                "schema_version": "2.0",
                "model_revision": "model-r1",
                "capture_group_id": "capture-run-1",
                "calibration_audit": {"calibration_id": "calibration-1"},
                "inputs": {"manifest": {"sha256": "b" * 64}},
            }
        )
        for result, pipe in zip(report["pipes"], manifest["model"]["pipes"], strict=True):
            result["cad_object_id"] = pipe["cad_object_id"]

        dashboard = self.gui.build_dashboard_model(
            manifest,
            report,
            manifest_sha256="b" * 64,
            model_actual_sha256=MODEL_HASH,
        )
        self.assertTrue(dashboard["binding_valid"])

        mutations = {
            "REPORT_MANIFEST_HASH_MISMATCH": lambda value: value["inputs"]["manifest"].update(
                sha256="c" * 64
            ),
            "MODEL_REVISION_MISMATCH": lambda value: value.update(model_revision="model-r0"),
            "CAPTURE_GROUP_MISMATCH": lambda value: value.update(capture_group_id="old-run"),
            "CALIBRATION_ID_MISMATCH": lambda value: value["calibration_audit"].update(
                calibration_id="old-calibration"
            ),
            "CAD_OBJECT_BINDING_MISMATCH": lambda value: value["pipes"][0].update(
                cad_object_id="guid-blue"
            ),
        }
        for expected_code, mutate in mutations.items():
            with self.subTest(expected_code=expected_code):
                stale = copy.deepcopy(report)
                mutate(stale)
                result = self.gui.build_dashboard_model(
                    manifest,
                    stale,
                    manifest_sha256="b" * 64,
                    model_actual_sha256=MODEL_HASH,
                )
                self.assertFalse(result["binding_valid"])
                self.assertEqual(result["counts"]["UNKNOWN"], 2)
                self.assertIn(
                    expected_code,
                    {issue["code"] for issue in result["binding_issues"]},
                )

    def test_schema_two_missing_left_or_right_view_fails_closed(self) -> None:
        manifest, report = _schema_two_manifest_report()
        report["pipes"][0]["visibility_by_view"].pop("right")

        dashboard = self.gui.build_dashboard_model(manifest, report)

        self.assertFalse(dashboard["binding_valid"])
        self.assertEqual(dashboard["counts"]["UNKNOWN"], 2)
        self.assertIn(
            "REPORT_VISIBILITY_INVALID",
            {issue["code"] for issue in dashboard["binding_issues"]},
        )

    def test_schema_two_invalid_evidence_or_occlusion_fails_closed_globally(self) -> None:
        manifest, report = _schema_two_manifest_report()
        report["pipes"][0]["visibility_by_view"]["left"][
            "installation_evidence"
        ] = "FORGED_EVIDENCE"
        report["pipes"][1]["visibility_by_view"]["right"]["occlusion_state"] = (
            "FORGED_OCCLUSION"
        )

        dashboard = self.gui.build_dashboard_model(manifest, report)

        self.assertFalse(dashboard["binding_valid"])
        self.assertEqual(dashboard["counts"]["UNKNOWN"], 2)
        issue_codes = {issue["code"] for issue in dashboard["binding_issues"]}
        self.assertIn("REPORT_EVIDENCE_INVALID", issue_codes)
        self.assertIn("REPORT_OCCLUSION_INVALID", issue_codes)

    def test_schema_two_decisive_state_without_typed_evidence_fails_closed(self) -> None:
        manifest, report = _schema_two_manifest_report()
        for view in report["pipes"][0]["visibility_by_view"].values():
            view["installation_evidence"] = "INCONCLUSIVE"
            view["direct_instance_evidence"] = False
            view["negative_candidate"] = False

        dashboard = self.gui.build_dashboard_model(manifest, report)

        self.assertFalse(dashboard["binding_valid"])
        self.assertEqual(dashboard["counts"]["UNKNOWN"], 2)
        self.assertIn(
            "REPORT_STATE_EVIDENCE_CONFLICT",
            {issue["code"] for issue in dashboard["binding_issues"]},
        )

    def test_schema_two_conflicting_direct_and_negative_flags_fail_closed(self) -> None:
        manifest, report = _schema_two_manifest_report()
        for view in report["pipes"][0]["visibility_by_view"].values():
            view["negative_candidate"] = True

        dashboard = self.gui.build_dashboard_model(manifest, report)

        self.assertFalse(dashboard["binding_valid"])
        self.assertEqual(dashboard["counts"]["UNKNOWN"], 2)
        self.assertIn(
            "REPORT_EVIDENCE_FLAG_CONFLICT",
            {issue["code"] for issue in dashboard["binding_issues"]},
        )

    def test_schema_two_report_schema_version_is_required(self) -> None:
        manifest, report = _schema_two_manifest_report()
        report.pop("schema_version")

        dashboard = self.gui.build_dashboard_model(manifest, report)

        self.assertFalse(dashboard["binding_valid"])
        self.assertEqual(dashboard["counts"]["UNKNOWN"], 2)
        self.assertIn(
            "REPORT_SCHEMA_MISMATCH",
            {issue["code"] for issue in dashboard["binding_issues"]},
        )

    def test_schema_two_manifest_requires_unique_nonempty_cad_bindings(self) -> None:
        for case in ("empty", "duplicate"):
            with self.subTest(case=case):
                manifest, report = _schema_two_manifest_report()
                if case == "empty":
                    manifest["model"]["pipes"][0]["cad_object_id"] = ""
                    expected_code = "MANIFEST_CAD_OBJECT_ID_INVALID"
                else:
                    manifest["model"]["pipes"][1]["cad_object_id"] = (
                        manifest["model"]["pipes"][0]["cad_object_id"]
                    )
                    expected_code = "MANIFEST_CAD_OBJECT_ID_DUPLICATE"

                dashboard = self.gui.build_dashboard_model(manifest, report)

                self.assertFalse(dashboard["binding_valid"])
                self.assertEqual(dashboard["counts"]["UNKNOWN"], 2)
                self.assertIn(
                    expected_code,
                    {issue["code"] for issue in dashboard["binding_issues"]},
                )

    def test_schema_two_photo_hash_change_suppresses_old_decisive_state(self) -> None:
        manifest, report = _schema_two_manifest_report()
        # Add a complete latest capture audit so the provenance check can
        # compare the current bytes with the report's recorded pair.
        manifest["capture"] = {
            "capture_group_id": "capture-run-1",
            "capture_groups": [
                {
                    "capture_id": "pair-1",
                    "views": {
                        "left": {"path": "left.png", "sha256": "a" * 64},
                        "right": {"path": "right.png", "sha256": "c" * 64},
                    },
                }
            ],
        }
        report["capture_audit"] = {
            "groups": [
                {
                    "capture_id": "pair-1",
                    "photos": {
                        "left": {"actual_sha256": "a" * 64},
                        "right": {"actual_sha256": "c" * 64},
                    },
                }
            ]
        }

        dashboard = self.gui.build_dashboard_model(
            manifest,
            report,
            photo_actual_sha256={"left": "b" * 64, "right": "c" * 64},
        )

        self.assertFalse(dashboard["binding_valid"])
        self.assertEqual(dashboard["counts"]["UNKNOWN"], 2)
        self.assertIn(
            "CURRENT_PHOTO_HASH_MISMATCH",
            {issue["code"] for issue in dashboard["binding_issues"]},
        )

    def test_source_load_failure_clears_previous_decisive_dashboard(self) -> None:
        gui = self.gui
        app = object.__new__(gui._PipeTwinApplication)
        app.manifest_path = Path("old-manifest.json").resolve()
        app.report_path = Path("old-report.json").resolve()
        app.manifest = _manifest()
        app.report = _report()
        app.dashboard = {
            "binding_valid": True,
            "binding_issues": [],
            "display_issues": [],
            "model": {"path": "pipes.3dm", "sha256": MODEL_HASH, "format": "3dm"},
            "counts": {"INSTALLED": 1, "NOT_INSTALLED": 1, "UNKNOWN": 0},
            "pipes": [],
        }
        app.selected_pipe_id = "P-RED-22"
        app._manifest_sha256 = "d" * 64
        app._model_actual_sha256 = MODEL_HASH
        app._photo_actual_sha256 = {}
        app._photo_paths = {}
        app._photo_images = {}
        app._photo_geometry = {}
        app.messagebox = Mock()
        app._refresh_dashboard = Mock()
        app._render_photo = Mock()

        with tempfile.TemporaryDirectory() as temporary_directory:
            missing = Path(temporary_directory) / "does-not-exist.json"
            app._load_sources(missing, None)

        self.assertIsNone(app.report)
        self.assertFalse(app.dashboard["binding_valid"])
        self.assertNotEqual(app.dashboard["counts"]["INSTALLED"], 1)
        self.assertIsNone(app.selected_pipe_id)
        app.messagebox.showerror.assert_called_once()

    def test_report_load_failure_replaces_old_report_with_unknown_dashboard(self) -> None:
        gui = self.gui
        app = object.__new__(gui._PipeTwinApplication)
        app.manifest_path = Path("manifest.json").resolve()
        app.report_path = Path("old-report.json").resolve()
        app.manifest = _manifest()
        app.report = _report()
        app.dashboard = {
            "binding_valid": True,
            "binding_issues": [],
            "display_issues": [],
            "model": {"path": "pipes.3dm", "sha256": MODEL_HASH, "format": "3dm"},
            "counts": {"INSTALLED": 1, "NOT_INSTALLED": 1, "UNKNOWN": 0},
            "pipes": [],
        }
        app.selected_pipe_id = "P-RED-22"
        app._manifest_sha256 = None
        app._model_actual_sha256 = None
        app._photo_actual_sha256 = {}
        app._photo_paths = {}
        app._photo_images = {}
        app._photo_geometry = {}
        app.filedialog = Mock()
        app.messagebox = Mock()
        app._refresh_dashboard = Mock()
        app._render_photo = Mock()
        with tempfile.TemporaryDirectory() as temporary_directory:
            bad_report = Path(temporary_directory) / "bad-report.json"
            bad_report.write_text("not-json", encoding="utf-8")
            app.filedialog.askopenfilename.return_value = str(bad_report)
            app._choose_report()

        self.assertIsNone(app.report)
        self.assertFalse(app.dashboard["binding_valid"])
        self.assertEqual(app.dashboard["counts"]["UNKNOWN"], 2)
        self.assertEqual(app.dashboard["counts"]["INSTALLED"], 0)
        self.assertIsNone(app.selected_pipe_id)
        app.messagebox.showerror.assert_called_once()

    def test_manifest_photo_preview_rejects_dataset_escape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "capture.json"
            manifest_path.write_text("{}", encoding="utf-8")
            manifest = {
                "capture": {
                    "capture_groups": [
                        {
                            "views": {
                                "left": {"path": "../outside.png"},
                                "right": {"path": "missing.png"},
                            }
                        }
                    ]
                }
            }

            paths = self.gui._first_manifest_stereo_paths(manifest_path, manifest)

            self.assertEqual(paths, {})

    def test_module_import_does_not_eagerly_import_tkinter(self) -> None:
        # This assertion is meaningful only if another test has not imported Tk.
        if "tkinter" in sys.modules:
            self.skipTest("tkinter was imported by the surrounding test process")
        sys.modules.pop("pipe_twin.gui", None)
        importlib.import_module("pipe_twin.gui")
        self.assertNotIn("tkinter", sys.modules)


if __name__ == "__main__":
    unittest.main()
