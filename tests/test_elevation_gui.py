from __future__ import annotations

import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np

from pipe_twin.dxf_elevation import catalog_from_dxf
from pipe_twin.elevation_gui import ElevationCaptureDialog, ZoneEditorDialog, image_region
from test_elevation_dataset import _calibration


class ElevationRegionTests(unittest.TestCase):
    def test_zoomed_reversed_drag_uses_original_pixels(self):
        self.assertEqual(image_region((210, 130), (30, 50), (2, 10, 10), (640, 480)), [10, 20, 90, 40])
        self.assertEqual(image_region((-100, -100), (50, 50), (1, 0, 0), (640, 480)), [0, 0, 50, 50])
        self.assertIsNone(image_region((5, 5), (8, 8), (1, 0, 0), (640, 480)))


class ElevationGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tkinter as tk
        from tkinter import ttk
        try:
            cls.root = tk.Tk()
        except tk.TclError as error:
            raise unittest.SkipTest(f"Tk display unavailable: {error}") from error
        cls.root.withdraw()
        cls.tk, cls.ttk = tk, ttk

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name) / "workbench"
        self.app = SimpleNamespace(root=self.root, tk=self.tk, ttk=self.ttk, messagebox=mock.Mock(), filedialog=mock.Mock())
        self.profile_patch = mock.patch("pipe_twin.workbench_profile.load_profile", return_value=(None, None))
        self.profile_patch.start()
        self.dialog = ElevationCaptureDialog(self.app, output_root=self.output, restore=False)
        self.dialog.window.withdraw()
        self.dialog.calibration_override = _calibration()
        model = Path(__file__).resolve().parents[1] / "test_model" / "管道布置.dxf"
        self.dialog.load_model(model)
        self.spec = copy.deepcopy(self.dialog.pipes[0])
        for role, value in (("left", 50), ("right", 60)):
            path = Path(self.tmp.name) / f"{role}.png"
            cv2.imwrite(str(path), np.full((480, 640, 3), value, np.uint8))
            self.dialog.fields[role].set(str(path))
            self.dialog.fields[f"{role}_time"].set("2026-09-12T10:00:00+08:00")
        self.dialog.load_images()
        self.dialog.refresh_table()
        self.dialog.confirmed.set(True)

    def tearDown(self):
        self.dialog.close()
        self.root.update()
        self.profile_patch.stop()
        self.tmp.cleanup()

    def finish_worker(self):
        deadline = time.monotonic() + 15
        while self.dialog.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertFalse(self.dialog.busy, "GUI worker did not finish")

    def test_scene_status_is_visible_before_cad_assignment_and_clears_on_edit(self):
        from test_temporal_sections import stage, check
        value = check([stage(), stage(), stage()])
        self.dialog.report = {'section_evaluation': value}
        self.dialog.refresh_table()
        self.root.update()
        self.assertEqual(len(self.dialog.solver_tree.get_children()), 2)
        self.assertIsNone(self.dialog.selected_solver_row()['pipe_id'])
        self.assertIn('3/3', self.dialog.solver_tree.item('0:0', 'values'))
        for view in self.dialog.views.values():
            self.assertEqual(len(view.canvas.find_withtag('solver-selection')), 2)
        self.dialog.solver_tree.selection_set('0:1')
        self.root.update()
        self.assertEqual(self.dialog.selected_solver_row()['label'], 'S2')
        self.dialog.invalidate()
        self.root.update()
        self.assertFalse(self.dialog.solver_tree.get_children())
        self.assertIsNone(self.dialog.selected_solver_row())

    def test_candidate_ground_trial_toggle_is_explicit_and_invalidates_old_report(self):
        from pipe_twin.elevation_dataset import normalize_registration_settings
        self.assertFalse(self.dialog.ground_candidate_trial.get())
        self.dialog.report = {'old': True}
        self.dialog.ground_candidate_trial.set(True)
        self.dialog._ground_trial_changed()
        self.assertIsNone(self.dialog.report)
        settings = normalize_registration_settings(self.dialog.registration_settings)
        self.assertTrue(settings['ground_candidate_trial'])
        self.dialog.ground_candidate_trial.set(False)
        self.dialog._ground_trial_changed()
        self.assertNotIn('ground_candidate_trial', normalize_registration_settings(self.dialog.registration_settings))
        with self.assertRaises(ValueError):
            normalize_registration_settings({'ground_candidate_trial': 'true'})
        self.app.messagebox.showerror.assert_not_called()

    def test_settings_do_not_consume_photo_space_and_block_edits_while_evaluating(self):
        self.dialog.window.deiconify()
        self.dialog.window.geometry('1024x720')
        self.root.update()
        height = self.dialog.views['left'].canvas.winfo_height()
        self.assertGreater(height, 180)
        # A long report/path must remain accessible without taking photo space.
        self.dialog.message.set('long diagnostic path ' * 100)
        self.root.update()
        self.assertEqual(self.dialog.views['left'].canvas.winfo_height(), height)
        self.assertIn(self.dialog.message.get(), self.dialog.status_text.get('1.0', 'end'))
        for page in self.dialog.settings_pages:
            self.dialog.open_settings(page)
            self.root.update()
            self.assertEqual(self.dialog.views['left'].canvas.winfo_height(), height)
        self.dialog._set_controls(False)
        self.assertIn('disabled', self.dialog.local_observation_box.state())
        self.dialog._set_controls(True)
        self.assertNotIn('disabled', self.dialog.local_observation_box.state())
        self.dialog.hide_settings()
        self.assertEqual(self.dialog.settings_window.state(), 'withdrawn')

    def test_unified_details_follow_selection_and_close_when_inputs_change(self):
        from test_temporal_sections import stage
        value = stage()
        for views in value['views'].values():
            for item in views:
                item['boundary_stable'] = True
        self.dialog.report = {'section_evaluation': value}
        self.dialog.refresh_table()
        self.dialog.show_analysis_details()
        window = self.dialog.analysis_details_window
        self.assertEqual(len(self.dialog.detail_tabs.tabs()), 7)
        self.dialog.solver_tree.selection_set('0:1')
        self.root.update()
        self.assertIn('/ S2', self.dialog.detail_texts['candidate'].get('1.0', 'end'))
        self.dialog.show_analysis_details('ground')
        self.assertIs(self.dialog.analysis_details_window, window)
        self.assertEqual(self.dialog.detail_tabs.select(), str(self.dialog.detail_pages['ground']))
        self.dialog.show_analysis_details('geometry')
        self.assertEqual(self.dialog.detail_tabs.select(), str(self.dialog.detail_pages['geometry']))
        self.assertIn('不是实时角度', self.dialog.detail_texts['geometry'].get('1.0', 'end'))
        self.dialog.invalidate()
        self.assertFalse(window.winfo_exists())

    def test_only_cad_tab_exposes_color_mutation_actions(self):
        self.root.update()
        self.assertFalse(self.dialog.cad_actions.winfo_manager())
        self.dialog.status_tabs.select(self.dialog.cad_status_tab)
        self.root.update()
        self.assertEqual(self.dialog.cad_actions.winfo_manager(), 'pack')
        self.dialog.status_tabs.select(0)
        self.root.update()
        self.assertFalse(self.dialog.cad_actions.winfo_manager())

    def test_assistance_without_metric_fit_links_photo_and_one_candidate_plan_only(self):
        from test_assist_display import report_fixture
        report = report_fixture()
        before = copy.deepcopy(report)
        self.dialog.report = report
        self.dialog.refresh_table()
        self.root.update()
        self.assertEqual(len(self.dialog.assist_tree.get_children()), 2)
        self.assertFalse(self.dialog.solver_tree.get_children())
        self.assertEqual(self.dialog.status_tabs.select(), str(self.dialog.assist_status_tab))
        self.assertEqual(self.dialog.selected_id(), 'P001')
        self.assertTrue(self.dialog.views['left'].canvas.find_withtag('assist-candidate'))
        self.dialog.open_model_viewer()
        viewer = self.dialog.model_viewer
        self.root.update_idletasks()
        self.assertEqual(viewer.installation_states['P001'], 'ASSIST_CANDIDATE')
        self.assertEqual(viewer.installation_states['P003'], 'UNKNOWN')
        self.dialog.assist_plan_box.current(1)
        self.dialog._assist_plan_selected()
        self.root.update()
        self.assertEqual(self.dialog.selected_id(), 'P003')
        self.assertEqual(viewer.installation_states['P003'], 'ASSIST_CANDIDATE')
        self.assertEqual(viewer.installation_states['P001'], 'UNKNOWN')
        self.dialog.select_model_pipe('P004')
        self.root.update()
        self.assertEqual(self.dialog.selected_assist_row()['label'], 'A2')
        self.dialog.select_model_pipe('P009')
        self.root.update()
        self.assertIsNone(self.dialog.selected_assist_row())
        self.assertEqual(report, before)
        self.dialog.show_analysis_details('assist')
        self.assertIn('采用已知管径辅助定位', self.dialog.detail_texts['assist'].get('1.0', 'end'))
        self.dialog.invalidate()
        self.assertFalse(self.dialog.assist_tree.get_children())
        self.assertFalse(viewer.section_canvas.find_withtag('state:ASSIST_CANDIDATE'))

    def test_purchase_colour_hints_invalidate_old_suggestions_without_rewriting_model(self):
        from test_assist_display import report_fixture
        self.dialog.report = report_fixture()
        before = copy.deepcopy(self.dialog.pipes)
        self.assertTrue(self.dialog.apply_assist_hints({'BLUE': 50., 'RED': 40., 'WHITE': 25.}))
        self.assertEqual(self.dialog.registration_settings['installation_assist']['appearance_diameter_hints']['BLUE'], 50.)
        self.assertIsNone(self.dialog.report)
        self.assertEqual(self.dialog.pipes, before)
        with self.assertRaises(ValueError):
            self.dialog.apply_assist_hints({'BLUE': -50.})

    def test_empty_pipe_roi_band_intersection_stops_analysis_before_saving(self):
        self.dialog.set_axial_band({'image_size': [640, 480], 'axis_segment_px': [[300, 240], [600, 240]]})
        self.dialog.add_left_region([10, 10, 100, 100])
        self.assertIn('范围冲突', self.dialog.setup_summary.get())
        self.assertIn('Z01', self.dialog.assessment_region_problem())
        with mock.patch.object(self.dialog, '_worker') as worker:
            self.dialog.submit(True)
        worker.assert_not_called()
        self.assertFalse(self.dialog.busy)
        self.assertIsNone(self.dialog.last_manifest)
        self.assertIn('没有交集', self.app.messagebox.showerror.call_args.args[1])

    def test_pipe_scope_reset_preserves_independent_ground_and_pooling(self):
        self.dialog.set_axial_band({'image_size': [640, 480], 'axis_segment_px': [[300, 240], [600, 240]]})
        self.dialog.set_ground_region([10, 10, 100, 100])
        self.dialog.add_left_region([350, 200, 200, 160])
        self.assertIsNone(self.dialog.assessment_region_problem())
        original = copy.deepcopy(self.dialog.registration_settings)
        self.dialog.clear_assessment_regions()
        self.assertEqual(self.dialog.registration_settings, original)
        self.assertEqual(self.dialog.zone_settings['scope'], 'full_frame')
        self.assertEqual(self.dialog.zone_settings['zones'], [])
        self.assertIsNone(self.dialog.assessment_region_problem())

    def test_analysis_button_follows_scope_controls_on_screen(self):
        self.dialog.window.deiconify()
        self.root.update()
        self.assertGreater(self.dialog.run_button.winfo_rooty(), self.dialog.axial_button.winfo_rooty())
        self.assertGreater(self.dialog.run_button.winfo_rooty(), self.dialog.region_button.winfo_rooty())

    def test_redraw_replaces_old_region_preserves_ground_band_and_saved_history(self):
        self.dialog.set_axial_band({'image_size': [640, 480], 'axis_segment_px': [[300, 240], [600, 240]]})
        self.dialog.set_ground_region([10, 10, 100, 100])
        self.dialog.zone_settings['right_roi_padding_px'] = 80
        self.dialog.add_left_region([10, 10, 100, 100])
        self.dialog.submit(False)
        self.finish_worker()
        old_path = self.dialog.last_manifest
        old_bytes = old_path.read_bytes()
        original = copy.deepcopy(self.dialog.registration_settings)
        self.assertEqual(self.dialog.region_conflict_button.winfo_manager(), 'pack')
        self.dialog.region_conflict_button.invoke()
        self.assertTrue(self.dialog.region_selection_active)
        self.dialog.report = {'old': True}
        view = self.dialog.views['left']
        view.transform = (1., 0., 0.)
        view.start_region(SimpleNamespace(x=350, y=200))
        view.finish_region(SimpleNamespace(x=550, y=360))
        self.assertIsNone(self.dialog.report)
        self.assertEqual(self.dialog.registration_settings, original)
        self.assertEqual(len(self.dialog.zone_settings['zones']), 1)
        self.assertEqual(self.dialog.zone_settings['zones'][0]['roi_rect_px'], [350, 200, 200, 160])
        self.assertEqual(self.dialog.zone_settings['right_roi_padding_px'], 80)
        self.assertFalse(self.dialog.region_conflict_button.winfo_manager())
        self.dialog.submit(False)
        self.finish_worker()
        saved = copy.deepcopy(self.dialog.zone_settings)
        self.assertNotEqual(self.dialog.last_manifest, old_path)
        self.dialog.load_session(self.dialog.last_manifest)
        self.assertEqual(self.dialog.zone_settings, saved)
        self.assertIsNone(self.dialog.assessment_region_problem())
        self.assertEqual(old_path.read_bytes(), old_bytes)
        self.app.messagebox.showerror.assert_not_called()

    def test_legacy_regions_restore_without_accumulation_or_modifying_source(self):
        self.dialog.add_left_region([350, 200, 200, 160])
        self.dialog.submit(False)
        self.finish_worker()
        path = self.dialog.last_manifest
        payload = json.loads(path.read_text(encoding='utf-8'))
        scope = payload['analysis']['elevation_auto']
        scope['zones'] = [
            {'zone_id': 'Z01', 'enabled': False, 'roi_rect_px': [10, 10, 100, 100]},
            {'zone_id': 'Z02', 'enabled': False, 'roi_rect_px': [20, 20, 100, 100]},
            {'zone_id': 'Z03', 'enabled': True, 'roi_rect_px': [350, 200, 200, 160]}]
        for active_count in (1, 2, 0):
            with self.subTest(active_count=active_count):
                scope['scope'] = 'zones' if active_count else 'full_frame'
                scope['zones'][0]['enabled'] = active_count == 2
                scope['zones'][2]['enabled'] = active_count > 0
                path.write_text(json.dumps(payload), encoding='utf-8')
                original = path.read_bytes()
                self.dialog.load_session(path)
                self.assertEqual(path.read_bytes(), original)
                self.assertIsNone(self.dialog.report)
                if active_count == 1:
                    self.assertEqual([z['zone_id'] for z in self.dialog.zone_settings['zones']], ['Z03'])
                    self.assertIn('沿用历史', self.dialog.message.get())
                else:
                    self.assertEqual(self.dialog.zone_settings['zones'], [])
                    self.assertEqual(self.dialog.zone_settings['scope'], 'full_frame')
                    self.assertIn('恢复全图', self.dialog.message.get())
        self.app.messagebox.showerror.assert_not_called()

    def test_cancel_or_invalid_replacement_keeps_current_region_and_result(self):
        self.dialog.add_left_region([350, 200, 200, 160])
        self.dialog.report = {'old': True}
        original = copy.deepcopy(self.dialog.zone_settings)
        self.dialog.toggle_region_selection()
        view = self.dialog.views['left']
        view.transform = (1., 0., 0.)
        view.start_region(SimpleNamespace(x=80, y=80))
        view.finish_region(SimpleNamespace(x=90, y=90))
        self.assertTrue(self.dialog.region_selection_active)
        self.dialog.cancel_image_selections()
        self.dialog.add_left_region([600, 400, 200, 200])
        self.assertEqual(self.dialog.zone_settings, original)
        self.assertEqual(self.dialog.report, {'old': True})
        self.assertFalse(self.dialog.region_selection_active)

    def test_unmigrated_multiple_regions_cannot_be_saved_or_evaluated(self):
        self.dialog.add_left_region([20, 20, 80, 80])
        self.dialog.zone_settings['zones'].append({
            'zone_id': 'Z02', 'enabled': True, 'roi_rect_px': [200, 200, 80, 80]})
        for analyze in (False, True):
            with mock.patch.object(self.dialog, '_worker') as worker:
                self.dialog.submit(analyze)
            worker.assert_not_called()
            self.assertIsNone(self.dialog.last_manifest)
            self.assertIn('一个管段', self.app.messagebox.showerror.call_args.args[1])

    def test_save_reopen_and_repeat_save_without_json_editing(self):
        self.dialog.submit(False)
        self.finish_worker()
        saved = self.dialog.last_manifest
        self.assertTrue(saved.is_file())
        payload = json.loads(saved.read_text(encoding="utf-8"))
        self.assertEqual(payload["analysis"]["mode"], "elevation_auto")
        self.assertTrue(payload["model"]["path"].endswith(".dxf"))
        self.assertFalse(payload["stereo_calibration"]["registration_validated"])
        self.dialog.load_session(saved)
        self.assertNotIn("left_region_px", self.dialog.pipes[0])
        self.assertFalse(self.dialog.confirmed.get())
        self.dialog.confirmed.set(True)
        self.dialog.submit(False)
        self.finish_worker()
        repeated = json.loads(self.dialog.last_manifest.read_text(encoding="utf-8"))
        self.assertEqual(len(repeated["capture"]["capture_groups"]), 1)
        self.assertNotIn("已开始新历史", self.dialog.message.get())

    def test_zoomed_direction_drag_handle_resize_save_and_reopen(self):
        from pipe_twin.axial_selection import band_geometry
        view = self.dialog.views['left']
        self.dialog.toggle_axial_selection()
        view.transform = (2, 10, 10)
        view.start_region(SimpleNamespace(x=210, y=490))
        view.move_region(SimpleNamespace(x=1010, y=450))
        view.finish_region(SimpleNamespace(x=1010, y=450))
        band = self.dialog.registration_settings['aggregation_band']
        self.assertEqual(band['axis_segment_px'], [[100., 240.], [500., 220.]])
        self.assertFalse(self.dialog.axial_selection_active)
        self.assertEqual(len(view.canvas.find_withtag('axial-band-limit')), 2)
        self.assertEqual(len(view.canvas.find_withtag('axial-band-handle')), 2)
        self.assertFalse(self.dialog.views['right'].canvas.find_withtag('axial-band'))
        self.dialog.report = {'old': True}
        # Resize one boundary without changing the chosen common image direction.
        view.transform = (2, 10, 10)
        view.start_region(SimpleNamespace(x=1010, y=450))
        view.move_region(SimpleNamespace(x=1190, y=400))
        view.finish_region(SimpleNamespace(x=1190, y=400))
        moved = self.dialog.registration_settings['aggregation_band']
        np.testing.assert_allclose(band_geometry(moved)[1], band_geometry(band)[1])
        self.assertGreater(band_geometry(moved)[2], band_geometry(band)[2])
        self.assertIsNone(self.dialog.report)
        self.dialog.submit(False)
        self.finish_worker()
        saved = self.dialog.last_manifest
        self.dialog.clear_axial_band()
        self.assertNotIn('aggregation_band', self.dialog.registration_settings)
        self.dialog.load_session(saved)
        self.assertEqual(self.dialog.registration_settings['aggregation_band'], moved)
        self.assertEqual(len(view.canvas.find_withtag('axial-band-limit')), 2)
        self.dialog.fields['calibration'].set('different-calibration.json')
        self.assertNotIn('aggregation_band', self.dialog.registration_settings)

    def test_pooling_selection_cancel_exclusive_modes_and_stale_drag(self):
        view = self.dialog.views['left']
        band = {'image_size': [640, 480], 'axis_segment_px': [[100, 240], [500, 240]]}
        self.dialog.set_axial_band(band)
        stored = copy.deepcopy(self.dialog.registration_settings['aggregation_band'])
        self.dialog.toggle_axial_selection()
        view.transform = (1, 0, 0)
        view.start_region(SimpleNamespace(x=100, y=100))
        view.move_region(SimpleNamespace(x=400, y=120))
        self.dialog.cancel_image_selections()
        self.assertFalse(view.band_controller.dragging)
        self.assertEqual(self.dialog.registration_settings['aggregation_band'], stored)
        self.dialog.toggle_axial_selection()
        self.dialog.toggle_region_selection()
        self.assertFalse(self.dialog.axial_selection_active)
        self.assertTrue(self.dialog.region_selection_active)
        self.dialog.cancel_image_selections()
        view.transform = (1, 0, 0)
        view.start_region(SimpleNamespace(x=100, y=240))
        self.assertTrue(view.band_controller.dragging)
        self.dialog.invalidate()
        self.assertFalse(view.band_controller.dragging)
        view.finish_region(SimpleNamespace(x=200, y=240))
        self.assertEqual(self.dialog.registration_settings['aggregation_band'], stored)
        view.set_image(np.zeros((240, 320, 3), np.uint8))
        self.assertNotIn('aggregation_band', self.dialog.registration_settings)

    def test_ground_roi_is_separate_from_pooling_and_persists(self):
        self.dialog.set_axial_band({'image_size': [640, 480], 'axis_segment_px': [[100, 240], [500, 220]]})
        band = copy.deepcopy(self.dialog.registration_settings['aggregation_band'])
        self.dialog.toggle_ground_selection()
        view = self.dialog.views['left']; view.transform = (2, 10, 10)
        view.start_region(SimpleNamespace(x=210, y=610))
        view.finish_region(SimpleNamespace(x=1010, y=910))
        self.assertEqual(self.dialog.registration_settings['ground_region']['rect_px'], [100, 300, 400, 150])
        self.assertEqual(self.dialog.registration_settings['aggregation_band'], band)
        self.assertEqual(self.dialog.zone_settings['scope'], 'full_frame')
        self.assertEqual(len(view.canvas.find_withtag('ground-region')), 2)
        self.dialog.submit(True)
        self.finish_worker()
        self.assertNotEqual(self.dialog.report['ground_check']['status'], 'CONSISTENT_GROUND_PLANE')
        saved = self.dialog.last_manifest
        self.dialog.clear_ground_region()
        self.dialog.load_session(saved)
        self.assertEqual(self.dialog.registration_settings['ground_region']['rect_px'], [100, 300, 400, 150])
        self.dialog.show_ground_check()
        windows = [w for w in self.dialog.window.winfo_children() if isinstance(w, self.tk.Toplevel)]
        self.assertTrue(windows)
        windows[-1].destroy()
        self.dialog.fields['calibration'].set('different-ground-calibration.json')
        self.assertNotIn('ground_region', self.dialog.registration_settings)

    def test_standalone_tilt_draws_angles_without_cad_and_clears_on_input_change(self):
        from test_ground_check import scene, add_ground_images
        group, calibration, region, normal = scene()
        add_ground_images(group, calibration, normal)
        group['depth'].audit = {'status': 'VALID'}
        for role in ('left', 'right'):
            cv2.imwrite(self.dialog.fields[role].get(), group[role])
        self.dialog.load_images()
        self.dialog.pipes = []
        self.dialog.fields['model'].set('')
        self.dialog.set_ground_region(region['rect_px'])
        original = self.dialog.views['left'].image.copy()
        with mock.patch('pipe_twin.ground_inspection._compute_stereo_depth', return_value=group['depth']), \
                mock.patch('pipe_twin.ground_inspection._quality', return_value={'passed': True}):
            self.dialog.inspect_ground()
            self.finish_worker()
        self.assertEqual(self.dialog.ground_inspection['status'], 'CONSISTENT_GROUND_PLANE')
        canvas = self.dialog.views['left'].canvas
        labels = canvas.find_withtag('ground-angle')
        self.assertEqual(len(labels), 1)
        self.assertIn('光轴偏离地面法线', canvas.itemcget(labels[0], 'text'))
        self.assertIn('地面检查通过', canvas.itemcget(labels[0], 'text'))
        self.assertFalse(self.dialog.views['right'].canvas.find_withtag('ground-angle'))
        np.testing.assert_array_equal(original, self.dialog.views['left'].image)
        self.assertTrue(list((self.output/'ground_inspections').glob('*/inspection.json')))
        self.assertIsNone(self.dialog.report)
        self.assertIsNotNone(self.dialog.ground_inspection_window)
        self.dialog.set_ground_region([100, 300, 400, 150])
        self.assertIsNone(self.dialog.ground_inspection)
        self.assertIsNone(self.dialog.ground_inspection_window)
        self.assertFalse(canvas.find_withtag('ground-angle'))

    def test_stale_tilt_worker_cannot_publish_previous_photo_angles(self):
        self.dialog.set_ground_region([100, 300, 400, 150])
        old_generation = self.dialog.generation
        self.dialog.fields['left_time'].set('2026-10-23T10:00:00+08:00')
        self.dialog.messages.put((old_generation, 'ground_ok', (Path('stale.json'), {'status': 'CONSISTENT_GROUND_PLANE'})))
        self.dialog.poll()
        self.assertIsNone(self.dialog.ground_inspection)
        self.assertFalse(self.dialog.views['left'].canvas.find_withtag('ground-angle'))

    def test_startup_clears_all_old_selections_but_explicit_open_restores_them(self):
        self.dialog.set_axial_band({'image_size': [640, 480], 'axis_segment_px': [[300, 240], [600, 240]]})
        self.dialog.set_ground_region([10, 10, 100, 100])
        self.dialog.add_left_region([350, 200, 200, 160])
        self.dialog._set_camera_side([0, -1, 0])
        original_registration = copy.deepcopy(self.dialog.registration_settings)
        original_zones = copy.deepcopy(self.dialog.zone_settings)
        self.dialog.submit(False)
        self.finish_worker()
        source = self.dialog.last_manifest
        original_bytes = source.read_bytes()
        resume = self.output/'last_scene.json'
        resume_bytes = resume.read_bytes()
        tags = ('assessment-zone', 'axial-band', 'ground-region')
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                reopened = ElevationCaptureDialog(self.app, source if explicit else None,
                                                 output_root=self.output, restore=True)
                try:
                    self.root.update()
                    canvas = reopened.views['left'].canvas
                    if not explicit:
                        self.assertEqual(reopened.zone_settings['zones'], [])
                        self.assertEqual(reopened.zone_settings['scope'], 'full_frame')
                        self.assertNotIn('aggregation_band', reopened.registration_settings)
                        self.assertNotIn('ground_region', reopened.registration_settings)
                        self.assertTrue(all(not canvas.find_withtag(tag) for tag in tags))
                        self.assertIn('已清空', reopened.message.get())
                        self.assertIn('上次照片', reopened.message.get())
                        self.assertEqual(reopened.registration_settings['camera_side_world'], [0, -1, 0])
                        self.assertEqual(reopened.current_calibration(), self.dialog.current_calibration())
                        self.assertEqual(reopened.image_hashes, self.dialog.image_hashes)
                        self.assertEqual(
                            [(p['pipe_id'], p['nominal_diameter_mm'], p['color_srgb']) for p in reopened.pipes],
                            [(p['pipe_id'], p['nominal_diameter_mm'], p['color_srgb']) for p in self.dialog.pipes])
                        self.assertIsNone(reopened.report)
                        reopened.load_session(source)
                        self.root.update()
                    self.assertEqual(reopened.registration_settings, original_registration)
                    self.assertEqual(reopened.zone_settings, original_zones)
                    self.assertTrue(all(canvas.find_withtag(tag) for tag in tags))
                    self.assertEqual(source.read_bytes(), original_bytes)
                    self.assertEqual(resume.read_bytes(), resume_bytes)
                finally:
                    reopened.close()

    def test_auto_restore_uses_new_profile_calibration_and_clears_old_photos(self):
        from pipe_twin.workbench_profile import save_camera_calibration_bundle
        self.dialog.submit(False)
        self.finish_worker()
        historical = self.dialog.last_manifest
        old_id = self.dialog.current_calibration()['calibration_id']
        active = _calibration()
        active['calibration_id'] = 'FIELD-NEW-001'
        destination = Path(self.tmp.name)/'new-calibration.json'
        save_camera_calibration_bundle(destination, active)
        profile = {'calibration_current': active, 'calibration_path': str(destination),
                   'rectification_recipe': None}
        with mock.patch('pipe_twin.workbench_profile.load_profile', return_value=(profile, None)):
            reopened = ElevationCaptureDialog(self.app, output_root=self.output, restore=True)
        try:
            self.assertEqual(reopened.current_calibration()['calibration_id'], 'FIELD-NEW-001')
            self.assertGreater(len(reopened.pipes), 0)
            self.assertIsNone(reopened.last_manifest)
            self.assertEqual(reopened.temporal_frames, [])
            for role in ('left', 'right'):
                self.assertEqual(reopened.fields[role].get(), '')
                self.assertIsNone(reopened.views[role].image)
            reopened.load_session(historical)
            self.assertEqual(reopened.current_calibration()['calibration_id'], old_id)
            self.assertIsNotNone(reopened.views['left'].image)
        finally:
            reopened.close()

    def test_image_guided_selection_and_diagnostic_overlay(self):
        self.dialog.local_observation_label.set("图像引导圆柱")
        self.dialog._local_observation_selected()
        self.assertEqual(self.dialog.registration_settings["local_observation_mode"],"image_guided")
        self.assertIn("剔除异常视差",self.dialog.message.get())
        self.dialog.report={"local_surface":{"image_candidates":[
            {"role":"left","candidate_id":"IMG-left-001","region_px":[20,40,100,30],
             "reason_codes":["NO_DOMINANT_DISPARITY_SUPPORT"]}]}}
        self.dialog.redraw()
        self.assertEqual(len(self.dialog.views['left'].canvas.find_withtag('image-candidate')),2)
        self.assertFalse(self.dialog.views['right'].canvas.find_withtag('image-candidate'))
        self.dialog.show_measurement()
        windows=[w for w in self.dialog.window.winfo_children() if isinstance(w,self.tk.Toplevel)]
        self.assertTrue(windows)
        texts=[w for w in windows[-1].winfo_children() if isinstance(w,self.tk.Text)]
        self.assertIn('尚无可靠的管间尺寸对比',texts[0].get('1.0','end'))
        windows[-1].destroy()

    def test_saved_scene_runs_real_analysis_and_renders_unknown_for_blank_images(self):
        self.dialog.submit(True)
        self.finish_worker()
        self.assertEqual(self.dialog.results["P001"]["installation_state"], "UNKNOWN")
        self.assertIn("不确定 12", self.dialog.summary.get())
        self.assertTrue((self.dialog.last_manifest.parent / "report.json").is_file())
        self.assertGreaterEqual(len(self.dialog.report["evidence_files"]), 2)

    def test_left_drag_saves_source_pixels_reopens_and_runs_zone_analysis(self):
        view = self.dialog.views["left"]
        self.assertTrue(view.canvas.bind("<ButtonPress-1>"))
        self.assertFalse(self.dialog.views["right"].canvas.bind("<ButtonPress-1>"))
        self.dialog.report = {"old": True}
        self.dialog.results = {"P001": {"installation_state": "INSTALLED"}}
        self.dialog.zone_settings["right_roi_padding_px"] = 80
        self.dialog.region_button.invoke()
        view.transform = (2.0, 10.0, 10.0)
        view.start_region(SimpleNamespace(x=210, y=130))
        view.move_region(SimpleNamespace(x=30, y=50))
        self.assertTrue(view.canvas.find_withtag("pending-zone"))
        view.finish_region(SimpleNamespace(x=30, y=50))
        self.assertEqual(self.dialog.zone_settings["zones"][0]["roi_rect_px"], [10, 20, 90, 40])
        self.assertEqual(self.dialog.scope_label.get(), "单一区域")
        self.assertFalse(self.dialog.region_selection_active)
        self.assertIsNone(self.dialog.report)
        self.assertFalse(self.dialog.results)
        self.assertTrue(view.canvas.find_withtag("assessment-zone"))
        self.assertTrue(all("left_region_px" not in p for p in self.dialog.pipes))
        self.dialog.submit(False)
        self.finish_worker()
        saved = self.dialog.last_manifest
        payload = json.loads(saved.read_text(encoding="utf-8"))["analysis"]["elevation_auto"]
        self.assertEqual(payload["scope"], "zones")
        self.assertEqual(payload["right_roi_padding_px"], 80)
        self.assertEqual(payload["zones"][0]["roi_rect_px"], [10, 20, 90, 40])
        self.dialog.load_session(saved)
        self.assertTrue(view.canvas.find_withtag("assessment-zone"))
        self.dialog.confirmed.set(True)
        self.dialog.submit(True)
        self.finish_worker()
        self.assertEqual(self.dialog.report["scope"], "zones")
        self.assertEqual(self.dialog.report["counts"]["UNKNOWN"], 12)
        self.assertTrue(self.dialog.report["zone_registration_audit"])

    def test_left_drag_clamps_edges_rejects_small_boxes_and_blocks_while_busy(self):
        view = self.dialog.views["left"]
        self.dialog.toggle_region_selection()
        view.transform = (1., 0., 0.)
        view.start_region(SimpleNamespace(x=-10, y=-10))
        view.finish_region(SimpleNamespace(x=80, y=80))
        self.assertFalse(self.dialog.zone_settings["zones"])
        view.start_region(SimpleNamespace(x=10, y=10))
        view.finish_region(SimpleNamespace(x=25, y=25))
        self.assertTrue(self.dialog.region_selection_active)
        self.assertFalse(self.dialog.zone_settings["zones"])
        view.start_region(SimpleNamespace(x=80, y=80))
        view.finish_region(SimpleNamespace(x=-50, y=-20))
        self.assertEqual(self.dialog.zone_settings["zones"][0]["roi_rect_px"], [0, 0, 80, 80])
        self.dialog.toggle_region_selection()
        view.transform = (1., 0., 0.)
        view.start_region(SimpleNamespace(x=80, y=80))
        self.dialog.busy = True
        try:
            view.finish_region(SimpleNamespace(x=250, y=250))
            self.dialog.add_left_region([100, 100, 80, 80])
            self.assertEqual(len(self.dialog.zone_settings["zones"]), 1)
        finally:
            self.dialog.busy = False

    def test_repeated_regions_replace_without_limit_and_clear_to_full_frame(self):
        for x in range(20, 32):
            self.dialog.add_left_region([x, 20, 80, 80])
            self.assertEqual(len(self.dialog.zone_settings['zones']), 1)
            self.assertEqual(self.dialog.zone_settings['zones'][0]['roi_rect_px'], [x, 20, 80, 80])
        self.dialog.toggle_region_selection()
        self.assertTrue(self.dialog.region_selection_active)
        self.dialog.clear_assessment_regions()
        self.assertFalse(self.dialog.zone_settings['zones'])
        self.assertFalse(self.dialog.region_selection_active)
        self.assertFalse(self.dialog.views['left'].canvas.find_withtag('assessment-zone'))

    def test_new_photo_or_input_change_cancels_pending_drag(self):
        view = self.dialog.views["left"]
        self.dialog.add_left_region([20, 20, 80, 80])
        self.dialog.toggle_region_selection()
        view.transform = (1., 0., 0.)
        view.start_region(SimpleNamespace(x=50, y=50))
        self.dialog.invalidate()
        view.finish_region(SimpleNamespace(x=150, y=150))
        self.assertEqual(len(self.dialog.zone_settings["zones"]), 1)
        self.assertFalse(self.dialog.region_selection_active)
        self.dialog.toggle_region_selection()
        self.dialog.load_images()
        self.assertFalse(self.dialog.region_selection_active)
        self.assertEqual(len(self.dialog.zone_settings["zones"]), 1)
        cv2.imwrite(self.dialog.fields["left"].get(), np.zeros((240, 320, 3), np.uint8))
        self.dialog.load_images()
        self.assertFalse(self.dialog.zone_settings["zones"])
        self.assertEqual(self.dialog.scope_label.get(), "全幅自动")

    def test_zone_editor_cannot_overwrite_new_main_view_selection(self):
        editor = ZoneEditorDialog(self.dialog)
        editor.window.withdraw()
        try:
            self.dialog.add_left_region([20, 20, 80, 80])
            editor.zones = [{"zone_id": "STALE", "roi_rect_px": [50, 50, 80, 80]}]
            editor._save()
            self.assertIn("已改变", editor.status.get())
            self.assertEqual(self.dialog.zone_settings["zones"][0]["zone_id"], "Z01")
        finally:
            editor.window.destroy()

    def test_matching_controls_roundtrip_and_preserve_other_settings(self):
        self.dialog.analysis_settings = {"minimum_depth_mm": 350, "stereo_matching": {"block_size": 7}}
        self.dialog.disparity_count.set("256")
        self.dialog.matching_preset.set("弱光降噪")
        self.dialog.submit(False)
        self.finish_worker()
        saved = self.dialog.last_manifest
        self.dialog.disparity_count.set("64")
        self.dialog.load_session(saved)
        self.assertEqual(self.dialog.disparity_count.get(), "256")
        self.assertEqual(self.dialog.matching_preset.get(), "弱光降噪")
        self.assertEqual(self.dialog.analysis_settings["minimum_depth_mm"], 350)
        self.assertEqual(self.dialog.analysis_settings["stereo_matching"]["block_size"], 7)
        self.dialog.confirmed.set(True)
        self.dialog.submit(False)
        self.finish_worker()
        analysis = json.loads(self.dialog.last_manifest.read_text(encoding="utf-8"))["analysis"]
        self.assertEqual(analysis["stereo_matching"]["num_disparities"], 256)
        self.assertEqual(analysis["minimum_depth_mm"], 350)

    def test_measurement_detail_opens_without_inventing_field_dimensions(self):
        self.dialog.show_measurement()
        self.root.update_idletasks()
        dialogs = [w for w in self.dialog.window.winfo_children() if w.winfo_class() == "Toplevel"]
        self.assertTrue(dialogs)
        widget = next(w for w in dialogs[-1].winfo_children() if w.winfo_class() == "Text")
        self.assertIn("未获得可靠实测", widget.get("1.0", "end"))
        dialogs[-1].destroy()

    def test_external_photo_change_is_rejected_and_old_result_removed(self):
        self.dialog.report = {"old": True}
        self.dialog.results = {"P001": {"installation_state": "INSTALLED"}}
        Path(self.dialog.fields["left"].get()).write_bytes(b"externally changed")
        self.dialog.submit(False)
        self.assertFalse(self.dialog.busy)
        self.assertIsNone(self.dialog.report)
        self.assertFalse(self.dialog.results)
        self.app.messagebox.showerror.assert_called_once()
        self.assertIn("预览后已改变", self.dialog.message.get())

    def test_first_run_stl_catalog_and_model_id_preview(self):
        model = Path(__file__).resolve().parents[1] / "test_model" / "管道布置.stl"
        self.dialog.load_model(model)
        self.assertTrue(self.dialog.pipes)
        self.assertTrue(self.dialog.selected_id())
        self.assertTrue(all(row.get("color_source") == "stl_synthetic_by_diameter" for row in self.dialog.pipes))
        colors_by_diameter = {}
        for row in self.dialog.pipes:
            diameter = round(row["nominal_diameter_mm"], 1)
            colors_by_diameter.setdefault(diameter, row["color_srgb"])
            self.assertEqual(row["color_srgb"], colors_by_diameter[diameter])
        self.assertTrue(all("left_region_px" not in row for row in self.dialog.pipes))
        self.dialog.show_catalog()
        self.root.update_idletasks()

    def test_stl_zone_candidates_can_be_bound_confirmed_and_saved(self):
        model = Path(__file__).resolve().parents[1] / "test_model" / "管道布置.stl"
        self.dialog.load_model(model)
        editor = ZoneEditorDialog(self.dialog)
        editor.window.withdraw()
        try:
            editor._auto_propose()
            self.assertEqual(len(editor.zones), 1)
            self.assertFalse(editor.zones[0]["confirmed"])
            self.assertFalse(editor.zones[0]["enabled"])
            editor.listbox.selection_set(0)
            editor._selected()
            editor.pending_rect = [0, 0, 640, 480]
            editor._bind_pending()
            self.assertEqual(editor.zones[0]["roi_source"], "user")
            editor._confirm()
            self.assertTrue(editor.zones[0]["confirmed"])
            self.assertTrue(editor.zones[0]["enabled"])
            editor._auto_propose()
            self.assertEqual(len(editor.zones), 1)
            self.assertTrue(editor.zones[0]["enabled"])
            editor._save()
            self.assertEqual(self.dialog.zone_settings["scope"], "zones")
            self.assertEqual(self.dialog.zone_settings["zones"][0]["model_pipe_ids"],
                             sorted(spec["pipe_id"] for spec in self.dialog.pipes))
        finally:
            if editor.window.winfo_exists():
                editor.window.destroy()

    def test_auto_mode_is_roi_free_and_direction_is_normalized(self):
        model = Path(__file__).resolve().parents[1] / "test_model" / "管道布置.stl"
        self.dialog.load_model(model)
        # The production workflow is automatic; model geometry owns the pipe
        # catalogue and the analyzer owns photo regions.
        self.dialog.pipes[0]["left_region_px"] = [20, 20, 20, 20]
        self.assertTrue(all("left_region_px" not in row for row in self.dialog.pipes[1:]))
        self.dialog._set_axis_world([0.0, 0.0, 4.0])
        self.assertEqual(self.dialog.registration_settings["axis_world"], [0.0, 0.0, 1.0])
        self.assertEqual(self.dialog.registration_settings["anchors"], {})

    def test_auto_canvas_draws_unmatched_observation_overlay(self):
        self.dialog.report = {
            "registration": {"status": "INSUFFICIENT_OBSERVATIONS"},
            "local_surface": {"observations": [{
                "observation_id": "OBS-0007", "left_region_px": [24, 30, 80, 42],
                "right_region_px": [20, 30, 78, 42], "color_srgb": "#00FF00",
            }]},
        }
        self.dialog.views["left"].set_image(np.zeros((120, 160, 3), dtype=np.uint8))
        items = self.dialog.views["left"].canvas.find_all()
        rectangles = [item for item in items if self.dialog.views["left"].canvas.type(item) == "rectangle"]
        labels = [self.dialog.views["left"].canvas.itemcget(item, "text") for item in items if self.dialog.views["left"].canvas.type(item) == "text"]
        self.assertTrue(rectangles)
        self.assertIn("OBS-0007", labels)

    def test_model_viewer_keeps_axis_in_stl_coordinates(self):
        model = Path(__file__).resolve().parents[1] / "test_model" / "管道布置.stl"
        self.dialog.load_model(model)
        self.dialog.report = {"registration": {"status": "MATCHED", "rotation_model_to_camera": [[1, 0, 0], [0, -1, 0], [0, 0, -1]], "translation_model_to_camera_mm": [0, 0, 1000]}, "local_surface": {"point_cloud": {"points_camera_mm": [[0, 0, 1000]], "colors_srgb": ["#FFFFFF"]}}}
        self.dialog.registration_settings["axis_world"] = [0, 0, 1]
        self.dialog.open_model_viewer()
        viewer = self.dialog.model_viewer
        self.assertTrue(np.allclose(viewer.display_axis, [0, 0, -1]))
        viewer.direction_mode.set("Z")
        viewer.apply_direction()
        self.assertTrue(np.allclose(self.dialog.registration_settings["axis_world"], [0, 0, 1]))

    def test_camera_side_control_persists_and_clears_stale_results(self):
        self.dialog.report = {"old": True}
        self.dialog.camera_side_label.set("从 -Y 侧朝 +Y 看")
        self.dialog._camera_side_selected()
        self.assertIsNone(self.dialog.report)
        self.assertEqual(self.dialog.registration_settings["camera_side_world"], [0, -1, 0])
        self.assertEqual(self.dialog.registration_settings["axis_world"], [0, 0, 1])
        self.dialog.submit(False)
        self.finish_worker()
        saved = self.dialog.last_manifest
        self.dialog._set_camera_side([0, 1, 0])
        self.dialog.load_session(saved)
        self.assertEqual(self.dialog.camera_side_label.get(), "从 -Y 侧朝 +Y 看")
        self.dialog.load_model(Path(self.dialog.fields["model"].get()))
        self.assertNotIn("camera_side_world", self.dialog.registration_settings)
        self.assertEqual(self.dialog.camera_side_label.get(), "自动判断（未指定）")

    def test_unapplied_view_direction_is_not_silently_discarded_by_save(self):
        self.dialog.open_model_viewer()
        viewer = self.dialog.model_viewer
        viewer.view_mode.set("从 -Y 侧朝 +Y 看")
        viewer.preview_side()
        self.dialog.submit(True)
        self.assertFalse(self.dialog.busy)
        self.assertIsNone(self.dialog.last_manifest)
        self.assertIn('方向仍是预览', self.dialog.message.get())
        self.assertTrue(viewer.has_pending_camera_side())
        self.assertNotIn('camera_side_world', self.dialog.registration_settings)
        # Reverting is explicit and does not silently apply the preview.
        viewer.restore_camera_side()
        self.assertFalse(viewer.has_pending_camera_side())
        self.assertNotIn('camera_side_world', self.dialog.registration_settings)
        self.dialog.submit(False)
        self.finish_worker()
        self.assertIsNotNone(self.dialog.last_manifest)

    def test_viewer_separates_preview_from_applied_model_camera_side(self):
        self.dialog.report = {"registration": {"status": "MATCHED",
            "rotation_model_to_camera": [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
            "translation_model_to_camera_mm": [0, 0, 1000]}}
        self.dialog.open_model_viewer()
        viewer = self.dialog.model_viewer
        self.assertIn("未指定", viewer.camera_side_status())
        viewer.view_mode.set("从 -Y 侧朝 +Y 看")
        viewer.preview_side()
        self.assertIn("仅预览", viewer.info.get())
        combined = viewer._rotation() @ viewer.registration_rotation
        self.assertTrue(np.allclose(combined @ [0, -1, 0], [0, 0, 1]))
        self.assertNotIn("camera_side_world", self.dialog.registration_settings)
        viewer.apply_camera_side()
        self.assertEqual(self.dialog.registration_settings["camera_side_world"], [0, -1, 0])
        self.assertIsNone(self.dialog.report)
        caption = viewer.section_canvas.itemcget(viewer.section_canvas.find_withtag("camera_side_status")[0], "text")
        self.assertIn("已应用", caption)
        self.assertNotIn("待应用", caption)
        self.dialog.open_model_viewer()
        stale = self.dialog.model_viewer
        self.dialog._set_camera_side([1, 0, 0])
        self.assertEqual(stale.view_mode.get(), "从 +X 侧朝 -X 看")
        self.assertTrue(stale._is_current())
        # Main-window side changes now update the linked viewer live. Other
        # input changes still invalidate old model/photo geometry.
        self.dialog.invalidate()
        stale.apply_camera_side()
        self.assertIn("输入已改变", stale.info.get())
        self.assertIn("输入已改变", stale.camera_side_status())
        self.assertEqual(self.dialog.registration_settings["camera_side_world"], [1, 0, 0])

    def test_catalog_and_direction_share_selection_without_creating_identity(self):
        self.dialog.show_catalog()
        viewer = self.dialog.model_viewer
        self.root.update_idletasks()
        self.dialog.open_model_viewer()
        self.assertIs(self.dialog.model_viewer, viewer)
        generation = self.dialog.generation
        settings = copy.deepcopy(self.dialog.registration_settings)
        report = self.dialog.report
        pid, x, y, _ = next(hit for hit in viewer.section_hits if hit[0] == "P006")
        viewer.pick_section_pipe(SimpleNamespace(x=x, y=y))
        self.assertEqual(self.dialog.selected_id(), pid)
        for canvas in (viewer.canvas, viewer.section_canvas):
            self.assertTrue(canvas.find_withtag("selected"))
            self.assertTrue(all(f"pipe:{pid}" in canvas.gettags(item) for item in canvas.find_withtag("selected")))
        line = viewer.projected_lines[2]
        midpoint = line.mean(axis=0)
        viewer.pick_model_pipe(SimpleNamespace(x=midpoint[0], y=midpoint[1]))
        self.assertEqual(self.dialog.selected_id(), "P003")
        self.dialog.tree.selection_set("P009")
        self.root.update()
        self.assertTrue(all("pipe:P009" in viewer.section_canvas.gettags(item)
                            for item in viewer.section_canvas.find_withtag("selected")))
        self.assertEqual(self.dialog.generation, generation)
        self.assertEqual(self.dialog.registration_settings, settings)
        self.assertIs(self.dialog.report, report)

    def test_scene_is_visible_without_registration_and_live_report_updates_cad_states(self):
        from test_relative_scene import report_fixture
        from pipe_twin.relative_scene import STATE_COLORS
        self.dialog.open_model_viewer()
        viewer = self.dialog.model_viewer
        report = report_fixture()
        report['pipes'] = [{'pipe_id': 'P001', 'installation_state': 'INSTALLED', 'installation_evidence_qualified': True}]
        self.dialog.report = report
        self.dialog.redraw(); self.root.update_idletasks()
        self.assertEqual(len(viewer.scene_hits), 2)
        self.assertEqual(viewer.scene_tabs.index(viewer.scene_tabs.select()), 0)
        circle = next(i for i in viewer.section_canvas.find_withtag('pipe:P001')
                      if viewer.section_canvas.type(i) == 'oval')
        self.assertEqual(viewer.section_canvas.itemcget(circle, 'fill'), STATE_COLORS['INSTALLED'])
        report['pipes'][0].pop('installation_evidence_qualified')
        self.dialog.redraw(); self.root.update_idletasks()
        self.assertFalse(viewer.section_canvas.find_withtag('state:INSTALLED'))
        # Looking at a conditional section creates no identity/installation.
        old_selection = self.dialog.selected_id()
        _, x, y, _ = viewer.scene_hits[1]
        viewer.pick_scene_pipe(SimpleNamespace(x=x, y=y))
        self.assertEqual(self.dialog.selected_id(), old_selection)
        self.assertIn('尚未唯一对应', viewer.selection_info.get())
        self.dialog.invalidate()
        self.assertEqual(viewer.scene_hits, [])
        self.assertFalse(viewer.section_canvas.find_withtag('state:INSTALLED'))

    def test_completed_assessment_refreshes_viewer_invalidated_by_submit(self):
        from test_relative_scene import report_fixture
        self.dialog.open_model_viewer()
        viewer = self.dialog.model_viewer
        self.dialog.invalidate()
        self.assertFalse(viewer._is_current())
        report = report_fixture()
        report.update(pipes=[], counts={})
        self.dialog.messages.put((self.dialog.generation, 'ok', (self.output/'manifest.json', report, False)))
        self.dialog.poll()
        self.assertTrue(viewer._is_current())
        self.assertEqual(len(viewer.scene_hits), 2)

    def test_camera_arrow_updates_while_cross_section_remains_fixed(self):
        self.dialog.show_catalog()
        viewer = self.dialog.model_viewer
        self.root.update_idletasks()
        positions = {pid: (x, y) for pid, x, y, _ in viewer.section_hits}
        directions = []
        for label in ("从 +X 侧朝 -X 看", "从 -X 侧朝 +X 看"):
            viewer.view_mode.set(label)
            viewer.preview_side()
            self.assertEqual(positions, {pid: (x, y) for pid, x, y, _ in viewer.section_hits})
            line = next(item for item in viewer.section_canvas.find_withtag("camera_direction")
                        if viewer.section_canvas.type(item) == "line")
            x1, _, x2, _ = viewer.section_canvas.coords(line)
            directions.append(x2-x1)
        self.assertLess(directions[0], 0)
        self.assertGreater(directions[1], 0)
        self.assertNotIn("camera_side_world", self.dialog.registration_settings)
        viewer.view_mode.set("从 +Z 侧朝 -Z 看")
        viewer.preview_side()
        captions = [viewer.section_canvas.itemcget(i, "text") for i in viewer.section_canvas.find_withtag("camera_direction")]
        self.assertTrue(any("与截面同侧" in text for text in captions))

    def test_axis_reverse_applies_to_both_views_and_invalid_axis_stays_unsaved(self):
        self.dialog.show_catalog()
        viewer = self.dialog.model_viewer
        viewer.reverse_axis()
        viewer.apply_direction()
        self.assertEqual(self.dialog.registration_settings["axis_world"], [0, 0, -1])
        self.assertTrue(np.allclose(viewer.section_basis[2], [0, 0, -1]))
        self.assertTrue(np.allclose(viewer.display_axis, [0, 0, -1]))
        self.assertTrue(viewer._is_current())
        viewer.custom.set("invalid")
        viewer.apply_direction()
        self.assertEqual(self.dialog.registration_settings["axis_world"], [0, 0, -1])
        self.assertIn("格式", viewer.info.get())

    def test_restore_keeps_verified_registration_settings_after_photo_load(self):
        manifest = {"model": {"source_unit": "millimeter", "path": None}, "capture": {"capture_groups": [{"views": {"left": {"timestamp_source": "MANIFEST_OPERATOR_CONFIRMED"}, "right": {"timestamp_source": "MANIFEST_OPERATOR_CONFIRMED"}}}]}}
        loaded = {"manifest": manifest, "mode": "elevation_auto", "registration_settings": {"axis_world": [0, 0, 1], "anchors": {}}, "calibration": _calibration(), "pipe_specs": [copy.deepcopy(self.spec)], "left_path": Path(self.tmp.name) / "left.png", "right_path": Path(self.tmp.name) / "right.png", "left_time": "2026-09-12T10:00:00+08:00", "right_time": "2026-09-12T10:00:00+08:00", "rectification_recipe": None}
        with mock.patch("pipe_twin.elevation_dataset.load_elevation_dataset", return_value=loaded), mock.patch.object(self.dialog, "load_images"):
            self.dialog.load_session(Path(self.tmp.name) / "manifest.json")
        self.assertEqual(self.dialog.registration_settings["anchors"], {})

    def test_calibration_change_requires_new_images_and_regions(self):
        self.dialog.fields["calibration"].set("new-calibration.json")
        self.assertIsNone(self.dialog.calibration_override)
        self.assertFalse(self.dialog.confirmed.get())
        self.assertFalse(self.dialog.image_hashes)
        self.assertNotIn("left_region_px", self.dialog.pipes[0])
        self.assertIsNone(self.dialog.views["left"].image)

    def test_missing_eye_provides_actionable_error(self):
        self.dialog.views["right"].set_image(None)
        self.dialog.submit(False)
        self.app.messagebox.showerror.assert_called_once()
        self.assertIn("完整的左右照片", self.dialog.message.get())

    def test_camera_preferences_keep_imported_rectification_recipe(self):
        recipe = {"calibration_id": "imported-bundle"}
        self.dialog.profile = {"rectification_recipe": recipe}
        with mock.patch("pipe_twin.workbench_profile.update_profile") as persist:
            self.dialog._persist_profile(("camera",), camera={"left_index": 2})
        persist.assert_called_once_with({"camera": {"left_index": 2}})
        self.assertEqual(self.dialog.profile["rectification_recipe"], recipe)

    def test_quality_check_operates_without_observations_and_preserves_analysis_regions(self):
        original = copy.deepcopy(self.dialog.pipes)
        self.dialog.open_quality_check()
        quality = self.dialog.quality_dialog
        quality.window.withdraw()
        quality.set_region("left", [20, 20, 40, 40])
        self.assertEqual(quality.pipes[0]["metrics"]["summary"]["status"], "INCOMPLETE")
        quality.set_region("right", [18, 20, 40, 40])
        self.assertEqual(quality.pipes[0]["metrics"]["summary"]["status"], "COMPLETE")
        self.assertEqual(self.dialog.pipes, original)
        self.assertFalse(self.dialog.results)
        saved = quality.record_path()
        self.assertTrue(saved.is_file())
        record = json.loads(saved.read_text(encoding="utf-8"))
        self.assertEqual(record["photo_sha256"], self.dialog.image_hashes)
        quality.close()
        self.dialog.open_quality_check()
        restored = self.dialog.quality_dialog
        restored.window.withdraw()
        self.assertEqual(restored.pipes[0]["left_region_px"], [20, 20, 40, 40])
        self.assertEqual(restored.pipes[0]["metrics"]["summary"]["status"], "COMPLETE")

    def test_quality_check_clears_changed_photos_and_retains_old_record(self):
        self.dialog.open_quality_check()
        quality = self.dialog.quality_dialog
        quality.window.withdraw()
        for role in ("left", "right"):
            quality.set_region(role, [20, 20, 40, 40])
        saved = quality.record_path()
        original_record = saved.read_bytes()
        cv2.imwrite(self.dialog.fields["left"].get(), np.full((480, 640, 3), 180, np.uint8))
        self.dialog.load_images()
        self.assertNotEqual(quality.record_path(), saved)
        self.assertNotIn("left_region_px", quality.pipes[0])
        self.assertIsNone(quality.pipes[0]["metrics"]["summary"]["color_coverage_percent"])
        self.assertEqual(saved.read_bytes(), original_record)

    def test_quality_mouse_selection_and_matching_policy_changes(self):
        self.dialog.open_quality_check()
        quality = self.dialog.quality_dialog
        quality.window.withdraw()
        quality.set_color("#505050")
        for role in ("left", "right"):
            view = quality.views[role]
            self.assertTrue(view.canvas.bind("<ButtonPress-1>"))
            view.transform = (1.0, 0.0, 0.0)
            view.start(SimpleNamespace(x=20, y=20))
            view.move(SimpleNamespace(x=60, y=60))
            view.finish(SimpleNamespace(x=60, y=60))
            self.assertEqual(quality.pipes[0][f"{role}_region_px"], [20, 20, 40, 40])
        self.dialog.color_delta_lab.set("1")
        self.assertEqual(quality.pipes[0]["metrics"]["summary"]["color_coverage_percent"], 0)
        self.dialog.color_delta_lab.set("100")
        self.assertEqual(quality.pipes[0]["metrics"]["summary"]["color_coverage_percent"], 100)
        self.assertIn("筛选关闭", quality.policy_status.get())
        quality.save()
        record = json.loads(quality.record_path().read_text(encoding="utf-8"))
        self.assertEqual(record["matching_context"]["color_delta_lab_used"], 100)
        self.dialog.color_delta_lab.set("")
        self.assertIsNone(quality.pipes[0]["metrics"]["summary"]["color_coverage_percent"])
        self.assertIn("请修正", quality.policy_status.get())
        self.dialog.color_delta_lab.set("45")
        self.dialog.local_observation_label.set("仅双目深度几何")
        self.assertIn("仅使用深度几何", quality.policy_status.get())

    def test_quality_color_application_invalidates_old_result_only_on_explicit_apply(self):
        originals = copy.deepcopy(self.dialog.pipes)
        self.dialog.open_quality_check()
        quality = self.dialog.quality_dialog
        quality.window.withdraw()
        self.dialog.report = {"old": True}
        self.dialog.results = {"P001": {"installation_state": "INSTALLED"}}
        quality.set_color("#ABCD12")
        self.assertEqual(self.dialog.pipes[0]["color_srgb"], originals[0]["color_srgb"])
        self.assertTrue(self.dialog.results)
        quality.apply_color()
        for row, old in zip(self.dialog.pipes, originals):
            self.assertEqual(row["color_srgb"], "#ABCD12" if abs(row["nominal_diameter_mm"] - self.spec["nominal_diameter_mm"]) <= 0.2 else old["color_srgb"])
        self.assertFalse(self.dialog.color_filter_enabled.get())
        self.assertFalse(self.dialog.results)
        self.assertIsNone(self.dialog.report)

    def test_live_dialogs_use_basic_owner_and_close_with_scene(self):
        self.dialog.open_calibration()
        wizard = self.dialog.calibration_dialog
        self.assertIn("基础立面模式标定后可直接抓拍", wizard.message.get())
        self.dialog.capture_camera()
        camera = self.dialog.camera_dialog
        self.assertIs(camera.owner, self.dialog)
        self.assertIsNone(camera.session)
        self.dialog.close()
        self.assertFalse(wizard.window.winfo_exists())
        self.assertFalse(camera.window.winfo_exists())
        # tearDown owns the final root destruction.

    def test_continuous_capture_save_reopen_and_calibration_change(self):
        from pipe_twin.stereo_camera import CapturedStereoPair
        self.dialog.capture_camera()
        camera = self.dialog.camera_dialog
        camera.window.withdraw()
        self.assertEqual(camera.burst_count.get(), "10")
        camera.burst_count.set("20")
        frames = []
        for i in range(20):
            left = np.full((480, 640, 3), 80+i, np.uint8)
            right = np.full_like(left, 100+i)
            stamp = f"2026-10-12T10:00:{i:02d}+08:00"
            frames.append(CapturedStereoPair(left, right, stamp, stamp, 0., "HOST_SYSTEM_CLOCK", {}))
        session = mock.Mock(exposure_ms=None)
        session.read_pair.side_effect = frames
        camera.session = session
        camera._show_pair = mock.Mock()
        with mock.patch("pipe_twin.measurement_gui.OUTPUT_ROOT", Path(self.tmp.name)), mock.patch.object(self.dialog, "_persist_profile"):
            camera.capture_burst()
            for _ in range(20):
                camera.window.after_cancel(camera._burst_after_id)
                camera._burst_tick()
        self.assertEqual(len(self.dialog.temporal_frames), 20)
        self.assertTrue(self.dialog.confirmed.get())
        self.assertTrue(session.close.called)
        self.dialog.submit(False)
        self.finish_worker()
        self.app.messagebox.showerror.assert_not_called()
        path = self.dialog.last_manifest
        self.dialog.load_session(path)
        self.assertEqual(len(self.dialog.temporal_frames), 20)
        self.assertEqual(len(json.loads(path.read_text(encoding="utf-8"))["capture"]["capture_groups"]), 1)
        self.dialog.fields["calibration"].set("new-calibration.json")
        self.assertEqual(self.dialog.temporal_frames, [])
        self.assertIsNone(self.dialog.last_manifest)

    def test_stopping_burst_never_adopts_partial_capture(self):
        self.dialog.capture_camera()
        camera = self.dialog.camera_dialog
        camera.session = mock.Mock()
        old_paths = {role: self.dialog.fields[role].get() for role in ("left", "right")}
        with mock.patch("pipe_twin.measurement_gui.OUTPUT_ROOT", Path(self.tmp.name)):
            camera.capture_burst()
        camera.stop()
        self.assertFalse(camera._burst_active)
        self.assertIsNone(camera._burst_after_id)
        self.assertEqual(self.dialog.temporal_frames, [])
        self.assertEqual(old_paths, {role: self.dialog.fields[role].get() for role in old_paths})

    def test_print_dimension_is_not_silently_used_as_measured_dimension(self):
        self.dialog.open_calibration()
        wizard = self.dialog.calibration_dialog
        wizard.window.withdraw()
        wizard.square_mm.set("19")
        wizard.measured_square_mm.set("17")
        wizard.columns.set("9")
        wizard.rows.set("7")
        self.assertEqual(wizard._pattern(), (8, 6))
        self.assertIn("本次求解格长：17", wizard.fit_hint.get())
        self.app.filedialog.asksaveasfilename.return_value = str(Path(self.tmp.name)/"board.png")
        wizard.export_target()
        self.assertEqual(wizard.measured_square_mm.get(), "")

    def test_camera_rejects_stale_rectification_after_calibration_change(self):
        self.dialog.capture_camera()
        camera = self.dialog.camera_dialog
        self.dialog.calibration_override = dict(_calibration(), calibration_id="FIELD-CHANGED")
        with self.assertRaisesRegex(ValueError, "标定发生改变"):
            camera._check_calibration_binding()

    def test_wizard_and_measurement_release_each_others_camera(self):
        self.dialog.open_calibration()
        wizard_session = mock.Mock()
        self.dialog.calibration_dialog.session = wizard_session
        self.dialog.capture_camera()
        wizard_session.close.assert_called_once()
        capture_session = mock.Mock()
        self.dialog.camera_dialog.session = capture_session
        self.dialog.open_calibration()
        capture_session.close.assert_called_once()
        self.dialog = ElevationCaptureDialog(self.app, output_root=self.output, restore=False)
        self.dialog.window.withdraw()


if __name__ == "__main__":
    unittest.main()
