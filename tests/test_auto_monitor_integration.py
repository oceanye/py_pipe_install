"""Main-window ownership and immutable setup for automatic monitoring."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import test_elevation_gui as fixtures


class MonitorIntegrationTests(unittest.TestCase):
    setUpClass = classmethod(fixtures.ElevationGuiTests.setUpClass.__func__)
    tearDownClass = classmethod(fixtures.ElevationGuiTests.tearDownClass.__func__)
    setUp = fixtures.ElevationGuiTests.setUp
    tearDown = fixtures.ElevationGuiTests.tearDown

    def test_snapshot_uses_setup_but_never_old_photos_or_history(self):
        self.dialog.profile = {'camera': {'layout': 'side_by_side_lr', 'left_index': 2,
                                         'exposure_ms': None}}
        self.dialog.registration_settings['camera_side_world'] = [0., 1., 0.]
        self.dialog.temporal_frames = [{'old': 'burst'}]
        self.dialog.history.set(True)
        self.dialog.confirmed.set(False)  # A new capture supplies its own pairing.
        config = self.dialog.monitor_configuration()
        args = config['dataset_arguments']
        self.assertIsNone(args['previous_manifest'])
        self.assertEqual(args['temporal_frames'], [])
        self.assertNotIn('left_path', args)
        self.assertNotIn('right_path', args)
        self.assertEqual(args['registration_settings']['camera_side_world'], [0., 1., 0.])
        self.assertIsNone(config['camera_options']['exposure_ms'])
        self.assertEqual(config['camera_options']['left_index'], 2)
        self.assertEqual(config['mode'], 'local')
        self.dialog.pipes[0]['nominal_diameter_mm'] += 5
        self.dialog.registration_settings['camera_side_world'][0] = 1
        self.assertNotEqual(args['pipe_specs'][0], self.dialog.pipes[0])
        self.assertEqual(args['registration_settings']['camera_side_world'], [0., 1., 0.])

    def test_frontend_snapshot_routes_to_saved_server(self):
        from pipe_twin.analysis_runtime_settings import AnalysisRuntimeSettings
        self.dialog.runtime.settings = AnalysisRuntimeSettings(mode='client', address='100.64.20.1', port=8871)
        config = self.dialog.monitor_configuration()
        self.assertEqual(config['mode'], 'client')
        self.assertEqual(config['server_url'], 'http://100.64.20.1:8871')
        self.assertFalse(self.dialog.busy)

    def test_running_monitor_locks_camera_analysis_and_mode_until_release(self):
        self.assertTrue(self.dialog.monitor_started())
        self.assertTrue(self.dialog.busy)
        with mock.patch('pipe_twin.capture_gui.StereoCameraDialog') as camera, \
             mock.patch.object(self.dialog, '_worker') as analyzer:
            self.dialog.capture_camera()
            self.dialog.submit(True)
        camera.assert_not_called()
        analyzer.assert_not_called()
        self.dialog.runtime.mode_label.set('服务器端')
        self.assertFalse(self.dialog.runtime.apply())
        self.dialog.monitor_stopped()
        self.assertFalse(self.dialog.busy)
        self.assertNotIn('disabled', self.dialog.run_button.state())
        self.dialog.busy = True  # An unrelated subsequent task must stay locked.
        self.dialog.monitor_stopped()
        self.assertTrue(self.dialog.busy)
        self.dialog.busy = False

    def test_existing_camera_or_unapplied_direction_blocks_monitor_start(self):
        camera = SimpleNamespace(window=mock.Mock(), _opening=False)
        camera.window.winfo_exists.return_value = True
        self.dialog.camera_dialog = camera
        try:
            with self.assertRaisesRegex(ValueError, '释放相机'):
                self.dialog.monitor_configuration()
        finally:
            self.dialog.camera_dialog = None
        viewer = mock.Mock()
        viewer.window.winfo_exists.return_value = True
        viewer.has_pending_camera_side.return_value = True
        self.dialog.model_viewer = viewer
        try:
            with self.assertRaisesRegex(ValueError, '观察方向尚未应用'):
                self.dialog.monitor_configuration()
        finally:
            self.dialog.model_viewer = None

    def test_pending_camera_open_and_ground_release_still_block_after_window_close(self):
        camera = SimpleNamespace(window=mock.Mock(), _opening=True)
        camera.window.winfo_exists.return_value = False
        self.dialog.camera_dialog = camera
        try:
            with self.assertRaisesRegex(ValueError, '释放相机'):
                self.dialog.monitor_started()
        finally:
            self.dialog.camera_dialog = None
        ground = SimpleNamespace(window=mock.Mock(), worker=SimpleNamespace(active=True))
        ground.window.winfo_exists.return_value = False
        self.dialog.ground_monitor = ground
        try:
            with self.assertRaisesRegex(ValueError, '相机正在释放'):
                self.dialog.monitor_started()
        finally:
            self.dialog.ground_monitor = None

    def test_server_mode_cannot_own_monitor_camera(self):
        from pipe_twin.analysis_runtime_settings import AnalysisRuntimeSettings
        self.dialog.runtime.settings = AnalysisRuntimeSettings(mode='server')
        with self.assertRaisesRegex(ValueError, '前端或本地运算'):
            self.dialog.monitor_configuration()
        self.assertFalse(self.dialog.busy)

    def test_closing_main_window_stops_monitor_without_altering_report(self):
        report = {'historical': 'result'}
        self.dialog.report = report
        monitor = mock.Mock()
        monitor.window.winfo_exists.return_value = True
        self.dialog.auto_monitor = monitor
        self.dialog.close()
        monitor.close.assert_called_once()
        self.assertIs(self.dialog.report, report)


if __name__ == '__main__':
    unittest.main()
