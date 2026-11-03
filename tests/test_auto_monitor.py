from __future__ import annotations

import copy
import hashlib
import json
import queue
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import numpy as np

from pipe_twin.auto_monitor import SingleFrameMonitor
from pipe_twin.dxf_elevation import catalog_from_dxf
from pipe_twin.stereo_camera import CapturedStereoPair, LAYOUT_SIDE_BY_SIDE_LR
from tests.test_elevation_dataset import MODEL_PATH, _calibration


class FakeClock:
    def __init__(self):
        self.value = 100.0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.value

    def advance(self, delta):
        with self.lock:
            self.value += delta
            return self.value


class FakeSession:
    def __init__(self, clock, **options):
        self.clock = clock
        self.options = options
        self.reads = 0
        self.opened = False
        self.closed = threading.Event()
        self.problem = None
        self.block_read = None
        self.fixed_time = None
        self.clock_step = .05

    def open(self):
        self.opened = True

    def read_pair(self):
        if self.block_read is not None:
            self.block_read.wait()
        time.sleep(.004)
        now = self.clock.advance(self.clock_step)
        self.reads += 1
        if self.problem == 'disconnect':
            raise OSError('camera unplugged')
        shape = (self.options['eye_height'], self.options['eye_width'], 3)
        left = np.full(shape, 90, np.uint8)
        right = np.full(shape, 140, np.uint8)
        left[0, 0] = (self.reads % 256, self.reads // 256 % 256, 25)
        right[0, 0] = (self.reads % 256, 20, 99)
        if self.problem == 'black':
            left.fill(0); right.fill(0)
        elif self.problem == 'identical':
            right = left.copy()
        elif self.problem == 'size':
            left = left[:-1]
        captured = self.fixed_time or (datetime(2026, 11, 3, tzinfo=timezone.utc) + timedelta(seconds=now)).isoformat()
        return CapturedStereoPair(left, right, captured, captured,
            1000 if self.problem == 'sync' else 0, 'HOST_SYSTEM_CLOCK',
            {'left': {'capture_backend': 'OpenCV', 'capture_layout': 'SINGLE_FRAME_SIDE_BY_SIDE',
                      'capture_device_index': 0, 'capture_sync_method': 'SAME_UVC_FRAME'},
             'right': {'capture_backend': 'OpenCV', 'capture_layout': 'SINGLE_FRAME_SIDE_BY_SIDE',
                       'capture_device_index': 0, 'capture_sync_method': 'SAME_UVC_FRAME'}})

    def close(self):
        self.closed.set()


def fake_analyzer(manifest_path, report_output_path, evidence_dir, *, progress, cancel_event=None):
    progress('fake compute')
    report = {'inputs': {'manifest': hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()},
              'counts': {'UNKNOWN': 12}, 'evidence_files': []}
    Path(report_output_path).write_text(json.dumps(report), encoding='utf-8')
    return report


class AutoMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.specs, _ = catalog_from_dxf(MODEL_PATH, axis_world=(0, 0, 1), unitless_unit='millimeter')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.clock = FakeClock()
        self.sessions = []
        self.arguments = {'output_root': self.root / 'scenes', 'calibration': _calibration(),
                          'pipe_specs': copy.deepcopy(self.specs), 'model_path': MODEL_PATH,
                          'stl_unit': 'millimeter', 'registration_settings': {
                              'axis_world': [0, 0, 1], 'anchors': {'OLD-OBS-01': self.specs[0]['pipe_id']}},
                          'previous_manifest': self.root / 'must-not-read.json',
                          'temporal_frames': [{'must': 'not carry over'}]}
        self.camera = {'layout': LAYOUT_SIDE_BY_SIDE_LR, 'left_index': 0, 'right_index': None,
                       'eye_width': 640, 'eye_height': 480, 'exposure_ms': None}
        self.events = []

    def factory(self, **options):
        session = FakeSession(self.clock, **options)
        self.sessions.append(session)
        return session

    def monitor(self, **options):
        params = dict(interval_seconds=.8, warmup_seconds=.2, session_factory=self.factory,
                      analyzer=fake_analyzer, clock=self.clock)
        params.update(options)
        monitor = SingleFrameMonitor(self.arguments, self.camera, **params)
        self.addCleanup(lambda: self.finish(monitor))
        return monitor

    def finish(self, monitor):
        monitor.stop()
        deadline = time.monotonic() + 5
        while monitor.active and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertFalse(monitor.active, 'worker or camera did not stop')

    def event(self, monitor, kind, predicate=lambda event: True):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                event = monitor.events.get(timeout=.2)
            except queue.Empty:
                continue
            self.events.append(event)
            if event['kind'] == kind and predicate(event):
                return event
        self.fail(f'timeout for {kind}: {self.events[-8:]}')

    def test_two_cycles_save_new_single_pairs_without_history_or_burst(self):
        monitor = self.monitor().start()
        # Configuration mutations after construction cannot change this run.
        self.arguments['registration_settings']['axis_world'] = [1, 0, 0]
        self.camera['left_index'] = 17
        first = self.event(monitor, 'result')
        second = self.event(monitor, 'result')
        self.finish(monitor)
        self.assertEqual((first['cycle_id'], second['cycle_id']), (1, 2))
        self.assertNotEqual(first['manifest_path'], second['manifest_path'])
        self.assertNotEqual(first['captured_at'], second['captured_at'])
        hashes = []
        for result in (first, second):
            manifest = json.loads(Path(result['manifest_path']).read_bytes())
            groups = manifest['capture']['capture_groups']
            self.assertEqual(len(groups), 1)
            self.assertFalse(groups[0].get('temporal_frames'))
            self.assertEqual(manifest['analysis']['elevation_auto']['registration']['axis_world'], [0, 0, 1])
            self.assertEqual(manifest['analysis']['elevation_auto']['registration']['anchors'], {})
            self.assertNotIn('anchor_pair_sha256', manifest['analysis']['elevation_auto'])
            view = groups[0]['views']['left']
            data = (Path(result['manifest_path']).parent / view['path']).read_bytes()
            self.assertEqual(view['sha256'], hashlib.sha256(data).hexdigest())
            hashes.append(view['sha256'])
            self.assertTrue(Path(result['report_path']).exists())
            state_index = next(i for i,e in enumerate(self.events) if e['cycle_id']==result['cycle_id'] and e.get('state')=='CAPTURING')
            capture_index = next(i for i,e in enumerate(self.events) if e['cycle_id']==result['cycle_id'] and e['kind']=='capture')
            self.assertLess(state_index, capture_index)
        self.assertNotEqual(*hashes)
        self.assertEqual(self.sessions[0].options['left_index'], 0)
        self.assertTrue(self.sessions[0].closed.is_set())
        self.assertIsNone(monitor.next_due)

    def test_slow_analysis_keeps_reader_live_and_never_overlaps_or_catches_up(self):
        calls = []
        def analyze(*args, **kwargs):
            self.assertFalse(calls and calls[-1]['end'] is None)
            before = self.sessions[0].reads
            item = {'start': self.clock(), 'end': None}
            calls.append(item)
            time.sleep(.08)
            self.assertGreater(self.sessions[0].reads, before + 2)
            result = fake_analyzer(*args, **kwargs)
            item['end'] = self.clock()
            return result
        monitor = self.monitor(analyzer=analyze, interval_seconds=.1).start()
        first = self.event(monitor, 'result'); second = self.event(monitor, 'result')
        self.finish(monitor)
        self.assertGreaterEqual(second['capture_started'], calls[0]['end'])
        self.assertGreaterEqual(second['capture_started'], first['next_due'])
        self.assertGreaterEqual(first['next_due'], calls[0]['end'] + monitor.interval_seconds)
        self.assertGreater(second['capture_started'] - first['capture_started'], monitor.interval_seconds)
        self.assertEqual(second['cycle_id'], 2)

    def test_overdue_result_waits_a_full_interval_before_next_capture(self):
        def analyze(*args, **kwargs):
            # Simulate computation longer than the interval, then hold the
            # clock still while the device continues returning new reads.
            self.clock.advance(30)
            self.sessions[0].clock_step = 0
            return fake_analyzer(*args, **kwargs)
        monitor = self.monitor(analyzer=analyze, interval_seconds=10).start()
        first = self.event(monitor, 'result')
        self.assertAlmostEqual(first['next_due'] - first['at'], 10)
        self.assertEqual(monitor.next_due, first['next_due'])
        self.clock.advance(9)
        time.sleep(.04)
        pending = []
        while not monitor.events.empty():
            pending.append(monitor.events.get_nowait())
        self.assertFalse(any(e.get('state')=='CAPTURING' for e in pending))
        self.assertEqual(monitor._cycle_id, 1)
        self.clock.advance(1.1)
        second_capture = self.event(monitor, 'state', lambda e: e['state']=='CAPTURING')
        self.assertEqual(second_capture['cycle_id'], 2)
        self.assertGreaterEqual(second_capture['at'], first['next_due'])
        self.finish(monitor)

    def test_stop_releases_camera_during_analysis_and_blocks_restart_until_done(self):
        entered, release = threading.Event(), threading.Event()
        def analyze(*args, **kwargs):
            entered.set()
            release.wait(5)
            return fake_analyzer(*args, **kwargs)
        monitor = self.monitor(analyzer=analyze).start()
        self.assertTrue(entered.wait(5))
        monitor.stop()
        self.assertTrue(self.sessions[0].closed.wait(2))
        self.assertTrue(monitor.active)
        with self.assertRaises(RuntimeError):
            monitor.start()
        release.set()
        self.finish(monitor)
        remainder = []
        while not monitor.events.empty():
            remainder.append(monitor.events.get_nowait())
        self.assertFalse(any(e['kind']=='result' for e in remainder))
        self.assertEqual(remainder[-1]['kind'], 'stopped')

    def test_failure_does_not_report_success_and_next_cycle_recovers(self):
        count = 0
        def analyze(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 1:
                raise ValueError('deliberate analysis failure')
            return fake_analyzer(*args, **kwargs)
        monitor = self.monitor(analyzer=analyze).start()
        error = self.event(monitor, 'error')
        result = self.event(monitor, 'result')
        self.finish(monitor)
        self.assertEqual(error['cycle_id'], 1)
        self.assertEqual(error['stage'], 'analysis')
        self.assertEqual(result['cycle_id'], 2)
        self.assertFalse(any(e['kind']=='result' and e['cycle_id']==1 for e in self.events))

    def test_black_identical_wrong_size_and_unsynchronized_pairs_are_not_saved(self):
        for problem in ('black', 'identical', 'size', 'sync'):
            with self.subTest(problem=problem):
                def factory(**options):
                    session = self.factory(**options)
                    session.problem = problem
                    return session
                monitor = self.monitor(session_factory=factory).start()
                error = self.event(monitor, 'error')
                self.finish(monitor)
                self.assertEqual(error['stage'], 'capture')
                self.assertFalse(list((self.root/'scenes').glob('*/manifest.json')))

    def test_same_timestamp_cannot_reuse_cached_pair_on_next_cycle(self):
        def factory(**options):
            session = self.factory(**options)
            session.fixed_time = '2026-11-03T12:00:00+08:00'
            return session
        monitor = self.monitor(session_factory=factory).start()
        self.event(monitor, 'result')
        error = self.event(monitor, 'error')
        self.finish(monitor)
        self.assertEqual(error['cycle_id'], 2)
        self.assertIn('旧时间戳', error['message'])

    def test_warmup_clock_and_capture_request_both_require_new_read(self):
        monitor = self.monitor(warmup_seconds=2).start()
        capture = self.event(monitor, 'capture')
        self.finish(monitor)
        self.assertGreaterEqual(capture['capture_started'], 102)
        recorded = datetime.fromisoformat(capture['captured_at'])
        offset = (recorded - datetime(2026,11,3,tzinfo=timezone.utc)).total_seconds()
        self.assertGreater(offset, capture['capture_started'])

    def test_stop_while_warming_up_does_not_capture(self):
        monitor = self.monitor(warmup_seconds=100).start()
        self.event(monitor, 'state', lambda e: e['state']=='WARMING_UP')
        self.finish(monitor)
        self.assertTrue(self.sessions[0].closed.is_set())
        self.assertFalse(list((self.root/'scenes').glob('*/manifest.json')))

    def test_bad_rectification_binding_rejected_before_camera_open(self):
        self.arguments['calibration']['calibration_id'] = 'field-chess-missing-recipe'
        monitor = self.monitor().start()
        error = self.event(monitor, 'error')
        self.finish(monitor)
        self.assertEqual(error['stage'], 'configuration')
        self.assertEqual(self.sessions, [])

    def test_cancellation_at_analyzer_return_never_publishes_success(self):
        def analyze(*args, **kwargs):
            report = fake_analyzer(*args, **kwargs)
            monitor.stop()
            return report
        monitor = self.monitor(analyzer=analyze).start()
        self.event(monitor, 'stopped')
        self.assertFalse(any(e['kind']=='result' for e in self.events))
        self.assertTrue(self.sessions[0].closed.is_set())

    def test_unrectified_field_calibration_is_rejected_before_camera_open(self):
        self.arguments['calibration']['rectified'] = False
        monitor = self.monitor().start()
        error = self.event(monitor, 'error')
        self.finish(monitor)
        self.assertEqual(error['stage'], 'configuration')
        self.assertEqual(self.sessions, [])

    def test_separate_devices_need_manual_sync_confirmation_and_cannot_auto_capture(self):
        self.camera.update(layout='separate_devices', right_index=1)
        monitor = self.monitor().start()
        error = self.event(monitor, 'error')
        self.finish(monitor)
        self.assertEqual(error['stage'], 'configuration')
        self.assertIn('人工同步确认', error['message'])
        self.assertEqual(self.sessions, [])

    def test_client_mode_receives_current_manifest_and_cancel_event(self):
        with mock.patch('pipe_twin.auto_monitor.RemoteAnalysisClient') as client:
            client.return_value.analyze.side_effect = fake_analyzer
            monitor = self.monitor(mode='client', server_url='http://127.0.0.1:8771').start()
            self.event(monitor, 'result')
            self.finish(monitor)
            client.return_value.health.assert_called_once()
            call = client.return_value.analyze.call_args
            self.assertEqual(call.kwargs['cancel_event'], monitor._stop)
            self.assertEqual(len(json.loads(Path(call.args[0]).read_bytes())['capture']['capture_groups']), 1)

    def test_camera_disconnect_stops_and_does_not_publish_current_success(self):
        monitor = self.monitor().start()
        self.event(monitor, 'result')
        self.sessions[0].problem = 'disconnect'
        error = self.event(monitor, 'error')
        self.event(monitor, 'stopped')
        self.assertIn('unplugged', error['message'])
        self.assertFalse(monitor.active)

    def test_configuration_validation(self):
        for options in ({'mode':'server'}, {'mode':'client'}, {'interval_seconds':0},
                        {'interval_seconds':float('nan')}, {'warmup_seconds':-1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.monitor(**options)


if __name__ == '__main__':
    unittest.main()
