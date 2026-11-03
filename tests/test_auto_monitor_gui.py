from __future__ import annotations

import copy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import queue
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import numpy as np
import ezdxf

from pipe_twin.auto_monitor_gui import AutoMonitorDialog
from test_temporal_sections import stage


class FakeMonitor:
    def __init__(self, **configuration):
        self.configuration = configuration
        self.active = False
        self.next_due = 160.0
        self.events = queue.Queue()
        self.starts = self.stops = 0

    def start(self):
        self.starts += 1
        self.active = True
        return self

    def stop(self):
        self.stops += 1


class AutoMonitorGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import tkinter as tk
        from tkinter import ttk
        try:
            cls.root = tk.Tk()
        except tk.TclError as error:
            raise unittest.SkipTest(str(error)) from error
        cls.tk, cls.ttk = tk, ttk
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        from pipe_twin.dxf_elevation import catalog_from_dxf
        from test_elevation_dataset import _calibration
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = 100.0
        self.model_path = Path(self.tmp.name) / "model.dxf"
        model = ezdxf.new()
        model.units = 4
        model.modelspace().add_circle((0, 0), radius=20)
        model.modelspace().add_circle((100, 40), radius=25)
        model.saveas(self.model_path)
        specs, _ = catalog_from_dxf(self.model_path, axis_world=(0, 0, 1), unitless_unit="millimeter")
        self.configuration = {"dataset_arguments": {"pipe_specs": specs, "model_path": self.model_path,
            "output_root": Path(self.tmp.name) / 'captures',
            "calibration": _calibration(), "stl_unit": "millimeter",
            "registration_settings": {"axis_world": [0, 0, 1], "camera_side_world": [0, 1, 0]}},
            "camera_options": {"device_index": 0}, "mode": "local", "server_url": None}
        self.owner = SimpleNamespace(window=self.root, closed=False, report={"owner": "unchanged"}, last_manifest=None,
            app=SimpleNamespace(tk=self.tk, ttk=self.ttk, messagebox=mock.Mock()),
            monitor_configuration=mock.Mock(side_effect=lambda: self.configuration),
            monitor_started=mock.Mock(return_value=True), monitor_stopped=mock.Mock())
        self.factory = mock.Mock(side_effect=FakeMonitor)
        self.dialog = AutoMonitorDialog(self.owner, monitor_factory=self.factory, clock=lambda: self.now)
        self.dialog.window.withdraw()

    def tearDown(self):
        if self.dialog.monitor is not None:
            self.dialog.monitor.active = False
        self.dialog.close()
        self.root.update()

    def capture(self, cycle):
        from pipe_twin.elevation_dataset import create_elevation_dataset
        root = Path(self.tmp.name) / str(cycle)
        root.mkdir(exist_ok=True)
        paths = {}
        for role, value in (("left", 80), ("right", 100)):
            path = root / f"{role}.png"
            image = np.full((480, 640, 3), value, np.uint8)
            cv2.rectangle(image, (180, 30), (240, 340), (40, 40, 220), -1)
            cv2.rectangle(image, (350, 30), (425, 340), (210, 90, 35), -1)
            cv2.imwrite(str(path), image)
            paths[role + "_path"] = path
        arguments = copy.deepcopy(self.configuration['dataset_arguments'])
        arguments['output_root'] = root / 'scenes'
        captured_at = (datetime.fromisoformat('2026-11-03T10:30:00+08:00') + timedelta(minutes=cycle-1)).isoformat()
        path = create_elevation_dataset(**arguments, **paths, pair_confirmed=True, previous_manifest=None,
            left_time=captured_at, right_time=captured_at)
        return {"kind": "capture", "cycle_id": cycle, "manifest_path": str(path), "captured_at": captured_at}

    def result(self, capture):
        from test_assist_display import report_fixture
        report = report_fixture()
        assistance = report["installation_assist"]
        for observation, identity in zip(assistance["observations"], ("0", "1")):
            observation["observation_id"] = identity
        for choice in assistance["ranked_hypotheses"]:
            for assignment, identity in zip(choice["assignments"], ("0", "1")):
                assignment["observation_id"] = identity
        report["section_evaluation"] = stage()
        report['generated_at'] = (datetime.fromisoformat(capture['captured_at']) + timedelta(seconds=12)).isoformat()
        raw = Path(capture["manifest_path"]).read_bytes()
        manifest = json.loads(raw)
        report["inputs"] = {"manifest": hashlib.sha256(raw).hexdigest(), "model": manifest["model"]["sha256"]}
        for group in manifest["capture"]["capture_groups"]:
            report["inputs"].update({f"{group['capture_id']}_{role}": view["sha256"] for role, view in group["views"].items()})
        return dict(capture, kind="result", report=report)

    def complete_cycle(self, cycle=1):
        self.dialog.handle_event({"kind": "state", "cycle_id": cycle, "state": "CAPTURING"})
        capture = self.capture(cycle)
        self.dialog.handle_event(capture)
        result = self.result(capture)
        self.dialog.handle_event(result)
        self.root.update()
        return result

    def assert_results_empty(self):
        self.assertIsNone(self.dialog.report)
        self.assertFalse(self.dialog.table.get_children())
        self.assertFalse(self.dialog.cad_canvas.find_withtag("pipe"))
        self.assertFalse(self.dialog.cad_canvas.find_withtag("state:ASSIST_CANDIDATE"))
        self.assertFalse(self.dialog.section_canvas.find_withtag("scene-pipe"))

    def test_opening_is_idle_and_does_not_acquire_camera_or_configuration(self):
        self.factory.assert_not_called()
        self.owner.monitor_configuration.assert_not_called()
        self.owner.monitor_started.assert_not_called()
        self.assertFalse(self.dialog.active)
        self.assertEqual(self.dialog.interval.get(), "1 分钟")
        self.assertEqual(tuple(self.dialog.interval_box.cget("values")), ("1 分钟", "2 分钟", "3 分钟", "5 分钟", "10 分钟"))
        self.assert_results_empty()

    def test_start_uses_snapshot_interval_and_preserves_owner_results(self):
        self.dialog.interval.set("3 分钟")
        self.dialog.start()
        self.assertTrue(self.dialog.active)
        self.assertEqual(self.dialog.monitor.configuration["interval_seconds"], 180)
        self.owner.monitor_started.assert_called_once()
        self.configuration["dataset_arguments"]["pipe_specs"][0]["nominal_diameter_mm"] = 999
        self.assertEqual(self.dialog.pipes[0]["nominal_diameter_mm"], 40)
        self.dialog.start()
        self.factory.assert_called_once()
        self.assertEqual(self.owner.report, {"owner": "unchanged"})
        self.assertIsNone(self.owner.last_manifest)

    def test_server_mode_or_camera_conflict_never_constructs_backend(self):
        self.configuration["mode"] = "server"
        self.dialog.start()
        self.assertIn("服务器端", self.dialog.error.get())
        self.factory.assert_not_called()
        self.owner.monitor_started.assert_not_called()
        self.configuration["mode"] = "local"
        self.owner.monitor_started.side_effect = ValueError("相机正在使用")
        self.dialog.start()
        self.assertIn("相机正在使用", self.dialog.error.get())
        self.factory.assert_not_called()
        self.owner.app.messagebox.showerror.assert_not_called()

    def test_new_capture_immediately_clears_all_sections_table_and_old_late_result(self):
        self.dialog.start()
        result = self.complete_cycle()
        self.assertTrue(self.dialog.table.get_children())
        self.assertTrue(self.dialog.section_canvas.find_withtag("scene-pipe"))
        self.assertTrue(self.dialog.cad_canvas.find_withtag("state:ASSIST_CANDIDATE"))
        self.dialog.handle_event({"kind": "state", "cycle_id": 2, "state": "CAPTURING"})
        self.assert_results_empty()
        self.dialog.handle_event(result)
        self.assert_results_empty()
        self.assertIn("第 2 轮", self.dialog.result_caption.get())
        self.dialog.handle_event(self.capture(2))
        self.dialog.handle_event({"kind": "state", "cycle_id": 2, "state": "ANALYZING"})
        self.assert_results_empty()
        self.assertIn("分析中", self.dialog.result_caption.get())

    def test_analysis_state_alone_also_suppresses_previous_results(self):
        self.dialog.start()
        self.complete_cycle()
        self.dialog.handle_event({"kind": "state", "cycle_id": 2, "state": "ANALYZING"})
        self.assert_results_empty()

    def test_pre_capture_red_warning_blinks_for_ten_seconds_without_hiding_last_result(self):
        self.dialog.start()
        self.complete_cycle()
        current = self.dialog.report
        self.dialog.monitor.next_due = 110.
        self.dialog.handle_event({'kind': 'state', 'cycle_id': 1, 'state': 'PREPARING'})
        self.dialog._countdown()
        self.assertIn('10 秒', self.dialog.capture_warning.get())
        self.assertEqual(str(self.dialog.warning_label.cget('foreground')), '#C62828')
        self.now = 100.6
        self.dialog._countdown()
        self.assertEqual(self.dialog.capture_warning.get(), '')
        self.now = 105.
        self.dialog._countdown()
        self.assertIn('5 秒', self.dialog.capture_warning.get())
        self.assertIs(self.dialog.report, current)
        self.dialog.handle_event({'kind': 'state', 'cycle_id': 2, 'state': 'CAPTURING'})
        self.assert_results_empty()
        self.assertEqual(self.dialog.capture_warning.get(), '')
        self.assertTrue(self.dialog.history_table.get_children())

    def hinted_cycle(self, cycle, pipes):
        self.dialog.handle_event({'kind': 'state', 'cycle_id': cycle, 'state': 'CAPTURING'})
        capture = self.capture(cycle)
        self.dialog.handle_event(capture)
        event = self.result(capture)
        data = event['report']['installation_assist']
        data['status'] = 'LIKELY'
        data['observations'] = data['observations'][:len(pipes)]
        data['ranked_hypotheses'] = data['ranked_hypotheses'][:1]
        data['ranked_hypotheses'][0]['assignments'] = [
            {'observation_id': str(i), 'pipe_id': pipe} for i, pipe in enumerate(pipes)]
        for observation, pipe in zip(data['observations'], pipes):
            observation.update(status='LIKELY', suggested_pipe_id=pipe)
        self.dialog.handle_event(event)
        return event

    def test_history_records_added_removed_time_and_ignores_duplicate_result(self):
        self.dialog.start()
        first = self.hinted_cycle(1, ['P001'])
        second = self.hinted_cycle(2, ['P002'])
        self.dialog.handle_event(second)
        rows = [self.dialog.history_table.item(item, 'values') for item in self.dialog.history_table.get_children()]
        self.assertEqual([row[2:] for row in rows], [('基线', 'P001'), ('新增提示', 'P002'), ('减少提示', 'P001')])
        saved = [json.loads(line) for line in self.dialog.history_path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[1]['captured_at'], second['captured_at'])
        self.assertEqual(saved[1]['removed'], ['P001'])
        self.assertEqual(saved[1]['added'], ['P002'])
        self.dialog.handle_event({'kind': 'state', 'cycle_id': 3, 'state': 'CAPTURING'})
        self.dialog.handle_event({'kind': 'error', 'cycle_id': 3, 'message': 'test failure'})
        self.assert_results_empty()
        saved = [json.loads(line) for line in self.dialog.history_path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(saved[-1]['kind'], 'ERROR')
        self.assertEqual(saved[-1]['removed'], [])
        self.dialog.handle_event(first)
        self.assertEqual(len(self.dialog.history_path.read_text(encoding='utf-8').splitlines()), 3)

    def test_failed_history_file_write_does_not_hide_valid_current_result(self):
        self.dialog.start()
        self.dialog.history_path = Path(self.tmp.name)  # Existing directory cannot be appended as a file.
        self.hinted_cycle(1, ['P001'])
        self.assertIsNotNone(self.dialog.report)
        self.assertTrue(self.dialog.history_table.get_children())
        self.assertIn('记录保存失败', self.dialog.error.get())

    def test_restart_retains_visible_history_but_creates_new_baseline(self):
        self.dialog.start()
        self.hinted_cycle(1, ['P001'])
        first_path = self.dialog.history_path
        self.dialog.monitor.active = False
        self.dialog.start()
        self.hinted_cycle(1, ['P002'])
        self.assertNotEqual(self.dialog.history_path, first_path)
        rows = [self.dialog.history_table.item(item, 'values') for item in self.dialog.history_table.get_children()]
        self.assertEqual([row[2] for row in rows], ['基线', '基线'])

    def test_failure_keeps_sections_empty_and_late_same_cycle_result_cannot_reappear(self):
        self.dialog.start()
        result = self.complete_cycle()
        self.dialog.handle_event({"kind": "error", "cycle_id": 1, "message": "服务器连接中断"})
        self.assert_results_empty()
        self.dialog.handle_event(result)
        self.assert_results_empty()
        self.assertIn("服务器连接中断", self.dialog.error.get())
        self.owner.app.messagebox.showerror.assert_not_called()

    def test_result_metadata_and_geometry_remain_source_bound(self):
        self.dialog.start()
        capture = self.capture(1)
        self.dialog.handle_event(capture)
        event = self.result(capture)
        before = copy.deepcopy(event["report"])
        self.dialog.handle_event(event)
        self.assertEqual(before, event["report"])
        self.assertIn("第 1 轮", self.dialog.result_caption.get())
        self.assertIn("10:30:00", self.dialog.result_caption.get())
        self.assertIn("10:30:12", self.dialog.result_caption.get())
        values = [self.dialog.table.item(i, "values") for i in self.dialog.table.get_children()]
        self.assertEqual([row[1] for row in values], ["40.00", "50.00"])
        self.assertEqual([row[2] for row in values], ["P001", "P002"])
        self.assertEqual(self.owner.report, {"owner": "unchanged"})

    def test_missing_metric_fit_never_substitutes_nominal_or_prior_diameter(self):
        self.dialog.start()
        capture = self.capture(1)
        self.dialog.handle_event(capture)
        event = self.result(capture)
        event["report"].pop("section_evaluation")
        event["report"]["installation_assist"]["observations"][0]["prior_diameter_mm"] = 100
        self.dialog.handle_event(event)
        values = [self.dialog.table.item(i, "values") for i in self.dialog.table.get_children()]
        self.assertEqual([row[1] for row in values], ["—", "—"])

    def deliver(self, event):
        self.dialog.monitor.events.put(event)
        self.dialog.poll()

    def asset(self, capture, name):
        path = Path(capture["manifest_path"])
        manifest = json.loads(path.read_text(encoding="utf-8"))
        relative = (manifest["model"]["path"] if name == "model"
                    else manifest["capture"]["capture_groups"][0]["views"][name]["path"])
        return path.parent / relative

    def test_missing_manifest_model_or_photo_report_binding_is_rejected(self):
        self.dialog.start()
        for cycle, missing in enumerate(("manifest", "model", "left"), 1):
            with self.subTest(binding=missing):
                capture = self.capture(cycle)
                self.dialog.handle_event(capture)
                event = self.result(capture)
                key = (next(key for key in event["report"]["inputs"] if key.endswith("_left"))
                       if missing == "left" else missing)
                del event["report"]["inputs"][key]
                self.deliver(event)
                self.assert_results_empty()
                self.assertIn("来源绑定", self.dialog.error.get())

    def test_capture_rejects_photo_or_model_changed_before_display(self):
        self.dialog.start()
        for cycle, name in enumerate(("left", "right", "model"), 1):
            with self.subTest(asset=name):
                capture = self.capture(cycle)
                path = self.asset(capture, name)
                path.write_bytes(path.read_bytes() + b"changed-after-snapshot")
                self.deliver(capture)
                self.assert_results_empty()
                self.assertTrue(self.dialog.error.get())
                self.assertTrue(all(pane.image is None for pane in self.dialog.photos.values()))

    def test_result_rechecks_every_displayed_photo_and_model_file(self):
        self.dialog.start()
        for cycle, name in enumerate(("left", "right", "model"), 1):
            with self.subTest(asset=name):
                capture = self.capture(cycle)
                self.dialog.handle_event(capture)
                event = self.result(capture)
                path = self.asset(capture, name)
                path.write_bytes(path.read_bytes() + b"changed-during-analysis")
                self.deliver(event)
                self.assert_results_empty()
                self.assertIn("分析期间改变", self.dialog.error.get())
                self.assertEqual(self.owner.report, {"owner": "unchanged"})

    def test_updated_manifest_cannot_rebind_a_new_photo_to_an_old_capture(self):
        self.dialog.start()
        capture = self.capture(1)
        self.dialog.handle_event(capture)
        path = self.asset(capture, "left")
        path.write_bytes(path.read_bytes() + b"replaced")
        manifest_path = Path(capture["manifest_path"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["capture"]["capture_groups"][0]["views"]["left"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.deliver(self.result(capture))  # Even a self-consistent new report belongs to a different snapshot.
        self.assert_results_empty()
        self.assertIn("现场文件已改变", self.dialog.error.get())

    def test_valid_capture_fixture_is_a_portable_single_pair_dataset(self):
        from pipe_twin.elevation_dataset import load_elevation_dataset
        capture = self.capture(1)
        loaded = load_elevation_dataset(Path(capture["manifest_path"]))
        groups = loaded["manifest"]["capture"]["capture_groups"]
        self.assertEqual(len(groups), 1)
        self.assertFalse(groups[0].get("temporal_frames"))
        self.assertTrue(loaded["manifest"]["model"]["sha256"])

    def test_countdown_uses_monotonic_and_does_not_schedule_catchup_jobs(self):
        self.dialog.start()
        self.dialog.handle_event({"kind": "state", "cycle_id": 1, "state": "ANALYZING"})
        self.now = 159.1
        self.dialog._countdown()
        self.assertIn("1 秒", self.dialog.countdown.get())
        self.now = 300
        self.dialog._countdown()
        self.assertIn("等待本轮完成", self.dialog.countdown.get())
        self.assertIn("不累积任务", self.dialog.countdown.get())
        self.assertEqual(self.dialog.monitor.starts, 1)

    def test_stop_does_not_release_lock_or_allow_restart_until_workers_idle(self):
        self.dialog.start()
        self.complete_cycle()
        self.dialog.handle_event({"kind": "state", "cycle_id": 2, "state": "CAPTURING"})
        capture = self.capture(2)
        self.dialog.handle_event(capture)
        self.dialog.stop()
        self.dialog.handle_event(self.result(capture))
        self.assert_results_empty()
        self.dialog.start()
        self.factory.assert_called_once()
        self.owner.monitor_stopped.assert_not_called()
        self.dialog.monitor.active = False
        self.dialog.poll()
        self.owner.monitor_stopped.assert_called_once()
        self.dialog.poll()
        self.owner.monitor_stopped.assert_called_once()

    def test_close_waits_for_release_and_owner_close_detaches_tk_immediately(self):
        self.dialog.start()
        self.dialog.close()
        self.assertFalse(self.dialog.closed)
        self.assertTrue(self.dialog.closing)
        self.owner.monitor_stopped.assert_not_called()
        self.owner.closed = True
        self.dialog.close()
        self.assertTrue(self.dialog.closed)
        self.assertGreaterEqual(self.dialog.monitor.stops, 1)

    def test_constructor_failure_releases_acquired_lock_inline(self):
        self.factory.side_effect = ValueError("配置不可用")
        self.dialog.start()
        self.owner.monitor_stopped.assert_called_once()
        self.assertFalse(self.dialog.locked)
        self.assertIn("配置不可用", self.dialog.error.get())

    def test_layout_keeps_photos_sections_table_and_controls_visible_at_both_sizes(self):
        self.dialog.window.deiconify()
        self.dialog.start()
        self.complete_cycle()
        for size in ("1024x720", "1280x860"):
            with self.subTest(size=size):
                self.dialog.window.geometry(size)
                self.root.update()
                for widget in [self.dialog.start_button, self.dialog.stop_button, self.dialog.cad_canvas,
                               self.dialog.section_canvas, self.dialog.table, *(pane.canvas for pane in self.dialog.photos.values())]:
                    self.assertTrue(widget.winfo_viewable())
                    self.assertGreater(widget.winfo_width(), 30)
                    bottom = widget.winfo_rooty() + widget.winfo_height()
                    self.assertLessEqual(bottom, self.dialog.window.winfo_rooty() + self.dialog.window.winfo_height())
                self.assertGreater(self.dialog.cad_canvas.winfo_height(), 250)
                self.assertGreater(self.dialog.photos["left"].canvas.winfo_height(), 140)


if __name__ == "__main__":
    unittest.main()
