import copy
from datetime import datetime, timedelta, timezone
import queue
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import pipe_twin.ground_monitor_gui as module
from pipe_twin.stereo_camera import CapturedStereoPair
from test_elevation_dataset import _calibration


def pair(value=80):
    stamp = datetime.now(timezone.utc).isoformat()
    return CapturedStereoPair(np.full((480, 640, 3), value, np.uint8),
        np.full((480, 640, 3), value+1, np.uint8), stamp, stamp, 0., 'HOST_SYSTEM_CLOCK', {})


def result(sequence=1, *, age=0, status='CONSISTENT_GROUND_PLANE'):
    valid = status == 'CONSISTENT_GROUND_PLANE'
    return {'sequence': sequence, 'captured_at': (datetime.now(timezone.utc)-timedelta(seconds=age)).isoformat(),
        'config_signature': 'fixed', 'status': status, 'valid': valid,
        'pose': {'off_normal_deg': 0., 'tilt_right_deg': 0., 'tilt_down_deg': 0.,
                 'right_off_normal_deg': 0., 'physical_axes_known': True,
                 'axis_target_camera': [0, 0, 1], 'right_axis_target_camera': [0, 0, 1]} if valid else None,
        'ground_check': {'status': status, 'surface_identity_source': 'OPERATOR_MARKED_GROUND'}}


def test_camera_settings_keep_auto_exposure_and_calibrated_resolution():
    options = module.camera_options(_calibration(), {'camera': {
        'exposure_ms': None, 'layout': 'separate_devices', 'left_index': 2, 'right_index': 4}})
    assert options['exposure_ms'] is None
    assert (options['eye_width'], options['eye_height']) == (640, 480)
    assert (options['left_index'], options['right_index']) == (2, 4)


def test_preview_stop_during_native_read_releases_on_the_same_worker_and_discards_late_frame():
    reading, release = threading.Event(), threading.Event()
    calls = []
    class Session:
        exposure_summary = 'auto'
        def __init__(self, **_kwargs): pass
        def open(self): calls.append(('open', threading.get_ident()))
        def read_pair(self):
            calls.append(('read', threading.get_ident()))
            reading.set(); release.wait(3)
            return pair()
        def close(self): calls.append(('close', threading.get_ident()))
    worker = module.PreviewWorker({}, None, session_factory=Session)
    worker.start()
    assert reading.wait(2)
    worker.stop()
    assert worker.active
    assert not any(c[0] == 'close' for c in calls)
    release.set(); worker.thread.join(2)
    assert not worker.active and worker.frames.empty()
    assert [c[0] for c in calls] == ['open', 'read', 'close']
    assert len({c[1] for c in calls}) == 1
    assert calls[0][1] != threading.get_ident()


def test_stop_during_open_does_not_start_preview():
    opening, release = threading.Event(), threading.Event()
    session = Mock(exposure_summary='auto')
    def open_camera():
        opening.set(); release.wait(3)
    session.open.side_effect = open_camera
    worker = module.PreviewWorker({}, None, session_factory=lambda **_: session)
    worker.start(); assert opening.wait(2)
    worker.stop(); release.set(); worker.thread.join(2)
    session.read_pair.assert_not_called()
    session.close.assert_called_once()
    assert worker.frames.empty()


@pytest.fixture
def dialog():
    import tkinter as tk
    from tkinter import ttk
    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(str(error))
    root.withdraw()
    owner = SimpleNamespace(window=root, app=SimpleNamespace(tk=tk, ttk=ttk),
        current_calibration=lambda: _calibration(), profile={'camera': {'exposure_ms': None}},
        analysis_settings={}, disparity_count=tk.StringVar(value='128'), matching_preset=tk.StringVar(value='正常'),
        message=tk.StringVar(), closed=False)
    d = module.GroundMonitorDialog(owner)
    d.window.withdraw()
    try:
        yield d
    finally:
        d.close()
        root.destroy()


def enable_fake_preview(d):
    stop = threading.Event()
    d.worker = SimpleNamespace(active=True, stopping=stop, stop=stop.set,
        frames=queue.Queue(maxsize=1), events=queue.Queue())
    d.latest_pair, d.latest_sequence = pair(90), 5
    d.last_preview_at = time.monotonic()


def test_result_is_bound_to_sample_not_newest_preview_and_roi_change_discards_late_result(dialog):
    d = dialog
    enable_fake_preview(d)
    sample = pair(30)
    token = d.generation
    assert d._adopt_result((token, sample, 1, result(), None, 1.2))
    d._render_state()
    assert d.sample_pair is sample and d.latest_pair.left[0, 0, 0] == 90
    assert '帧 1' in d.sample_caption.get() and ':' in d.sample_caption.get()
    d.clear_region()
    assert not d._adopt_result((token, sample, 2, result(2), None, 1.2))
    assert d.sample_pair is None and d.angle.get() == '—'


def test_no_more_frames_immediately_removes_old_support_and_angle(dialog, monkeypatch):
    d = dialog
    enable_fake_preview(d)
    paint = Mock(side_effect=lambda image, *_a, **_kw: image.copy())
    monkeypatch.setattr(module, 'paint_support', paint)
    d._adopt_result((d.generation, pair(), 1, result(), None, 1.))
    d._render_state()
    assert paint.called and d.angle.get() == '0.00°'
    paint.reset_mock()
    d.last_preview_at = time.monotonic()-3
    d._render_state()  # No incoming frame, result, or configure event.
    assert d.angle.get() == '—'
    paint.assert_not_called()
    text = [d.canvases['sample'].itemcget(i, 'text') for i in d.canvases['sample'].find_all()
            if d.canvases['sample'].type(i) == 'text']
    assert any('不显示支持涂色' in t for t in text)


def test_dark_result_and_expired_result_cannot_retain_last_angle(dialog):
    d = dialog
    enable_fake_preview(d)
    d._adopt_result((d.generation, pair(), 1, result(), None, 1.))
    d._render_state()
    d._adopt_result((d.generation, pair(), 2, result(2, status='NEAR_BLACK_FRAME'), None, .1))
    d._render_state()
    assert d.angle.get() == '—' and '接近全黑' in d.pose_detail.get()
    d._adopt_result((d.generation, pair(), 3, result(3, age=10), None, 10.))
    d._render_state()
    assert d.angle.get() == '—' and not d.state.snapshot()['green']


def test_active_drag_survives_preview_redraw(dialog):
    d = dialog
    enable_fake_preview(d)
    d.draw_images()
    d.begin_region(SimpleNamespace(x=15, y=20))
    d.drag_region(SimpleNamespace(x=110, y=120))
    d.draw_images()
    ids = d.canvases['preview'].find_withtag('drag')
    assert len(ids) == 1
    assert d.canvases['preview'].coords(ids[0]) == [15., 20., 110., 120.]


def test_stop_discards_late_measurement_and_does_not_change_owner_scene(dialog):
    d = dialog
    enable_fake_preview(d)
    before = copy.deepcopy((d.owner.profile, d.owner.analysis_settings))
    token = d.generation
    d.stop()
    assert d.worker.stopping.is_set()
    assert not d._adopt_result((token, pair(), 1, result(), None, 1.))
    assert d.sample_pair is None and d.angle.get() == '—'
    assert before == (d.owner.profile, d.owner.analysis_settings)


def test_tolerance_change_resets_previous_stability(dialog):
    d = dialog
    enable_fake_preview(d)
    d._adopt_result((d.generation, pair(), 1, result(), None, 1.))
    d.tolerance.set('1.5'); d.apply_tolerance()
    assert d.state.tolerance_deg == 1.5
    assert d.state.snapshot()['stable_count'] == 0 and d.sample_pair is None


def test_sample_roi_and_support_count_belong_to_frozen_measurement_not_current_drag(dialog):
    d = dialog
    enable_fake_preview(d)
    measured = result()
    measured['ground_check'].update(selection={'rect_px': [10, 20, 200, 200]},
                                    observed_support={'pixel_count': 1511})
    d.region = {'rect_px': [40, 50, 100, 100]}
    d._adopt_result((d.generation, pair(), 1, measured, None, 1.))
    canvas = d.canvases['sample']
    scale, ox, oy = d.transforms['sample']
    rectangle = canvas.find_withtag('sample-ground-region')
    assert len(rectangle) == 1
    assert canvas.coords(rectangle[0]) == [ox+10*scale, oy+20*scale, ox+210*scale, oy+220*scale]
    label = canvas.find_withtag('sample-support-count')
    assert '实测支持 1,511 像素' in canvas.itemcget(label[0], 'text')
