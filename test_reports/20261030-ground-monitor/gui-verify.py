"""Exercise the real Tk monitor, separately labelling replay and live capture."""
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import hashlib
import json
import runpy
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk
import ctypes
from ctypes import wintypes

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.elevation_gui import ElevationCaptureDialog
from pipe_twin.ground_monitor_gui import GroundMonitorDialog
from pipe_twin.stereo_camera import CapturedStereoPair
import pipe_twin.ground_monitor as backend

HELPERS = runpy.run_path(str(ROOT/'test_reports/20261030-function-review/gui-verify.py'))
SOURCE = ROOT/'outputs/elevation_workbench/captures/elevation-20261030-183635-b6b7b875'
OUT = ROOT/'outputs/ground-monitor-20261030/gui'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def screenshot(window, path):
    """Capture only this window, even when unrelated desktop windows cover it."""
    user, gdi = ctypes.windll.user32, ctypes.windll.gdi32
    user.GetAncestor.argtypes = [wintypes.HWND, wintypes.UINT]
    user.GetAncestor.restype = wintypes.HWND
    user.GetDC.argtypes = [wintypes.HWND]; user.GetDC.restype = wintypes.HDC
    user.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
    user.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    user.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
    gdi.CreateCompatibleDC.argtypes = [wintypes.HDC]; gdi.CreateCompatibleDC.restype = wintypes.HDC
    gdi.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
    gdi.CreateCompatibleBitmap.restype = wintypes.HANDLE
    gdi.SelectObject.argtypes = [wintypes.HDC, wintypes.HANDLE]; gdi.SelectObject.restype = wintypes.HANDLE
    gdi.DeleteObject.argtypes = [wintypes.HANDLE]; gdi.DeleteDC.argtypes = [wintypes.HDC]
    class BitmapHeader(ctypes.Structure):
        _fields_ = [('size', wintypes.DWORD), ('width', wintypes.LONG), ('height', wintypes.LONG),
            ('planes', wintypes.WORD), ('bits', wintypes.WORD), ('compression', wintypes.DWORD),
            ('image_size', wintypes.DWORD), ('xppm', wintypes.LONG), ('yppm', wintypes.LONG),
            ('used', wintypes.DWORD), ('important', wintypes.DWORD)]
    gdi.GetDIBits.argtypes = [wintypes.HDC, wintypes.HANDLE, wintypes.UINT, wintypes.UINT,
                            ctypes.c_void_p, ctypes.POINTER(BitmapHeader), wintypes.UINT]
    hwnd = user.GetAncestor(window.winfo_id(), 2)
    rect = wintypes.RECT(); assert user.GetClientRect(hwnd, ctypes.byref(rect))
    width, height = rect.right, rect.bottom
    dc = user.GetDC(hwnd); memory = gdi.CreateCompatibleDC(dc)
    bitmap = gdi.CreateCompatibleBitmap(dc, width, height)
    previous = gdi.SelectObject(memory, bitmap)
    try:
        assert user.PrintWindow(hwnd, memory, 3), 'Window-only screenshot failed'
        gdi.SelectObject(memory, previous)
        header = BitmapHeader(ctypes.sizeof(BitmapHeader), width, -height, 1, 32, 0, width*height*4, 0, 0, 0, 0)
        buffer = ctypes.create_string_buffer(width*height*4)
        assert gdi.GetDIBits(memory, bitmap, 0, height, buffer, ctypes.byref(header), 0) == height
        pixels = np.frombuffer(buffer.raw, np.uint8).reshape(height, width, 4)
        assert cv2.imwrite(str(path), pixels[:, :, :3])
    finally:
        gdi.SelectObject(memory, previous)
        gdi.DeleteObject(bitmap); gdi.DeleteDC(memory); user.ReleaseDC(hwnd, dc)


def wait_ui(root, predicate, seconds=20):
    deadline = time.monotonic()+seconds
    while time.monotonic() < deadline:
        root.update()
        if predicate():
            return
        time.sleep(.03)
    raise AssertionError('GUI condition timed out')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [SOURCE/'manifest.json', SOURCE/'report.json',
                 ROOT/'outputs/elevation_workbench/last_scene.json',
                 ROOT/'outputs/measurement_workbench/workbench_profile.json',
                 ROOT/'outputs/measurement_workbench/calibration_current.json']
    protected += list((SOURCE/'photos').rglob('*.png'))
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    manifest = json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
    profile = json.loads((ROOT/'outputs/measurement_workbench/workbench_profile.json').read_text(encoding='utf-8-sig'))
    result = json.loads((ROOT/'outputs/ground-monitor-20261030/historical-monitor.json').read_text(encoding='utf-8'))
    frame = manifest['capture']['capture_groups'][-1]['views']
    pair = CapturedStereoPair(*(cv2.imread(str(SOURCE/frame[r]['path'])) for r in ('left', 'right')),
                             frame['left']['captured_at'], frame['right']['captured_at'], 0., 'ARCHIVED_REPLAY', {})
    evidence = {'replay_scope': 'OFFLINE_UI_REPLAY_TIME_NOT_CURRENT_LIVE_ANGLE', 'screens': {}, 'live': {}}
    root = tk.Tk()
    root.withdraw()
    errors = []
    root.report_callback_exception = lambda *e: errors.append(''.join(traceback.format_exception(*e)))
    style = ttk.Style(root); style.theme_use('clam')
    style.configure('.', font=('Microsoft YaHei UI', 10))
    app = SimpleNamespace(root=root, tk=tk, ttk=ttk, messagebox=Mock(), filedialog=Mock())
    with patch('pipe_twin.workbench_profile.load_profile', return_value=(None, None)):
        owner = ElevationCaptureDialog(app, SOURCE/'manifest.json', output_root=OUT/'isolated', restore=False)
    owner.profile = profile
    monitor = None
    try:
        monitor = GroundMonitorDialog(owner)
        monitor.window.after_cancel(monitor.poll_id)
        monitor.poll_id = None
        monitor.window.title('离线界面核验 · 历史测角，不代表当前机位')
        event = threading.Event()
        monitor.worker = SimpleNamespace(active=True, stopping=event, stop=event.set)
        monitor.latest_pair, monitor.latest_sequence = pair, result['sequence']
        monitor.region = manifest['analysis']['elevation_auto']['registration']['ground_region']
        monitor.last_preview_at = time.monotonic()
        replay_now = datetime.fromisoformat(result['captured_at'])+timedelta(seconds=1)
        real_time = backend._time
        with patch('pipe_twin.ground_monitor._time', side_effect=lambda value=None: real_time(replay_now if value is None else value)):
            assert monitor._adopt_result((monitor.generation, pair, result['sequence'], result, None, 4.2))
            for size in ('1024x720', '1280x860'):
                monitor.window.geometry(size+'+10+10')
                monitor.window.deiconify()
                root.update()
                monitor.last_preview_at = time.monotonic()
                monitor._render_state(); monitor.draw_images(); root.update()
                clipped = HELPERS['clipped_controls'](monitor.window)
                assert not clipped, clipped
                assert monitor.angle.get() == '14.83°'
                assert '15.60' in monitor.pose_detail.get()
                assert monitor.sample_pair is pair
                assert monitor.state.snapshot()['status'] == 'ADJUST_TILT'
                evidence['screens'][size] = {'clipped_controls': clipped, 'left_angle': monitor.angle.get(),
                    'detail': monitor.pose_detail.get(), 'sample_caption': monitor.sample_caption.get(),
                    'image_heights': {k: v.winfo_height() for k, v in monitor.canvases.items()}}
                screenshot(monitor.window, OUT/(size+'-historical.png'))
        monitor._render_state(); monitor.draw_images(); root.update()
        assert monitor.angle.get() == '—' and not monitor.state.snapshot()['green']
        evidence['historical_now_expired'] = True
        monitor.worker.active = False
        monitor.close(); monitor = None

        # Actual live camera smoke test. No replay timestamps or time mocking.
        monitor = GroundMonitorDialog(owner)
        monitor.window.geometry('1024x720+10+10')
        monitor.window.title('实机核验 · 当前地面监控')
        root.update()
        assert monitor.start()
        wait_ui(root, lambda: monitor.latest_pair is not None and monitor.latest_sequence >= 8, 30)
        region = manifest['analysis']['elevation_auto']['registration']['ground_region']
        # Exercise the same mouse-coordinate mapping as the operator drag.
        monitor.draw_images(); root.update()
        scale, ox, oy = monitor.transforms['preview']
        x, y, w, h = region['rect_px']
        monitor.begin_region(SimpleNamespace(x=ox+x*scale, y=oy+y*scale))
        monitor.finish_region(SimpleNamespace(x=ox+(x+w)*scale, y=oy+(y+h)*scale))
        wait_ui(root, lambda: monitor.sample_result is not None, 30)
        measured_sequence = monitor.sample_result['sequence']
        wait_ui(root, lambda: monitor.latest_sequence >= measured_sequence+3, 15)
        monitor._render_state(); root.update()
        current = monitor.sample_result
        evidence['live'] = {k: current[k] for k in ('sequence', 'captured_at', 'status', 'valid', 'pose', 'sources')}
        evidence['live'].update(angle_text=monitor.angle.get(), detail=monitor.pose_detail.get(),
                                preview_sequence=monitor.latest_sequence, sample_is_bound=monitor.sample_pair is not None)
        assert monitor.latest_sequence > measured_sequence
        assert monitor.latest_sequence >= current['sequence']
        if current['valid']:
            assert current['pose']['physical_axes_known'] and monitor.angle.get() != '—'
        else:
            assert current['pose'] is None and monitor.angle.get() == '—'
            assert not monitor.state.snapshot()['green']
        screenshot(monitor.window, OUT/'live-monitor.png')
        monitor.stop()
        assert monitor.angle.get() == '—' and monitor.sample_pair is None
        wait_ui(root, lambda: not monitor.camera_running, 12)
        evidence['live']['stop_released_camera'] = True
        assert monitor.start()
        wait_ui(root, lambda: monitor.latest_pair is not None, 30)
        assert monitor.region is None and monitor.sample_result is None
        evidence['live']['restart_received_new_preview'] = True
        monitor.stop()
        wait_ui(root, lambda: not monitor.camera_running, 12)
        monitor.close(); monitor = None
        assert not errors, errors
    finally:
        if monitor is not None and not monitor.closed:
            monitor.close()
        owner.close()
        root.destroy()
    assert before == {str(p.relative_to(ROOT)): sha(p) for p in protected}
    evidence.update(source_unchanged=True, protected_source_sha256=before, callback_errors=errors)
    Path(__file__).with_name('gui-evidence.json').write_text(json.dumps(evidence, ensure_ascii=False,
                                                                    indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'screens': evidence['screens'], 'live': evidence['live']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
