"""Live camera pointing aid; camera ownership and measurements stay isolated."""
from __future__ import annotations

import base64
import copy
from dataclasses import replace
from datetime import datetime
import math
import queue
import threading
import time

import cv2

from .ground_check import normalize_ground_region
from .ground_monitor import GroundMonitorState, inspect_live_ground
from .ground_visualization import paint_support
from .stereo_camera import LAYOUT_SIDE_BY_SIDE_LR, StereoCameraSession


def _latest(target, value):
    try:
        target.get_nowait()
    except queue.Empty:
        pass
    target.put_nowait(value)


def camera_options(calibration, profile):
    """Preserve saved automatic exposure (None) and the calibrated image size."""
    from .stereo_analyzer import _calibration_from_manifest
    parsed = _calibration_from_manifest(calibration)
    camera = (profile or {}).get('camera') or {}
    options = dict(layout=camera.get('layout', LAYOUT_SIDE_BY_SIDE_LR),
                   left_index=camera.get('left_index', 0), right_index=camera.get('right_index', 1),
                   eye_width=parsed.left.width, eye_height=parsed.left.height,
                   exposure_ms=camera.get('exposure_ms', 5.))
    if type(camera.get('backend')) is int:
        options['backend'] = camera['backend']
    return options


class PreviewWorker:
    """The only thread touching a session, including its final release."""
    def __init__(self, options, rectifier, *, session_factory=None):
        self.options, self.rectifier = options, rectifier
        self.session_factory = session_factory or StereoCameraSession
        self.frames = queue.Queue(maxsize=1)
        self.events = queue.Queue()
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    @property
    def active(self):
        return self.thread.is_alive()

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopping.set()

    def _run(self):
        session = None
        try:
            session = self.session_factory(**self.options)
            session.open()
            if self.stopping.is_set():
                return
            self.events.put(('opened', session.exposure_summary))
            sequence = 0
            while not self.stopping.is_set():
                pair = session.read_pair()
                if self.stopping.is_set():
                    break
                if self.rectifier is not None:
                    pair = replace(pair, left=self.rectifier.rectify('left', pair.left),
                                   right=self.rectifier.rectify('right', pair.right))
                if self.stopping.is_set():
                    break
                sequence += 1
                _latest(self.frames, (sequence, pair))
                self.stopping.wait(.02)
        except Exception as error:
            self.events.put(('error', str(error)))
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception as error:
                    self.events.put(('error', str(error)))
            self.events.put(('closed', None))


STATE_TEXT = {
    'WAITING_FOR_MEASUREMENT': '等待选区和有效测量', 'ADJUST_TILT': '尚未进入辅助容差，请按方向缓慢调整',
    'TARGET_STABILIZING': '已进入辅助容差，请保持不动', 'ALIGNED_STABLE': '两眼均进入辅助容差且连续稳定',
    'RECTIFIED_REFERENCE_ONLY': '仅矫正光轴参考，无法确认实际镜头是否垂直',
    'EXPIRED': '测量已过期，请等待新的有效结果', 'EXPIRED_STABILITY_WINDOW': '稳定记录已过期，重新检查中',
    'CAPTURE_UNHEALTHY': '画面质量未通过，无法测角；请检查照明/曝光',
    'NEAR_BLACK_FRAME': '当前画面接近全黑，无法测角；请检查照明/曝光',
    'GROUND_PLANE_INCONSISTENT': '左右地面不一致，请重新框选连续裸地',
    'INSUFFICIENT_GROUND_SUBPIXEL_SUPPORT': '有效地面匹配不足，请选择有纹理的裸地',
    'STEREO_UNSYNCHRONIZED': '左右拍摄不同步，无法测角',
    'INSUFFICIENT_GROUND_DEPTH': '有效地面深度不足，请调整裸地选区',
}


class GroundMonitorDialog:
    def __init__(self, owner):
        self.owner, self.app = owner, owner.app
        tk, ttk = self.app.tk, self.app.ttk
        self.closed = False
        self.generation = 0
        self.worker = None
        self.analysis_thread = None
        self.results = queue.Queue(maxsize=1)
        self.state = GroundMonitorState()
        self.region = None
        self.latest_pair = None
        self.latest_sequence = None
        self.sample_pair = None
        self.sample_result = None
        self.sample_generation = None
        self.next_measurement_at = 0.
        self.last_preview_at = 0.
        self.drag_start = None
        self.drag_end = None
        self._paint_valid = False
        self.photos = {}
        self.transforms = {}
        self.window = tk.Toplevel(owner.window)
        self.window.title('实时地面调姿 · 垂直向下')
        self.window.geometry('1000x680')
        self.window.minsize(900, 600)
        self.window.transient(owner.window)
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.window.bind('<Escape>', lambda _e: self.stop())
        frame = ttk.Frame(self.window, padding=10)
        frame.pack(fill='both', expand=True)
        controls = ttk.Frame(frame); controls.pack(fill='x')
        self.start_button = ttk.Button(controls, text='开始实时调姿', command=self.start)
        self.start_button.pack(side='left', padx=(0, 6))
        self.stop_button = ttk.Button(controls, text='停止并释放相机', command=self.stop, state='disabled')
        self.stop_button.pack(side='left')
        ttk.Label(controls, text='辅助容差 ±').pack(side='left', padx=(20, 0))
        self.tolerance = tk.StringVar(value='2')
        ttk.Entry(controls, textvariable=self.tolerance, width=5).pack(side='left')
        ttk.Label(controls, text='°').pack(side='left')
        ttk.Button(controls, text='应用', command=self.apply_tolerance).pack(side='left', padx=5)
        ttk.Button(controls, text='清除地面框', command=self.clear_region).pack(side='right')
        ttk.Label(frame, text='① 开始预览  ② 在左图拖框选裸地  ③ 缓慢调整相机并停稳；目标是光轴垂直地面（0°）。\n辅助容差不代表精度；绿色涂色只是地面支持，不等于调姿达标。地面不能确定朝向角。',
                  wraplength=950).pack(anchor='w', pady=(8, 4))
        reading = ttk.Frame(frame); reading.pack(fill='x')
        numbers = ttk.Frame(reading); numbers.pack(side='left', fill='both', expand=True)
        self.axis_label = tk.StringVar(value='左镜头偏离地面法线（目标 0°）')
        ttk.Label(numbers, textvariable=self.axis_label).pack(anchor='w')
        self.angle = tk.StringVar(value='—')
        self.angle_label = ttk.Label(numbers, textvariable=self.angle, font=('Microsoft YaHei UI', 36), foreground='#7A5600')
        self.angle_label.pack(anchor='w')
        self.pose_detail = tk.StringVar(value='请开始预览并框选裸地')
        ttk.Label(numbers, textvariable=self.pose_detail, wraplength=700).pack(anchor='w')
        self.direction = tk.Canvas(reading, width=210, height=142, bg='#F3F5F7', highlightthickness=0)
        self.direction.pack(side='right', padx=(10, 0))
        pictures = ttk.Frame(frame); pictures.pack(fill='both', expand=True, pady=8)
        pictures.columnconfigure(0, weight=1); pictures.columnconfigure(1, weight=1)
        pictures.rowconfigure(1, weight=1)
        ttk.Label(pictures, text='实时左目矫正预览 · 拖框选裸地').grid(row=0, column=0, sticky='w')
        self.sample_caption = tk.StringVar(value='最近测量证据 · 尚无结果')
        ttk.Label(pictures, textvariable=self.sample_caption).grid(row=0, column=1, sticky='w', padx=(8, 0))
        self.canvases = {}
        for i, role in enumerate(('preview', 'sample')):
            canvas = tk.Canvas(pictures, bg='#152235', highlightthickness=0, width=420, height=210)
            canvas.grid(row=1, column=i, sticky='nsew', padx=(0 if i == 0 else 8, 0))
            canvas.bind('<Configure>', lambda _e: self.draw_images())
            self.canvases[role] = canvas
        preview = self.canvases['preview']
        preview.bind('<ButtonPress-1>', self.begin_region)
        preview.bind('<B1-Motion>', self.drag_region)
        preview.bind('<ButtonRelease-1>', self.finish_region)
        self.status = tk.StringVar(value='未打开相机；使用③双目抓拍保存的设备和曝光设置。')
        self.status_label = ttk.Label(frame, textvariable=self.status, wraplength=950, foreground='#355371')
        self.status_label.pack(anchor='w')
        ttk.Label(frame, text='有矫正配方时显示实际镜头光轴；历史报告采用矫正后的虚拟光轴，数值可略有不同。\n调姿完成后：停止 → 重新双目抓拍 → 重画地面与管段范围 → 保存并评估。此窗口不更新现场照片和报告。',
                  wraplength=950).pack(anchor='w', pady=(7, 0))
        self._render_state()
        self.poll_id = self.window.after(60, self.poll)

    @property
    def camera_running(self):
        return bool(self.worker and self.worker.active)

    def start(self):
        if self.closed or self.camera_running:
            return False
        try:
            from .calibration_wizard import rectifier_for_calibration
            self.calibration = copy.deepcopy(self.owner.current_calibration())
            profile = copy.deepcopy(self.owner.profile)
            rectifier = rectifier_for_calibration(self.calibration, profile)
            self.recipe = copy.deepcopy(rectifier.recipe) if rectifier is not None else None
            self.settings = copy.deepcopy(self.owner.analysis_settings)
            self.settings['stereo_matching'] = {**self.settings.get('stereo_matching', {}),
                'num_disparities': int(self.owner.disparity_count.get()),
                'preprocessing': 'low_light' if self.owner.matching_preset.get() == '弱光降噪' else 'none'}
            options = camera_options(self.calibration, profile)
            self._clear_measurement(clear_region=True)
            self.worker = PreviewWorker(options, rectifier)
            self.worker.start()
            self.last_preview_at = time.monotonic()
            self.start_button.configure(state='disabled')
            self.stop_button.configure(state='normal')
            self.status.set('正在打开保存的相机设备；驱动连接期间窗口仍可停止。')
            return True
        except (OSError, ValueError) as error:
            self.status.set(str(error))
            return False

    def _clear_measurement(self, *, clear_region=False):
        self.generation += 1
        self.state.reset()
        self.sample_pair = self.sample_result = None
        self.sample_generation = None
        if clear_region:
            self.region = None
        self.next_measurement_at = 0.
        self._render_state()
        self.draw_images()

    def stop(self):
        if self.worker:
            self.worker.stop()
        self.latest_pair = None
        self.latest_sequence = None
        self.drag_start = self.drag_end = None
        self._clear_measurement()
        self.stop_button.configure(state='disabled')
        self.start_button.configure(state='disabled' if self.camera_running else 'normal')
        self.status.set('正在停止并释放相机，请稍候…' if self.camera_running else '已停止；重新开始后请重新框选裸地。')

    def apply_tolerance(self):
        try:
            self.state = GroundMonitorState(tolerance_deg=float(self.tolerance.get()))
        except ValueError:
            self.status.set('辅助容差请输入大于 0 且不超过 45 的角度。')
            return
        self._clear_measurement()
        self.status.set('辅助容差已更新，重新检查连续稳定性；此数值不是精度承诺。')

    def clear_region(self):
        self._clear_measurement(clear_region=True)
        self.status.set('请在实时左图重新拖框选择连续裸地，避开管道、支架和阴影边缘。')

    def begin_region(self, event):
        if self.latest_pair is not None and self.camera_running and not self.worker.stopping.is_set():
            self.drag_start = (event.x, event.y)
            self.drag_end = self.drag_start
            self._clear_measurement(clear_region=True)

    def drag_region(self, event):
        canvas = self.canvases['preview']
        canvas.delete('drag')
        if self.drag_start:
            self.drag_end = (event.x, event.y)
            canvas.create_rectangle(*self.drag_start, event.x, event.y, outline='#E7A9FF', width=2, tags='drag')

    def finish_region(self, event):
        if self.drag_start is None or self.latest_pair is None:
            return
        from .elevation_gui import image_region
        rect = image_region(self.drag_start, (event.x, event.y), self.transforms['preview'], self.latest_pair.left.shape[1::-1])
        self.drag_start = self.drag_end = None
        self.canvases['preview'].delete('drag')
        try:
            self.region = normalize_ground_region({'image_size': list(self.latest_pair.left.shape[1::-1]), 'rect_px': rect})
            self.status.set('已框选裸地；请缓慢调姿，每次调整后停稳。测角按计算耗时更新。')
        except ValueError as error:
            self.status.set(str(error))
        self.draw_images()

    def _measure(self, pair, sequence):
        generation = self.generation
        self.next_measurement_at = float('inf')
        arguments = dict(pair=pair, sequence=sequence, calibration=copy.deepcopy(self.calibration),
                         region=copy.deepcopy(self.region), analysis_settings=copy.deepcopy(self.settings),
                         rectification_recipe=copy.deepcopy(self.recipe))
        def task():
            started = time.monotonic()
            try:
                result, error = inspect_live_ground(**arguments), None
            except Exception as problem:
                result, error = None, str(problem)
            _latest(self.results, (generation, pair, sequence, result, error, time.monotonic()-started))
        self.analysis_thread = threading.Thread(target=task, daemon=True)
        self.analysis_thread.start()

    def _adopt_result(self, item):
        generation, pair, sequence, result, error, elapsed = item
        if self.closed or generation != self.generation or not self.camera_running or self.worker.stopping.is_set():
            return False
        self.next_measurement_at = time.monotonic()+1.
        if error:
            self.state.reset()
            self.sample_result = self.sample_pair = None
            self.status.set('本次测角失败：'+error)
        else:
            self.sample_pair, self.sample_result, self.sample_generation = pair, result, generation
            self.state.update(result, expected_sequence=sequence)
            self.status.set(f'本次计算 {elapsed:.1f} 秒；右图只显示该次采样证据，左图继续预览。')
        self.draw_images()
        return True

    def poll(self):
        if self.closed:
            return
        if self.worker:
            try:
                while True:
                    event, payload = self.worker.events.get_nowait()
                    if event == 'opened' and not self.worker.stopping.is_set():
                        self.status.set(str(payload)+'；请在左图拖框选裸地。')
                    elif event == 'error':
                        self.stop()
                        self.status.set('相机无法继续预览：'+str(payload))
                    elif event == 'closed':
                        self.start_button.configure(state='normal')
                        self.stop_button.configure(state='disabled')
            except queue.Empty:
                pass
            if not self.worker.stopping.is_set():
                try:
                    self.latest_sequence, self.latest_pair = self.worker.frames.get_nowait()
                    self.last_preview_at = time.monotonic()
                    self.draw_images()
                except queue.Empty:
                    pass
        try:
            while True:
                self._adopt_result(self.results.get_nowait())
        except queue.Empty:
            pass
        running = self.camera_running and not self.worker.stopping.is_set()
        if (running and self.region and self.latest_pair is not None and self.drag_start is None
                and time.monotonic()-self.last_preview_at < 2.
                and not (self.analysis_thread and self.analysis_thread.is_alive())
                and time.monotonic() >= self.next_measurement_at):
            self._measure(self.latest_pair, self.latest_sequence)
        self._render_state()
        self.poll_id = self.window.after(60, self.poll)

    def _render_state(self):
        snapshot = self.state.snapshot()
        if self.worker and time.monotonic()-self.last_preview_at > 2.:
            self.state.reset()
            snapshot = self.state.snapshot()
        if self._paint_valid != bool(snapshot.get('valid')):
            self._paint_valid = bool(snapshot.get('valid'))
            self.draw_images()
        pose = snapshot.get('pose')
        busy = bool(self.analysis_thread and self.analysis_thread.is_alive())
        green = snapshot['green'] and not busy
        self.angle_label.configure(foreground='#166534' if green else '#7A5600')
        self.angle.set(f"{pose['off_normal_deg']:.2f}°" if pose else '—')
        self.axis_label.set('左镜头偏离地面法线（目标 0°）' if not pose or pose.get('physical_axes_known')
                            else '矫正光轴参考（不能确认实际镜头垂直）')
        detail = STATE_TEXT.get(snapshot['status'], '本次地面检查未通过：'+snapshot['status'])
        if busy:
            detail = '正在计算新采样；下方数值仅属于最近测量图。' if pose else '正在计算地面，请保持相机和裸地可见。'
        if pose:
            right = pose.get('right_off_normal_deg')
            detail += f" · 右镜头 {right:.2f}°" if right is not None else ' · 缺少实际镜头转轴配方'
            detail += f" · 稳定 {snapshot['stable_count']}/{snapshot['required_samples']}"
        self.pose_detail.set(detail)
        if self.sample_result:
            age = snapshot.get('age_s')
            age_text = f'{age:.1f} 秒前' if age is not None else '已失效'
            stamp = datetime.fromisoformat(self.sample_result['captured_at']).astimezone().strftime('%H:%M:%S')
            self.sample_caption.set(f"采样 {stamp} · 帧 {self.sample_result['sequence']} · {age_text}")
        else:
            self.sample_caption.set('最近测量证据 · 尚无有效结果')
        self._draw_direction(pose, green)

    def _draw_direction(self, pose, green):
        canvas = self.direction
        canvas.delete('all')
        x, y, radius = 105, 65, 43
        canvas.create_oval(x-radius, y-radius, x+radius, y+radius, outline='#64748B')
        canvas.create_line(x-radius, y, x+radius, y, fill='#94A3B8')
        canvas.create_line(x, y-radius, x, y+radius, fill='#94A3B8')
        canvas.create_text(x, 10, text='朝箭头方向调光轴 · 中心为 0°', fill='#26374A')
        canvas.create_text(x+radius+12, y, text='右', fill='#26374A')
        canvas.create_text(x, y+radius+9, text='下', fill='#26374A')
        if pose:
            right, down = pose['tilt_right_deg'], pose['tilt_down_deg']
            scale = radius/max(10., math.hypot(right, down))
            color = '#166534' if green else '#7A5600'
            canvas.create_line(x, y, x+right*scale, y+down*scale, arrow='last', width=3, fill=color)
            canvas.create_oval(x+right*scale-3, y+down*scale-3, x+right*scale+3, y+down*scale+3, fill=color, outline=color)
            ref = '原始左镜头' if pose.get('physical_axes_known') else '矫正图参考'
            canvas.create_text(x, 135, text=f'{ref}：右 {right:+.1f}° / 下 {down:+.1f}°', fill='#26374A')
        else:
            canvas.create_text(x, y, text='等待有效地面', fill='#26374A')

    def draw_images(self):
        if not hasattr(self, 'canvases') or self.closed:
            return
        valid_sample = self.sample_result and self.state.snapshot().get('valid')
        for role, canvas in self.canvases.items():
            canvas.delete('all')
            pair = self.latest_pair if role == 'preview' else self.sample_pair
            if pair is None:
                canvas.create_text(max(100, canvas.winfo_width())/2, max(80, canvas.winfo_height())/2,
                                   text='等待相机画面' if role == 'preview' else '尚无采样证据', fill='white')
                continue
            image = pair.left
            cw, ch = max(80, canvas.winfo_width()), max(80, canvas.winfo_height())
            scale = min(cw/image.shape[1], ch/image.shape[0])
            width, height = max(1, round(image.shape[1]*scale)), max(1, round(image.shape[0]*scale))
            ox, oy = (cw-width)/2, (ch-height)/2
            self.transforms[role] = (scale, ox, oy)
            rendered = cv2.resize(image, (width, height))
            if role == 'sample' and valid_sample:
                rendered = paint_support(rendered, self.sample_result['ground_check'],
                    transform=(scale, 0, 0), image_size=image.shape[1::-1])
            ok, payload = cv2.imencode('.png', rendered)
            if ok:
                self.photos[role] = self.app.tk.PhotoImage(data=base64.b64encode(payload).decode('ascii'), format='png')
                canvas.create_image(ox, oy, image=self.photos[role], anchor='nw')
            if role == 'preview' and self.region:
                x, y, w, h = self.region['rect_px']
                canvas.create_rectangle(ox+x*scale, oy+y*scale, ox+(x+w)*scale, oy+(y+h)*scale,
                                        outline='#E7A9FF', width=2, tags='ground-region')
            if role == 'preview' and self.drag_start and self.drag_end:
                canvas.create_rectangle(*self.drag_start, *self.drag_end, outline='#E7A9FF', width=2, tags='drag')
            if role == 'sample' and self.sample_result:
                check = self.sample_result.get('ground_check') or {}
                selection = check.get('selection')
                if selection:
                    x, y, w, h = selection['rect_px']
                    canvas.create_rectangle(ox+x*scale, oy+y*scale, ox+(x+w)*scale, oy+(y+h)*scale,
                                            outline='#E7A9FF', width=2, tags='sample-ground-region')
                if valid_sample:
                    count = (check.get('observed_support') or {}).get('pixel_count', 0)
                    label = canvas.create_text(10, 10, anchor='nw', text=f'地面检查通过 · 实测支持 {count:,} 像素',
                                               fill='#BFE9CA', tags='sample-support-count')
                    bounds = canvas.bbox(label)
                    background = canvas.create_rectangle(bounds[0]-4, bounds[1]-3, bounds[2]+4, bounds[3]+3,
                                                         fill='#152235', outline='')
                    canvas.tag_lower(background, label)
            if role == 'sample' and not valid_sample:
                canvas.create_text(10, 10, anchor='nw', text='本次无效或已过期 · 不显示支持涂色', fill='#FFD166')

    def close(self):
        if self.closed:
            return
        self.stop()
        self.closed = True
        if self.poll_id:
            self.window.after_cancel(self.poll_id)
        self.window.destroy()
        if not getattr(self.owner, 'closed', False):
            self.owner.message.set('实时调姿已结束；若机位有调整，请重新双目抓拍，重画地面与管段范围后再评估。')
