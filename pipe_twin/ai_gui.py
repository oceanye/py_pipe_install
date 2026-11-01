"""Independent AI task lifecycle and Kimi settings for the field GUI.

Workers only publish messages. The Tk thread checks generation and cancellation
before accepting or saving a provider result. No credentials enter scene files.
"""
from __future__ import annotations

import hashlib
import copy
import json
import threading
from datetime import datetime
from pathlib import Path


def report_matches_scene(path, report):
    """Check the production A report's source hashes before sharing a scene."""
    if not isinstance(report, dict):
        return False
    path = Path(path)
    raw = path.read_bytes()
    manifest = json.loads(raw)
    expected = {'manifest': hashlib.sha256(raw).hexdigest(), 'model': manifest['model']['sha256']}
    for group in manifest['capture']['capture_groups']:
        for role, view in group['views'].items():
            expected[f"{group['capture_id']}_{role}"] = view['sha256']
        for index, frame in enumerate(group.get('temporal_frames', [])):
            for role, view in frame['views'].items():
                expected[f"{group['capture_id']}_burst_{index}_{role}"] = view['sha256']
    actual = report.get('inputs') or {}
    return isinstance(actual, dict) and all(actual.get(key) == value for key, value in expected.items())


def load_bound_local_report(path):
    """Only source-bound A reports may be restored alongside an AI sidecar."""
    report_path = Path(path).with_name('report.json')
    if not report_path.is_file():
        return None
    if report_path.stat().st_size > 32 * 1024 * 1024:
        return None
    try:
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if report_matches_scene(path, report):
            report.pop('ai_section_assist', None)
            return apply_current_identity_policy(path, report)
    except (ValueError, KeyError, TypeError):
        pass
    return None


def apply_current_identity_policy(path, report):
    """Old archives must not revive identities forbidden by today's burst gate."""
    from .installation_assist import _invalid_temporal_sequence, analyze_installation_assist
    result = copy.deepcopy(report)
    if _invalid_temporal_sequence(result):
        manifest = json.loads(Path(path).read_text(encoding='utf-8'))
        settings = manifest['analysis']['elevation_auto']
        result['installation_assist'] = analyze_installation_assist(
            result, settings['pipes'], settings.get('registration'))
    return result


class AISectionController:
    def __init__(self, owner):
        self.owner = owner
        self.active = None
        self.settings_window = None

    def configured(self):
        from .ai_settings import configured_status
        state = configured_status()
        if not state['configured']:
            self.owner.message.set('请先打开 Kimi 设置并填写 API key，再运行 AI 断面评估。')
            self.open_settings()
            return False
        return True

    def start(self, arguments, generation):
        from .ai_settings import load_ai_settings
        settings = load_ai_settings()
        event = threading.Event()
        self.active = (generation, event)
        self.owner.ai_cancel_button.configure(state='normal')
        self.owner.message.set('正在核对当前现场并请求 Kimi；结果只用于断面候选判断。')
        prior_path = self.owner.last_manifest
        prior_report = copy.deepcopy(self.owner.report)
        threading.Thread(target=self._worker, args=(arguments, generation, event, settings, prior_path, prior_report), daemon=True).start()

    @staticmethod
    def _reusable_scene(path, arguments):
        """Reuse only a complete, validated snapshot of the current GUI input."""
        if path is None:
            return False
        from .elevation_dataset import elevation_history_compatible, load_elevation_dataset
        names = ('calibration', 'pipe_specs', 'model_path', 'stl_unit', 'rectification_recipe',
                 'mode', 'registration_settings', 'analysis_settings', 'scope_settings')
        if not elevation_history_compatible(path, **{name: arguments.get(name) for name in names}):
            return False
        latest = load_elevation_dataset(path)['manifest']['capture']['capture_groups'][-1]

        def same_view(view, item, role, expected_hash=None):
            data_hash = hashlib.sha256(Path(item[role+'_path']).read_bytes()).hexdigest()
            if data_hash != view['sha256'] or (expected_hash is not None and data_hash != expected_hash):
                return False
            if datetime.fromisoformat(item[role+'_time']) != datetime.fromisoformat(view['captured_at']):
                return False
            source = (item.get('timestamp_sources') or {}).get(role)
            if source is not None and source != view.get('timestamp_source'):
                return False
            provenance = (item.get('camera_capture_provenance') or {}).get(role) or {}
            old_provenance = {k: v for k, v in view.items() if k.startswith('capture_') or k == 'side_by_side_order'}
            return provenance == old_provenance

        for role in ('left', 'right'):
            if not same_view(latest['views'][role], arguments, role, arguments['_preview_hashes'][role]):
                return False
        old_frames, frames = latest.get('temporal_frames', []), arguments.get('temporal_frames', [])
        if len(old_frames) != len(frames):
            return False
        for old, current in zip(old_frames, frames):
            for role in ('left', 'right'):
                if not same_view(old['views'][role], current, role, (current.get('sha256') or {}).get(role)):
                    return False
        return True

    def _worker(self, arguments, generation, event, settings, prior_path=None, prior_report=None):
        from .elevation_dataset import create_elevation_dataset, load_elevation_dataset
        from .ai_workflow import evaluate_ai_scene
        path = None
        try:
            arguments = dict(arguments)
            if event.is_set():
                return
            reuse = self._reusable_scene(prior_path, arguments)
            expected = arguments.pop('_preview_hashes')
            if reuse:
                path = prior_path
                base_report = apply_current_identity_policy(path, prior_report) if report_matches_scene(path, prior_report) else load_bound_local_report(path)
            else:
                arguments['previous_manifest'] = None
                path, base_report = create_elevation_dataset(**arguments), None
            latest = load_elevation_dataset(path)['manifest']['capture']['capture_groups'][-1]['views']
            if any(latest[role]['sha256'] != expected[role] for role in ('left', 'right')):
                raise ValueError('保存期间照片发生变化，请重新载入。')
            if event.is_set():
                return
            self.owner.messages.put((generation, 'ai_snapshot', (event, path, reuse, base_report)))
            result = evaluate_ai_scene(path, report=base_report, settings=settings, cancel_event=event,
                progress=lambda text: self.owner.messages.put((generation, 'ai_progress', (event, text))))
            self.owner.messages.put((generation, 'ai_ok', (event, path, result, base_report)))
        except Exception as error:
            self.owner.messages.put((generation, 'ai_error', (event, path, str(error))))

    def cancel(self, *, notify=True):
        if self.active is None:
            return
        self.active[1].set()
        self.active = None
        owner = self.owner
        owner.busy = False
        owner._set_controls(True)
        owner.ai_cancel_button.configure(state='disabled')
        if notify and not owner.closed:
            owner.message.set('AI 评估已取消；迟到响应不会保存或覆盖现场。')

    def handle(self, generation, kind, payload):
        owner = self.owner
        event = payload[0]
        if self.active != (generation, event) or event.is_set() or owner.closed:
            return
        if generation != owner.generation:
            self.cancel(notify=False)
            owner.message.set('输入已改变，旧 AI 结果已丢弃；请重新评估。')
            return
        if kind == 'ai_progress':
            owner.message.set(str(payload[1]))
            return
        if kind == 'ai_snapshot':
            if payload[2] and payload[3] is not None:
                owner.report = copy.deepcopy(payload[3])
                owner.results = {r['pipe_id']: r for r in owner.report.get('pipes', [])}
                owner.refresh_table()
            elif not payload[2]:
                owner.report, owner.results = None, {}
                owner.refresh_table()
                owner.message.set('输入已形成新现场；正在独立执行 AI 断面评估。')
            return
        self.active = None
        owner.busy = False
        owner._set_controls(True)
        owner.ai_cancel_button.configure(state='disabled')
        _, path, value, *extra = payload
        if kind == 'ai_error':
            suffix = f'\n输入现场已保存：{path}' if path else ''
            owner.show_error(ValueError(str(value) + suffix))
            return
        try:
            # Files can change independently of Tk traces. Recheck both preview
            # files and the immutable package before the atomic sidecar save.
            for role in ('left', 'right'):
                source = Path(owner.fields[role].get()).resolve()
                if str(source) != owner.image_paths.get(role) or hashlib.sha256(source.read_bytes()).hexdigest() != owner.image_hashes.get(role):
                    raise ValueError('当前照片已改变，AI 结果未保存，请重新载入。')
            from .ai_workflow import accept_ai_result
            base_report = extra[0] if extra else None
            report = accept_ai_result(path, value, report=base_report, save=True)
            owner.last_manifest = Path(path)
            owner.report = report
            owner.results = {r['pipe_id']: r for r in report.get('pipes', [])}
            owner.refresh_table()
            owner.status_tabs.select(owner.ai_section_status_tab)
            if owner.model_viewer is not None and owner.model_viewer.window.winfo_exists():
                owner.model_viewer.sync_from_owner()
            data = report.get('ai_section_assist') or {}
            from .ai_section_display import STATUS_LABELS
            status = STATUS_LABELS.get(data.get('status'), data.get('status', '已完成'))
            owner.summary.set('AI 与本地断面结果已并列保存，严格安装状态保持原值。' if base_report is not None
                              else 'AI 独立断面判断；本现场未运行严格安装验收。')
            owner.message.set(f'Kimi 评估完成：{status}。结果已保存；点击“查看 AI 断面图”查看模型管位分布，打开现场可恢复。')
            from .artifact_io import atomic_write_text
            atomic_write_text(owner.output_root / 'last_scene.json', json.dumps({'manifest_path': str(path)}, ensure_ascii=False))
        except (OSError, ValueError) as error:
            owner.show_error(error)

    def open_settings(self):
        owner = self.owner
        if owner.busy:
            return
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.lift()
            return
        from .ai_settings import load_ai_settings, save_ai_settings, configured_status, AISettings
        try:
            settings = load_ai_settings()
            state = configured_status()
        except ValueError as error:
            owner.show_error(error)
            return
        tk, ttk = owner.app.tk, owner.app.ttk
        window = self.settings_window = tk.Toplevel(owner.window)
        window.title('Kimi 设置')
        window.transient(owner.window)
        window.resizable(False, False)
        panel = ttk.Frame(window, padding=16)
        panel.pack(fill='both', expand=True)
        model = tk.StringVar(value=settings.model)
        base = tk.StringVar(value=settings.base_url)
        timeout = tk.StringVar(value=str(settings.timeout_seconds))
        key = tk.StringVar(value='')
        hint = tk.StringVar(value='已配置凭据（' + str(state.get('credential_source')) + '）' if state['configured'] else '尚未配置凭据')
        ttk.Label(panel, text='Kimi 用于图片与模型断面候选对照，独立于本地双目算法。').grid(row=0, column=0, columnspan=2, sticky='w', pady=(0, 10))
        for row, label, variable in ((1, '模型', model), (3, '请求超时（秒）', timeout), (4, 'API key', key)):
            ttk.Label(panel, text=label).grid(row=row, column=0, sticky='w', pady=5)
            entry = ttk.Entry(panel, textvariable=variable, width=44, show='*' if row == 4 else '')
            entry.grid(row=row, column=1, sticky='ew', padx=(12, 0), pady=5)
        ttk.Label(panel, text='服务地址').grid(row=2, column=0, sticky='w')
        ttk.Combobox(panel, textvariable=base, values=('https://api.moonshot.cn/v1', 'https://api.moonshot.ai/v1'), state='readonly', width=42).grid(row=2, column=1, padx=(12, 0), pady=5)
        ttk.Label(panel, textvariable=hint, wraplength=470, foreground='#355371').grid(row=5, column=0, columnspan=2, sticky='w', pady=8)
        ttk.Label(panel, text='密钥留空保留现有配置。Windows 加密保存在当前用户目录；\n环境变量 MOONSHOT_API_KEY 优先。密钥不随现场或 Git 同步。', wraplength=470).grid(row=6, column=0, columnspan=2, sticky='w')

        def save():
            if owner.busy:
                hint.set('请先等待或取消当前评估。')
                return
            try:
                candidate = AISettings(base_url=base.get(), model=model.get(), timeout_seconds=float(timeout.get()),
                                       max_tokens=settings.max_tokens, reasoning_effort=settings.reasoning_effort)
                save_ai_settings(candidate, api_key=key.get().strip() or None)
                key.set('')
                hint.set('设置已保存。可返回主窗口点击“AI 断面评估”。')
                owner.message.set('Kimi 设置已保存；可点击 AI 断面评估。')
            except (OSError, ValueError) as error:
                hint.set(str(error))

        ttk.Button(panel, text='保存设置', command=save).grid(row=7, column=1, sticky='e', pady=(12, 0))

    def close(self):
        self.cancel(notify=False)
        if self.settings_window is not None and self.settings_window.winfo_exists():
            self.settings_window.destroy()
