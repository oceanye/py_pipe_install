"""Audit real Tk geometry against the previous GUI and replay the field workflow."""
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
import hashlib
import json
import subprocess
import sys
import time
import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.gui import _start_application

SOURCE = ROOT/'outputs/ground-trial-20261026/gui/captures/elevation-20261026-133314-f4a6779d/manifest.json'
OUTPUT = ROOT/'outputs/gui-audit-20261027'


def widgets(parent):
    for child in parent.winfo_children():
        if isinstance(child, tk.Toplevel):
            continue
        yield child
        yield from widgets(child)


def layout(window):
    buttons, clipped = [], []
    for w in widgets(window):
        if not w.winfo_ismapped():
            continue
        cls = w.winfo_class()
        if cls not in ('TButton', 'TMenubutton', 'TCombobox', 'TEntry', 'TCheckbutton'):
            continue
        label = str(w.cget('text')) if 'text' in w.keys() else cls
        if cls in ('TButton', 'TMenubutton'):
            buttons.append(label)
        x, y = w.winfo_rootx()-window.winfo_rootx(), w.winfo_rooty()-window.winfo_rooty()
        if x < 0 or y < 0 or x+w.winfo_width() > window.winfo_width()+1 or y+w.winfo_height() > window.winfo_height()+1:
            clipped.append({'label': label, 'box': [x, y, w.winfo_width(), w.winfo_height()]})
    return {'buttons': buttons, 'button_count': len(buttons), 'clipped_controls': clipped}


def without_capture_ids(value):
    """A replay has new capture IDs; preserve every numerical/decision field."""
    if isinstance(value, dict):
        return {k: without_capture_ids(v) for k, v in value.items() if k != 'capture_id'}
    if isinstance(value, list):
        return [without_capture_ids(v) for v in value]
    return value


def compare_results(actual, source):
    checked = ['section_evaluation', 'ground_check', 'counts', 'position_assessment']
    for key in checked:
        assert without_capture_ids(actual[key]) == without_capture_ids(source[key]), key
    return checked


def audit(*, old=False, evaluate=False):
    root = tk.Tk(); root.withdraw()
    report = json.loads(SOURCE.with_name('report.json').read_text(encoding='utf8'))
    with patch('pipe_twin.workbench_profile.load_profile', return_value=(None, None)):
        if old:
            source = subprocess.run(['git', 'show', '3358460:pipe_twin/elevation_gui.py'], cwd=ROOT,
                check=True, capture_output=True).stdout.decode('utf8')
            module = ModuleType('pipe_twin.elevation_gui_before')
            module.__file__ = str(ROOT/'pipe_twin/elevation_gui.py')
            exec(compile(source, module.__file__, 'exec'), module.__dict__)
            style = ttk.Style(root); style.theme_use('clam')
            style.configure('.', font=('Microsoft YaHei UI', 10)); style.configure('Treeview', rowheight=28)
            app = SimpleNamespace(root=root, tk=tk, ttk=ttk, filedialog=Mock(), messagebox=Mock())
            d = module.ElevationCaptureDialog(app, SOURCE, output_root=OUTPUT/'before', restore=False)
        else:
            d = _start_application(root, SOURCE)
            d.output_root = OUTPUT/'current'
            d.app.messagebox = Mock(); d.app.filedialog = Mock()
        try:
            d.report = report; d.results = {r['pipe_id']: r for r in report['pipes']}
            d.refresh_table()
            sizes = {}
            for size in ('1024x720', '1280x860'):
                d.window.geometry(size); d.window.deiconify(); root.update()
                sizes[size] = layout(d.window) | {'left_preview_height': d.views['left'].canvas.winfo_height(),
                    'right_preview_height': d.views['right'].canvas.winfo_height()}
                if not old:
                    assert not sizes[size]['clipped_controls'], sizes[size]
                    assert d.views['left'].canvas.winfo_height() >= 180, sizes[size]
            result = {'sizes': sizes}
            if not old:
                settings = {}
                for page in d.settings_pages:
                    d.open_settings(page); root.update()
                    settings[page] = layout(d.settings_window)
                    assert not settings[page]['clipped_controls'], settings[page]
                d.hide_settings()
                d.show_analysis_details(); root.update()
                assert len(d.detail_tabs.tabs()) == 5
                d.open_model_viewer(); root.update()
                assert len(d.model_viewer.scene_hits) == 4
                d.open_quality_check(); root.update()
                d.quality_dialog.close()
                result.update(settings=settings, details_tabs=5, scene_count=4, quality_dialog_opened=True)
                dialogs = {}
                for name, opener, attr in [('camera', d.capture_camera, 'camera_dialog'),
                                            ('calibration', d.open_calibration, 'calibration_dialog')]:
                    opener(); root.update()
                    child = getattr(d, attr)
                    dialogs[name] = layout(child.window)
                    assert not dialogs[name]['clipped_controls'], dialogs[name]
                    child.close()
                result['capture_and_calibration_dialogs'] = dialogs
                if evaluate:
                    d.history.set(False); d.confirmed.set(True)
                    d.submit(True)
                    previous, deadline = '', time.monotonic()+300
                    while d.busy and time.monotonic() < deadline:
                        root.update(); time.sleep(.03)
                        if d.message.get() != previous:
                            previous = d.message.get(); print(previous, flush=True)
                    assert not d.busy, 'GUI_EVALUATION_TIMEOUT'
                    d.app.messagebox.showerror.assert_not_called()
                    assert d.report['section_evaluation']['conditional_section_count'] == 4
                    assert d.report['ground_check']['correction_status'] == 'TRIAL_APPLIED'
                    assert len(d.solver_rows) == 4
                    result['unchanged_result_fields_except_capture_ids'] = compare_results(d.report, report)
                    d.window.geometry('1024x720'); root.update()
                    result['evaluated_layout'] = layout(d.window) | {
                        'left_preview_height': d.views['left'].canvas.winfo_height()}
                    assert d.views['left'].canvas.winfo_height() >= 180, result['evaluated_layout']
                    result.update(manifest=str(d.last_manifest.relative_to(ROOT)),
                        section_count=4, installation_counts=d.report['counts'],
                        position_counts=d.report['position_assessment']['counts'],
                        trial_state=d.report['ground_check']['correction_status'],
                        temporal_frames=d.report['section_evaluation']['temporal_sections']['sampled_frames'])
                    d.load_session(d.last_manifest)
                    assert d.ground_candidate_trial.get() and len(d.temporal_frames) == 5
                    result['saved_settings_and_five_frames_restored'] = True
            return result
        finally:
            d.close()
            if old:
                root.update(); root.destroy()


def main():
    calibration = ROOT/'outputs/measurement_workbench/calibration_current.json'
    bound = [SOURCE, SOURCE.with_name('report.json'), calibration]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in bound}
    before = audit(old=True)
    after = audit(evaluate='--evaluate' in sys.argv)
    for name, digest in hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    result = {'baseline_commit': '3358460', 'source_hashes_unchanged': hashes,
              'replay_only_no_new_capture': True, 'before': before, 'after': after}
    (Path(__file__).parent/'gui-validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
