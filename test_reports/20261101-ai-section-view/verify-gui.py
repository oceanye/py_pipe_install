"""Read-only real-window verification; never capture or request AI."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import hashlib
import json
import runpy
import sys
import traceback
import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/ai-section-view-20261101'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
from pipe_twin.elevation_gui import ElevationCaptureDialog

screenshot = runpy.run_path(str(ROOT / 'test_reports/20261030-ground-monitor/gui-verify.py'))['screenshot']
clipped_controls = runpy.run_path(str(ROOT / 'test_reports/20261030-function-review/gui-verify.py'))['clipped_controls']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def widgets(widget):
    yield widget
    for child in widget.winfo_children():
        yield from widgets(child)


def styles(canvas):
    result = {}
    for item in canvas.find_withtag('ai-section-pipe'):
        tags = canvas.gettags(item)
        pipe = next(tag[5:] for tag in tags if tag.startswith('pipe:'))
        result[pipe] = {key: canvas.itemcget(item, key) for key in ('fill', 'outline', 'dash', 'width')}
        result[pipe]['status'] = next(tag[9:] for tag in tags if tag.startswith('ai-state:'))
    return result


def main():
    pointer = ROOT / 'outputs/elevation_workbench/last_scene.json'
    source = Path(read(pointer)['manifest_path']).resolve()
    protected = set(source.parent.rglob('*')) | {
        pointer, ROOT / 'outputs/measurement_workbench/workbench_profile.json',
        ROOT / 'outputs/measurement_workbench/calibration_current.json',
    }
    before = {str(p): sha(p) for p in sorted(protected) if p.is_file()}
    saved = read(source.with_name('ai_section_result.json'))['ai_section_assist']
    evidence = {'scope': 'READ_ONLY_REAL_TK_AI_SECTION_PLOT', 'source': str(source),
                'started_at': datetime.now(timezone.utc).isoformat(),
                'new_camera_capture': False, 'new_provider_calls': 0, 'checks': [],
                'screens': {}, 'callback_errors': []}

    def check(name, passed, **detail):
        evidence['checks'].append({'name': name, 'passed': bool(passed), **detail})

    root = tk.Tk()
    root.withdraw()
    root.report_callback_exception = lambda *exc: evidence['callback_errors'].append(''.join(traceback.format_exception(*exc)))
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', font=('Microsoft YaHei UI', 10))
    style.configure('Treeview', rowheight=25)
    app = SimpleNamespace(root=root, tk=tk, ttk=ttk, messagebox=Mock(), filedialog=Mock())
    owner = None
    try:
        with patch('pipe_twin.workbench_profile.load_profile', return_value=(None, None)), \
             patch('pipe_twin.stereo_camera.StereoCameraSession.open', side_effect=AssertionError('Camera forbidden')), \
             patch('pipe_twin.ai_provider.request_section_assessment', side_effect=AssertionError('Network forbidden')):
            owner = ElevationCaptureDialog(app, source, output_root=OUT / 'isolated', restore=False)
            owner.window.title('AI 断面只读验收 · 已保存真实结果')
            owner.window.geometry('1280x860+20+20')
            owner.status_tabs.select(owner.ai_section_status_tab)
            root.update()
            owner.ai_section_view_button.invoke()
            root.update()
            viewer = owner.model_viewer
            original_report = json.dumps(owner.report, sort_keys=True, ensure_ascii=False)
            expected = {row['pipe_id']: row['section_status'] for row in saved['candidates']}
            actual = {pid: row['status'] for pid, row in viewer.ai_plot['by_pipe'].items()}
            check('current_real_ai_result_loaded', owner.report['ai_section_assist'] == saved,
                  run_status=saved['run_status'], states=dict(Counter(expected.values())))
            check('ai_plot_matches_saved_states', actual == expected, states=actual)
            check('show_ai_section_button_selects_ai', viewer.section_result_mode.get() == 'AI' and owner.ai_display_active())
            check('all_cad_pipes_have_ai_circles', len(viewer.section_canvas.find_withtag('ai-section-pipe')) == len(owner.pipes))
            check('all_ai_states_have_text_labels', len(viewer.section_canvas.find_withtag('ai-state-label')) == len(owner.pipes))
            check('no_local_installation_legend_in_ai_section', not viewer.section_canvas.find_withtag('installation-legend'))
            check('nominal_geometry_scope_visible', 'CAD' in viewer.section_canvas.itemcget(viewer.section_canvas.find_withtag('ai-section-caption')[0], 'text'))
            initial_styles = styles(viewer.section_canvas)

            selected_key = list(owner.ai_section_rows)[-1]
            selected_pipe = owner.ai_section_rows[selected_key]['pipe_id']
            owner.ai_section_tree.selection_set(selected_key)
            owner.ai_section_tree.see(selected_key)
            root.update()
            ring = viewer.section_canvas.find_withtag('ai-selection-ring')
            check('table_selection_highlights_section', owner.selected_id() == selected_pipe and len(ring) == 1 and
                  f'pipe:{selected_pipe}' in viewer.section_canvas.gettags(ring[0]))
            check('selection_keeps_ai_status_style', styles(viewer.section_canvas) == initial_styles)

            hit = next(item for item in viewer.section_hits if item[0] != selected_pipe)
            viewer.section_canvas.event_generate('<Button-1>', x=round(hit[1]), y=round(hit[2]))
            root.update()
            check('section_click_selects_ai_table', owner.selected_id() == hit[0] and owner.selected_ai_section_row()['pipe_id'] == hit[0],
                  selected_pipe=owner.selected_id())
            check('selected_info_names_ai_and_nominal_diameter', 'CAD 名义外径' in viewer.selection_info.get() and 'AI' in viewer.selection_info.get())

            for size in ('1000x660', '1320x820'):
                viewer.window.geometry(size + '+40+30')
                viewer.window.deiconify()
                root.update()
                viewer.draw()
                root.update()
                canvas = viewer.section_canvas
                clipped = clipped_controls(viewer.window)
                required = {'应用方向', '反向', '应用相机观察侧', '恢复已应用方向'}
                mapped_buttons = {w.cget('text') for w in widgets(viewer.window)
                                  if isinstance(w, ttk.Button) and w.winfo_ismapped()}
                missing_buttons = sorted(required - mapped_buttons)
                outside = []
                for item in (*canvas.find_withtag('ai-state-label'), *canvas.find_withtag('ai-section-pipe'),
                             *canvas.find_withtag('ai-section-legend'), *canvas.find_withtag('ai-section-caption')):
                    box = canvas.bbox(item)
                    if box and (box[0] < 0 or box[1] < 0 or box[2] > canvas.winfo_width() or box[3] > canvas.winfo_height()):
                        outside.append({'tags': canvas.gettags(item), 'bbox': box})
                legend_box = canvas.bbox(canvas.find_withtag('ai-section-legend')[0])
                legend_collisions = []
                for item in canvas.find_all():
                    if canvas.type(item) != 'text' or not canvas.itemcget(item, 'text').startswith('+'):
                        continue
                    box = canvas.bbox(item)
                    if box and max(box[0], legend_box[0]) < min(box[2], legend_box[2]) and max(box[1], legend_box[1]) < min(box[3], legend_box[3]):
                        legend_collisions.append(canvas.itemcget(item, 'text'))
                evidence['screens'][size] = {'canvas_width': canvas.winfo_width(), 'canvas_height': canvas.winfo_height(),
                                            'clipped_controls': clipped, 'outside_canvas': outside,
                                            'unmapped_required_buttons': missing_buttons, 'legend_axis_collisions': legend_collisions}
                check('layout_' + size, not clipped and not outside and not missing_buttons and not legend_collisions,
                      clipped_controls=clipped, outside_canvas=outside,
                      unmapped_required_buttons=missing_buttons, legend_axis_collisions=legend_collisions)
                screenshot(viewer.window, OUT / ('ai-section-' + size + '.png'))
            screenshot(owner.window, OUT / 'ai-list-and-entry.png')

            controls = {w.cget('text'): w for w in widgets(viewer.window) if isinstance(w, ttk.Radiobutton)}
            controls['本地结果'].invoke()
            root.update()
            check('local_switch_restores_local_mode_and_list', viewer.section_result_mode.get() == 'LOCAL' and
                  owner.status_tabs.select() == str(owner.assist_status_tab) and
                  not viewer.section_canvas.find_withtag('ai-section-pipe'))
            controls['AI 候选'].invoke()
            root.update()
            check('ai_switch_restores_real_results', owner.ai_display_active() and styles(viewer.section_canvas) == initial_styles)
            check('selection_and_mode_switch_do_not_mutate_report', original_report == json.dumps(owner.report, sort_keys=True, ensure_ascii=False))
            check('no_messagebox_errors', not app.messagebox.showerror.called)
    except Exception:
        evidence['fatal_error'] = traceback.format_exc()
    finally:
        if owner is not None:
            owner.close()
        root.destroy()
        after = {path: sha(path) for path in before}
        check('protected_scene_photos_report_and_defaults_unchanged', before == after, protected_file_count=len(before))
        check('no_tk_callback_errors', not evidence['callback_errors'])
        evidence.update(source_sha256=before, sources_unchanged=before == after,
                        finished_at=datetime.now(timezone.utc).isoformat(),
                        passed=not evidence.get('fatal_error') and all(row['passed'] for row in evidence['checks']))
        (OUT / 'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'passed': evidence['passed'], 'checks': len(evidence['checks']),
                      'failed': [row for row in evidence['checks'] if not row['passed']],
                      'fatal_error': evidence.get('fatal_error')}, ensure_ascii=False, indent=2))
    return 0 if evidence['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
