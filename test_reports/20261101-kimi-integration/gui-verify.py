"""Read-only real-Tk acceptance of an archived real Kimi response.

Run after pytest's Tk checks finish. No network, camera, credential reads,
provider requests or scene writes are allowed in this verification.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import argparse
import hashlib
import json
import runpy
import sys
import traceback
import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.elevation_gui import ElevationCaptureDialog

HELPERS = runpy.run_path(str(ROOT / 'test_reports/20261030-function-review/gui-verify.py'))
screenshot = runpy.run_path(str(ROOT / 'test_reports/20261030-ground-monitor/gui-verify.py'))['screenshot']
OUT = ROOT / 'outputs/kimi-integration-20261101/gui'
DEFAULT = ROOT / 'outputs/kimi-integration-20261101/live-20261101-105332/scene/manifest.json'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected_inputs(source):
    paths = {source, source.with_name('report.json'), source.with_name('ai_section_result.json'),
             ROOT / 'outputs/elevation_workbench/last_scene.json',
             ROOT / 'outputs/measurement_workbench/workbench_profile.json',
             ROOT / 'outputs/measurement_workbench/calibration_current.json'}
    manifest = read(source)
    paths.add(source.parent / manifest['model']['path'])
    for group in manifest['capture']['capture_groups']:
        for record in (group, *group.get('temporal_frames', [])):
            paths.update(source.parent / v['path'] for v in record['views'].values())
    return {str(p.resolve().relative_to(ROOT)): sha(p) for p in sorted(paths) if p.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=Path, default=DEFAULT)
    args = parser.parse_args()
    source = args.scene.resolve()
    before = protected_inputs(source)
    saved = read(source.with_name('ai_section_result.json'))
    original_a = read(source.with_name('report.json'))
    evidence = {'scope': 'REAL_TK_REOPEN_OF_ARCHIVED_REAL_KIMI_RESPONSE',
                'source_scene': str(source), 'new_capture': False, 'external_ai_called_by_this_script': False,
                'camera_opened': False, 'credentials_read': False,
                'checks': [], 'screens': {}, 'callback_errors': []}

    def check(name, passed, **detail):
        evidence['checks'].append({'name': name, 'passed': bool(passed), **detail})

    OUT.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    root.withdraw()
    root.report_callback_exception = lambda *exc: evidence['callback_errors'].append(''.join(traceback.format_exception(*exc)))
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', font=('Microsoft YaHei UI', 10))
    style.configure('Treeview', rowheight=25)
    owner = None
    app = SimpleNamespace(root=root, tk=tk, ttk=ttk, messagebox=Mock(), filedialog=Mock())
    try:
        with patch('pipe_twin.workbench_profile.load_profile', return_value=(None, None)), \
             patch('pipe_twin.stereo_camera.StereoCameraSession.open', side_effect=AssertionError('Camera forbidden')), \
             patch('pipe_twin.ai_provider.request_section_assessment', side_effect=AssertionError('Network forbidden')):
            owner = ElevationCaptureDialog(app, source, output_root=OUT / 'isolated', restore=False)
            owner.window.title('Kimi 断面实窗验收 · 已保存真实响应 · 本次未发请求')
            owner.status_tabs.select(owner.ai_section_status_tab)
            root.update()
            check('saved_real_ai_namespace_restored', owner.report['ai_section_assist'] == saved['ai_section_assist'])
            check('strict_a_states_restored_unchanged', owner.report['pipes'] == original_a['pipes'] and
                  owner.report['counts'] == original_a['counts'])
            check('old_invalid_burst_identities_not_revived',
                  owner.report['installation_assist']['counts']['LIKELY'] == 0 and
                  owner.report['installation_assist']['reason_codes'] == ['TEMPORAL_SEQUENCE_INVALID_FOR_IDENTITY'])
            states = [row['status'] for row in owner.ai_section_rows.values()]
            check('twelve_real_ambiguous_points', len(states) == 12 and all(state == 'AMBIGUOUS' for state in states), states=states)
            check('ai_tab_no_local_img_proposals', all(not view.canvas.find_withtag('image-candidate') for view in owner.views.values()))
            check('ai_control_line_visible', bool(owner.views['left'].canvas.find_withtag('ai-control-section')))
            check('ai_run_after_preparation', owner.ai_run_button.winfo_rooty() == owner.run_button.winfo_rooty() and
                  owner.ai_run_button.winfo_rooty() > owner.region_button.winfo_rooty())

            for size in ('1024x720', '1280x860'):
                owner.window.geometry(size + '+10+10')
                owner.window.deiconify()
                root.update()
                clipped = HELPERS['clipped_controls'](owner.window)
                height = owner.views['left'].canvas.winfo_height()
                evidence['screens'][size] = {'clipped_controls': clipped, 'photo_height': height}
                check('layout_' + size, not clipped and height >= 180, photo_height=height, clipped_controls=clipped)
                screenshot(owner.window, OUT / ('real-kimi-' + size + '.png'))

            owner.show_analysis_details('ai_section')
            root.update()
            detail = owner.detail_texts['ai_section'].get('1.0', 'end')
            check('provider_and_capture_provenance_visible', 'kimi-k3.1-pro' in detail and
                  '结果生成时间' in detail and '实拍时间' in detail and '仅当前参考双目对' in detail and
                  '不是重新采集' in detail and '图像控制断面' in detail, detail_text=detail)
            screenshot(owner.analysis_details_window, OUT / 'real-kimi-details.png')
            owner.analysis_details_window.withdraw()
            owner.open_model_viewer()
            root.update()
            viewer = owner.model_viewer
            model_states = dict(viewer.installation_states)
            row_key = list(owner.ai_section_rows)[-1]
            expected_pipe = owner.ai_section_rows[row_key]['pipe_id']
            owner.ai_section_tree.selection_set(row_key)
            owner._ai_section_selected()
            root.update()
            check('ai_selection_only_navigates_model', owner.selected_id() == expected_pipe and
                  dict(viewer.installation_states) == model_states, selected_pipe=owner.selected_id())
            screenshot(viewer.window, OUT / 'real-kimi-model-selection.png')
            owner.invalidate()
            root.update()
            check('input_change_clears_ai_and_a', owner.report is None and
                  [r['status'] for r in owner.ai_section_rows.values()] == ['NOT_RUN'])
            check('no_messagebox_errors', not app.messagebox.showerror.called)
    except Exception:
        evidence['fatal_error'] = traceback.format_exc()
    finally:
        if owner is not None:
            owner.close()
        root.destroy()
        after = {name: sha(ROOT / name) for name in before}
        check('protected_sources_unchanged', before == after)
        check('no_tk_callback_errors', not evidence['callback_errors'])
        evidence.update(source_sha256=before, sources_unchanged=before == after,
                        passed=not evidence.get('fatal_error') and all(c['passed'] for c in evidence['checks']))
        Path(__file__).with_name('gui-evidence.json').write_text(
            json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'passed': evidence['passed'], 'checks': len(evidence['checks']),
                      'failed': [c for c in evidence['checks'] if not c['passed']],
                      'fatal_error': evidence.get('fatal_error')}, ensure_ascii=False, indent=2))
    return 0 if evidence['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
