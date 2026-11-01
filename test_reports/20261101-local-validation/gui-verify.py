"""Real Tk UI validation; archived images and explicitly simulated AI only.

Run after other Tk tests finish. Never opens a camera, calls a provider, saves a
scene, or updates the default scene pointer. A displayed fixture is not evidence
that an external AI request works.
"""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import argparse
import copy
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
from pipe_twin.ai_section_assist import attach_ai_section_assist
from pipe_twin.model_section_reference import build_model_section_reference

HELPERS = runpy.run_path(str(ROOT / 'test_reports/20261030-function-review/gui-verify.py'))
screenshot = runpy.run_path(str(ROOT / 'test_reports/20261030-ground-monitor/gui-verify.py'))['screenshot']
OUT = ROOT / 'outputs/local-validation-20261101/gui'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def protected_inputs(source, report_path):
    paths = {source, report_path,
        ROOT / 'outputs/elevation_workbench/last_scene.json',
        ROOT / 'outputs/measurement_workbench/workbench_profile.json',
        ROOT / 'outputs/measurement_workbench/calibration_current.json'}
    manifest = read(source)
    model = manifest.get('model') or {}
    if model.get('path'):
        paths.add(source.parent / model['path'])
    for group in manifest['capture']['capture_groups']:
        for record in (group, *group.get('temporal_frames', [])):
            paths.update(source.parent / v['path'] for v in record['views'].values())
    return {str(p.resolve().relative_to(ROOT)): sha(p) for p in sorted(paths) if p.is_file()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    source = (args.scene or Path(read(ROOT / 'outputs/installation-assist-20261031/latest-scene.json')['manifest_path'])).resolve()
    report_path = (args.report or source.with_name('report.json')).resolve()
    before = protected_inputs(source, report_path)
    archived = read(report_path)
    archived_digest = digest(archived)
    evidence = {'scope': 'REAL_TK_ARCHIVED_SCENE_AND_SYNTHETIC_AI_CONTRACT_ONLY',
        'source_scene': str(source), 'source_report': str(report_path),
        'new_capture': False, 'external_ai_called': False, 'ai_network_validated': False,
        'fixture_notice': 'SIMULATED_AI_RESPONSE_IS_NOT_AN_ACTUAL_MODEL_JUDGEMENT',
        'checks': [], 'screens': {}, 'callback_errors': []}

    def check(name, condition, **detail):
        evidence['checks'].append({'name': name, 'passed': bool(condition), **detail})

    OUT.mkdir(parents=True, exist_ok=True)
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
             patch('pipe_twin.stereo_camera.StereoCameraSession.open', side_effect=AssertionError('Camera forbidden in UI validation')):
            owner = ElevationCaptureDialog(app, source, output_root=OUT / 'isolated', restore=False)
            model_before = digest(owner.pipes)

            def display(report, title):
                owner.report = report
                owner.results = {r['pipe_id']: r for r in report.get('pipes', [])} if report else {}
                owner.window.title(title)
                owner.refresh_table()
                owner.status_tabs.select(owner.ai_section_status_tab)
                root.update()

            display(copy.deepcopy(archived), 'AI 分页本地核验 · 历史报告，无外部 AI 调用')
            statuses = [r['status'] for r in owner.ai_section_rows.values()]
            check('archived_without_provider_shows_not_run', statuses == ['NOT_RUN'], statuses=statuses)
            for size in ('1024x720', '1280x860'):
                owner.window.geometry(size + '+10+10')
                owner.window.deiconify()
                root.update()
                clipped = HELPERS['clipped_controls'](owner.window)
                photo_height = owner.views['left'].canvas.winfo_height()
                check('layout_' + size, not clipped and photo_height >= 180,
                      clipped_controls=clipped, photo_height=photo_height)
                screenshot(owner.window, OUT / ('not-run-' + size + '.png'))
                evidence['screens'][size] = {'clipped_controls': clipped, 'photo_height': photo_height}
            owner.show_analysis_details('ai_section')
            root.update()
            text = owner.detail_texts['ai_section'].get('1.0', 'end')
            check('not_run_details_explicit', '未运行' in text and 'AI_PROVIDER_NOT_CONFIGURED' in text)
            screenshot(owner.analysis_details_window, OUT / 'not-run-details.png')

            point_map = build_model_section_reference(owner.pipes, owner.registration_settings.get('axis_world'))
            no_response = attach_ai_section_assist(archived, point_map)
            display(no_response, 'AI 分页本地核验 · 显式 NOT_RUN 契约')
            statuses = [r['status'] for r in owner.ai_section_rows.values()]
            check('explicit_not_run_namespace_shows_not_run', statuses == ['NOT_RUN'], statuses=statuses,
                  core_status=no_response['ai_section_assist']['status'])

            display(copy.deepcopy(archived), 'AI 分页本地核验 · 安装状态基准')
            owner.open_model_viewer()
            root.update()
            viewer = owner.model_viewer
            viewer.window.geometry('1100x760+10+10')
            root.update()
            baseline_states = copy.deepcopy(viewer.installation_states)
            control = {'schema_version': 'pooled-control-section-v1', 'status': 'DEFINED',
                'station_mm': 321.5, 'interval_mm': [300.0, 343.0],
                'source': 'PIPE_SEGMENT_AGGREGATION_COMMON_SUPPORT_MIDPOINT',
                'fixture_only': True}
            fixture_map = copy.deepcopy(point_map)
            fixture_map['points'][1]['projection_ambiguity'] = 'OVERLAPPING_MODEL_POINTS'
            selected = fixture_map['points'][:4]
            response = {'point_assessments': [
                {'point_id': p['point_id'], 'section_status': 'SUPPORTED_VISIBLE',
                 'evidence_quality': 'HIGH', 'reason_codes': ['LOCAL_UI_FIXTURE_NOT_MODEL_OUTPUT'],
                 'observation_refs': ['fixture:left', 'fixture:right']} for p in selected]}
            simulated = attach_ai_section_assist(archived, fixture_map, response,
                section_evidence={'control_section': control, 'pipes': []},
                source={'provider': 'LOCAL_UI_FIXTURE_NO_NETWORK', 'model': 'synthetic-contract-fixture'})
            simulated_before = digest(simulated)
            display(simulated, 'AI 分页本地核验 · 模拟响应，不代表实网 AI')
            root.update()
            viewer.draw()
            states = {row['pipe_id']: row['status'] for row in owner.ai_section_rows.values()}
            check('simulated_candidates_render', len(states) == len(point_map['points']) and
                  states[selected[0]['pipe_id']] == 'SUPPORTED_VISIBLE', statuses=states)
            check('overlap_downgraded_to_ambiguous', states[selected[1]['pipe_id']] == 'AMBIGUOUS')
            check('missing_assessments_stay_unresolved', all(states[p['pipe_id']] == 'UNRESOLVED' for p in fixture_map['points'][4:]))
            check('ai_candidates_do_not_repaint_installation_states', viewer.installation_states == baseline_states)
            selected_key = next(k for k, row in owner.ai_section_rows.items() if row['pipe_id'] == selected[1]['pipe_id'])
            owner.ai_section_tree.selection_set(selected_key)
            owner._ai_section_selected()
            root.update()
            check('ai_selection_only_navigates_model', owner.selected_id() == selected[1]['pipe_id'] and
                  digest(owner.pipes) == model_before and viewer.installation_states == baseline_states,
                  selected_pipe=owner.selected_id())
            check('strict_report_unchanged', simulated['pipes'] == archived['pipes'] and simulated['counts'] == archived['counts'])
            check('ui_does_not_mutate_simulated_report', digest(simulated) == simulated_before)
            screenshot(owner.window, OUT / 'simulated-candidates.png')
            screenshot(viewer.window, OUT / 'simulated-selection-model.png')
            owner.show_analysis_details('ai_section')
            root.update()
            text = owner.detail_texts['ai_section'].get('1.0', 'end')
            check('control_section_and_ambiguity_visible', '321.5' in text and '候选有歧义' in text and
                  '不写入安装验收状态' in text, detail_text=text)
            screenshot(owner.analysis_details_window, OUT / 'simulated-control-section-details.png')

            old_generation = owner.generation
            owner.invalidate()
            root.update()
            check('input_change_removes_ai_candidates', owner.report is None and
                  [r['status'] for r in owner.ai_section_rows.values()] == ['NOT_RUN'])
            if owner.poll_id:
                owner.window.after_cancel(owner.poll_id)
                owner.poll_id = None
            owner.messages.put((old_generation, 'ok', (source, simulated, False)))
            owner.poll()
            root.update()
            check('stale_worker_result_not_restored', owner.report is None and
                  [r['status'] for r in owner.ai_section_rows.values()] == ['NOT_RUN'] and
                  '已丢弃旧结果' in owner.message.get(), message=owner.message.get())
            screenshot(owner.window, OUT / 'stale-result-discarded.png')
    except Exception:
        evidence['fatal_error'] = traceback.format_exc()
    finally:
        if owner is not None:
            owner.close()
        root.destroy()
        after = {name: sha(ROOT / name) for name in before}
        check('protected_sources_unchanged', before == after)
        check('archive_report_object_unchanged', digest(archived) == archived_digest)
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
