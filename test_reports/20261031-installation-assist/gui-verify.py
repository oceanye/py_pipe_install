"""Real Tk check of the separate assistance UI using archived, portable evidence."""
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
sys.path.insert(0, str(ROOT))
from pipe_twin.elevation_gui import ElevationCaptureDialog
from pipe_twin.assist_display import assist_preview_states

HELPERS = runpy.run_path(str(ROOT/'test_reports/20261030-function-review/gui-verify.py'))
screenshot = runpy.run_path(str(ROOT/'test_reports/20261030-ground-monitor/gui-verify.py'))['screenshot']
OUT = ROOT/'outputs/installation-assist-20261031/gui'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pointer = json.loads((ROOT/'outputs/installation-assist-20261031/latest-scene.json').read_text(encoding='utf-8'))
    source = Path(pointer['manifest_path'])
    original = ROOT/'outputs/elevation_workbench/captures/elevation-20261030-183635-b6b7b875'
    protected = [source, source.with_name('report.json'), original/'manifest.json', original/'report.json',
                 ROOT/'outputs/elevation_workbench/last_scene.json',
                 ROOT/'outputs/measurement_workbench/workbench_profile.json',
                 ROOT/'outputs/measurement_workbench/calibration_current.json']
    before = {str(p.relative_to(ROOT)): sha(p) for p in protected}
    report = json.loads(source.with_name('report.json').read_text(encoding='utf-8'))
    root = tk.Tk(); root.withdraw()
    errors = []
    root.report_callback_exception = lambda *e: errors.append(''.join(traceback.format_exception(*e)))
    style = ttk.Style(root); style.theme_use('clam'); style.configure('.', font=('Microsoft YaHei UI', 10))
    app = SimpleNamespace(root=root, tk=tk, ttk=ttk, messagebox=Mock(), filedialog=Mock())
    with patch('pipe_twin.workbench_profile.load_profile', return_value=(None, None)):
        owner = ElevationCaptureDialog(app, source, output_root=OUT/'isolated', restore=False)
    evidence = {'scope': 'ARCHIVED_SCENE_UI_NOT_NEW_CAPTURE_OR_METRIC_VALIDATION', 'screens': {}}
    try:
        owner.window.title('辅助安装识别 · 历史照片核验')
        owner.report = report; owner.results = {r['pipe_id']: r for r in report['pipes']}
        owner.refresh_table(); root.update()
        assert len(owner.assist_rows) == 4
        assert owner.status_tabs.select() == str(owner.assist_status_tab)
        assert '辅助识别' in owner.workflow_status.get()
        evidence['strict_counts'] = report['counts']
        evidence['assistance_rows'] = [{k: v for k, v in row.items() if k in ('label', 'values', 'preview_pipe_id')}
                                        for row in owner.assist_rows.values()]
        for size in ('1024x720', '1280x860'):
            owner.window.geometry(size+'+10+10'); owner.window.deiconify(); root.update()
            clipped = HELPERS['clipped_controls'](owner.window)
            height = owner.views['left'].canvas.winfo_height()
            assert not clipped and height >= 180, (size, clipped, height)
            assert owner.assist_actions.winfo_ismapped()
            assert owner.views['left'].canvas.find_withtag('assist-candidate')
            assert not owner.views['left'].canvas.find_withtag('image-candidate')
            screenshot(owner.window, OUT/(size+'.png'))
            evidence['screens'][size] = {'clipped_controls': clipped, 'photo_height': height,
                'workflow': owner.workflow_status.get(), 'selected_photo_labels': True}
        owner.show_analysis_details('assist'); root.update()
        detail = owner.detail_texts['assist'].get('1.0', 'end')
        assert '采用已知管径辅助定位' in detail and '不是本次实测' in detail
        assert '横向位' in detail and '层' in detail
        screenshot(owner.analysis_details_window, OUT/'details.png')
        evidence['detail'] = detail
        owner.analysis_details_window.destroy()
        owner.open_model_viewer(); root.update()
        viewer = owner.model_viewer
        viewer.window.geometry('1100x760+10+10'); root.update()
        assert not HELPERS['clipped_controls'](viewer.window)
        candidates = {pid for pid, state in viewer.installation_states.items() if state in ('SUSPECTED_INSTALLED', 'ASSIST_CANDIDATE')}
        assert len(candidates) == 4
        assert not viewer.section_canvas.find_withtag('state:INSTALLED')
        screenshot(viewer.window, OUT/'model-primary.png')
        evidence['first_plan_yellow_pipes'] = sorted(candidates)
        scene_row, x, y, _ = viewer.scene_hits[1]
        viewer.pick_scene_pipe(SimpleNamespace(x=x, y=y)); root.update()
        assert scene_row['pipe_id'] is None and scene_row['assist_pipe_id']
        assert owner.selected_id() == scene_row['assist_pipe_id']
        assert owner.selected_assist_row()['preview_pipe_id'] == scene_row['assist_pipe_id']
        other = next(hit for hit in viewer.section_hits if hit[0] in candidates and hit[0] != scene_row['assist_pipe_id'])
        viewer.pick_section_pipe(SimpleNamespace(x=other[1], y=other[2])); root.update()
        assert owner.selected_assist_row()['preview_pipe_id'] == other[0]
        evidence['scene_and_model_clicks_link_to_the_current_photo_candidate_without_assigning_identity'] = True
        if len(owner.assist_plan_box.cget('values')) > 1:
            owner.assist_plan_box.current(1); owner._assist_plan_selected(); root.update()
            expected = assist_preview_states(report, {p['pipe_id']: 'UNKNOWN' for p in owner.pipes}, 1)
            actual = {pid for pid, state in viewer.installation_states.items() if state in ('SUSPECTED_INSTALLED', 'ASSIST_CANDIDATE')}
            assert actual == {pid for pid, state in expected.items() if state in ('SUSPECTED_INSTALLED', 'ASSIST_CANDIDATE')}
            assert len(actual) == 4
            screenshot(viewer.window, OUT/'model-alternative.png')
            evidence['second_plan_yellow_pipes'] = sorted(actual)
            evidence['switch_does_not_light_union_of_plans'] = True
        assert all(row['installation_state'] == 'UNKNOWN' for row in report['pipes'])
        owner.invalidate(); root.update()
        assert not owner.assist_rows and not viewer.section_canvas.find_withtag('state:ASSIST_CANDIDATE')
        assert not viewer.section_canvas.find_withtag('state:SUSPECTED_INSTALLED')
        evidence['input_change_clears_suggestions'] = True
        assert not errors, errors
    finally:
        owner.close(); root.destroy()
    assert before == {str(p.relative_to(ROOT)): sha(p) for p in protected}
    evidence.update(callback_errors=errors, sources_unchanged=True, source_sha256=before)
    Path(__file__).with_name('gui-evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(evidence['screens'], ensure_ascii=False))


if __name__ == '__main__':
    main()
