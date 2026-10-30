"""Real Tk, archived report, no camera access and no scene writes."""
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
from pipe_twin.view_geometry import report_view_geometry

HELPERS = runpy.run_path(str(ROOT/'test_reports/20261030-function-review/gui-verify.py'))
screenshot = runpy.run_path(str(ROOT/'test_reports/20261030-ground-monitor/gui-verify.py'))['screenshot']
SOURCE = ROOT/'outputs/elevation_workbench/captures/elevation-20261030-183635-b6b7b875'
OUT = ROOT/'outputs/geometry-path-20261031/gui'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [SOURCE/'manifest.json', SOURCE/'report.json',
                 ROOT/'outputs/elevation_workbench/last_scene.json',
                 ROOT/'outputs/measurement_workbench/workbench_profile.json',
                 ROOT/'outputs/measurement_workbench/calibration_current.json']
    before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    report_file = ROOT/'outputs/geometry-path-20261031/report.json'
    report = json.loads(report_file.read_text(encoding='utf-8'))
    values = report_view_geometry(report)
    assert values and values[0]['status'] == 'CONDITIONAL_GEOMETRY'
    root = tk.Tk(); root.withdraw()
    errors = []
    root.report_callback_exception = lambda *e: errors.append(''.join(traceback.format_exception(*e)))
    style = ttk.Style(root); style.theme_use('clam')
    style.configure('.', font=('Microsoft YaHei UI', 10))
    app = SimpleNamespace(root=root, tk=tk, ttk=ttk, messagebox=Mock(), filedialog=Mock())
    evidence = {'scope': 'ARCHIVED_REPORT_GUI_NOT_LIVE_CAMERA', 'screens': {}, 'geometry': values}
    with patch('pipe_twin.workbench_profile.load_profile', return_value=(None, None)):
        owner = ElevationCaptureDialog(app, SOURCE/'manifest.json', output_root=OUT/'isolated', restore=False)
    try:
        owner.window.title('机位几何核验 · 历史照片')
        owner.report = report
        owner.results = {r['pipe_id']: r for r in report['pipes']}
        owner.refresh_table()
        rows = list(owner.solver_rows.values())
        assert [row['label'] for row in rows] == ['S1', 'S2', 'S3', 'S4']
        assert all(row['solved'] and row['diameter_mm'] is not None for row in rows)
        assert all('全幅有效深度' in row['detail'] and '圆柱拟合残差过大 55 次' in row['detail'] for row in rows)
        evidence['all_four_sections_keep_solved_labels_and_values_with_depth_diagnostic'] = True
        for size in ('1024x720', '1280x860'):
            owner.window.geometry(size+'+10+10'); owner.window.deiconify(); root.update()
            clipped = HELPERS['clipped_controls'](owner.window)
            assert not clipped, clipped
            buttons = [w for w in HELPERS['children'](owner.window)
                       if w.winfo_class() == 'TButton' and w.cget('text') == '机位几何检查']
            assert len(buttons) == 1 and buttons[0].winfo_ismapped()
            height = owner.views['left'].canvas.winfo_height()
            assert height >= 180
            screenshot(owner.window, OUT/(size+'.png'))
            buttons[0].invoke(); root.update()
            assert owner.detail_tabs.select() == str(owner.detail_pages['geometry'])
            text = owner.detail_texts['geometry'].get('1.0', 'end')
            assert '不是实时角度' in text and '59.82 mm' in text
            assert '条件诊断' in text and '不能凭此页' in text
            assert not HELPERS['clipped_controls'](owner.analysis_details_window)
            screenshot(owner.analysis_details_window, OUT/(size+'-details.png'))
            evidence['screens'][size] = {'clipped_controls': clipped, 'photo_height': height,
                                        'geometry_text': text, 'button_opens_geometry_tab': True}
            owner.analysis_details_window.destroy()
        # Updating a live details window must not keep stale geometry.
        owner.report = None; owner.show_analysis_details('geometry'); root.update()
        text = owner.detail_texts['geometry'].get('1.0', 'end')
        assert '请先抓拍并评估' in text and '59.82 mm' not in text
        evidence['cleared_report_removes_old_geometry'] = True
        assert not errors, errors
    finally:
        owner.close(); root.destroy()
    assert before == {str(p.relative_to(ROOT)): digest(p) for p in protected}
    evidence.update(callback_errors=errors, source_unchanged=True, protected_source_sha256=before)
    Path(__file__).with_name('gui-evidence.json').write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(evidence['screens'], ensure_ascii=False))


if __name__ == '__main__':
    main()
