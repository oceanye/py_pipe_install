"""Render the real replay result in Tk without camera or source-file writes."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import hashlib
import json
import runpy
import sys
import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.elevation_gui import ElevationCaptureDialog
from pipe_twin.section_display import boundary_overlay_segments

HELPERS = runpy.run_path(str(ROOT/'test_reports/20261030-function-review/gui-verify.py'))
SCENE = ROOT/'outputs/elevation_workbench/captures/elevation-20261030-183635-b6b7b875'
REPLAY = ROOT/'outputs/ground-color-boundaries-20261030/report.json'
OUT = ROOT/'outputs/ground-color-boundaries-20261030/gui'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report = json.loads(REPLAY.read_text(encoding='utf-8'))
    assert report['ground_check']['source_scope'] == 'REFERENCE_STEREO_PAIR_ONLY'
    assert report['temporal_fusion']['status'] == 'INVALID'
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [SCENE/'manifest.json', SCENE/'report.json', REPLAY,
                 ROOT/'outputs/elevation_workbench/last_scene.json',
                 ROOT/'outputs/measurement_workbench/calibration_current.json',
                 ROOT/'outputs/measurement_workbench/workbench_profile.json']
    protected += list((SCENE/'photos').rglob('*.png'))
    before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    root = tk.Tk()
    root.withdraw()
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', font=('Microsoft YaHei UI', 10))
    style.configure('Treeview', rowheight=25)
    app = SimpleNamespace(root=root, tk=tk, ttk=ttk, messagebox=Mock(), filedialog=Mock())
    with patch('pipe_twin.workbench_profile.load_profile', return_value=(None, None)):
        d = ElevationCaptureDialog(app, SCENE/'manifest.json', output_root=OUT/'isolated', restore=False)
    result = {'source_scene': SCENE.name, 'replay_report': str(REPLAY.relative_to(ROOT)), 'screens': {}}
    try:
        d.window.title('同图复核 · 当前帧地面与可见边界')
        d.report = report
        d.results = {row['pipe_id']: row for row in report['pipes']}
        d.refresh_table()
        for geometry in ['1024x720', '1280x860']:
            d.window.geometry(geometry+'+10+10')
            d.window.deiconify()
            root.update()
            status = d.workflow_status.get()
            assert '10 对，多帧检查未通过' in status
            assert '地面（仅当前帧；连拍未通过）' in status
            height = d.views['left'].canvas.winfo_height()
            clipped = HELPERS['clipped_controls'](d.window)
            assert height >= 180 and not clipped, (geometry, height, clipped)
            overlays = {}
            for role, view in d.views.items():
                canvas = view.canvas
                line_count = len(canvas.find_withtag('pooled-boundary'))
                assert line_count == len(boundary_overlay_segments(report, role))
                labels = [canvas.itemcget(i, 'text') for i in canvas.find_withtag('ground-angle')]
                overlays[role] = {'stable_edge_segments': line_count, 'ground_labels': labels}
            assert '仅当前帧' in '\n'.join(overlays['left']['ground_labels'])
            assert '连拍未通过' in '\n'.join(overlays['left']['ground_labels'])
            result['screens'][geometry] = {'photo_height': height, 'clipped_controls': clipped,
                'workflow_text': status, 'workflow_label_height': d.workflow_label.winfo_height(),
                'overlays': overlays}
            HELPERS['screenshot'](d.window, OUT/(geometry+'.png'))
    finally:
        d.close()
        root.destroy()
    assert before == {str(p.relative_to(ROOT)): digest(p) for p in protected}
    result['source_unchanged'] = True
    result['protected_source_sha256'] = before
    Path(__file__).with_name('gui-evidence.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result['screens'], ensure_ascii=False))


if __name__ == '__main__':
    main()
