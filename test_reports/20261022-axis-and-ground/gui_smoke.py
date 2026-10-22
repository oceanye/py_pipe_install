"""Load the real replay in Tk; check controls, restored overlays and messages."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch
import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.elevation_gui import ElevationCaptureDialog
from pipe_twin.section_display import section_summary_lines


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


def dialog_text(owner, callback):
    callback()
    window = [w for w in owner.window.winfo_children() if isinstance(w, tk.Toplevel)][-1]
    window.withdraw()
    text = next(w for w in descendants(window) if isinstance(w, tk.Text)).get('1.0', 'end')
    window.destroy()
    return text


def main():
    evidence = json.loads((Path(__file__).parent/'evidence.json').read_text(encoding='utf-8'))
    manifest = ROOT/evidence['new_manifest']
    report = json.loads((manifest.parent/'report.json').read_text(encoding='utf-8'))
    root = tk.Tk(); root.withdraw()
    app = SimpleNamespace(root=root, tk=tk, ttk=ttk, messagebox=Mock(), filedialog=Mock())
    with patch('pipe_twin.workbench_profile.load_profile', return_value=(None, None)):
        dialog = ElevationCaptureDialog(app, output_root=ROOT/'outputs/axial-algorithm-audit-20261022/gui-smoke', restore=False)
        try:
            dialog.load_session(manifest)
            dialog.report = report
            dialog.results = {row['pipe_id']: row for row in report['pipes']}
            dialog.refresh_table()
            layout = []
            for width, height in ((1120, 860), (940, 660)):
                dialog.window.geometry(f'{width}x{height}'); root.update()
                controls = []
                for widget in descendants(dialog.window):
                    if isinstance(widget, ttk.Button) and widget.cget('text') in (
                            '管长方向 / 汇聚范围', '左图框选地面', '地面距离 / 姿态检查', '清除地面区域', '④ 保存并评估', '展开拍摄 / 评估设置'):
                        right = widget.winfo_rootx()-dialog.window.winfo_rootx()+widget.winfo_width()
                        bottom = widget.winfo_rooty()-dialog.window.winfo_rooty()+widget.winfo_height()
                        visible = widget.winfo_ismapped() and right <= dialog.window.winfo_width() and bottom <= dialog.window.winfo_height()
                        controls.append({'button': widget.cget('text'), 'right': right, 'bottom': bottom, 'visible': bool(visible)})
                layout.append({'window_size': [dialog.window.winfo_width(), dialog.window.winfo_height()],
                    'controls': controls, 'photo_canvas_size': [dialog.views['left'].canvas.winfo_width(), dialog.views['left'].canvas.winfo_height()],
                    'photo_canvas_visible': bool(dialog.views['left'].canvas.winfo_ismapped())})
                assert dialog.views['left'].canvas.winfo_ismapped()
                assert dialog.views['left'].canvas.winfo_height() >= 100
            values = {k: v.get() for k, v in dialog.fields.items()}
            dialog.toggle_capture_settings(); root.update()
            assert dialog.capture_settings.winfo_ismapped()
            dialog.toggle_capture_settings(); root.update()
            assert not dialog.capture_settings.winfo_ismapped()
            assert values == {k: v.get() for k, v in dialog.fields.items()}
            dialog.window.withdraw()
            assert all(button['visible'] for row in layout for button in row['controls']), layout
            assert len(dialog.views['left'].canvas.find_withtag('axial-band-limit')) == 2
            assert len(dialog.views['left'].canvas.find_withtag('ground-region')) == 2
            assert not dialog.views['right'].canvas.find_withtag('ground-region')
            assert dialog.registration_settings['aggregation_band'] == evidence['band']
            assert dialog.registration_settings['ground_region']['selection_source'] == 'ASSISTANT_DIAGNOSTIC'
            ground_text = dialog_text(dialog, dialog.show_ground_check)
            assert '尚未经你确认' in ground_text and '检查未通过' in ground_text
            assert '垂直距离（暂估）' not in ground_text
            section_text = dialog_text(dialog, dialog.show_section_aggregation)
            assert '共同轴向区间' in section_text and '鼠标仅提供大致方向' in section_text
            old_text = '\n'.join(section_summary_lines({'section_evaluation': evidence['previous']}))
            assert '共同轴向区间' not in old_text and '实际边缘方向从图像重新估计' not in old_text
            dialog.report = None
            pending = dialog_text(dialog, dialog.show_ground_check)
            assert '已设置地面区域' in pending and '尚未设置地面区域' not in pending
            dialog.report = report
            app.messagebox.showerror.assert_not_called()
            result = {'layout': layout, 'band_and_ground_restored': True, 'real_report_loaded': True,
                'camera_opened': False, 'failed_ground_does_not_publish_distance': True,
                'pending_ground_and_historical_algorithm_labels_correct': True,
                'section_explains_common_station_and_mouse_hint': True}
            (Path(__file__).parent/'gui-validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            print(json.dumps(result, ensure_ascii=False))
        finally:
            dialog.close(); root.update(); root.destroy()


if __name__ == '__main__':
    main()
