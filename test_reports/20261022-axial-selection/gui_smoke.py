"""Check real saved selection/report in Tk, including minimum window layout."""
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
from pipe_twin.axial_selection import band_geometry


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


def main():
    evidence = json.loads((Path(__file__).parent/'evidence.json').read_text(encoding='utf-8'))
    manifest = ROOT/evidence['new_manifest']
    report = json.loads((manifest.parent/'report.json').read_text(encoding='utf-8'))
    root = tk.Tk(); root.withdraw()
    app = SimpleNamespace(root=root, tk=tk, ttk=ttk, messagebox=Mock(), filedialog=Mock())
    with patch('pipe_twin.workbench_profile.load_profile', return_value=(None, None)):
        dialog = ElevationCaptureDialog(app, output_root=ROOT/'outputs/axial-selection-20261022/gui-smoke', restore=False)
        try:
            dialog.load_session(manifest)
            dialog.report = report
            dialog.results = {row['pipe_id']: row for row in report['pipes']}
            dialog.refresh_table()
            layout = []
            for width in (1120, 940):
                dialog.window.geometry(f'{width}x860')
                root.update()
                for widget in descendants(dialog.window):
                    if isinstance(widget, ttk.Button) and widget.cget('text') in ('管长方向 / 汇聚范围', '清除汇聚范围', '④ 保存并评估'):
                        right = widget.winfo_rootx()-dialog.window.winfo_rootx()+widget.winfo_width()
                        visible = widget.winfo_ismapped() and right <= dialog.window.winfo_width()
                        layout.append({'window_width': dialog.window.winfo_width(), 'button': widget.cget('text'), 'right': right, 'visible': bool(visible)})
            dialog.window.withdraw()
            view = dialog.views['left']
            assert len(view.canvas.find_withtag('axial-band-limit')) == 2
            assert len(view.canvas.find_withtag('axial-band-handle')) == 2
            assert dialog.registration_settings['aggregation_band'] == evidence['band']
            assert not dialog.views['right'].canvas.find_withtag('axial-band-limit')
            assert all(item['visible'] for item in layout), layout
            dialog.show_section_aggregation()
            windows = [w for w in dialog.window.winfo_children() if isinstance(w, tk.Toplevel)]
            windows[-1].withdraw()
            text = next(w for w in descendants(windows[-1]) if isinstance(w, tk.Text)).get('1.0', 'end')
            assert '手动汇聚范围' in text and f"{band_geometry(evidence['band'])[2]:.1f} px" in text
            windows[-1].destroy()
            app.messagebox.showerror.assert_not_called()
            result = {'layout': layout, 'two_perpendicular_lines_restored': True, 'two_handles_restored': True,
                      'real_report_loaded': True, 'camera_opened': False, 'summary_mentions_band': True}
            (Path(__file__).parent/'gui-validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            print(json.dumps(result, ensure_ascii=False))
        finally:
            dialog.close(); root.update(); root.destroy()


if __name__ == '__main__':
    main()
