"""Load the real replay report in isolated Tk widgets without opening cameras."""
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


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


def main():
    output = ROOT/'outputs/axial-aggregation-20261022'
    report = json.loads((output/'report.json').read_text(encoding='utf-8'))
    manifest = ROOT/'outputs/retry-20261020-131151/active-static-v2/burst/elevation-20261020-151239-63c7f244/manifest.json'
    root = tk.Tk(); root.withdraw()
    app = SimpleNamespace(root=root, tk=tk, ttk=ttk, messagebox=Mock(), filedialog=Mock())
    with patch('pipe_twin.workbench_profile.load_profile', return_value=(None, None)):
        dialog = ElevationCaptureDialog(app, output_root=output/'gui-smoke', restore=False)
        dialog.window.withdraw()
        try:
            dialog.load_session(manifest)
            dialog.report = report
            dialog.results = {row['pipe_id']: row for row in report['pipes']}
            dialog.refresh_table()
            root.update_idletasks()
            buttons = [str(w.cget('text')) for w in descendants(dialog.window) if isinstance(w, ttk.Button)]
            assert '管段汇聚' in buttons and '断面尺寸与就位' in buttons
            assert dialog.tree.heading('position')['text'] == '断面就位'
            assert len(dialog.tree.get_children()) == 12
            assert all(dialog.tree.set(row, 'position') == '就位待判定' for row in dialog.tree.get_children())
            segments = {role: len(view.canvas.find_withtag('pooled-boundary')) for role, view in dialog.views.items()}
            assert all(segments.values())
            dialog.show_section_aggregation()
            windows = [w for w in dialog.window.winfo_children() if isinstance(w, tk.Toplevel)]
            for window in windows:
                window.withdraw()
            text = next(w for w in descendants(windows[-1]) if isinstance(w, tk.Text)).get('1.0', 'end')
            assert '不会恢复遮挡' in text and '尚未验证尺寸精度' in text and 'POOL-L' in text
            windows[-1].destroy()
            dialog.show_measurement()
            windows = [w for w in dialog.window.winfo_children() if isinstance(w, tk.Toplevel)]
            for window in windows:
                window.withdraw()
            text = next(w for w in descendants(windows[-1]) if isinstance(w, tk.Text)).get('1.0', 'end')
            assert '模型管道的断面相对就位' in text
            windows[-1].destroy()
            app.messagebox.showerror.assert_not_called()
            result = {'real_report_loaded': True, 'model_rows': 12, 'all_relative_positions_pending': True,
                      'boundary_overlay_line_count': segments, 'aggregation_dialog': 'PASS',
                      'dimension_position_dialog': 'PASS', 'camera_opened': False}
            (Path(__file__).parent/'gui-validation.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
            print(json.dumps(result))
        finally:
            dialog.close(); root.update(); root.destroy()


if __name__ == '__main__':
    main()
