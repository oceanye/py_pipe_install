"""Verify the ground correction entry and honest real-photo failure message."""
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
    evidence = json.loads((Path(__file__).parent/'evidence.json').read_text(encoding='utf-8'))
    manifest = ROOT/evidence['new_manifest']
    report = json.loads((manifest.parent/'report.json').read_text(encoding='utf-8'))
    root = tk.Tk(); root.withdraw()
    app = SimpleNamespace(root=root, tk=tk, ttk=ttk, messagebox=Mock(), filedialog=Mock())
    with patch('pipe_twin.workbench_profile.load_profile', return_value=(None, None)):
        dialog = ElevationCaptureDialog(app, output_root=ROOT/'outputs/ground-correction-20261022/gui-smoke', restore=False)
        try:
            dialog.load_session(manifest); dialog.report = report
            dialog.results = {row['pipe_id']: row for row in report['pipes']}
            dialog.refresh_table()
            dialog.window.geometry('940x660'); root.update()
            button = next(w for w in descendants(dialog.window) if isinstance(w, ttk.Button) and w.cget('text') == '地面修正')
            assert button.winfo_ismapped()
            assert dialog.views['left'].canvas.winfo_ismapped()
            button.invoke()
            window = [w for w in dialog.window.winfo_children() if isinstance(w, tk.Toplevel)][-1]
            window.withdraw()
            text = next(w for w in descendants(window) if isinstance(w, tk.Text)).get('1.0', 'end')
            assert '未应用地面修正' in text and '地面平面检查未通过' in text
            assert '保留自由轴向结果' in text and '相机到地面的垂直距离（暂估）' not in text
            result = {'real_report_loaded': True, 'button': button.cget('text'), 'window_title': window.title(),
                'minimum_window': [dialog.window.winfo_width(), dialog.window.winfo_height()],
                'photo_canvas_visible': True, 'failed_ground_correction_not_reported_as_applied': True, 'camera_opened': False}
            window.destroy(); app.messagebox.showerror.assert_not_called()
            (Path(__file__).parent/'gui-validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            print(json.dumps(result, ensure_ascii=False))
        finally:
            dialog.close(); root.update(); root.destroy()


if __name__ == '__main__':
    main()
