"""Exercise the real tilt button and inspect small-window overlay placement."""
from pathlib import Path
import json
import sys
import subprocess
import time
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


def screenshot(path, x, y, width, height):
    # Use Windows' existing drawing API; no extra GUI/runtime dependency.
    escaped = str(path).replace("'", "''")
    command = f"""Add-Type -AssemblyName System.Drawing
$bitmap = New-Object System.Drawing.Bitmap({width}, {height})
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
try {{
    $graphics.CopyFromScreen({x}, {y}, 0, 0, $bitmap.Size)
    $bitmap.Save('{escaped}', [System.Drawing.Imaging.ImageFormat]::Png)
}} finally {{ $graphics.Dispose(); $bitmap.Dispose() }}
"""
    subprocess.run(['powershell', '-NoProfile', '-Command', command], check=True, capture_output=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)


def main():
    evidence = json.loads((Path(__file__).parent/'evidence.json').read_text(encoding='utf-8'))
    manifest = ROOT/evidence['new_manifest']
    report = json.loads((manifest.parent/'report.json').read_text(encoding='utf-8'))
    root = tk.Tk(); root.withdraw()
    app = SimpleNamespace(root=root, tk=tk, ttk=ttk, messagebox=Mock(), filedialog=Mock())
    output = ROOT/'outputs/tilt-inspection-20261023/gui-smoke'
    with patch('pipe_twin.workbench_profile.load_profile', return_value=(None, None)):
        dialog = ElevationCaptureDialog(app, output_root=output, restore=False)
        try:
            dialog.load_session(manifest); dialog.confirmed.set(True); dialog.report = report
            dialog.results = {row['pipe_id']: row for row in report['pipes']}
            dialog.refresh_table()
            dialog.window.geometry('940x660+20+20'); root.update()
            button = next(w for w in descendants(dialog.window) if isinstance(w, ttk.Button) and w.cget('text') == '倾角检查')
            assert button.winfo_ismapped()
            assert button.winfo_rootx()+button.winfo_width() < dialog.window.winfo_rootx()+dialog.window.winfo_width()
            canvas = dialog.views['left'].canvas
            assert canvas.winfo_ismapped() and canvas.winfo_height() > 100
            minimum_photo_canvas_height = canvas.winfo_height()
            labels = canvas.find_withtag('ground-angle')
            assert len(labels) == 1
            assert '候选 / 不用于修正' in canvas.itemcget(labels[0], 'text')
            assert not dialog.views['right'].canvas.find_withtag('ground-angle')
            button.invoke()
            deadline = time.monotonic()+45
            while dialog.busy and time.monotonic() < deadline:
                root.update(); time.sleep(.03)
            assert not dialog.busy
            app.messagebox.showerror.assert_not_called()
            window = [w for w in dialog.window.winfo_children() if isinstance(w, tk.Toplevel)][-1]
            text = next(w for w in descendants(window) if isinstance(w, tk.Text)).get('1.0', 'end')
            assert '候选 / 不用于修正' in text and '检查不重算管径' in text
            window.geometry('1000x760+30+30'); window.attributes('-topmost', True); root.update()
            preview = next(w for w in descendants(window) if isinstance(w, tk.Canvas))
            assert len(preview.find_withtag('ground-preview-angle')) == 1
            assert len(preview.find_withtag('ground-preview-roi')) == 1
            assert len(preview.find_withtag('ground-preview')) == 1
            screenshot(output/'ground-only-preview.png', window.winfo_rootx(), window.winfo_rooty(), 1000, 760)
            window.destroy()
            assert dialog.report is report
            dialog.window.attributes('-topmost', True)
            dialog.window.lift(); root.update()
            x, y = dialog.window.winfo_rootx(), dialog.window.winfo_rooty()
            screenshot(output/'gui-940x660.png', x, y, 940, 660)
            dialog.window.geometry('1440x900+20+20'); root.update()
            x, y = dialog.window.winfo_rootx(), dialog.window.winfo_rooty()
            screenshot(output/'gui-1440x900.png', x, y, 1440, 900)
            result = {'real_report_loaded': True, 'real_standalone_button_exercised': True,
                'minimum_window': [940, 660], 'minimum_photo_canvas_height': minimum_photo_canvas_height,
                'candidate_angles_visible_on_left_only': True, 'rejected_candidate_not_used_as_correction': True,
                'prior_pipe_report_preserved': True, 'camera_opened': False,
                'clean_ground_preview_visible': True,
                'inspection_status': dialog.ground_inspection['status'], 'mask_pixels': dialog.ground_inspection['observed_support']['pixel_count'],
                'screenshots': [str((output/name).relative_to(ROOT)) for name in ('gui-940x660.png', 'gui-1440x900.png', 'ground-only-preview.png')]}
            (Path(__file__).parent/'gui-validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            print(json.dumps(result, ensure_ascii=False))
        finally:
            dialog.close(); root.update(); root.destroy()


if __name__ == '__main__':
    main()
