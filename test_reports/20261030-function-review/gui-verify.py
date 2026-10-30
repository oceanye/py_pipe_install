"""Real Tk workflow review against the frozen user scene, without camera access."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import hashlib
import json
import subprocess
import sys
import time
import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.elevation_gui import ElevationCaptureDialog

SCENE = ROOT / 'outputs/elevation_workbench/captures/elevation-20261030-161444-3ab7011e'
OUT = ROOT / 'outputs/function-review-20261030/gui'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def children(parent):
    for child in parent.winfo_children():
        if isinstance(child, tk.Toplevel):
            continue
        yield child
        yield from children(child)


def clipped_controls(window):
    clipped = []
    for widget in children(window):
        if not widget.winfo_ismapped() or widget.winfo_class() not in ('TButton', 'TMenubutton', 'TCombobox', 'TEntry', 'TCheckbutton'):
            continue
        x, y = widget.winfo_rootx()-window.winfo_rootx(), widget.winfo_rooty()-window.winfo_rooty()
        w, h = widget.winfo_width(), widget.winfo_height()
        if x < 0 or y < 0 or x+w > window.winfo_width()+1 or y+h > window.winfo_height()+1:
            clipped.append([str(widget), x, y, w, h])
    return clipped


def screenshot(window, path):
    window.lift()
    window.update()
    time.sleep(.35)  # Let the native window activation animation settle.
    script = f"""Add-Type -AssemblyName System.Drawing
$bitmap=New-Object System.Drawing.Bitmap({window.winfo_width()},{window.winfo_height()})
$graphics=[System.Drawing.Graphics]::FromImage($bitmap)
$graphics.CopyFromScreen({window.winfo_rootx()},{window.winfo_rooty()},0,0,$bitmap.Size)
$bitmap.Save('{str(path).replace(chr(39), chr(39)*2)}')
$graphics.Dispose()
$bitmap.Dispose()
"""
    subprocess.run(['powershell', '-NoProfile', '-Command', script], check=True, capture_output=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protected = [SCENE/'manifest.json', SCENE/'report.json', SCENE/'photos/left.png', SCENE/'photos/right.png',
                 ROOT/'outputs/elevation_workbench/last_scene.json', ROOT/'outputs/measurement_workbench/calibration_current.json']
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
    result = {'source_dataset': SCENE.name, 'screens': {}}
    try:
        d.window.title('功能复核 · 独立只读现场')
        d.report = json.loads((SCENE/'report.json').read_text(encoding='utf8'))
        d.results = {row['pipe_id']: row for row in d.report['pipes']}
        d.refresh_table()
        for geometry in ['1024x720', '1280x860']:
            d.window.geometry(geometry+'+10+10')
            d.window.deiconify()
            root.update()
            status = d.workflow_status.get()
            assert '单帧' in status and '试验已应用，未验证' in status and 'CAD 已对应 0' in status
            height = d.views['left'].canvas.winfo_height()
            clipped = clipped_controls(d.window)
            assert height >= 180 and not clipped, (geometry, height, clipped)
            result['screens'][geometry] = {'photo_height': height, 'clipped_controls': clipped,
                'workflow_text': status, 'workflow_label_height': d.workflow_label.winfo_height()}
            screenshot(d.window, OUT/(geometry+'.png'))
        d.open_model_viewer()
        viewer = d.model_viewer
        viewer.window.geometry('1100x760+10+10')
        viewer.view_mode.set('从 +Y 侧朝 -Y 看')
        viewer.preview_side()
        d.submit(True)
        assert not d.busy and viewer.has_pending_camera_side()
        assert '预览' in d.message.get()
        assert d.registration_settings.get('camera_side_world') is None
        result['pending_direction'] = d.message.get()
        screenshot(viewer.window, OUT/'pending-direction.png')
        viewer.restore_camera_side()
        assert not viewer.has_pending_camera_side()
        assert d.registration_settings.get('camera_side_world') is None
        result['preview_reverted_without_changing_field_direction'] = True
        assert not clipped_controls(viewer.window)
    finally:
        d.close()
        root.destroy()
    assert before == {str(p.relative_to(ROOT)): digest(p) for p in protected}
    result['protected_source_sha256'] = before
    result['source_unchanged'] = True
    Path(__file__).with_name('gui-evidence.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result['screens'], ensure_ascii=False))


if __name__ == '__main__':
    main()
