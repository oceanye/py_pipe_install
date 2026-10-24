"""Exercise actual Tk widgets against the saved offline replay."""
from pathlib import Path
import json
import sys
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from pipe_twin.elevation_gui import ElevationCaptureDialog

OUT = ROOT/'outputs/local-support-20261012-173724'
report = json.loads((OUT/'report.json').read_text(encoding='utf-8'))
root = tk.Tk()
root.withdraw()
app = SimpleNamespace(root=root,tk=tk,ttk=ttk,filedialog=Mock(),messagebox=Mock())
with patch('pipe_twin.workbench_profile.load_profile',return_value=(None,None)):
    owner = ElevationCaptureDialog(app,output_root=OUT/'gui-workbench',restore=False)
owner.report = report

def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)

diagnostic = next(w for w in descendants(owner.window) if isinstance(w,ttk.Button) and w.cget('text')=='识别诊断')
records = []
for size in ('940x660','1120x760','1120x860'):
    owner.window.geometry(size+'+20+20')
    root.update()
    controls = {}
    for label,widget in [('evaluate',owner.run_button),('region',owner.region_button),('diagnostic',diagnostic)]:
        x = widget.winfo_rootx()-owner.window.winfo_rootx()
        y = widget.winfo_rooty()-owner.window.winfo_rooty()
        w,h = widget.winfo_width(),widget.winfo_height()
        visible = bool(widget.winfo_ismapped() and x>=0 and y>=0 and x+w<=owner.window.winfo_width() and y+h<=owner.window.winfo_height())
        assert visible,(size,label,[x,y,w,h])
        controls[label] = {'rect':[x,y,w,h],'fully_visible':visible}
    records.append({'size':size,'controls':controls})
diagnostic.invoke()
root.update()
dialog = next(w for w in owner.window.winfo_children() if isinstance(w,tk.Toplevel))
widget = next(w for w in descendants(dialog) if isinstance(w,tk.Text))
text = widget.get('1.0','end')
assert 'Z02:IMG-left-004' in text
assert '局部 W03' in text and '拟合误差' in text
assert '不代表测量合格率' in text
assert widget.cget('state')=='disabled'
assert any(isinstance(w,ttk.Scrollbar) for w in descendants(dialog))
(OUT/'gui.json').write_text(json.dumps({'layouts':records,'diagnostic_text_verified':True,
    'read_only_scrollable':True,'camera_opened':False},indent=2),encoding='utf-8')
dialog.destroy()
owner.close()
root.destroy()
print('GUI_DIAGNOSTICS_LAYOUT_AND_CONTENT_PASSED')
