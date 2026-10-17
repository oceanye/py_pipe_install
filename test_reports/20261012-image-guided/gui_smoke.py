from pathlib import Path
import json
import sys
import tkinter as tk
from tkinter import ttk
from types import SimpleNamespace
from unittest.mock import Mock, patch
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from pipe_twin.elevation_gui import ElevationCaptureDialog,recognition_summary

OUT=ROOT/'outputs/image-guided-20261012'
OUT.mkdir(parents=True,exist_ok=True)
source=ROOT/'outputs/elevation_workbench/captures/elevation-20261012-135039-b075dca2'
report=json.loads((OUT/'middle.json').read_text(encoding='utf-8'))
root=tk.Tk();root.withdraw()
app=SimpleNamespace(root=root,tk=tk,ttk=ttk,filedialog=Mock(),messagebox=Mock())
with patch('pipe_twin.workbench_profile.load_profile',return_value=(None,None)):
    owner=ElevationCaptureDialog(app,output_root=OUT/'gui-workbench',restore=False)
owner.load_model(next((ROOT/'test_model').glob('*.dxf')))
owner.report=report
owner.results={p['pipe_id']:p for p in report['pipes']}
for eye in ('left','right'):
    owner.views[eye].set_image(cv2.imdecode(np.fromfile(source/f'photos/{eye}.png',np.uint8),1))
owner.summary.set('安装 0 · 未安装 0 · 不确定 12\n'+recognition_summary(report))
owner.message.set('离线 GUI 验证：未调用相机，未更改现场设置。')
owner.refresh_table()
def descendants(w):
    for child in w.winfo_children():
        yield child
        yield from descendants(child)
relative=next(w for w in descendants(owner.window) if isinstance(w,ttk.Button) and w.cget('text')=='管间尺寸对比')
records=[]
for size in ('940x660','1120x760','1120x860'):
    owner.window.geometry(size+'+20+20');root.update()
    controls={}
    for label,widget in [('evaluate',owner.run_button),('region',owner.region_button),('relative',relative)]:
        x=widget.winfo_rootx()-owner.window.winfo_rootx();y=widget.winfo_rooty()-owner.window.winfo_rooty()
        w=widget.winfo_width();h=widget.winfo_height()
        visible=bool(widget.winfo_ismapped() and x>=0 and y>=0 and x+w<=owner.window.winfo_width() and y+h<=owner.window.winfo_height())
        controls[label]={'rect':[x,y,w,h],'fully_visible':visible}
        assert visible,(size,label,controls[label])
    for eye in ('left','right'):
        assert len(owner.views[eye].canvas.find_withtag('image-candidate'))==8
    records.append({'size':size,'controls':controls})
relative.invoke();root.update()
dialogs=[w for w in owner.window.winfo_children() if isinstance(w,tk.Toplevel)]
text=next(w for w in dialogs[-1].winfo_children() if isinstance(w,tk.Text)).get('1.0','end')
assert '尚无可靠的管间尺寸对比' in text
assert text.index('管道之间的轴心间距')<text.index('相机坐标诊断')
dialogs[-1].destroy()
(OUT/'gui.json').write_text(json.dumps({'layouts':records,'candidate_overlay_count_per_eye':4,
    'relative_dialog_precedes_camera_diagnostics':True},ensure_ascii=False,indent=2),encoding='utf-8')
owner.close();root.destroy()
print('GUI_LAYOUT_CANDIDATES_AND_RELATIVE_DIALOG_PASSED')
