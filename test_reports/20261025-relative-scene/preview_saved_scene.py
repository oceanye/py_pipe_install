"""Open the verified saved result in the production GUI for operator review."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import sys
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.elevation_gui import ElevationCaptureDialog

manifest = Path(sys.argv[1]).resolve()
report = json.loads((manifest.parent/'report.json').read_text(encoding='utf8'))
assert hashlib.sha256(manifest.read_bytes()).hexdigest() == report['inputs']['manifest']
root = tk.Tk(); root.withdraw()
app = SimpleNamespace(root=root, tk=tk, ttk=ttk, messagebox=messagebox, filedialog=filedialog)
dialog = ElevationCaptureDialog(app, output_root=manifest.parents[2], restore=False)
dialog.load_session(manifest)  # Validates all bound image/model/calibration snapshots.
dialog.report = report
dialog.results = {r['pipe_id']: r for r in report['pipes']}
dialog.refresh_table()
dialog.summary.set('已恢复保存的实拍评估；现场模型中的虚线圆为条件估计。')
dialog.open_model_viewer()
viewer = dialog.model_viewer
viewer.window.geometry('1320x820+30+30')
viewer.window.lift()
root.update()
viewer.window.attributes('-topmost', True)
viewer.window.after(1500, lambda: viewer.window.attributes('-topmost', False) if viewer.window.winfo_exists() else None)
dialog.window.protocol('WM_DELETE_WINDOW', lambda: (dialog.close(), root.destroy()))
root.mainloop()
