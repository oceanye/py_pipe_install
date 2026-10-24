"""Exercise production GUI tilt inspection on the newly captured fifth pair."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import json
import sys
import time
import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.elevation_gui import ElevationCaptureDialog


def main():
    output = ROOT/'outputs/tilt-effect-check-20261024'
    capture = json.loads((output/'raw-20261024-083322/capture.json').read_text(encoding='utf-8'))
    pair = capture['captures'][-1]
    root = tk.Tk(); root.withdraw()
    app = SimpleNamespace(root=root, tk=tk, ttk=ttk, messagebox=Mock(), filedialog=Mock())
    with patch('pipe_twin.workbench_profile.load_profile', return_value=(None, None)):
        dialog = ElevationCaptureDialog(app, output_root=output/'gui', restore=False)
        dialog.window.withdraw()
        try:
            dialog.load_session(ROOT/'outputs/tilt-inspection-20261023/captures/elevation-20261023-084738-cbfc8998/manifest.json')
            for role in ('left', 'right'):
                dialog.fields[role].set(str(output/'rectified'/pair[role]['path']))
                dialog.fields[role+'_time'].set(pair[role]['captured_at'])
            dialog.load_images()
            dialog.registration_settings['ground_region'] = {
                'image_size': [1920, 1080], 'rect_px': [700, 40, 850, 200],
                'coordinate_space': 'rectified_left', 'selection_source': 'ASSISTANT_DIAGNOSTIC'}
            dialog.confirmed.set(True)
            dialog.inspect_ground()
            deadline = time.monotonic()+40
            while dialog.busy and time.monotonic() < deadline:
                root.update(); time.sleep(.03)
            assert not dialog.busy
            app.messagebox.showerror.assert_not_called()
            check = dialog.ground_inspection
            assert check['status'] == 'GROUND_PLANE_INCONSISTENT'
            assert not check['pipe_axis_constrained_by_ground']
            window = dialog.ground_inspection_window; root.update()
            preview = next(w for w in window.winfo_children() if isinstance(w, tk.Canvas))
            label = preview.find_withtag('ground-preview-angle')[0]
            label_text = preview.itemcget(label, 'text')
            assert '候选 / 不用于修正' in label_text, repr(label_text)
            path = max((output/'gui/ground_inspections').glob('*/inspection.json'), key=lambda p: p.stat().st_mtime)
            saved = json.loads(path.read_text(encoding='utf-8'))
            assert saved['sources']['left']['captured_at'] == pair['left']['captured_at']
            reference = json.loads((output/'checks/pair-05/rear_large/ground_check.json').read_text(encoding='utf-8'))
            assert check == reference
            result = {'new_camera_pair_used': True, 'capture_time': pair['left']['captured_at'],
                      'production_gui_worker_passed': True, 'ground_only_preview_created': True,
                      'candidate_label': label_text, 'saved_inspection': str(path.relative_to(ROOT)),
                      'ground_status': check['status'], 'pipe_axis_constrained_by_ground': False,
                      'standalone_matches_five_frame_audit': True}
            (Path(__file__).parent/'gui-validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            print(json.dumps(result, ensure_ascii=False), flush=True)
        finally:
            dialog.close(); root.update(); root.destroy()


if __name__ == '__main__':
    main()
