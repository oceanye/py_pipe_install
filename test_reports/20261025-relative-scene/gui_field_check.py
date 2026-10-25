"""Run production Save/Evaluate on the fresh burst, then inspect real widgets."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import hashlib
import json
import sys
import time
import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.elevation_gui import ElevationCaptureDialog


def main():
    output = ROOT/'outputs/relative-scene-20261025'
    prefix = sys.argv[1] if len(sys.argv) > 1 else 'fresh'
    capture = json.loads((output/f'{prefix}-capture/capture.json').read_text(encoding='utf8'))
    root = tk.Tk(); root.withdraw()
    app = SimpleNamespace(root=root, tk=tk, ttk=ttk, messagebox=Mock(), filedialog=Mock())
    with patch('pipe_twin.workbench_profile.load_profile', return_value=(None, None)):
        dialog = ElevationCaptureDialog(app, output_root=output/f'{prefix}-gui', restore=False)
        dialog.window.withdraw()
        try:
            dialog.load_session(ROOT/'outputs/tilt-inspection-20261023/captures/elevation-20261023-084738-cbfc8998/manifest.json')
            saved_settings = dict(dialog.registration_settings)
            last = capture['captures'][-1]
            for role in ('left', 'right'):
                dialog.fields[role].set(str(output/f'{prefix}-{len(capture["captures"]):02d}-{role}.png'))
                dialog.fields[role+'_time'].set(last[role]['captured_at'])
            dialog.load_images()
            dialog.registration_settings = saved_settings
            dialog.camera_capture_provenance = last['provenance']
            dialog.timestamp_sources = {r: last['timestamp_source'] for r in ('left', 'right')}
            dialog.temporal_frames = []
            for index, frame in enumerate(capture['captures'], 1):
                paths = {r: output/f'{prefix}-{index:02d}-{r}.png' for r in ('left', 'right')}
                dialog.temporal_frames.append({**{r+'_path': str(paths[r]) for r in paths},
                    **{r+'_time': frame[r]['captured_at'] for r in paths},
                    'sha256': {r: hashlib.sha256(paths[r].read_bytes()).hexdigest() for r in paths},
                    'timestamp_sources': {r: frame['timestamp_source'] for r in paths},
                    'camera_capture_provenance': frame['provenance']})
            dialog.history.set(False)
            dialog.confirmed.set(True)
            dialog.submit(True)
            deadline = time.monotonic()+220
            while dialog.busy and time.monotonic() < deadline:
                root.update(); time.sleep(.03)
            assert not dialog.busy, 'GUI_ANALYSIS_TIMEOUT'
            app.messagebox.showerror.assert_not_called()
            assert dialog.report is not None
            dialog.open_model_viewer(); root.update()
            viewer = dialog.model_viewer
            viewer.window.geometry('1320x820+30+30'); viewer.window.lift(); root.update()
            count = dialog.report['section_evaluation']['conditional_section_count']
            assert len(viewer.scene_hits) == count
            assert viewer.section_canvas.find_withtag('installation-legend')
            assert viewer.scene_canvas.find_withtag('scene-limitations')
            assert not viewer.section_canvas.find_withtag('state:IN_POSITION')
            paths = dialog.report['evidence_files']
            assert any(p.endswith('section-model.json') for p in paths)
            assert any(p.endswith('index.html') for p in paths)
            result = {'manifest': str(dialog.last_manifest.relative_to(ROOT)),
                      'fresh_capture_time': last['left']['captured_at'], 'scene_count': count,
                      'fresh_burst_fusion': dialog.report['temporal_fusion'],
                      'installation_counts': dialog.report['counts'],
                      'position_counts': dialog.report['position_assessment']['counts'],
                      'automatic_section_artifacts_written': True,
                      'scene_and_cad_widgets_checked': True}
            (Path(__file__).parent/f'{prefix}-gui-validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
            print(json.dumps(result, ensure_ascii=False), flush=True)
        finally:
            dialog.close(); root.update(); root.destroy()


if __name__ == '__main__':
    main()
