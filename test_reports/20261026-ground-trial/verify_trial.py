"""Run explicit candidate-ground trial through the production GUI worker."""
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
from pipe_twin.ground_visualization import tilt_label_lines


def main():
    source = ROOT/'outputs/relative-scene-20261025/settled-gui/captures/elevation-20261025-122438-e4146031/manifest.json'
    calibration = ROOT/'outputs/measurement_workbench/calibration_current.json'
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, source.parent/'report.json', calibration)}
    root = tk.Tk(); root.withdraw()
    app = SimpleNamespace(root=root, tk=tk, ttk=ttk, messagebox=Mock(), filedialog=Mock())
    with patch('pipe_twin.workbench_profile.load_profile', return_value=(None, None)):
        dialog = ElevationCaptureDialog(app, output_root=ROOT/'outputs/ground-trial-20261026/gui', restore=False)
        dialog.window.withdraw()
        try:
            dialog.load_session(source)
            dialog.ground_candidate_trial.set(True); dialog._ground_trial_changed()
            dialog.history.set(False); dialog.confirmed.set(True)
            dialog.submit(True)
            deadline, previous = time.monotonic()+300, ''
            while dialog.busy and time.monotonic() < deadline:
                root.update(); time.sleep(.03)
                if dialog.message.get() != previous:
                    previous = dialog.message.get(); print(previous, flush=True)
            assert not dialog.busy, 'TRIAL_TIMED_OUT'
            app.messagebox.showerror.assert_not_called()
            report = dialog.report; assert report is not None
            stage = report['section_evaluation']; ground = report['ground_check']
            assert ground['correction_status'] == 'TRIAL_APPLIED'
            assert ground['status'] == 'GROUND_PLANE_INCONSISTENT'
            assert stage['ground_correction']['boundary_consistency_bypassed']
            assert stage['temporal_sections']['sampled_frames'] == 5
            assert report['position_assessment']['counts']['IN_POSITION'] == 0
            for row in dialog.solver_rows.values():
                assert '倾角试验' in row['values'][-1]
                assert '同图对照外径' in row['detail']
            assert '试验' in '\n'.join(tilt_label_lines(ground, compact=True))
            saved_manifest = str(dialog.last_manifest.relative_to(ROOT))
            assert json.loads(dialog.last_manifest.read_text(encoding='utf8'))['analysis']['elevation_auto']['registration']['ground_candidate_trial']
            result = {'source_manifest': str(source.relative_to(ROOT)), 'trial_manifest': saved_manifest,
                'replay_only_no_new_capture': True, 'source_hashes': hashes,
                'ground_check_status': ground['status'], 'ground_correction_status': ground['correction_status'],
                'ground_correction': stage['ground_correction'], 'comparison': stage['ground_trial_comparison'],
                'temporal_sections': stage['temporal_sections'], 'solver_status': report['solver_status'],
                'installation_counts': report['counts'], 'position_counts': report['position_assessment']['counts'],
                'gui_worker_and_rows_checked': True}
        finally:
            dialog.close(); root.update(); root.destroy()
    for name, digest in hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    (Path(__file__).parent/'field-validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'manifest': saved_manifest, 'comparison': result['comparison']},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
