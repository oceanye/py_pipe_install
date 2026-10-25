"""Replay the preserved Oct 25 burst through production evaluation and real Tk."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import hashlib
import json
import sys
import tkinter as tk
from tkinter import ttk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.elevation_auto import analyze_elevation_auto_manifest
from pipe_twin.elevation_gui import ElevationCaptureDialog


def main():
    source = ROOT/'outputs/relative-scene-20261025/settled-gui/captures/elevation-20261025-122438-e4146031/manifest.json'
    output = ROOT/'outputs/stable-sections-20261026/production'
    output.mkdir(parents=True, exist_ok=True)
    calibration = ROOT/'outputs/measurement_workbench/calibration_current.json'
    original = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, source.parent/'report.json', calibration)}
    if '--reuse-report' in sys.argv:
        report = json.loads((output/'report.json').read_text(encoding='utf8'))
    else:
        report = analyze_elevation_auto_manifest(source, report_output_path=output/'report.json',
            evidence_dir=output/'evidence', progress=lambda value: print(value, flush=True))
    for p, digest in original.items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest
    section = report['section_evaluation']
    assert section['conditional_section_count'] == 4
    assert section['temporal_sections']['status'] == 'CHECKED'
    assert section['temporal_sections']['sampled_frames'] == 5
    assert all(c['temporal_consistency']['solved_frames'] == 5 for c in section['candidates'])
    root = tk.Tk(); root.withdraw()
    app = SimpleNamespace(root=root, tk=tk, ttk=ttk, messagebox=Mock(), filedialog=Mock())
    with patch('pipe_twin.workbench_profile.load_profile', return_value=(None, None)):
        dialog = ElevationCaptureDialog(app, output_root=output/'gui', restore=False)
        dialog.window.withdraw()
        try:
            dialog.load_session(source)
            dialog.report = report
            dialog.results = {r['pipe_id']: r for r in report['pipes']}
            dialog.refresh_table(); root.update()
            assert len(dialog.solver_tree.get_children()) == len(report['solver_status'])
            for row in report['solver_status']:
                dialog.solver_tree.selection_set(row['key']); root.update()
                assert dialog.selected_solver_row()['candidate_id'] == row['candidate_id']
                assert dialog.views['left'].canvas.find_withtag('solver-selection')
            dialog.open_model_viewer(); root.update()
            assert len(dialog.model_viewer.scene_hits) == 4
            assert not dialog.model_viewer.section_canvas.find_withtag('state:IN_POSITION')
            app.messagebox.showerror.assert_not_called()
        finally:
            dialog.close(); root.update(); root.destroy()
    evidence = {'source_manifest': str(source.relative_to(ROOT)), 'replay_only_no_new_capture': True,
        'source_hashes_unchanged': original, 'report_path': str((output/'report.json').relative_to(ROOT)),
        'conditional_section_count': section['conditional_section_count'],
        'temporal_sections': section['temporal_sections'], 'solver_status': report['solver_status'],
        'installation_counts': report['counts'], 'position_counts': report['position_assessment']['counts'],
        'real_tk_status_selection_and_scene_checked': True,
        'evidence_files': report['evidence_files']}
    (Path(__file__).parent/'field-validation.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps({k: evidence[k] for k in ('conditional_section_count','installation_counts','position_counts')},ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
