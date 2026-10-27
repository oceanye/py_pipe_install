"""Audit the operator's 17:05 capture and replay with only the pipe ROI reset."""
from pathlib import Path
from unittest.mock import Mock, patch
import hashlib
import json
from collections import Counter
import runpy
import sys
import time
import tkinter as tk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.gui import _start_application

SOURCE = ROOT/'outputs/elevation_workbench/captures/elevation-20261027-170823-c8d2285a/manifest.json'
OUTPUT = ROOT/'outputs/operation-inspection-20261027-1708'


def summarize(report):
    stages = report.get('section_evaluations') or [report.get('section_evaluation', {})]
    temporal = report.get('temporal_fusion') or {}
    return {'counts': report['counts'], 'position_counts': report['position_assessment']['counts'],
            'ground_status': report['ground_check']['status'],
            'ground_correction': report['ground_check'].get('correction_status'),
            'temporal': {k: temporal.get(k) for k in ('status', 'requested_frames', 'accepted_frames', 'reason_codes')},
            'frame_failure_counts': dict(Counter(code for frame in temporal.get('frames', [])
                                                for code in frame.get('reason_codes', []))),
            'stages': [{'zone_id': s.get('zone_id'), 'status': s.get('status'),
                        'conditional_section_count': s.get('conditional_section_count', 0),
                        'aggregation_band': s.get('aggregation_band'),
                        'candidates': [{k: c.get(k) for k in ('candidate_id', 'left_region_px', 'right_region_px',
                                         'dimension_status', 'reason_codes')} | {
                            'trial_diameter_mm': (c.get('fit') or {}).get('trial_diameter_mm'),
                            'accepted_diameter_mm': (c.get('fit') or {}).get('accepted_diameter_mm')}
                            for c in s.get('candidates', [])],
                        'blue_boundaries': [{'role': role, 'candidate_id': b['candidate_id'],
                                            'boundary_stable': b['boundary_stable']}
                                           for role, rows in s.get('views', {}).items() for b in rows
                                           if b.get('appearance') == 'hue-105']} for s in stages]}


def main():
    original = json.loads(SOURCE.with_name('report.json').read_text(encoding='utf8'))
    bound = [SOURCE, SOURCE.with_name('report.json'), SOURCE.parent/'photos/left.png',
             SOURCE.parent/'photos/right.png', ROOT/'outputs/elevation_workbench/last_scene.json',
             ROOT/'outputs/measurement_workbench/calibration_current.json']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in bound}
    root = tk.Tk()
    with patch('pipe_twin.workbench_profile.load_profile', return_value=(None, None)):
        dialog = _start_application(root, SOURCE)
    dialog.app.messagebox = Mock()
    dialog.output_root = OUTPUT
    result = {'source': str(SOURCE.relative_to(ROOT)), 'original': summarize(original)}
    try:
        root.geometry('1024x720'); root.update()
        layout = runpy.run_path(str(ROOT/'test_reports/20261027-gui-cleanup/verify_gui.py'))['layout']
        result['layout'] = layout(root) | {'preview_height': dialog.views['left'].canvas.winfo_height(),
            'scope_y': dialog.region_button.winfo_rooty(), 'evaluate_y': dialog.run_button.winfo_rooty()}
        assert not result['layout']['clipped_controls']
        assert result['layout']['preview_height'] >= 180
        assert result['layout']['evaluate_y'] > result['layout']['scope_y']
        dialog.confirmed.set(True)
        dialog.submit(True)
        assert not dialog.busy and dialog.app.messagebox.showerror.called
        result['original_configuration_blocked'] = dialog.assessment_region_problem()
        dialog.app.messagebox.reset_mock()
        result['replays'] = {}
        for mode in ('full_frame_burst', 'full_frame_single_diagnostic'):
            dialog.load_session(SOURCE)
            dialog.clear_assessment_regions()
            assert dialog.registration_settings['ground_region']
            assert dialog.registration_settings['aggregation_band']
            if mode == 'full_frame_single_diagnostic':
                dialog.temporal_frames = []
            dialog.history.set(False); dialog.confirmed.set(True)
            dialog.submit(True)
            deadline, previous = time.monotonic()+360, ''
            while dialog.busy and time.monotonic() < deadline:
                root.update(); time.sleep(.03)
                if dialog.message.get() != previous:
                    previous = dialog.message.get(); print(mode, previous, flush=True)
            assert not dialog.busy, 'evaluation timed out'
            dialog.app.messagebox.showerror.assert_not_called()
            result['replays'][mode] = {'manifest': str(dialog.last_manifest.relative_to(ROOT)),
                                      'summary': summarize(dialog.report)}
    finally:
        dialog.close()
    for p, digest in hashes.items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest() == digest, p
    result['source_hashes_unchanged'] = hashes
    output = Path(__file__).with_name('operation-validation.json')
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    for key, replay in result['replays'].items():
        summary = replay['summary']
        print(key, 'sections', [s['conditional_section_count'] for s in summary['stages']],
              'ground', summary['ground_correction'], 'temporal', summary['temporal'], flush=True)
    print(output, flush=True)


if __name__ == '__main__':
    main()
