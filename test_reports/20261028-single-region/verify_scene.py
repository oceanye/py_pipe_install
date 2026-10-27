"""Replay the saved pair through the single-region GUI; never recapture images."""
from pathlib import Path
from unittest.mock import Mock, patch
import copy
import hashlib
import json
import runpy
import sys
import time
import tkinter as tk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.gui import _start_application
from pipe_twin.pipeline import atomic_write_text

SOURCE = ROOT/'outputs/region-conflict-recovery-20261027/captures/elevation-20261027-175505-a2a82c73/manifest.json'
OUTPUT = ROOT/'outputs/single-region-20261028'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    original = json.loads(SOURCE.read_text(encoding='utf8'))
    baseline = json.loads(SOURCE.with_name('report.json').read_text(encoding='utf8'))
    protected = [SOURCE, SOURCE.with_name('report.json'),
                 ROOT/'outputs/measurement_workbench/calibration_current.json']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    resume = ROOT/'outputs/elevation_workbench/last_scene.json'
    previous_resume = resume.read_bytes()
    helpers = runpy.run_path(str(ROOT/'test_reports/20261027-gui-cleanup/verify_gui.py'))
    root = tk.Tk(); root.withdraw()
    with patch('pipe_twin.workbench_profile.load_profile', return_value=(None, None)):
        d = _start_application(root, SOURCE)
    d.output_root = OUTPUT
    d.app.messagebox = Mock()
    try:
        assert [z['zone_id'] for z in d.zone_settings['zones']] == ['Z03']
        assert d.zone_settings['zones'][0]['roi_rect_px'] == [1112, 295, 439, 461]
        assert d.scope_label.get() == '单一区域'
        assert d.assessment_region_problem() is None
        registration = copy.deepcopy(d.registration_settings)
        root.geometry('1024x720'); root.deiconify(); root.update()
        layouts = {'main': helpers['layout'](root)}
        d.open_settings('model'); root.update()
        layouts['model_settings'] = helpers['layout'](d.settings_window)
        for layout in layouts.values():
            assert not layout['clipped_controls'], layout
            assert '分区管理' not in layout['buttons']
        d.hide_settings(); root.withdraw()
        d.confirmed.set(True); d.history.set(False)
        d.submit(True)
        deadline, previous = time.monotonic()+240, ''
        while d.busy and time.monotonic() < deadline:
            root.update(); time.sleep(.03)
            if previous != d.message.get():
                previous = d.message.get(); print(previous, flush=True)
        assert not d.busy, 'GUI evaluation timed out'
        d.app.messagebox.showerror.assert_not_called()
        path, report = d.last_manifest, d.report
        assert path != SOURCE
        compared = ['counts', 'section_evaluations', 'ground_check', 'position_assessment']
        for key in compared:
            assert helpers['without_capture_ids'](report[key]) == helpers['without_capture_ids'](baseline[key]), key
        d.load_session(path)
        assert d.registration_settings == registration
        assert [z['zone_id'] for z in d.zone_settings['zones']] == ['Z03']
        assert d.temporal_frames == []
        saved = json.loads(path.read_text(encoding='utf8'))
        source_views = original['capture']['capture_groups'][-1]['views']
        saved_views = saved['capture']['capture_groups'][-1]['views']
        for role in ('left', 'right'):
            for key in ('sha256', 'captured_at', 'timestamp_source'):
                assert saved_views[role][key] == source_views[role][key]
        result = {
            'source': str(SOURCE.relative_to(ROOT)), 'new_manifest': str(path.relative_to(ROOT)),
            'zone_settings': d.zone_settings, 'layouts': layouts,
            'captured_at': source_views['left']['captured_at'], 'replayed_pair_only': True,
            'unchanged_result_fields': compared, 'registration_preserved': registration,
            'counts': report['counts'],
            'conditional_section_count': sum(s['conditional_section_count'] for s in report['section_evaluations']),
            'source_hashes_unchanged': hashes,
        }
    finally:
        d.close()
    for name, expected in hashes.items():
        assert digest(ROOT/name) == expected, name
    if '--publish-resume' in sys.argv:
        assert resume.read_bytes() == previous_resume, 'Default scene changed during validation'
        backup = OUTPUT/'previous_last_scene.json'
        if not backup.exists():
            backup.write_bytes(previous_resume)
        atomic_write_text(resume, json.dumps({'manifest_path': str(path)}, ensure_ascii=False))
        result['default_resume_updated'] = True
    Path(__file__).with_name('field-validation.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
