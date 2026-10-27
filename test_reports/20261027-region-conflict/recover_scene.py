"""Recover the stopped GUI's latest verified pair and selection events.

The completed ground inspection binds the last pair's hashes and timestamps.
The interrupted burst has no persisted frame metadata: do not fabricate it.
"""
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

SOURCE = ROOT/'outputs/elevation_workbench/captures/elevation-20261027-170823-c8d2285a/manifest.json'
INSPECTION = ROOT/'outputs/elevation_workbench/ground_inspections/tilt-20261027-174748-4f02e075/inspection.json'
RUN_ID = 'f0101579971444a6acec7316e995e61c'
OUTPUT = ROOT/'outputs/region-conflict-recovery-20261027'


def main():
    inspection = json.loads(INSPECTION.read_text(encoding='utf8'))
    events = []
    for line in (ROOT/'logs/pipe_twin.log.jsonl').read_text(encoding='utf8').splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get('run_id') == RUN_ID:
            events.append(row)
    band = next(r['fields']['aggregation_band'] for r in reversed(events) if r['event'] == 'elevation_axial_band_changed')
    new_region = next(r['fields'] for r in reversed(events) if r['event'] == 'elevation_left_region_added')
    assert new_region['zone_id'] == 'Z03'
    sources = inspection['sources']
    for source in sources.values():
        assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest() == source['sha256']
    bound = [SOURCE, SOURCE.with_name('report.json'), INSPECTION,
             ROOT/'outputs/measurement_workbench/calibration_current.json']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in bound}
    resume = ROOT/'outputs/elevation_workbench/last_scene.json'
    previous_resume = resume.read_bytes()
    root = tk.Tk(); root.withdraw()
    with patch('pipe_twin.workbench_profile.load_profile', return_value=(None, None)):
        d = _start_application(root, SOURCE)
    d.output_root = OUTPUT; d.app.messagebox = Mock()
    try:
        assert d.current_calibration()['calibration_id'] == inspection['calibration']['calibration_id']
        for role, source in sources.items():
            d.fields[role].set(source['path'])
            d.fields[role+'_time'].set(source['captured_at'])
        d.load_images()
        d.temporal_frames = []
        # These are the host timestamps captured by StereoCameraDialog and
        # copied verbatim into the completed ground inspection, not file mtimes.
        d.timestamp_sources = {role: 'HOST_SYSTEM_CLOCK' for role in sources}
        d.camera_capture_provenance = {}
        d.set_axial_band(band)
        d.set_ground_region(inspection['ground_check']['selection']['rect_px'])
        d.add_left_region(new_region['roi_rect_px'])
        assert [z['zone_id'] for z in d.zone_settings['zones']] == ['Z01', 'Z02', 'Z03']
        assert d.conflicting_assessment_zones() == ['Z01', 'Z02']
        registration = copy.deepcopy(d.registration_settings)
        root.geometry('1024x720'); root.deiconify(); root.update()
        layout = runpy.run_path(str(ROOT/'test_reports/20261027-gui-cleanup/verify_gui.py'))['layout'](root)
        assert not layout['clipped_controls'], layout
        d.region_conflict_button.invoke()
        assert d.registration_settings == registration
        assert d.conflicting_assessment_zones() == []
        assert [(z['zone_id'], z['enabled']) for z in d.zone_settings['zones']] == [
            ('Z01', False), ('Z02', False), ('Z03', True)]
        root.withdraw()
        d.confirmed.set(True); d.history.set(False)
        d.submit(True)
        deadline, previous = time.monotonic()+240, ''
        while d.busy and time.monotonic() < deadline:
            root.update(); time.sleep(.03)
            if previous != d.message.get():
                previous = d.message.get(); print(previous, flush=True)
        assert not d.busy
        d.app.messagebox.showerror.assert_not_called()
        path = d.last_manifest
        report = d.report
        assert path != SOURCE
        d.load_session(path)
        assert d.assessment_region_problem() is None
        assert d.registration_settings == registration
        assert d.temporal_frames == []
        payload = json.loads(path.read_text(encoding='utf8'))
        for role, record in payload['capture']['capture_groups'][-1]['views'].items():
            assert record['sha256'] == sources[role]['sha256']
            assert record['captured_at'] == sources[role]['captured_at']
        result = {'source': str(SOURCE.relative_to(ROOT)), 'inspection': str(INSPECTION.relative_to(ROOT)),
                  'new_manifest': str(path.relative_to(ROOT)), 'recovered_pair_only': True,
                  'captured_at': sources['left']['captured_at'], 'zone_settings': d.zone_settings,
                  'registration_preserved': registration, 'layout_with_repair_button': layout,
                  'counts': report['counts'], 'position_counts': report['position_assessment']['counts'],
                  'sections': [{'zone_id': s['zone_id'], 'count': s.get('conditional_section_count', 0),
                                'status': s.get('status')} for s in report.get('section_evaluations', [])],
                  'source_hashes_unchanged': hashes}
    finally:
        d.close()
    for name, digest in hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest
    if '--publish-resume' in sys.argv:
        assert resume.read_bytes() == previous_resume, 'GUI resume changed during recovery'
        (OUTPUT/'previous_last_scene.json').write_bytes(previous_resume)
        atomic_write_text(resume, json.dumps({'manifest_path': str(path)}, ensure_ascii=False))
        result['default_resume_updated'] = True
    Path(__file__).with_name('recovery-validation.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
