"""Display an archived production result, then verify the next cycle clears it.

No hardware, network, AI or user preferences are accessed. Pass the successful
verify_monitor.py summary JSON to replay one source-bound result in real Tk.
"""
from pathlib import Path
import json
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from test_auto_monitor_gui import AutoMonitorGuiTests
from pipe_twin.elevation_dataset import load_elevation_dataset

screenshot = runpy.run_path(str(ROOT / 'test_reports/20261030-ground-monitor/gui-verify.py'))['screenshot']


def main():
    evidence_path = Path(sys.argv[1])
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    source = evidence['cycles'][0]
    manifest_path = Path(source['manifest_path'])
    loaded = load_elevation_dataset(manifest_path)
    report = json.loads(Path(source['report_path']).read_text(encoding='utf-8'))
    output = Path(__file__).resolve().parent
    AutoMonitorGuiTests.setUpClass()
    case = AutoMonitorGuiTests('test_opening_is_idle_and_does_not_acquire_camera_or_configuration')
    try:
        case.setUp()
        case.configuration['dataset_arguments'].update(pipe_specs=loaded['pipe_specs'],
            registration_settings=loaded['registration_settings'])
        dialog = case.dialog
        dialog.window.deiconify()
        dialog.start()
        dialog.handle_event({'kind': 'state', 'cycle_id': 1, 'state': 'CAPTURING'})
        captured_at = loaded['manifest']['capture']['capture_groups'][0]['views']['left']['captured_at']
        event = {'kind': 'capture', 'cycle_id': 1, 'manifest_path': str(manifest_path), 'captured_at': captured_at}
        dialog.handle_event(event)
        dialog.handle_event(dict(event, kind='result', report=report))
        case.root.update()
        rows = [list(dialog.table.item(item, 'values')) for item in dialog.table.get_children()]
        assert dialog.report is report and len(rows) == 4, rows
        assert dialog.section_canvas.find_withtag('scene-pipe')
        assert len(dialog.cad_canvas.find_withtag('state:SUSPECTED_INSTALLED')) == 4
        local_images = ROOT / 'outputs' / 'monitor-validation-local'
        local_images.mkdir(parents=True, exist_ok=True)
        screenshot(dialog.window, local_images / 'archived-real-result.png')
        dialog.handle_event({'kind': 'state', 'cycle_id': 2, 'state': 'CAPTURING'})
        case.assert_results_empty()
        # A delayed success for the archived first cycle cannot restore its model.
        dialog.handle_event(dict(event, kind='result', report=report))
        case.assert_results_empty()
        case.root.update()
        screenshot(dialog.window, output / 'archived-next-cycle-cleared.png')
        (output / 'real-gui-evidence.json').write_text(json.dumps({
            'source': str(evidence_path), 'real_camera_opened': False, 'network_requests': 0,
            'ai_requests': 0, 'cad_pipe_count': len(dialog.pipes), 'result_rows': rows,
            'next_cycle_section_empty': dialog.report is None,
            'late_previous_result_rejected': True, 'passed': True,
        }, ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        if hasattr(case, 'dialog'):
            case.tearDown()
            case.doCleanups()
        AutoMonitorGuiTests.tearDownClass()


if __name__ == '__main__':
    main()
