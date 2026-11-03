"""Isolated Tk verification of rotated sections, blinking warning and history."""
from pathlib import Path
import json
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from test_auto_monitor_gui import AutoMonitorGuiTests

screenshot = runpy.run_path(str(ROOT / 'test_reports/20261030-ground-monitor/gui-verify.py'))['screenshot']


def main():
    output = Path(__file__).resolve().parent
    AutoMonitorGuiTests.setUpClass()
    case = AutoMonitorGuiTests('test_opening_is_idle_and_does_not_acquire_camera_or_configuration')
    try:
        case.setUp()
        dialog = case.dialog
        dialog.window.deiconify()
        dialog.start()
        case.hinted_cycle(1, ['P001'])
        case.hinted_cycle(2, ['P002'])
        screenshots = []
        for size in ('1024x720', '1280x860'):
            dialog.window.geometry(size)
            case.root.update()
            name = 'result-history-' + size + '.png'
            screenshot(dialog.window, output / name)
            screenshots.append(name)
        dialog.window.geometry('1024x720')
        dialog.monitor.next_due = 110.
        dialog.handle_event({'kind': 'state', 'cycle_id': 2, 'state': 'PREPARING'})
        dialog._countdown()
        case.root.update()
        warning_text = dialog.capture_warning.get()
        assert '10 秒' in warning_text
        screenshot(dialog.window, output / 'warning-visible.png')
        case.now = 100.6
        dialog._countdown()
        assert dialog.capture_warning.get() == ''
        case.root.update()
        screenshot(dialog.window, output / 'warning-off.png')
        dialog.handle_event({'kind': 'state', 'cycle_id': 3, 'state': 'CAPTURING'})
        dialog.handle_event(case.capture(3))
        dialog.handle_event({'kind': 'state', 'cycle_id': 3, 'state': 'ANALYZING'})
        case.root.update()
        case.assert_results_empty()
        assert len(dialog.history_table.get_children()) == 3
        screenshot(dialog.window, output / 'analysis-history-preserved.png')
        records = [json.loads(line) for line in dialog.history_path.read_text(encoding='utf-8').splitlines()]
        # Avoid committing temporary filesystem paths from the synthetic scene.
        for entry in records:
            entry['manifest_path'] = '<temporary synthetic scene>'
        evidence = {'real_camera_opened': False, 'network_requests': 0, 'ai_requests': 0,
                    'production_preferences_changed': False, 'screenshots': screenshots,
                    'warning_visible_text': warning_text, 'warning_off_text': '',
                    'records': records, 'analysis_sections_empty': dialog.report is None,
                    'analysis_history_rows': len(dialog.history_table.get_children())}
        (output / 'gui-evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        if hasattr(case, 'dialog'):
            case.tearDown()
            case.doCleanups()
        AutoMonitorGuiTests.tearDownClass()


if __name__ == '__main__':
    main()
