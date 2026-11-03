"""Exercise both feature GUI entries with temporary profiles and no camera."""
from pathlib import Path
import json
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))
from test_elevation_gui import ElevationGuiTests

screenshot = runpy.run_path(str(ROOT / 'test_reports/20261030-ground-monitor/gui-verify.py'))['screenshot']


def main():
    output = Path(__file__).resolve().parent
    ElevationGuiTests.setUpClass()
    case = ElevationGuiTests('test_settings_do_not_consume_photo_space_and_block_edits_while_evaluating')
    try:
        case.setUp()
        dialog = case.dialog
        dialog.window.deiconify()
        dialog.window.geometry('1024x720')
        case.root.update()
        entries = {}
        for name, button in [('settings', dialog.settings_button), ('monitor', dialog.monitor_button)]:
            assert button.winfo_viewable() and 'disabled' not in button.state()
            x = button.winfo_rootx() - dialog.window.winfo_rootx()
            assert x >= 0 and x + button.winfo_width() <= dialog.window.winfo_width()
            entries[name] = {'text': button.cget('text'), 'x': x, 'width': button.winfo_width(), 'enabled': True}
        screenshot(dialog.window, output / 'main-gui-entries-1024x720.png')
        dialog.settings_button.invoke()
        dialog.capture_settings.select(dialog.settings_pages['runtime'])
        case.root.update()
        assert dialog.settings_window.winfo_viewable() and dialog.runtime.mode_box.winfo_viewable()
        modes = list(dialog.runtime.mode_box.cget('values'))
        assert modes == ['服务器端', '前端', '本地运算']
        applied = []
        for label, mode in [('服务器端', 'server'), ('前端', 'client'), ('本地运算', 'local')]:
            dialog.runtime.mode_label.set(label)
            dialog.runtime.apply_button.invoke()
            case.root.update()
            assert dialog.runtime.mode == mode
            applied.append(mode)
        screenshot(dialog.settings_window, output / 'runtime-settings-entry.png')
        dialog.hide_settings()
        dialog.monitor_button.invoke()
        case.root.update()
        assert dialog.auto_monitor.window.winfo_viewable() and not dialog.auto_monitor.active
        assert dialog.auto_monitor.start_button.winfo_viewable()
        assert dialog.auto_monitor.stop_button.winfo_viewable()
        (output / 'entries-evidence.json').write_text(json.dumps({
            'passed': True, 'entries': entries, 'runtime_choices': modes, 'applied_modes': applied,
            'monitor_opens_from_button': True, 'camera_opened': False, 'server_started': False,
            'network_requests': 0, 'ai_requests': 0, 'production_preferences_written': False,
        }, ensure_ascii=False, indent=2), encoding='utf-8')
    finally:
        if hasattr(case, 'dialog'):
            case.tearDown()
            case.doCleanups()
        ElevationGuiTests.tearDownClass()


if __name__ == '__main__':
    main()
