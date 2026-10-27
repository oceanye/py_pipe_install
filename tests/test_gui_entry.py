import json
from unittest.mock import patch

import pytest


def test_default_entry_uses_root_as_field_window_and_closes_event_loop(tmp_path):
    import tkinter as tk
    from pipe_twin.gui import _start_application
    from pipe_twin.elevation_gui import ElevationCaptureDialog
    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(str(error))
    try:
        with patch('pipe_twin.workbench_profile.load_profile', return_value=(None, None)), patch('pipe_twin.elevation_gui.ROOT', tmp_path):
            dialog = _start_application(root)
        assert isinstance(dialog, ElevationCaptureDialog) and dialog.window is root
        assert dialog.settings_window.state() == 'withdrawn'
        root.after(10, dialog.close)
        root.mainloop()
        assert dialog.closed
    finally:
        try:
            root.destroy()
        except tk.TclError:
            pass


def test_explicit_legacy_manifest_still_routes_to_compatible_viewer(tmp_path):
    from pipe_twin.gui import _start_application
    path = tmp_path/'manifest.json'
    path.write_text(json.dumps({'analysis': {'mode': 'historical'}}), encoding='utf8')
    root = object()
    with patch('pipe_twin.gui._PipeTwinApplication') as legacy:
        result = _start_application(root, path)
        assert result is legacy.return_value
        legacy.assert_called_once_with(root, path, None)
