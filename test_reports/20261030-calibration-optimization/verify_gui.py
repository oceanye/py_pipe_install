"""Exercise background solve and save in a separate profile/calibration directory."""
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
from pipe_twin.calibration_wizard import ChessboardWizardDialog, read_calibration_diagnostic


def main():
    out = ROOT / 'outputs/calibration-optimization-20261030/gui-test'
    out.mkdir(parents=True, exist_ok=True)
    active = ROOT / 'outputs/measurement_workbench/calibration_current.json'
    digest = hashlib.sha256(active.read_bytes()).hexdigest()
    pairs, metadata = read_calibration_diagnostic(
        ROOT / 'outputs/measurement_workbench/calibration_diagnostics/calibration_20261030_120105_953411.npz')
    root = tk.Tk(); root.geometry('100x60+0+0')
    messages = Mock()
    dialog = ChessboardWizardDialog(SimpleNamespace(root=root, tk=tk, ttk=ttk, messagebox=messages),
                                     profile_path=out / 'profile.json')
    dialog.window.title('标定解算验证 · 独立测试目录')
    dialog.measured_square_mm.set('17'); dialog.expected_baseline_mm.set('60')
    dialog.pairs = pairs
    dialog._pair_capture_ids = metadata['capture_archive']['pair_capture_ids']
    dialog._refresh_pairs()
    checks = {'heartbeat': 0}
    def heartbeat():
        checks['heartbeat'] += 1
        if dialog._solving:
            root.after(100, heartbeat)
    try:
        with patch('pipe_twin.workbench_profile.default_calibration_path', return_value=out / 'calibration_current.json'), \
             patch('pipe_twin.calibration_wizard._DIAGNOSTIC_ROOT', out / 'diagnostics'):
            dialog.finish()
            assert dialog._solving
            assert 'disabled' in dialog.settings_button.state()
            dialog.finish()  # A second click cannot start another solve.
            heartbeat()
            deadline = time.monotonic() + 90
            while dialog._solving and time.monotonic() < deadline:
                root.update(); time.sleep(.02)
            assert not dialog._solving
            assert checks['heartbeat'] >= 5
            assert 'disabled' not in dialog.settings_button.state()
            assert not messages.showerror.called and not messages.showwarning.called, messages.mock_calls
            saved = json.loads((out / 'calibration_current.json').read_text(encoding='utf8'))
            calibration = saved['stereo_calibration']
            assert calibration['validated'] and calibration['quality_audit']['holdout']['status'] == 'PASSED'
            checks.update(message=dialog.message.get(), guidance=dialog.guidance_status.get(),
                          baseline_mm=calibration['baseline_mm'], calibration_id=calibration['calibration_id'])
            saved_digest = hashlib.sha256((out / 'calibration_current.json').read_bytes()).hexdigest()
            dialog.pairs = pairs[:3]
            dialog._pair_capture_ids = dialog._pair_capture_ids[:3]
            dialog.finish()
            deadline = time.monotonic() + 10
            while dialog._solving and time.monotonic() < deadline:
                root.update(); time.sleep(.02)
            assert not dialog._solving and messages.showerror.call_count == 1
            assert '解算未完成' in dialog.message.get() and '留出' in dialog.message.get()
            assert 'disabled' not in dialog.settings_button.state()
            assert hashlib.sha256((out / 'calibration_current.json').read_bytes()).hexdigest() == saved_digest
            checks['insufficient_photos_error'] = dialog.message.get()
    finally:
        dialog.close(); root.destroy()
    assert hashlib.sha256(active.read_bytes()).hexdigest() == digest
    checks['production_calibration_unchanged'] = digest
    Path(__file__).with_name('gui-validation.json').write_text(json.dumps(checks, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(json.dumps(checks, ensure_ascii=False))


if __name__ == '__main__':
    main()
