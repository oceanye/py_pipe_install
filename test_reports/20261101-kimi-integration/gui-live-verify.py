"""One explicitly enabled real GUI -> Kimi -> save -> reopen acceptance run.

Default invocation is preparation only. --execute-live sends the archived
2026-10-30 reference stereo pair through the application's configured provider.
No camera is opened and no key is read or printed by this script. Existing
encrypted local configuration is resolved exclusively by application code.
"""
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import argparse
import copy
import hashlib
import json
import runpy
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'outputs/kimi-integration-20261101'
DEFAULT = OUT / 'live-20261101-105332/scene/manifest.json'
HELPERS = runpy.run_path(str(ROOT / 'test_reports/20261101-kimi-integration/gui-verify.py'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=DEFAULT)
    parser.add_argument('--execute-live', action='store_true')
    args = parser.parse_args()
    source = args.source.resolve()
    if not args.execute_live:
        print(json.dumps({'mode': 'PREPARED_ONLY', 'source': str(source),
                          'maximum_provider_requests': 1, 'camera_access': False,
                          'archived_capture_date': '2026-10-30',
                          'requires': '--execute-live'}, ensure_ascii=False, indent=2))
        return 0

    import tkinter as tk
    from tkinter import ttk
    from pipe_twin.elevation_gui import ElevationCaptureDialog
    from pipe_twin.ai_provider import request_section_assessment

    protected = HELPERS['protected_inputs'](source)
    run = OUT / ('gui-live-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
    run.mkdir(parents=True, exist_ok=False)
    scene = run / 'scene'
    shutil.copytree(source.parent, scene, ignore=shutil.ignore_patterns('ai_section_result.json'))
    path = scene / 'manifest.json'
    a_bytes_before = sha(path.with_name('report.json'))
    evidence = {'scope': 'REAL_GUI_CLICK_TO_KIMI_SAVE_AND_REOPEN',
                'started_at': datetime.now(timezone.utc).isoformat(),
                'source': str(source), 'manifest': str(path),
                'new_capture': False, 'camera_opened': False,
                'credentials_read_by_script': False, 'provider_call_count': 0,
                'checks': [], 'callback_errors': []}

    def check(name, condition, **detail):
        evidence['checks'].append({'name': name, 'passed': bool(condition), **detail})

    def counted_provider(*arguments, **kwargs):
        evidence['provider_call_count'] += 1
        if evidence['provider_call_count'] > 1:
            raise AssertionError('This acceptance run permits only one real provider request.')
        print('REAL_PROVIDER_REQUEST_STARTED count=1', flush=True)
        return request_section_assessment(*arguments, **kwargs)

    root = tk.Tk()
    root.withdraw()
    root.report_callback_exception = lambda *exc: evidence['callback_errors'].append(''.join(traceback.format_exception(*exc)))
    style = ttk.Style(root)
    style.theme_use('clam')
    style.configure('.', font=('Microsoft YaHei UI', 10))
    style.configure('Treeview', rowheight=25)
    app = SimpleNamespace(root=root, tk=tk, ttk=ttk, messagebox=Mock(), filedialog=Mock())
    owner = None
    try:
        with patch('pipe_twin.workbench_profile.load_profile', return_value=(None, None)), \
             patch('pipe_twin.stereo_camera.StereoCameraSession.open', side_effect=AssertionError('Camera forbidden')), \
             patch('pipe_twin.ai_provider.request_section_assessment', side_effect=counted_provider):
            owner = ElevationCaptureDialog(app, path, output_root=run / 'gui-state', restore=False)
            owner.window.title('Kimi 实网 GUI 验收 · 10月30日归档照片 · 单次请求')
            owner.window.geometry('1280x860+10+10')
            owner.window.deiconify()
            owner.status_tabs.select(owner.ai_section_status_tab)
            root.update()
            baseline = copy.deepcopy(owner.report)
            check('source_bound_a_restored', bool(baseline and baseline.get('pipes')))
            check('invalid_burst_gate_applied_before_ai', baseline['installation_assist']['counts']['LIKELY'] == 0 and
                  baseline['installation_assist']['reason_codes'] == ['TEMPORAL_SEQUENCE_INVALID_FOR_IDENTITY'])
            check('no_preexisting_ai_sidecar', not path.with_name('ai_section_result.json').exists())
            owner.confirmed.set(True)
            owner.start_ai_evaluation()
            deadline = time.monotonic() + 660
            while owner.busy and time.monotonic() < deadline:
                root.update()
                time.sleep(0.025)
            if owner.busy:
                owner.ai_controller.cancel()
                raise TimeoutError('GUI acceptance deadline exceeded; no automatic retry.')
            root.update()
            check('no_messagebox_errors', not app.messagebox.showerror.called)
            if app.messagebox.showerror.called:
                raise RuntimeError('Application reported an error during the single live GUI run.')
            check('one_real_provider_request', evidence['provider_call_count'] == 1)
            check('same_manifest_reused', owner.last_manifest == path)
            check('strict_a_preserved_in_gui', owner.report['pipes'] == baseline['pipes'] and
                  owner.report['counts'] == baseline['counts'])
            check('a_report_file_not_rewritten', sha(path.with_name('report.json')) == a_bytes_before)
            check('ai_sidecar_saved_by_gui', path.with_name('ai_section_result.json').is_file())
            ai = copy.deepcopy(owner.report['ai_section_assist'])
            check('real_ai_rows_present', bool(ai.get('point_assessments')) and ai.get('provider') == 'kimi',
                  run_status=ai.get('run_status'), decision=ai.get('status'),
                  states={state: sum(r['section_status'] == state for r in ai.get('point_assessments', []))
                          for state in ('SUPPORTED_VISIBLE', 'AMBIGUOUS', 'UNRESOLVED', 'RECAPTURE_RECOMMENDED')})
            check('archive_timestamp_preserved', any(str(asset.get('captured_at', '')).startswith('2026-10-30')
                  for asset in ai.get('asset_refs', []) if asset.get('kind') == 'FIELD_IMAGE'))
            HELPERS['screenshot'](owner.window, run / 'gui-live-result.png')
            owner.show_analysis_details('ai_section')
            root.update()
            detail = owner.detail_texts['ai_section'].get('1.0', 'end')
            check('source_and_scope_visible', '实拍时间' in detail and '结果生成时间' in detail and '仅当前参考双目对' in detail)
            HELPERS['screenshot'](owner.analysis_details_window, run / 'gui-live-details.png')
            owner.load_session(path)
            root.update()
            check('reopened_ai_matches_saved', owner.report['ai_section_assist'] == ai)
            check('reopened_a_preserved', owner.report['pipes'] == baseline['pipes'] and owner.report['counts'] == baseline['counts'])
            check('isolated_latest_pointer_written', (run / 'gui-state/last_scene.json').is_file())
            evidence['generated_at'] = ai.get('generated_at')
            evidence['input_hash'] = ai.get('input_hash')
            evidence['sidecar_sha256'] = sha(path.with_name('ai_section_result.json'))
            print('REAL_GUI_ACCEPTANCE_FINISHED', flush=True)
    except Exception:
        evidence['fatal_error'] = traceback.format_exc()
    finally:
        if owner is not None:
            owner.close()
        root.destroy()
        after = {name: sha(ROOT / name) for name in protected}
        check('original_sources_and_default_pointer_unchanged', after == protected)
        check('no_tk_callback_errors', not evidence['callback_errors'])
        evidence.update(source_sha256=protected, sources_unchanged=after == protected,
                        finished_at=datetime.now(timezone.utc).isoformat(),
                        passed=not evidence.get('fatal_error') and all(c['passed'] for c in evidence['checks']))
        content = json.dumps(evidence, ensure_ascii=False, indent=2) + '\n'
        (run / 'gui-live-evidence.json').write_text(content, encoding='utf-8', newline='\n')
        Path(__file__).with_name('gui-live-evidence.json').write_text(content, encoding='utf-8', newline='\n')
    print(json.dumps({'passed': evidence['passed'], 'provider_call_count': evidence['provider_call_count'],
                      'checks': len(evidence['checks']), 'failed': [c for c in evidence['checks'] if not c['passed']],
                      'run_directory': str(run), 'fatal_error': evidence.get('fatal_error')}, ensure_ascii=False, indent=2))
    return 0 if evidence['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
