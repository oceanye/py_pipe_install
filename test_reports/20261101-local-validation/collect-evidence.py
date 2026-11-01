"""Collect completed local checks without running tests, cameras or AI services."""
from datetime import datetime, timezone
from importlib import metadata
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/local-validation-20261101'
HERE = Path(__file__).resolve().parent


def read_log(path):
    raw = path.read_bytes()
    return raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig').replace('\r\n', '\n')


def junit(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter('testcase'))
    suites = list(root.iter('testsuite'))
    return {
        'path': path.relative_to(ROOT).as_posix(),
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'suites': [dict(suite.attrib) for suite in suites],
        'testcase_elements': len(cases),
        'skips': [dict(case.attrib, reason=case.find('skipped').get('message'))
                  for case in cases if case.find('skipped') is not None],
        'failures': [dict(case.attrib) for case in cases
                     if case.find('failure') is not None or case.find('error') is not None],
    }


def main():
    import cv2
    import tkinter
    versions = {name: metadata.version(name) for name in
                ('numpy', 'scipy', 'opencv-python-headless', 'rhino3dm', 'ezdxf', 'pytest')}
    pins = dict(line.split('==', 1) for line in
                (ROOT / 'requirements.txt').read_text(encoding='utf-8').splitlines()
                if '==' in line and not line.startswith('#'))
    log = read_log(OUT / 'pytest-output.txt')
    summary = [line.strip() for line in log.splitlines()
               if re.search(r'\d+ passed', line)]
    if not summary:
        raise RuntimeError('Full test output is incomplete; do not report a pass.')
    report = {
        'recorded_at': datetime.now(timezone.utc).isoformat(),
        'tested_commit': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'scope': 'LOCAL_REGRESSION_ARCHIVED_REPLAY_REAL_TK_AI_CONTRACT_AND_PROVIDER_AUTH_AUDIT',
        'python': sys.version,
        'platform': platform.platform(),
        'opencv': cv2.__version__,
        'tk': tkinter.TkVersion,
        'versions': versions,
        'repository_pins': pins,
        'pin_mismatches': {name: {'installed': versions.get(name), 'required': version}
                           for name, version in pins.items() if versions.get(name) != version},
        'environment_was_upgraded': False,
        'clean_pinned_environment_tested': False,
        'pytest_opencv_threads': 2,
        'full_pytest_summary': summary[-1],
        'full_pytest': junit(OUT / 'pytest.xml'),
        'isolated_pytest': [junit(path) for path in sorted(OUT.glob('pytest-isolated*.xml'))],
        'external_ai_inference_called': False,
        'field_images_uploaded_to_ai': False,
        'application_ai_end_to_end_validated': False,
        'camera_opened': False,
        'new_field_capture': False,
        'production_code_changed': False,
        'credentials_saved': False,
    }
    for name in ('requirements-dry-run.txt', 'pip-check.txt', 'cli-help.txt'):
        path = OUT / name
        if path.is_file():
            report[name] = read_log(path).strip()
    provider = HERE / 'kimi-connectivity.json'
    if provider.is_file():
        probe = json.loads(provider.read_text(encoding='utf-8'))
        report['provider_model_list_called'] = True
        report['provider_model_list_authentication_verified'] = probe.get('model_list_authentication_verified', False)
    for name in ('replay-evidence.json', 'gui-evidence.json', 'ai-readiness.json', 'kimi-connectivity.json'):
        path = HERE / name
        if path.is_file():
            report[name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (HERE / 'validation-evidence.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    # Keep human-readable logs with the report; raw GUI images remain in outputs.
    (HERE / 'pytest-output.txt').write_text(log, encoding='utf-8', newline='\n')
    print(json.dumps({key: report[key] for key in
                      ('tested_commit', 'full_pytest_summary', 'pin_mismatches')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
