"""Summarize completed tests and verify archived-input/measurement preservation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs/installation-assist-20261031'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tests(filename, subtests_passed=0):
    path = OUTPUT / filename
    suite = ET.parse(path).getroot().find('testsuite')
    result = {k: int(suite.attrib[k]) for k in ('tests', 'failures', 'errors', 'skipped')}
    result['junit_testcase_count_including_subtests'] = result['tests']
    result['tests'] -= subtests_passed
    result['subtests_passed'] = subtests_passed
    result['passed'] = result['tests'] - result['failures'] - result['errors'] - result['skipped']
    result['seconds'] = float(suite.attrib['time'])
    result['xml_sha256'] = sha(path)
    result['skips'] = [{'test': case.attrib['classname'] + '::' + case.attrib['name'],
                        'reason': case.find('skipped').attrib.get('message')}
                       for case in suite.findall('testcase') if case.find('skipped') is not None]
    assert result['failures'] == result['errors'] == 0
    return result


def main():
    replay = read(HERE / 'evidence.json')
    gui = read(HERE / 'gui-evidence.json')
    preserved = {name: sha(ROOT / name) == digest for name, digest in replay['source_sha256'].items()}
    assert all(preserved.values())
    assert all(sha(ROOT / name) == digest for name, digest in gui['source_sha256'].items())
    old = read(ROOT / 'outputs/geometry-path-20261031/report.json')
    current = read((ROOT / replay['derived_scene']).with_name('report.json'))
    before = {r['id']: r['diameter_mm'] for r in old['section_evaluation']['model']['candidates']}
    after = {r['id']: r['diameter_mm'] for r in current['section_evaluation']['model']['candidates']}
    assert before.keys() == after.keys()
    deltas = {key: after[key] - before[key] for key in before}
    assert all(value == 0 for value in deltas.values())
    checks = {name: tests(filename, subtests_passed) for name, filename, subtests_passed in (
        ('full_suite', 'pytest.xml', 272),
        ('isolated_lazy_tk', 'pytest-lazy-tk.xml', 0),
        ('isolated_ground_gui', 'pytest-ground-gui.xml', 0),
        ('final_assist_gui', 'pytest-final-gui.xml', 5))}
    assert all(checks[k]['skipped'] == 0 for k in checks if k != 'full_suite')
    result = {
        'recorded_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'ARCHIVED_REPLAY_QUALITATIVE_ASSISTANCE_NOT_METRIC_ACCEPTANCE',
        'tests': checks,
        'full_suite_subtests_passed': 272,
        'test_notes': [
            'The full suite had two Tk environment skips; each affected test passed in an isolated run.',
            'The final GUI navigation/display edits were also checked after full-suite collection.',
            'OpenCV used two threads in the full-suite process; production settings were not relaxed.',
        ],
        'protected_file_count': len(preserved),
        'original_sources_unchanged': all(preserved.values()),
        'gui_source_hashes_unchanged': True,
        'free_diameter_delta_mm_by_candidate': deltas,
        'assist_counts': current['installation_assist']['counts'],
        'strict_counts': current['counts'],
        'new_capture': False,
        'gui_evidence_sha256': sha(HERE / 'gui-evidence.json'),
        'replay_evidence_sha256': sha(HERE / 'evidence.json'),
    }
    (HERE / 'validation-evidence.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
