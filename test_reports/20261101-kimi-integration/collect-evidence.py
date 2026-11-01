"""Collect completed checks without invoking the camera or provider."""
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = ROOT/'outputs/kimi-integration-20261101'
sys.path.insert(0, str(ROOT))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8', newline='\n')


def text_log(path):
    blob=path.read_bytes()
    return blob.decode('utf-16' if blob.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig').replace('\r\n','\n')


def summary(path):
    log=text_log(path)
    rows=[line for line in log.splitlines() if re.search(r'\d+ passed',line)]
    if not rows:
        raise ValueError('Test run has not completed.')
    return rows[-1]


def junit(path):
    tree=ET.parse(path)
    return {'suites':[dict(s.attrib) for s in tree.iter('testsuite')],
            'failures':[dict(c.attrib) for c in tree.iter('testcase')
                        if c.find('failure') is not None or c.find('error') is not None],
            'skips':[dict(c.attrib, reason=c.find('skipped').get('message'))
                     for c in tree.iter('testcase') if c.find('skipped') is not None]}


def main():
    from pipe_twin.ai_settings import configured_status
    packages=['numpy','scipy','opencv-python-headless','rhino3dm','ezdxf','pytest',
              'colorama','iniconfig','packaging','pluggy','pygments','pyparsing','typing_extensions','fonttools']
    active={name:metadata.version(name) for name in packages}
    query='import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in '+repr(packages)+'}))'
    pinned=json.loads(subprocess.check_output(
        [str(OUT/'venv/Scripts/python.exe'),'-c',query],text=True))
    required=dict(line.split('==') for line in (ROOT/'requirements-lock-windows-py312.txt').read_text().splitlines() if '==' in line)
    files=subprocess.check_output(['git','diff','--name-only','b00c170..HEAD','--','pipe_twin','tests','.github','requirements.txt','requirements-lock-windows-py312.txt'],cwd=ROOT,text=True).splitlines()
    snapshot={name:hashlib.sha256((ROOT/name).read_text(encoding='utf-8').replace('\r\n','\n').encode()).hexdigest() for name in files}
    evidence={
        'recorded_at':datetime.now(timezone.utc).isoformat(),
        'implementation_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'starting_local_and_origin_commit':'b00c1702e99b984c7d16d8c761b1d00539891eaa',
        'python':sys.version, 'active_environment_versions':active, 'clean_environment_versions':pinned,
        'active_and_clean_versions_equal':active==pinned,
        'required_versions_equal':all(active.get(k)==v and pinned.get(k)==v for k,v in required.items()),
        'pytest_opencv_threads':2,
        'first_full_summary':summary(OUT/'pytest-first-output.txt'),
        'first_full':junit(OUT/'pytest-first.xml'),
        'final_full_summary':summary(OUT/'pytest-output.txt'),
        'final_full':junit(OUT/'pytest.xml'),
        'isolated_tests':[junit(p) for p in OUT.glob('pytest-isolated*.xml')],
        'safe_ai_configuration_status':configured_status(),
        'credentials_in_repository':False,
        'new_camera_capture':False,
        'real_provider_call_count':2,
        'real_provider_result':'AMBIGUOUS_ON_ALL_12_MODEL_POINTS',
        'source_hash_normalization':'UTF8_LF',
        'tested_changed_sources_sha256':snapshot,
    }
    for name in ('live-evidence.json','gui-evidence.json','gui-live-evidence.json','auth-rejection-evidence.json'):
        p=HERE/name
        if not p.is_file():
            raise ValueError('Missing evidence: '+name)
        evidence[name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    write(HERE/'validation-evidence.json',evidence)
    (HERE/'pytest-output.txt').write_text(text_log(OUT/'pytest-output.txt'),encoding='utf-8',newline='\n')
    (HERE/'requirements-validated.txt').write_text('\n'.join(k+'=='+v for k,v in pinned.items())+'\n',encoding='utf-8',newline='\n')
    print(json.dumps({k:evidence[k] for k in ('implementation_commit','final_full_summary',
                      'active_and_clean_versions_equal','required_versions_equal')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
