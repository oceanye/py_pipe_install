"""Run the production Kimi route on an isolated copy of archived field images.

Default prepares only. --execute makes one paid vision request with the locally
configured key. Never saves credentials or sends local installation conclusions.
"""
import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import runpy
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.ai_inputs import prepare_ai_request
from pipe_twin.ai_workflow import evaluate_ai_scene, accept_ai_result, load_ai_result


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
                    encoding='utf-8', newline='\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    source = ROOT / 'outputs/installation-assist-20261031/scenes/elevation-20261031-071031-189194c4/manifest.json'
    helpers = runpy.run_path(str(ROOT / 'test_reports/20261101-local-validation/replay.py'))
    _, protected = helpers['prepare'](source)
    run = ROOT / 'outputs/kimi-integration-20261101' / ('live-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    run.mkdir(parents=True)
    scene = run / 'scene'
    shutil.copytree(source.parent, scene)
    manifest = scene / 'manifest.json'
    # Existing results are not evidence for the new AI request.
    prepared = prepare_ai_request(manifest)
    summary = {k: prepared[k] for k in
               ('input_hash', 'point_map', 'control_section', 'asset_refs', 'source_binding', 'qualification_reason_codes')}
    write(run / 'prepared-input.json', summary)
    content = prepared['messages'][1]['content']
    ref = None
    for part in content:
        if part['type'] == 'text' and part['text'].startswith('asset_ref='):
            ref = part['text'].split('=', 1)[1]
        if part['type'] == 'image_url' and ref:
            url = part['image_url']['url']
            suffix = '.png' if url.startswith('data:image/png;') else '.jpg'
            (run / (ref+suffix)).write_bytes(base64.b64decode(url.split(',', 1)[1]))
    print(json.dumps({'run': str(run), 'manifest': str(manifest),
                      'asset_count': len(prepared['asset_refs']),
                      'point_count': len(prepared['point_map']['points']),
                      'qualification': prepared['qualification_reason_codes'],
                      'mode': 'LIVE_PROVIDER' if args.execute else 'PREPARED_ONLY'}, ensure_ascii=False), flush=True)
    if not args.execute:
        return
    before_report = (scene / 'report.json').read_bytes()
    report = json.loads(before_report)
    started = time.monotonic()
    evidence = {'started_at': datetime.now(timezone.utc).isoformat(),
                'scope': 'LIVE_KIMI_ARCHIVED_REFERENCE_PAIR', 'new_capture': False,
                'manifest': str(manifest.relative_to(ROOT)), 'input_hash': prepared['input_hash'],
                'credentials_saved_in_report': False, 'request_assets': prepared['asset_refs']}
    try:
        result = evaluate_ai_scene(manifest, progress=lambda message: print(message, flush=True))
        merged = accept_ai_result(manifest, result, report, save=True)
        restored = load_ai_result(manifest, report)
        unchanged = all(merged.get(k) == v for k,v in report.items() if k != 'ai_section_assist')
        assert unchanged and restored == merged
        assert (scene / 'report.json').read_bytes() == before_report
        ai = merged['ai_section_assist']
        evidence.update(status='COMPLETED', provider=result['provider'], model=result['model'],
                        usage=result['usage'], strict_report_unchanged=unchanged,
                        save_and_reopen_identical=True,
                        candidate_states=dict(Counter(c['section_status'] for c in ai['candidates'])),
                        result_status=ai['status'], candidates=ai['candidates'],
                        sidecar_sha256=hashlib.sha256((scene/'ai_section_result.json').read_bytes()).hexdigest())
        write(run / 'merged-display-report.json', merged)
    except Exception as error:
        evidence.update(status='FAILED', error_code=getattr(error, 'code', type(error).__name__),
                        error_message=str(error))
        raise
    finally:
        changed = [name for name,h in protected.items() if helpers['sha'](ROOT/name) != h]
        evidence.update(seconds=time.monotonic()-started, protected_source_count=len(protected),
                        protected_sources_unchanged=not changed, changed_source_files=changed)
        write(run/'live-evidence.json', evidence)
        write(ROOT/'outputs/kimi-integration-20261101/latest-live.json', {'run':str(run.relative_to(ROOT)),
              'manifest':str(manifest.relative_to(ROOT)), 'evidence':str((run/'live-evidence.json').relative_to(ROOT))})
        print(json.dumps({k:v for k,v in evidence.items() if k not in ('candidates','request_assets')}, ensure_ascii=False, indent=2), flush=True)
        assert not changed


if __name__ == '__main__':
    main()
