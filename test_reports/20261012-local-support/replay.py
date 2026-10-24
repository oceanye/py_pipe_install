"""Read-only replay of the 17:37 capture; originals and GUI state stay intact."""
from pathlib import Path
import hashlib
import json
import sys
import time
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from pipe_twin.elevation_auto import analyze_elevation_auto_manifest
from pipe_twin.image_regions import REVISION, POLICY, POLICY_SHA256
from pipe_twin.stereo_correspondence import POLICY as PATCH_POLICY, POLICY_SHA256 as PATCH_SHA
from pipe_twin.elevation_gui import recognition_diagnostic_lines

SOURCE = ROOT/'outputs/elevation_workbench/captures/elevation-20261012-173724-476aa8d1'
OUT = ROOT/'outputs/local-support-20261012-173724'
OUT.mkdir(parents=True,exist_ok=True)
paths = ['manifest.json','report.json','photos/left.png','photos/right.png']
hashes = {s:hashlib.sha256((SOURCE/s).read_bytes()).hexdigest() for s in paths}
baseline = json.loads((SOURCE/'report.json').read_text(encoding='utf-8'))
(OUT/'baseline.json').write_text(json.dumps(baseline,ensure_ascii=False,indent=2),encoding='utf-8')
started = time.perf_counter()
result = analyze_elevation_auto_manifest(SOURCE/'manifest.json',
    report_output_path=OUT/'report.json',evidence_dir=OUT/'evidence')
elapsed = time.perf_counter()-started
assert all(hashlib.sha256((SOURCE/s).read_bytes()).hexdigest()==sha for s,sha in hashes.items())
(OUT/'diagnostic.txt').write_text('\n'.join(recognition_diagnostic_lines(result)),encoding='utf-8')
def summary(report):
    surface = report['local_surface']
    return {'counts':report['counts'],'observations':len(surface['observations']),
            'candidates_by_zone_eye':dict(Counter(c.get('zone_id','full')+':'+c['role'] for c in surface['image_candidates'])),
            'rejections':dict(Counter(r['reason'] for r in surface['rejected'])),
            'image_candidates':surface['image_candidates']}
files = ['pipe_twin/image_regions.py','pipe_twin/local_surface.py','pipe_twin/metrology.py',
         'pipe_twin/stereo_correspondence.py','pipe_twin/elevation_auto.py','pipe_twin/elevation_gui.py']
evidence = {'dataset_id':result['dataset_id'],'captured_at':result['capture_audit']['groups'][0]['captured_at'],
            'source_hashes':hashes,'source_unchanged':True,'actual_camera_capture':False,
            'code_sha256':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in files},
            'revision':REVISION,'policy':POLICY,'policy_sha256':POLICY_SHA256,
            'patch_policy':PATCH_POLICY,'patch_policy_sha256':PATCH_SHA,
            'replay_elapsed_s':elapsed,'before':summary(baseline),'after':summary(result)}
(OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
print(json.dumps({key:{k:v for k,v in evidence[key].items() if k!='image_candidates'} for key in ['before','after']},indent=2),flush=True)
for candidate in result['local_surface']['image_candidates']:
    diagnostic = candidate.get('correspondence_diagnostic',{})
    print(candidate['candidate_id'],candidate['region_px'],
          'valid',round(100*candidate['valid_depth_fraction'],1),
          'unique',round(100*diagnostic.get('sampled_unique_match_fraction',0),1),
          'attempts',len(candidate.get('attempts',[])),flush=True)
print('READ_ONLY_REPLAY_PASSED',round(elapsed,2),flush=True)
