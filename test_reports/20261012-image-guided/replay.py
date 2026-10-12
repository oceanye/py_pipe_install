from pathlib import Path
from collections import Counter
import hashlib
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from pipe_twin.elevation_auto import analyze_elevation_auto_groups, analyze_elevation_auto_zones, analyze_elevation_auto_manifest
from pipe_twin.stereo_analyzer import _calibration_from_manifest, StereoDepthResult
from pipe_twin.image_regions import REVISION, POLICY, POLICY_SHA256

OUT=ROOT/'outputs/image-guided-20261012'
OUT.mkdir(parents=True,exist_ok=True)
SOURCE=ROOT/'outputs/elevation_workbench/captures/elevation-20261012-135039-b075dca2'
paths=[SOURCE/'manifest.json',SOURCE/'report.json',SOURCE/'photos/left.png',SOURCE/'photos/right.png']
hashes={str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
manifest=json.loads((SOURCE/'manifest.json').read_text(encoding='utf-8'))
cal=_calibration_from_manifest(manifest['stereo_calibration'])
a=np.load(ROOT/'outputs/capture-check-20261012-135039/depth.npz')
metadata=json.loads((ROOT/'outputs/capture-check-20261012-135039/summary.json').read_text(encoding='utf-8'))
depth=StereoDepthResult(a['left_depth_mm'],a['right_depth_mm'],a['left_valid'],a['right_valid'],metadata['depth_audit'])
group={eye:cv2.imdecode(np.fromfile(SOURCE/f'photos/{eye}.png',np.uint8),1) for eye in ['left','right']}
group.update(depth=depth,pair_healthy=True,capture_id=manifest['dataset_id'],captured_at=metadata['captured_at'])
records=[]
for name,rect in [('user',[1008,10,163,498]),('middle',[900,10,400,425]),('wide',[800,10,800,425]),('full',None)]:
    started=time.perf_counter()
    kwargs=dict(calibration=cal,pipe_specs=manifest['analysis']['elevation_auto']['pipes'],
        registration_settings=manifest['analysis']['elevation_auto']['registration'],matching_settings=manifest['analysis']['matching'])
    result=(analyze_elevation_auto_groups([group],**kwargs) if rect is None else
        analyze_elevation_auto_zones([group],scope_settings={'scope':'zones','zones':[{'zone_id':'Z01','roi_rect_px':rect}]},**kwargs))
    (OUT/f'{name}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    candidates=result['local_surface'].get('image_candidates',[])
    records.append({'name':name,'roi_rect_px':rect,'elapsed_s':time.perf_counter()-started,
        'counts':result['counts'],'candidates_by_eye':dict(Counter(p['role'] for p in candidates)),
        'observation_count':len(result['local_surface']['observations']),
        'rejection_counts':dict(Counter(p['reason'] for p in result['local_surface']['rejected'])),
        'candidates':candidates,'rejected':result['local_surface']['rejected'],
        'surface_audit':result['local_surface']['audit']})
    print(name,records[-1]['counts'],records[-1]['candidates_by_eye'],records[-1]['rejection_counts'],flush=True)
    for eye in ['left','right']:
        canvas=group[eye].copy()
        if eye=='left' and rect:
            x,y,w,h=rect;cv2.rectangle(canvas,(x,y),(x+w,y+h),(0,210,0),2)
        for candidate in candidates:
            if candidate['role']!=eye:continue
            x,y,w,h=candidate['region_px']
            cv2.rectangle(canvas,(x,y),(x+w,y+h),(0,210,255),2)
            cv2.putText(canvas,candidate['candidate_id']+' IMAGE ONLY',(x,max(18,y-4)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,210,255),1)
        cv2.imencode('.jpg',canvas)[1].tofile(OUT/f'{name}_{eye}.jpg')
# Exercise the real saved-manifest path, including saved overlay coordinates.
real=analyze_elevation_auto_manifest(SOURCE/'manifest.json',report_output_path=OUT/'manifest_path_report.json',evidence_dir=OUT/'evidence')
assert real['counts']==records[0]['counts']
assert real['local_surface']['image_candidates']==json.loads((OUT/'user.json').read_text(encoding='utf-8'))['local_surface']['image_candidates']
assert all(hashlib.sha256((SOURCE/p).read_bytes()).hexdigest()==sha for p,sha in hashes.items())
codepaths=['pipe_twin/image_regions.py','pipe_twin/local_surface.py','pipe_twin/recognition/__init__.py',
           'pipe_twin/elevation_auto.py','pipe_twin/elevation_registration.py','pipe_twin/elevation_gui.py']
evidence={'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'working_code_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in codepaths},
 'revision':REVISION,'policy':POLICY,'policy_sha256':POLICY_SHA256,
 'dataset_id':manifest['dataset_id'],'captured_at':metadata['captured_at'],'source_hashes':hashes,
 'source_unchanged':True,'saved_manifest_path_verified':True,'results':records}
(OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
print('READ_ONLY_FIELD_REPLAY_AND_SAVED_MANIFEST_PATH_PASSED',flush=True)
