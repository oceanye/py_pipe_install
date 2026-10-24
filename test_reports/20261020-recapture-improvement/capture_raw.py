"""Capture 20 original stereo pairs; output directory must not already exist.

Usage: python capture_raw.py outputs/new-check/raw-burst 20
The second argument is the pre-capture warm-up duration in seconds.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import cv2
import numpy as np
from pipe_twin.stereo_camera import StereoCameraSession

out=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'outputs/improvement-20261020/raw-burst'
warmup_seconds=float(sys.argv[2]) if len(sys.argv)>2 else 20.
if not 0 <= warmup_seconds <= 60:
    raise ValueError('warmup duration must be between 0 and 60 seconds')
out.mkdir(parents=True,exist_ok=False)
profile=json.loads((ROOT/'outputs/measurement_workbench/workbench_profile.json').read_text(encoding='utf-8'))
camera=profile['camera']
session=StereoCameraSession(layout=camera['layout'],left_index=camera['left_index'],
    right_index=camera['right_index'],eye_width=1920,eye_height=1080,exposure_ms=camera['exposure_ms'])
records=[]
try:
    session.open()
    started=time.monotonic()
    count=0
    while time.monotonic()-started<warmup_seconds or count<15:
        session.read_pair()
        count+=1
    print('WARMUP',count,session.exposure_summary,flush=True)
    started=time.monotonic()
    for i in range(20):
        pair=session.read_pair()
        assert pair.sync_delta_ms<=1. and not np.array_equal(pair.left,pair.right)
        record={'index':i,'left_time':pair.left_captured_at,'right_time':pair.right_captured_at,
                'timestamp_source':pair.timestamp_source,'provenance':pair.provenance,'sync_delta_ms':pair.sync_delta_ms,
                'files':{},'sha256':{},'mean':{}}
        for role in ('left','right'):
            frame=getattr(pair,role)
            assert frame.shape==(1080,1920,3)
            assert float(np.mean(frame))>5,'Black image'
            path=out/f'{i:03d}_{role}.png'
            ok,data=cv2.imencode('.png',frame)
            assert ok
            path.write_bytes(data.tobytes())
            record['files'][role]=path.name
            record['sha256'][role]=hashlib.sha256(path.read_bytes()).hexdigest()
            record['mean'][role]=float(np.mean(frame))
        records.append(record)
        print('CAPTURE',i+1,record['mean'],flush=True)
        time.sleep(.15)
    manifest={'kind':'RAW_STEREO_BURST_FOR_CALIBRATION_COMPARISON','created_at':datetime.now(timezone.utc).isoformat(),
              'camera':camera,'eye_size':[1920,1080],'warmup_frames':count,'warmup_seconds':warmup_seconds,'rectified':False,
              'duration_s':time.monotonic()-started,'frames':records}
    (out/'raw-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
finally:
    session.close()
print('SAVED',str(out),flush=True)
