"""Read-only capture diagnosis: reuse full-frame depth for ROI comparisons."""
from pathlib import Path
from collections import Counter
import copy
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from pipe_twin.stereo_analyzer import _analysis_config, _calibration_from_manifest, _compute_stereo_depth
from pipe_twin.elevation_auto import analyze_elevation_auto_zones
from pipe_twin.capture_quality import measure_patch

OUT = ROOT / 'outputs/capture-check-20261012-135039'
OUT.mkdir(parents=True, exist_ok=True)
SOURCE = ROOT / 'outputs/elevation_workbench/captures/elevation-20261012-135039-b075dca2'
manifest = json.loads((SOURCE / 'manifest.json').read_text(encoding='utf-8'))
original = json.loads((SOURCE / 'report.json').read_text(encoding='utf-8'))
images = {eye: cv2.imdecode(np.fromfile(SOURCE / f'photos/{eye}.png', np.uint8), cv2.IMREAD_COLOR)
          for eye in ('left', 'right')}
source_hashes = {str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in [SOURCE/'manifest.json', SOURCE/'report.json', SOURCE/'photos/left.png', SOURCE/'photos/right.png']}
cal = _calibration_from_manifest(manifest['stereo_calibration'])
config = _analysis_config(manifest['analysis'])
depth = _compute_stereo_depth(images['left'], images['right'], cal, config)
np.savez_compressed(OUT/'depth.npz', left_depth_mm=depth.left_depth_mm, right_depth_mm=depth.right_depth_mm,
                    left_valid=depth.left_valid, right_valid=depth.right_valid)
group = {'left': images['left'], 'right': images['right'], 'depth': depth,
         'pair_healthy': True, 'capture_id': manifest['dataset_id'],
         'captured_at': manifest['capture']['capture_groups'][0]['views']['left']['captured_at']}
results = []
rects = [('user_163px', [1008,10,163,498]), ('middle_400px', [900,10,400,425]),
         ('wide_800px', [800,10,800,425])]
for label, rect in rects:
    started = time.perf_counter()
    settings = {'scope':'zones','right_roi_padding_px':128,
                'zones':[{'zone_id':'Z01','roi_rect_px':rect}]}
    result = analyze_elevation_auto_zones([group], calibration=cal,
        pipe_specs=manifest['analysis']['elevation_auto']['pipes'], scope_settings=settings,
        matching_settings=manifest['analysis']['matching'],
        registration_settings=manifest['analysis']['elevation_auto']['registration'])
    (OUT/f'{label}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    audit = result['capture_audit']['groups'][0]['zone_audits'][0]['surface_audit']
    entry = {'name':label,'roi_rect_px':rect,'seconds':time.perf_counter()-started,
        'counts':result['counts'], 'observation_count':len(result['local_surface']['observations']),
        'registration_status':result['registration']['status'],
        'registration_reasons':result['registration']['reason_codes'],
        'rejection_counts':dict(Counter(r.get('reason') for r in result['local_surface']['rejected'])),
        'surface_audit':{k:audit.get(k) for k in ['status','truncated','candidate_budgets','source_rejection_counts','depth_surfaces']}}
    results.append(entry)
    print(json.dumps(entry,ensure_ascii=True),flush=True)

# Manually chosen interior quadrilaterals, for image/depth diagnosis only.
# These masks do not supply pipe identities to the recognizer.
patches = {
 'red_upper': {'left': [[900,50],[1500,40],[1500,85],[900,94]],
               'right': [[790,50],[1390,40],[1390,85],[790,94]]},
 'red_lower': {'left': [[900,143],[1500,127],[1500,189],[900,205]],
               'right': [[770,143],[1370,127],[1370,189],[770,205]]},
 'white': {'left': [[900,260],[1500,244],[1500,267],[900,283]],
           'right': [[795,260],[1395,244],[1395,267],[795,283]]},
 'blue': {'left': [[900,332],[1500,314],[1500,390],[900,408]],
          'right': [[755,332],[1355,314],[1355,390],[755,408]]},
}
patch_stats = {}
for label, eyes in patches.items():
    patch_stats[label] = {}
    for eye, polygon in eyes.items():
        mask = np.zeros(images[eye].shape[:2], np.uint8)
        cv2.fillPoly(mask,[np.asarray(polygon,np.int32)],1)
        mask = mask.astype(bool)
        valid = getattr(depth,f'{eye}_valid') & mask
        z = getattr(depth,f'{eye}_depth_mm')[valid]
        pixels = images[eye][mask]
        gray = cv2.cvtColor(images[eye],cv2.COLOR_BGR2GRAY)[mask]
        patch_stats[label][eye] = {'polygon_px':polygon, 'pixels':int(mask.sum()),
           'valid_depth_fraction':float(valid.sum()/mask.sum()),
           'depth_z_percentiles_mm_diagnostic_only':np.percentile(z,[5,25,50,75,95]).tolist() if len(z) else [],
           'highlights_fraction_two_channels_ge250':float(np.mean(np.sum(pixels>=250,axis=1)>=2)),
           'gray_median':float(np.median(gray))}
        color = {'red_upper':'#FF0000','red_lower':'#FF0000','white':'#FFFFFF','blue':'#0000FF'}[label]
        quality = measure_patch(images[eye],color,mask)
        patch_stats[label][eye]['quality_gui_policy'] = quality
        patch_stats[label][eye]['quality_sampled_reference'] = measure_patch(
            images[eye],quality['sampled_color_srgb'],mask)
overlay = images['left'].copy()
for (label,rect),color in zip(rects,[(0,255,0),(0,220,255),(255,180,0)]):
    x,y,w,h = rect
    cv2.rectangle(overlay,(x,y),(x+w,y+h),color,3)
    label_x = x if label!='wide_800px' else x+w-180
    cv2.putText(overlay,label,(label_x,y+h+25),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2,cv2.LINE_AA)
cv2.imencode('.jpg',overlay)[1].tofile(OUT/'roi_comparison.jpg')
for eye in ('left','right'):
    overlay = images[eye].copy()
    valid = getattr(depth,f'{eye}_valid')
    tint = overlay.copy()
    tint[~valid] = (0,0,160)
    overlay = cv2.addWeighted(overlay,.55,tint,.45,0)
    for label,eyes in patches.items():
        polygon = np.asarray(eyes[eye],np.int32)
        cv2.polylines(overlay,[polygon],True,(0,255,255),2)
        cv2.putText(overlay,label,tuple(polygon[0]),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),1,cv2.LINE_AA)
    cv2.imencode('.jpg',overlay)[1].tofile(OUT/f'{eye}_depth_patches.jpg')
unchanged = all(hashlib.sha256((SOURCE/name).read_bytes()).hexdigest()==sha for name,sha in source_hashes.items())
summary = {'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
 'dataset_id':manifest['dataset_id'],'captured_at':group['captured_at'],
 'original_report_generated_at':original['generated_at'],'source_hashes':source_hashes,
 'source_unchanged':unchanged,'depth_audit':depth.audit, 'roi_comparisons':results,
 'manual_diagnostic_patches':patch_stats,
 'notes':['Depth percentiles are unvalidated diagnostic values, not pipe distance measurements.',
          'No camera, GUI, production parameters, or original capture files were changed.']}
(OUT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
assert unchanged
assert results[0]['counts']==original['counts']
assert results[0]['rejection_counts']==dict(Counter(r.get('reason') for r in original['local_surface']['rejected']))
print('SOURCE_UNCHANGED_AND_ORIGINAL_REPRODUCED',flush=True)
print(json.dumps(patch_stats,ensure_ascii=True),flush=True)
