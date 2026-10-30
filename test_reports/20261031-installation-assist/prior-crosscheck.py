"""Independent coarse identity experiment using actual archived boundary samples.

No prior pipe IDs; physical/CAD diameter classes are explicit assumptions.
This is not a measurement or installation acceptance result.
"""
from pathlib import Path
import sys,json,itertools,hashlib
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from pipe_twin.axial_selection import band_sampling_masks
from pipe_twin.pooled_sections import aggregate_image, _bounded
from pipe_twin.projection_diagnostics import fit_silhouettes
from pipe_twin.elevation_dataset import load_elevation_dataset
from pipe_twin.stereo_analyzer import _calibration_from_manifest

source=ROOT/'outputs/elevation_workbench/captures/elevation-20261030-183635-b6b7b875/manifest.json'
report=ROOT/'outputs/geometry-path-20261031/report.json'
loaded=load_elevation_dataset(source)
manifest=json.loads(source.read_text(encoding='utf-8'))
stage=json.loads(report.read_text(encoding='utf-8'))['section_evaluation']
cal=_calibration_from_manifest(loaded['calibration'])
photos={role:source.parent/manifest['capture']['capture_groups'][-1]['views'][role]['path'] for role in ('left','right')}
protected=[source,report,*photos.values(),ROOT/'outputs/measurement_workbench/calibration_current.json',ROOT/'outputs/elevation_workbench/last_scene.json']
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
images={r:cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),cv2.IMREAD_COLOR) for r,p in photos.items()}
masks,band=band_sampling_masks(loaded['registration_settings']['aggregation_band'],images['left'].shape,images['right'].shape,disparity_range=(0.,255.))
items={r:aggregate_image(images[r],r,sampling_mask=masks[r],axis_hint_image=band['axis_image'])[0] for r in ('left','right')}
specs=manifest['analysis']['elevation_auto']['pipes']
xy=np.array([np.mean(s['centerline_world_mm'],axis=0)[:2] for s in specs]);xy-=xy.mean(axis=0)
classes=[round(s['nominal_diameter_mm']) for s in specs]
observed=[]
for candidate in stage['candidates']:
    a=next(v for v in items['left'] if v['candidate_id']==candidate['left_candidate_id'])
    b=next(v for v in items['right'] if v['candidate_id']==candidate['right_candidate_id'])
    image_axis=np.array(a['image_axis']);flip=image_axis@b['image_axis']<0
    obs=[*a['pooled_edges'],*[{**o,'edge':('bottom' if o['edge']=='top' else 'top') if flip else o['edge'],'image_axis':image_axis.tolist()} for o in b['pooled_edges']]]
    # Known user colour/size mapping, not the synthetic STL display colour.
    bucket=41 if a['appearance']=='hue-0' else 26 if a['appearance']=='neutral' else 51
    observed.append({'id':candidate['candidate_id'],'appearance':a['appearance'],'class':bucket,
                     'observations':obs,'image_axis':image_axis,'reference':np.mean(a['pooled_edges'][0]['points'],axis=0)})

results=[]
for prior_source in ('USER_PURCHASE_40_25_50','CAD_41_26_51'):
 for axis_source in ('axis','unconstrained_axis'):
  points=[];fits=[]
  for value in observed:
   diameter=value['class']-(1 if prior_source.startswith('USER') else 0)
   fit=fit_silhouettes(_bounded(value['observations']),cal.left.rectified_intrinsic,cal.baseline_mm,stage[axis_source]['axis'],
                      image_axis=value['image_axis'],reference_pixel=value['reference'],known_diameter_mm=diameter)
   assert fit['status']=='CONDITIONAL_FIT'
   points.append(fit['section_center_parameters_mm'])
   fits.append({k:fit[k] for k in ('known_diameter_mm','diameter_source','reprojection_rms_px','section_center_parameters_mm')})
  points=np.array(points);scores=[]
  groups=[[i for i,d in enumerate(classes) if d==o['class']] for o in observed]
  for ids in itertools.product(*groups):
   if len(set(ids))<len(ids):continue
   a=xy[list(ids)];ac=a-a.mean(axis=0);bc=points-points.mean(axis=0)
   u,_,vt=np.linalg.svd(ac.T@bc)
   for sign in (1,-1):
    q=u@np.diag([1,sign])@vt;t=points.mean(axis=0)-a.mean(axis=0)@q
    camera=(-t)@q.T
    if camera[1]<=0:continue
    error=np.linalg.norm(a@q+t-points,axis=1)
    scores.append({'rms_mm':float(np.sqrt(np.mean(error**2))),'pipe_ids':[specs[i]['pipe_id'] for i in ids],
      'residuals_mm':error.tolist(),'assumed_camera_cross_section_mm':camera.tolist(),'determinant':float(np.linalg.det(q))})
  scores.sort(key=lambda s:s['rms_mm'])
  results.append({'prior_source':prior_source,'axis_source':axis_source,'relative_centers_mm':(points-points[0]).tolist(),
                  'fits':fits,'top10':scores[:10], 'evaluated_compatible_hypotheses':len(scores)})

result={'scope':'DIAMETER_PRIOR_AUXILIARY_IDENTITY_NOT_MEASUREMENT_OR_ACCEPTANCE','input_pipe_ids_used_as_prior':False,
        'observed_order':[{'observation_id':o['id'],'appearance':o['appearance'],'cad_diameter_class':o['class']} for o in observed],
        'source_sha256':hashes,'source_unchanged':all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items()),
        'results':results}
assert result['source_unchanged']
Path(__file__).with_name('prior-crosscheck.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps([{k:v for k,v in r.items() if k not in ('fits','top10')}|{'top2':r['top10'][:2]} for r in results],ensure_ascii=False,indent=2))
