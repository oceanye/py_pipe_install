"""Compare a saved raw burst with active or fixed-baseline diagnostic calibration."""
from pathlib import Path
from datetime import datetime,timezone
import copy,hashlib,json,sys,time
import argparse
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/verification-20261017'
sys.path.insert(0,str(ROOT))
from pipe_twin.calibration_wizard import Rectifier,_camera_dict,build_rectification_recipe
from pipe_twin.workbench_profile import save_camera_calibration_bundle
from pipe_twin.elevation_dataset import create_elevation_dataset,load_elevation_dataset
from pipe_twin.stereo_analyzer import analyze_stereo_capture


def main():
    cv2.setNumThreads(2)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('active','fixed60'))
    parser.add_argument('--raw',default='raw-burst')
    parser.add_argument('--tag',default='')
    parser.add_argument('--output-root',type=Path,default=OUT)
    parser.add_argument('--variants',nargs='+',choices=('single','burst'),default=['single','burst'])
    parser.add_argument('--num-disparities',type=int)
    parser.add_argument('--preprocessing',choices=('none','low_light'))
    args=parser.parse_args()
    mode=args.mode
    output_root=args.output_root.resolve()
    out=output_root/(mode+args.tag)
    out.mkdir(parents=True,exist_ok=True)
    raw=output_root/args.raw
    frames=json.loads((raw/'raw-manifest.json').read_text(encoding='utf-8'))['frames']
    active=ROOT/'outputs/measurement_workbench/calibration_current.json'
    active_hash=hashlib.sha256(active.read_bytes()).hexdigest()
    bundle=json.loads(active.read_text(encoding='utf-8'))
    calibration=bundle['stereo_calibration']; recipe=bundle['rectification_recipe']
    if mode=='fixed60':
        path=output_root/'fixed60-bounded-result.json'
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        solve=json.loads(path.read_text(encoding='utf-8'))
        c=solve['selected']
        assert not c['problems'] and abs(c['baseline_mm']-60)<1e-6
        cid='field-chess-fixed60-diagnostic-'+digest[:12]
        k=np.asarray(c['P1'])[:,:3]
        calibration={'calibration_id':cid,'validated':False,'registration_validated':False,
            'rectified':True,'baseline_mm':c['baseline_mm'],'max_sync_delta_ms':1.,
            'left_camera':_camera_dict('left',[0.,0.,0.],(1920,1080),k),
            'right_camera':_camera_dict('right',[c['baseline_mm'],0.,0.],(1920,1080),k),
            'quality_audit':{'solve_mode':'fixed_baseline_joint_diagnostic','square_mm':17.,
                'baseline_constraint_mm':60.,'input_pair_count':31,'training_pair_count':25,
                'training_input_indices':solve['training_indices'],'holdout_input_indices':solve['holdout_indices'],
                'holdout':solve['holdout'],'stereo_rms_px':c['rms'],'converged':c['converged'],
                'active_bounds':c['active_bounds'],'scale_verified_independently':False,
                'scope':'DIAGNOSTIC_RECOGNITION_NOT_ACTIVE_CALIBRATION',
                'operator_acceptance':{'status':'PROVISIONAL','authorized':True,'scope':'relative_elevation',
                    'calibration_id':cid,'accepted_at':datetime.now(timezone.utc).isoformat(),
                    'reason':'用户确认 17 mm 格长、60 mm 基线和实物尺寸，授权重新取样及识别验算',
                    'source_result_sha256':digest}}}
        recipe=build_rectification_recipe(calibration_id=cid,image_size=(1920,1080),
            **{k:np.asarray(c[k],np.float64) for k in ('K1','D1','K2','D2','R1','R2','P1','P2')})
    save_camera_calibration_bundle(out/'calibration.json',calibration,rectification_recipe=recipe)
    rectifier=Rectifier(recipe)
    rectified=out/'rectified'
    rectified.mkdir(exist_ok=True)
    records=[]
    for frame in frames:
        record={'timestamp_sources':{role:frame['timestamp_source'] for role in ('left','right')},
                'camera_capture_provenance':frame['provenance'],'sha256':{}}
        for role in ('left','right'):
            source=raw/frame['files'][role]
            assert hashlib.sha256(source.read_bytes()).hexdigest()==frame['sha256'][role]
            image=cv2.imdecode(np.fromfile(source,np.uint8),cv2.IMREAD_COLOR)
            corrected=rectifier.rectify(role,image)
            path=rectified/source.name
            ok,data=cv2.imencode('.png',corrected); assert ok
            path.write_bytes(data.tobytes())
            record[f'{role}_path']=str(path)
            record[f'{role}_time']=frame[f'{role}_time']
            record['sha256'][role]=hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(record)
    old=ROOT/'outputs/elevation_workbench/captures/elevation-20261014-150423-df1c14c6/manifest.json'
    loaded=load_elevation_dataset(old)
    manifest=loaded['manifest']
    analysis={k:v for k,v in manifest['analysis'].items() if k not in ('mode','elevation_auto')}
    if args.num_disparities is not None:
        analysis['stereo_matching']['num_disparities']=args.num_disparities
    if args.preprocessing is not None:
        analysis['stereo_matching']['preprocessing']=args.preprocessing
    results=[]
    for variant in args.variants:
        last=records[-1]
        path=create_elevation_dataset(output_root=out/variant,calibration=calibration,
            left_path=Path(last['left_path']),right_path=Path(last['right_path']),
            left_time=last['left_time'],right_time=last['right_time'],
            pipe_specs=loaded['pipe_specs'],pair_confirmed=True,
            model_path=old.parent/manifest['model']['path'],stl_unit='millimeter',
            rectification_recipe=recipe,camera_capture_provenance=last['camera_capture_provenance'],
            timestamp_sources=last['timestamp_sources'],analysis_settings=analysis,
            registration_settings=loaded['registration_settings'],
            temporal_frames=records if variant=='burst' else None)
        print(mode,variant,'SAVED',path,flush=True)
        started=time.monotonic()
        result=analyze_stereo_capture(path,report_output_path=path.parent/'report.json',
            evidence_dir=path.parent/'evidence',progress=lambda message:print(mode,variant,message,flush=True))
        summary={'mode':mode,'variant':variant,'manifest':str(path.relative_to(ROOT)),
            'report':str((path.parent/'report.json').relative_to(ROOT)),
            'seconds':time.monotonic()-started,'counts':result['counts'],
            'observation_count':len(result['local_surface']['observations']),
            'observations':result['local_surface']['observations'],
            'registration':result['registration'],'temporal_fusion':result.get('temporal_fusion')}
        results.append(summary)
        (out/f'{variant}-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        print('RESULT',mode,variant,summary['counts'],summary['observation_count'],flush=True)
    assert hashlib.sha256(active.read_bytes()).hexdigest()==active_hash
    (out/'summary.json').write_text(json.dumps({'active_calibration_sha256_unchanged':active_hash,
        'calibration_id':calibration['calibration_id'],'baseline_mm':calibration['baseline_mm'],
        'results':results},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
