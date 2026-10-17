"""Diagnostic bundle adjustment with an exact 60 mm baseline and 17 mm board.

Requires SciPy (isolated local install supported). Keeps the established six
holdout views out of optimization and seed selection. Never activates a result.
"""
from pathlib import Path
import json
import sys
import time
import hashlib

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/verification-20261017'
sys.path.insert(0, str(ROOT))
sys.path.append(str(OUT/'deps'))
import cv2
import numpy as np
from scipy.optimize import least_squares
from pipe_twin.calibration_wizard import (
    _object_points, _intrinsic_problems, read_calibration_diagnostic,
    align_pair_orientation, rectification_quality_report,
)
from pipe_twin.calibration_quality import holdout_quality


def camera_parameters(k, d):
    return np.r_[np.log([k[0,0], k[1,1]]), k[0,2], k[1,2], np.ravel(d)[:4]]


def decode_camera(p):
    return (np.array([[np.exp(p[0]),0,p[2]],[0,np.exp(p[1]),p[3]],[0,0,1.]]),
            np.r_[p[4:8], 0.])


def translation(p, baseline):
    v = np.r_[-1., p]
    norm = np.linalg.norm(v)
    derivative = baseline*(np.eye(3)[:,1:]/norm - np.outer(v,p)/norm**3)
    return baseline*v/norm, derivative


class Bundle:
    def __init__(self, objects, observations, baseline=60.):
        self.objects = np.asarray(objects, np.float64)
        self.observations = np.asarray(observations, np.float64)
        self.baseline = baseline
        self.last_x = None
        self.calls = 0
        self.last_print = time.monotonic()

    def evaluate(self, x):
        if self.last_x is not None and np.array_equal(x,self.last_x):
            return self.residual, self.jacobian
        k1,d1 = decode_camera(x[:8]); k2,d2 = decode_camera(x[8:16])
        rs = x[16:19]; ts, dts = translation(x[19:21], self.baseline)
        block = len(self.objects)*2
        residual = np.empty(len(self.observations)*2*block)
        jacobian = np.zeros((len(residual),len(x)))
        for i, (obs_l,obs_r) in enumerate(self.observations):
            pose_start = 21+i*6
            rb,tb = x[pose_start:pose_start+3], x[pose_start+3:pose_start+6]
            comp = cv2.composeRT(rb,tb,rs,ts)
            rr,tr = comp[:2]
            pl,jl = cv2.projectPoints(self.objects,rb,tb,k1,d1)
            pr,jr = cv2.projectPoints(self.objects,rr,tr,k2,d2)
            a,b = slice(i*2*block,(i*2+1)*block),slice((i*2+1)*block,(i*2+2)*block)
            residual[a] = (pl.reshape(-1,2)-obs_l).ravel()
            residual[b] = (pr.reshape(-1,2)-obs_r).ravel()
            jacobian[a,:8] = jl[:,6:14]
            jacobian[a,0] *= k1[0,0]; jacobian[a,1] *= k1[1,1]
            jacobian[a,pose_start:pose_start+6] = jl[:,:6]
            jacobian[b,8:16] = jr[:,6:14]
            jacobian[b,8] *= k2[0,0]; jacobian[b,9] *= k2[1,1]
            jacobian[b,16:19] = jr[:,:3]@comp[4] + jr[:,3:6]@comp[8]
            jacobian[b,19:21] = jr[:,3:6]@dts
            jacobian[b,pose_start:pose_start+3] = jr[:,:3]@comp[2] + jr[:,3:6]@comp[6]
            jacobian[b,pose_start+3:pose_start+6] = jr[:,:3]@comp[3] + jr[:,3:6]@comp[7]
        self.last_x = x.copy()
        self.residual,self.jacobian = residual,jacobian
        self.calls += 1
        if time.monotonic()-self.last_print > 15:
            print('ITER',self.calls,'RMS',float(np.sqrt(np.mean(residual**2)*2)),
                  'FX',float(k1[0,0]),flush=True)
            self.last_print = time.monotonic()
        return residual,jacobian

    def fun(self,x):
        return self.evaluate(x)[0]

    def jac(self,x):
        return self.evaluate(x)[1]


def verify_math():
    objects = _object_points((8,6),17.).astype(np.float64)
    k = np.array([[1100.,0,960],[0,1080.,540],[0,0,1.]])
    p = camera_parameters(k, [0.03,-0.1,0.001,-0.002,0])
    rng = np.random.default_rng(123)
    poses = np.array([np.r_[rng.uniform(-.4,.4,3),rng.uniform(-150,40,2),rng.uniform(500,1100)] for _ in range(10)])
    truth = np.r_[p,p,[.01,.02,-.005],[.002,.015],poses.ravel()]
    observations = []
    ts,_ = translation(truth[19:21],60.)
    for pose in poses:
        rr,tr,*_ = cv2.composeRT(pose[:3],pose[3:],truth[16:19],ts)
        d = decode_camera(p)[1]
        observations.append([cv2.projectPoints(objects,pose[:3],pose[3:],k,d)[0].reshape(-1,2),
                             cv2.projectPoints(objects,rr,tr,k,d)[0].reshape(-1,2)])
    bundle = Bundle(objects,observations)
    initial = truth.copy()
    initial[[0,1,8,9]] += .025
    initial[[2,3,10,11]] += 2
    initial[16:19] += .002
    # Check the full analytic Jacobian, including composition and constrained T.
    jac = bundle.jac(initial).copy()
    errors = []
    for column in range(len(initial)):
        step = 1e-5 if column in (2,3,10,11) or column>=21 and (column-21)%6>=3 else 1e-6
        delta = np.zeros(len(initial)); delta[column] = step
        numeric = (bundle.fun(initial+delta)-bundle.fun(initial-delta))/(2*step)
        errors.append(float(np.max(np.abs(jac[:,column]-numeric))))
    assert max(errors)<1e-4, max(errors)
    fitted = least_squares(bundle.fun,initial,jac=bundle.jac,method='lm',x_scale='jac',max_nfev=100,
                           ftol=1e-10,xtol=1e-10,gtol=1e-10)
    rms = float(np.sqrt(np.mean(fitted.fun**2)*2))
    assert rms < 1e-7 and abs(np.exp(fitted.x[0])-1100)<1e-3
    return {'jacobian_max_abs_error':max(errors),'synthetic_rms_px':rms,
            'recovered_fx_px':float(np.exp(fitted.x[0])),'true_fx_px':1100.,'true_baseline_mm':60.}


def main():
    cv2.setNumThreads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    checks = verify_math()
    print('MATH',checks,flush=True)
    checkpoint = ROOT/'outputs/calibration-20261013/accepted-31-before-final.npz'
    pairs,metadata = read_calibration_diagnostic(checkpoint)
    assert len(pairs)==31 and metadata['square_mm']==17 and metadata['pattern_inner_corners']==[8,6]
    holdout = [4,8,12,17,21,25]
    train = [i for i in range(31) if i not in holdout]
    objects = _object_points((8,6),17.).astype(np.float64)
    observations = [[pairs[i][0].corners_px,align_pair_orientation(*pairs[i],pattern=(8,6)).corners_px] for i in train]
    observations = np.asarray(observations).reshape(len(train),2,-1,2)
    source = json.loads((ROOT/'outputs/calibration-20261014/joint-continuation.json').read_text(encoding='utf-8'))['selected']
    rs = cv2.Rodrigues(np.asarray(source['R']))[0].ravel()
    ts = np.ravel(source['T']); direction = ts[1:]/(-ts[0])
    candidates = []
    bounded = '--bounded' in sys.argv
    prefix = 'fixed60-bounded' if bounded else 'fixed60'
    for seed in (1000.,2000.,4000.):
        if seed==4000:
            k1,d1,k2,d2 = [np.asarray(source[k]) for k in ('K1','D1','K2','D2')]
        else:
            k1 = np.array([[seed,0,959.5],[0,seed,539.5],[0,0,1.]])
            k2=k1.copy(); d1=np.zeros(5); d2=np.zeros(5)
        poses=[]
        for obs_l,_ in observations:
            ok,rv,tv=cv2.solvePnP(objects,obs_l,k1,d1)
            assert ok
            poses.extend(np.r_[rv.ravel(),tv.ravel()])
        initial=np.r_[camera_parameters(k1,d1),camera_parameters(k2,d2),rs,direction,poses]
        bundle=Bundle(objects,observations)
        started=time.monotonic()
        options = {'method':'lm'}
        if bounded:
            lower = np.full(len(initial),-np.inf); upper = -lower
            for start in (0,8):
                lower[start:start+8] = [np.log(384),np.log(384),0,0,-5,-5,-5,-5]
                upper[start:start+8] = [np.log(7680),np.log(7680),1920,1080,5,5,5,5]
            options = {'method':'trf','bounds':(lower,upper)}
        fit=least_squares(bundle.fun,initial,jac=bundle.jac,x_scale='jac',
                          max_nfev=500,ftol=1e-9,xtol=1e-9,gtol=1e-9,**options)
        k1,d1=decode_camera(fit.x[:8]); k2,d2=decode_camera(fit.x[8:16])
        rotation=cv2.Rodrigues(fit.x[16:19])[0]; t,_=translation(fit.x[19:21],60.)
        r1,r2,p1,p2,*_=cv2.stereoRectify(k1,d1,k2,d2,(1920,1080),rotation,t,flags=cv2.CALIB_ZERO_DISPARITY,alpha=0.)
        candidate={'seed_fx':seed,'rms':float(np.sqrt(np.mean(fit.fun**2)*2)),
                   'nfev':fit.nfev,'converged':bool(fit.success),'termination':fit.message,
                   'optimality':float(fit.optimality),'elapsed_seconds':time.monotonic()-started,
                   'bounded_existing_intrinsic_policy':bounded,
                   'active_bounds':np.flatnonzero(fit.active_mask).tolist(),
                   'baseline_mm':float(np.linalg.norm(t)),
                   'problems':_intrinsic_problems(k1,d1,(1920,1080),'left')+_intrinsic_problems(k2,d2,(1920,1080),'right'),
                   **{k:v.tolist() for k,v in zip(('K1','D1','K2','D2','R','T','R1','R2','P1','P2'),(k1,d1,k2,d2,rotation,t,r1,r2,p1,p2))},
                   'per_pair_rms_px':np.sqrt(np.mean(fit.fun.reshape(len(train),2,-1,2)**2,axis=(2,3))*2).tolist(),
                   'training_geometry':rectification_quality_report(list(observations[:,0]),list(observations[:,1]),k1,d1,k2,d2,r1,r2,p1,p2)}
        candidates.append(candidate)
        print('CANDIDATE',seed,candidate['rms'],candidate['converged'],candidate['K1'][0][0],candidate['problems'],flush=True)
        (OUT/f'{prefix}-seed-{int(seed)}.json').write_text(json.dumps(candidate,indent=2)+'\n',encoding='utf-8')
    selected=min(candidates,key=lambda c:(bool(c['problems']),c['rms']))
    quality=holdout_quality([pairs[i] for i in holdout],recipe=selected,pattern=(8,6),square_mm=17.,image_size=(1920,1080))
    record={'scope':'FIXED_BASELINE_JOINT_DIAGNOSTIC_NOT_ACTIVE','baseline_constraint_mm':60.,'square_mm':17.,
            'math_verification':checks,'training_indices':[i+1 for i in train],'holdout_indices':[i+1 for i in holdout],
            'source_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            'candidates':candidates,'selected':selected,'holdout':quality,
            'automatic_quality_passed':bool(selected['converged'] and not selected['problems'] and selected['rms']<=1.5 and quality['status']=='PASSED')}
    (OUT/f'{prefix}-result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('RESULT',selected['rms'],quality['cross_reprojection_rms_px'],quality['status'],flush=True)


if __name__=='__main__':
    main()
