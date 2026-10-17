"""Independent sparse correspondence checks on the new raw pipe scene."""
from pathlib import Path
import json
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/verification-20261017'
cv2.setNumThreads(2)
images=[cv2.imdecode(np.fromfile(OUT/'raw-burst'/f'019_{role}.png',np.uint8),cv2.IMREAD_COLOR) for role in ('left','right')]
sift=cv2.SIFT_create(nfeatures=5000)
features=[sift.detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),None) for image in images]
matcher=cv2.BFMatcher()
forward=matcher.knnMatch(features[0][1],features[1][1],k=2)
backward=matcher.knnMatch(features[1][1],features[0][1],k=2)
reverse={a.queryIdx:a.trainIdx for a,b in backward if a.distance<.65*b.distance}
matches=[a for a,b in forward if a.distance<.65*b.distance and reverse.get(a.trainIdx)==a.queryIdx]
left=np.array([features[0][0][m.queryIdx].pt for m in matches],np.float64)
right=np.array([features[1][0][m.trainIdx].pt for m in matches],np.float64)
f,mask=cv2.findFundamentalMat(left,right,cv2.USAC_MAGSAC,1.,.999,10000)
assert f is not None and mask is not None
inliers=mask.ravel().astype(bool)
result={'scope':'SPARSE_SCENE_GEOMETRY_DIAGNOSTIC_NOT_PIPE_MEASUREMENT',
        'ratio_and_mutual_matches':len(matches),'raw_fundamental_inliers':int(inliers.sum()),
        'fundamental_matrix':f.tolist(),'calibrations':{}}
for mode in ('active','fixed60'):
    recipe=json.loads((OUT/mode/'calibration.json').read_text(encoding='utf-8'))['rectification_recipe']
    mapped=[]
    for idx,points in enumerate((left,right),1):
        mapped.append(cv2.undistortPoints(points.reshape(-1,1,2),
            np.asarray(recipe[f'K{idx}']),np.asarray(recipe[f'D{idx}']),
            R=np.asarray(recipe[f'R{idx}']),P=np.asarray(recipe[f'P{idx}'])).reshape(-1,2))
    # Compare the same raw-image correspondence set for both calibrations.
    dy=(mapped[0]-mapped[1])[inliers,1]
    dx=(mapped[0]-mapped[1])[inliers,0]
    inside=np.ones(len(left),bool)
    for p in mapped:
        inside &= (p[:,0]>=0)&(p[:,0]<1920)&(p[:,1]>=0)&(p[:,1]<1080)
    result['calibrations'][mode]={'all_inlier_vertical_rms_px':float(np.sqrt(np.mean(dy**2))),
        'all_inlier_vertical_abs_p95_px':float(np.percentile(np.abs(dy),95)),
        'median_vertical_signed_px':float(np.median(dy)),
        'disparity_percentiles_5_50_95_px':np.percentile(dx,[5,50,95]).tolist(),
        'inside_both_rectified_count':int((inside&inliers).sum()),
        'correspondence_within_1_row_fraction':float(np.mean(np.abs(dy)<=1)),
        'disparity_over_255_fraction':float(np.mean(dx>255))}
(OUT/'scene-geometry.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
