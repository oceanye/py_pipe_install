"""Diagnostic only: test whether a depth-scale error could explain CAD mismatch.

The fitted scale and candidate identities are NOT used by production analysis,
calibration, measurements, or installation states. No absolute CAD distance or
pipe length is used. A fitted-to-CAD hypothesis is not an independent validation.
"""
from pathlib import Path
from itertools import combinations, product
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.section_diagnostics import section_basis

manifest = json.loads((ROOT/'outputs/tilt-inspection-20261023/captures/elevation-20261023-084738-cbfc8998/manifest.json').read_text(encoding='utf8'))
specs = manifest['analysis']['elevation_auto']['pipes']
basis = section_basis(np.diff(specs[0]['centerline_world_mm'], axis=0)[0])
centers = np.array([np.mean(p['centerline_world_mm'], axis=0)@basis for p in specs])
results = {'diagnostic_only': True, 'applied_to_production': False, 'scale_fitted_to_cad': True,
           'confirms_focal_length_or_pipe_identity': False, 'runs': []}
for name in ('reference-20261022', 'fresh-01'):
    report = json.loads((ROOT/f'outputs/relative-scene-20261025/{name}/section-report.json').read_text(encoding='utf8'))
    observed = report['section_evaluation']['model']['candidates']
    uv = np.array([c['estimated_center_uv_mm'] for c in observed]); uv -= uv.mean(axis=0)
    choices = [[j for j, p in enumerate(specs) if abs(p['nominal_diameter_mm']-c['diameter_mm']) < max(3., .1*p['nominal_diameter_mm'])] for c in observed]
    pairs = list(combinations(range(len(observed)), 2))
    delta = np.array([uv[j]-uv[i] for i, j in pairs])
    trials = []
    for ids in product(*choices):
        if len(set(ids)) != len(ids):
            continue
        model = centers[list(ids)].copy(); model -= model.mean(axis=0)
        expected = np.array([model[j]-model[i] for i, j in pairs])
        squared = np.sum(delta[:, 1]**2*(np.sum(expected**2, axis=1)-delta[:, 0]**2))/np.sum(delta[:, 1]**4)
        if squared <= 0:
            continue
        scale = float(np.sqrt(squared))
        scaled = uv*[1., scale]
        u, _, vt = np.linalg.svd(model.T@scaled)
        errors = np.linalg.norm(model@(u@vt)-scaled, axis=1)
        trials.append({'rms_mm': float(np.sqrt(np.mean(errors**2))), 'v_scale': scale,
                       'candidate_pipe_ids': [specs[i]['pipe_id'] for i in ids],
                       'point_errors_mm': errors.tolist(), 'identity_confirmed': False})
    trials.sort(key=lambda r: r['rms_mm'])
    results['runs'].append({'name': name, 'observation_order': [c['id'] for c in observed],
                            'top_hypotheses': trials[:3]})
(Path(__file__).parent/'depth-scale-diagnostic.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
print(json.dumps(results, ensure_ascii=False))
