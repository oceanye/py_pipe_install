"""Qualitative assistance replay with user-provided purchase-size hints.

Create a new portable scene from the same archived photos. Original images,
scene, camera calibration and default pointer are never edited.
"""
from pathlib import Path
import copy
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import cv2
from pipe_twin.elevation_dataset import load_elevation_dataset, create_elevation_dataset
from pipe_twin.stereo_analyzer import analyze_stereo_capture

SOURCE = ROOT/'outputs/elevation_workbench/captures/elevation-20261030-183635-b6b7b875/manifest.json'
OUT = ROOT/'outputs/installation-assist-20261031'
HINTS = {'appearance_diameter_hints': {'BLUE': 50., 'RED': 40., 'WHITE': 25.},
         'hint_source': 'USER_STATED_PURCHASE_DIAMETERS'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(views):
    return {role: {k: v for k, v in view.items() if k.startswith('capture_') or k == 'side_by_side_order'}
            for role, view in views.items()}


def main():
    cv2.setNumThreads(2)
    loaded = load_elevation_dataset(SOURCE)
    manifest = loaded['manifest']; group = manifest['capture']['capture_groups'][-1]
    views = group['views']
    protected = {SOURCE, SOURCE.with_name('report.json'),
        ROOT/'outputs/measurement_workbench/calibration_current.json',
        ROOT/'outputs/measurement_workbench/workbench_profile.json',
        ROOT/'outputs/elevation_workbench/last_scene.json'}
    for record in [group, *group.get('temporal_frames', [])]:
        for view in record['views'].values():
            path = SOURCE.parent/view['path']
            assert sha(path) == view['sha256']
            protected.add(path)
    protected_before = {str(p.relative_to(ROOT)): sha(p) for p in sorted(protected)}
    settings = copy.deepcopy(loaded['registration_settings']); settings['installation_assist'] = HINTS
    frames = [{**{f'{r}_path': str(SOURCE.parent/f['views'][r]['path']) for r in ('left', 'right')},
               **{f'{r}_time': f['views'][r]['captured_at'] for r in ('left', 'right')},
               'sha256': {r: f['views'][r]['sha256'] for r in ('left', 'right')},
               'timestamp_sources': {r: f['views'][r]['timestamp_source'] for r in ('left', 'right')},
               'camera_capture_provenance': provenance(f['views'])} for f in group.get('temporal_frames', [])]
    OUT.mkdir(parents=True, exist_ok=True)
    scene = create_elevation_dataset(output_root=OUT/'scenes', calibration=loaded['calibration'],
        left_path=loaded['left_path'], right_path=loaded['right_path'], left_time=loaded['left_time'],
        right_time=loaded['right_time'], pipe_specs=loaded['pipe_specs'], pair_confirmed=True,
        model_path=SOURCE.parent/manifest['model']['path'], stl_unit='millimeter',
        rectification_recipe=loaded['rectification_recipe'], camera_capture_provenance=provenance(views),
        timestamp_sources={r: views[r]['timestamp_source'] for r in ('left', 'right')},
        registration_settings=settings, scope_settings=loaded['scope_settings'], temporal_frames=frames,
        analysis_settings={k: v for k, v in manifest['analysis'].items() if k not in ('mode', 'elevation_auto')})
    started = time.monotonic()
    report = analyze_stereo_capture(scene, report_output_path=scene.with_name('report.json'),
        evidence_dir=scene.parent/'evidence', progress=lambda value: print(value, flush=True))
    assert protected_before == {str(p.relative_to(ROOT)): sha(p) for p in sorted(protected)}
    data = {'source_scene': str(SOURCE.relative_to(ROOT)), 'derived_scene': str(scene.relative_to(ROOT)),
            'source_sha256': protected_before, 'sources_unchanged': True, 'new_capture': False,
            'seconds': time.monotonic()-started, 'purchase_hints': HINTS,
            'installation_assist': report['installation_assist'], 'strict_counts': report['counts'],
            'strict_position_counts': report['position_assessment']['counts'],
            'temporal_fusion': {k: (report.get('temporal_fusion') or {}).get(k)
                               for k in ('status', 'accepted_frames', 'requested_frames', 'reason_codes')},
            'scope': 'QUALITATIVE_IDENTIFICATION_USING_STATED_PRIORS_NOT_INDEPENDENT_DIMENSION_VALIDATION'}
    (OUT/'latest-scene.json').write_text(json.dumps({'manifest_path': str(scene)}, indent=2)+'\n', encoding='utf-8')
    Path(__file__).with_name('evidence.json').write_text(json.dumps(data, ensure_ascii=False,
        indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print('SCENE', scene, flush=True)
    print(json.dumps(report['installation_assist'], ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
