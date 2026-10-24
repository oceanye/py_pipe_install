"""Replay the latest field pair with a central image band; keep originals intact."""
from pathlib import Path
import hashlib
import json
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.axial_selection import normalize_axial_band, band_geometry, band_sampling_masks, band_overlay
from pipe_twin.elevation_dataset import load_elevation_dataset, create_elevation_dataset
from pipe_twin.elevation_auto import analyze_elevation_auto_manifest
from pipe_twin.pooled_sections import aggregate_image
from pipe_twin.section_display import section_summary_lines
from pipe_twin.capture_gui import _CAPTURE_PROVENANCE_KEYS

SOURCE = ROOT/'outputs/elevation_workbench/captures/elevation-20261022-103057-545d1d88/manifest.json'
OUTPUT = ROOT/'outputs/axial-selection-20261022'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    loaded = load_elevation_dataset(SOURCE)
    manifest = loaded['manifest']
    views = manifest['capture']['capture_groups'][-1]['views']
    paths = [SOURCE, SOURCE.parent/'report.json', ROOT/'outputs/measurement_workbench/calibration_current.json',
             loaded['left_path'], loaded['right_path']]
    before = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    band = normalize_axial_band({'image_size': [1920, 1080], 'axis_segment_px': [[900, 600], [1420, 558]]})
    package = create_elevation_dataset(output_root=OUTPUT/'captures', calibration=loaded['calibration'],
        left_path=loaded['left_path'], right_path=loaded['right_path'], left_time=loaded['left_time'], right_time=loaded['right_time'],
        pipe_specs=loaded['pipe_specs'], pair_confirmed=True, model_path=SOURCE.parent/manifest['model']['path'],
        stl_unit=manifest['model'].get('source_unit'), rectification_recipe=loaded['rectification_recipe'],
        camera_capture_provenance={r: {k: v[k] for k in _CAPTURE_PROVENANCE_KEYS if k in v} for r, v in views.items()},
        timestamp_sources={r: v.get('timestamp_source', 'MANIFEST_OPERATOR_CONFIRMED') for r, v in views.items()},
        analysis_settings={k: v for k, v in manifest['analysis'].items() if k not in ('mode', 'elevation_auto')},
        registration_settings=loaded['registration_settings'] | {'aggregation_band': band}, scope_settings=loaded['scope_settings'])
    started = time.perf_counter()
    report = analyze_elevation_auto_manifest(package, report_output_path=package.parent/'report.json',
        evidence_dir=package.parent/'evidence', progress=lambda msg: print(msg, flush=True))
    elapsed = time.perf_counter()-started
    stage = report['section_evaluation']
    previous = json.loads((SOURCE.parent/'report.json').read_text(encoding='utf-8'))['section_evaluation']
    image = cv2.imdecode(np.fromfile(loaded['left_path'], np.uint8), cv2.IMREAD_COLOR)
    masks, _ = band_sampling_masks(band, image.shape, image.shape)
    items, _ = aggregate_image(image, 'left', sampling_mask=masks['left'], axis_hint_image=band_geometry(band)[1])
    origin, axis, length = band_geometry(band)
    sample_checks = []
    for item in items:
        for edge in item['pooled_edges']:
            if not len(edge['points']):
                continue
            station = (edge['points']-origin)@axis
            assert station.min() >= 0 and station.max() <= length
            sample_checks.append({'candidate_id': item['candidate_id'], 'edge': edge['edge'],
                'count': len(station), 'station_range_px': [float(station.min()), float(station.max())]})
    for line in band_overlay(band)['boundary_lines_px']:
        if len(line) == 2:
            cv2.line(image, tuple(np.round(line[0]).astype(int)), tuple(np.round(line[1]).astype(int)), (71, 179, 255), 3)
    cv2.polylines(image, [np.asarray(band['axis_segment_px'], np.int32)], False, (122, 224, 255), 3)
    cv2.imencode('.png', image)[1].tofile(OUTPUT/'selected-left.png')
    assert {str(p.relative_to(ROOT)): sha(p) for p in paths} == before
    def brief(value):
        return {k: value.get(k) for k in ('status', 'axis', 'boundary_pair_count', 'conditional_section_count',
                                         'candidates', 'position_assessment', 'aggregation_band')} | {
            'views': {r: [{k: x.get(k) for k in ('candidate_id', 'appearance', 'region_px', 'width_px', 'boundary_status', 'reason_codes')}
                          for x in xs] for r, xs in value.get('views', {}).items()}}
    evidence = {'source_manifest': str(SOURCE.relative_to(ROOT)), 'new_manifest': str(package.relative_to(ROOT)),
        'new_camera_capture': False, 'band_source': 'ASSISTANT_OFFLINE_REPLAY_EXAMPLE', 'band': band,
        'originals_unchanged': True, 'original_sha256': before, 'production_seconds': elapsed,
        'legacy_counts': report['counts'], 'position_counts': report['position_assessment']['counts'],
        'previous_automatic': brief(previous), 'selected_band': brief(stage), 'retained_left_samples_inside_band': sample_checks}
    (Path(__file__).parent/'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    (OUTPUT/'gui-summary.txt').write_text('\n'.join(section_summary_lines(report, detailed=True)), encoding='utf-8')
    print(json.dumps({'manifest': str(package), 'seconds': elapsed, 'counts': report['counts'],
        'pairs': stage.get('boundary_pair_count'), 'sections': stage.get('conditional_section_count'), 'axis': stage.get('axis'),
        'views': evidence['selected_band']['views']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
