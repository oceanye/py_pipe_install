"""Reproduce independent section results on archived and fresh real images."""
from pathlib import Path
import hashlib
import json
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.calibration_wizard import rectifier_for_calibration
from pipe_twin.stereo_analyzer import _calibration_from_manifest
from pipe_twin.pooled_sections import analyze_pooled_sections
from pipe_twin.section_position import assess_pooled_sections
from pipe_twin.section_model import write_section_artifacts
from pipe_twin.section_display import boundary_overlay_segments


def main():
    output = ROOT/'outputs/relative-scene-20261025'
    reference = ROOT/'outputs/tilt-inspection-20261023/captures/elevation-20261023-084738-cbfc8998'
    manifest = json.loads((reference/'manifest.json').read_text(encoding='utf8'))
    settings = manifest['analysis']['elevation_auto']['registration']
    specs = manifest['analysis']['elevation_auto']['pipes']
    calibration_path = ROOT/'outputs/measurement_workbench/calibration_current.json'
    bundle = json.loads(calibration_path.read_text(encoding='utf8'))
    calibration = _calibration_from_manifest(bundle['stereo_calibration'])
    calibration_hash = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    pairs = [('reference-20261022', {r: reference/f'photos/{r}.png' for r in ('left', 'right')})]
    capture_path = output/'fresh-capture/capture.json'
    capture = json.loads(capture_path.read_text(encoding='utf8'))
    rectifier = rectifier_for_calibration(bundle['stereo_calibration'], bundle)
    for index, pair in enumerate(capture['captures'], 1):
        paths = {}
        for role in ('left', 'right'):
            source = capture_path.parent/pair[role]['path']
            assert hashlib.sha256(source.read_bytes()).hexdigest() == pair[role]['sha256']
            rectified = rectifier.rectify(role, cv2.imread(str(source)))
            paths[role] = output/f'fresh-{index:02d}-{role}.png'
            cv2.imwrite(str(paths[role]), rectified)
        pairs.append((f'fresh-{index:02d}', paths))
    evidence = {'calibration_sha256': calibration_hash, 'fresh_capture_status': capture['status'],
                'fresh_capture_times': [p['left']['captured_at'] for p in capture['captures']], 'runs': []}
    for label, paths in pairs:
        hashes = {r: hashlib.sha256(p.read_bytes()).hexdigest() for r, p in paths.items()}
        images = {r: cv2.imread(str(p)) for r, p in paths.items()}
        original = analyze_pooled_sections(images['left'], images['right'], calibration,
                                           aggregation_band=settings.get('aggregation_band'))
        refined = analyze_pooled_sections(images['left'], images['right'], calibration,
                                          aggregation_band=settings.get('aggregation_band'), refine_epipolar=True)
        assess_pooled_sections(refined, specs, calibration, capture_healthy=True, settings=settings, matching={})
        report = {'section_evaluation': refined, 'position_assessment': refined['position_assessment']}
        directory = output/label; directory.mkdir(parents=True, exist_ok=True)
        (directory/'section-report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        if refined['model']:
            write_section_artifacts(refined['model'], directory/'model')
        for role, image in images.items():
            for x1, y1, x2, y2 in boundary_overlay_segments(report, role):
                cv2.line(image, (round(x1), round(y1)), (round(x2), round(y2)), (239, 232, 103), 2)
            cv2.imwrite(str(directory/f'{role}-boundaries.png'), image)
        assert all(hashlib.sha256(p.read_bytes()).hexdigest() == hashes[r] for r, p in paths.items())
        evidence['runs'].append({'label': label, 'image_sha256': hashes,
            'before_sections': original.get('conditional_section_count', 0),
            'after_sections': refined.get('conditional_section_count', 0),
            'epipolar_refinement': refined['epipolar_refinement'],
            'axis_camera': refined.get('axis'), 'position_counts': refined['position_assessment']['counts'],
            'registration_status': refined['registration']['status'],
            'candidates': [{'id': c['candidate_id'], 'diameter_mm': c['fit'].get('trial_diameter_mm'),
                            'subset': c['window_sensitivity'], 'reason_codes': c['reason_codes']}
                           for c in refined['candidates']],
            'nominal_dimensions_used_in_fitting': refined['nominal_dimensions_used_in_fitting']})
        print(label, evidence['runs'][-1], flush=True)
    assert calibration_hash == hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    (Path(__file__).parent/'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
