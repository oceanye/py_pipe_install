"""Hold observed pixels, 3D axis and solver fixed; change only block assignment."""
from pathlib import Path
import json
import math
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.elevation_dataset import load_elevation_dataset
from pipe_twin.stereo_analyzer import _calibration_from_manifest
from pipe_twin.axial_selection import band_sampling_masks
from pipe_twin.pooled_sections import aggregate_image, _bounded
from pipe_twin.projection_diagnostics import fit_silhouettes
from pipe_twin.axial_consistency import synchronize_axial_blocks, conditional_axial_stations


def difference(fits):
    return {'diameter_difference_mm': abs(fits[0]['trial_diameter_mm']-fits[1]['trial_diameter_mm']),
        'center_difference_mm': float(np.linalg.norm(np.array(fits[0]['section_center_parameters_mm'])-
                                                     fits[1]['section_center_parameters_mm']))}


def main():
    evidence = json.loads((Path(__file__).parent/'evidence.json').read_text(encoding='utf-8'))
    loaded = load_elevation_dataset(ROOT/evidence['source_manifest'])
    calibration = _calibration_from_manifest(loaded['calibration'])
    k, kr = calibration.left.rectified_intrinsic, calibration.right.rectified_intrinsic
    images = {r: cv2.imdecode(np.fromfile(loaded[r+'_path'], np.uint8), cv2.IMREAD_COLOR) for r in ('left', 'right')}
    masks, audit = band_sampling_masks(evidence['band'], images['left'].shape, images['right'].shape,
        disparity_range=evidence['current']['aggregation_band']['disparity_range_px'])
    views = {role: aggregate_image(image, role, sampling_mask=masks[role], axis_hint_image=audit['axis_image'])[0]
             for role, image in images.items()}
    views = {role: {c['candidate_id']: c for c in rows} for role, rows in views.items()}
    axis = evidence['current']['axis']['axis']
    rows = []
    for candidate in evidence['current']['candidates']:
        a, b = views['left'][candidate['left_candidate_id']], views['right'][candidate['right_candidate_id']]
        observations = [*a['pooled_edges'], *[{**o, 'points': o['points']+k[:2, 2]-kr[:2, 2]} for o in b['pooled_edges']]]
        reference = a['aggregation']['boundaries'][0]['reference_u_px']
        initial = fit_silhouettes(_bounded(observations), k, calibration.baseline_mm, axis, reference)
        shared, partitions, support = synchronize_axial_blocks(observations, k, calibration.baseline_mm, initial)
        legacy, starts = [[], []], []
        for observation in observations:
            p = observation['points']; first = p[:, 0].min()
            ids = np.floor((p[:, 0]-first)/128).astype(int)
            starts.append({'role': observation['role'], 'edge': observation['edge'],
                           'legacy_first_station_mm': float(conditional_axial_stations(observation, k, calibration.baseline_mm, initial).min())})
            for parity in (0, 1):
                legacy[parity].append({**observation, 'points': p[ids % 2 == parity]})
        def fitted(groups):
            return [fit_silhouettes(_bounded(obs), k, calibration.baseline_mm, axis, reference) for obs in groups]
        rows.append({'candidate_id': candidate['candidate_id'], 'independent_pixel_origin_test': difference(fitted(legacy)),
            'common_station_test': difference(fitted(partitions)), 'legacy_boundary_station_starts': starts,
            'common_station_support': support})
    hints = []
    for delta in (-2., 2.):
        angle = math.atan2(*audit['axis_image'][::-1])+math.radians(delta)
        for role, image in images.items():
            changed, _ = aggregate_image(image, role, sampling_mask=masks[role], axis_hint_image=[math.cos(angle), math.sin(angle)])
            identical = len(changed) == len(views[role])
            for candidate in changed:
                old = views[role].get(candidate['candidate_id'])
                identical = identical and old is not None and all(np.array_equal(a['points'], b['points'])
                    for a, b in zip(candidate['pooled_edges'], old['pooled_edges']))
            hints.append({'role': role, 'hint_delta_deg': delta, 'same_sampling_mask': True, 'retained_edges_identical': bool(identical)})
    result = {'same_pixels_axis_intrinsics_solver': True, 'nominal_diameters_used': False, 'hint_only_checks': hints,
        'scope': 'PARTITION_CORRECTION_NOT_A_TOTAL_ACCURACY_CLAIM', 'candidates': rows}
    (Path(__file__).parent/'partition-ablation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps([{'candidate': r['candidate_id'], 'old': r['independent_pixel_origin_test'], 'new': r['common_station_test']}
                      for r in rows], ensure_ascii=False))


if __name__ == '__main__':
    main()
