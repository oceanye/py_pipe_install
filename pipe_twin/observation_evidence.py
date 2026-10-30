"""Separate reconstructed cylinder geometry from visible surface-strip hints.

A metric-cylinder grade describes the geometric fit, not calibration accuracy,
CAD identity, installation, or repeatability. Those gates remain independent.
"""
from __future__ import annotations

REVISION = 'local-observation-evidence-v1'
CYLINDER_BASES = frozenset({'DEPTH_ONLY_LOCAL_CYLINDER', 'IMAGE_GUIDED_STEREO_CYLINDER',
                           'DEPTH_AND_OPTIONAL_RGB_LOCAL_CYLINDER'})


def observation_evidence(observation):
    if (observation.get('parallel_local')
            or observation.get('diameter_source') == 'STEREO_LOCAL_RADIAL_P95_AND_PROJECTED_CHORD'):
        grade = 'SURFACE_STRIP_CANDIDATE'
        reasons = ['SURFACE_STRIP_AXIS_CENTER_UNRESOLVED', 'SURFACE_STRIP_DIAMETER_NOT_FREE_CYLINDER_FIT']
    elif (observation.get('diameter_source') == 'ROBUST_OUTER_CYLINDER_FIT_VISIBLE_SURFACE'
          and observation.get('observation_basis') in CYLINDER_BASES):
        grade, reasons = 'METRIC_CYLINDER', []
    else:
        grade, reasons = 'UNVERIFIED_OBSERVATION', ['LOCAL_GEOMETRY_EVIDENCE_UNVERIFIED']
    resolved = grade == 'METRIC_CYLINDER'
    return {'revision': REVISION, 'grade': grade, 'axis_center_resolved': resolved,
            'free_diameter_resolved': resolved, 'reason_codes': reasons,
            'installation_state_from_geometry_alone': False}


def metric_cylinder(observation):
    return observation_evidence(observation)['grade'] == 'METRIC_CYLINDER'
