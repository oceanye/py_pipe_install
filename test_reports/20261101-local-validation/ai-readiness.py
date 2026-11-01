"""Reproduce AI integration readiness findings without network or credentials.

Reads tracked Python source and runs tiny synthetic in-memory contract probes.
It does not call a model, open a camera, read user settings/keys, or alter code.
Assertions describe the audited revision; a later implementation may fix these
gaps and should replace the expected observations with its new acceptance tests.
"""
from __future__ import annotations

import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipe_twin.ai_section_assist import build_ai_section_assist


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True, encoding='utf-8').strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(result):
    return {
        'run_status': result['run_status'],
        'section_decision': result['section_decision'],
        'candidate_statuses': [row['section_status'] for row in result['candidates']],
        'reason_codes': result['reason_codes'],
        'installation_state': result['installation_state'],
    }


def main():
    commit = git('rev-parse', 'HEAD')
    targets = ('build_model_section_reference', 'build_ai_section_assist', 'attach_ai_section_assist')
    sites = {name: [] for name in targets}
    files = [p for p in git('ls-files', 'pipe_twin', 'scripts').splitlines() if p.endswith('.py')]
    source_hashes = {name: sha(ROOT / name) for name in files}
    for name in files:
        tree = ast.parse((ROOT / name).read_text(encoding='utf-8-sig'), filename=name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                called = (node.func.id if isinstance(node.func, ast.Name)
                          else node.func.attr if isinstance(node.func, ast.Attribute) else None)
                if called in sites:
                    sites[called].append({'file': name, 'line': node.lineno})
    assert not sites['build_model_section_reference']
    assert not sites['attach_ai_section_assist']
    assert sites['build_ai_section_assist'] == [{'file': 'pipe_twin/ai_section_assist.py', 'line': 473}]

    point_map = {
        'mapping_sha256': '1' * 64, 'view_id': 'SYNTHETIC_VIEW',
        'points': [{'point_id': 'P01', 'pipe_id': 'TEST_PIPE',
                    'section_uv_mm': [0., 0.], 'projected_center_px': [20., 20.],
                    'nominal_diameter_mm': 40., 'projection_ambiguity': 'NONE'}],
    }
    response = {'point_assessments': [{'point_id': 'P01', 'section_status': 'SUPPORTED_VISIBLE',
                                      'evidence_quality': 'HIGH', 'observation_refs': ['left:001']} ]}
    report = {
        'inputs': {'left': 'a' * 64, 'right': 'b' * 64, 'model': 'c' * 64},
        'counts': {'INSTALLED': 0, 'NOT_INSTALLED': 0, 'UNKNOWN': 1},
        'pipes': [{'pipe_id': 'TEST_PIPE', 'installation_state': 'UNKNOWN'}],
        'section_evaluation': {'position_assessment': {'pipes': []}},
    }
    originals = copy.deepcopy((point_map, response, report))

    # There is no expected-input-fingerprint parameter or comparison in the
    # contract helper. An arbitrary source value is accepted and echoed.
    expected_hash = hashlib.sha256(json.dumps(report['inputs'], sort_keys=True).encode()).hexdigest()
    submitted_hash = 'f' * 64
    hash_result = build_ai_section_assist(point_map, response, report=report,
        source={'provider': 'SYNTHETIC_LOCAL_PROBE', 'input_hash': submitted_hash})
    assert expected_hash != submitted_hash
    assert hash_result['input_hash'] == submitted_hash
    assert hash_result['run_status'] == 'VALID'

    # Reference strings are type-checked, but there is no lookup against the
    # actual submitted-image/observation catalogue in the helper.
    unknown_refs = ['MISSING_IMAGE:MISSING_OBSERVATION']
    refs_response = copy.deepcopy(response)
    refs_response['point_assessments'][0]['observation_refs'] = unknown_refs
    refs_result = build_ai_section_assist(point_map, refs_response, report=report)
    assert refs_result['run_status'] == 'VALID'
    assert refs_result['candidates'][0]['observation_refs'] == unknown_refs

    # The unresolved control plane is copied as metadata. Without a per-pipe
    # evidence reason, that metadata does not cap the supplied visible status.
    control = {
        'schema_version': 'pooled-control-section-v1',
        'source': 'PIPE_SEGMENT_AGGREGATION_COMMON_SUPPORT_MIDPOINT',
        'status': 'UNRESOLVED', 'station_mm': None, 'interval_mm': None,
        'reason_codes': ['COMMON_AXIAL_SUPPORT_NO_INTERSECTION'],
    }
    control_report = copy.deepcopy(report)
    control_report['section_evaluation']['control_section'] = control
    control_result = build_ai_section_assist(point_map, response, report=control_report)
    assert control_result['control_section']['status'] == 'UNRESOLVED'
    assert control_result['section_decision'] == 'CANDIDATES_READY'
    assert control_result['candidates'][0]['section_status'] == 'SUPPORTED_VISIBLE'
    assert (point_map, response, report) == originals
    assert all(sha(ROOT / name) == digest for name, digest in source_hashes.items())
    assert git('rev-parse', 'HEAD') == commit

    evidence = {
        'schema_version': 'ai-integration-readiness-audit-v1',
        'tested_commit': commit,
        'recorded_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'LOCAL_STATIC_CALL_GRAPH_AND_SYNTHETIC_MEMORY_PROBES_ONLY',
        'ready_with_api_key_only': False,
        'external_ai_called': False, 'network_used': False,
        'user_credentials_read_or_saved': False, 'camera_opened': False,
        'production_code_changed': False,
        'production_python_files_scanned': len(files),
        'tracked_production_call_sites': sites,
        'call_site_scan_limit': 'Direct function-name/attribute AST calls; not a general dynamic call-graph proof.',
        'current_gui_chain': [
            'elevation_gui.py:713 submit(True)',
            'elevation_gui.py:2331 _worker',
            'elevation_gui.py:2347 analyze_stereo_capture',
            'elevation_auto.py:1030 local installation_assist',
            'elevation_auto.py:1121 atomic local report save',
            'elevation_gui.py:1552 read-only AI table refresh',
        ],
        'implemented': [
            'Fixed model point-map builder (not wired to production GUI/analysis).',
            'Section-only response validation, fixed-ID checks and evidence status caps.',
            'Deep-copy attach helper preserving strict report states (not wired to producer).',
            'Read-only AI result table and explicit not-run fallback.',
        ],
        'missing_for_end_to_end_ai': [
            'GUI/CLI AI task trigger and per-request asynchronous lifecycle.',
            'Current-image/section-reference request packaging and prompt.',
            'Provider adapter, endpoint/model/credential configuration and network error handling.',
            'Actual request-response parsing with bound assets and task fingerprints.',
            'Production attach, atomic AI report persistence and reopen/display integration.',
            'Live-provider end-to-end acceptance on field images.',
        ],
        'probes': {
            'source_hash_not_verified': {
                'reference_input_hash_for_probe': expected_hash,
                'supplied_input_hash': submitted_hash,
                'echoed_input_hash': hash_result['input_hash'], 'observed': summary(hash_result),
                'source_lines': ['pipe_twin/ai_section_assist.py:260', 'pipe_twin/ai_section_assist.py:331'],
            },
            'unknown_observation_reference_not_resolved': {
                'reference_not_present_in_inputs': unknown_refs,
                'echoed_references': refs_result['candidates'][0]['observation_refs'],
                'observed': summary(refs_result),
                'source_lines': ['pipe_twin/ai_section_assist.py:374', 'pipe_twin/ai_section_assist.py:377',
                                 'pipe_twin/ai_section_assist.py:394'],
            },
            'unresolved_control_plane_does_not_cap_candidate': {
                'control_section': control_result['control_section'], 'observed': summary(control_result),
                'source_lines': ['pipe_twin/ai_section_assist.py:326', 'pipe_twin/ai_section_assist.py:380'],
            },
        },
        'existing_test_scope': {
            'static_response_fixture': 'tests/test_ai_section_assist.py:40',
            'provider_names_are_fixture_metadata': 'tests/test_ai_section_assist.py:63',
            'gui_direct_report_injection': 'tests/test_elevation_gui.py:203',
            'previous_full_test_evidence_commit': 'cb04f90d43a85fcb4f4aa0d5ca1c5afaaa258abc',
            'previous_evidence_precedes_ai_contract_and_control_plane_commits': True,
        },
        'interpretation': (
            'These observations identify safeguards still needed at the future integration boundary. '
            'No current production path sends images to an external AI, and these probes do not '
            'demonstrate a live-provider failure or validate recognition accuracy.'),
        'production_source_sha256': source_hashes,
    }
    destination = HERE / 'ai-readiness.json'
    destination.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'report': str(destination.relative_to(ROOT)), 'commit': commit,
                      'ready_with_api_key_only': False, 'synthetic_probes_reproduced': 3,
                      'external_ai_called': False, 'production_code_changed': False}, ensure_ascii=False))


if __name__ == '__main__':
    main()
