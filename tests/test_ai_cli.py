"""Command-line wiring: an explicit AI run is distinct from an input dry run."""
from unittest.mock import patch
import json
from pathlib import Path

from pipe_twin.cli import main


def test_prepare_only_does_not_call_provider_or_save(capsys, tmp_path):
    prepared = {"input_hash": "input", "point_map": {"points": []},
                "asset_refs": ["left", "right"], "messages": [{"secret_payload": "data:image/hidden"}]}
    with patch('pipe_twin.ai_inputs.prepare_ai_request', return_value=prepared) as prepare, \
         patch('pipe_twin.ai_workflow.evaluate_ai_scene') as evaluate, \
         patch('pipe_twin.ai_workflow.accept_ai_result') as accept:
        scene = tmp_path / 'manifest.json'
        assert main(['ai-section-evaluate', '--manifest', str(scene), '--prepare-only']) == 0
    prepare.assert_called_once_with(scene)
    evaluate.assert_not_called()
    accept.assert_not_called()
    output = capsys.readouterr().out
    assert 'data:image' not in output
    assert json.loads(output)['provider_called'] is False


def test_explicit_run_saves_only_through_acceptance(capsys, tmp_path):
    result = {'envelope': 'bound'}
    report = {'ai_section_assist': {'status': 'AMBIGUOUS', 'run_status': 'VALID',
                                   'candidates': [{'point_id': 'P01'}], 'model': 'example'}}
    with patch('pipe_twin.ai_workflow.evaluate_ai_scene', return_value=result) as evaluate, \
         patch('pipe_twin.ai_workflow.accept_ai_result', return_value=report) as accept:
        scene = tmp_path / 'manifest.json'
        assert main(['ai-section-evaluate', '--manifest', str(scene)]) == 0
    evaluate.assert_called_once_with(scene)
    accept.assert_called_once_with(scene, result, save=True)
    output = json.loads(capsys.readouterr().out)
    assert output['strict_installation_state_changed'] is False
    assert Path(output['result_path']) == scene.parent / 'ai_section_result.json'
