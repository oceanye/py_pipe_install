"""Launcher selection and isolation; no physical camera or services."""
from pathlib import Path
import subprocess
import sys
from unittest import mock

import pytest

import run_gui


def checkout(path, *, environment=False):
    package = path / 'pipe_twin'
    package.mkdir(parents=True)
    (package / '__init__.py').write_text('', encoding='utf-8')
    if environment:
        python = path / '.venv/Scripts/python.exe'
        python.parent.mkdir(parents=True)
        python.touch()
        return python


def test_nested_checkout_uses_parent_environment_without_switching_project(tmp_path):
    parent_python = checkout(tmp_path, environment=True)
    nested = tmp_path / 'outputs' / '开发 副本'
    checkout(nested)
    assert run_gui.find_python(nested, {}) == parent_python
    local_python = nested / '.venv/Scripts/python.exe'
    local_python.parent.mkdir(parents=True)
    local_python.touch()
    assert run_gui.find_python(nested, {}) == local_python


def test_explicit_environment_and_missing_override(tmp_path):
    python = checkout(tmp_path, environment=True)
    assert run_gui.find_python(tmp_path, {'PIPE_TWIN_PYTHON': str(python)}) == python
    with pytest.raises(ValueError, match='不存在'):
        run_gui.find_python(tmp_path, {'PIPE_TWIN_PYTHON': str(tmp_path / 'missing.exe')})


def test_unrelated_parent_environment_is_not_selected(tmp_path):
    unrelated = tmp_path / '.venv/Scripts/python.exe'
    unrelated.parent.mkdir(parents=True)
    unrelated.touch()
    nested = tmp_path / 'project'
    checkout(nested)
    assert run_gui.find_python(nested, {}) == Path(sys.executable)


def test_check_only_never_launches_gui_or_creates_log(tmp_path, capsys):
    with mock.patch.object(run_gui, 'find_python', return_value=Path(sys.executable)), \
         mock.patch.object(run_gui, 'check_environment', return_value={'version': '3.12'}), \
         mock.patch.object(run_gui.subprocess, 'Popen') as launch:
        assert run_gui.main(['--check'], root=tmp_path) == 0
    launch.assert_not_called()
    assert '"ok": true' in capsys.readouterr().out
    assert not (tmp_path / 'outputs').exists()


def test_failure_reports_project_and_install_command_without_launch(tmp_path, capsys):
    with mock.patch.object(run_gui, 'check_environment', side_effect=ValueError('missing cv2')), \
         mock.patch.object(run_gui.subprocess, 'Popen') as launch:
        assert run_gui.main([], root=tmp_path) == 1
    launch.assert_not_called()
    error = capsys.readouterr().err
    assert str(tmp_path) in error and 'missing cv2' in error and 'requirements.txt' in error


def test_normal_start_uses_current_checkout_and_preserves_arguments(tmp_path, monkeypatch):
    monkeypatch.setenv('PYTHONPATH', 'some-other-checkout')
    process = mock.Mock(poll=mock.Mock(return_value=None))
    with mock.patch.object(run_gui, 'check_environment', return_value={}), \
         mock.patch.object(run_gui, 'find_python', return_value=Path(sys.executable)), \
         mock.patch.object(run_gui.subprocess, 'Popen', return_value=process) as launch, \
         mock.patch.object(run_gui.time, 'sleep'):
        assert run_gui.main(['--capture-port', '8870'], root=tmp_path) == 0
    assert launch.call_args.args[0] == [sys.executable, '-m', 'pipe_twin', 'office-client', '--capture-port', '8870']
    assert launch.call_args.kwargs['cwd'] == tmp_path
    assert 'PYTHONPATH' not in launch.call_args.kwargs['env']
    assert (tmp_path / 'outputs/logs/gui-startup.log').is_file()


def test_dependency_probe_rejects_wrong_checkout(tmp_path):
    result = subprocess.CompletedProcess([], 0, '{"package": "wrong/pipe_twin/__init__.py"}', '')
    with mock.patch.object(run_gui.subprocess, 'run', return_value=result):
        with pytest.raises(ValueError, match='其他目录'):
            run_gui.check_environment(Path(sys.executable), tmp_path)


def test_early_process_failure_reports_log(tmp_path, capsys):
    process = mock.Mock(poll=mock.Mock(return_value=1))
    with mock.patch.object(run_gui, 'check_environment', return_value={}), \
         mock.patch.object(run_gui.subprocess, 'Popen', return_value=process), \
         mock.patch.object(run_gui.time, 'sleep'):
        assert run_gui.main([], root=tmp_path) == 1
    assert 'gui-startup.log' in capsys.readouterr().err
