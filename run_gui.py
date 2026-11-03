"""Validate the GUI environment before launching this checkout's office client.

`run_gui.bat --check` checks Python/dependencies without opening any camera,
GUI, or service. A nested checkout can share an ancestor project's environment.
"""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PROBE = """
import importlib, json, pathlib, sys
if sys.version_info < (3, 12):
    raise RuntimeError('Python 3.12 or newer is required')
modules = ('numpy', 'scipy', 'cv2', 'ezdxf', 'rhino3dm', 'tkinter')
versions = {}
for name in modules:
    module = importlib.import_module(name)
    versions[name] = str(getattr(module, '__version__', getattr(module, 'TkVersion', 'available')))
import pipe_twin
print(json.dumps({'python': sys.executable, 'version': sys.version.split()[0],
                  'package': str(pathlib.Path(pipe_twin.__file__).resolve()),
                  'dependencies': versions}, ensure_ascii=False))
"""


def find_python(root: Path, environ=None) -> Path:
    environ = os.environ if environ is None else environ
    override = environ.get('PIPE_TWIN_PYTHON')
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f'PIPE_TWIN_PYTHON 指定的 Python 不存在：{path}')
        return path
    for project in (root, *root.parents):
        if not (project / 'pipe_twin' / '__init__.py').is_file():
            continue
        for relative in ('.venv/Scripts/python.exe', '.venv/bin/python'):
            candidate = project / relative
            if candidate.is_file():
                return candidate
    return Path(sys.executable)


def check_environment(python: Path, root: Path) -> dict:
    env = dict(os.environ, PYTHONUTF8='1')
    # A GUI launched from a shell/editor must import this checkout, not PYTHONPATH.
    env.pop('PYTHONPATH', None)
    result = subprocess.run([str(python), '-c', PROBE], cwd=root, env=env,
                            capture_output=True, text=True, encoding='utf-8',
                            errors='replace', timeout=45,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    if result.returncode:
        raise ValueError((result.stderr or result.stdout).strip() or f'Python 检查失败：{result.returncode}')
    checked = json.loads(result.stdout)
    if Path(checked['package']).resolve() != (root / 'pipe_twin' / '__init__.py').resolve():
        raise ValueError('Python 加载了其他目录的 pipe_twin，请检查环境。')
    return checked


def main(arguments=None, *, root=ROOT) -> int:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    root = Path(root).resolve()
    python = None
    try:
        python = find_python(root)
        checked = check_environment(python, root)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(f'启动环境检查失败。\n项目：{root}\nPython：{python or "未找到"}\n{error}', file=sys.stderr)
        print('请安装包含 Tcl/Tk 的 Python 3.12，并在项目目录执行：\n'
              '  py -3.12 -m venv .venv\n'
              '  .venv\\Scripts\\python.exe -m pip install -r requirements.txt', file=sys.stderr)
        return 1
    if arguments == ['--check']:
        print(json.dumps({'ok': True, 'project': str(root), **checked}, ensure_ascii=False, indent=2))
        return 0
    log_path = root / 'outputs' / 'logs' / 'gui-startup.log'
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONUTF8='1', PYTHONUNBUFFERED='1')
        env.pop('PYTHONPATH', None)
        with log_path.open('ab') as log:
            log.write(f'\n[{datetime.now().isoformat(timespec="seconds")}] Project: {root}\nPython: {python}\n'.encode('utf-8'))
            log.flush()
            process = subprocess.Popen([str(python), '-m', 'pipe_twin', 'office-client', *arguments],
                cwd=root, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0), close_fds=True)
        time.sleep(1)
        exit_code = process.poll()
        if exit_code not in (None, 0):
            print(f'GUI 启动失败，请查看：{log_path}', file=sys.stderr)
            return 1
        print((f'已启动当前项目 GUI：{root}' if exit_code is None else '命令已完成。')
              + f'\n启动日志：{log_path}')
        return 0
    except OSError as error:
        print(f'GUI 启动失败：{error}\n启动日志：{log_path}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
