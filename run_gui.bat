@echo off
setlocal DisableDelayedExpansion
chcp 65001 >nul
set "PYTHONUTF8=1"
cd /d "%~dp0"
rem Bootstrap with the project environment, a parent checkout, or system Python.
rem run_gui.py validates dependencies and launches from THIS checkout's directory.
if defined PIPE_TWIN_PYTHON goto explicit_python
for %%D in ("%~dp0.") do set "PIPE_TWIN_SEARCH=%%~fD"
:find_environment
if exist "%PIPE_TWIN_SEARCH%\pipe_twin\__init__.py" if exist "%PIPE_TWIN_SEARCH%\.venv\Scripts\python.exe" (
    set "PIPE_TWIN_BOOTSTRAP=%PIPE_TWIN_SEARCH%\.venv\Scripts\python.exe"
    goto launch
)
for %%D in ("%PIPE_TWIN_SEARCH%\..") do set "PIPE_TWIN_PARENT=%%~fD"
if /I "%PIPE_TWIN_SEARCH%"=="%PIPE_TWIN_PARENT%" goto system_python
set "PIPE_TWIN_SEARCH=%PIPE_TWIN_PARENT%"
goto find_environment
:explicit_python
set "PIPE_TWIN_BOOTSTRAP=%PIPE_TWIN_PYTHON%"
goto launch
:system_python
where py >nul 2>nul
if not errorlevel 1 (
    py -3.12 "%~dp0run_gui.py" %*
    goto finished
)
where python >nul 2>nul
if not errorlevel 1 (
    python "%~dp0run_gui.py" %*
    goto finished
)
echo Python 3.12 was not found. Install Python with Tcl/Tk support.
echo Project: %~dp0
echo Then run: py -3.12 -m venv .venv
echo And: .venv\Scripts\python.exe -m pip install -r requirements.txt
if /I not "%~1"=="--check" pause
exit /b 1
:launch
"%PIPE_TWIN_BOOTSTRAP%" "%~dp0run_gui.py" %*
:finished
set "PIPE_TWIN_LAUNCH_EXIT=%ERRORLEVEL%"
if not "%PIPE_TWIN_LAUNCH_EXIT%"=="0" (
    echo Startup failed. Project: %~dp0
    echo Python: %PIPE_TWIN_BOOTSTRAP%
    if /I not "%~1"=="--check" pause
)
exit /b %PIPE_TWIN_LAUNCH_EXIT%
