@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project Python environment is missing. Install requirements first.
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m pipe_twin gui %*
if errorlevel 1 pause
endlocal
