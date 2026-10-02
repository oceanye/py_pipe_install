@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Project Python environment is missing. Install requirements first.
    pause
    exit /b 1
)
rem The integrated office client starts the GUI, the authenticated 8770
rem capture service, and the read-only 8765 evidence service in one process.
start "Pipe Twin Office Client" ".venv\Scripts\pythonw.exe" -m pipe_twin office-client %*
if errorlevel 1 pause
endlocal
