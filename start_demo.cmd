@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" scripts\wind_lab.py --demo
) else (
    python scripts\wind_lab.py --demo
)
if errorlevel 1 (
    echo See the installation steps in README.md.
    pause
    exit /b 1
)
