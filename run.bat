@echo off
cd /d "%~dp0"

if not exist venv\Scripts\python.exe (
    echo venv not found. Run setup.bat first.
    pause & exit /b 1
)

venv\Scripts\python.exe app.py
pause
