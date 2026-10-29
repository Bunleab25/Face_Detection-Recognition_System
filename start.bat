@echo off
echo ==================================================================
echo   Face Detection ^& Recognition System (Windows Launcher)
echo ==================================================================
if exist .venv\Scripts\activate.bat (
    call .venv\Scripts\activate.bat
)
python run.py
pause
