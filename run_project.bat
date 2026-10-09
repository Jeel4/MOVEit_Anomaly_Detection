@echo off
cd /d "%~dp0"
python main.py --quick
echo.
echo If the run failed, activate your Python environment and install requirements.txt.
pause
