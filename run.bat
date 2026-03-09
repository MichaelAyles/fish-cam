@echo off
setlocal

cd /d "%~dp0"

set "VENV_DIR=%~dp0.venv"

:: Create venv if it doesn't exist
if not exist "%VENV_DIR%" (
    echo Creating virtual environment...
    python -m venv "%VENV_DIR%"
)

call "%VENV_DIR%\Scripts\activate.bat"

:: Install/upgrade requirements (pip is fast when already satisfied)
pip install --quiet -r requirements.txt

python fishon.py %*
