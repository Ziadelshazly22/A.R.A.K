@echo off
REM A.R.A.K Quick Setup launcher (Windows)
REM This batch file invokes the PowerShell script to create a venv and install deps.

setlocal
set SCRIPT_DIR=%~dp0

REM Enable calling PowerShell with a permissive policy for this process only
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%QuickSetup.ps1"

if %ERRORLEVEL% NEQ 0 (
  echo.
  echo Quick setup failed. See messages above.
  exit /b %ERRORLEVEL%
)

echo.
echo Environment ready.
echo To activate:   .\venv\Scripts\Activate.ps1

echo To run the app: streamlit run src\ui\streamlit_app.py

endlocal
