@echo off
REM A.R.A.K Quick Start launcher (Windows)
REM Double-click to activate (or create) venv and run Streamlit app.

setlocal
set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

REM Change to project root directory
cd /d "%PROJECT_ROOT%"

REM Try to run QuickStart.ps1 with ExecutionPolicy Bypass for this process only
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%QuickStart.ps1"

if %ERRORLEVEL% NEQ 0 (
  echo.
  echo Quick Start failed. See messages above.
  exit /b %ERRORLEVEL%
)

endlocal
