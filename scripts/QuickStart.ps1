# Requires PowerShell 5.1+
$ErrorActionPreference = "Stop"

Write-Host "A.R.A.K Quick Start (Windows)" -ForegroundColor Cyan

# Resolve repo root (one level up from scripts/)
$root = Split-Path -Parent $PSScriptRoot
$venvPy = Join-Path $root "venv\Scripts\python.exe"
$appPath = Join-Path $root "src\ui\streamlit_app.py"
$quickSetup = Join-Path $PSScriptRoot "QuickSetup.ps1"

# Ensure Streamlit app path exists
if (-not (Test-Path $appPath)) {
    Write-Host "Streamlit app not found at: $appPath" -ForegroundColor Red
    exit 1
}

# Create venv if missing by invoking QuickSetup
if (-not (Test-Path $venvPy)) {
    Write-Host "Virtual environment not found. Running QuickSetup..." -ForegroundColor Yellow
    if (-not (Test-Path $quickSetup)) {
        Write-Host "QuickSetup.ps1 not found in scripts/." -ForegroundColor Red
        exit 1
    }
    powershell -NoProfile -ExecutionPolicy Bypass -File "$quickSetup"
}

# Re-check after setup
if (-not (Test-Path $venvPy)) {
    Write-Host "Failed to prepare virtual environment." -ForegroundColor Red
    exit 1
}

Write-Host "Starting Streamlit app..." -ForegroundColor Green
& "$venvPy" -m streamlit run "$appPath"
