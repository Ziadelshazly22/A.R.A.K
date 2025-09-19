# Requires PowerShell 5.1+
$ErrorActionPreference = "Stop"

Write-Host "A.R.A.K Quick Setup (Windows)" -ForegroundColor Cyan

# Locate Python (prefer py launcher, fallback to python)
$pythonExe = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $pythonExe = "py"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pythonExe = "python"
} else {
    Write-Host "Python not found. Please install Python 3.8+ and retry." -ForegroundColor Red
    exit 1
}

# Check Python version (>= 3.8)
$verOut = & $pythonExe -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
if (-not $verOut) { $verOut = "0.0" }
$parts = $verOut.Trim().Split('.')
if ($parts.Count -lt 2) { $parts = @('0','0') }
$maj = [int]$parts[0]
$min = [int]$parts[1]
if (($maj -lt 3) -or (($maj -eq 3) -and ($min -lt 8))) {
    Write-Host ("Python >=3.8 required. Found {0}" -f $verOut) -ForegroundColor Red
    exit 1
}

# Create venv in ./venv (same as Unix setup)
$venvDir = Join-Path $PSScriptRoot "..\venv"
Write-Host ("Creating virtual environment at {0}" -f $venvDir) -ForegroundColor Yellow
& $pythonExe -m venv "$venvDir"

# Use venv's Python to install requirements
$venvPy = Join-Path $venvDir "Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "Failed to locate venv's python.exe" -ForegroundColor Red
    exit 1
}

Write-Host "Upgrading pip..." -ForegroundColor Yellow
& "$venvPy" -m pip install --upgrade pip

$reqPath = Join-Path $PSScriptRoot "..\requirements.txt"
if (-not (Test-Path $reqPath)) {
    Write-Host "requirements.txt not found next to scripts folder." -ForegroundColor Red
    exit 1
}

Write-Host "Installing project requirements..." -ForegroundColor Yellow
& "$venvPy" -m pip install -r "$reqPath"

Write-Host "" 
Write-Host "Environment ready." -ForegroundColor Green
Write-Host "Activate with:  .\venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "Run app with:    streamlit run src\ui\streamlit_app.py" -ForegroundColor Green
