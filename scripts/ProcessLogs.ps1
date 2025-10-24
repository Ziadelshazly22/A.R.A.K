#!/usr/bin/env pwsh
<#
.SYNOPSIS
    A.R.A.K Log Data Processing Tool (PowerShell Wrapper)

.DESCRIPTION
    This script provides an easy way to process A.R.A.K log files on Windows.
    It automatically processes CSV log files to make them more human-readable.

.PARAMETER InputFile
    Specific CSV file to process

.PARAMETER OutputDir
    Output directory for processed files

.PARAMETER All
    Process all CSV files in logs directory

.PARAMETER LogsDir
    Directory containing log files (default: logs)

.PARAMETER Format
    Output format: excel or csv (default: excel)

.PARAMETER Verbose
    Enable verbose output

.EXAMPLE
    .\ProcessLogs.ps1
    Process all CSV files in logs directory

.EXAMPLE
    .\ProcessLogs.ps1 -InputFile "events_session1.csv"
    Process specific file

.EXAMPLE
    .\ProcessLogs.ps1 -OutputDir "processed_logs" -Verbose
    Process all files with custom output directory and verbose output
#>

param(
    [string]$InputFile,
    [string]$OutputDir,
    [switch]$All,
    [string]$LogsDir = "logs",
    [ValidateSet("excel", "csv")]
    [string]$Format = "excel",
    [switch]$Verbose
)

# Script configuration
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# Colors for output
$Colors = @{
    Success = "Green"
    Warning = "Yellow" 
    Error = "Red"
    Info = "Cyan"
    Header = "Magenta"
}

function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Colors[$Color]
}

function Test-PythonEnvironment {
    """Test if Python environment is properly configured."""
    try {
        $pythonVersion = python --version 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "Python not found"
        }
        
        Write-ColorOutput "✅ Python found: $pythonVersion" "Success"
        return $true
    }
    catch {
        Write-ColorOutput "❌ Python not found or not in PATH" "Error"
        Write-ColorOutput "💡 Please install Python or ensure it's in your PATH" "Info"
        return $false
    }
}

function Test-ProjectStructure {
    """Verify we're in the correct A.R.A.K project directory."""
    $requiredFiles = @("src\data_processor.py", "process_logs.py")
    
    foreach ($file in $requiredFiles) {
        if (-not (Test-Path $file)) {
            Write-ColorOutput "❌ Required file not found: $file" "Error"
            Write-ColorOutput "💡 Make sure you're running this script from the A.R.A.K project root directory" "Info"
            return $false
        }
    }
    
    Write-ColorOutput "✅ Project structure verified" "Success"
    return $true
}

function Invoke-LogProcessing {
    """Call the Python log processing script with appropriate parameters."""
    
    # Build Python command arguments
    $pythonArgs = @("process_logs.py")
    
    if ($InputFile) {
        $pythonArgs += @("--input", $InputFile)
    }
    
    if ($OutputDir) {
        $pythonArgs += @("--output", $OutputDir)
    }
    
    if ($All) {
        $pythonArgs += "--all"
    }
    
    if ($LogsDir -ne "logs") {
        $pythonArgs += @("--logs-dir", $LogsDir)
    }
    
    if ($Format -ne "excel") {
        $pythonArgs += @("--format", $Format)
    }
    
    if ($Verbose) {
        $pythonArgs += "--verbose"
    }
    
    # Execute Python script
    try {
        Write-ColorOutput "🔄 Executing log processing..." "Info"
        $result = & python @pythonArgs
        
        if ($LASTEXITCODE -eq 0) {
            Write-ColorOutput "✅ Log processing completed successfully!" "Success"
            return $true
        }
        else {
            Write-ColorOutput "❌ Log processing failed with exit code: $LASTEXITCODE" "Error"
            return $false
        }
    }
    catch {
        Write-ColorOutput "❌ Error executing Python script: $($_.Exception.Message)" "Error"
        return $false
    }
}

function Show-Usage {
    """Display usage information and examples."""
    Write-ColorOutput "`n📖 A.R.A.K Log Processing Tool - Usage Examples:" "Header"
    Write-ColorOutput "=" * 60 "Header"
    
    Write-ColorOutput "`n🔹 Process all log files:" "Info"
    Write-ColorOutput "  .\ProcessLogs.ps1" "White"
    
    Write-ColorOutput "`n🔹 Process specific file:" "Info"  
    Write-ColorOutput "  .\ProcessLogs.ps1 -InputFile `"events_session1.csv`"" "White"
    
    Write-ColorOutput "`n🔹 Custom output directory:" "Info"
    Write-ColorOutput "  .\ProcessLogs.ps1 -OutputDir `"my_processed_logs`"" "White"
    
    Write-ColorOutput "`n🔹 Verbose output:" "Info"
    Write-ColorOutput "  .\ProcessLogs.ps1 -Verbose" "White"
    
    Write-ColorOutput "`n🔹 CSV output format:" "Info"
    Write-ColorOutput "  .\ProcessLogs.ps1 -Format csv" "White"
    
    Write-ColorOutput "`n💡 For more options, run: python process_logs.py --help" "Info"
}

function Main {
    """Main execution function."""
    
    # Display header
    Write-ColorOutput "`n🔄 A.R.A.K Log Data Processing Tool" "Header"
    Write-ColorOutput "=" * 50 "Header"
    
    # Verify environment
    if (-not (Test-PythonEnvironment)) {
        exit 1
    }
    
    if (-not (Test-ProjectStructure)) {
        exit 1
    }
    
    # Show usage if no parameters provided
    if (-not $InputFile -and -not $All -and -not $PSBoundParameters.Count) {
        Write-ColorOutput "`n📋 No specific parameters provided. Processing all log files..." "Info"
        $All = $true
    }
    
    # Check if logs directory exists
    if (-not (Test-Path $LogsDir)) {
        Write-ColorOutput "❌ Logs directory '$LogsDir' not found." "Error"
        Write-ColorOutput "💡 Run some monitoring sessions first to generate log files." "Info"
        Show-Usage
        exit 1
    }
    
    # Execute log processing
    $success = Invoke-LogProcessing
    
    if ($success) {
        Write-ColorOutput "`n🎉 Processing completed successfully!" "Success"
        
        # Show output location
        $outputPath = if ($OutputDir) { $OutputDir } else { Join-Path $LogsDir "processed" }
        if (Test-Path $outputPath) {
            Write-ColorOutput "📁 Processed files saved to: $outputPath" "Info"
            
            # List processed files
            $processedFiles = Get-ChildItem $outputPath -Filter "*.xlsx" | Select-Object -First 5
            if ($processedFiles) {
                Write-ColorOutput "`n📋 Recent processed files:" "Info"
                foreach ($file in $processedFiles) {
                    Write-ColorOutput "  ✓ $($file.Name)" "Success"
                }
            }
        }
        
        Write-ColorOutput "`n💡 You can now open the processed Excel files for human-readable data analysis!" "Info"
        exit 0
    }
    else {
        Write-ColorOutput "`n💭 If you need help, try:" "Info"
        Show-Usage
        exit 1
    }
}

# Script execution
try {
    Main
}
catch {
    Write-ColorOutput "`n❌ Unexpected error: $($_.Exception.Message)" "Error"
    Write-ColorOutput "💡 Please check the error details and try again." "Info"
    exit 1
}
finally {
    # Restore preferences
    $ProgressPreference = "Continue"
}