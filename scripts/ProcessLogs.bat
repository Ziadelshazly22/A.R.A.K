@echo off
REM A.R.A.K Log Data Processing Tool (Batch Wrapper)
REM This script provides an easy way to process A.R.A.K log files.

setlocal enabledelayedexpansion

REM Script configuration
set "SCRIPT_NAME=A.R.A.K Log Processing Tool"
set "PYTHON_SCRIPT=process_logs.py"

REM Colors (if supported)
set "COLOR_SUCCESS=92"
set "COLOR_WARNING=93" 
set "COLOR_ERROR=91"
set "COLOR_INFO=96"
set "COLOR_HEADER=95"

echo.
echo [%COLOR_HEADER%m========================================[0m
echo [%COLOR_HEADER%m%SCRIPT_NAME%[0m
echo [%COLOR_HEADER%m========================================[0m
echo.

REM Check if Python is available
python --version >nul 2>&1
if !errorlevel! neq 0 (
    echo [%COLOR_ERROR%m❌ Python not found or not in PATH[0m
    echo [%COLOR_INFO%m💡 Please install Python or ensure it's in your PATH[0m
    echo.
    pause
    exit /b 1
)

REM Verify project structure
if not exist "%PYTHON_SCRIPT%" (
    echo [%COLOR_ERROR%m❌ Required file not found: %PYTHON_SCRIPT%[0m
    echo [%COLOR_INFO%m💡 Make sure you're running this from the A.R.A.K project root directory[0m
    echo.
    pause
    exit /b 1
)

if not exist "src\data_processor.py" (
    echo [%COLOR_ERROR%m❌ Required file not found: src\data_processor.py[0m
    echo [%COLOR_INFO%m💡 Make sure you're running this from the A.R.A.K project root directory[0m
    echo.
    pause
    exit /b 1
)

echo [%COLOR_SUCCESS%m✅ Python found and project structure verified[0m
echo.

REM Check if logs directory exists
if not exist "logs" (
    echo [%COLOR_ERROR%m❌ Logs directory not found[0m
    echo [%COLOR_INFO%m💡 Run some monitoring sessions first to generate log files[0m
    echo.
    pause
    exit /b 1
)

REM Check for CSV files
set "CSV_COUNT=0"
for %%f in (logs\*.csv) do (
    set /a CSV_COUNT+=1
)

if !CSV_COUNT! equ 0 (
    echo [%COLOR_WARNING%m📄 No CSV files found in logs directory[0m
    echo [%COLOR_INFO%m💡 Run some monitoring sessions to generate log files first[0m
    echo.
    pause
    exit /b 0
)

echo [%COLOR_INFO%m📊 Found !CSV_COUNT! CSV file(s) to process[0m
echo.

REM Show menu options
echo [%COLOR_INFO%mSelect processing option:[0m
echo 1. Process all CSV files (Recommended)
echo 2. Process specific file
echo 3. Show help and exit
echo.
set /p "choice=Enter your choice (1-3): "

if "!choice!"=="1" goto process_all
if "!choice!"=="2" goto process_specific
if "!choice!"=="3" goto show_help
goto invalid_choice

:process_all
echo.
echo [%COLOR_INFO%m🔄 Processing all CSV files...[0m
python "%PYTHON_SCRIPT%" --all --verbose
goto check_result

:process_specific
echo.
echo [%COLOR_INFO%mAvailable CSV files:[0m
set "file_num=0"
for %%f in (logs\*.csv) do (
    set /a file_num+=1
    echo !file_num!. %%~nxf
    set "file_!file_num!=%%~nxf"
)
echo.
set /p "file_choice=Enter file number: "

if defined file_!file_choice! (
    set "selected_file=!file_%file_choice%!"
    echo.
    echo [%COLOR_INFO%m🔄 Processing: !selected_file![0m
    python "%PYTHON_SCRIPT%" --input "logs\!selected_file!" --verbose
    goto check_result
) else (
    echo [%COLOR_ERROR%m❌ Invalid file number[0m
    goto end_script
)

:show_help
echo.
echo [%COLOR_INFO%m📖 A.R.A.K Log Processing Help:[0m
echo.
echo This tool processes A.R.A.K CSV log files to make them more human-readable.
echo.
echo [%COLOR_INFO%mWhat it does:[0m
echo • Converts timestamps to readable dates and times
echo • Translates event codes into human-readable descriptions  
echo • Categorizes suspicion levels (Normal, Low Risk, High Risk, etc.)
echo • Analyzes head pose and gaze direction data
echo • Creates comprehensive Excel reports with multiple sheets
echo • Provides summary statistics and insights
echo.
echo [%COLOR_INFO%mOutput location:[0m
echo Processed files are saved in: logs\processed\
echo.
echo [%COLOR_INFO%mFor advanced options, run:[0m
echo python process_logs.py --help
echo.
goto end_script

:check_result
if !errorlevel! equ 0 (
    echo.
    echo [%COLOR_SUCCESS%m✅ Processing completed successfully![0m
    echo [%COLOR_INFO%m📁 Processed files saved to: logs\processed\[0m
    echo.
    echo [%COLOR_INFO%m💡 You can now open the Excel files for human-readable data analysis![0m
    
    REM Ask if user wants to open the output folder
    set /p "open_folder=Open output folder? (y/n): "
    if /i "!open_folder!"=="y" (
        if exist "logs\processed" (
            explorer "logs\processed"
        )
    )
) else (
    echo.
    echo [%COLOR_ERROR%m❌ Processing failed[0m
    echo [%COLOR_INFO%m💡 Check the error messages above for details[0m
)
goto end_script

:invalid_choice
echo [%COLOR_ERROR%m❌ Invalid choice. Please enter 1, 2, or 3.[0m
goto end_script

:end_script
echo.
echo [%COLOR_INFO%mPress any key to exit...[0m
pause >nul
exit /b 0