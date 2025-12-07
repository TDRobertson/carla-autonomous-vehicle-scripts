@echo off
REM ============================================================================
REM ML Dataset Collection Automation Script (Windows)
REM ============================================================================
REM This script automates the collection of GPS/IMU data for training ML models
REM to detect GPS spoofing attacks. It provides menu-driven options for
REM different collection scenarios.
REM ============================================================================

setlocal enabledelayedexpansion

REM Color codes for better readability
set "ESC="
set "RED=%ESC%[91m"
set "GREEN=%ESC%[92m"
set "YELLOW=%ESC%[93m"
set "BLUE=%ESC%[94m"
set "MAGENTA=%ESC%[95m"
set "CYAN=%ESC%[96m"
set "RESET=%ESC%[0m"

:MENU
cls
echo ============================================================================
echo ML DATASET COLLECTION AUTOMATION
echo ============================================================================
echo.
echo Please select collection mode:
echo.
echo  1. Quick Test (5 runs x 60s) - Testing and validation
echo  2. One-Class Training (25 runs x 120s) - Standard training dataset
echo  3. One-Class Validation (5 runs x 180s, random) - Validation set
echo  4. Supervised Training (20 runs x 120s) - Balanced supervised learning
echo  5. Supervised Validation (10 runs x 150s, random) - Supervised validation
echo  6. Custom Parameters - Specify your own settings
echo  7. Exit
echo.
echo ============================================================================
echo.

set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto QUICK_TEST
if "%choice%"=="2" goto ONE_CLASS_TRAINING
if "%choice%"=="3" goto ONE_CLASS_VALIDATION
if "%choice%"=="4" goto SUPERVISED_TRAINING
if "%choice%"=="5" goto SUPERVISED_VALIDATION
if "%choice%"=="6" goto CUSTOM
if "%choice%"=="7" goto END

echo Invalid choice. Please try again.
timeout /t 2 >nul
goto MENU

REM ============================================================================
REM Mode 1: Quick Test
REM ============================================================================
:QUICK_TEST
cls
echo ============================================================================
echo QUICK TEST - 5 runs x 60 seconds
echo ============================================================================
echo.
echo This will collect 5 test runs with 60-second duration each.
echo Attack starts after 10 seconds (clean baseline).
echo Output directory: data/test
echo.
set /p confirm="Continue? (Y/N): "
if /i not "%confirm%"=="Y" goto MENU

mkdir data\test 2>nul

echo.
echo Starting data collection...
echo.

for /L %%i in (1,1,5) do (
    echo [%%i/5] Collecting run %%i...
    python ml_data_collector.py --duration 60 --attack-delay 10 --warmup 5 --output-dir data/test --label test_run%%i
    if errorlevel 1 (
        echo ERROR: Collection failed on run %%i
        pause
        goto MENU
    )
    echo Run %%i completed successfully.
    echo.
    timeout /t 2 >nul
)

echo.
echo ============================================================================
echo COLLECTION COMPLETE
echo ============================================================================
echo Total runs: 5
echo Estimated samples: ~3,000
echo Location: data/test/
echo.
pause
goto MENU

REM ============================================================================
REM Mode 2: One-Class Training
REM ============================================================================
:ONE_CLASS_TRAINING
cls
echo ============================================================================
echo ONE-CLASS TRAINING - 25 runs x 120 seconds
echo ============================================================================
echo.
echo This will collect 25 training runs with 120-second duration each.
echo Attack starts after 30 seconds (clean baseline for training).
echo Output directory: data/training
echo.
echo Estimated time: ~1 hour
echo.
set /p confirm="Continue? (Y/N): "
if /i not "%confirm%"=="Y" goto MENU

mkdir data\training 2>nul

echo.
echo Starting data collection...
echo This may take approximately 1 hour. Please be patient.
echo.

for /L %%i in (1,1,25) do (
    echo [%%i/25] Collecting run %%i...
    echo Time: %TIME%
    python ml_data_collector.py --duration 120 --attack-delay 30 --warmup 5 --output-dir data/training --label train_run%%i
    if errorlevel 1 (
        echo ERROR: Collection failed on run %%i
        pause
        goto MENU
    )
    echo Run %%i completed successfully.
    echo.
)

echo.
echo ============================================================================
echo COLLECTION COMPLETE
echo ============================================================================
echo Total runs: 25
echo Estimated samples: ~30,000
echo Location: data/training/
echo.
pause
goto MENU

REM ============================================================================
REM Mode 3: One-Class Validation
REM ============================================================================
:ONE_CLASS_VALIDATION
cls
echo ============================================================================
echo ONE-CLASS VALIDATION - 5 runs x 180 seconds (Random Attacks)
echo ============================================================================
echo.
echo This will collect 5 validation runs with 180-second duration each.
echo Random attack mode with unpredictable timing.
echo Output directory: data/validation
echo.
set /p confirm="Continue? (Y/N): "
if /i not "%confirm%"=="Y" goto MENU

mkdir data\validation 2>nul

echo.
echo Starting data collection...
echo.

for /L %%i in (1,1,5) do (
    echo [%%i/5] Collecting run %%i...
    python ml_data_collector.py --random-attacks --duration 180 --warmup 5 --min-attack-duration 5 --max-attack-duration 20 --min-clean-duration 5 --max-clean-duration 15 --output-dir data/validation --label val_run%%i
    if errorlevel 1 (
        echo ERROR: Collection failed on run %%i
        pause
        goto MENU
    )
    echo Run %%i completed successfully.
    echo.
    timeout /t 2 >nul
)

echo.
echo ============================================================================
echo COLLECTION COMPLETE
echo ============================================================================
echo Total runs: 5
echo Estimated samples: ~9,000
echo Location: data/validation/
echo.
pause
goto MENU

REM ============================================================================
REM Mode 4: Supervised Training
REM ============================================================================
:SUPERVISED_TRAINING
cls
echo ============================================================================
echo SUPERVISED TRAINING - 20 runs x 120 seconds
echo ============================================================================
echo.
echo This will collect 20 training runs with 120-second duration each.
echo Attack starts immediately (attack_delay=0) for maximum labeled pairs.
echo Output directory: data/supervised_training
echo.
echo Estimated time: ~45 minutes
echo.
set /p confirm="Continue? (Y/N): "
if /i not "%confirm%"=="Y" goto MENU

mkdir data\supervised_training 2>nul

echo.
echo Starting data collection...
echo This may take approximately 45 minutes.
echo.

for /L %%i in (1,1,20) do (
    echo [%%i/20] Collecting run %%i...
    echo Time: %TIME%
    python ml_data_collector.py --attack-delay 0 --duration 120 --warmup 5 --output-dir data/supervised_training --label sup_train_run%%i
    if errorlevel 1 (
        echo ERROR: Collection failed on run %%i
        pause
        goto MENU
    )
    echo Run %%i completed successfully.
    echo.
)

echo.
echo ============================================================================
echo COLLECTION COMPLETE
echo ============================================================================
echo Total runs: 20
echo Estimated samples: ~24,000
echo Location: data/supervised_training/
echo.
pause
goto MENU

REM ============================================================================
REM Mode 5: Supervised Validation
REM ============================================================================
:SUPERVISED_VALIDATION
cls
echo ============================================================================
echo SUPERVISED VALIDATION - 10 runs x 150 seconds (Random Attacks)
echo ============================================================================
echo.
echo This will collect 10 validation runs with 150-second duration each.
echo Random attack mode with immediate start for timing diversity.
echo Output directory: data/supervised_validation
echo.
set /p confirm="Continue? (Y/N): "
if /i not "%confirm%"=="Y" goto MENU

mkdir data\supervised_validation 2>nul

echo.
echo Starting data collection...
echo.

for /L %%i in (1,1,10) do (
    echo [%%i/10] Collecting run %%i...
    python ml_data_collector.py --random-attacks --attack-delay 0 --duration 150 --warmup 5 --min-attack-duration 8 --max-attack-duration 25 --min-clean-duration 8 --max-clean-duration 25 --output-dir data/supervised_validation --label sup_val_run%%i
    if errorlevel 1 (
        echo ERROR: Collection failed on run %%i
        pause
        goto MENU
    )
    echo Run %%i completed successfully.
    echo.
    timeout /t 2 >nul
)

echo.
echo ============================================================================
echo COLLECTION COMPLETE
echo ============================================================================
echo Total runs: 10
echo Estimated samples: ~15,000
echo Location: data/supervised_validation/
echo.
pause
goto MENU

REM ============================================================================
REM Mode 6: Custom Parameters
REM ============================================================================
:CUSTOM
cls
echo ============================================================================
echo CUSTOM PARAMETERS
echo ============================================================================
echo.
echo Enter your custom collection parameters:
echo.

set /p num_runs="Number of runs: "
set /p duration="Duration per run (seconds): "
set /p attack_delay="Attack delay (seconds, 0 for immediate): "
set /p output_dir="Output directory (e.g., data/custom): "
set /p label_prefix="Label prefix (e.g., custom_run): "

set /p use_random="Use random attacks? (Y/N): "

mkdir %output_dir% 2>nul

echo.
echo ============================================================================
echo SUMMARY
echo ============================================================================
echo Runs: %num_runs%
echo Duration: %duration%s
echo Attack delay: %attack_delay%s
echo Random attacks: %use_random%
echo Output: %output_dir%
echo Label: %label_prefix%
echo ============================================================================
echo.

set /p confirm="Start collection? (Y/N): "
if /i not "%confirm%"=="Y" goto MENU

echo.
echo Starting data collection...
echo.

for /L %%i in (1,1,%num_runs%) do (
    echo [%%i/%num_runs%] Collecting run %%i...
    
    if /i "%use_random%"=="Y" (
        python ml_data_collector.py --random-attacks --duration %duration% --attack-delay %attack_delay% --warmup 5 --output-dir %output_dir% --label %label_prefix%%%i
    ) else (
        python ml_data_collector.py --duration %duration% --attack-delay %attack_delay% --warmup 5 --output-dir %output_dir% --label %label_prefix%%%i
    )
    
    if errorlevel 1 (
        echo ERROR: Collection failed on run %%i
        pause
        goto MENU
    )
    echo Run %%i completed successfully.
    echo.
)

echo.
echo ============================================================================
echo COLLECTION COMPLETE
echo ============================================================================
echo Total runs: %num_runs%
echo Location: %output_dir%/
echo.
pause
goto MENU

REM ============================================================================
REM End
REM ============================================================================
:END
echo.
echo Exiting...
echo.
endlocal
exit /b 0

