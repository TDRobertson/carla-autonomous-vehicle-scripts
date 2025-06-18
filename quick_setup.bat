@echo off
REM Quick setup script for CARLA environment on Windows

echo CARLA Environment Quick Setup
echo =============================

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo ✗ Python not found in PATH
    echo Please install Python 3.10 and add it to PATH
    pause
    exit /b 1
)

REM Check if virtual environment already exists
if exist "venv" (
    echo Virtual environment already exists.
    set /p recreate="Do you want to recreate it? (y/N): "
    if /i "%recreate%"=="y" (
        echo Removing existing virtual environment...
        rmdir /s /q venv
    ) else (
        echo Setup cancelled.
        pause
        exit /b 0
    )
)

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ✗ Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ✗ Failed to activate virtual environment
    pause
    exit /b 1
)

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 (
    echo ✗ Failed to upgrade pip
    pause
    exit /b 1
)

REM Install requirements
echo Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ✗ Failed to install requirements
    pause
    exit /b 1
)

REM Install Windows-specific packages
echo Installing Windows-specific packages...
pip install pywin32==306
if errorlevel 1 (
    echo ⚠ Failed to install pywin32 (this is optional)
)

echo.
echo =============================
echo Setup completed successfully!
echo.
echo To activate the environment:
echo   venv\Scripts\activate.bat
echo.
echo To deactivate:
echo   deactivate
echo.
echo To test the environment:
echo   python test_environment.py
echo.
echo To run your CARLA scripts:
echo   1. Activate the environment
echo   2. Navigate to your script directory
echo   3. Run: python your_script.py
echo.
pause 