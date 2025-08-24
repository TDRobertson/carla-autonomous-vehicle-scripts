@echo off
REM Strict Python 3.10 quick setup for CARLA environment on Windows

echo CARLA Environment Quick Setup

REM Check for Python 3.10
py -3.10 --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.10 was not found on your system.
    echo Please install Python 3.10.11 or 3.10.14 and ensure it is in your PATH.
    pause
    exit /b 1
)
echo ✓ Python 3.10 found

REM Remove old venv if exists
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

REM Create venv with Python 3.10
py -3.10 -m venv venv
if errorlevel 1 (
    echo ✗ Failed to create virtual environment with Python 3.10
    pause
    exit /b 1
)

REM Activate and install requirements
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pywin32==306

echo.
echo =============================
echo Setup completed successfully!
echo To activate: venv\Scripts\activate.bat
echo To deactivate: deactivate
echo To test: python test_environment.py
echo To run: python your_script.py
pause 