#!/bin/bash
# Strict Python 3.10 quick setup for CARLA environment on Linux/WSL
set -e

echo "CARLA Environment Quick Setup (Strict Python 3.10)"

# Check for python3.10
if command -v python3.10 &> /dev/null; then
    echo "✓ python3.10 found"
    PYTHON_CMD="python3.10"
else
    echo "ERROR: python3.10 was not found on your system."
    echo "Please install Python 3.10.11 or 3.10.14 and ensure it is in your PATH."
    exit 1
fi

# Remove old venv if exists
if [ -d "venv" ]; then
    echo "Virtual environment already exists."
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing virtual environment..."
        rm -rf venv
    else
        echo "Setup cancelled."
        exit 0
    fi
fi

# Create venv with python3.10
$PYTHON_CMD -m venv venv

# Activate and install requirements
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "============================="
echo "Setup completed successfully!"
echo "To activate: source venv/bin/activate"
echo "To deactivate: deactivate"
echo "To test: python test_environment.py"
echo "To run: python your_script.py" 