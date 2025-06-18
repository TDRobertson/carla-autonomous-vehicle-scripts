#!/bin/bash
# Quick setup script for CARLA environment on WSL/Linux

set -e  # Exit on any error

echo "CARLA Environment Quick Setup"
echo "============================="

# Check if we're on a supported system
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "✓ Linux system detected"
elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
    echo "✓ WSL system detected"
else
    echo "⚠ Unsupported system: $OSTYPE"
    echo "This script is designed for Linux/WSL systems"
    exit 1
fi

# Check Python version
echo "Checking Python version..."
if command -v python3.10 &> /dev/null; then
    echo "✓ Python 3.10 found"
    PYTHON_CMD="python3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
    if [[ "$PYTHON_VERSION" == "3.10" ]]; then
        echo "✓ Python 3.10 found"
        PYTHON_CMD="python3"
    else
        echo "✗ Python 3.10 not found. Found version: $PYTHON_VERSION"
        echo "Please install Python 3.10 first"
        exit 1
    fi
else
    echo "✗ Python 3 not found"
    echo "Please install Python 3.10 first"
    exit 1
fi

# Check if virtual environment already exists
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

# Create virtual environment
echo "Creating virtual environment..."
$PYTHON_CMD -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "============================="
echo "Setup completed successfully!"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate:"
echo "  deactivate"
echo ""
echo "To test the environment:"
echo "  python test_environment.py"
echo ""
echo "To run your CARLA scripts:"
echo "  1. Activate the environment"
echo "  2. Navigate to your script directory"
echo "  3. Run: python your_script.py" 