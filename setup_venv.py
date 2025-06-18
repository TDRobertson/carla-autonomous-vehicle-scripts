#!/usr/bin/env python3
"""
Setup script for CARLA autonomous vehicle environment.
This script helps create a virtual environment with the correct Python version and dependencies.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major != 3 or version.minor != 10:
        print(f"Warning: Current Python version is {version.major}.{version.minor}")
        print("This environment was designed for Python 3.10.14")
        print("You may encounter compatibility issues.")
        return False
    return True

def create_venv(venv_path):
    """Create a virtual environment."""
    try:
        subprocess.run([sys.executable, "-m", "venv", venv_path], check=True)
        print(f"Virtual environment created at: {venv_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        return False

def get_activation_script(venv_path):
    """Get the appropriate activation script based on OS."""
    system = platform.system().lower()
    
    if system == "windows":
        return venv_path / "Scripts" / "activate.bat"
    else:
        return venv_path / "bin" / "activate"

def install_requirements(venv_path):
    """Install requirements in the virtual environment."""
    system = platform.system().lower()
    
    if system == "windows":
        pip_path = venv_path / "Scripts" / "pip.exe"
    else:
        pip_path = venv_path / "bin" / "pip"
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    try:
        # Upgrade pip first
        subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
        
        # Install requirements
        subprocess.run([str(pip_path), "install", "-r", str(requirements_file)], check=True)
        print("Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False

def main():
    """Main setup function."""
    print("CARLA Autonomous Vehicle Environment Setup")
    print("=" * 50)
    
    # Check Python version
    check_python_version()
    
    # Get current directory
    current_dir = Path(__file__).parent
    venv_path = current_dir / "venv"
    
    # Check if virtual environment already exists
    if venv_path.exists():
        print(f"Virtual environment already exists at: {venv_path}")
        response = input("Do you want to recreate it? (y/N): ").lower()
        if response == 'y':
            import shutil
            shutil.rmtree(venv_path)
        else:
            print("Setup cancelled.")
            return
    
    # Create virtual environment
    print("Creating virtual environment...")
    if not create_venv(venv_path):
        return
    
    # Install requirements
    print("Installing requirements...")
    if not install_requirements(venv_path):
        return
    
    # Get activation script
    activation_script = get_activation_script(venv_path)
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("\nTo activate the virtual environment:")
    
    system = platform.system().lower()
    if system == "windows":
        print(f"  {activation_script}")
        print("  or")
        print(f"  {venv_path}\\Scripts\\activate.ps1  # For PowerShell")
    else:
        print(f"  source {activation_script}")
    
    print("\nTo deactivate:")
    print("  deactivate")
    
    print("\nTo run your CARLA scripts:")
    print("  1. Activate the virtual environment")
    print("  2. Navigate to your script directory")
    print("  3. Run: python your_script.py")

if __name__ == "__main__":
    main() 