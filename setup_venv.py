#!/usr/bin/env python3
"""
Strict Python 3.10 setup script for CARLA autonomous vehicle environment.
"""
import os
import sys
import subprocess
import platform
from pathlib import Path

def find_python310():
    system = platform.system().lower()
    if system == "windows":
        # Try 'py -3.10'
        try:
            out = subprocess.check_output(["py", "-3.10", "--version"], stderr=subprocess.STDOUT)
            return ["py", "-3.10"]
        except Exception:
            return None
    else:
        # Try 'python3.10'
        for exe in ["python3.10", "python3"]:
            try:
                out = subprocess.check_output([exe, "--version"], stderr=subprocess.STDOUT)
                if b"3.10" in out:
                    return [exe]
            except Exception:
                continue
        return None

def main():
    print("CARLA Strict Python 3.10 Environment Setup")
    print("=" * 50)
    py310 = find_python310()
    if not py310:
        print("ERROR: Python 3.10 was not found on your system.")
        print("Please install Python 3.10.11 or 3.10.14 and ensure it is in your PATH.")
        sys.exit(1)
    else:
        print(f"✓ Found Python 3.10: {' '.join(py310)}")

    current_dir = Path(__file__).parent
    venv_path = current_dir / "venv"
    if venv_path.exists():
        print(f"Virtual environment already exists at: {venv_path}")
        response = input("Do you want to recreate it? (y/N): ").lower()
        if response == 'y':
            import shutil
            shutil.rmtree(venv_path)
        else:
            print("Setup cancelled.")
            return
    print("Creating virtual environment with Python 3.10...")
    try:
        subprocess.run(py310 + ["-m", "venv", str(venv_path)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        sys.exit(1)
    print(f"Virtual environment created at: {venv_path}")
    # Install requirements
    if platform.system().lower() == "windows":
        pip_path = venv_path / "Scripts" / "pip.exe"
    else:
        pip_path = venv_path / "bin" / "pip"
    requirements_file = current_dir / "requirements.txt"
    print("Upgrading pip and installing requirements...")
    try:
        subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
        subprocess.run([str(pip_path), "install", "-r", str(requirements_file)], check=True)
        print("Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)
    print("\nSetup completed successfully!")
    print("To activate the virtual environment:")
    if platform.system().lower() == "windows":
        print(f"  {venv_path}\\Scripts\\activate.bat")
        print(f"  or {venv_path}\\Scripts\\activate.ps1  # For PowerShell")
    else:
        print(f"  source {venv_path}/bin/activate")
    print("To deactivate: deactivate")
    print("To run your CARLA scripts: python your_script.py")

if __name__ == "__main__":
    main() 