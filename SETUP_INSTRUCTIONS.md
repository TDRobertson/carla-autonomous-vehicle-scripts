# CARLA Environment Setup Instructions

This guide helps you migrate from the conda environment to a standard Python virtual environment that can run outside of conda or within WSL.

## Prerequisites

### Windows

- Python 3.10.14 installed
- Git (optional, for cloning)
- Visual Studio Build Tools (for some packages)

### WSL/Linux

- Python 3.10.14 installed
- System dependencies (see below)

## System Dependencies (WSL/Linux)

Before setting up the Python environment, install these system dependencies:

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgcc-s1 \
    libstdc++6 \
    libgfortran5 \
    libopenblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    libhdf5-serial-dev \
    libhdf5-103 \
    libqtgui4 \
    libqtwebkit4 \
    libqt4-test \
    python3-pyqt5 \
    libgtk-3-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libatlas-base-dev \
    gfortran \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-alsa \
    gstreamer1.0-gl \
    gstreamer1.0-gtk3 \
    gstreamer1.0-qt5 \
    gstreamer1.0-pulseaudio

# CentOS/RHEL/Fedora
sudo yum install -y \
    python3.10 \
    python3.10-devel \
    gcc \
    gcc-c++ \
    make \
    cmake \
    mesa-libGL \
    mesa-libGL-devel \
    libXext \
    libXrender \
    libXrender-devel \
    gcc-gfortran \
    openblas-devel \
    lapack-devel \
    atlas-devel \
    hdf5-devel \
    qt4-devel \
    gtk3-devel \
    libcanberra-gtk3 \
    ffmpeg-devel \
    libjpeg-devel \
    libpng-devel \
    libtiff-devel \
    gstreamer1-devel \
    gstreamer1-plugins-base-devel \
    gstreamer1-plugins-bad-free-devel
```

## Setup Methods

### Method 1: Automated Setup (Recommended)

1. **Run the setup script:**

   ```bash
   python setup_venv.py
   ```

2. **Follow the prompts and wait for installation to complete.**

### Method 2: Manual Setup

#### Windows

1. **Create virtual environment:**

   ```cmd
   python -m venv venv
   ```

2. **Activate virtual environment:**

   ```cmd
   venv\Scripts\activate
   ```

   or for PowerShell:

   ```powershell
   venv\Scripts\Activate.ps1
   ```

3. **Upgrade pip:**

   ```cmd
   python -m pip install --upgrade pip
   ```

4. **Install requirements:**

   ```cmd
   pip install -r requirements.txt
   ```

5. **Install Windows-specific packages:**
   ```cmd
   pip install pywin32==306
   ```

#### WSL/Linux

1. **Create virtual environment:**

   ```bash
   python3.10 -m venv venv
   ```

2. **Activate virtual environment:**

   ```bash
   source venv/bin/activate
   ```

3. **Upgrade pip:**

   ```bash
   pip install --upgrade pip
   ```

4. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

## Environment Activation

### Windows

```cmd
venv\Scripts\activate
```

### WSL/Linux

```bash
source venv/bin/activate
```

## Running Your Code

1. **Activate the virtual environment** (see above)

2. **Navigate to your script directory:**

   ```bash
   cd sensor_fusion_testing
   ```

3. **Run your scripts:**
   ```bash
   python your_script.py
   ```

## Troubleshooting

### Common Issues

#### 1. Python Version Mismatch

**Error:** "Python version not found" or compatibility issues
**Solution:** Ensure you have Python 3.10.14 installed

#### 2. Package Installation Failures (WSL)

**Error:** Missing system libraries
**Solution:** Install the system dependencies listed above

#### 3. OpenCV Issues (WSL)

**Error:** ImportError with OpenCV
**Solution:**

```bash
sudo apt install -y libgl1-mesa-glx libglib2.0-0
```

#### 4. PyQt5 Issues (WSL)

**Error:** Qt platform plugin issues
**Solution:**

```bash
sudo apt install -y python3-pyqt5
export QT_QPA_PLATFORM=offscreen
```

#### 5. CARLA Import Issues

**Error:** Cannot import carla
**Solution:** Ensure CARLA is properly installed and the Python API is in your PYTHONPATH

#### 6. Memory Issues

**Error:** Out of memory during installation
**Solution:**

```bash
pip install --no-cache-dir -r requirements.txt
```

### WSL-Specific Notes

1. **Display Issues:** If you encounter display-related errors:

   ```bash
   export DISPLAY=:0
   export QT_QPA_PLATFORM=offscreen
   ```

2. **Performance:** WSL2 may have slower I/O. Consider using WSL1 for better performance with file operations.

3. **CARLA Server:** If running CARLA server in WSL, you may need to run it in a separate terminal or use X11 forwarding or ip targeting.

## Environment Verification

To verify your environment is set up correctly:

```python
# Test script: test_environment.py
import sys
print(f"Python version: {sys.version}")

try:
    import carla
    print("✓ CARLA imported successfully")
except ImportError as e:
    print(f"✗ CARLA import failed: {e}")

try:
    import numpy
    print("✓ NumPy imported successfully")
except ImportError as e:
    print(f"✗ NumPy import failed: {e}")

try:
    import cv2
    print("✓ OpenCV imported successfully")
except ImportError as e:
    print(f"✗ OpenCV import failed: {e}")

try:
    import matplotlib
    print("✓ Matplotlib imported successfully")
except ImportError as e:
    print(f"✗ Matplotlib import failed: {e}")

try:
    import jupyter
    print("✓ Jupyter imported successfully")
except ImportError as e:
    print(f"✗ Jupyter import failed: {e}")
```

Run this script to verify all key packages are working:

```bash
python test_environment.py
```

## Additional Notes

- The virtual environment will be created in the `venv/` directory
- You can deactivate the environment anytime with `deactivate`
- The environment includes all the packages from your original conda environment
- Some Windows-specific packages (like `pywin32`) are not included in the requirements.txt and need to be installed separately on Windows
- For WSL, some packages may require additional system dependencies not listed in requirements.txt
