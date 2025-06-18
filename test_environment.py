#!/usr/bin/env python3
"""
Environment test script for CARLA autonomous vehicle environment.
This script verifies that all key packages are working correctly.
"""

import sys
import platform

def test_python_version():
    """Test Python version."""
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    print()

def test_core_packages():
    """Test core scientific computing packages."""
    packages = [
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("matplotlib", "Matplotlib"),
        ("scipy", "SciPy"),
        ("scikit-learn", "Scikit-learn"),
        ("seaborn", "Seaborn"),
    ]
    
    print("Testing core packages:")
    for package, name in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✓ {name} imported successfully (version: {version})")
        except ImportError as e:
            print(f"✗ {name} import failed: {e}")
    print()

def test_carla_packages():
    """Test CARLA and related packages."""
    packages = [
        ("carla", "CARLA"),
        ("pygame", "Pygame"),
        ("cv2", "OpenCV"),
        ("open3d", "Open3D"),
    ]
    
    print("Testing CARLA and related packages:")
    for package, name in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✓ {name} imported successfully (version: {version})")
        except ImportError as e:
            print(f"✗ {name} import failed: {e}")
    print()

def test_jupyter_packages():
    """Test Jupyter ecosystem packages."""
    packages = [
        ("jupyter", "Jupyter"),
        ("jupyterlab", "JupyterLab"),
        ("ipython", "IPython"),
        ("notebook", "Notebook"),
    ]
    
    print("Testing Jupyter ecosystem:")
    for package, name in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✓ {name} imported successfully (version: {version})")
        except ImportError as e:
            print(f"✗ {name} import failed: {e}")
    print()

def test_visualization_packages():
    """Test visualization packages."""
    packages = [
        ("plotly", "Plotly"),
        ("dash", "Dash"),
        ("flask", "Flask"),
    ]
    
    print("Testing visualization packages:")
    for package, name in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✓ {name} imported successfully (version: {version})")
        except ImportError as e:
            print(f"✗ {name} import failed: {e}")
    print()

def test_utility_packages():
    """Test utility packages."""
    packages = [
        ("requests", "Requests"),
        ("yaml", "PyYAML"),
        ("configargparse", "ConfigArgParse"),
        ("psutil", "psutil"),
        ("PIL", "Pillow"),
    ]
    
    print("Testing utility packages:")
    for package, name in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'Unknown')
            print(f"✓ {name} imported successfully (version: {version})")
        except ImportError as e:
            print(f"✗ {name} import failed: {e}")
    print()

def test_platform_specific():
    """Test platform-specific packages."""
    system = platform.system().lower()
    
    print("Testing platform-specific packages:")
    if system == "windows":
        try:
            import win32api
            print("✓ pywin32 imported successfully")
        except ImportError as e:
            print(f"✗ pywin32 import failed: {e}")
    else:
        print("✓ Skipping Windows-specific packages (not on Windows)")
    print()

def test_carla_connection():
    """Test CARLA connection (if possible)."""
    print("Testing CARLA connection:")
    try:
        import carla
        print("✓ CARLA module imported successfully")
        
        # Try to connect to CARLA server (optional)
        try:
            client = carla.Client('localhost', 2000)
            client.set_timeout(2.0)
            world = client.get_world()
            print("✓ Successfully connected to CARLA server")
        except Exception as e:
            print(f"⚠ CARLA server not running or not accessible: {e}")
            print("  This is normal if CARLA server is not started")
            
    except ImportError as e:
        print(f"✗ CARLA import failed: {e}")
    print()

def main():
    """Main test function."""
    print("CARLA Environment Test")
    print("=" * 50)
    
    test_python_version()
    test_core_packages()
    test_carla_packages()
    test_jupyter_packages()
    test_visualization_packages()
    test_utility_packages()
    test_platform_specific()
    test_carla_connection()
    
    print("=" * 50)
    print("Environment test completed!")
    print("\nIf you see any ✗ marks, those packages failed to import.")
    print("Check the SETUP_INSTRUCTIONS.md for troubleshooting steps.")

if __name__ == "__main__":
    main() 