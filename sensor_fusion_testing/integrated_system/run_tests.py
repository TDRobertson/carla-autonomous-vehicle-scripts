import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import test modules
from tests.test_kalman_filter import TestExtendedKalmanFilter
from tests.test_spoofing_detection import TestSpoofingDetection
from tests.test_gps_spoofer import TestGPSSpoofer
from tests.test_car import TestCar
from tests.test_rotations import TestRotations, TestQuaternion
from tests.test_integrated_system import TestIntegratedSystem

def run_tests():
    """Run all tests."""
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestExtendedKalmanFilter))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestSpoofingDetection))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGPSSpoofer))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCar))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRotations))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestQuaternion))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestIntegratedSystem))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success/failure
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1) 