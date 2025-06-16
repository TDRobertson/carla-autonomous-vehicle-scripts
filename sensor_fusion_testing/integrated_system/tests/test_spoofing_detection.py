import unittest
import numpy as np
from ..core.kalman_filter import ExtendedKalmanFilter
from ..core.spoofing_detector import SpoofingDetector
from ..core.gps_spoofer import GPSSpoofer, SpoofingStrategy

class TestSpoofingDetection(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        # Initialize EKF with default parameters
        self.ekf = ExtendedKalmanFilter()
        
        # Initialize spoofing detector
        self.detector = SpoofingDetector(self.ekf)
        
        # Initialize GPS spoofer
        self.spoofer = GPSSpoofer()
        
        # Initial position and velocity
        self.initial_position = np.array([0.0, 0.0, 0.0])
        self.initial_velocity = np.array([1.0, 0.0, 0.0])
        
        # Initialize EKF with initial state
        self.ekf.initialize_with_gnss(
            self.initial_position,
            self.initial_velocity
        )
        
    def test_no_spoofing(self):
        """Test detection with no spoofing."""
        # Update with normal measurements
        position = self.initial_position + self.initial_velocity
        velocity = self.initial_velocity
        
        # Update EKF
        self.ekf.predict(np.zeros(3), np.zeros(3))  # No IMU data
        self.ekf.correct_with_gnss(position, velocity)
        
        # Check spoofing detection
        spoofing_detected, _ = self.detector.update(position, velocity)
        self.assertFalse(spoofing_detected)
        
    def test_jump_spoofing(self):
        """Test detection with jump spoofing."""
        # Set jump spoofing strategy
        self.spoofer.set_strategy(SpoofingStrategy.JUMP)
        
        # Get spoofed position
        true_position = self.initial_position + self.initial_velocity
        spoofed_position = self.spoofer.spoof_position(true_position)
        
        # Update EKF
        self.ekf.predict(np.zeros(3), np.zeros(3))  # No IMU data
        self.ekf.correct_with_gnss(true_position, self.initial_velocity)
        
        # Check spoofing detection
        spoofing_detected, _ = self.detector.update(spoofed_position, self.initial_velocity)
        self.assertTrue(spoofing_detected)
        
    def test_drift_spoofing(self):
        """Test detection with drift spoofing."""
        # Set drift spoofing strategy
        self.spoofer.set_strategy(SpoofingStrategy.DRIFT)
        
        # Get spoofed position
        true_position = self.initial_position + self.initial_velocity
        spoofed_position = self.spoofer.spoof_position(true_position)
        
        # Update EKF
        self.ekf.predict(np.zeros(3), np.zeros(3))  # No IMU data
        self.ekf.correct_with_gnss(true_position, self.initial_velocity)
        
        # Check spoofing detection
        spoofing_detected, _ = self.detector.update(spoofed_position, self.initial_velocity)
        self.assertTrue(spoofing_detected)
        
    def test_random_spoofing(self):
        """Test detection with random spoofing."""
        # Set random spoofing strategy
        self.spoofer.set_strategy(SpoofingStrategy.RANDOM)
        
        # Get spoofed position
        true_position = self.initial_position + self.initial_velocity
        spoofed_position = self.spoofer.spoof_position(true_position)
        
        # Update EKF
        self.ekf.predict(np.zeros(3), np.zeros(3))  # No IMU data
        self.ekf.correct_with_gnss(true_position, self.initial_velocity)
        
        # Check spoofing detection
        spoofing_detected, _ = self.detector.update(spoofed_position, self.initial_velocity)
        self.assertTrue(spoofing_detected)
        
    def test_sequential_spoofing(self):
        """Test detection with sequential spoofing."""
        # Set sequential spoofing strategy
        self.spoofer.set_strategy(SpoofingStrategy.SEQUENTIAL)
        
        # Get spoofed position
        true_position = self.initial_position + self.initial_velocity
        spoofed_position = self.spoofer.spoof_position(true_position)
        
        # Update EKF
        self.ekf.predict(np.zeros(3), np.zeros(3))  # No IMU data
        self.ekf.correct_with_gnss(true_position, self.initial_velocity)
        
        # Check spoofing detection
        spoofing_detected, _ = self.detector.update(spoofed_position, self.initial_velocity)
        self.assertTrue(spoofing_detected)
        
if __name__ == '__main__':
    unittest.main() 