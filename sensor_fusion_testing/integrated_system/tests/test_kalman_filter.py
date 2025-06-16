import unittest
import numpy as np
from ..core.kalman_filter import ExtendedKalmanFilter
from ..utils.rotations import Quaternion

class TestExtendedKalmanFilter(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.ekf = ExtendedKalmanFilter()
        
        # Initial state
        self.initial_position = np.array([0.0, 0.0, 0.0])
        self.initial_velocity = np.array([1.0, 0.0, 0.0])
        self.initial_orientation = Quaternion()
        
    def test_initialization(self):
        """Test EKF initialization with GNSS data."""
        # Initialize EKF
        self.ekf.initialize_with_gnss(
            self.initial_position,
            self.initial_velocity
        )
        
        # Get state
        state = self.ekf.get_state()
        
        # Check position
        np.testing.assert_array_almost_equal(
            state[:3],
            self.initial_position
        )
        
        # Check velocity
        np.testing.assert_array_almost_equal(
            state[3:6],
            self.initial_velocity
        )
        
        # Check orientation (should be identity quaternion)
        orientation = state[6:10]
        np.testing.assert_array_almost_equal(
            orientation,
            np.array([1.0, 0.0, 0.0, 0.0])
        )
        
    def test_prediction(self):
        """Test EKF prediction step."""
        # Initialize EKF
        self.ekf.initialize_with_gnss(
            self.initial_position,
            self.initial_velocity
        )
        
        # IMU measurements
        accel = np.array([0.0, 0.0, 0.0])
        gyro = np.array([0.0, 0.0, 0.0])
        
        # Predict
        self.ekf.predict(accel, gyro)
        
        # Get state
        state = self.ekf.get_state()
        
        # Check position (should be initial + velocity)
        expected_position = self.initial_position + self.initial_velocity
        np.testing.assert_array_almost_equal(
            state[:3],
            expected_position
        )
        
        # Check velocity (should be unchanged)
        np.testing.assert_array_almost_equal(
            state[3:6],
            self.initial_velocity
        )
        
    def test_correction(self):
        """Test EKF correction step with GNSS data."""
        # Initialize EKF
        self.ekf.initialize_with_gnss(
            self.initial_position,
            self.initial_velocity
        )
        
        # Predict
        self.ekf.predict(np.zeros(3), np.zeros(3))
        
        # New GNSS measurement
        new_position = self.initial_position + self.initial_velocity
        new_velocity = self.initial_velocity
        
        # Correct
        self.ekf.correct_with_gnss(new_position, new_velocity)
        
        # Get state
        state = self.ekf.get_state()
        
        # Check position
        np.testing.assert_array_almost_equal(
            state[:3],
            new_position
        )
        
        # Check velocity
        np.testing.assert_array_almost_equal(
            state[3:6],
            new_velocity
        )
        
    def test_imu_integration(self):
        """Test EKF with IMU measurements."""
        # Initialize EKF
        self.ekf.initialize_with_gnss(
            self.initial_position,
            self.initial_velocity
        )
        
        # IMU measurements
        accel = np.array([1.0, 0.0, 0.0])  # 1 m/s^2 in x direction
        gyro = np.array([0.0, 0.0, 0.0])  # No rotation
        
        # Predict
        self.ekf.predict(accel, gyro)
        
        # Get state
        state = self.ekf.get_state()
        
        # Check velocity (should be initial + acceleration)
        expected_velocity = self.initial_velocity + accel
        np.testing.assert_array_almost_equal(
            state[3:6],
            expected_velocity
        )
        
        # Check position (should be initial + average velocity)
        expected_position = self.initial_position + 0.5 * (self.initial_velocity + expected_velocity)
        np.testing.assert_array_almost_equal(
            state[:3],
            expected_position
        )
        
    def test_orientation_update(self):
        """Test EKF orientation update with gyroscope."""
        # Initialize EKF
        self.ekf.initialize_with_gnss(
            self.initial_position,
            self.initial_velocity
        )
        
        # IMU measurements with rotation
        accel = np.zeros(3)
        gyro = np.array([0.0, 0.0, 1.0])  # 1 rad/s around z-axis
        
        # Predict
        self.ekf.predict(accel, gyro)
        
        # Get state
        state = self.ekf.get_state()
        
        # Check orientation (should be rotated around z-axis)
        orientation = state[6:10]
        expected_orientation = Quaternion.from_euler(0, 0, 1.0)  # 1 rad around z
        np.testing.assert_array_almost_equal(
            orientation,
            expected_orientation.to_numpy()
        )
        
if __name__ == '__main__':
    unittest.main() 