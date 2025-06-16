import unittest
import carla
import numpy as np
from ..core.car import Car
from ..core.kalman_filter import ExtendedKalmanFilter
from ..core.spoofing_detector import SpoofingDetector
from ..core.gps_spoofer import GPSSpoofer, SpoofingStrategy

class TestIntegratedSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for the entire test class."""
        # Connect to CARLA server
        cls.client = carla.Client('localhost', 2000)
        cls.client.set_timeout(10.0)
        
        # Get world
        cls.world = cls.client.get_world()
        
        # Set synchronous mode
        settings = cls.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        cls.world.apply_settings(settings)
        
    def setUp(self):
        """Set up test fixtures for each test method."""
        # Get spawn point
        spawn_points = self.world.get_map().get_spawn_points()
        self.spawn_point = spawn_points[0]
        
        # Create car
        self.car = Car(self.world, self.client, self.spawn_point)
        
        # Create EKF
        self.ekf = ExtendedKalmanFilter()
        
        # Create spoofing detector
        self.detector = SpoofingDetector(self.ekf)
        
        # Create GPS spoofer
        self.spoofer = GPSSpoofer()
        
        # Initialize EKF with initial state
        initial_position = np.array([
            self.spawn_point.location.x,
            self.spawn_point.location.y,
            self.spawn_point.location.z
        ])
        initial_velocity = np.array([0.0, 0.0, 0.0])
        self.ekf.initialize_with_gnss(initial_position, initial_velocity)
        
    def tearDown(self):
        """Clean up after each test method."""
        self.car.cleanup()
        
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Reset world settings
        settings = cls.world.get_settings()
        settings.synchronous_mode = False
        cls.world.apply_settings(settings)
        
    def test_normal_operation(self):
        """Test normal operation without spoofing."""
        # Get sensor readings
        readings = self.car.get_sensor_readings(0)
        
        # Get GNSS data
        gnss = readings['gnss']
        position = np.array([gnss.latitude, gnss.longitude, gnss.altitude])
        velocity = np.array([gnss.velocity.x, gnss.velocity.y, gnss.velocity.z])
        
        # Get IMU data
        imu = readings['imu']
        accel = np.array([imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z])
        gyro = np.array([imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z])
        
        # Update EKF
        self.ekf.predict(accel, gyro)
        self.ekf.correct_with_gnss(position, velocity)
        
        # Check spoofing detection
        spoofing_detected, _ = self.detector.update(position, velocity)
        self.assertFalse(spoofing_detected)
        
    def test_jump_spoofing(self):
        """Test system with jump spoofing."""
        # Set jump spoofing strategy
        self.spoofer.set_strategy(SpoofingStrategy.JUMP)
        
        # Get sensor readings
        readings = self.car.get_sensor_readings(0)
        
        # Get GNSS data
        gnss = readings['gnss']
        true_position = np.array([gnss.latitude, gnss.longitude, gnss.altitude])
        velocity = np.array([gnss.velocity.x, gnss.velocity.y, gnss.velocity.z])
        
        # Spoof position
        spoofed_position = self.spoofer.spoof_position(true_position)
        
        # Get IMU data
        imu = readings['imu']
        accel = np.array([imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z])
        gyro = np.array([imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z])
        
        # Update EKF
        self.ekf.predict(accel, gyro)
        self.ekf.correct_with_gnss(true_position, velocity)  # Use true position for EKF
        
        # Check spoofing detection
        spoofing_detected, corrected_position = self.detector.update(spoofed_position, velocity)
        self.assertTrue(spoofing_detected)
        self.assertIsNotNone(corrected_position)
        
    def test_drift_spoofing(self):
        """Test system with drift spoofing."""
        # Set drift spoofing strategy
        self.spoofer.set_strategy(SpoofingStrategy.DRIFT)
        
        # Get sensor readings
        readings = self.car.get_sensor_readings(0)
        
        # Get GNSS data
        gnss = readings['gnss']
        true_position = np.array([gnss.latitude, gnss.longitude, gnss.altitude])
        velocity = np.array([gnss.velocity.x, gnss.velocity.y, gnss.velocity.z])
        
        # Spoof position
        spoofed_position = self.spoofer.spoof_position(true_position)
        
        # Get IMU data
        imu = readings['imu']
        accel = np.array([imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z])
        gyro = np.array([imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z])
        
        # Update EKF
        self.ekf.predict(accel, gyro)
        self.ekf.correct_with_gnss(true_position, velocity)  # Use true position for EKF
        
        # Check spoofing detection
        spoofing_detected, corrected_position = self.detector.update(spoofed_position, velocity)
        self.assertTrue(spoofing_detected)
        self.assertIsNotNone(corrected_position)
        
    def test_random_spoofing(self):
        """Test system with random spoofing."""
        # Set random spoofing strategy
        self.spoofer.set_strategy(SpoofingStrategy.RANDOM)
        
        # Get sensor readings
        readings = self.car.get_sensor_readings(0)
        
        # Get GNSS data
        gnss = readings['gnss']
        true_position = np.array([gnss.latitude, gnss.longitude, gnss.altitude])
        velocity = np.array([gnss.velocity.x, gnss.velocity.y, gnss.velocity.z])
        
        # Spoof position
        spoofed_position = self.spoofer.spoof_position(true_position)
        
        # Get IMU data
        imu = readings['imu']
        accel = np.array([imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z])
        gyro = np.array([imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z])
        
        # Update EKF
        self.ekf.predict(accel, gyro)
        self.ekf.correct_with_gnss(true_position, velocity)  # Use true position for EKF
        
        # Check spoofing detection
        spoofing_detected, corrected_position = self.detector.update(spoofed_position, velocity)
        self.assertTrue(spoofing_detected)
        self.assertIsNotNone(corrected_position)
        
    def test_sequential_spoofing(self):
        """Test system with sequential spoofing."""
        # Set sequential spoofing strategy
        self.spoofer.set_strategy(SpoofingStrategy.SEQUENTIAL)
        
        # Get sensor readings
        readings = self.car.get_sensor_readings(0)
        
        # Get GNSS data
        gnss = readings['gnss']
        true_position = np.array([gnss.latitude, gnss.longitude, gnss.altitude])
        velocity = np.array([gnss.velocity.x, gnss.velocity.y, gnss.velocity.z])
        
        # Spoof position
        spoofed_position = self.spoofer.spoof_position(true_position)
        
        # Get IMU data
        imu = readings['imu']
        accel = np.array([imu.accelerometer.x, imu.accelerometer.y, imu.accelerometer.z])
        gyro = np.array([imu.gyroscope.x, imu.gyroscope.y, imu.gyroscope.z])
        
        # Update EKF
        self.ekf.predict(accel, gyro)
        self.ekf.correct_with_gnss(true_position, velocity)  # Use true position for EKF
        
        # Check spoofing detection
        spoofing_detected, corrected_position = self.detector.update(spoofed_position, velocity)
        self.assertTrue(spoofing_detected)
        self.assertIsNotNone(corrected_position)
        
if __name__ == '__main__':
    unittest.main() 