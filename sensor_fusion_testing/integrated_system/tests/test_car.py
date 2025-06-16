import unittest
import carla
import numpy as np
from ..core.car import Car

class TestCar(unittest.TestCase):
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
        
    def test_vehicle_spawn(self):
        """Test vehicle spawning."""
        # Check that vehicle exists
        self.assertIsNotNone(self.car.vehicle)
        
        # Check vehicle location
        location = self.car.get_location()
        self.assertIsNotNone(location)
        self.assertAlmostEqual(location.x, self.spawn_point.location.x)
        self.assertAlmostEqual(location.y, self.spawn_point.location.y)
        self.assertAlmostEqual(location.z, self.spawn_point.location.z)
        
    def test_sensor_setup(self):
        """Test sensor setup."""
        # Check that sensors exist
        self.assertIn('camera', self.car.sensors)
        self.assertIn('imu', self.car.sensors)
        self.assertIn('gnss', self.car.sensors)
        
        # Check camera attributes
        camera = self.car.sensors['camera']
        self.assertEqual(camera.attributes['image_size_x'], '800')
        self.assertEqual(camera.attributes['image_size_y'], '600')
        self.assertEqual(camera.attributes['fov'], '90')
        
    def test_sensor_readings(self):
        """Test sensor readings."""
        # Get sensor readings
        readings = self.car.get_sensor_readings(0)
        
        # Check that readings exist
        self.assertIsNotNone(readings['image'])
        self.assertIsNotNone(readings['imu'])
        self.assertIsNotNone(readings['gnss'])
        
    def test_vehicle_state(self):
        """Test vehicle state methods."""
        # Get vehicle state
        location = self.car.get_location()
        velocity = self.car.get_velocity()
        acceleration = self.car.get_acceleration()
        angular_velocity = self.car.get_angular_velocity()
        
        # Check that state exists
        self.assertIsNotNone(location)
        self.assertIsNotNone(velocity)
        self.assertIsNotNone(acceleration)
        self.assertIsNotNone(angular_velocity)
        
    def test_cleanup(self):
        """Test cleanup method."""
        # Clean up
        self.car.cleanup()
        
        # Check that vehicle is destroyed
        self.assertIsNone(self.car.vehicle)
        
        # Check that sensors are destroyed
        self.assertEqual(len(self.car.sensors), 0)
        
if __name__ == '__main__':
    unittest.main() 