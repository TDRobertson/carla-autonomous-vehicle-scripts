import carla
import numpy as np
import time
from typing import Dict, Optional

class Car:
    def __init__(self, world, client, spawn_point):
        self.world = world
        self.client = client
        self.spawn_point = spawn_point
        
        # Spawn vehicle
        self.vehicle = self._spawn_vehicle()
        
        # Setup sensors
        self.sensors = {}
        self.setup_sensors()
        
    def _spawn_vehicle(self) -> carla.Actor:
        """Spawn the vehicle at the specified spawn point."""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        vehicle = self.world.spawn_actor(vehicle_bp, self.spawn_point)
        
        # Wait for spawn
        time.sleep(2.0)
        
        return vehicle
        
    def setup_sensors(self):
        """Setup vehicle sensors."""
        # Setup camera
        camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        self.sensors['camera'] = self.world.spawn_actor(
            camera_bp,
            camera_transform,
            attach_to=self.vehicle
        )
        
        # Setup IMU
        imu_bp = self.world.get_blueprint_library().find('sensor.other.imu')
        self.sensors['imu'] = self.world.spawn_actor(
            imu_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        
        # Setup GPS
        gps_bp = self.world.get_blueprint_library().find('sensor.other.gnss')
        self.sensors['gnss'] = self.world.spawn_actor(
            gps_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        
    def get_sensor_readings(self, frame: int) -> Dict:
        """Get current sensor readings."""
        readings = {
            'image': None,
            'imu': None,
            'gnss': None
        }
        
        # Get camera image
        if 'camera' in self.sensors:
            image = self.sensors['camera'].get_data()
            if image.frame == frame:
                readings['image'] = image
                
        # Get IMU data
        if 'imu' in self.sensors:
            imu = self.sensors['imu'].get_data()
            if imu.frame == frame:
                readings['imu'] = imu
                
        # Get GPS data
        if 'gnss' in self.sensors:
            gnss = self.sensors['gnss'].get_data()
            if gnss.frame == frame:
                readings['gnss'] = gnss
                
        return readings
        
    def get_location(self) -> Optional[carla.Location]:
        """Get current vehicle location."""
        if self.vehicle:
            return self.vehicle.get_location()
        return None
        
    def get_velocity(self) -> Optional[carla.Vector3D]:
        """Get current vehicle velocity."""
        if self.vehicle:
            return self.vehicle.get_velocity()
        return None
        
    def get_acceleration(self) -> Optional[carla.Vector3D]:
        """Get current vehicle acceleration."""
        if self.vehicle:
            return self.vehicle.get_acceleration()
        return None
        
    def get_angular_velocity(self) -> Optional[carla.Vector3D]:
        """Get current vehicle angular velocity."""
        if self.vehicle:
            return self.vehicle.get_angular_velocity()
        return None
        
    def cleanup(self):
        """Clean up vehicle and sensors."""
        for sensor in self.sensors.values():
            sensor.destroy()
        if self.vehicle:
            self.vehicle.destroy() 