#!/usr/bin/env python3

import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import math
import argparse

# Add the CARLA Python API directory to the path
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

class SensorFusionVehicle:
    def __init__(self, enable_camera=True, enable_lidar=True, enable_radar=True, num_traffic_vehicles=20, spectator_mode=False):
        # Connect to CARLA server
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Set up synchronous mode
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = 0.05  # 20 FPS
        self.world.apply_settings(self.settings)
        
        self.blueprint_library = self.world.get_blueprint_library()
        
        # Get the map spawn points
        self.spawn_points = self.world.get_map().get_spawn_points()
        
        # Initialize the traffic manager
        self.traffic_manager = self.client.get_trafficmanager(8000)  # Port 8000
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(42)  # Set seed for deterministic behavior
        self.traffic_manager.global_percentage_speed_difference(10.0)  # Make traffic move at reasonable speed
        
        # Store sensor flags and traffic settings
        self.enable_camera = enable_camera
        self.enable_lidar = enable_lidar
        self.enable_radar = enable_radar
        self.num_traffic_vehicles = num_traffic_vehicles
        self.spectator_mode = spectator_mode
        
        # List to store actors for cleanup
        self.actor_list = []
        self.traffic_vehicles = []
        
        # Sensor data storage
        self.sensor_data = {
            'camera_image': None,
            'lidar_points': None,
            'radar_points': None
        }
        
        # Vehicle control parameters
        self.emergency_stop = False
        self.emergency_stop_duration = 0
        
        # Get the spectator
        self.spectator = self.world.get_spectator()
        
    def update_spectator_camera(self):
        """Update spectator camera to follow ego vehicle"""
        if not self.spectator_mode or not hasattr(self, 'vehicle'):
            return
            
        # Get vehicle transform
        vehicle_transform = self.vehicle.get_transform()
        
        # Calculate camera position behind and above the vehicle
        camera_offset = carla.Location(x=-8, z=4)  # 8 meters behind, 4 meters up
        camera_location = vehicle_transform.transform(camera_offset)
        
        # Calculate camera rotation to look at vehicle
        camera_rotation = vehicle_transform.rotation
        camera_rotation.pitch = -15  # Look down slightly
        
        # Set spectator transform
        camera_transform = carla.Transform(camera_location, camera_rotation)
        self.spectator.set_transform(camera_transform)
        
    def spawn_traffic(self):
        """Spawn traffic vehicles"""
        print(f"Spawning {self.num_traffic_vehicles} traffic vehicles...")
        
        # Get all vehicle blueprints
        vehicle_bps = self.blueprint_library.filter('vehicle.*')
        
        # Remove ego vehicle spawn point from available points
        available_spawn_points = list(self.spawn_points)
        if hasattr(self, 'vehicle'):
            ego_spawn_point = self.vehicle.get_transform()
            if ego_spawn_point in available_spawn_points:
                available_spawn_points.remove(ego_spawn_point)
        
        # Spawn vehicles
        for _ in range(self.num_traffic_vehicles):
            if not available_spawn_points:
                print("No more spawn points available")
                break
                
            # Choose random vehicle blueprint and spawn point
            vehicle_bp = random.choice(vehicle_bps)
            
            # Ensure the vehicle will autopilot properly
            if vehicle_bp.has_attribute('color'):
                vehicle_bp.set_attribute('color', random.choice(vehicle_bp.get_attribute('color').recommended_values))
            if vehicle_bp.has_attribute('driver_id'):
                vehicle_bp.set_attribute('driver_id', random.choice(vehicle_bp.get_attribute('driver_id').recommended_values))
            
            spawn_point = random.choice(available_spawn_points)
            available_spawn_points.remove(spawn_point)  # Remove used spawn point
            
            # Try to spawn the vehicle
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle is not None:
                self.traffic_vehicles.append(vehicle)
                self.actor_list.append(vehicle)
                
                # Set autopilot
                vehicle.set_autopilot(True, self.traffic_manager.get_port())
                
                # Add some randomness to the traffic behavior
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle, random.uniform(-20, 10))
                self.traffic_manager.distance_to_leading_vehicle(vehicle, random.uniform(1.0, 4.0))
                self.traffic_manager.auto_lane_change(vehicle, True)
        
        # Wait for a few ticks to let the vehicles settle
        for _ in range(10):
            self.world.tick()
        
        print(f"Successfully spawned {len(self.traffic_vehicles)} traffic vehicles")
        
    def spawn_ego_vehicle(self):
        """Spawn the ego vehicle and attach sensors based on flags"""
        # Choose a random spawn point
        spawn_point = random.choice(self.spawn_points)
        
        # Spawn the vehicle
        vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
        self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.actor_list.append(self.vehicle)
        
        # Enable autopilot with traffic manager
        self.vehicle.set_autopilot(True, self.traffic_manager.get_port())
        
        # Attach sensors based on flags
        if self.enable_camera:
            self._attach_camera()
        if self.enable_lidar:
            self._attach_lidar()
        if self.enable_radar:
            self._attach_radar()
            
        # If spectator mode is enabled, position the camera
        if self.spectator_mode:
            self.update_spectator_camera()
            
        return self.vehicle
    
    def _attach_camera(self):
        """Attach and configure RGB camera"""
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)
        self.actor_list.append(self.camera)
        
        # Set up camera callback
        self.camera.listen(lambda image: self._camera_callback(image))
        
    def _attach_lidar(self):
        """Attach and configure LiDAR sensor"""
        lidar_bp = self.blueprint_library.find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels', '32')
        lidar_bp.set_attribute('points_per_second', '100000')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('range', '20')
        
        lidar_transform = carla.Transform(carla.Location(x=0, z=2.4))
        self.lidar = self.world.spawn_actor(lidar_bp, lidar_transform, attach_to=self.vehicle)
        self.actor_list.append(self.lidar)
        
        # Set up LiDAR callback
        self.lidar.listen(lambda point_cloud: self._lidar_callback(point_cloud))
        
    def _attach_radar(self):
        """Attach and configure radar sensor"""
        radar_bp = self.blueprint_library.find('sensor.other.radar')
        radar_bp.set_attribute('horizontal_fov', '30')
        radar_bp.set_attribute('vertical_fov', '10')
        radar_bp.set_attribute('range', '20')
        
        radar_transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        self.radar = self.world.spawn_actor(radar_bp, radar_transform, attach_to=self.vehicle)
        self.actor_list.append(self.radar)
        
        # Set up radar callback
        self.radar.listen(lambda radar_data: self._radar_callback(radar_data))
    
    def _camera_callback(self, image):
        """Process camera data"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        self.sensor_data['camera_image'] = array
        
    def _lidar_callback(self, point_cloud):
        """Process LiDAR data"""
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        self.sensor_data['lidar_points'] = data
        
    def _radar_callback(self, radar_data):
        """Process radar data and adjust vehicle behavior"""
        # Get current vehicle velocity
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
        
        # Process radar detections
        for detection in radar_data:
            # Calculate distance to detected object
            distance = detection.depth
            
            # If object is too close and we're moving
            if distance < 5.0 and speed > 5.0:  # 5 meters and 5 km/h thresholds
                self._handle_close_object(distance)
                break
        else:
            # No close objects detected, resume normal operation
            self._resume_normal_operation()
    
    def _handle_close_object(self, distance):
        """Handle detection of close objects"""
        # Set emergency stop flag
        self.emergency_stop = True
        self.emergency_stop_duration = time.time()
        
        # Get autopilot control and modify it
        control = self.vehicle.get_control()
        
        # Apply stronger braking the closer the object
        brake_intensity = min(1.0, (5.0 - distance) / 5.0)
        control.brake = brake_intensity
        control.throttle = 0.0
        
        # Apply the modified control
        self.vehicle.apply_control(control)
        
    def _resume_normal_operation(self):
        """Resume normal operation after emergency stop"""
        if self.emergency_stop:
            # Only resume if emergency stop has been active for at least 1 second
            if time.time() - self.emergency_stop_duration > 1.0:
                self.emergency_stop = False
                # Re-enable normal autopilot behavior
                self.vehicle.set_autopilot(True, self.traffic_manager.get_port())
    
    def cleanup(self):
        """Cleanup all actors"""
        print("Cleaning up actors...")
        
        # Disable synchronous mode
        self.settings.synchronous_mode = False
        self.world.apply_settings(self.settings)
        
        for actor in self.actor_list:
            if actor is not None:
                actor.destroy()
        print("Cleanup complete.")

def main():
    """Main function to run the sensor fusion vehicle"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run CARLA sensor fusion vehicle')
    parser.add_argument('--no-camera', action='store_true', help='Disable camera sensor')
    parser.add_argument('--no-lidar', action='store_true', help='Disable LiDAR sensor')
    parser.add_argument('--no-radar', action='store_true', help='Disable radar sensor')
    parser.add_argument('--num-traffic', type=int, default=20, help='Number of traffic vehicles to spawn')
    parser.add_argument('--spectator', action='store_true', help='Enable spectator camera to follow ego vehicle')
    args = parser.parse_args()
    
    # Create sensor fusion vehicle instance
    sensor_fusion = SensorFusionVehicle(
        enable_camera=not args.no_camera,
        enable_lidar=not args.no_lidar,
        enable_radar=not args.no_radar,
        num_traffic_vehicles=args.num_traffic,
        spectator_mode=args.spectator
    )
    
    try:
        # Spawn the ego vehicle
        vehicle = sensor_fusion.spawn_ego_vehicle()
        print("Ego vehicle spawned successfully!")
        
        # Spawn traffic
        sensor_fusion.spawn_traffic()
        print("Traffic spawned successfully!")
        print("Press Ctrl+C to exit...")
        
        # Main simulation loop
        while True:
            # Update spectator camera if enabled
            if args.spectator:
                sensor_fusion.update_spectator_camera()
                
            # Tick the world to advance the simulation
            sensor_fusion.world.tick()
            time.sleep(0.05)  # Cap at 20 FPS
            
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
    finally:
        # Cleanup
        sensor_fusion.cleanup()

if __name__ == '__main__':
    main() 