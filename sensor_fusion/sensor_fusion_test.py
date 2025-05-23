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
import threading
import pygame

# Add the CARLA Python API directory to the path
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

class SensorFusionTest:
    def __init__(self, enable_camera=True, enable_lidar=True, enable_radar=True, num_traffic_vehicles=20):
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
        self.spawn_points = self.world.get_map().get_spawn_points()
        
        # Initialize traffic manager
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.0)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(42)
        self.traffic_manager.global_percentage_speed_difference(10.0)
        
        # Store sensor flags and settings
        self.enable_camera = enable_camera
        self.enable_lidar = enable_lidar
        self.enable_radar = enable_radar
        self.num_traffic_vehicles = num_traffic_vehicles
        
        # Lists to store actors
        self.actor_list = []
        self.traffic_vehicles = []
        
        # ACC parameters
        self.acc_enabled = False
        self.target_speed = 50.0  # km/h
        self.min_distance = 5.0   # meters
        self.max_distance = 20.0  # meters
        self.time_gap = 1.5       # seconds
        self.acceleration = 0.0
        self.braking = 0.0
        self.steering = 0.0
        
        # Vehicle control state
        self.current_speed = 0.0
        self.leading_vehicle_distance = float('inf')
        self.leading_vehicle_speed = 0.0
        self.last_control_time = 0.0
        self.control_update_interval = 0.1  # seconds
        self.emergency_braking = False
        
        # Safety parameters
        self.max_speed = 80.0  # km/h
        self.emergency_distance = 3.0  # meters
        self.speed_threshold = 1.0  # km/h
        
        # Sensor data storage
        self.sensor_data = {
            'camera_image': None,
            'lidar_points': None,
            'radar_points': None
        }
        
        # Detection results
        self.detection_results = {
            'camera_detections': [],
            'lidar_detections': [],
            'radar_detections': []
        }
        
    def spawn_traffic(self):
        """Spawn traffic vehicles"""
        for _ in range(self.num_traffic_vehicles):
            # Choose random vehicle blueprint
            vehicle_bp = random.choice(self.blueprint_library.filter('vehicle.*'))
            
            # Choose random spawn point
            spawn_point = random.choice(self.spawn_points)
            
            # Spawn vehicle
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            
            if vehicle is not None:
                self.traffic_vehicles.append(vehicle)
                self.actor_list.append(vehicle)
                
                # Set autopilot
                vehicle.set_autopilot(True, self.traffic_manager.get_port())
                
                # Add randomness to traffic behavior
                self.traffic_manager.vehicle_percentage_speed_difference(vehicle, random.uniform(-20, 10))
                self.traffic_manager.distance_to_leading_vehicle(vehicle, random.uniform(1.0, 4.0))
                self.traffic_manager.auto_lane_change(vehicle, True)
        
        # Wait for vehicles to settle
        for _ in range(10):
            self.world.tick()
            
        print(f"Successfully spawned {len(self.traffic_vehicles)} traffic vehicles")
        
    def spawn_ego_vehicle(self):
        """Spawn ego vehicle with sensors"""
        # Try to find a clear spawn point
        max_attempts = 50
        spawn_point = None
        vehicle = None
        
        for _ in range(max_attempts):
            # Choose a random spawn point
            spawn_point = random.choice(self.spawn_points)
            
            # Check if the spawn point is clear
            # Get all actors in the world
            actors = self.world.get_actors()
            
            # Check if any vehicle is too close to the spawn point
            spawn_location = spawn_point.location
            is_clear = True
            
            for actor in actors:
                if actor.type_id.startswith('vehicle.'):
                    actor_location = actor.get_location()
                    distance = spawn_location.distance(actor_location)
                    if distance < 5.0:  # Minimum distance of 5 meters
                        is_clear = False
                        break
            
            if is_clear:
                # Try to spawn the vehicle
                vehicle_bp = self.blueprint_library.find('vehicle.tesla.model3')
                vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                if vehicle is not None:
                    break
        
        if vehicle is None:
            raise RuntimeError("Could not find a clear spawn point for the ego vehicle")
        
        self.vehicle = vehicle
        self.actor_list.append(self.vehicle)
        
        # Enable traffic manager for path following
        self.vehicle.set_autopilot(True, self.traffic_manager.get_port())
        
        # Configure traffic manager for the ego vehicle
        self.traffic_manager.auto_lane_change(self.vehicle, True)
        self.traffic_manager.distance_to_leading_vehicle(self.vehicle, self.min_distance)
        self.traffic_manager.vehicle_percentage_speed_difference(self.vehicle, 0)  # No speed difference
        
        # Attach sensors
        if self.enable_camera:
            self._attach_camera()
        if self.enable_lidar:
            self._attach_lidar()
        if self.enable_radar:
            self._attach_radar()
            
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
        
        self.radar.listen(lambda radar_data: self._radar_callback(radar_data))
    
    def _camera_callback(self, image):
        """Process camera data"""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        self.sensor_data['camera_image'] = array
        
        # TODO: Add object detection processing
        # This would typically involve running a neural network for object detection
        
    def _lidar_callback(self, point_cloud):
        """Process LiDAR data"""
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        self.sensor_data['lidar_points'] = data
        
        # Use LiDAR as backup for distance measurement if radar fails
        if self.acc_enabled and self.leading_vehicle_distance == float('inf'):
            # Simple clustering to find closest point cloud cluster
            # This is a simplified version - in practice, you'd want more sophisticated clustering
            points = data[:, :3]  # x, y, z coordinates
            distances = np.linalg.norm(points, axis=1)
            if len(distances) > 0:
                min_distance = np.min(distances)
                if min_distance < self.max_distance:
                    self.leading_vehicle_distance = min_distance
                    self._update_acc_control()
        
    def _radar_callback(self, radar_data):
        """Process radar data for ACC"""
        radar_points = []
        closest_detection = None
        min_distance = float('inf')
        
        for detection in radar_data:
            # Store all radar points
            radar_points.append({
                'depth': detection.depth,
                'azimuth': detection.azimuth,
                'altitude': detection.altitude,
                'velocity': detection.velocity
            })
            
            # Find closest detection in front of vehicle
            if abs(detection.azimuth) < math.radians(30):  # Within radar FOV
                if detection.depth < min_distance:
                    min_distance = detection.depth
                    closest_detection = detection
        
        self.sensor_data['radar_points'] = radar_points
        
        # Update ACC if enabled
        if self.acc_enabled and closest_detection is not None:
            self.leading_vehicle_distance = closest_detection.depth
            self.leading_vehicle_speed = self.current_speed + closest_detection.velocity
            self._update_acc_control()
    
    def _update_acc_control(self):
        """Update vehicle control based on ACC parameters"""
        if not self.acc_enabled:
            return
            
        current_time = time.time()
        if current_time - self.last_control_time < self.control_update_interval:
            return
            
        self.last_control_time = current_time
        
        # Get the current control from traffic manager
        control = self.vehicle.get_control()
        
        # Get current vehicle state
        velocity = self.vehicle.get_velocity()
        self.current_speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
        
        # Safety check - emergency braking
        if self.leading_vehicle_distance < self.emergency_distance:
            self.emergency_braking = True
            control.throttle = 0.0
            control.brake = 1.0
            self.vehicle.apply_control(control)
            return
        
        # Calculate desired distance based on time gap
        desired_distance = max(self.min_distance, self.current_speed * self.time_gap / 3.6)  # Convert km/h to m/s
        
        # Calculate distance error
        distance_error = self.leading_vehicle_distance - desired_distance
        
        # Calculate speed error
        speed_error = self.target_speed - self.current_speed
        
        # Speed control logic
        if distance_error < 0:  # Too close to leading vehicle
            # Gradual speed reduction based on distance
            speed_reduction = min(1.0, abs(distance_error) / self.min_distance)
            target_speed = self.target_speed * (1.0 - speed_reduction)
            speed_error = target_speed - self.current_speed
            
            if speed_error < -self.speed_threshold:
                control.throttle = 0.0
                control.brake = min(1.0, abs(speed_error) / self.target_speed)
            else:
                control.throttle = 0.0
                control.brake = 0.0
        else:
            # Normal speed control
            if abs(speed_error) > self.speed_threshold:
                if speed_error > 0:  # Need to accelerate
                    control.throttle = min(1.0, speed_error / self.target_speed)
                    control.brake = 0.0
                else:  # Need to decelerate
                    control.throttle = 0.0
                    control.brake = min(1.0, abs(speed_error) / self.target_speed)
            else:
                # Maintain current speed
                control.throttle = 0.0
                control.brake = 0.0
        
        # Apply the modified control
        self.vehicle.apply_control(control)

    def toggle_acc(self):
        """Toggle ACC on/off"""
        self.acc_enabled = not self.acc_enabled
        if not self.acc_enabled:
            # Reset control when disabling ACC
            control = self.vehicle.get_control()
            control.throttle = 0.0
            control.brake = 0.0
            self.vehicle.apply_control(control)
            # Re-enable traffic manager control
            self.vehicle.set_autopilot(True, self.traffic_manager.get_port())
            self.emergency_braking = False
        else:
            # When enabling ACC, keep traffic manager for path following
            self.vehicle.set_autopilot(True, self.traffic_manager.get_port())
            self.last_control_time = time.time()
        print(f"ACC {'enabled' if self.acc_enabled else 'disabled'}")

    def run_test(self, duration=60):
        """Run the sensor fusion test for specified duration"""
        print(f"Running sensor fusion test for {duration} seconds...")
        print("Press 'A' to toggle ACC, 'Q' to quit")
        
        start_time = time.time()
        pygame.init()
        screen = pygame.display.set_mode((400, 100))
        pygame.display.set_caption('ACC Control')
        
        while time.time() - start_time < duration:
            self.world.tick()
            
            # Update current speed
            velocity = self.vehicle.get_velocity()
            self.current_speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
            
            # Process sensor data
            if self.sensor_data['camera_image'] is not None:
                # Create a copy of the image for display
                display_image = self.sensor_data['camera_image'].copy()
                
                # Add ACC status overlay
                if self.acc_enabled:
                    status_text = f"ACC: ON | Speed: {self.current_speed:.1f} km/h | Distance: {self.leading_vehicle_distance:.1f} m"
                else:
                    status_text = f"ACC: OFF | Speed: {self.current_speed:.1f} km/h"
                
                cv2.putText(display_image, status_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display the modified image
                cv2.imshow('Camera Feed', display_image)
                cv2.waitKey(1)
            
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        self.toggle_acc()
                    elif event.key == pygame.K_q:
                        return
            
            # Update ACC control
            if self.acc_enabled:
                self._update_acc_control()
        
        cv2.destroyAllWindows()
        pygame.quit()
        
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
    """Main function to run the sensor fusion test"""
    parser = argparse.ArgumentParser(description='Run CARLA sensor fusion test')
    parser.add_argument('--no-camera', action='store_true', help='Disable camera sensor')
    parser.add_argument('--no-lidar', action='store_true', help='Disable LiDAR sensor')
    parser.add_argument('--no-radar', action='store_true', help='Disable radar sensor')
    parser.add_argument('--num-traffic', type=int, default=20, help='Number of traffic vehicles')
    parser.add_argument('--duration', type=int, default=60, help='Test duration in seconds')
    args = parser.parse_args()
    
    # Create sensor fusion test instance
    test = SensorFusionTest(
        enable_camera=not args.no_camera,
        enable_lidar=not args.no_lidar,
        enable_radar=not args.no_radar,
        num_traffic_vehicles=args.num_traffic
    )
    
    try:
        # Spawn traffic and ego vehicle
        test.spawn_traffic()
        test.spawn_ego_vehicle()
        
        # Run the test
        test.run_test(duration=args.duration)
        
    finally:
        test.cleanup()

if __name__ == '__main__':
    main() 