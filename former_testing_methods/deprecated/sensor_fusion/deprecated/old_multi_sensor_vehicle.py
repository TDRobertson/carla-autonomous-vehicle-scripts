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
        self.warning_active = False  # New flag for radar warning
        self.last_detected_distance = None  # Store the last detected distance
        self.last_detected_world_location = None  # Store the last detected object's world location
        
        # Get the spectator
        self.spectator = self.world.get_spectator()
        
        # Manual control state
        pass  # Manual mode now handled in main loop
        
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
        print(f"Camera callback: warning_active={self.warning_active}, last_detected_distance={self.last_detected_distance}")
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        
        # Add warning overlay if radar detected something
        if self.warning_active and self.last_detected_distance is not None:
            # Create a copy of the image to avoid modifying the original
            array = array.copy()
            
            # Add warning text
            warning_text = f"WARNING: Object detected at {self.last_detected_distance:.1f}m"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_thickness = 2
            text_color = (0, 0, 255)  # Red color in BGR
            
            # Get text size
            text_size = cv2.getTextSize(warning_text, font, font_scale, font_thickness)[0]
            
            # Calculate text position (centered horizontally, near top vertically)
            text_x = (array.shape[1] - text_size[0]) // 2
            text_y = 50  # 50 pixels from top
            
            # Add semi-transparent red rectangle behind text
            padding = 10
            overlay = array.copy()
            cv2.rectangle(overlay, 
                         (text_x - padding, text_y - text_size[1] - padding),
                         (text_x + text_size[0] + padding, text_y + padding),
                         (0, 0, 255),
                         -1)
            # Apply transparency
            alpha = 0.3
            array = cv2.addWeighted(overlay, alpha, array, 1 - alpha, 0)
            
            # Add text
            cv2.putText(array, warning_text, (text_x, text_y), font, font_scale, text_color, font_thickness)

            # Draw a line from camera center to detected object if available
            if self.last_detected_world_location is not None:
                # Camera intrinsics
                w = image.width
                h = image.height
                fov = float(self.camera.attributes.get('fov', 90))
                K = self._build_projection_matrix(w, h, fov)
                # Get camera world transform
                camera_transform = self.camera.get_transform()
                # Project camera origin and detected point
                points3d = [camera_transform.location, self.last_detected_world_location]
                points_2d = self._get_screen_points(self.camera, K, w, h, points3d)
                # Draw line if both points are in front of camera and within image bounds
                p0, p1 = points_2d[0], points_2d[1]
                if all(0 <= p < w for p in [p0[0], p1[0]]) and all(0 <= p < h for p in [p0[1], p1[1]]):
                    cv2.line(array, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), (0,255,255), 3)
        
        self.sensor_data['camera_image'] = array
        
    def _lidar_callback(self, point_cloud):
        """Process LiDAR data"""
        data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))
        self.sensor_data['lidar_points'] = data
        
    def _radar_callback(self, radar_data):
        """Process radar data and adjust vehicle behavior"""
        print(f"Radar callback: {len(radar_data)} detections")
        # Get current vehicle velocity
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
        
        # Reset warning flag
        self.warning_active = False
        self.last_detected_distance = None
        self.last_detected_world_location = None
        
        # Process radar detections
        min_distance = float('inf')
        closest_location = None
        for detection in radar_data:
            print(f"Detection: depth={detection.depth}, azimuth={detection.azimuth}, altitude={detection.altitude}")
            # Calculate distance to detected object
            distance = detection.depth
            # Calculate world location of detection
            radar_transform = self.radar.get_transform()
            rel_x = distance * math.cos(detection.azimuth) * math.cos(detection.altitude)
            rel_y = distance * math.sin(detection.azimuth) * math.cos(detection.altitude)
            rel_z = distance * math.sin(detection.altitude)
            detection_location = radar_transform.transform(carla.Location(x=rel_x, y=rel_y, z=rel_z))
            # Update warning status for any object within 10 meters
            if distance < 10.0 and distance < min_distance:
                self.warning_active = True
                self.last_detected_distance = distance
                min_distance = distance
                closest_location = detection_location
            # If object is too close and we're moving
            if distance < 5.0 and speed > 5.0:  # 5 meters and 5 km/h thresholds
                self._handle_close_object(distance)
                break
        if self.warning_active and closest_location is not None:
            self.last_detected_world_location = closest_location
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

    def _build_projection_matrix(self, w, h, fov):
        focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
        K = np.identity(3)
        K[0, 0] = K[1, 1] = focal
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0
        return K

    def _get_screen_points(self, camera, K, image_w, image_h, points3d):
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
        points_temp = []
        for p in points3d:
            points_temp += [p.x, p.y, p.z, 1]
        points = np.array(points_temp).reshape(-1, 4).T
        points_camera = np.dot(world_2_camera, points)
        points = np.array([
            points_camera[1],
            points_camera[2] * -1,
            points_camera[0]])
        points_2d = np.dot(K, points)
        points_2d = np.array([
            points_2d[0, :] / points_2d[2, :],
            points_2d[1, :] / points_2d[2, :],
            points_2d[2, :]]).T
        return points_2d

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
    
    pygame.init()
    screen = pygame.display.set_mode((400, 100))
    pygame.display.set_caption('CARLA Manual Mode Toggle (press M)')

    manual_mode = False

    try:
        # Spawn the ego vehicle
        vehicle = sensor_fusion.spawn_ego_vehicle()
        print("Ego vehicle spawned successfully!")
        
        # Spawn traffic
        sensor_fusion.spawn_traffic()
        print("Traffic spawned successfully!")
        print("Press Ctrl+C or close the window to exit...")
        
        # Main simulation loop
        while True:
            # Listen for manual mode toggle and WASD control
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_m:
                        manual_mode = not manual_mode
                        if manual_mode:
                            print("Manual mode ON. Use WASD to control the vehicle. Press 'm' to toggle manual mode off.")
                            sensor_fusion.vehicle.set_autopilot(False)
                        else:
                            print("Manual mode OFF. Returning to autopilot.")
                            sensor_fusion.vehicle.set_autopilot(True, sensor_fusion.traffic_manager.get_port())
                if event.type == pygame.QUIT:
                    raise KeyboardInterrupt

            if manual_mode:
                keys = pygame.key.get_pressed()
                control = carla.VehicleControl()
                if keys[pygame.K_w]:
                    control.throttle = 1.0
                if keys[pygame.K_s]:
                    control.brake = 1.0
                if keys[pygame.K_a]:
                    control.steer = -1.0
                if keys[pygame.K_d]:
                    control.steer = 1.0
                sensor_fusion.vehicle.apply_control(control)

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