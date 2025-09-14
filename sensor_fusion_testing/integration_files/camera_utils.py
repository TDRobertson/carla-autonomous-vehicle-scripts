"""
Camera utilities for CARLA
Includes auto-follow camera functionality.
"""
import carla
import time
import threading
from typing import Optional

class AutoFollowCamera:
    """
    Automatically follows a vehicle with the spectator camera.
    """
    
    def __init__(self, world: carla.World, vehicle: carla.Vehicle, 
                 height: float = 50.0, distance: float = 20.0, 
                 follow_speed: float = 2.0):
        """
        Initialize auto-follow camera.
        
        Args:
            world: CARLA world instance
            vehicle: Vehicle to follow
            height: Height above vehicle
            distance: Distance behind vehicle
            follow_speed: Speed of camera movement (higher = smoother)
        """
        self.world = world
        self.vehicle = vehicle
        self.height = height
        self.distance = distance
        self.follow_speed = follow_speed
        self.is_following = False
        self.follow_thread = None
        self.spectator = world.get_spectator()
        
    def start_following(self):
        """Start auto-following the vehicle"""
        if self.is_following:
            return
            
        self.is_following = True
        self.follow_thread = threading.Thread(target=self._follow_loop, daemon=True)
        self.follow_thread.start()
        print("Auto-follow camera started")
        
    def stop_following(self):
        """Stop auto-following the vehicle"""
        self.is_following = False
        if self.follow_thread:
            self.follow_thread.join(timeout=1.0)
        print("Auto-follow camera stopped")
        
    def _follow_loop(self):
        """Main follow loop running in separate thread"""
        while self.is_following:
            try:
                if self.vehicle and self.vehicle.is_alive:
                    self._update_camera_position()
                time.sleep(1.0 / self.follow_speed)  # Update frequency
            except Exception as e:
                print(f"Camera follow error: {e}")
                break
                
    def _update_camera_position(self):
        """Update camera position to follow vehicle"""
        try:
            # Get vehicle transform
            vehicle_transform = self.vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            
            # Calculate camera position
            if self.distance == 0.0:
                # Overhead camera - directly above vehicle
                camera_location = carla.Location(
                    x=vehicle_location.x,
                    y=vehicle_location.y,
                    z=vehicle_location.z + self.height
                )
                # Look straight down
                camera_rotation = carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0)
            else:
                # Angled camera - behind and above vehicle
                forward_vector = vehicle_transform.get_forward_vector()
                
                camera_location = carla.Location(
                    x=vehicle_location.x - forward_vector.x * self.distance,
                    y=vehicle_location.y - forward_vector.y * self.distance,
                    z=vehicle_location.z + self.height
                )
                
                # Calculate camera rotation to look down at vehicle
                import math
                
                # Calculate the vector from camera to vehicle
                look_at_vector = vehicle_location - camera_location
                horizontal_distance = math.sqrt(look_at_vector.x**2 + look_at_vector.y**2)
                vertical_distance = look_at_vector.z
                
                # Calculate pitch (negative to look down)
                pitch = math.degrees(math.atan2(-vertical_distance, horizontal_distance))
                
                # Calculate yaw to face the vehicle
                yaw = math.degrees(math.atan2(look_at_vector.y, look_at_vector.x))
                
                camera_rotation = carla.Rotation(pitch=pitch, yaw=yaw, roll=0.0)
            
            # Set spectator transform
            camera_transform = carla.Transform(camera_location, camera_rotation)
            self.spectator.set_transform(camera_transform)
            
        except Exception as e:
            print(f"Camera update error: {e}")
            
    def set_follow_parameters(self, height: float = None, distance: float = None, 
                            follow_speed: float = None):
        """Update follow parameters"""
        if height is not None:
            self.height = height
        if distance is not None:
            self.distance = distance
        if follow_speed is not None:
            self.follow_speed = follow_speed
            
    def cleanup(self):
        """Cleanup camera resources"""
        self.stop_following()

def setup_auto_follow_camera(world: carla.World, vehicle: carla.Vehicle, 
                            height: float = 50.0, distance: float = 20.0) -> AutoFollowCamera:
    """
    Setup auto-follow camera for a vehicle.
    
    Args:
        world: CARLA world instance
        vehicle: Vehicle to follow
        height: Height above vehicle
        distance: Distance behind vehicle
        
    Returns:
        AutoFollowCamera instance
    """
    camera = AutoFollowCamera(world, vehicle, height, distance)
    camera.start_following()
    return camera

def setup_simple_overhead_camera(world: carla.World, vehicle: carla.Vehicle, 
                                height: float = 50.0) -> AutoFollowCamera:
    """
    Setup a simple overhead camera that looks straight down at the vehicle.
    
    Args:
        world: CARLA world instance
        vehicle: Vehicle to follow
        height: Height above vehicle
        
    Returns:
        AutoFollowCamera instance
    """
    camera = AutoFollowCamera(world, vehicle, height, 0.0)  # No horizontal distance
    camera.start_following()
    return camera
