import numpy as np
import time
import sys
import glob
import os
import math
from typing import List, Optional, Tuple, Dict, Any

# Add the CARLA Python API to PYTHONPATH
try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from .sensor_fusion import SensorFusion
from .gps_spoofer import SpoofingStrategy

class PIDController:
    """PID controller for vehicle control"""
    def __init__(self, Kp: float, Ki: float, Kd: float, output_limits: Tuple[float, float] = (-1.0, 1.0)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.output_limits = output_limits
        self.integral = 0.0
        self.previous_error = 0.0
        self.last_time = None
        
    def compute(self, error: float, dt: float) -> float:
        """Compute PID output"""
        if dt <= 0:
            return 0.0
            
        # Integral term
        self.integral += error * dt
        
        # Derivative term
        derivative = (error - self.previous_error) / dt if dt > 0 else 0.0
        
        # PID output
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        # Apply output limits
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        
        # Update previous error
        self.previous_error = error
        
        return output
    
    def reset(self):
        """Reset PID controller state"""
        self.integral = 0.0
        self.previous_error = 0.0
        self.last_time = None

class WaypointNavigator:
    """
    Advanced waypoint navigation system using sensor fusion for position tracking.
    Integrates with CARLA's traffic manager and supports various navigation modes.
    """
    
    def __init__(self, 
                 vehicle: carla.Vehicle,
                 sensor_fusion: Optional[SensorFusion] = None,
                 enable_spoofing: bool = False,
                 spoofing_strategy: SpoofingStrategy = SpoofingStrategy.GRADUAL_DRIFT,
                 waypoint_reach_distance: float = 3.0,
                 waypoint_lookahead_distance: float = 10.0,
                 max_speed: float = 30.0,
                 pid_config: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Initialize the waypoint navigator.
        
        Args:
            vehicle: CARLA vehicle actor
            sensor_fusion: Optional SensorFusion instance (will create one if None)
            enable_spoofing: Whether to enable GPS spoofing for testing
            spoofing_strategy: Spoofing strategy to use
            waypoint_reach_distance: Distance threshold to consider waypoint reached
            waypoint_lookahead_distance: Distance to look ahead for next waypoint
            max_speed: Maximum vehicle speed in m/s
            pid_config: Custom PID parameters for throttle and steering
        """
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        
        # Navigation parameters
        self.waypoint_reach_distance = waypoint_reach_distance
        self.waypoint_lookahead_distance = waypoint_lookahead_distance
        self.max_speed = max_speed
        
        # Initialize sensor fusion if not provided
        if sensor_fusion is None:
            self.sensor_fusion = SensorFusion(
                vehicle, 
                enable_spoofing=enable_spoofing,
                spoofing_strategy=spoofing_strategy
            )
        else:
            self.sensor_fusion = sensor_fusion
        
        # Initialize PID controllers
        default_pid_config = {
            'throttle': {'Kp': 0.3, 'Ki': 0.01, 'Kd': 0.05},
            'steering': {'Kp': 0.8, 'Ki': 0.01, 'Kd': 0.1},
            'speed': {'Kp': 0.5, 'Ki': 0.02, 'Kd': 0.1}
        }
        
        if pid_config:
            default_pid_config.update(pid_config)
        
        self.throttle_pid = PIDController(**default_pid_config['throttle'], output_limits=(0.0, 1.0))
        self.steering_pid = PIDController(**default_pid_config['steering'], output_limits=(-1.0, 1.0))
        self.speed_pid = PIDController(**default_pid_config['speed'], output_limits=(0.0, self.max_speed))
        
        # Navigation state
        self.waypoints = []
        self.current_waypoint_index = 0
        self.is_navigating = False
        self.navigation_stats = {
            'start_time': None,
            'total_distance': 0.0,
            'waypoints_reached': 0,
            'average_speed': 0.0,
            'max_speed_reached': 0.0,
            'navigation_time': 0.0
        }
        
        # Performance tracking
        self.last_position = None
        self.last_time = None
        # Use true position for a brief warmup to allow Kalman filter to initialize
        self.position_warmup_duration = 3.0
        
    def set_waypoints(self, waypoints: List[carla.Waypoint]):
        """Set the waypoint route to follow"""
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        print(f"Set {len(waypoints)} waypoints for navigation")
        
    def add_waypoint(self, waypoint: carla.Waypoint):
        """Add a single waypoint to the route"""
        self.waypoints.append(waypoint)
        print(f"Added waypoint at {waypoint.transform.location}")
        
    def clear_waypoints(self):
        """Clear all waypoints"""
        self.waypoints = []
        self.current_waypoint_index = 0
        self.is_navigating = False
        print("Cleared all waypoints")
        
    def get_current_waypoint(self) -> Optional[carla.Waypoint]:
        """Get the current target waypoint"""
        if self.current_waypoint_index < len(self.waypoints):
            return self.waypoints[self.current_waypoint_index]
        return None
        
    def get_next_waypoint(self) -> Optional[carla.Waypoint]:
        """Get the next waypoint in the sequence"""
        if self.current_waypoint_index + 1 < len(self.waypoints):
            return self.waypoints[self.current_waypoint_index + 1]
        return None
        
    def get_position_estimate(self) -> Optional[np.ndarray]:
        """Get the current position estimate from sensor fusion"""
        if self.sensor_fusion:
            # During the first few seconds after navigation starts, use true position
            # to avoid initial bias while the Kalman filter converges
            try:
                start_time = self.navigation_stats.get('start_time')
                if start_time is not None and (time.time() - start_time) < self.position_warmup_duration:
                    position = self.sensor_fusion.get_true_position()
                else:
                    position = self.sensor_fusion.get_fused_position()
            except Exception:
                position = self.sensor_fusion.get_fused_position()

            # Fallback to vehicle transform if sensor fusion returns None
            if position is None:
                transform = self.vehicle.get_transform()
                return np.array([transform.location.x, transform.location.y, transform.location.z])
            return position
        else:
            # Fallback to vehicle transform
            transform = self.vehicle.get_transform()
            return np.array([transform.location.x, transform.location.y, transform.location.z])
            
    def get_true_position(self) -> Optional[np.ndarray]:
        """Get the true vehicle position"""
        if self.sensor_fusion:
            return self.sensor_fusion.get_true_position()
        else:
            # Fallback to vehicle transform
            transform = self.vehicle.get_transform()
            return np.array([transform.location.x, transform.location.y, transform.location.z])
            
    def calculate_distance_to_waypoint(self, waypoint: carla.Waypoint) -> float:
        """Calculate distance from current position to waypoint"""
        position = self.get_position_estimate()
        if position is None:
            return float('inf')
            
        waypoint_location = waypoint.transform.location
        return np.linalg.norm(position[:2] - np.array([waypoint_location.x, waypoint_location.y]))
        
    def calculate_heading_error(self, waypoint: carla.Waypoint) -> float:
        """Calculate heading error between vehicle and waypoint"""
        position = self.get_position_estimate()
        if position is None:
            return 0.0
            
        # Get vehicle heading
        vehicle_transform = self.vehicle.get_transform()
        vehicle_yaw = math.radians(vehicle_transform.rotation.yaw)
        
        # Calculate desired heading to waypoint
        waypoint_location = waypoint.transform.location
        dx = waypoint_location.x - position[0]
        dy = waypoint_location.y - position[1]
        desired_yaw = math.atan2(dy, dx)
        
        # Calculate heading error
        heading_error = desired_yaw - vehicle_yaw
        
        # Normalize to [-pi, pi]
        while heading_error > math.pi:
            heading_error -= 2 * math.pi
        while heading_error < -math.pi:
            heading_error += 2 * math.pi
            
        return heading_error
        
    def calculate_speed_error(self, target_speed: float) -> float:
        """Calculate speed error for speed control"""
        velocity = self.vehicle.get_velocity()
        current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return target_speed - current_speed
        
    def update_navigation_stats(self, dt: float):
        """Update navigation statistics"""
        current_time = time.time()
        
        if self.navigation_stats['start_time'] is None:
            self.navigation_stats['start_time'] = current_time
            
        # Update total distance
        current_position = self.get_position_estimate()
        if current_position is not None and self.last_position is not None:
            distance = np.linalg.norm(current_position - self.last_position)
            self.navigation_stats['total_distance'] += distance
            
        # Update speed statistics
        velocity = self.vehicle.get_velocity()
        current_speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        self.navigation_stats['max_speed_reached'] = max(self.navigation_stats['max_speed_reached'], current_speed)
        
        # Update average speed - fix division by zero
        elapsed_time = current_time - self.navigation_stats['start_time']
        if dt > 0 and elapsed_time > 0:
            self.navigation_stats['average_speed'] = (
                (self.navigation_stats['average_speed'] * (elapsed_time - dt) + 
                 current_speed * dt) / elapsed_time
            )
            
        self.last_position = current_position
        self.navigation_stats['navigation_time'] = elapsed_time
        
    def navigate_step(self, dt: float) -> Dict[str, Any]:
        """
        Perform one navigation step.
        
        Args:
            dt: Time delta since last step
            
        Returns:
            Dictionary containing navigation status and control outputs
        """
        if not self.waypoints or self.current_waypoint_index >= len(self.waypoints):
            return {
                'status': 'completed',
                'throttle': 0.0,
                'steering': 0.0,
                'current_waypoint': None,
                'distance_to_waypoint': 0.0,
                'heading_error': 0.0
            }
            
        current_waypoint = self.get_current_waypoint()
        if current_waypoint is None:
            return {
                'status': 'no_waypoint',
                'throttle': 0.0,
                'steering': 0.0,
                'current_waypoint': None,
                'distance_to_waypoint': 0.0,
                'heading_error': 0.0
            }
            
        # Calculate navigation metrics
        distance_to_waypoint = self.calculate_distance_to_waypoint(current_waypoint)
        heading_error = self.calculate_heading_error(current_waypoint)
        
        # Check if waypoint reached
        if distance_to_waypoint < self.waypoint_reach_distance:
            self.current_waypoint_index += 1
            self.navigation_stats['waypoints_reached'] += 1
            print(f"Reached waypoint {self.current_waypoint_index - 1}/{len(self.waypoints)}")
            
            if self.current_waypoint_index >= len(self.waypoints):
                return {
                    'status': 'completed',
                    'throttle': 0.0,
                    'steering': 0.0,
                    'current_waypoint': current_waypoint,
                    'distance_to_waypoint': distance_to_waypoint,
                    'heading_error': heading_error
                }
                
        # Calculate control outputs
        # Speed control based on distance to waypoint and road conditions
        target_speed = min(self.max_speed, distance_to_waypoint * 2.0)  # Slow down when approaching waypoint
        speed_error = self.calculate_speed_error(target_speed)
        throttle = self.throttle_pid.compute(speed_error, dt)
        
        # Steering control based on heading error
        steering = self.steering_pid.compute(heading_error, dt)
        
        # Apply controls
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=throttle,
            steer=steering,
            brake=0.0,
            hand_brake=False,
            reverse=False
        ))
        
        # Update statistics
        self.update_navigation_stats(dt)
        
        return {
            'status': 'navigating',
            'throttle': throttle,
            'steering': steering,
            'current_waypoint': current_waypoint,
            'distance_to_waypoint': distance_to_waypoint,
            'heading_error': heading_error,
            'target_speed': target_speed,
            'current_speed': math.sqrt(self.vehicle.get_velocity().x**2 + 
                                     self.vehicle.get_velocity().y**2 + 
                                     self.vehicle.get_velocity().z**2)
        }
        
    def start_navigation(self):
        """Start the navigation process"""
        if not self.waypoints:
            print("No waypoints set for navigation")
            return False
            
        self.is_navigating = True
        self.current_waypoint_index = 0
        self.navigation_stats = {
            'start_time': time.time(),
            'total_distance': 0.0,
            'waypoints_reached': 0,
            'average_speed': 0.0,
            'max_speed_reached': 0.0,
            'navigation_time': 0.0
        }
        print(f"Started navigation with {len(self.waypoints)} waypoints")
        return True
        
    def stop_navigation(self):
        """Stop the navigation process"""
        self.is_navigating = False
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=0.0,
            steer=0.0,
            brake=1.0,
            hand_brake=True,
            reverse=False
        ))
        print("Navigation stopped")
        
    def get_navigation_stats(self) -> Dict[str, Any]:
        """Get current navigation statistics"""
        stats = self.navigation_stats.copy()
        if stats['navigation_time'] > 0:
            stats['waypoints_per_minute'] = (stats['waypoints_reached'] / stats['navigation_time']) * 60
        else:
            stats['waypoints_per_minute'] = 0.0
        return stats
        
    def get_sensor_fusion_stats(self) -> Dict[str, Any]:
        """Get sensor fusion statistics"""
        if self.sensor_fusion:
            return {
                'fused_position': self.sensor_fusion.get_fused_position(),
                'true_position': self.sensor_fusion.get_true_position(),
                'innovation_stats': self.sensor_fusion.get_innovation_stats(),
                'kalman_metrics': self.sensor_fusion.get_kalman_metrics()
            }
        return {}
        
    def cleanup(self):
        """Cleanup resources"""
        self.stop_navigation()
        if self.sensor_fusion:
            self.sensor_fusion.cleanup() 