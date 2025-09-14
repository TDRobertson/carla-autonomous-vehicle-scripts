import numpy as np
import time
import sys
import glob
import os
from .advanced_kalman_filter import AdvancedKalmanFilter
from .gps_spoofer import GPSSpoofer, SpoofingStrategy
from .position_display import PositionDisplay

# Add the CARLA Python API to PYTHONPATH
try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

class SensorFusion:
    def __init__(self, vehicle, enable_spoofing=False, spoofing_strategy=SpoofingStrategy.GRADUAL_DRIFT, enable_display=True):
        self.vehicle = vehicle
        self.kf = AdvancedKalmanFilter()
        self.enable_spoofing = enable_spoofing
        self.enable_display = enable_display
        
        # Initialize spoofer first
        self.spoofer = None
        if self.enable_spoofing:
            self.spoofer = GPSSpoofer([0, 0, 0], strategy=spoofing_strategy)
        
        # Initialize position display BEFORE sensors (so it's available in callbacks)
        if self.enable_display:
            self.position_display = PositionDisplay(vehicle.get_world(), vehicle, enable_console_output=False)
        else:
            self.position_display = None
        
        # Data storage
        self.gps_data = None
        self.imu_data = None
        self.fused_position = None
        self.true_position = None
        self.last_imu_timestamp = None
        
        # Innovation tracking
        self.current_innovation = 0.0  # Initialize to 0.0 instead of None
        self.innovation_history = []
        self.max_innovation_history = 100
        
        # Initialize sensors AFTER position display
        self.setup_sensors()
        
    def setup_sensors(self):
        # Setup GPS
        gps_bp = self.vehicle.get_world().get_blueprint_library().find('sensor.other.gnss')
        self.gps_sensor = self.vehicle.get_world().spawn_actor(
            gps_bp,
            carla.Transform(carla.Location(x=0.0, z=0.0)),
            attach_to=self.vehicle
        )
        self.gps_sensor.listen(self.gps_callback)
        
        # Setup IMU
        imu_bp = self.vehicle.get_world().get_blueprint_library().find('sensor.other.imu')
        self.imu_sensor = self.vehicle.get_world().spawn_actor(
            imu_bp,
            carla.Transform(carla.Location(x=0.0, z=0.0)),
            attach_to=self.vehicle
        )
        self.imu_sensor.listen(self.imu_callback)
        
    def gps_callback(self, data):
        # Store true position
        self.true_position = np.array([
            data.transform.location.x,
            data.transform.location.y,
            data.transform.location.z
        ])
        
        # Apply spoofing if enabled
        if self.enable_spoofing and self.spoofer is not None:
            # Pass current innovation to spoofer for innovation-aware attacks
            innovation = getattr(self, 'current_innovation', 0.0)
            self.gps_data = self.spoofer.spoof_position(self.true_position, innovation)
        else:
            self.gps_data = self.true_position
            
        # Update Kalman filter with GPS measurement
        if self.gps_data is not None:
            # Get innovation from Kalman filter update
            innovation = self.kf.update_with_gps(self.gps_data)
            self.current_innovation = innovation
            
            # Track innovation history - add defensive check
            if not hasattr(self, 'innovation_history'):
                self.innovation_history = []
            if not hasattr(self, 'max_innovation_history'):
                self.max_innovation_history = 100
                
            self.innovation_history.append(innovation)
            if len(self.innovation_history) > self.max_innovation_history:
                self.innovation_history.pop(0)
                
            self.fused_position = self.kf.position.copy()
            
            # Update position display
            if hasattr(self, 'position_display') and self.position_display is not None:
                attack_type = self.spoofer.strategy.name if self.spoofer else "None"
                self.position_display.update_positions(
                    self.true_position, 
                    self.fused_position, 
                    attack_type=attack_type,
                    innovation=innovation
                )
                # Calculate velocity error for display
                true_velocity = self.get_true_velocity()
                fused_velocity = self.get_fused_velocity()
                if true_velocity is not None and fused_velocity is not None:
                    velocity_error = np.linalg.norm(true_velocity - fused_velocity)
                    self.position_display.update_velocity_error(velocity_error)
                self.position_display.draw_position_info()
                self.position_display.draw_console_output()
        
    def imu_callback(self, data):
        # Use CARLA's simulation time for timestamp
        timestamp = self.vehicle.get_world().get_snapshot().timestamp.elapsed_seconds
        self.imu_data = data
        self.last_imu_timestamp = timestamp
        # Predict step with IMU data
        self.kf.predict(data, timestamp)
        self.fused_position = self.kf.position.copy()
        
    def get_fused_position(self):
        return self.fused_position
        
    def get_true_position(self):
        return self.true_position
        
    def get_true_velocity(self):
        """Get the true velocity of the vehicle."""
        if self.vehicle:
            velocity = self.vehicle.get_velocity()
            return np.array([velocity.x, velocity.y, velocity.z])
        return None
        
    def get_fused_velocity(self):
        """Get the fused velocity estimate."""
        if self.fused_position is not None and hasattr(self, '_last_fused_position'):
            dt = 0.1  # Assuming 10Hz update rate
            velocity = (self.fused_position - self._last_fused_position) / dt
            self._last_fused_position = self.fused_position.copy()
            return velocity
        elif self.fused_position is not None:
            self._last_fused_position = self.fused_position.copy()
        return None
        
    def get_imu_data(self):
        """Get the current IMU data."""
        return self.imu_data
        
    def get_kalman_metrics(self):
        """Get current Kalman filter metrics."""
        if self.kf:
            return {
                'covariance': self.kf.P,
                'position': self.kf.position,
                'velocity': self.kf.velocity,
                'orientation': self.kf.orientation
            }
        return None
        
    def get_innovation_stats(self):
        """Get innovation statistics for monitoring."""
        if self.spoofer is not None:
            return self.spoofer.get_innovation_stats()
        else:
            return {
                'current_innovation': self.current_innovation if self.current_innovation is not None else 0.0,
                'mean_innovation': np.mean(self.innovation_history) if self.innovation_history else 0.0,
                'max_innovation': np.max(self.innovation_history) if self.innovation_history else 0.0,
                'suspicious_counter': 0,
                'is_suspicious': False
            }
        
    def toggle_spoofing(self, enable=None):
        if enable is not None:
            self.enable_spoofing = enable
        else:
            self.enable_spoofing = not self.enable_spoofing
        
    def set_spoofing_strategy(self, strategy):
        if self.enable_spoofing:
            self.spoofer.set_strategy(strategy)
        
    def cleanup(self):
        if self.gps_sensor:
            self.gps_sensor.destroy()
        if self.imu_sensor:
            self.imu_sensor.destroy()
        if self.position_display is not None:
            self.position_display.cleanup()

def find_spawn_point(world):
    """
    Find a valid spawn point for the vehicle
    """
    spawn_points = world.get_map().get_spawn_points()
    if spawn_points:
        return spawn_points[0]  # Use the first spawn point
    return carla.Transform(carla.Location(x=0.0, y=0.0, z=1.0))  # Fallback spawn point

def setup_spectator(world, vehicle):
    """
    Setup the spectator camera to view the vehicle spawn point
    """
    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(
        transform.location + carla.Location(z=50),
        carla.Rotation(pitch=-90)
    ))

def main():
    # Connect to CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Get a valid spawn point
    spawn_point = find_spawn_point(world)
    
    # Spawn a vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    
    # Wait for the vehicle to spawn
    time.sleep(2.0)
    
    # Setup spectator camera once at the start
    setup_spectator(world, vehicle)
    
    # Enable autopilot
    vehicle.set_autopilot(True)
    
    # Initialize sensor fusion with spoofing enabled
    # Change the strategy here to test different methods:
    # SpoofingStrategy.GRADUAL_DRIFT --
    # SpoofingStrategy.SUDDEN_JUMP
    # SpoofingStrategy.RANDOM_WALK
    # SpoofingStrategy.REPLAY
    fusion = SensorFusion(vehicle, enable_spoofing=True, spoofing_strategy=SpoofingStrategy.REPLAY)
    
    try:
        while True:
            # Get the positions
            fused_pos = fusion.get_fused_position()
            true_pos = fusion.get_true_position()
            
            if fused_pos is not None and true_pos is not None:
                print(f"True Position: {true_pos}")
                print(f"Fused Position: {fused_pos.flatten()}")
                print(f"Position Error: {np.linalg.norm(fused_pos.flatten() - true_pos)}")
                print("-" * 50)
                
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("Cleaning up...")
        fusion.cleanup()
        vehicle.destroy()

if __name__ == '__main__':
    main() 