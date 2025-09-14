import numpy as np
import time
import sys
import glob
import os
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

class GPSOnlySystem:
    """
    GPS-only positioning system without IMU or Kalman filter correction.
    This allows testing the raw effects of GPS spoofing attacks.
    """
    def __init__(self, vehicle, enable_spoofing=False, spoofing_strategy=SpoofingStrategy.GRADUAL_DRIFT, enable_display=True):
        self.vehicle = vehicle
        self.enable_spoofing = enable_spoofing
        self.enable_display = enable_display
        
        # Initialize spoofer with aggressive mode for GPS-only systems
        self.spoofer = None
        if self.enable_spoofing:
            self.spoofer = GPSSpoofer([0, 0, 0], strategy=spoofing_strategy, aggressive_mode=True)
        
        # Initialize position display BEFORE sensors (so it's available in callbacks)
        if self.enable_display:
            self.position_display = PositionDisplay(vehicle.get_world(), vehicle, enable_console_output=False)
        else:
            self.position_display = None
        
        # Data storage
        self.gps_data = None
        self.true_position = None
        self.gps_position = None  # This will be the "fused" position in GPS-only mode
        
        # Simple velocity estimation from GPS position differences
        self.last_gps_position = None
        self.last_gps_timestamp = None
        self.gps_velocity = None
        
        # Statistics tracking
        self.position_errors = []
        self.velocity_errors = []
        self.timestamps = []
        
        # Initialize GPS sensor AFTER position display
        self.setup_gps_sensor()
        
    def setup_gps_sensor(self):
        """Setup only the GPS sensor"""
        gps_bp = self.vehicle.get_world().get_blueprint_library().find('sensor.other.gnss')
        self.gps_sensor = self.vehicle.get_world().spawn_actor(
            gps_bp,
            carla.Transform(carla.Location(x=0.0, z=0.0)),
            attach_to=self.vehicle
        )
        self.gps_sensor.listen(self.gps_callback)
        
    def gps_callback(self, data):
        """GPS callback - this is the only sensor data we use"""
        # Store true position
        self.true_position = np.array([
            data.transform.location.x,
            data.transform.location.y,
            data.transform.location.z
        ])
        
        # Apply spoofing if enabled
        if self.enable_spoofing and self.spoofer is not None:
            # For GPS-only system, we don't have innovation tracking
            # so we pass 0.0 as innovation
            self.gps_data = self.spoofer.spoof_position(self.true_position, 0.0)
        else:
            self.gps_data = self.true_position
            
        # In GPS-only mode, the "fused" position is just the GPS position
        self.gps_position = self.gps_data.copy() if self.gps_data is not None else None
        
        # Calculate velocity from GPS position differences
        current_timestamp = time.time()
        if self.last_gps_position is not None and self.last_gps_timestamp is not None:
            dt = current_timestamp - self.last_gps_timestamp
            if dt > 0:
                self.gps_velocity = (self.gps_position - self.last_gps_position) / dt
            else:
                self.gps_velocity = np.zeros(3)
        else:
            self.gps_velocity = np.zeros(3)
            
        self.last_gps_position = self.gps_position.copy() if self.gps_position is not None else None
        self.last_gps_timestamp = current_timestamp
        
        # Calculate and store errors for analysis
        if self.true_position is not None and self.gps_position is not None:
            position_error = np.linalg.norm(self.true_position - self.gps_position)
            self.position_errors.append(position_error)
            
            # Calculate velocity error
            true_velocity = self.get_true_velocity()
            if true_velocity is not None and self.gps_velocity is not None:
                velocity_error = np.linalg.norm(true_velocity - self.gps_velocity)
                self.velocity_errors.append(velocity_error)
            else:
                self.velocity_errors.append(0.0)
                
            self.timestamps.append(current_timestamp)
            
            # Update position display
            if hasattr(self, 'position_display') and self.position_display is not None:
                attack_type = self.spoofer.strategy.name if self.spoofer else "None"
                self.position_display.update_positions(
                    self.true_position, 
                    self.gps_position, 
                    attack_type=attack_type,
                    innovation=0.0  # GPS-only doesn't have innovation
                )
                self.position_display.update_velocity_error(velocity_error)
                self.position_display.draw_position_info()
                self.position_display.draw_console_output()
            
            # Keep only recent history to prevent memory issues
            max_history = 1000
            if len(self.position_errors) > max_history:
                self.position_errors.pop(0)
                self.velocity_errors.pop(0)
                self.timestamps.pop(0)
        
    def get_fused_position(self):
        """Get the GPS position (no fusion in GPS-only mode)"""
        return self.gps_position
        
    def get_true_position(self):
        """Get the true vehicle position"""
        return self.true_position
        
    def get_true_velocity(self):
        """Get the true velocity of the vehicle"""
        if self.vehicle:
            velocity = self.vehicle.get_velocity()
            return np.array([velocity.x, velocity.y, velocity.z])
        return None
        
    def get_fused_velocity(self):
        """Get the GPS-derived velocity estimate"""
        return self.gps_velocity
        
    def get_gps_data(self):
        """Get the current GPS data"""
        return self.gps_data
        
    def get_position_error(self):
        """Get current position error"""
        if self.true_position is not None and self.gps_position is not None:
            return np.linalg.norm(self.true_position - self.gps_position)
        return 0.0
        
    def get_velocity_error(self):
        """Get current velocity error"""
        true_velocity = self.get_true_velocity()
        if true_velocity is not None and self.gps_velocity is not None:
            return np.linalg.norm(true_velocity - self.gps_velocity)
        return 0.0
        
    def get_error_statistics(self):
        """Get error statistics for analysis"""
        if not self.position_errors:
            return {
                'mean_position_error': 0.0,
                'max_position_error': 0.0,
                'std_position_error': 0.0,
                'mean_velocity_error': 0.0,
                'max_velocity_error': 0.0,
                'std_velocity_error': 0.0,
                'num_samples': 0
            }
            
        return {
            'mean_position_error': np.mean(self.position_errors),
            'max_position_error': np.max(self.position_errors),
            'std_position_error': np.std(self.position_errors),
            'mean_velocity_error': np.mean(self.velocity_errors),
            'max_velocity_error': np.max(self.velocity_errors),
            'std_velocity_error': np.std(self.velocity_errors),
            'num_samples': len(self.position_errors)
        }
        
    def get_innovation_stats(self):
        """Get innovation statistics (not applicable for GPS-only)"""
        return {
            'current_innovation': 0.0,
            'mean_innovation': 0.0,
            'max_innovation': 0.0,
            'suspicious_counter': 0,
            'is_suspicious': False
        }
        
    def get_kalman_metrics(self):
        """Get Kalman filter metrics (not applicable for GPS-only)"""
        return None
        
    def toggle_spoofing(self, enable=None):
        """Toggle GPS spoofing on/off"""
        if enable is not None:
            self.enable_spoofing = enable
        else:
            self.enable_spoofing = not self.enable_spoofing
        
    def set_spoofing_strategy(self, strategy):
        """Set the GPS spoofing strategy"""
        if self.enable_spoofing and self.spoofer is not None:
            self.spoofer.set_strategy(strategy)
        
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'gps_sensor') and self.gps_sensor:
            self.gps_sensor.destroy()
        if self.position_display is not None:
            self.position_display.cleanup()

def find_spawn_point(world):
    """Find a valid spawn point for the vehicle"""
    spawn_points = world.get_map().get_spawn_points()
    if spawn_points:
        return spawn_points[0]  # Use the first spawn point
    return carla.Transform(carla.Location(x=0.0, y=0.0, z=1.0))  # Fallback spawn point

def setup_spectator(world, vehicle):
    """Setup the spectator camera to view the vehicle spawn point"""
    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(
        transform.location + carla.Location(z=50),
        carla.Rotation(pitch=-90)
    ))

def main():
    """Main function for testing GPS-only system"""
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
    
    # Initialize GPS-only system with spoofing enabled
    # Change the strategy here to test different methods:
    # SpoofingStrategy.GRADUAL_DRIFT
    # SpoofingStrategy.SUDDEN_JUMP
    # SpoofingStrategy.RANDOM_WALK
    # SpoofingStrategy.REPLAY
    gps_system = GPSOnlySystem(vehicle, enable_spoofing=True, spoofing_strategy=SpoofingStrategy.GRADUAL_DRIFT)
    
    try:
        print("GPS-Only System Test")
        print("===================")
        print("Testing GPS positioning without IMU or Kalman filter correction")
        print("This will show the raw effects of GPS spoofing attacks")
        print()
        
        while True:
            # Get the positions
            gps_pos = gps_system.get_fused_position()
            true_pos = gps_system.get_true_position()
            
            if gps_pos is not None and true_pos is not None:
                position_error = gps_system.get_position_error()
                velocity_error = gps_system.get_velocity_error()
                
                print(f"True Position: [{true_pos[0]:.2f}, {true_pos[1]:.2f}, {true_pos[2]:.2f}]")
                print(f"GPS Position:  [{gps_pos[0]:.2f}, {gps_pos[1]:.2f}, {gps_pos[2]:.2f}]")
                print(f"Position Error: {position_error:.3f}m")
                print(f"Velocity Error: {velocity_error:.3f}m/s")
                print("-" * 50)
                
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nCleaning up...")
        stats = gps_system.get_error_statistics()
        print("\nFinal Error Statistics:")
        print(f"Mean Position Error: {stats['mean_position_error']:.3f}m")
        print(f"Max Position Error: {stats['max_position_error']:.3f}m")
        print(f"Mean Velocity Error: {stats['mean_velocity_error']:.3f}m/s")
        print(f"Max Velocity Error: {stats['max_velocity_error']:.3f}m/s")
        print(f"Total Samples: {stats['num_samples']}")
        
        gps_system.cleanup()
        vehicle.destroy()

if __name__ == '__main__':
    main()
