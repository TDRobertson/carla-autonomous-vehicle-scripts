import numpy as np
import time
import sys
import glob
import os
from advanced_kalman_filter import AdvancedKalmanFilter
from gps_spoofer import GPSSpoofer, SpoofingStrategy

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
    def __init__(self, vehicle, enable_spoofing=False, spoofing_strategy=SpoofingStrategy.GRADUAL_DRIFT):
        self.vehicle = vehicle
        self.kf = AdvancedKalmanFilter()
        self.enable_spoofing = enable_spoofing
        
        # Initialize spoofer first
        self.spoofer = None
        if self.enable_spoofing:
            self.spoofer = GPSSpoofer([0, 0, 0], strategy=spoofing_strategy)
        
        # Initialize sensors
        self.setup_sensors()
        
        # Data storage
        self.gps_data = None
        self.imu_data = None
        self.fused_position = None
        self.true_position = None
        self.last_imu_timestamp = None
        self.imu_predicted_position = None  # Track IMU prediction for bias detection
        self.gps_rejected_count = 0  # Track how many times GPS was rejected
        self.gps_accepted_count = 0  # Track how many times GPS was accepted
        
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
            self.gps_data = self.spoofer.spoof_position(self.true_position)
        else:
            self.gps_data = self.true_position
        # Update Kalman filter with GPS measurement and IMU prediction for bias detection
        if self.gps_data is not None:
            # Safety check: ensure imu_predicted_position exists
            if self.imu_predicted_position is None:
                # If no IMU prediction yet, use current GPS as fallback
                self.imu_predicted_position = self.gps_data.copy()
            
            gps_accepted = self.kf.update_with_gps(self.gps_data, self.imu_predicted_position)
            if gps_accepted:
                self.gps_accepted_count += 1
            else:
                self.gps_rejected_count += 1
            self.fused_position = self.kf.position.copy()
            
            # Update spoofer with current innovation for adaptive attacks
            if self.enable_spoofing and self.spoofer is not None:
                innovation_stats = self.kf.get_innovation_stats()
                if innovation_stats:
                    self.spoofer.update_innovation(innovation_stats['current_innovation'])
        
    def imu_callback(self, data):
        # Use CARLA's simulation time for timestamp
        timestamp = self.vehicle.get_world().get_snapshot().timestamp.elapsed_seconds
        self.imu_data = data
        self.last_imu_timestamp = timestamp
        # Predict step with IMU data
        self.kf.predict(data, timestamp)
        # Store IMU predicted position for bias detection
        self.imu_predicted_position = self.kf.position.copy()
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
        """Get innovation statistics for spoofing detection monitoring."""
        return self.kf.get_innovation_stats()
        
    def get_bias_stats(self):
        """Get bias statistics for constant bias detection monitoring."""
        return self.kf.get_bias_stats()
        
    def get_gps_stats(self):
        """Get GPS acceptance/rejection statistics."""
        total_gps = self.gps_accepted_count + self.gps_rejected_count
        if total_gps == 0:
            return {
                'accepted_count': 0,
                'rejected_count': 0,
                'acceptance_rate': 0.0
            }
        
        return {
            'accepted_count': self.gps_accepted_count,
            'rejected_count': self.gps_rejected_count,
            'acceptance_rate': self.gps_accepted_count / total_gps
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
    
    # Initialize sensor fusion with spoofing enabled and innovation-based mitigation
    # Change the strategy here to test different methods:
    # SpoofingStrategy.GRADUAL_DRIFT -- More subtle with random fluctuations
    # SpoofingStrategy.SUDDEN_JUMP -- Innovation-aware jumping
    # SpoofingStrategy.RANDOM_WALK -- Directional random walk
    # SpoofingStrategy.REPLAY -- Sophisticated replay with noise
    fusion = SensorFusion(vehicle, enable_spoofing=True, spoofing_strategy=SpoofingStrategy.SUDDEN_JUMP)
    
    print("=== Innovation-Based GPS Spoofing Mitigation System ===")
    print("Monitoring innovation values and GPS acceptance/rejection...")
    print("Innovation threshold: 5.0 meters")
    print("Suspicious GPS count threshold: 3")
    print("=" * 60)
    
    try:
        while True:
            # Get the positions
            fused_pos = fusion.get_fused_position()
            true_pos = fusion.get_true_position()
            
            if fused_pos is not None and true_pos is not None:
                # Calculate position error
                position_error = np.linalg.norm(fused_pos - true_pos)
                
                # Get monitoring statistics
                innovation_stats = fusion.get_innovation_stats()
                bias_stats = fusion.get_bias_stats()
                gps_stats = fusion.get_gps_stats()
                
                # Display basic position information
                print(f"\nTrue Position: {true_pos}")
                print(f"Fused Position: {fused_pos}")
                print(f"Position Error: {position_error:.3f}m")
                
                # Display innovation monitoring
                if innovation_stats:
                    print(f"Current Innovation: {innovation_stats['current_innovation']:.3f}m")
                    print(f"Mean Innovation: {innovation_stats['mean_innovation']:.3f}m")
                    print(f"Max Innovation: {innovation_stats['max_innovation']:.3f}m")
                    print(f"Suspicious GPS Count: {innovation_stats['suspicious_count']}")
                
                # Display bias monitoring
                if bias_stats:
                    print(f"GPS-IMU Bias: {bias_stats['current_bias']:.3f}m")
                    print(f"Mean Bias: {bias_stats['mean_bias']:.3f}m")
                    print(f"Bias Std: {bias_stats['bias_std']:.3f}m")
                
                # Display GPS acceptance statistics
                if gps_stats:
                    print(f"GPS Accepted: {gps_stats['accepted_count']}")
                    print(f"GPS Rejected: {gps_stats['rejected_count']}")
                    print(f"GPS Acceptance Rate: {gps_stats['acceptance_rate']:.2%}")
                
                print("-" * 60)
                
            time.sleep(1.0)  # Update every second for readability
            
    except KeyboardInterrupt:
        print("\nCleaning up...")
        fusion.cleanup()
        vehicle.destroy()

if __name__ == '__main__':
    main() 