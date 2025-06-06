import carla
import numpy as np
import time
from typing import Optional, Dict, List, Tuple
from collections import deque
from dataclasses import dataclass
from kalman_filter import KalmanFilter
from gps_spoofer import GPSSpoofer, SpoofingStrategy
import warnings

@dataclass
class SensorData:
    timestamp: float
    data: np.ndarray
    type: str  # 'gps' or 'imu'

class SensorFusion:
    def __init__(self, vehicle, enable_spoofing=False, spoofing_strategy=SpoofingStrategy.GRADUAL_DRIFT):
        self.vehicle = vehicle
        
        # Initialize data buffers first
        self.gps_buffer = deque(maxlen=100)  # Store last 100 GPS measurements
        self.imu_buffer = deque(maxlen=100)  # Store last 100 IMU measurements
        
        # Initialize state variables
        self.fused_position = None
        self.fused_velocity = None
        self.true_position = None
        self.true_velocity = None
        
        # Initialize Kalman filter
        self.kf = KalmanFilter()
        
        # Initialize spoofer
        self.spoofer = None
        if enable_spoofing:
            self.spoofer = GPSSpoofer([0, 0, 0], strategy=spoofing_strategy)
        self.enable_spoofing = enable_spoofing
        
        # Synchronization parameters
        self.sync_window = 0.1  # 100ms window for synchronization
        self.last_sync_time = None
        
        # Traffic state detection
        self.velocity_threshold = 0.5  # m/s
        self.stop_duration_threshold = 1.0  # seconds
        self.last_stop_time = None
        self.is_stopped = False
        
        # Initialize sensors last
        self.setup_sensors()
        
        # Suppress numpy warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')
        
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
        try:
            # Store true position
            self.true_position = np.array([
                data.transform.location.x,
                data.transform.location.y,
                data.transform.location.z
            ])
            
            # Apply spoofing if enabled
            if self.enable_spoofing and self.spoofer is not None:
                gps_data = self.spoofer.spoof_position(self.true_position)
            else:
                gps_data = self.true_position
                
            # Store GPS data with timestamp
            self.gps_buffer.append(SensorData(
                timestamp=time.time(),
                data=gps_data,
                type='gps'
            ))
            
            self.update()
        except Exception as e:
            print(f"Error in GPS callback: {str(e)}")
        
    def imu_callback(self, data):
        try:
            # Store IMU data with timestamp
            self.imu_buffer.append(SensorData(
                timestamp=time.time(),
                data={
                    'acceleration': np.array([
                        data.accelerometer.x,
                        data.accelerometer.y,
                        data.accelerometer.z
                    ]),
                    'gyroscope': np.array([
                        data.gyroscope.x,
                        data.gyroscope.y,
                        data.gyroscope.z
                    ])
                },
                type='imu'
            ))
        except Exception as e:
            print(f"Error in IMU callback: {str(e)}")
        
    def _detect_traffic_state(self) -> bool:
        """Detect if the vehicle is stopped at a traffic light."""
        if self.true_velocity is None:
            return False
            
        velocity_magnitude = np.linalg.norm(self.true_velocity)
        current_time = time.time()
        
        if velocity_magnitude < self.velocity_threshold:
            if not self.is_stopped:
                self.last_stop_time = current_time
                self.is_stopped = True
            elif current_time - self.last_stop_time > self.stop_duration_threshold:
                return True
        else:
            self.is_stopped = False
            self.last_stop_time = None
            
        return False
        
    def _synchronize_data(self) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Synchronize GPS and IMU data within the time window."""
        if not self.gps_buffer or not self.imu_buffer:
            return None, None
            
        current_time = time.time()
        
        # Find the most recent GPS measurement
        gps_data = self.gps_buffer[-1]
        
        # Find the closest IMU measurement within the sync window
        closest_imu = None
        min_time_diff = float('inf')
        
        for imu_data in reversed(self.imu_buffer):
            time_diff = abs(imu_data.timestamp - gps_data.timestamp)
            if time_diff < self.sync_window and time_diff < min_time_diff:
                closest_imu = imu_data
                min_time_diff = time_diff
                
        if closest_imu is None:
            return gps_data.data, None
            
        return gps_data.data, closest_imu.data
        
    def update(self):
        """Update the sensor fusion state with synchronized data."""
        try:
            # Synchronize data
            gps_data, imu_data = self._synchronize_data()
            if gps_data is None:
                return
                
            # Detect traffic state
            is_at_traffic_light = self._detect_traffic_state()
            
            # Get current time
            current_time = time.time()
            
            # Predict step
            predicted_position, predicted_velocity = self.kf.predict(current_time)
            
            # Update step with GPS measurement
            self.fused_position, self.fused_velocity = self.kf.update(gps_data, current_time)
            
            # Get true velocity
            if self.vehicle:
                velocity = self.vehicle.get_velocity()
                self.true_velocity = np.array([velocity.x, velocity.y, velocity.z])
        except Exception as e:
            print(f"Error in update: {str(e)}")
            
    def get_fused_position(self):
        return self.fused_position
        
    def get_fused_velocity(self):
        return self.fused_velocity
        
    def get_true_position(self):
        return self.true_position
        
    def get_true_velocity(self):
        return self.true_velocity
        
    def get_imu_data(self):
        """Get the current IMU data."""
        if self.imu_buffer:
            return self.imu_buffer[-1].data
        return None
        
    def get_kalman_metrics(self):
        """Get current Kalman filter metrics."""
        return {
            'covariance': self.kf.get_covariance(),
            'kalman_gain': self.kf.get_kalman_gain(),
            'innovation': self.kf.get_innovation()
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