import carla
import numpy as np
import time
from kalman_filter import KalmanFilter
from gps_spoofer import GPSSpoofer, SpoofingStrategy

class SensorFusion:
    def __init__(self, vehicle, enable_spoofing=False, spoofing_strategy=SpoofingStrategy.GRADUAL_DRIFT):
        self.vehicle = vehicle
        self.kf = KalmanFilter()
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
            
        self.update()
        
    def imu_callback(self, data):
        self.imu_data = {
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
        }
        
    def update(self):
        """Update the sensor fusion state."""
        if self.gps_data is not None:
            # Predict step
            predicted_position = self.kf.predict()
            
            # Update step with GPS measurement
            self.fused_position = self.kf.update(self.gps_data)
            
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
            metrics = {
                'covariance': self.kf.P,
                'kalman_gain': self.kf.K if self.kf.K is not None else np.zeros((6, 3))
            }
            if self.kf.y is not None:
                metrics['innovation'] = self.kf.y
            return metrics
        return None
        
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