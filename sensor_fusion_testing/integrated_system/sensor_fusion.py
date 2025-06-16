import carla
import numpy as np
import time
from typing import Dict, Optional, Tuple
from kalman_filter import ExtendedKalmanFilter
from gps_spoofer import GPSSpoofer, SpoofingStrategy
from spoofing_detector import SpoofingDetector

class IntegratedSensorFusion:
    def __init__(self, vehicle, enable_spoofing=False, spoofing_strategy=SpoofingStrategy.GRADUAL_DRIFT):
        self.vehicle = vehicle
        self.ekf = ExtendedKalmanFilter()
        self.enable_spoofing = enable_spoofing
        self.spoofing_detector = SpoofingDetector()
        
        # Initialize spoofer if enabled
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
        self.last_update_time = time.time()
        
    def setup_sensors(self):
        """Setup GPS and IMU sensors."""
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
        """Handle GPS data updates."""
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
        """Handle IMU data updates."""
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
        current_time = time.time()
        dt = current_time - self.last_update_time
        self.last_update_time = current_time
        
        if not self.ekf.is_initialized():
            if self.gps_data is not None:
                self.ekf.initialize_with_gnss(self.gps_data)
            return
            
        # Predict step with IMU if available
        if self.imu_data is not None:
            self.ekf.predict_state_with_imu(self.imu_data)
            
        # Update step with GPS if available
        if self.gps_data is not None:
            # Check for spoofing
            detection_result = self.spoofing_detector.detect_spoofing(
                self.gps_data,
                self.get_velocity(),
                current_time
            )
            
            if detection_result.detected:
                # Apply correction based on detected spoofing type
                corrected_position = self.spoofing_detector.correct_position(
                    detection_result,
                    self.gps_data,
                    self.get_velocity()
                )
                self.ekf.correct_state_with_gnss(corrected_position)
            else:
                self.ekf.correct_state_with_gnss(self.gps_data)
                
        # Update fused position
        self.fused_position = np.array(self.ekf.get_location())
        
    def get_fused_position(self) -> Optional[np.ndarray]:
        """Get the current fused position estimate."""
        return self.fused_position
        
    def get_true_position(self) -> Optional[np.ndarray]:
        """Get the true vehicle position."""
        return self.true_position
        
    def get_velocity(self) -> Optional[np.ndarray]:
        """Get the current vehicle velocity."""
        if self.vehicle:
            velocity = self.vehicle.get_velocity()
            return np.array([velocity.x, velocity.y, velocity.z])
        return None
        
    def get_imu_data(self) -> Optional[Dict]:
        """Get the current IMU data."""
        return self.imu_data
        
    def get_kalman_metrics(self) -> Optional[Dict]:
        """Get current Kalman filter metrics."""
        if self.ekf:
            return {
                'covariance': self.ekf.p_cov,
                'state': {
                    'position': self.ekf.p,
                    'velocity': self.ekf.v,
                    'orientation': self.ekf.q
                }
            }
        return None
        
    def toggle_spoofing(self, enable: Optional[bool] = None):
        """Toggle GPS spoofing on/off."""
        if enable is not None:
            self.enable_spoofing = enable
        else:
            self.enable_spoofing = not self.enable_spoofing
            
    def set_spoofing_strategy(self, strategy: SpoofingStrategy):
        """Set the GPS spoofing strategy."""
        if self.enable_spoofing and self.spoofer is not None:
            self.spoofer.set_strategy(strategy)
            
    def cleanup(self):
        """Clean up sensors."""
        if self.gps_sensor:
            self.gps_sensor.destroy()
        if self.imu_sensor:
            self.imu_sensor.destroy() 