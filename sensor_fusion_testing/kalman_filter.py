import numpy as np

class KalmanFilter:
    def __init__(self):
        # State vector [x, y, z, vx, vy, vz]
        self.x = np.zeros(6)
        
        # State transition matrix
        self.F = np.eye(6)
        self.F[0:3, 3:6] = np.eye(3) * 0.1  # dt = 0.1s
        
        # Measurement matrices for both GPS and IMU
        self.H_gps = np.zeros((3, 6))
        self.H_gps[0:3, 0:3] = np.eye(3)
        
        self.H_imu = np.zeros((6, 6))
        self.H_imu[0:3, 0:3] = np.eye(3)  # Position
        self.H_imu[3:6, 3:6] = np.eye(3)  # Velocity
        
        # Process noise covariance
        self.Q = np.eye(6) * 0.1
        
        # Measurement noise covariance for GPS and IMU
        self.R_gps = np.eye(3) * 1.0  # GPS is less reliable
        self.R_imu = np.eye(6) * 0.1  # IMU is more reliable short-term
        
        # Error covariance matrix
        self.P = np.eye(6) * 1000
        
        # Innovation
        self.y = None
        
        # Kalman gain
        self.K = None
        
        # IMU integration
        self.last_imu_time = None
        self.imu_position = np.zeros(3)
        self.imu_velocity = np.zeros(3)
        
    def predict(self):
        """Predict step of the Kalman filter."""
        # Predict state
        self.x = self.F @ self.x
        
        # Predict error covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x[0:3]  # Return predicted position
        
    def update_with_imu(self, imu_data, timestamp):
        """Update state using IMU data."""
        if self.last_imu_time is None:
            self.last_imu_time = timestamp
            return self.x[0:3]
            
        dt = timestamp - self.last_imu_time
        self.last_imu_time = timestamp
        
        # Integrate IMU data
        accel = imu_data['acceleration']
        gyro = imu_data['gyroscope']
        
        # Update velocity using acceleration
        self.imu_velocity += accel * dt
        
        # Update position using velocity
        self.imu_position += self.imu_velocity * dt
        
        # Create measurement vector
        z = np.concatenate([self.imu_position, self.imu_velocity])
        
        # Calculate innovation
        self.y = z - self.H_imu @ self.x
        
        # Calculate innovation covariance
        S = self.H_imu @ self.P @ self.H_imu.T + self.R_imu
        
        # Calculate Kalman gain
        self.K = self.P @ self.H_imu.T @ np.linalg.inv(S)
        
        # Update state estimate
        self.x = self.x + self.K @ self.y
        
        # Update error covariance
        self.P = (np.eye(6) - self.K @ self.H_imu) @ self.P
        
        return self.x[0:3]
        
    def update_with_gps(self, gps_position):
        """Update state using GPS data."""
        # Calculate innovation
        self.y = gps_position - self.H_gps @ self.x
        
        # Calculate innovation covariance
        S = self.H_gps @ self.P @ self.H_gps.T + self.R_gps
        
        # Calculate Kalman gain
        self.K = self.P @ self.H_gps.T @ np.linalg.inv(S)
        
        # Update state estimate
        self.x = self.x + self.K @ self.y
        
        # Update error covariance
        self.P = (np.eye(6) - self.K @ self.H_gps) @ self.P
        
        return self.x[0:3]
        
    def get_reliability_metrics(self):
        """Get reliability metrics for GPS and IMU."""
        gps_reliability = 1.0 / np.trace(self.R_gps)
        imu_reliability = 1.0 / np.trace(self.R_imu)
        
        return {
            'gps_reliability': gps_reliability,
            'imu_reliability': imu_reliability,
            'innovation': self.y if self.y is not None else np.zeros(3),
            'kalman_gain': self.K if self.K is not None else np.zeros((6, 3))
        } 