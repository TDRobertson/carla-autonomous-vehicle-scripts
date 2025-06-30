import numpy as np
from scipy.spatial.transform import Rotation

class AdvancedKalmanFilter:
    def __init__(self):
        # State vector: [position, velocity, orientation]
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.array([0, 0, 0, 1])  # quaternion [x,y,z,w]
        
        # Covariance matrix
        self.P = np.eye(9) * 0.1
        
        # Process noise
        self.Q = np.eye(9) * 0.01
        
        # Measurement noise
        self.R = np.eye(3) * 0.1
        
        # Last timestamp
        self.last_timestamp = None
        
        # Gravity vector
        self.g = np.array([0, 0, -9.81])
        
        # Innovation tracking for spoofing detection
        self.innovation_history = []
        self.innovation_threshold = 5.0  # Minimum 5 meters
        self.max_innovation_history = 50  # Keep last 50 innovation values
        self.suspicious_gps_count = 0
        self.max_suspicious_count = 3  # Number of suspicious readings before mitigation
        
        # Bias detection
        self.gps_imu_bias_history = []
        self.max_bias_history = 20
        self.bias_threshold = 2.0  # meters - threshold for constant bias detection

    def predict(self, imu_data, timestamp):
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            return
        
        # Time delta
        dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp
        
        # Get IMU measurements
        acc = np.array([imu_data.accelerometer.x, 
                       imu_data.accelerometer.y, 
                       imu_data.accelerometer.z])
        gyro = np.array([imu_data.gyroscope.x,
                        imu_data.gyroscope.y,
                        imu_data.gyroscope.z])
        
        # Convert quaternion to rotation matrix
        R = Rotation.from_quat(self.orientation).as_matrix()
        
        # Rotate acceleration to world frame and remove gravity
        acc_world = R @ acc + self.g
        
        # Update state
        self.position += self.velocity * dt + 0.5 * acc_world * dt**2
        self.velocity += acc_world * dt
        
        # Update orientation using gyroscope
        angle = gyro * dt
        dq = Rotation.from_rotvec(angle).as_quat()
        self.orientation = Rotation.from_quat(self.orientation).as_quat()
        self.orientation = (Rotation.from_quat(dq) * Rotation.from_quat(self.orientation)).as_quat()
        
        # Predict covariance
        F = self._get_state_transition_matrix(dt, R, acc)
        self.P = F @ self.P @ F.T + self.Q

    def update_with_gps(self, gps_pos, imu_predicted_pos=None):
        """Correct state using GPS measurement with innovation-based spoofing detection"""
        # Measurement matrix (we only observe position)
        H = np.zeros((3, 9))
        H[:3, :3] = np.eye(3)
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Innovation
        innovation = gps_pos - self.position
        innovation_magnitude = np.linalg.norm(innovation)
        
        # Track innovation history
        self.innovation_history.append(innovation_magnitude)
        if len(self.innovation_history) > self.max_innovation_history:
            self.innovation_history.pop(0)
        
        # Check for suspicious GPS data
        is_suspicious = self._check_suspicious_gps(innovation_magnitude, gps_pos, imu_predicted_pos)
        
        if is_suspicious:
            self.suspicious_gps_count += 1
            print(f"WARNING: Suspicious GPS detected! Innovation: {innovation_magnitude:.2f}m, Count: {self.suspicious_gps_count}")
            
            # If too many suspicious readings, fall back to IMU
            if self.suspicious_gps_count >= self.max_suspicious_count:
                print(f"MITIGATION: Falling back to IMU prediction due to {self.suspicious_gps_count} suspicious GPS readings")
                if imu_predicted_pos is not None:
                    # Reset to IMU prediction
                    self.position = imu_predicted_pos.copy()
                    # Increase uncertainty to reflect the fallback
                    self.P[:3, :3] *= 2.0
                    self.suspicious_gps_count = 0  # Reset counter
                    return False  # Indicate that GPS was rejected
        else:
            # Reset suspicious counter if GPS looks good
            self.suspicious_gps_count = max(0, self.suspicious_gps_count - 1)
        
        # Normal Kalman update
        state_update = K @ innovation
        self.position += state_update[:3]
        self.velocity += state_update[3:6]
        
        # Update orientation using small angle approximation
        angle_update = state_update[6:]
        dq = Rotation.from_rotvec(angle_update).as_quat()
        self.orientation = (Rotation.from_quat(dq) * Rotation.from_quat(self.orientation)).as_quat()
        
        # Update covariance
        self.P = (np.eye(9) - K @ H) @ self.P
        
        return True  # Indicate that GPS was accepted

    def _check_suspicious_gps(self, innovation_magnitude, gps_pos, imu_predicted_pos):
        """Check if GPS data is suspicious based on innovation and bias analysis"""
        # Check if innovation is too large (sudden jump detection)
        if innovation_magnitude > self.innovation_threshold:
            return True
        
        # Check for constant bias if we have IMU prediction
        if imu_predicted_pos is not None:
            bias = np.linalg.norm(gps_pos - imu_predicted_pos)
            self.gps_imu_bias_history.append(bias)
            if len(self.gps_imu_bias_history) > self.max_bias_history:
                self.gps_imu_bias_history.pop(0)
            
            # Check for consistent bias (constant bias attack)
            if len(self.gps_imu_bias_history) >= 5:
                bias_std = np.std(self.gps_imu_bias_history)
                bias_mean = np.mean(self.gps_imu_bias_history)
                
                # If bias is consistent (low std) and significant (high mean), it's suspicious
                if bias_std < 0.5 and bias_mean > self.bias_threshold:
                    return True
        
        return False

    def get_innovation_stats(self):
        """Get innovation statistics for monitoring"""
        if not self.innovation_history:
            return None
        
        return {
            'current_innovation': self.innovation_history[-1] if self.innovation_history else 0,
            'mean_innovation': np.mean(self.innovation_history),
            'max_innovation': np.max(self.innovation_history),
            'suspicious_count': self.suspicious_gps_count,
            'innovation_history': self.innovation_history.copy()
        }

    def get_bias_stats(self):
        """Get bias statistics for monitoring"""
        if not self.gps_imu_bias_history:
            return None
        
        return {
            'current_bias': self.gps_imu_bias_history[-1] if self.gps_imu_bias_history else 0,
            'mean_bias': np.mean(self.gps_imu_bias_history),
            'bias_std': np.std(self.gps_imu_bias_history),
            'bias_history': self.gps_imu_bias_history.copy()
        }

    def _get_state_transition_matrix(self, dt, R, acc):
        """Calculate the state transition matrix F"""
        F = np.eye(9)
        F[:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:] = -dt * self._skew_symmetric(R @ acc)
        return F

    def _skew_symmetric(self, v):
        """Convert vector to skew symmetric matrix"""
        return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])

    def get_state(self):
        return {
            'position': self.position,
            'velocity': self.velocity,
            'orientation': self.orientation
        } 