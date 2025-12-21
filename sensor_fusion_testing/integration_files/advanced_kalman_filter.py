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

    def update_with_gps(self, gps_pos):
        """Correct state using GPS measurement"""
        # Measurement matrix (we only observe position)
        H = np.zeros((3, 9))
        H[:3, :3] = np.eye(3)
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Innovation
        innovation = gps_pos - self.position
        
        # Update state
        state_update = K @ innovation
        self.position += state_update[:3]
        self.velocity += state_update[3:6]
        
        # Update orientation using small angle approximation
        angle_update = state_update[6:]
        dq = Rotation.from_rotvec(angle_update).as_quat()
        self.orientation = (Rotation.from_quat(dq) * Rotation.from_quat(self.orientation)).as_quat()
        
        # Update covariance
        self.P = (np.eye(9) - K @ H) @ self.P
        
        # Return innovation magnitude for monitoring
        return np.linalg.norm(innovation)
    
    def update_with_gps_detailed(self, gps_pos):
        """
        Correct state using GPS measurement and return detailed innovation info.
        
        This is a non-breaking extension that returns additional diagnostic
        information useful for ML-based spoofing detection.
        
        Args:
            gps_pos: GPS position measurement (3D array in meters)
            
        Returns:
            Dictionary containing:
                - innovation_vector: 3D innovation vector (z - H*x_pred)
                - innovation_norm: Euclidean norm of innovation
                - S_matrix: 3x3 innovation covariance matrix
                - S_diag: Diagonal elements of S (position uncertainties)
                - nis: Normalized Innovation Squared (chi-squared statistic)
                - position_pred: Position before update (for comparison)
                - position_upd: Position after update
                - velocity_upd: Velocity after update
                - P_pos_diag: Diagonal of position covariance after update
        """
        # Store pre-update position for comparison
        position_pred = self.position.copy()
        
        # Measurement matrix (we only observe position)
        H = np.zeros((3, 9))
        H[:3, :3] = np.eye(3)
        
        # Innovation covariance
        S = H @ self.P @ H.T + self.R
        S_inv = np.linalg.inv(S)
        
        # Kalman gain
        K = self.P @ H.T @ S_inv
        
        # Innovation vector
        innovation = gps_pos - self.position
        innovation_norm = np.linalg.norm(innovation)
        
        # Normalized Innovation Squared (NIS) - chi-squared distributed with 3 DOF
        # NIS = innovation^T * S^-1 * innovation
        nis = float(innovation.T @ S_inv @ innovation)
        
        # Update state
        state_update = K @ innovation
        self.position += state_update[:3]
        self.velocity += state_update[3:6]
        
        # Update orientation using small angle approximation
        angle_update = state_update[6:]
        dq = Rotation.from_rotvec(angle_update).as_quat()
        self.orientation = (Rotation.from_quat(dq) * Rotation.from_quat(self.orientation)).as_quat()
        
        # Update covariance
        self.P = (np.eye(9) - K @ H) @ self.P
        
        return {
            'innovation_vector': innovation.copy(),
            'innovation_norm': innovation_norm,
            'S_matrix': S.copy(),
            'S_diag': np.diag(S).copy(),
            'nis': nis,
            'position_pred': position_pred,
            'position_upd': self.position.copy(),
            'velocity_upd': self.velocity.copy(),
            'P_pos_diag': np.diag(self.P[:3, :3]).copy()
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