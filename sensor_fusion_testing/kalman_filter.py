import numpy as np

class KalmanFilter:
    def __init__(self):
        # State vector [x, y, z, vx, vy, vz]
        self.x = np.zeros(6)
        
        # State transition matrix
        self.F = np.eye(6)
        self.F[0:3, 3:6] = np.eye(3) * 0.1  # dt = 0.1s
        
        # Measurement matrix (we only measure position)
        self.H = np.zeros((3, 6))
        self.H[0:3, 0:3] = np.eye(3)
        
        # Process noise covariance
        self.Q = np.eye(6) * 0.1
        
        # Measurement noise covariance
        self.R = np.eye(3) * 1.0
        
        # Error covariance matrix
        self.P = np.eye(6) * 1000
        
        # Innovation
        self.y = None
        
        # Kalman gain
        self.K = None
        
    def predict(self):
        """Predict step of the Kalman filter."""
        # Predict state
        self.x = self.F @ self.x
        
        # Predict error covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x[0:3]  # Return predicted position
        
    def update(self, measurement):
        """Update step of the Kalman filter."""
        # Calculate innovation
        self.y = measurement - self.H @ self.x
        
        # Calculate innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Calculate Kalman gain
        self.K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state estimate
        self.x = self.x + self.K @ self.y
        
        # Update error covariance
        self.P = (np.eye(6) - self.K @ self.H) @ self.P
        
        return self.x[0:3]  # Return updated position 