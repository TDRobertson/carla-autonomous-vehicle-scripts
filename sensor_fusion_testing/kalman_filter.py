import numpy as np

class KalmanFilter:
    def __init__(self, dt=0.1):
        # State transition matrix
        self.A = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.eye(6) * 0.1
        
        # Measurement noise covariance
        self.R = np.eye(3) * 0.1
        
        # Initial state estimate
        self.x = np.zeros((6, 1))
        
        # Initial error covariance
        self.P = np.eye(6) * 1.0
        
    def predict(self):
        # Predict state
        self.x = self.A @ self.x
        
        # Predict error covariance
        self.P = self.A @ self.P @ self.A.T + self.Q
        
        return self.x[:3]  # Return position estimate
        
    def update(self, measurement):
        # Calculate Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state estimate
        y = measurement.reshape(3, 1) - self.H @ self.x
        self.x = self.x + K @ y
        
        # Update error covariance
        self.P = (np.eye(6) - K @ self.H) @ self.P
        
        return self.x[:3]  # Return updated position estimate 