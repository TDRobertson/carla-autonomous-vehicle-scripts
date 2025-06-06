import numpy as np
from typing import Optional, Tuple
import time

class KalmanFilter:
    def __init__(self):
        # State vector [x, y, z, vx, vy, vz]
        self.x = np.zeros(6)
        
        # State transition matrix (will be updated with actual dt)
        self.F = np.eye(6)
        
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
        
        # Time tracking
        self.last_update_time = None
        
        # Velocity constraints
        self.MAX_VELOCITY = 50.0  # m/s
        self.MAX_ACCELERATION = 10.0  # m/s^2
        
        # State history for smoothing
        self.state_history = []
        self.max_history_size = 10
        
    def _update_state_transition_matrix(self, dt: float) -> None:
        """Update the state transition matrix with the current time step."""
        self.F = np.eye(6)
        self.F[0:3, 3:6] = np.eye(3) * dt
        
    def _constrain_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """Apply velocity constraints to prevent unrealistic values."""
        velocity_magnitude = np.linalg.norm(velocity)
        
        if velocity_magnitude > self.MAX_VELOCITY:
            # Scale down velocity while maintaining direction
            velocity = velocity * (self.MAX_VELOCITY / velocity_magnitude)
            
        return velocity
        
    def _smooth_velocity_transition(self, new_velocity: np.ndarray) -> np.ndarray:
        """Smooth velocity transitions to prevent sudden changes."""
        if not self.state_history:
            return new_velocity
            
        # Get last velocity
        last_velocity = self.x[3:6]
        
        # Calculate velocity change
        velocity_change = new_velocity - last_velocity
        change_magnitude = np.linalg.norm(velocity_change)
        
        if change_magnitude > self.MAX_ACCELERATION:
            # Scale down the change while maintaining direction
            velocity_change = velocity_change * (self.MAX_ACCELERATION / change_magnitude)
            new_velocity = last_velocity + velocity_change
            
        return new_velocity
        
    def _update_state_history(self) -> None:
        """Update the state history for smoothing."""
        self.state_history.append(self.x.copy())
        if len(self.state_history) > self.max_history_size:
            self.state_history.pop(0)
            
    def _get_smoothed_state(self) -> np.ndarray:
        """Get smoothed state estimate using exponential weighting."""
        if not self.state_history:
            return self.x
            
        # Calculate exponential weights
        weights = np.exp(-np.arange(len(self.state_history)))
        weights = weights / np.sum(weights)
        
        # Calculate weighted average
        smoothed_state = np.zeros_like(self.x)
        for i, state in enumerate(self.state_history):
            smoothed_state += state * weights[i]
            
        return smoothed_state
        
    def predict(self, current_time: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Predict step of the Kalman filter with adaptive time step."""
        if current_time is None:
            current_time = time.time()
            
        # Calculate time step
        if self.last_update_time is None:
            dt = 0.1  # Default time step
        else:
            dt = current_time - self.last_update_time
            
        # Update state transition matrix
        self._update_state_transition_matrix(dt)
        
        # Predict state
        self.x = self.F @ self.x
        
        # Constrain velocity
        self.x[3:6] = self._constrain_velocity(self.x[3:6])
        
        # Predict error covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x[0:3], self.x[3:6]  # Return predicted position and velocity
        
    def update(self, measurement: np.ndarray, current_time: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Update step of the Kalman filter."""
        if current_time is None:
            current_time = time.time()
            
        # Calculate innovation
        self.y = measurement - self.H @ self.x
        
        # Calculate innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Calculate Kalman gain
        self.K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state estimate
        self.x = self.x + self.K @ self.y
        
        # Constrain and smooth velocity
        self.x[3:6] = self._constrain_velocity(self.x[3:6])
        self.x[3:6] = self._smooth_velocity_transition(self.x[3:6])
        
        # Update error covariance
        self.P = (np.eye(6) - self.K @ self.H) @ self.P
        
        # Update state history
        self._update_state_history()
        
        # Get smoothed state
        smoothed_state = self._get_smoothed_state()
        
        # Update last update time
        self.last_update_time = current_time
        
        return smoothed_state[0:3], smoothed_state[3:6]  # Return smoothed position and velocity
        
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current state estimate."""
        return self.x[0:3], self.x[3:6]  # Return position and velocity
        
    def get_covariance(self) -> np.ndarray:
        """Get current error covariance matrix."""
        return self.P
        
    def get_kalman_gain(self) -> Optional[np.ndarray]:
        """Get current Kalman gain."""
        return self.K
        
    def get_innovation(self) -> Optional[np.ndarray]:
        """Get current innovation."""
        return self.y 