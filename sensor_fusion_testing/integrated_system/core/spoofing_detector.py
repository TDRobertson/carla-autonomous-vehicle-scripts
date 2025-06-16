import numpy as np
from typing import Tuple, Optional
from .kalman_filter import ExtendedKalmanFilter

class SpoofingDetector:
    def __init__(self, ekf: ExtendedKalmanFilter):
        self.ekf = ekf
        self.spoofing_detected = False
        self.spoofing_threshold = 5.0  # meters
        self.history_size = 10
        self.position_history = []
        self.velocity_history = []
        
    def update(self, gnss_position: np.ndarray, gnss_velocity: np.ndarray) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Update spoofing detection with new GNSS measurements.
        
        Args:
            gnss_position: GNSS position measurement
            gnss_velocity: GNSS velocity measurement
            
        Returns:
            Tuple of (spoofing_detected, corrected_position)
        """
        # Get current state estimate
        state = self.ekf.get_state()
        estimated_position = state[:3]
        estimated_velocity = state[3:6]
        
        # Update history
        self.position_history.append(gnss_position)
        self.velocity_history.append(gnss_velocity)
        if len(self.position_history) > self.history_size:
            self.position_history.pop(0)
            self.velocity_history.pop(0)
            
        # Check for position jumps
        position_jump = np.linalg.norm(gnss_position - estimated_position)
        velocity_jump = np.linalg.norm(gnss_velocity - estimated_velocity)
        
        # Check for consistency in history
        position_consistency = self._check_position_consistency()
        velocity_consistency = self._check_velocity_consistency()
        
        # Detect spoofing
        self.spoofing_detected = (
            position_jump > self.spoofing_threshold or
            velocity_jump > self.spoofing_threshold or
            not position_consistency or
            not velocity_consistency
        )
        
        # If spoofing detected, use EKF estimate
        corrected_position = None
        if self.spoofing_detected:
            corrected_position = estimated_position
            
        return self.spoofing_detected, corrected_position
        
    def _check_position_consistency(self) -> bool:
        """Check if position history is consistent."""
        if len(self.position_history) < 2:
            return True
            
        # Calculate position differences
        diffs = []
        for i in range(1, len(self.position_history)):
            diff = np.linalg.norm(
                self.position_history[i] - self.position_history[i-1]
            )
            diffs.append(diff)
            
        # Check if any difference is too large
        return all(diff < self.spoofing_threshold for diff in diffs)
        
    def _check_velocity_consistency(self) -> bool:
        """Check if velocity history is consistent."""
        if len(self.velocity_history) < 2:
            return True
            
        # Calculate velocity differences
        diffs = []
        for i in range(1, len(self.velocity_history)):
            diff = np.linalg.norm(
                self.velocity_history[i] - self.velocity_history[i-1]
            )
            diffs.append(diff)
            
        # Check if any difference is too large
        return all(diff < self.spoofing_threshold for diff in diffs)
        
    def get_spoofing_status(self) -> bool:
        """Get current spoofing detection status."""
        return self.spoofing_detected 