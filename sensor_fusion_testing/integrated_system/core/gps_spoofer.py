import numpy as np
from typing import Optional, Tuple
from enum import Enum

class SpoofingStrategy(Enum):
    NONE = "none"
    JUMP = "jump"
    DRIFT = "drift"
    RANDOM = "random"
    SEQUENTIAL = "sequential"

class GPSSpoofer:
    def __init__(self):
        self.strategy = SpoofingStrategy.NONE
        self.spoofing_active = False
        self.jump_magnitude = 10.0  # meters
        self.drift_rate = 0.1  # meters/second
        self.random_std = 5.0  # meters
        self.sequential_step = 1.0  # meters
        self.sequential_direction = np.array([1.0, 0.0, 0.0])  # x-direction
        self.sequential_count = 0
        
    def set_strategy(self, strategy: SpoofingStrategy):
        """Set the spoofing strategy."""
        self.strategy = strategy
        self.spoofing_active = strategy != SpoofingStrategy.NONE
        self.sequential_count = 0
        
    def spoof_position(self, true_position: np.ndarray) -> np.ndarray:
        """
        Apply spoofing to the true position based on the current strategy.
        
        Args:
            true_position: The true position from GNSS
            
        Returns:
            The spoofed position
        """
        if not self.spoofing_active:
            return true_position
            
        if self.strategy == SpoofingStrategy.JUMP:
            return self._apply_jump(true_position)
        elif self.strategy == SpoofingStrategy.DRIFT:
            return self._apply_drift(true_position)
        elif self.strategy == SpoofingStrategy.RANDOM:
            return self._apply_random(true_position)
        elif self.strategy == SpoofingStrategy.SEQUENTIAL:
            return self._apply_sequential(true_position)
        else:
            return true_position
            
    def _apply_jump(self, true_position: np.ndarray) -> np.ndarray:
        """Apply a position jump."""
        jump = np.random.randn(3) * self.jump_magnitude
        return true_position + jump
        
    def _apply_drift(self, true_position: np.ndarray) -> np.ndarray:
        """Apply a gradual drift."""
        drift = self.sequential_direction * self.drift_rate
        return true_position + drift
        
    def _apply_random(self, true_position: np.ndarray) -> np.ndarray:
        """Apply random noise."""
        noise = np.random.randn(3) * self.random_std
        return true_position + noise
        
    def _apply_sequential(self, true_position: np.ndarray) -> np.ndarray:
        """Apply sequential steps."""
        step = self.sequential_direction * self.sequential_step
        self.sequential_count += 1
        return true_position + step * self.sequential_count
        
    def get_spoofing_status(self) -> Tuple[bool, str]:
        """Get current spoofing status and strategy."""
        return self.spoofing_active, self.strategy.value 