import numpy as np
import time
import random
from enum import Enum

class SpoofingStrategy(Enum):
    GRADUAL_DRIFT = 1
    SUDDEN_JUMP = 2
    RANDOM_WALK = 3
    REPLAY = 4

class GPSSpoofer:
    def __init__(self, initial_position, strategy=SpoofingStrategy.GRADUAL_DRIFT):
        # Ensure initial_position is converted to float64
        self.initial_position = np.array(initial_position, dtype=np.float64)
        self.current_position = np.array(initial_position, dtype=np.float64)
        self.strategy = strategy
        self.time_start = time.time()
        self.replay_buffer = []
        self.replay_index = 0
        
        # Spoofing parameters
        self.drift_rate = 0.1  # meters per second
        self.jump_magnitude = 5.0  # meters
        self.random_walk_step = 0.5  # meters
        self.replay_delay = 2.0  # seconds
        
    def spoof_position(self, true_position):
        """
        Generate a spoofed position based on the selected strategy
        """
        # Convert true_position to float64
        true_position = np.array(true_position, dtype=np.float64)
        
        if self.strategy == SpoofingStrategy.GRADUAL_DRIFT:
            return self._gradual_drift(true_position)
        elif self.strategy == SpoofingStrategy.SUDDEN_JUMP:
            return self._sudden_jump(true_position)
        elif self.strategy == SpoofingStrategy.RANDOM_WALK:
            return self._random_walk(true_position)
        elif self.strategy == SpoofingStrategy.REPLAY:
            return self._replay_attack(true_position)
        
    def _gradual_drift(self, true_position):
        """
        Gradually drift away from the true position
        """
        elapsed_time = time.time() - self.time_start
        drift = np.array([
            np.sin(elapsed_time) * self.drift_rate,
            np.cos(elapsed_time) * self.drift_rate,
            0.0  # Keep z-coordinate unchanged
        ], dtype=np.float64)
        return true_position + drift
    
    def _sudden_jump(self, true_position):
        """
        Create sudden jumps in position
        """
        if random.random() < 0.01:  # 1% chance of jumping
            jump = np.array([
                random.uniform(-self.jump_magnitude, self.jump_magnitude),
                random.uniform(-self.jump_magnitude, self.jump_magnitude),
                0.0
            ], dtype=np.float64)
            self.current_position = true_position + jump
        return self.current_position
    
    def _random_walk(self, true_position):
        """
        Create a random walk pattern
        """
        step = np.array([
            random.uniform(-self.random_walk_step, self.random_walk_step),
            random.uniform(-self.random_walk_step, self.random_walk_step),
            0.0
        ], dtype=np.float64)
        self.current_position = true_position + step
        return self.current_position
    
    def _replay_attack(self, true_position):
        """
        Implement a replay attack by recording and replaying previous positions
        """
        # Record current position
        self.replay_buffer.append(true_position)
        
        # If we have enough data, start replaying
        if len(self.replay_buffer) > 100:
            if self.replay_index >= len(self.replay_buffer):
                self.replay_index = 0
            position = self.replay_buffer[self.replay_index]
            self.replay_index += 1
            return position
        return true_position
    
    def set_strategy(self, strategy):
        """
        Change the spoofing strategy
        """
        self.strategy = strategy
        self.time_start = time.time()
        self.current_position = self.initial_position.copy()
        self.replay_buffer = []
        self.replay_index = 0 