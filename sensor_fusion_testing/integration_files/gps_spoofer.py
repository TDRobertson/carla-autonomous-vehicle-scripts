import numpy as np
import time
import random
from enum import Enum

class SpoofingStrategy(Enum):
    GRADUAL_DRIFT = 1
    SUDDEN_JUMP = 2
    RANDOM_WALK = 3
    REPLAY = 4
    INNOVATION_AWARE_GRADUAL_DRIFT = 5  # NEW: Innovation-aware gradual drift

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
        
        # Innovation-aware parameters
        self.innovation_threshold = 5.0  # meters - threshold for detection
        self.innovation_history = []
        self.max_innovation_history = 50
        self.current_innovation = 0.0
        self.suspicious_counter = 0
        self.max_suspicious_readings = 3
        
        # Enhanced gradual drift parameters
        self.adaptive_drift_rate = 0.05  # m/s - reduced for subtlety
        self.drift_direction = np.array([1.0, 0.0, 0.0])  # Start drifting in X direction
        self.drift_amplitude = 0.02  # meters - small fluctuations
        self.drift_frequency = 0.1  # Hz - slow oscillations
        
    def spoof_position(self, true_position, innovation=None):
        """
        Generate a spoofed position based on the selected strategy
        """
        # Convert true_position to float64
        true_position = np.array(true_position, dtype=np.float64)
        
        # Update innovation if provided
        if innovation is not None:
            self.update_innovation(innovation)
        
        if self.strategy == SpoofingStrategy.GRADUAL_DRIFT:
            return self._gradual_drift(true_position)
        elif self.strategy == SpoofingStrategy.SUDDEN_JUMP:
            return self._sudden_jump(true_position)
        elif self.strategy == SpoofingStrategy.RANDOM_WALK:
            return self._random_walk(true_position)
        elif self.strategy == SpoofingStrategy.REPLAY:
            return self._replay_attack(true_position)
        elif self.strategy == SpoofingStrategy.INNOVATION_AWARE_GRADUAL_DRIFT:
            return self._innovation_aware_gradual_drift(true_position)
        
    def update_innovation(self, innovation):
        """
        Update innovation history and track suspicious readings
        """
        self.current_innovation = innovation
        self.innovation_history.append(innovation)
        
        # Keep only recent history
        if len(self.innovation_history) > self.max_innovation_history:
            self.innovation_history.pop(0)
        
        # Track suspicious readings
        if abs(innovation) > self.innovation_threshold:
            self.suspicious_counter += 1
        else:
            self.suspicious_counter = max(0, self.suspicious_counter - 1)
    
    def _innovation_aware_gradual_drift(self, true_position):
        """
        Innovation-aware gradual drift that adapts to avoid detection
        """
        elapsed_time = time.time() - self.time_start
        
        # Calculate base drift
        base_drift = self.adaptive_drift_rate * elapsed_time
        
        # Add small oscillations to make the drift more realistic
        oscillation = self.drift_amplitude * np.sin(2 * np.pi * self.drift_frequency * elapsed_time)
        
        # Adaptive drift direction based on innovation
        if self.current_innovation > self.innovation_threshold * 0.8:
            # If approaching threshold, reduce drift rate
            adaptive_rate = self.adaptive_drift_rate * 0.5
        elif self.suspicious_counter > 1:
            # If suspicious readings detected, pause drift
            adaptive_rate = 0.0
        else:
            # Normal drift rate
            adaptive_rate = self.adaptive_drift_rate
        
        # Calculate drift vector
        drift_vector = self.drift_direction * (base_drift * adaptive_rate / self.adaptive_drift_rate + oscillation)
        
        # Add small random perturbations
        random_perturbation = np.array([
            random.uniform(-0.01, 0.01),
            random.uniform(-0.01, 0.01),
            0.0
        ], dtype=np.float64)
        
        # Apply drift
        spoofed_position = true_position + drift_vector + random_perturbation
        
        # Update current position for consistency
        self.current_position = spoofed_position.copy()
        
        return spoofed_position
    
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
        self.innovation_history = []
        self.suspicious_counter = 0
    
    def get_innovation_stats(self):
        """
        Get innovation statistics for monitoring
        """
        if len(self.innovation_history) == 0:
            return {
                'current_innovation': 0.0,
                'mean_innovation': 0.0,
                'max_innovation': 0.0,
                'suspicious_counter': 0,
                'is_suspicious': False
            }
        
        return {
            'current_innovation': self.current_innovation,
            'mean_innovation': np.mean(self.innovation_history),
            'max_innovation': np.max(self.innovation_history),
            'suspicious_counter': self.suspicious_counter,
            'is_suspicious': self.suspicious_counter >= self.max_suspicious_readings
        } 