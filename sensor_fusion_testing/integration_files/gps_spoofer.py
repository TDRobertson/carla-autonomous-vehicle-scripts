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
        
        # More subtle spoofing parameters
        self.drift_rate = 0.05  
        self.jump_magnitude = 3.0  
        self.random_walk_step = 0.2  
        self.replay_delay = 2.0  # seconds
        
        # Innovation-aware parameters
        self.innovation_threshold = 4.5  # Stay just under 5m threshold
        self.last_innovation = 0.0
        self.attack_phase = 0  # Track attack progression
        
        # Gradual drift with random fluctuations
        self.drift_base = np.array([0.0, 0.0, 0.0])
        self.drift_fluctuation = 0.02  # Small random fluctuations
        
        # Sudden jump with innovation awareness
        self.jump_probability = 0.005  #0.5% chance of jump per second
        self.last_jump_time = 0.0
        self.jump_cooldown = 10.0  # Minimum time between jumps
        
    def spoof_position(self, true_position):
        """
        Generate a spoofed position based on the selected strategy
        """
        # Convert true_position to float64
        true_position = np.array(true_position, dtype=np.float64)
        
        if self.strategy == SpoofingStrategy.GRADUAL_DRIFT:
            return self._gradual_drift_improved(true_position)
        elif self.strategy == SpoofingStrategy.SUDDEN_JUMP:
            return self._sudden_jump_improved(true_position)
        elif self.strategy == SpoofingStrategy.RANDOM_WALK:
            return self._random_walk_improved(true_position)
        elif self.strategy == SpoofingStrategy.REPLAY:
            return self._replay_attack_improved(true_position)
        
    def _gradual_drift_improved(self, true_position):
        """
        Gradually drift away from the true position with random fluctuations
        This makes the attack more subtle and harder to detect
        """
        elapsed_time = time.time() - self.time_start
        
        # Base drift with sine/cosine pattern
        base_drift = np.array([
            np.sin(elapsed_time * 0.1) * self.drift_rate,
            np.cos(elapsed_time * 0.1) * self.drift_rate,
            0.0  # Keep z-coordinate unchanged
        ], dtype=np.float64)
        
        # Add random fluctuations within a small radius
        fluctuation = np.array([
            random.uniform(-self.drift_fluctuation, self.drift_fluctuation),
            random.uniform(-self.drift_fluctuation, self.drift_fluctuation),
            0.0
        ], dtype=np.float64)
        
        # Combine base drift and fluctuation
        total_drift = base_drift + fluctuation
        
        # Ensure we don't exceed innovation threshold
        drift_magnitude = np.linalg.norm(total_drift)
        if drift_magnitude > self.innovation_threshold:
            # Scale down to stay within threshold
            total_drift = total_drift * (self.innovation_threshold / drift_magnitude)
        
        return true_position + total_drift
    
    def _sudden_jump_improved(self, true_position):
        """
        Create sudden jumps in position with innovation awareness
        """
        current_time = time.time()
        
        # Check cooldown and probability
        if (current_time - self.last_jump_time > self.jump_cooldown and 
            random.random() < self.jump_probability):
            
            # Calculate jump magnitude based on current innovation
            # If innovation is low, we can make a larger jump
            # If innovation is high, we make a smaller jump to avoid detection
            if self.last_innovation < 2.0:
                jump_size = self.jump_magnitude
            elif self.last_innovation < 4.0:
                jump_size = self.jump_magnitude * 0.7
            else:
                jump_size = self.jump_magnitude * 0.4
            
            # Create jump in random direction
            jump_direction = np.array([
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                0.0
            ], dtype=np.float64)
            jump_direction = jump_direction / np.linalg.norm(jump_direction)
            
            jump = jump_direction * jump_size
            self.current_position = true_position + jump
            self.last_jump_time = current_time
            
            print(f"SUDDEN JUMP: Applied jump of {jump_size:.2f}m magnitude")
        
        return self.current_position
    
    def _random_walk_improved(self, true_position):
        """
        Create a random walk pattern
        """
        # Smaller steps with some directionality
        step_size = random.uniform(0, self.random_walk_step)
        
        # Add some persistence to the walk (not completely random)
        if hasattr(self, '_last_walk_direction'):
            # 70% chance to continue in similar direction
            if random.random() < 0.7:
                direction = self._last_walk_direction + np.array([
                    random.uniform(-0.3, 0.3),
                    random.uniform(-0.3, 0.3),
                    0.0
                ], dtype=np.float64)
            else:
                direction = np.array([
                    random.uniform(-1, 1),
                    random.uniform(-1, 1),
                    0.0
                ], dtype=np.float64)
        else:
            direction = np.array([
                random.uniform(-1, 1),
                random.uniform(-1, 1),
                0.0
            ], dtype=np.float64)
        
        # Normalize direction
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)
        
        self._last_walk_direction = direction
        
        step = direction * step_size
        self.current_position = true_position + step
        
        return self.current_position
    
    def _replay_attack_improved(self, true_position):
        """
        Implement a more sophisticated replay attack
        """
        # Record current position
        self.replay_buffer.append(true_position)
        
        # If we have enough data, start replaying with some modifications
        if len(self.replay_buffer) > 100:
            if self.replay_index >= len(self.replay_buffer):
                self.replay_index = 0
            
            # Get the recorded position
            recorded_position = self.replay_buffer[self.replay_index]
            
            # Add small noise to make it less obvious
            noise = np.array([
                random.uniform(-0.1, 0.1),
                random.uniform(-0.1, 0.1),
                0.0
            ], dtype=np.float64)
            
            position = recorded_position + noise
            self.replay_index += 1
            return position
        
        return true_position
    
    def update_innovation(self, innovation_magnitude):
        """
        Update the spoofer with current innovation magnitude for adaptive attacks
        """
        self.last_innovation = innovation_magnitude
    
    def set_strategy(self, strategy):
        """
        Change the spoofing strategy
        """
        self.strategy = strategy
        self.time_start = time.time()
        self.current_position = self.initial_position.copy()
        self.replay_buffer = []
        self.replay_index = 0
        self.attack_phase = 0
        self.last_innovation = 0.0 