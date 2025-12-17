import numpy as np
import time
import random
from enum import Enum
from typing import Optional, Dict, Tuple

class SpoofingStrategy(Enum):
    GRADUAL_DRIFT = 1
    SUDDEN_JUMP = 2
    RANDOM_WALK = 3
    REPLAY = 4
    INNOVATION_AWARE_GRADUAL_DRIFT = 5  # Innovation-aware gradual drift


class ChaoticScheduler:
    """
    Manages chaotic on/off attack scheduling with strength modulation.
    
    State machine: CLEAN <-> ATTACK with random durations and varying strength.
    """
    
    def __init__(
        self,
        min_clean_s: float = 5.0,
        max_clean_s: float = 20.0,
        min_attack_s: float = 10.0,
        max_attack_s: float = 40.0,
        strength_min: float = 0.3,
        strength_max: float = 1.5,
        strength_hold_s: float = 5.0,
        seed: Optional[int] = None
    ):
        """
        Initialize the chaotic scheduler.
        
        Args:
            min_clean_s: Minimum duration of clean (no attack) windows
            max_clean_s: Maximum duration of clean windows
            min_attack_s: Minimum duration of attack windows
            max_attack_s: Maximum duration of attack windows
            strength_min: Minimum strength multiplier during attacks
            strength_max: Maximum strength multiplier during attacks
            strength_hold_s: How long to hold a strength level before changing
            seed: Optional RNG seed for reproducibility
        """
        self.min_clean_s = min_clean_s
        self.max_clean_s = max_clean_s
        self.min_attack_s = min_attack_s
        self.max_attack_s = max_attack_s
        self.strength_min = strength_min
        self.strength_max = strength_max
        self.strength_hold_s = strength_hold_s
        
        # RNG for reproducibility
        self.rng = random.Random(seed)
        
        # State machine
        self.is_attacking = False
        self.current_strength = 1.0
        self.state_start_time = 0.0
        self.state_duration = 0.0
        self.strength_change_time = 0.0
        
        # Initialize first state (start clean)
        self._transition_to_clean(0.0)
        
    def _transition_to_clean(self, current_time: float):
        """Transition to clean state."""
        self.is_attacking = False
        self.current_strength = 0.0
        self.state_start_time = current_time
        self.state_duration = self.rng.uniform(self.min_clean_s, self.max_clean_s)
        
    def _transition_to_attack(self, current_time: float):
        """Transition to attack state."""
        self.is_attacking = True
        self.state_start_time = current_time
        self.state_duration = self.rng.uniform(self.min_attack_s, self.max_attack_s)
        self._sample_new_strength(current_time)
        
    def _sample_new_strength(self, current_time: float):
        """Sample a new strength multiplier."""
        self.current_strength = self.rng.uniform(self.strength_min, self.strength_max)
        self.strength_change_time = current_time
        
    def update(self, current_time: float) -> Tuple[bool, float]:
        """
        Update the scheduler state based on current time.
        
        Args:
            current_time: Current simulation time in seconds
            
        Returns:
            Tuple of (is_attacking, strength_multiplier)
        """
        time_in_state = current_time - self.state_start_time
        
        # Check for state transition
        if time_in_state >= self.state_duration:
            if self.is_attacking:
                self._transition_to_clean(current_time)
            else:
                self._transition_to_attack(current_time)
                
        # If attacking, check for strength change
        if self.is_attacking:
            time_since_strength_change = current_time - self.strength_change_time
            if time_since_strength_change >= self.strength_hold_s:
                self._sample_new_strength(current_time)
                
        return self.is_attacking, self.current_strength
    
    def get_state_info(self) -> Dict:
        """Get current state information for logging."""
        return {
            'is_attacking': self.is_attacking,
            'current_strength': self.current_strength,
            'time_in_state': 0.0,  # Will be updated by caller
            'state_duration': self.state_duration
        }
    
    def reset(self, seed: Optional[int] = None):
        """Reset the scheduler state."""
        if seed is not None:
            self.rng = random.Random(seed)
        self.is_attacking = False
        self.current_strength = 1.0
        self.state_start_time = 0.0
        self.state_duration = 0.0
        self.strength_change_time = 0.0
        self._transition_to_clean(0.0)


class GPSSpoofer:
    def __init__(self, initial_position, strategy=SpoofingStrategy.GRADUAL_DRIFT, aggressive_mode=False):
        # Ensure initial_position is converted to float64
        self.initial_position = np.array(initial_position, dtype=np.float64)
        self.current_position = np.array(initial_position, dtype=np.float64)
        self.strategy = strategy
        self.time_start = time.time()
        self.replay_buffer = []
        self.replay_index = 0
        self.aggressive_mode = aggressive_mode
        
        # Spoofing parameters - MADE MORE AGGRESSIVE
        self.drift_rate = 0.2  # meters per second (was 0.1)
        self.jump_magnitude = 5.0  # meters (was 5.0)
        self.jump_probability = 0.02  # probability per step for sudden jumps (was 0.01)
        self.random_walk_step = 2.0  # meters (was 0.5)
        self.replay_delay = 2.0  # seconds (was 2.0)
        
        # Aggressive mode parameters for GPS-only systems
        if self.aggressive_mode:
            self.replay_delay = 0.2  # Much faster replay (was 2.0)
            self.replay_buffer_size = 50  # Larger buffer for more dramatic replay
        
        # Innovation-aware parameters
        self.innovation_threshold = 5.0  # meters - threshold for detection
        self.innovation_history = []
        self.max_innovation_history = 50
        self.current_innovation = 0.0
        self.suspicious_counter = 0
        self.max_suspicious_readings = 3
        
        # Enhanced gradual drift parameters
        self.adaptive_drift_rate = 0.15  # m/s - increased for more effectiveness
        self.drift_direction = np.array([1.0, 0.0, 0.0])  # Start drifting in X direction
        self.drift_amplitude = 0.05  # meters - increased fluctuations
        self.drift_frequency = 0.05  # Hz - slower oscillations for more persistent drift
        self.min_drift_rate = 0.02  # m/s - minimum drift rate to maintain effectiveness
        self.max_drift_rate = 0.25  # m/s - maximum drift rate
        self.innovation_safety_margin = 0.7  # Reduce drift when innovation > 70% of threshold
        
        # Chaotic mode parameters
        self.chaotic_mode = False
        self.chaotic_scheduler = None
        self.chaotic_attack_active = False
        self.chaotic_strength = 1.0
        
    def enable_chaotic_mode(
        self,
        min_clean_s: float = 5.0,
        max_clean_s: float = 20.0,
        min_attack_s: float = 10.0,
        max_attack_s: float = 40.0,
        strength_min: float = 0.3,
        strength_max: float = 1.5,
        strength_hold_s: float = 5.0,
        seed: Optional[int] = None
    ):
        """
        Enable chaotic attack scheduling with random on/off windows and strength modulation.
        
        Args:
            min_clean_s: Minimum duration of clean (no attack) windows
            max_clean_s: Maximum duration of clean windows
            min_attack_s: Minimum duration of attack windows
            max_attack_s: Maximum duration of attack windows
            strength_min: Minimum strength multiplier during attacks
            strength_max: Maximum strength multiplier during attacks
            strength_hold_s: How long to hold a strength level before changing
            seed: Optional RNG seed for reproducibility
        """
        self.chaotic_mode = True
        self.chaotic_scheduler = ChaoticScheduler(
            min_clean_s=min_clean_s,
            max_clean_s=max_clean_s,
            min_attack_s=min_attack_s,
            max_attack_s=max_attack_s,
            strength_min=strength_min,
            strength_max=strength_max,
            strength_hold_s=strength_hold_s,
            seed=seed
        )
        
    def disable_chaotic_mode(self):
        """Disable chaotic attack scheduling."""
        self.chaotic_mode = False
        self.chaotic_scheduler = None
        self.chaotic_attack_active = False
        self.chaotic_strength = 1.0
        
    def is_chaotic_attack_active(self) -> bool:
        """Check if the chaotic scheduler currently has an attack active."""
        return self.chaotic_mode and self.chaotic_attack_active
    
    def get_chaotic_state(self) -> Dict:
        """Get current chaotic scheduler state for logging."""
        if not self.chaotic_mode or self.chaotic_scheduler is None:
            return {
                'chaotic_mode': False,
                'is_attacking': False,
                'strength': 1.0
            }
        return {
            'chaotic_mode': True,
            'is_attacking': self.chaotic_attack_active,
            'strength': self.chaotic_strength,
            **self.chaotic_scheduler.get_state_info()
        }
        
    def spoof_position(self, true_position, innovation=None, elapsed_time: Optional[float] = None):
        """
        Generate a spoofed position based on the selected strategy.
        
        Args:
            true_position: True GPS position
            innovation: Current Kalman filter innovation value
            elapsed_time: Optional elapsed time for chaotic scheduling (uses wall clock if None)
            
        Returns:
            Spoofed position (or true position if chaotic mode says no attack)
        """
        # Convert true_position to float64
        true_position = np.array(true_position, dtype=np.float64)
        
        # Update innovation if provided
        if innovation is not None:
            self.update_innovation(innovation)
        
        # Handle chaotic mode scheduling
        if self.chaotic_mode and self.chaotic_scheduler is not None:
            # Use provided elapsed_time or calculate from wall clock
            if elapsed_time is None:
                elapsed_time = time.time() - self.time_start
            
            # Update chaotic scheduler
            self.chaotic_attack_active, self.chaotic_strength = self.chaotic_scheduler.update(elapsed_time)
            
            # If chaotic mode says no attack, return true position
            if not self.chaotic_attack_active:
                self.current_position = true_position.copy()
                return true_position
        
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
        Innovation-aware gradual drift that adapts to avoid detection.
        
        In chaotic mode, the strength multiplier modulates the base drift rate
        and amplitude, creating more varied attack patterns.
        """
        elapsed_time = time.time() - self.time_start
        
        # Get strength multiplier from chaotic mode (default 1.0 if not chaotic)
        strength = self.chaotic_strength if self.chaotic_mode else 1.0
        
        # Apply strength to base parameters
        effective_drift_rate = self.adaptive_drift_rate * strength
        effective_amplitude = self.drift_amplitude * strength
        effective_max_rate = self.max_drift_rate * strength
        
        # Calculate base drift with exponential growth for more persistent effect
        base_drift = effective_drift_rate * elapsed_time * (1 + 0.1 * elapsed_time)
        
        # Add oscillations to make the drift more realistic
        oscillation = effective_amplitude * np.sin(2 * np.pi * self.drift_frequency * elapsed_time)
        
        # Enhanced adaptive logic based on innovation (safety guardrail - always applies)
        if self.current_innovation > self.innovation_threshold * self.innovation_safety_margin:
            # If approaching threshold, reduce drift rate but maintain minimum
            adaptive_rate = max(self.min_drift_rate, effective_drift_rate * 0.3)
        elif self.suspicious_counter > 2:
            # If multiple suspicious readings detected, pause drift temporarily
            adaptive_rate = 0.0
        elif self.suspicious_counter > 0:
            # If some suspicious readings, reduce drift rate
            adaptive_rate = max(self.min_drift_rate, effective_drift_rate * 0.6)
        else:
            # Normal drift rate with some variation
            adaptive_rate = effective_drift_rate * (0.8 + 0.4 * random.random())
            adaptive_rate = min(effective_max_rate, adaptive_rate)
        
        # Calculate drift vector with enhanced direction changes
        if effective_drift_rate > 0:
            drift_vector = self.drift_direction * (base_drift * adaptive_rate / effective_drift_rate + oscillation)
        else:
            drift_vector = self.drift_direction * oscillation
        
        # Add directional changes over time to make attack more sophisticated
        if elapsed_time > 10.0:  # After 10 seconds, start changing direction
            direction_change = np.array([
                np.sin(elapsed_time * 0.1) * 0.3 * strength,
                np.cos(elapsed_time * 0.1) * 0.3 * strength,
                0.0
            ], dtype=np.float64)
            drift_vector += direction_change
        
        # Add small random perturbations (scaled by strength)
        random_perturbation = np.array([
            random.uniform(-0.02, 0.02) * strength,
            random.uniform(-0.02, 0.02) * strength,
            0.0
        ], dtype=np.float64)
        
        # Apply drift
        spoofed_position = true_position + drift_vector + random_perturbation
        
        # Update current position for consistency
        self.current_position = spoofed_position.copy()
        
        return spoofed_position
    
    def _gradual_drift(self, true_position):
        """
        Gradually drift away from the true position - ENHANCED FOR VISIBILITY
        """
        elapsed_time = time.time() - self.time_start
        
        # Create a more aggressive drift pattern
        # Use exponential growth to make drift more visible over time
        drift_amplitude = self.drift_rate * elapsed_time * (1 + 0.2 * elapsed_time)
        
        # Create a figure-8 pattern for more dramatic movement
        drift = np.array([
            np.sin(elapsed_time * 0.5) * drift_amplitude,
            np.sin(elapsed_time * 0.3) * drift_amplitude * 0.7,
            0.0  # Keep z-coordinate unchanged
        ], dtype=np.float64)
        
        # Add some random variation to make it less predictable
        random_variation = np.array([
            random.uniform(-0.5, 0.5),
            random.uniform(-0.5, 0.5),
            0.0
        ], dtype=np.float64)
        
        return true_position + drift + random_variation
    
    def _sudden_jump(self, true_position):
        """
        Create sudden jumps in position - ENHANCED FOR VISIBILITY
        """
        # Increase probability over time to make jumps more frequent
        elapsed_time = time.time() - self.time_start
        dynamic_probability = self.jump_probability * (1 + elapsed_time * 0.1)
        
        if random.random() < dynamic_probability:
            # Create larger, more dramatic jumps
            jump_magnitude = self.jump_magnitude * (1 + random.random())
            jump = np.array([
                random.uniform(-jump_magnitude, jump_magnitude),
                random.uniform(-jump_magnitude, jump_magnitude),
                0.0
            ], dtype=np.float64)
            self.current_position = true_position + jump
        return self.current_position
    
    def _random_walk(self, true_position):
        """
        Create a random walk pattern - ENHANCED FOR VISIBILITY
        """
        # Make steps larger and more frequent over time
        elapsed_time = time.time() - self.time_start
        step_multiplier = 1 + elapsed_time * 0.1
        
        step = np.array([
            random.uniform(-self.random_walk_step, self.random_walk_step) * step_multiplier,
            random.uniform(-self.random_walk_step, self.random_walk_step) * step_multiplier,
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
        
        # In aggressive mode, limit buffer size to create more dramatic replay
        if self.aggressive_mode and len(self.replay_buffer) > self.replay_buffer_size:
            self.replay_buffer.pop(0)  # Remove oldest position
        
        # If we have enough buffered history, start replaying
        # Approximate GPS callback at ~10 Hz -> gate by replay_delay seconds
        gate = max(1, int(self.replay_delay * 10.0))
        if len(self.replay_buffer) > gate:
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
        
        # Reset chaotic scheduler if enabled
        if self.chaotic_mode and self.chaotic_scheduler is not None:
            self.chaotic_scheduler.reset()
    
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