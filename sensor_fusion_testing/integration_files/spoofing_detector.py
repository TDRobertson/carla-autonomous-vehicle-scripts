import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class SpoofingType(Enum):
    GRADUAL_DRIFT = "gradual_drift"
    SUDDEN_JUMP = "sudden_jump"
    RANDOM_WALK = "random_walk"
    REPLAY = "replay"

@dataclass
class DetectionResult:
    detected: bool
    confidence: float
    attack_type: SpoofingType
    details: Dict

class SpoofingDetector:
    def __init__(self, window_size: int = 100):
        # Detection thresholds (in meters)
        self.gradual_drift_threshold = 0.15
        self.sudden_jump_threshold = 1.0
        self.random_walk_threshold = 0.4
        self.replay_threshold = 0.1
        
        # History buffers
        self.window_size = window_size
        self.position_history: List[np.ndarray] = []
        self.velocity_history: List[np.ndarray] = []
        self.timestamp_history: List[float] = []
        
        # Statistical analysis
        self.error_mean = 0.0
        self.error_variance = 0.0
        
    def _update_history(self, position: np.ndarray, velocity: np.ndarray, timestamp: float) -> None:
        """Update history buffers with new measurements."""
        self.position_history.append(position)
        self.velocity_history.append(velocity)
        self.timestamp_history.append(timestamp)
        
        # Maintain window size
        if len(self.position_history) > self.window_size:
            self.position_history.pop(0)
            self.velocity_history.pop(0)
            self.timestamp_history.pop(0)
            
        # Update error statistics
        if len(self.position_history) > 1:
            errors = [np.linalg.norm(p2 - p1) for p1, p2 in zip(self.position_history[:-1], self.position_history[1:])]
            self.error_mean = np.mean(errors)
            self.error_variance = np.var(errors)
    
    def detect_spoofing(self, current_position: np.ndarray, current_velocity: np.ndarray, 
                       timestamp: float) -> DetectionResult:
        """Main detection method that checks for all types of spoofing."""
        self._update_history(current_position, current_velocity, timestamp)
        
        # Run all detection methods
        drift_result = self._detect_gradual_drift()
        jump_result = self._detect_sudden_jump()
        walk_result = self._detect_random_walk()
        replay_result = self._detect_replay()
        
        # Find the most likely attack type
        results = [
            (drift_result, SpoofingType.GRADUAL_DRIFT),
            (jump_result, SpoofingType.SUDDEN_JUMP),
            (walk_result, SpoofingType.RANDOM_WALK),
            (replay_result, SpoofingType.REPLAY)
        ]
        
        # Return the result with highest confidence
        best_result = max(results, key=lambda x: x[0]['confidence'])
        return DetectionResult(
            detected=best_result[0]['detected'],
            confidence=best_result[0]['confidence'],
            attack_type=best_result[1],
            details=best_result[0]
        )
    
    def _detect_gradual_drift(self) -> Dict:
        """Detect gradual drift by analyzing position trends and velocity consistency."""
        if len(self.position_history) < 2:
            return {'detected': False, 'confidence': 0.0}
            
        # Calculate position trend
        positions = np.array(self.position_history)
        velocities = np.array(self.velocity_history)
        
        # Calculate drift rate
        position_diffs = np.diff(positions, axis=0)
        drift_rate = np.mean(np.linalg.norm(position_diffs, axis=1))
        
        # Check velocity consistency
        velocity_changes = np.diff(velocities, axis=0)
        velocity_consistency = 1.0 - np.mean(np.linalg.norm(velocity_changes, axis=1))
        
        # Calculate confidence
        confidence = min(1.0, drift_rate / self.gradual_drift_threshold)
        
        return {
            'detected': drift_rate > self.gradual_drift_threshold,
            'confidence': confidence,
            'drift_rate': drift_rate,
            'velocity_consistency': velocity_consistency
        }
    
    def _detect_sudden_jump(self) -> Dict:
        """Detect sudden jumps by analyzing position discontinuities."""
        if len(self.position_history) < 2:
            return {'detected': False, 'confidence': 0.0}
            
        # Calculate position differences
        positions = np.array(self.position_history)
        position_diffs = np.diff(positions, axis=0)
        max_diff = np.max(np.linalg.norm(position_diffs, axis=1))
        
        # Check for velocity anomalies
        velocities = np.array(self.velocity_history)
        velocity_diffs = np.diff(velocities, axis=0)
        velocity_anomaly = np.max(np.linalg.norm(velocity_diffs, axis=1))
        
        # Calculate confidence
        confidence = min(1.0, max_diff / self.sudden_jump_threshold)
        
        return {
            'detected': max_diff > self.sudden_jump_threshold,
            'confidence': confidence,
            'max_position_diff': max_diff,
            'velocity_anomaly': velocity_anomaly
        }
    
    def _detect_random_walk(self) -> Dict:
        """Detect random walk by analyzing movement statistics."""
        if len(self.position_history) < 2:
            return {'detected': False, 'confidence': 0.0}
            
        # Calculate movement statistics
        positions = np.array(self.position_history)
        movements = np.diff(positions, axis=0)
        
        # Calculate statistics
        movement_mean = np.mean(movements, axis=0)
        movement_std = np.std(movements, axis=0)
        movement_variance = np.var(movements, axis=0)
        
        # Check for random walk characteristics
        is_random = np.all(movement_std > 0.1) and np.all(np.abs(movement_mean) < 0.1)
        
        # Calculate confidence
        confidence = min(1.0, np.mean(movement_variance) / self.random_walk_threshold)
        
        return {
            'detected': is_random and np.mean(movement_variance) > self.random_walk_threshold,
            'confidence': confidence,
            'movement_variance': movement_variance,
            'is_random': is_random
        }
    
    def _detect_replay(self) -> Dict:
        """Detect replay attacks by analyzing temporal patterns."""
        if len(self.position_history) < self.window_size:
            return {'detected': False, 'confidence': 0.0}
            
        # Calculate temporal patterns
        positions = np.array(self.position_history)
        timestamps = np.array(self.timestamp_history)
        
        # Check for repeating patterns
        position_diffs = np.diff(positions, axis=0)
        time_diffs = np.diff(timestamps)
        
        # Look for periodic patterns
        pattern_length = self._find_pattern_length(position_diffs)
        temporal_consistency = self._check_temporal_consistency(time_diffs)
        
        # Calculate confidence
        confidence = min(1.0, pattern_length / self.window_size)
        
        return {
            'detected': pattern_length > 0 and temporal_consistency > self.replay_threshold,
            'confidence': confidence,
            'pattern_length': pattern_length,
            'temporal_consistency': temporal_consistency
        }
    
    def _find_pattern_length(self, diffs: np.ndarray) -> int:
        """Find the length of repeating patterns in the differences."""
        for length in range(2, len(diffs) // 2):
            if self._check_pattern(diffs, length):
                return length
        return 0
    
    def _check_pattern(self, diffs: np.ndarray, length: int) -> bool:
        """Check if there's a repeating pattern of given length."""
        if len(diffs) < 2 * length:
            return False
            
        pattern = diffs[:length]
        for i in range(length, len(diffs) - length, length):
            if not np.allclose(diffs[i:i+length], pattern, atol=0.1):
                return False
        return True
    
    def _check_temporal_consistency(self, time_diffs: np.ndarray) -> float:
        """Check the consistency of time differences."""
        if len(time_diffs) < 2:
            return 0.0
            
        # Calculate the variance of time differences
        time_variance = np.var(time_diffs)
        return 1.0 / (1.0 + time_variance)
    
    def correct_position(self, detection_result: DetectionResult, 
                        current_position: np.ndarray, 
                        current_velocity: np.ndarray) -> np.ndarray:
        """Apply appropriate correction based on detected spoofing type."""
        if not detection_result.detected:
            return current_position
            
        if detection_result.attack_type == SpoofingType.GRADUAL_DRIFT:
            return self._correct_gradual_drift(current_position, current_velocity)
        elif detection_result.attack_type == SpoofingType.SUDDEN_JUMP:
            return self._correct_sudden_jump(current_position, current_velocity)
        elif detection_result.attack_type == SpoofingType.RANDOM_WALK:
            return self._correct_random_walk(current_position, current_velocity)
        elif detection_result.attack_type == SpoofingType.REPLAY:
            return self._correct_replay(current_position, current_velocity)
        
        return current_position
    
    def _correct_gradual_drift(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Correct gradual drift by estimating and removing the drift component."""
        if len(self.position_history) < 2:
            return position
            
        # Estimate drift direction
        positions = np.array(self.position_history)
        drift_direction = np.mean(np.diff(positions, axis=0), axis=0)
        drift_magnitude = np.linalg.norm(drift_direction)
        
        if drift_magnitude > 0:
            drift_direction = drift_direction / drift_magnitude
            # Remove drift component
            corrected = position - drift_direction * drift_magnitude
            return corrected
        
        return position
    
    def _correct_sudden_jump(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Correct sudden jumps by using velocity-based prediction."""
        if len(self.position_history) < 2:
            return position
            
        # Use last valid position and velocity to predict
        last_position = self.position_history[-1]
        last_velocity = self.velocity_history[-1]
        
        # Predict position based on velocity
        predicted = last_position + last_velocity * (self.timestamp_history[-1] - self.timestamp_history[-2])
        
        # If prediction is close to current position, use it
        if np.linalg.norm(predicted - position) < self.sudden_jump_threshold:
            return predicted
        
        return position
    
    def _correct_random_walk(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Correct random walk using low-pass filtering."""
        if len(self.position_history) < 2:
            return position
            
        # Apply simple moving average
        positions = np.array(self.position_history)
        weights = np.exp(-np.arange(len(positions)) / 5.0)  # Exponential weights
        weights = weights / np.sum(weights)
        
        # Calculate weighted average
        corrected = np.sum(positions * weights[:, np.newaxis], axis=0)
        return corrected
    
    def _correct_replay(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Correct replay attacks using temporal validation."""
        if len(self.position_history) < 2:
            return position
            
        # Use velocity-based prediction
        last_position = self.position_history[-1]
        last_velocity = self.velocity_history[-1]
        time_diff = self.timestamp_history[-1] - self.timestamp_history[-2]
        
        # Predict next position
        predicted = last_position + last_velocity * time_diff
        
        # If current position is too different from prediction, use prediction
        if np.linalg.norm(predicted - position) > self.replay_threshold:
            return predicted
        
        return position 