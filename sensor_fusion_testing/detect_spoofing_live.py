#!/usr/bin/env python3
"""
Real-Time GPS Spoofing Detection Demo

Tests trained ML models by applying GPS spoofing attacks and detecting them
in real-time within the CARLA simulator.

Usage:
    python detect_spoofing_live.py [--model-dir trained_models] [--duration 60] [--no-display]
"""

import sys
import os
import glob
import time
import argparse
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List
from collections import deque

# Add CARLA to path
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from integration_files.gps_spoofer import GPSSpoofer, SpoofingStrategy
from integration_files.advanced_kalman_filter import AdvancedKalmanFilter
from ml_models.ensemble import EnsembleVoting
import joblib

# Try to import pygame for visualization
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


class LiveSpoofingDetector:
    """
    Real-time GPS spoofing detection using trained ML models.
    """
    
    def __init__(
        self,
        model_dir: str = "trained_models",
        enable_display: bool = True,
        attack_start_delay: float = 10.0
    ):
        """
        Initialize the live detector.
        
        Args:
            model_dir: Directory containing trained models
            enable_display: Enable pygame visualization
            attack_start_delay: Seconds before starting attack (allows baseline)
        """
        self.model_dir = model_dir
        self.enable_display = enable_display and HAS_PYGAME
        self.attack_start_delay = attack_start_delay
        
        # Load trained models
        print(f"[INFO] Loading models from {model_dir}...")
        self.ensemble = EnsembleVoting.load(os.path.join(model_dir, 'ensemble.pkl'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        print(f"[INFO] Loaded ensemble with {len(self.ensemble.models)} models")
        
        # CARLA connection
        self.client = None
        self.world = None
        self.vehicle = None
        self.gps_sensor_true = None
        self.gps_sensor_spoof = None
        self.imu_sensor = None
        
        # GPS Spoofer
        self.spoofer = GPSSpoofer(
            [0, 0, 0],
            strategy=SpoofingStrategy.INNOVATION_AWARE_GRADUAL_DRIFT
        )
        self.current_innovation = 0.0
        
        # Dual Kalman filters
        self.kf_true = AdvancedKalmanFilter()
        self.kf_spoof = AdvancedKalmanFilter()
        self.kf_initialized = False
        
        # Data buffers
        self.imu_buffer = deque(maxlen=100)  # Store last 100 IMU samples
        self.data_buffer = deque(maxlen=10)   # Store last 10 data points for rolling stats
        
        # Timing
        self.t0 = None
        self.start_time = None
        self.attack_active = False
        
        # Detection tracking
        self.detection_history = []
        self.last_detection_time = 0
        self.detection_count = 0
        
        # Previous values
        self.prev_position = None
        self.prev_timestamp = None
        
        # Pygame display
        if self.enable_display:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("GPS Spoofing Detection - Live Demo")
            self.font = pygame.font.Font(None, 48)
            self.small_font = pygame.font.Font(None, 24)
            self.clock = pygame.time.Clock()
            self.alert_active = False
        
    def connect_to_carla(self, host: str = 'localhost', port: int = 2000):
        """Connect to CARLA and spawn vehicle."""
        print(f"[INFO] Connecting to CARLA at {host}:{port}...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Spawn vehicle
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.find('vehicle.tesla.model3')
        spawn_points = self.world.get_map().get_spawn_points()
        spawn_point = spawn_points[0] if spawn_points else carla.Transform()
        
        self.vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
        if not self.vehicle:
            raise RuntimeError("Failed to spawn vehicle")
            
        print(f"[INFO] Spawned vehicle at ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f}, {spawn_point.location.z:.1f})")
        
        # Enable autopilot
        self.vehicle.set_autopilot(True)
        
    def setup_sensors(self):
        """Setup dual GPS sensors and IMU sensor."""
        bp_lib = self.world.get_blueprint_library()
        
        # GPS Sensor 1: True readings
        gps_bp_true = bp_lib.find('sensor.other.gnss')
        gps_bp_true.set_attribute('sensor_tick', '0.10')  # 10 Hz
        self.gps_sensor_true = self.world.spawn_actor(
            gps_bp_true,
            carla.Transform(carla.Location(x=0.0, z=2.0)),
            attach_to=self.vehicle
        )
        self.gps_sensor_true.listen(self._gps_callback)
        
        # IMU Sensor
        imu_bp = bp_lib.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', '0.02')  # 50 Hz
        self.imu_sensor = self.world.spawn_actor(
            imu_bp,
            carla.Transform(carla.Location(x=0.0, z=2.0)),
            attach_to=self.vehicle
        )
        self.imu_sensor.listen(self._imu_callback)
        
        print("[INFO] Sensors initialized")
        
    def _set_t0_if_needed(self, sensor_timestamp: float):
        """Set the time origin on first sensor reading."""
        if self.t0 is None:
            self.t0 = sensor_timestamp
            self.start_time = time.time()
            
    def _imu_callback(self, data):
        """IMU callback - collect raw IMU data."""
        self._set_t0_if_needed(data.timestamp)
        t = data.timestamp - self.t0
        
        imu_sample = {
            'timestamp': t,
            'accel_x': data.accelerometer.x,
            'accel_y': data.accelerometer.y,
            'accel_z': data.accelerometer.z,
            'gyro_x': data.gyroscope.x,
            'gyro_y': data.gyroscope.y,
            'gyro_z': data.gyroscope.z,
            'compass': data.compass
        }
        self.imu_buffer.append(imu_sample)
        
        # Update Kalman filter predictions
        if self.kf_initialized:
            try:
                self.kf_true.predict(data, timestamp=t)
                self.kf_spoof.predict(data, timestamp=t)
            except Exception:
                pass
                
    def _gps_callback(self, data):
        """GPS callback - main processing happens here."""
        self._set_t0_if_needed(data.timestamp)
        t = data.timestamp - self.t0
        
        # Get true position from vehicle transform
        tf = self.vehicle.get_transform()
        true_pos = np.array([tf.location.x, tf.location.y, tf.location.z])
        
        # Determine if attack should be active
        self.attack_active = t >= self.attack_start_delay
        
        # Apply spoofing if attack is active
        if self.attack_active:
            spoofed_pos = self.spoofer.spoof_position(true_pos, self.current_innovation)
        else:
            spoofed_pos = true_pos.copy()
            
        # Initialize Kalman filters on first GPS reading
        if not self.kf_initialized:
            try:
                self.kf_true.position = true_pos.copy()
                self.kf_spoof.position = true_pos.copy()
                self.kf_initialized = True
            except Exception:
                pass
                
        # Update Kalman filters with GPS measurements
        innovation_true = 0.0
        innovation_spoof = 0.0
        kf_true_pos = true_pos.copy()
        kf_spoof_pos = true_pos.copy()
        
        try:
            innovation_true = self.kf_true.update_with_gps(true_pos)
            kf_true_state = self.kf_true.get_state()
            kf_true_pos = kf_true_state['position']
        except Exception:
            pass
            
        try:
            innovation_spoof = self.kf_spoof.update_with_gps(spoofed_pos)
            kf_spoof_state = self.kf_spoof.get_state()
            kf_spoof_pos = kf_spoof_state['position']
            self.current_innovation = innovation_spoof
        except Exception:
            pass
            
        # Interpolate IMU data
        imu_interp = self._interpolate_imu(t)
        
        # Calculate velocity
        velocity_magnitude = 0.0
        if self.prev_position is not None and self.prev_timestamp is not None:
            dt = t - self.prev_timestamp
            if dt > 0:
                displacement = true_pos - self.prev_position
                velocity_magnitude = np.linalg.norm(displacement) / dt
                
        self.prev_position = true_pos.copy()
        self.prev_timestamp = t
        
        # Calculate derived metrics
        position_error = np.linalg.norm(true_pos - spoofed_pos)
        kf_tracking_error = np.linalg.norm(true_pos - kf_spoof_pos)
        
        # Store data point
        data_point = {
            'timestamp': t,
            'true_gps_x': true_pos[0],
            'true_gps_y': true_pos[1],
            'true_gps_z': true_pos[2],
            'spoofed_gps_x': spoofed_pos[0],
            'spoofed_gps_y': spoofed_pos[1],
            'spoofed_gps_z': spoofed_pos[2],
            'imu_accel_x': imu_interp['accel_x'],
            'imu_accel_y': imu_interp['accel_y'],
            'imu_accel_z': imu_interp['accel_z'],
            'imu_gyro_x': imu_interp['gyro_x'],
            'imu_gyro_y': imu_interp['gyro_y'],
            'imu_gyro_z': imu_interp['gyro_z'],
            'kf_true_x': float(kf_true_pos[0]),
            'kf_true_y': float(kf_true_pos[1]),
            'kf_true_z': float(kf_true_pos[2]),
            'kf_spoof_x': float(kf_spoof_pos[0]),
            'kf_spoof_y': float(kf_spoof_pos[1]),
            'kf_spoof_z': float(kf_spoof_pos[2]),
            'innovation_true': float(innovation_true),
            'innovation_spoof': float(innovation_spoof),
            'position_error': float(position_error),
            'kf_tracking_error': float(kf_tracking_error),
            'velocity_magnitude': float(velocity_magnitude),
            'is_attack_active': 1 if self.attack_active else 0
        }
        self.data_buffer.append(data_point)
        
        # Run detection every second
        if t - self.last_detection_time >= 1.0:
            self._run_detection()
            self.last_detection_time = t
        
    def _interpolate_imu(self, gps_timestamp: float) -> Dict[str, float]:
        """Interpolate IMU data to GPS timestamp."""
        if len(self.imu_buffer) < 2:
            return {
                'accel_x': 0.0, 'accel_y': 0.0, 'accel_z': 0.0,
                'gyro_x': 0.0, 'gyro_y': 0.0, 'gyro_z': 0.0
            }
            
        # Find closest IMU samples
        imu_list = list(self.imu_buffer)
        timestamps = [s['timestamp'] for s in imu_list]
        
        # Find surrounding samples
        idx = np.searchsorted(timestamps, gps_timestamp)
        if idx == 0:
            return {k: imu_list[0][k] for k in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']}
        if idx >= len(timestamps):
            return {k: imu_list[-1][k] for k in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']}
            
        # Linear interpolation
        t0 = timestamps[idx-1]
        t1 = timestamps[idx]
        alpha = (gps_timestamp - t0) / (t1 - t0) if t1 > t0 else 0
        
        result = {}
        for key in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']:
            v0 = imu_list[idx-1][key]
            v1 = imu_list[idx][key]
            result[key] = v0 + alpha * (v1 - v0)
            
        return result
        
    def _run_detection(self):
        """Run ML detection on current data."""
        if len(self.data_buffer) < 10:
            return  # Need minimum samples for rolling stats
            
        features = self._calculate_features()
        if features is None:
            return
            
        # Normalize features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get prediction from ensemble
        prediction = self.ensemble.predict(features_scaled)[0]
        
        # Get individual model predictions
        individual_preds = self.ensemble.get_individual_predictions(features_scaled)
        
        # Get anomaly scores
        scores = {}
        for name, model in self.ensemble.models.items():
            score = model.score_samples(features_scaled)[0]
            scores[name] = score
        
        # Is attack detected?
        is_attack_detected = prediction == -1
        
        # Get current state
        current_data = self.data_buffer[-1]
        is_actually_attacking = current_data['is_attack_active'] == 1
        
        # Update alert status for display
        if self.enable_display:
            self.alert_active = is_attack_detected
        
        # Track detection
        if is_attack_detected:
            self.detection_count += 1
            
        self.detection_history.append({
            'timestamp': current_data['timestamp'],
            'detected': is_attack_detected,
            'actual_attack': is_actually_attacking,
            'position_error': current_data['position_error'],
            'innovation_spoof': current_data['innovation_spoof']
        })
        
        # Print detection results
        self._print_detection_result(
            current_data['timestamp'],
            is_attack_detected,
            is_actually_attacking,
            individual_preds,
            scores,
            current_data
        )
        
    def _calculate_features(self) -> Optional[np.ndarray]:
        """
        Calculate features from buffered data (same as training).
        
        Returns:
            Feature vector matching training data format
        """
        if len(self.data_buffer) < 10:
            return None
            
        # Get recent data points
        data_list = list(self.data_buffer)
        
        # Current data point (most recent)
        current = data_list[-1]
        
        # Calculate acceleration magnitude
        accel_magnitude = np.sqrt(
            current['imu_accel_x']**2 + 
            current['imu_accel_y']**2 + 
            current['imu_accel_z']**2
        )
        
        # Calculate gyroscope magnitude
        gyro_magnitude = np.sqrt(
            current['imu_gyro_x']**2 + 
            current['imu_gyro_y']**2 + 
            current['imu_gyro_z']**2
        )
        
        # Calculate jerk (rate of change of acceleration)
        if len(data_list) >= 2:
            prev = data_list[-2]
            dt = current['timestamp'] - prev['timestamp']
            if dt > 0:
                jerk_x = (current['imu_accel_x'] - prev['imu_accel_x']) / dt
                jerk_y = (current['imu_accel_y'] - prev['imu_accel_y']) / dt
                jerk_z = (current['imu_accel_z'] - prev['imu_accel_z']) / dt
                jerk_magnitude = np.sqrt(jerk_x**2 + jerk_y**2 + jerk_z**2)
            else:
                jerk_magnitude = 0.0
        else:
            jerk_magnitude = 0.0
            
        # Calculate rolling statistics
        innovations_spoof = [d['innovation_spoof'] for d in data_list]
        position_errors = [d['position_error'] for d in data_list]
        kf_tracking_errors = [d['kf_tracking_error'] for d in data_list]
        accel_mags = []
        
        for d in data_list:
            am = np.sqrt(d['imu_accel_x']**2 + d['imu_accel_y']**2 + d['imu_accel_z']**2)
            accel_mags.append(am)
        
        # Rolling statistics
        innovation_spoof_ma = np.mean(innovations_spoof)
        innovation_spoof_std = np.std(innovations_spoof)
        innovation_spoof_max = np.max(innovations_spoof)
        
        position_error_ma = np.mean(position_errors)
        position_error_std = np.std(position_errors)
        
        kf_tracking_error_ma = np.mean(kf_tracking_errors)
        kf_tracking_error_std = np.std(kf_tracking_errors)
        
        accel_magnitude_ma = np.mean(accel_mags)
        accel_magnitude_std = np.std(accel_mags)
        
        # Innovation difference
        innovation_diff = current['innovation_spoof'] - current['innovation_true']
        
        # KF difference magnitude
        kf_diff_magnitude = np.sqrt(
            (current['kf_spoof_x'] - current['kf_true_x'])**2 +
            (current['kf_spoof_y'] - current['kf_true_y'])**2 +
            (current['kf_spoof_z'] - current['kf_true_z'])**2
        )
        
        # Feature vector (must match training feature order)
        features = np.array([
            current['innovation_spoof'],
            innovation_spoof_ma,
            innovation_spoof_std,
            innovation_spoof_max,
            current['position_error'],
            position_error_ma,
            position_error_std,
            current['kf_tracking_error'],
            kf_tracking_error_ma,
            kf_tracking_error_std,
            accel_magnitude,
            accel_magnitude_ma,
            accel_magnitude_std,
            gyro_magnitude,
            jerk_magnitude,
            innovation_diff,
            kf_diff_magnitude
        ])
        
        return features
        
    def _print_detection_result(
        self,
        timestamp: float,
        detected: bool,
        actual_attack: bool,
        individual_preds: Dict,
        scores: Dict,
        data: Dict
    ):
        """Print detection result to console."""
        print("\n" + "="*70)
        print(f"DETECTION at t={timestamp:.2f}s")
        print("="*70)
        
        # Detection status
        if detected:
            if actual_attack:
                status = "CORRECTLY DETECTED"
                symbol = "✓"
            else:
                status = "FALSE ALARM"
                symbol = "✗"
        else:
            if actual_attack:
                status = "MISSED ATTACK"
                symbol = "✗"
            else:
                status = "CORRECTLY CLEAN"
                symbol = "✓"
                
        print(f"Status: {status} {symbol}")
        print(f"Ensemble Decision: {'SPOOFED DETECTED!' if detected else 'CLEAN'}")
        print(f"Ground Truth: {'Attack Active' if actual_attack else 'Clean Data'}")
        
        # Key metrics
        print(f"\nKey Metrics:")
        print(f"  Position Error: {data['position_error']:.3f}m")
        print(f"  Innovation (Spoof): {data['innovation_spoof']:.3f}")
        print(f"  KF Tracking Error: {data['kf_tracking_error']:.3f}m")
        
        # Individual model predictions
        print(f"\nIndividual Models:")
        for name, pred in individual_preds.items():
            pred_str = "SPOOFED" if pred == -1 else "CLEAN"
            score = scores.get(name, 0.0)
            print(f"  {name:20s}: {pred_str:8s} (score: {score:+.3f})")
            
        print("="*70)
        
    def update_display(self):
        """Update pygame display with detection status."""
        if not self.enable_display:
            return
            
        self.screen.fill((20, 20, 30))  # Dark background
        
        # Title
        title = self.font.render("GPS Spoofing Detection", True, (255, 255, 255))
        self.screen.blit(title, (20, 20))
        
        # Status
        y_offset = 100
        if self.alert_active:
            status_text = "STATUS: ATTACK DETECTED!"
            status_color = (255, 50, 50)  # Red
        else:
            status_text = "STATUS: NORMAL"
            status_color = (50, 255, 50)  # Green
            
        status = self.font.render(status_text, True, status_color)
        self.screen.blit(status, (20, y_offset))
        
        # Detection count
        y_offset += 80
        count_text = self.small_font.render(
            f"Total Detections: {self.detection_count}",
            True, (200, 200, 200)
        )
        self.screen.blit(count_text, (20, y_offset))
        
        # Recent history
        y_offset += 60
        history_text = self.small_font.render("Recent History:", True, (200, 200, 200))
        self.screen.blit(history_text, (20, y_offset))
        
        for i, detection in enumerate(self.detection_history[-10:]):
            y_offset += 30
            t = detection['timestamp']
            detected = detection['detected']
            actual = detection['actual_attack']
            
            if detected and actual:
                text = f"  {t:.1f}s: DETECTED (Correct)"
                color = (100, 255, 100)
            elif detected and not actual:
                text = f"  {t:.1f}s: DETECTED (False Alarm)"
                color = (255, 200, 100)
            elif not detected and actual:
                text = f"  {t:.1f}s: MISSED"
                color = (255, 100, 100)
            else:
                text = f"  {t:.1f}s: Clean (Correct)"
                color = (150, 150, 150)
                
            line = self.small_font.render(text, True, color)
            self.screen.blit(line, (20, y_offset))
        
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS
        
    def run_demo(self, duration: float = 60.0):
        """
        Run live detection demo.
        
        Args:
            duration: Demo duration in seconds
        """
        print("\n" + "="*70)
        print("LIVE GPS SPOOFING DETECTION DEMO")
        print("="*70)
        print(f"Duration: {duration}s")
        print(f"Attack delay: {self.attack_start_delay}s")
        print(f"Attack type: INNOVATION_AWARE_GRADUAL_DRIFT")
        print(f"Loaded models: {len(self.ensemble.models)}")
        print("="*70 + "\n")
        
        # Setup sensors
        self.setup_sensors()
        
        print(f"[INFO] Collecting baseline data for {self.attack_start_delay}s...")
        print(f"[INFO] Attack will start at t={self.attack_start_delay}s")
        print(f"[INFO] Running detection every 1 second")
        print(f"[INFO] Press Ctrl+C to stop early\n")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # Update display
                if self.enable_display:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("\n[INFO] Demo stopped by user")
                            return
                    self.update_display()
                    
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n[INFO] Demo interrupted by user")
            
        # Print summary
        print("\n" + "="*70)
        print("DEMO COMPLETE - SUMMARY")
        print("="*70)
        print(f"Total runtime: {time.time() - start_time:.1f}s")
        print(f"Total detections: {self.detection_count}")
        
        if len(self.detection_history) > 0:
            # Calculate accuracy
            correct = sum(1 for d in self.detection_history 
                         if d['detected'] == d['actual_attack'])
            accuracy = correct / len(self.detection_history) * 100
            
            # Count detection types
            true_positives = sum(1 for d in self.detection_history 
                                if d['detected'] and d['actual_attack'])
            false_positives = sum(1 for d in self.detection_history 
                                 if d['detected'] and not d['actual_attack'])
            true_negatives = sum(1 for d in self.detection_history 
                                if not d['detected'] and not d['actual_attack'])
            false_negatives = sum(1 for d in self.detection_history 
                                 if not d['detected'] and d['actual_attack'])
            
            print(f"\nAccuracy: {accuracy:.1f}%")
            print(f"True Positives: {true_positives}")
            print(f"False Positives: {false_positives}")
            print(f"True Negatives: {true_negatives}")
            print(f"False Negatives: {false_negatives}")
            
            if true_positives + false_negatives > 0:
                recall = true_positives / (true_positives + false_negatives) * 100
                print(f"Detection Rate (Recall): {recall:.1f}%")
                
            if true_positives + false_positives > 0:
                precision = true_positives / (true_positives + false_positives) * 100
                print(f"Precision: {precision:.1f}%")
        
        print("="*70 + "\n")
        
    def cleanup(self):
        """Cleanup sensors and display."""
        print("\n[INFO] Cleaning up...")
        if self.gps_sensor_true:
            self.gps_sensor_true.destroy()
        if self.imu_sensor:
            self.imu_sensor.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        if self.enable_display:
            pygame.quit()
        print("[INFO] Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description="Live GPS spoofing detection demo"
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default='trained_models',
        help='Directory containing trained models (default: trained_models)'
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=60.0,
        help='Demo duration in seconds (default: 60)'
    )
    parser.add_argument(
        '--attack-delay',
        type=float,
        default=10.0,
        help='Seconds before starting attack (default: 10)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable pygame visualization'
    )
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='CARLA server host (default: localhost)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=2000,
        help='CARLA server port (default: 2000)'
    )
    
    args = parser.parse_args()
    
    # Create detector
    detector = LiveSpoofingDetector(
        model_dir=args.model_dir,
        enable_display=not args.no_display,
        attack_start_delay=args.attack_delay
    )
    
    try:
        # Connect and run
        detector.connect_to_carla(args.host, args.port)
        detector.run_demo(args.duration)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        detector.cleanup()


if __name__ == '__main__':
    main()
