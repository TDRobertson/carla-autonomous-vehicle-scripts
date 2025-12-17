#!/usr/bin/env python3
"""
Real-Time GPS Spoofing Detection

Runs trained ML models to detect GPS spoofing attacks in real-time within the
CARLA simulator. Applies spoofing attacks and evaluates detection performance.

Supports two detection modes:
    - supervised: Uses original one-class classifiers (trained_models/)
    - unsupervised: Uses pure unsupervised ensemble (trained_models_unsupervised/)

Supports chaotic attack scheduling:
    - Random on/off attack windows with configurable durations
    - Strength modulation during attacks
    - Full logging to CSV + JSON summary

Usage:
    # Basic unsupervised mode (default)
    python detect_spoofing_live.py --mode unsupervised --duration 120
    
    # Chaotic mode with random attack scheduling
    python detect_spoofing_live.py --chaotic --duration 300
    
    # Chaotic mode with custom parameters
    python detect_spoofing_live.py --chaotic --min-attack 15 --max-attack 60 --seed 42
    
    # With logging output
    python detect_spoofing_live.py --chaotic --output-dir results/live_runs --run-label test1
"""

import sys
import os
import glob
import time
import argparse
import json
import csv
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
    
    Supports both supervised (original) and unsupervised (new) detection modes.
    Supports chaotic attack scheduling with random on/off windows and strength modulation.
    """
    
    def __init__(
        self,
        model_dir: str = "trained_models",
        mode: str = "unsupervised",
        enable_display: bool = True,
        attack_start_delay: float = 10.0,
        use_smoothing: bool = True,
        # Chaotic mode parameters
        chaotic_mode: bool = False,
        min_clean_s: float = 5.0,
        max_clean_s: float = 20.0,
        min_attack_s: float = 10.0,
        max_attack_s: float = 40.0,
        strength_min: float = 0.3,
        strength_max: float = 1.5,
        strength_hold_s: float = 5.0,
        chaos_seed: Optional[int] = None,
        # Logging parameters
        output_dir: Optional[str] = None,
        run_label: str = "run"
    ):
        """
        Initialize the live detector.
        
        Args:
            model_dir: Directory containing trained models
            mode: Detection mode - 'supervised' or 'unsupervised'
            enable_display: Enable pygame visualization
            attack_start_delay: Seconds before starting attack (allows baseline)
            use_smoothing: Use temporal smoothing for unsupervised mode
            chaotic_mode: Enable chaotic attack scheduling
            min_clean_s: Minimum duration of clean (no attack) windows
            max_clean_s: Maximum duration of clean windows
            min_attack_s: Minimum duration of attack windows
            max_attack_s: Maximum duration of attack windows
            strength_min: Minimum strength multiplier during attacks
            strength_max: Maximum strength multiplier during attacks
            strength_hold_s: How long to hold a strength level before changing
            chaos_seed: Optional RNG seed for reproducibility
            output_dir: Directory for logging output (None to disable logging)
            run_label: Label prefix for output files
        """
        self.model_dir = model_dir
        self.mode = mode.lower()
        self.enable_display = enable_display and HAS_PYGAME
        self.attack_start_delay = attack_start_delay
        self.use_smoothing = use_smoothing
        
        # Chaotic mode settings
        self.chaotic_mode = chaotic_mode
        self.min_clean_s = min_clean_s
        self.max_clean_s = max_clean_s
        self.min_attack_s = min_attack_s
        self.max_attack_s = max_attack_s
        self.strength_min = strength_min
        self.strength_max = strength_max
        self.strength_hold_s = strength_hold_s
        self.chaos_seed = chaos_seed
        
        # Logging settings
        self.output_dir = output_dir
        self.run_label = run_label
        self.log_data = []  # Stores per-detection records for CSV
        
        # Load appropriate models based on mode
        self._load_models()
        
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
        
        # Enable chaotic mode if requested
        if self.chaotic_mode:
            self.spoofer.enable_chaotic_mode(
                min_clean_s=min_clean_s,
                max_clean_s=max_clean_s,
                min_attack_s=min_attack_s,
                max_attack_s=max_attack_s,
                strength_min=strength_min,
                strength_max=strength_max,
                strength_hold_s=strength_hold_s,
                seed=chaos_seed
            )
        
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
            mode_str = "Unsupervised" if self.mode == "unsupervised" else "Supervised"
            chaotic_str = " [CHAOTIC]" if self.chaotic_mode else ""
            pygame.display.set_caption(f"GPS Spoofing Detection - {mode_str} Mode{chaotic_str}")
            self.font = pygame.font.Font(None, 48)
            self.small_font = pygame.font.Font(None, 24)
            self.clock = pygame.time.Clock()
            self.alert_active = False
            
    def _load_models(self):
        """Load models based on detection mode."""
        print(f"\n[INFO] Loading models in {self.mode.upper()} mode from {self.model_dir}...")
        
        if self.mode == "unsupervised":
            # Load unsupervised ensemble
            from ml_models.unsupervised_ensemble import UnsupervisedEnsemble
            
            ensemble_path = os.path.join(self.model_dir, 'unsup_ensemble.pkl')
            if not os.path.exists(ensemble_path):
                raise FileNotFoundError(
                    f"Unsupervised ensemble not found at {ensemble_path}. "
                    f"Run train_unsupervised_ensemble.py first."
                )
                
            self.ensemble = UnsupervisedEnsemble.load(ensemble_path)
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
            
            summary = self.ensemble.get_summary()
            print(f"[INFO] Loaded unsupervised ensemble:")
            print(f"       Models: {summary['model_names']}")
            print(f"       Voting: {summary['voting_threshold']} (majority)")
            print(f"       Smoothing: {summary['smoothing_required']}/{summary['smoothing_window']}")
            for name, thresh in summary['thresholds'].items():
                print(f"       {name} threshold: {thresh:.4f}")
                
        else:
            # Load supervised ensemble (original)
            from ml_models.ensemble import EnsembleVoting
            
            ensemble_path = os.path.join(self.model_dir, 'ensemble.pkl')
            if not os.path.exists(ensemble_path):
                raise FileNotFoundError(
                    f"Supervised ensemble not found at {ensemble_path}. "
                    f"Run train_models.py first."
                )
                
            self.ensemble = EnsembleVoting.load(ensemble_path)
            self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
            print(f"[INFO] Loaded supervised ensemble with {len(self.ensemble.models)} models")
        
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
        if self.chaotic_mode:
            # In chaotic mode, the spoofer manages attack scheduling internally
            # But we still respect the initial attack_start_delay as a baseline period
            if t < self.attack_start_delay:
                spoofed_pos = true_pos.copy()
                self.attack_active = False
            else:
                # Let the spoofer handle chaotic scheduling
                spoofed_pos = self.spoofer.spoof_position(
                    true_pos, 
                    self.current_innovation, 
                    elapsed_time=t - self.attack_start_delay
                )
                self.attack_active = self.spoofer.is_chaotic_attack_active()
        else:
            # Non-chaotic mode: simple delay-based attack
            self.attack_active = t >= self.attack_start_delay
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
        
        # Get detection based on mode
        if self.mode == "unsupervised":
            is_attack_detected, info = self.ensemble.predict_single(
                features_scaled[0],
                use_smoothing=self.use_smoothing
            )
            individual_preds = info['votes']
            scores = info['scores']
        else:
            # Supervised mode
            prediction = self.ensemble.predict(features_scaled)[0]
            is_attack_detected = prediction == -1
            individual_preds = {}
            scores = {}
            for name, model in self.ensemble.models.items():
                pred = model.predict(features_scaled)[0]
                individual_preds[name] = pred == -1
                try:
                    scores[name] = model.score_samples(features_scaled)[0]
                except Exception:
                    scores[name] = 0.0
        
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
        
        # Get chaotic state info if in chaotic mode
        chaotic_state = self.spoofer.get_chaotic_state() if self.chaotic_mode else {}
        
        # Log data for CSV output
        log_record = {
            'timestamp': current_data['timestamp'],
            'is_attack_active': 1 if is_actually_attacking else 0,
            'detected': 1 if is_attack_detected else 0,
            'position_error': current_data['position_error'],
            'innovation_spoof': current_data['innovation_spoof'],
            'kf_tracking_error': current_data['kf_tracking_error'],
        }
        
        # Add per-model scores and votes
        for name, score in scores.items():
            log_record[f'score_{name}'] = score
        for name, vote in individual_preds.items():
            vote_val = vote if isinstance(vote, bool) else (vote == -1)
            log_record[f'vote_{name}'] = 1 if vote_val else 0
            
        # Add chaotic mode info if enabled
        if self.chaotic_mode:
            log_record['chaotic_strength'] = chaotic_state.get('strength', 1.0)
            
        self.log_data.append(log_record)
        
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
        
        Must match PRIMARY_FEATURES from data_loader.py exactly:
            'innovation_spoof', 'innovation_spoof_ma', 'innovation_spoof_std',
            'innovation_spoof_max', 'position_error', 'kf_tracking_error',
            'accel_magnitude', 'gyro_magnitude', 'jerk_magnitude',
            'position_error_ma', 'position_error_std', 'kf_tracking_error_ma',
            'kf_tracking_error_std', 'innovation_diff', 'kf_diff_magnitude'
        
        Returns:
            Feature vector matching training data format (15 features)
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
        
        # Rolling statistics
        innovation_spoof_ma = np.mean(innovations_spoof)
        innovation_spoof_std = np.std(innovations_spoof)
        innovation_spoof_max = np.max(innovations_spoof)
        
        position_error_ma = np.mean(position_errors)
        position_error_std = np.std(position_errors)
        
        kf_tracking_error_ma = np.mean(kf_tracking_errors)
        kf_tracking_error_std = np.std(kf_tracking_errors)
        
        # Innovation difference
        innovation_diff = current['innovation_spoof'] - current['innovation_true']
        
        # KF difference magnitude
        kf_diff_magnitude = np.sqrt(
            (current['kf_spoof_x'] - current['kf_true_x'])**2 +
            (current['kf_spoof_y'] - current['kf_true_y'])**2 +
            (current['kf_spoof_z'] - current['kf_true_z'])**2
        )
        
        # Feature vector - MUST match PRIMARY_FEATURES order from data_loader.py (15 features)
        features = np.array([
            current['innovation_spoof'],      # 1
            innovation_spoof_ma,              # 2
            innovation_spoof_std,             # 3
            innovation_spoof_max,             # 4
            current['position_error'],        # 5
            current['kf_tracking_error'],     # 6
            accel_magnitude,                  # 7
            gyro_magnitude,                   # 8
            jerk_magnitude,                   # 9
            position_error_ma,                # 10
            position_error_std,               # 11
            kf_tracking_error_ma,             # 12
            kf_tracking_error_std,            # 13
            innovation_diff,                  # 14
            kf_diff_magnitude                 # 15
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
        mode_str = "UNSUPERVISED" if self.mode == "unsupervised" else "SUPERVISED"
        print(f"DETECTION [{mode_str}] at t={timestamp:.2f}s")
        print("="*70)
        
        # Detection status
        if detected:
            if actual_attack:
                status = "TRUE POSITIVE (Attack Detected)"
                symbol = "[TP]"
            else:
                status = "FALSE POSITIVE (False Alarm)"
                symbol = "[FP]"
        else:
            if actual_attack:
                status = "FALSE NEGATIVE (Missed Attack)"
                symbol = "[FN]"
            else:
                status = "TRUE NEGATIVE (Correctly Clean)"
                symbol = "[TN]"
                
        print(f"Result: {symbol} {status}")
        print(f"Detection: {'SPOOFING DETECTED!' if detected else 'CLEAN'}")
        print(f"Ground Truth: {'ATTACK ACTIVE' if actual_attack else 'CLEAN DATA'}")
        
        # Key metrics
        print(f"\nSensor Metrics:")
        print(f"  Position Error: {data['position_error']:.3f}m")
        print(f"  Innovation (Spoof): {data['innovation_spoof']:.3f}")
        print(f"  KF Tracking Error: {data['kf_tracking_error']:.3f}m")
        
        # Individual model votes
        print(f"\nModel Votes:")
        for name, vote in individual_preds.items():
            if self.mode == "unsupervised":
                # Unsupervised: vote is boolean
                vote_str = "ANOMALY" if vote else "NORMAL"
                score = scores.get(name, 0.0)
                # Get threshold from ensemble
                thresh = self.ensemble.models[name].threshold
                print(f"  {name:15s}: {vote_str:8s} (score: {score:+.4f}, thresh: {thresh:+.4f})")
            else:
                # Supervised: vote is boolean or -1/1
                vote_val = vote if isinstance(vote, bool) else (vote == -1)
                vote_str = "ANOMALY" if vote_val else "NORMAL"
                score = scores.get(name, 0.0)
                print(f"  {name:15s}: {vote_str:8s} (score: {score:+.4f})")
            
        print("="*70)
        
    def update_display(self):
        """Update pygame display with detection status."""
        if not self.enable_display:
            return
            
        self.screen.fill((20, 20, 30))  # Dark background
        
        # Title
        mode_str = "Unsupervised" if self.mode == "unsupervised" else "Supervised"
        title = self.font.render(f"GPS Spoofing Detection ({mode_str})", True, (255, 255, 255))
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
                text = f"  {t:.1f}s: TP - Detected (Correct)"
                color = (100, 255, 100)
            elif detected and not actual:
                text = f"  {t:.1f}s: FP - False Alarm"
                color = (255, 200, 100)
            elif not detected and actual:
                text = f"  {t:.1f}s: FN - Missed"
                color = (255, 100, 100)
            else:
                text = f"  {t:.1f}s: TN - Clean (Correct)"
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
        mode_str = "UNSUPERVISED" if self.mode == "unsupervised" else "SUPERVISED"
        chaotic_str = " [CHAOTIC]" if self.chaotic_mode else ""
        
        print("\n" + "="*70)
        print(f"LIVE GPS SPOOFING DETECTION DEMO - {mode_str} MODE{chaotic_str}")
        print("="*70)
        print(f"Duration: {duration}s")
        print(f"Attack delay: {self.attack_start_delay}s")
        print(f"Attack type: INNOVATION_AWARE_GRADUAL_DRIFT")
        if self.chaotic_mode:
            print(f"Chaotic mode: ENABLED")
            print(f"  Clean windows: {self.min_clean_s}-{self.max_clean_s}s")
            print(f"  Attack windows: {self.min_attack_s}-{self.max_attack_s}s")
            print(f"  Strength range: {self.strength_min}-{self.strength_max}x")
            print(f"  Strength hold: {self.strength_hold_s}s")
            if self.chaos_seed is not None:
                print(f"  RNG seed: {self.chaos_seed}")
        if self.mode == "unsupervised":
            print(f"Temporal smoothing: {'Enabled' if self.use_smoothing else 'Disabled'}")
            summary = self.ensemble.get_summary()
            print(f"Models: {summary['model_names']}")
        else:
            print(f"Models: {len(self.ensemble.models)}")
        if self.output_dir:
            print(f"Logging to: {self.output_dir}")
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
        print(f"DEMO COMPLETE - SUMMARY ({mode_str} MODE)")
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
            
            print(f"\nConfusion Matrix:")
            print(f"  True Positives (TP):  {true_positives}")
            print(f"  False Positives (FP): {false_positives}")
            print(f"  True Negatives (TN):  {true_negatives}")
            print(f"  False Negatives (FN): {false_negatives}")
            
            print(f"\nMetrics:")
            print(f"  Accuracy: {accuracy:.1f}%")
            
            if true_positives + false_negatives > 0:
                recall = true_positives / (true_positives + false_negatives) * 100
                print(f"  Detection Rate (Recall): {recall:.1f}%")
                
            if true_positives + false_positives > 0:
                precision = true_positives / (true_positives + false_positives) * 100
                print(f"  Precision: {precision:.1f}%")
                
            if true_negatives + false_positives > 0:
                fpr = false_positives / (true_negatives + false_positives) * 100
                print(f"  False Positive Rate: {fpr:.1f}%")
        
        print("="*70 + "\n")
        
        # Write logs if output directory specified
        if self.output_dir and len(self.log_data) > 0:
            self._write_logs(duration)
        
    def _write_logs(self, duration: float):
        """
        Write detection logs to CSV and JSON summary.
        
        Args:
            duration: Total run duration in seconds
        """
        # Create output directory if needed
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.run_label}_{timestamp_str}"
        
        csv_path = os.path.join(self.output_dir, f"{base_name}_live_log.csv")
        json_path = os.path.join(self.output_dir, f"{base_name}_summary.json")
        
        # Write CSV
        if len(self.log_data) > 0:
            fieldnames = list(self.log_data[0].keys())
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.log_data)
            print(f"[INFO] Wrote {len(self.log_data)} detection records to {csv_path}")
        
        # Calculate metrics for summary
        true_positives = sum(1 for d in self.detection_history 
                            if d['detected'] and d['actual_attack'])
        false_positives = sum(1 for d in self.detection_history 
                             if d['detected'] and not d['actual_attack'])
        true_negatives = sum(1 for d in self.detection_history 
                            if not d['detected'] and not d['actual_attack'])
        false_negatives = sum(1 for d in self.detection_history 
                             if not d['detected'] and d['actual_attack'])
        
        total = len(self.detection_history)
        accuracy = (true_positives + true_negatives) / total * 100 if total > 0 else 0
        recall = true_positives / (true_positives + false_negatives) * 100 if (true_positives + false_negatives) > 0 else 0
        precision = true_positives / (true_positives + false_positives) * 100 if (true_positives + false_positives) > 0 else 0
        fpr = false_positives / (true_negatives + false_positives) * 100 if (true_negatives + false_positives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Get model thresholds
        thresholds = {}
        if self.mode == "unsupervised":
            summary = self.ensemble.get_summary()
            thresholds = summary.get('thresholds', {})
        
        # Build summary JSON
        summary_data = {
            'run_info': {
                'timestamp': timestamp_str,
                'label': self.run_label,
                'duration_s': duration,
                'mode': self.mode,
                'attack_type': 'INNOVATION_AWARE_GRADUAL_DRIFT',
                'attack_start_delay_s': self.attack_start_delay,
                'temporal_smoothing': self.use_smoothing
            },
            'chaotic_mode': {
                'enabled': self.chaotic_mode,
                'min_clean_s': self.min_clean_s if self.chaotic_mode else None,
                'max_clean_s': self.max_clean_s if self.chaotic_mode else None,
                'min_attack_s': self.min_attack_s if self.chaotic_mode else None,
                'max_attack_s': self.max_attack_s if self.chaotic_mode else None,
                'strength_min': self.strength_min if self.chaotic_mode else None,
                'strength_max': self.strength_max if self.chaotic_mode else None,
                'strength_hold_s': self.strength_hold_s if self.chaotic_mode else None,
                'seed': self.chaos_seed if self.chaotic_mode else None
            },
            'model_info': {
                'model_dir': self.model_dir,
                'thresholds': thresholds
            },
            'confusion_matrix': {
                'true_positives': true_positives,
                'false_positives': false_positives,
                'true_negatives': true_negatives,
                'false_negatives': false_negatives
            },
            'metrics': {
                'accuracy': round(accuracy, 2),
                'recall': round(recall, 2),
                'precision': round(precision, 2),
                'false_positive_rate': round(fpr, 2),
                'f1_score': round(f1, 2),
                'total_detections': self.detection_count,
                'total_samples': total
            },
            'files': {
                'csv_log': csv_path,
                'summary': json_path
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        print(f"[INFO] Wrote summary to {json_path}")
        
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
        description="Live GPS spoofing detection demo with chaotic attack scheduling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run with unsupervised detection
  python detect_spoofing_live.py --duration 120
  
  # Chaotic mode with random attack scheduling
  python detect_spoofing_live.py --chaotic --duration 300
  
  # Chaotic mode with custom parameters and logging
  python detect_spoofing_live.py --chaotic --min-attack 15 --max-attack 60 \\
      --output-dir results/live_runs --run-label experiment1 --seed 42
  
  # Reproducible chaotic run
  python detect_spoofing_live.py --chaotic --seed 12345 --duration 180
        """
    )
    
    # Detection mode
    parser.add_argument(
        '--mode',
        type=str,
        choices=['supervised', 'unsupervised'],
        default='unsupervised',
        help='Detection mode: supervised (original) or unsupervised (new, default)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help='Directory containing trained models (default: auto-select based on mode)'
    )
    
    # Run parameters
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
        help='Seconds before starting attack/chaotic scheduling (default: 10)'
    )
    
    # Chaotic mode parameters
    chaotic_group = parser.add_argument_group('Chaotic Mode Options')
    chaotic_group.add_argument(
        '--chaotic',
        action='store_true',
        help='Enable chaotic attack scheduling with random on/off windows'
    )
    chaotic_group.add_argument(
        '--min-clean',
        type=float,
        default=5.0,
        help='Minimum duration of clean (no attack) windows in seconds (default: 5)'
    )
    chaotic_group.add_argument(
        '--max-clean',
        type=float,
        default=20.0,
        help='Maximum duration of clean windows in seconds (default: 20)'
    )
    chaotic_group.add_argument(
        '--min-attack',
        type=float,
        default=10.0,
        help='Minimum duration of attack windows in seconds (default: 10)'
    )
    chaotic_group.add_argument(
        '--max-attack',
        type=float,
        default=40.0,
        help='Maximum duration of attack windows in seconds (default: 40)'
    )
    chaotic_group.add_argument(
        '--strength-min',
        type=float,
        default=0.3,
        help='Minimum strength multiplier during attacks (default: 0.3)'
    )
    chaotic_group.add_argument(
        '--strength-max',
        type=float,
        default=1.5,
        help='Maximum strength multiplier during attacks (default: 1.5)'
    )
    chaotic_group.add_argument(
        '--strength-hold',
        type=float,
        default=5.0,
        help='How long to hold a strength level before changing in seconds (default: 5)'
    )
    chaotic_group.add_argument(
        '--seed',
        type=int,
        default=None,
        help='RNG seed for reproducible chaotic scheduling (default: random)'
    )
    
    # Logging parameters
    logging_group = parser.add_argument_group('Logging Options')
    logging_group.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory for logging output (default: no logging)'
    )
    logging_group.add_argument(
        '--run-label',
        type=str,
        default='run',
        help='Label prefix for output files (default: run)'
    )
    
    # Display options
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable pygame visualization'
    )
    parser.add_argument(
        '--no-smoothing',
        action='store_true',
        help='Disable temporal smoothing for unsupervised mode'
    )
    
    # CARLA connection
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
    
    # Auto-select model directory based on mode if not specified
    if args.model_dir is None:
        if args.mode == 'unsupervised':
            args.model_dir = 'trained_models_unsupervised'
        else:
            args.model_dir = 'trained_models'
    
    # Create detector
    detector = LiveSpoofingDetector(
        model_dir=args.model_dir,
        mode=args.mode,
        enable_display=not args.no_display,
        attack_start_delay=args.attack_delay,
        use_smoothing=not args.no_smoothing,
        # Chaotic mode
        chaotic_mode=args.chaotic,
        min_clean_s=args.min_clean,
        max_clean_s=args.max_clean,
        min_attack_s=args.min_attack,
        max_attack_s=args.max_attack,
        strength_min=args.strength_min,
        strength_max=args.strength_max,
        strength_hold_s=args.strength_hold,
        chaos_seed=args.seed,
        # Logging
        output_dir=args.output_dir,
        run_label=args.run_label
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
