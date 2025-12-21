#!/usr/bin/env python3
"""
Real-Time GPS Spoofing Detection - GPS+IMU+KF-only Mode

Runs trained ML models to detect GPS spoofing attacks in real-time using ONLY
victim-side sensor data:
    - GPS from GNSS sensor (lat/lon/alt converted to local ENU meters)
    - IMU accelerometer and gyroscope
    - Kalman filter state and innovation statistics
    
NO ground-truth vehicle position (vehicle.get_transform()) is used for features.
Ground truth is only used for:
    - Applying simulated spoofing attacks
    - Logging is_attack_active for evaluation

This makes the detector independent of any "golden" reference dataset.

Supports chaotic attack scheduling:
    - Random on/off attack windows with configurable durations
    - Strength modulation during attacks
    - Full logging to CSV + JSON summary

Usage:
    # Basic run
    python detect_spoofing_live_gps_imu_kf.py --duration 120
    
    # Chaotic mode with random attack scheduling
    python detect_spoofing_live_gps_imu_kf.py --chaotic --duration 300
    
    # With logging
    python detect_spoofing_live_gps_imu_kf.py --chaotic --output-dir results/gps_imu_kf --run-label test1
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
from typing import Optional, Dict, List, Tuple
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
from utils.geo_utils import GNSSToLocalConverter
from ml_models.data_loader_gps_imu_kf import PRIMARY_FEATURES_GPS_IMU_KF
import joblib

# Try to import pygame for visualization
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


class LiveSpoofingDetectorGPSIMUKF:
    """
    Real-time GPS spoofing detection using GPS+IMU+KF-only features.
    
    Key difference from the standard detector:
    - Uses GNSS sensor output (lat/lon/alt) converted to local ENU meters
    - Does NOT use vehicle.get_transform() for feature extraction
    - Innovation vector, norm, NIS, and covariance diagnostics as features
    
    Ground truth (vehicle.get_transform()) is used only for:
    - Applying simulated spoofing attacks to GPS
    - Logging is_attack_active for evaluation
    """
    
    def __init__(
        self,
        model_dir: str = "trained_models_unsupervised_gps_imu_kf",
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
        Initialize the GPS+IMU+KF-only live detector.
        
        Args:
            model_dir: Directory containing trained models
            enable_display: Enable pygame visualization
            attack_start_delay: Seconds before starting attack (allows baseline)
            use_smoothing: Use temporal smoothing
            chaotic_mode: Enable chaotic attack scheduling
            min_clean_s: Minimum duration of clean windows
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
        self.log_data = []
        
        # Load models
        self._load_models()
        
        # CARLA connection
        self.client = None
        self.world = None
        self.vehicle = None
        self.gnss_sensor = None
        self.imu_sensor = None
        
        # GPS Spoofer
        self.spoofer = GPSSpoofer(
            [0, 0, 0],
            strategy=SpoofingStrategy.INNOVATION_AWARE_GRADUAL_DRIFT
        )
        
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
        
        # GNSS to local converter
        self.geo_converter = GNSSToLocalConverter()
        
        # Kalman filter (single filter - no "true" filter since we don't have ground truth)
        self.kf = AdvancedKalmanFilter()
        self.kf_initialized = False
        self.current_innovation_norm = 0.0
        
        # Data buffers
        self.imu_buffer = deque(maxlen=100)
        self.data_buffer = deque(maxlen=10)
        
        # Timing
        self.t0 = None
        self.start_time = None
        self.attack_active = False
        
        # Detection tracking
        self.detection_history = []
        self.last_detection_time = 0
        self.detection_count = 0
        
        # Previous values for derived features
        self.prev_gnss_timestamp = None
        
        # Pygame display
        if self.enable_display:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            chaotic_str = " [CHAOTIC]" if self.chaotic_mode else ""
            pygame.display.set_caption(f"GPS+IMU+KF-only Spoofing Detection{chaotic_str}")
            self.font = pygame.font.Font(None, 48)
            self.small_font = pygame.font.Font(None, 24)
            self.clock = pygame.time.Clock()
            self.alert_active = False
            
    def _load_models(self):
        """Load the GPS+IMU+KF-only trained models."""
        print(f"\n[INFO] Loading GPS+IMU+KF-only models from {self.model_dir}...")
        
        from ml_models.unsupervised_ensemble import UnsupervisedEnsemble
        
        ensemble_path = os.path.join(self.model_dir, 'unsup_ensemble.pkl')
        if not os.path.exists(ensemble_path):
            raise FileNotFoundError(
                f"Ensemble not found at {ensemble_path}. "
                f"Train the GPS+IMU+KF-only model first using:\n"
                f"  python train_unsupervised_ensemble_gps_imu_kf.py --train-dir data/training_gps_imu_kf"
            )
            
        self.ensemble = UnsupervisedEnsemble.load(ensemble_path)
        self.scaler = joblib.load(os.path.join(self.model_dir, 'scaler.pkl'))
        
        # Load config to get feature names
        config_path = os.path.join(self.model_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.feature_names = config.get('feature_names', PRIMARY_FEATURES_GPS_IMU_KF)
        else:
            self.feature_names = PRIMARY_FEATURES_GPS_IMU_KF
        
        summary = self.ensemble.get_summary()
        print(f"[INFO] Loaded GPS+IMU+KF-only ensemble:")
        print(f"       Models: {summary['model_names']}")
        print(f"       Features: {len(self.feature_names)}")
        print(f"       Voting: {summary['voting_threshold']} (majority)")
        print(f"       Smoothing: {summary['smoothing_required']}/{summary['smoothing_window']}")
        
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
            
        print(f"[INFO] Spawned vehicle at ({spawn_point.location.x:.1f}, {spawn_point.location.y:.1f})")
        
        # Enable autopilot
        self.vehicle.set_autopilot(True)
        
    def setup_sensors(self):
        """Setup GNSS and IMU sensors."""
        bp_lib = self.world.get_blueprint_library()
        
        # GNSS Sensor (provides lat/lon/alt)
        gnss_bp = bp_lib.find('sensor.other.gnss')
        gnss_bp.set_attribute('sensor_tick', '0.10')  # 10 Hz
        self.gnss_sensor = self.world.spawn_actor(
            gnss_bp,
            carla.Transform(carla.Location(x=0.0, z=2.0)),
            attach_to=self.vehicle
        )
        self.gnss_sensor.listen(self._gnss_callback)
        
        # IMU Sensor
        imu_bp = bp_lib.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', '0.02')  # 50 Hz
        self.imu_sensor = self.world.spawn_actor(
            imu_bp,
            carla.Transform(carla.Location(x=0.0, z=2.0)),
            attach_to=self.vehicle
        )
        self.imu_sensor.listen(self._imu_callback)
        
        print("[INFO] Sensors initialized (GNSS 10Hz, IMU 50Hz)")
        print("[INFO] Using GNSS sensor output for GPS (no ground truth for features)")
        
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
        
        # Update Kalman filter prediction
        if self.kf_initialized:
            try:
                self.kf.predict(data, timestamp=t)
            except Exception:
                pass
                
    def _gnss_callback(self, data):
        """GNSS callback - main processing happens here."""
        self._set_t0_if_needed(data.timestamp)
        t = data.timestamp - self.t0
        
        # Get GNSS measurement (lat/lon/alt)
        lat = data.latitude
        lon = data.longitude
        alt = data.altitude
        
        # Convert to local ENU meters (reference set on first sample)
        gps_x, gps_y, gps_z = self.geo_converter.convert(lat, lon, alt)
        gps_pos = np.array([gps_x, gps_y, gps_z])
        
        # For spoofing simulation, we need ground truth to know what to spoof
        # This is ONLY for attack simulation, not for features
        tf = self.vehicle.get_transform()
        true_pos_world = np.array([tf.location.x, tf.location.y, tf.location.z])
        
        # Determine if attack should be active
        if self.chaotic_mode:
            if t < self.attack_start_delay:
                spoofed_gps = gps_pos.copy()
                self.attack_active = False
            else:
                # Spoof the GPS position (in local ENU meters)
                spoofed_gps = self.spoofer.spoof_position(
                    gps_pos, 
                    self.current_innovation_norm,
                    elapsed_time=t - self.attack_start_delay
                )
                self.attack_active = self.spoofer.is_chaotic_attack_active()
        else:
            self.attack_active = t >= self.attack_start_delay
            if self.attack_active:
                spoofed_gps = self.spoofer.spoof_position(gps_pos, self.current_innovation_norm)
            else:
                spoofed_gps = gps_pos.copy()
        
        # Initialize Kalman filter on first GPS reading
        if not self.kf_initialized:
            try:
                self.kf.position = spoofed_gps.copy()
                self.kf_initialized = True
            except Exception:
                pass
                
        # Update Kalman filter with (potentially spoofed) GPS and get detailed info
        kf_info = {
            'innovation_vector': np.zeros(3),
            'innovation_norm': 0.0,
            'S_diag': np.ones(3),
            'nis': 0.0,
            'position_pred': spoofed_gps.copy(),
            'position_upd': spoofed_gps.copy(),
            'velocity_upd': np.zeros(3),
            'P_pos_diag': np.ones(3)
        }
        
        try:
            kf_info = self.kf.update_with_gps_detailed(spoofed_gps)
            self.current_innovation_norm = kf_info['innovation_norm']
        except Exception:
            pass
        
        # Interpolate IMU data
        imu_interp = self._interpolate_imu(t)
        
        # Calculate IMU magnitudes
        accel_magnitude = np.sqrt(
            imu_interp['accel_x']**2 + 
            imu_interp['accel_y']**2 + 
            imu_interp['accel_z']**2
        )
        gyro_magnitude = np.sqrt(
            imu_interp['gyro_x']**2 + 
            imu_interp['gyro_y']**2 + 
            imu_interp['gyro_z']**2
        )
        
        # Store data point with GPS+IMU+KF-only features
        data_point = {
            'timestamp': t,
            # GPS position (local ENU from GNSS - potentially spoofed)
            'gps_x': float(spoofed_gps[0]),
            'gps_y': float(spoofed_gps[1]),
            'gps_z': float(spoofed_gps[2]),
            # KF updated state
            'kf_x': float(kf_info['position_upd'][0]),
            'kf_y': float(kf_info['position_upd'][1]),
            'kf_z': float(kf_info['position_upd'][2]),
            'kf_vx': float(kf_info['velocity_upd'][0]),
            'kf_vy': float(kf_info['velocity_upd'][1]),
            'kf_vz': float(kf_info['velocity_upd'][2]),
            # Innovation
            'innov_x': float(kf_info['innovation_vector'][0]),
            'innov_y': float(kf_info['innovation_vector'][1]),
            'innov_z': float(kf_info['innovation_vector'][2]),
            'innov_norm': float(kf_info['innovation_norm']),
            # NIS
            'nis': float(kf_info['nis']),
            # Innovation covariance diagonal
            'S_x': float(kf_info['S_diag'][0]),
            'S_y': float(kf_info['S_diag'][1]),
            'S_z': float(kf_info['S_diag'][2]),
            # IMU
            'accel_magnitude': float(accel_magnitude),
            'gyro_magnitude': float(gyro_magnitude),
            # Ground truth (for evaluation only, NOT for features)
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
            
        imu_list = list(self.imu_buffer)
        timestamps = [s['timestamp'] for s in imu_list]
        
        idx = np.searchsorted(timestamps, gps_timestamp)
        if idx == 0:
            return {k: imu_list[0][k] for k in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']}
        if idx >= len(timestamps):
            return {k: imu_list[-1][k] for k in ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']}
            
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
            return
            
        features = self._calculate_features()
        if features is None:
            return
            
        # Normalize features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get detection
        is_attack_detected, info = self.ensemble.predict_single(
            features_scaled[0],
            use_smoothing=self.use_smoothing
        )
        individual_preds = info['votes']
        scores = info['scores']
        
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
            'innov_norm': current_data['innov_norm'],
            'nis': current_data['nis']
        })
        
        # Get chaotic state info
        chaotic_state = self.spoofer.get_chaotic_state() if self.chaotic_mode else {}
        
        # Log data
        log_record = {
            'timestamp': current_data['timestamp'],
            'is_attack_active': 1 if is_actually_attacking else 0,
            'detected': 1 if is_attack_detected else 0,
            'innov_norm': current_data['innov_norm'],
            'nis': current_data['nis'],
        }
        
        for name, score in scores.items():
            log_record[f'score_{name}'] = score
        for name, vote in individual_preds.items():
            vote_val = vote if isinstance(vote, bool) else (vote == -1)
            log_record[f'vote_{name}'] = 1 if vote_val else 0
            
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
        Calculate GPS+IMU+KF-only features from buffered data.
        
        Returns feature vector matching PRIMARY_FEATURES_GPS_IMU_KF.
        """
        if len(self.data_buffer) < 10:
            return None
            
        data_list = list(self.data_buffer)
        current = data_list[-1]
        
        # Calculate rolling statistics on innovation norm and NIS
        innov_norms = [d['innov_norm'] for d in data_list]
        nis_values = [d['nis'] for d in data_list]
        
        innov_norm_ma = np.mean(innov_norms)
        innov_norm_std = np.std(innov_norms)
        nis_ma = np.mean(nis_values)
        nis_std = np.std(nis_values)
        
        # Build feature vector matching PRIMARY_FEATURES_GPS_IMU_KF order
        features = np.array([
            # GPS position (local ENU)
            current['gps_x'],
            current['gps_y'],
            current['gps_z'],
            # KF state
            current['kf_x'],
            current['kf_y'],
            current['kf_z'],
            current['kf_vx'],
            current['kf_vy'],
            current['kf_vz'],
            # Innovation
            current['innov_x'],
            current['innov_y'],
            current['innov_z'],
            current['innov_norm'],
            # NIS
            current['nis'],
            # Innovation covariance
            current['S_x'],
            current['S_y'],
            current['S_z'],
            # IMU
            current['accel_magnitude'],
            current['gyro_magnitude'],
            # Rolling stats
            innov_norm_ma,
            innov_norm_std,
            nis_ma,
            nis_std,
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
        print(f"DETECTION [GPS+IMU+KF-ONLY] at t={timestamp:.2f}s")
        print("="*70)
        
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
        
        print(f"\nInnovation Statistics:")
        print(f"  Innovation Norm: {data['innov_norm']:.3f} m")
        print(f"  NIS: {data['nis']:.3f}")
        print(f"  Innovation Vector: ({data['innov_x']:.2f}, {data['innov_y']:.2f}, {data['innov_z']:.2f})")
        
        print(f"\nModel Votes:")
        for name, vote in individual_preds.items():
            vote_str = "ANOMALY" if vote else "NORMAL"
            score = scores.get(name, 0.0)
            thresh = self.ensemble.models[name].threshold
            print(f"  {name:15s}: {vote_str:8s} (score: {score:+.4f}, thresh: {thresh:+.4f})")
            
        print("="*70)
        
    def update_display(self):
        """Update pygame display."""
        if not self.enable_display:
            return
            
        self.screen.fill((20, 20, 30))
        
        title = self.font.render("GPS+IMU+KF-only Spoofing Detection", True, (255, 255, 255))
        self.screen.blit(title, (20, 20))
        
        subtitle = self.small_font.render("(No ground-truth reference)", True, (150, 150, 150))
        self.screen.blit(subtitle, (20, 70))
        
        y_offset = 120
        if self.alert_active:
            status_text = "STATUS: ATTACK DETECTED!"
            status_color = (255, 50, 50)
        else:
            status_text = "STATUS: NORMAL"
            status_color = (50, 255, 50)
            
        status = self.font.render(status_text, True, status_color)
        self.screen.blit(status, (20, y_offset))
        
        y_offset += 80
        count_text = self.small_font.render(
            f"Total Detections: {self.detection_count}",
            True, (200, 200, 200)
        )
        self.screen.blit(count_text, (20, y_offset))
        
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
        self.clock.tick(30)
        
    def run_demo(self, duration: float = 60.0):
        """Run live detection demo."""
        chaotic_str = " [CHAOTIC]" if self.chaotic_mode else ""
        
        print("\n" + "="*70)
        print(f"LIVE GPS SPOOFING DETECTION - GPS+IMU+KF-ONLY MODE{chaotic_str}")
        print("="*70)
        print("This detector uses ONLY victim-side sensor data:")
        print("  - GPS from GNSS sensor (lat/lon/alt -> local ENU)")
        print("  - IMU accelerometer and gyroscope")
        print("  - Kalman filter state and innovation statistics")
        print("  - NO ground-truth vehicle position for features")
        print("="*70)
        print(f"Duration: {duration}s")
        print(f"Attack delay: {self.attack_start_delay}s")
        print(f"Attack type: INNOVATION_AWARE_GRADUAL_DRIFT")
        if self.chaotic_mode:
            print(f"Chaotic mode: ENABLED")
            print(f"  Clean windows: {self.min_clean_s}-{self.max_clean_s}s")
            print(f"  Attack windows: {self.min_attack_s}-{self.max_attack_s}s")
            print(f"  Strength range: {self.strength_min}-{self.strength_max}x")
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
        print("DEMO COMPLETE - GPS+IMU+KF-ONLY SUMMARY")
        print("="*70)
        print(f"Total runtime: {time.time() - start_time:.1f}s")
        print(f"Total detections: {self.detection_count}")
        
        if len(self.detection_history) > 0:
            correct = sum(1 for d in self.detection_history 
                         if d['detected'] == d['actual_attack'])
            accuracy = correct / len(self.detection_history) * 100
            
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
        
        # Write logs
        if self.output_dir and len(self.log_data) > 0:
            self._write_logs(duration)
            
    def _write_logs(self, duration: float):
        """Write detection logs to CSV and JSON summary."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.run_label}_{timestamp_str}"
        
        csv_path = os.path.join(self.output_dir, f"{base_name}_gps_imu_kf_log.csv")
        json_path = os.path.join(self.output_dir, f"{base_name}_gps_imu_kf_summary.json")
        
        # Write CSV
        if len(self.log_data) > 0:
            fieldnames = list(self.log_data[0].keys())
            with open(csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.log_data)
            print(f"[INFO] Wrote {len(self.log_data)} detection records to {csv_path}")
        
        # Calculate metrics
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
        
        # Get thresholds
        summary = self.ensemble.get_summary()
        thresholds = summary.get('thresholds', {})
        
        # Build summary
        summary_data = {
            'run_info': {
                'timestamp': timestamp_str,
                'label': self.run_label,
                'duration_s': duration,
                'mode': 'gps_imu_kf_only',
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
                'seed': self.chaos_seed if self.chaotic_mode else None
            },
            'model_info': {
                'model_dir': self.model_dir,
                'feature_count': len(self.feature_names),
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
        if self.gnss_sensor:
            self.gnss_sensor.destroy()
        if self.imu_sensor:
            self.imu_sensor.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        if self.enable_display:
            pygame.quit()
        print("[INFO] Cleanup complete")


def main():
    parser = argparse.ArgumentParser(
        description="Live GPS spoofing detection using GPS+IMU+KF-only features (no ground truth)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run
  python detect_spoofing_live_gps_imu_kf.py --duration 120
  
  # Chaotic mode with random attack scheduling
  python detect_spoofing_live_gps_imu_kf.py --chaotic --duration 300
  
  # With logging
  python detect_spoofing_live_gps_imu_kf.py --chaotic --output-dir results/gps_imu_kf --run-label test1
        """
    )
    
    # Model
    parser.add_argument(
        '--model-dir',
        type=str,
        default='trained_models_unsupervised_gps_imu_kf',
        help='Directory containing trained models (default: trained_models_unsupervised_gps_imu_kf)'
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
        help='Seconds before starting attack (default: 10)'
    )
    
    # Chaotic mode
    chaotic_group = parser.add_argument_group('Chaotic Mode Options')
    chaotic_group.add_argument('--chaotic', action='store_true', help='Enable chaotic attack scheduling')
    chaotic_group.add_argument('--min-clean', type=float, default=5.0, help='Min clean window (s)')
    chaotic_group.add_argument('--max-clean', type=float, default=20.0, help='Max clean window (s)')
    chaotic_group.add_argument('--min-attack', type=float, default=10.0, help='Min attack window (s)')
    chaotic_group.add_argument('--max-attack', type=float, default=40.0, help='Max attack window (s)')
    chaotic_group.add_argument('--strength-min', type=float, default=0.3, help='Min strength multiplier')
    chaotic_group.add_argument('--strength-max', type=float, default=1.5, help='Max strength multiplier')
    chaotic_group.add_argument('--strength-hold', type=float, default=5.0, help='Strength hold time (s)')
    chaotic_group.add_argument('--seed', type=int, default=None, help='RNG seed for reproducibility')
    
    # Logging
    logging_group = parser.add_argument_group('Logging Options')
    logging_group.add_argument('--output-dir', type=str, default=None, help='Directory for log output')
    logging_group.add_argument('--run-label', type=str, default='run', help='Label for output files')
    
    # Display
    parser.add_argument('--no-display', action='store_true', help='Disable pygame visualization')
    parser.add_argument('--no-smoothing', action='store_true', help='Disable temporal smoothing')
    
    # CARLA
    parser.add_argument('--host', type=str, default='localhost', help='CARLA host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA port')
    
    args = parser.parse_args()
    
    # Create detector
    detector = LiveSpoofingDetectorGPSIMUKF(
        model_dir=args.model_dir,
        enable_display=not args.no_display,
        attack_start_delay=args.attack_delay,
        use_smoothing=not args.no_smoothing,
        chaotic_mode=args.chaotic,
        min_clean_s=args.min_clean,
        max_clean_s=args.max_clean,
        min_attack_s=args.min_attack,
        max_attack_s=args.max_attack,
        strength_min=args.strength_min,
        strength_max=args.strength_max,
        strength_hold_s=args.strength_hold,
        chaos_seed=args.seed,
        output_dir=args.output_dir,
        run_label=args.run_label
    )
    
    try:
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

