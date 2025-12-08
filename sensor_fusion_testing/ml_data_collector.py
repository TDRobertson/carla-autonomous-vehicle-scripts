#!/usr/bin/env python3
"""
ML Training Data Collector for GPS Spoofing Detection

This script collects synchronized GPS and IMU data for training machine learning
models to detect GPS spoofing attacks. It uses a dual-sensor approach:
- One GPS sensor provides true (unspoofed) readings
- A second GPS sensor at the same location has spoofing applied

Both sensors fire simultaneously, ensuring perfect timestamp synchronization.
IMU data is interpolated to GPS timestamps for alignment.

Features collected:
- True and spoofed GPS positions
- IMU accelerometer and gyroscope data (interpolated)
- Kalman filter states for both true and spoofed paths
- Innovation values (measurement residuals)
- Derived features: position error, velocity, acceleration magnitude, jerk

Output:
- JSON file with full metadata and raw data
- CSV file for direct ML training with sklearn/pandas

Usage:
    python ml_data_collector.py [--duration 60] [--warmup 5] [--output-dir ml_data]
"""

import sys
import os
import glob
import time
import json
import argparse
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

# Add the CARLA Python API to PYTHONPATH
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
from integration_files.gps_spoofer import GPSSpoofer, SpoofingStrategy
from integration_files.advanced_kalman_filter import AdvancedKalmanFilter
from integration_files.traffic_utils import setup_continuous_traffic, cleanup_traffic_setup


@dataclass
class IMUSample:
    """Raw IMU sample with timestamp."""
    timestamp: float
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    compass: float


@dataclass
class GPSSample:
    """GPS sample with true and spoofed positions."""
    timestamp: float
    true_x: float
    true_y: float
    true_z: float
    spoofed_x: float
    spoofed_y: float
    spoofed_z: float


@dataclass
class CollectedDataPoint:
    """Single synchronized data point for ML training."""
    # Timestamp
    timestamp: float
    
    # True GPS position
    true_gps_x: float
    true_gps_y: float
    true_gps_z: float
    
    # Spoofed GPS position
    spoofed_gps_x: float
    spoofed_gps_y: float
    spoofed_gps_z: float
    
    # IMU data (interpolated to GPS timestamp)
    imu_accel_x: float
    imu_accel_y: float
    imu_accel_z: float
    imu_gyro_x: float
    imu_gyro_y: float
    imu_gyro_z: float
    imu_compass: float
    
    # Kalman filter estimates (true path)
    kf_true_x: float
    kf_true_y: float
    kf_true_z: float
    
    # Kalman filter estimates (spoofed path)
    kf_spoof_x: float
    kf_spoof_y: float
    kf_spoof_z: float
    
    # Innovation values
    innovation_true: float
    innovation_spoof: float
    
    # Derived metrics
    position_error: float  # Euclidean distance: true vs spoofed GPS
    kf_tracking_error: float  # Euclidean distance: true GPS vs KF_spoof
    velocity_magnitude: float  # Derived from position change rate
    
    # Label
    is_attack_active: int  # 1 if spoofing is applied, 0 otherwise


class MLDataCollector:
    """
    Collects synchronized GPS and IMU data for ML training.
    
    Uses dual GPS sensors at the same vehicle location:
    - Sensor 1: True GPS readings (no spoofing)
    - Sensor 2: Spoofed GPS readings (INNOVATION_AWARE_GRADUAL_DRIFT)
    
    IMU data is collected at higher frequency and interpolated to GPS timestamps.
    """
    
    def __init__(
        self,
        vehicle,
        duration: float = 60.0,
        warmup_duration: float = 5.0,
        attack_start_delay: float = 10.0,
        output_dir: str = "data",
        random_attacks: bool = False,
        min_attack_duration: float = 5.0,
        max_attack_duration: float = 15.0,
        min_clean_duration: float = 5.0,
        max_clean_duration: float = 15.0,
        label: str = ""
    ):
        """
        Initialize the ML data collector.
        
        Args:
            vehicle: CARLA vehicle actor
            duration: Total collection duration in seconds
            warmup_duration: Initial warmup period before recording (seconds)
            attack_start_delay: Delay before starting spoofing attack (seconds).
                               Set to 0 to start attack immediately (useful for testing).
            output_dir: Directory to save output files
            random_attacks: If True, randomly start/stop attacks during collection
            min_attack_duration: Minimum duration of attack periods (random mode)
            max_attack_duration: Maximum duration of attack periods (random mode)
            min_clean_duration: Minimum duration of clean periods (random mode)
            max_clean_duration: Maximum duration of clean periods (random mode)
            label: Label prefix for output files (e.g., "train_run01", "val_run05")
        """
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.duration = duration
        self.warmup_duration = warmup_duration
        self.attack_start_delay = attack_start_delay
        self.output_dir = output_dir
        self.random_attacks = random_attacks
        self.min_attack_duration = min_attack_duration
        self.max_attack_duration = max_attack_duration
        self.min_clean_duration = min_clean_duration
        self.max_clean_duration = max_clean_duration
        self.label = label
        
        # Timing
        self.t0: Optional[float] = None  # Simulation time origin
        self.collection_start_time: Optional[float] = None  # Wall clock time when collection starts
        self.collection_start_sim_time: Optional[float] = None  # CARLA simulation time when collection starts
        
        # Sensors
        self.gps_sensor_true: Optional[carla.Actor] = None
        self.gps_sensor_spoof: Optional[carla.Actor] = None
        self.imu_sensor: Optional[carla.Actor] = None
        
        # GPS Spoofer (INNOVATION_AWARE_GRADUAL_DRIFT for stealthy attacks)
        self.spoofer = GPSSpoofer(
            [0, 0, 0],
            strategy=SpoofingStrategy.INNOVATION_AWARE_GRADUAL_DRIFT
        )
        self.current_innovation = 0.0
        
        # Dual Kalman filters
        self.kf_true = AdvancedKalmanFilter()
        self.kf_spoof = AdvancedKalmanFilter()
        self.kf_initialized = False
        
        # Raw data buffers
        self.imu_samples: List[IMUSample] = []
        self.gps_samples: List[GPSSample] = []
        
        # Synchronized data points
        self.data_points: List[CollectedDataPoint] = []
        
        # Previous values for velocity calculation
        self.prev_position: Optional[np.ndarray] = None
        self.prev_timestamp: Optional[float] = None
        
        # State tracking
        self.is_collecting = False
        self.attack_active = False
        
        # Random attack state tracking
        self.next_attack_transition_time: Optional[float] = None
        self.random_seed = int(time.time() * 1000) % 2**31
        np.random.seed(self.random_seed)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def setup_sensors(self):
        """Setup dual GPS sensors and IMU sensor on the vehicle."""
        bp_lib = self.world.get_blueprint_library()
        
        # GPS Sensor 1: True readings (no spoofing applied in callback)
        gps_bp_true = bp_lib.find('sensor.other.gnss')
        gps_bp_true.set_attribute('sensor_tick', '0.10')  # 10 Hz
        self.gps_sensor_true = self.world.spawn_actor(
            gps_bp_true,
            carla.Transform(carla.Location(x=0.0, z=2.0)),
            attach_to=self.vehicle
        )
        self.gps_sensor_true.listen(self._gps_callback_true)
        
        # GPS Sensor 2: For spoofed readings (same location)
        # Note: We use the same physical sensor location but apply spoofing in callback
        gps_bp_spoof = bp_lib.find('sensor.other.gnss')
        gps_bp_spoof.set_attribute('sensor_tick', '0.10')  # 10 Hz
        self.gps_sensor_spoof = self.world.spawn_actor(
            gps_bp_spoof,
            carla.Transform(carla.Location(x=0.0, z=2.0)),
            attach_to=self.vehicle
        )
        self.gps_sensor_spoof.listen(self._gps_callback_spoof)
        
        # IMU Sensor: Higher frequency for interpolation
        imu_bp = bp_lib.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', '0.02')  # 50 Hz
        self.imu_sensor = self.world.spawn_actor(
            imu_bp,
            carla.Transform(carla.Location(x=0.0, z=2.0)),
            attach_to=self.vehicle
        )
        self.imu_sensor.listen(self._imu_callback)
        
        print("[MLDataCollector] Sensors initialized:")
        print("  - GPS True: 10 Hz")
        print("  - GPS Spoof: 10 Hz (with INNOVATION_AWARE_GRADUAL_DRIFT)")
        print("  - IMU: 50 Hz")
        
    def _set_t0_if_needed(self, sensor_timestamp: float):
        """Set the time origin on first sensor reading."""
        if self.t0 is None:
            self.t0 = sensor_timestamp
            
    def _imu_callback(self, data):
        """IMU callback - collect raw IMU data at high frequency."""
        self._set_t0_if_needed(data.timestamp)
        
        if not self.is_collecting:
            return
        
        # Set collection start time on first collected sample
        if self.collection_start_sim_time is None:
            self.collection_start_sim_time = data.timestamp - self.t0
            
        t = data.timestamp - self.t0
        
        sample = IMUSample(
            timestamp=t,
            accel_x=data.accelerometer.x,
            accel_y=data.accelerometer.y,
            accel_z=data.accelerometer.z,
            gyro_x=data.gyroscope.x,
            gyro_y=data.gyroscope.y,
            gyro_z=data.gyroscope.z,
            compass=data.compass
        )
        self.imu_samples.append(sample)
        
        # Update Kalman filter predictions
        if self.kf_initialized:
            try:
                self.kf_true.predict(data, timestamp=t)
                self.kf_spoof.predict(data, timestamp=t)
            except Exception:
                pass
                
    def _gps_callback_true(self, data):
        """GPS callback for true (unspoofed) readings."""
        self._set_t0_if_needed(data.timestamp)
        
        if not self.is_collecting:
            return
        
        # Set collection start time on first collected sample
        if self.collection_start_sim_time is None:
            self.collection_start_sim_time = data.timestamp - self.t0
            
        t = data.timestamp - self.t0
        
        # Get true position from vehicle transform (ground truth)
        tf = self.vehicle.get_transform()
        true_pos = np.array([tf.location.x, tf.location.y, tf.location.z])
        
        # Determine attack state
        if self.random_attacks:
            # Random attack mode: randomly start/stop attacks
            if self.collection_start_sim_time is not None:
                elapsed = t - self.collection_start_sim_time
                
                # Initialize first transition time
                if self.next_attack_transition_time is None:
                    # Start with clean period if attack_start_delay > 0
                    if self.attack_start_delay > 0 and elapsed < self.attack_start_delay:
                        self.attack_active = False
                        self.next_attack_transition_time = self.attack_start_delay
                    else:
                        # Randomly decide to start with attack or clean
                        self.attack_active = np.random.random() < 0.5
                        if self.attack_active:
                            duration = np.random.uniform(self.min_attack_duration, self.max_attack_duration)
                        else:
                            duration = np.random.uniform(self.min_clean_duration, self.max_clean_duration)
                        self.next_attack_transition_time = elapsed + duration
                
                # Check if we need to transition
                if elapsed >= self.next_attack_transition_time:
                    # Toggle attack state
                    self.attack_active = not self.attack_active
                    
                    # Schedule next transition
                    if self.attack_active:
                        duration = np.random.uniform(self.min_attack_duration, self.max_attack_duration)
                    else:
                        duration = np.random.uniform(self.min_clean_duration, self.max_clean_duration)
                    self.next_attack_transition_time = elapsed + duration
        else:
            # Fixed attack mode: single attack period after delay
            if self.collection_start_sim_time is not None:
                elapsed = t - self.collection_start_sim_time
                self.attack_active = elapsed >= self.attack_start_delay
        
        # Apply spoofing if attack is active
        if self.attack_active:
            spoofed_pos = self.spoofer.spoof_position(true_pos, self.current_innovation)
        else:
            spoofed_pos = true_pos.copy()
            
        # Store GPS sample
        gps_sample = GPSSample(
            timestamp=t,
            true_x=true_pos[0],
            true_y=true_pos[1],
            true_z=true_pos[2],
            spoofed_x=spoofed_pos[0],
            spoofed_y=spoofed_pos[1],
            spoofed_z=spoofed_pos[2]
        )
        self.gps_samples.append(gps_sample)
        
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
            
        # Interpolate IMU data to this GPS timestamp
        imu_interp = self._interpolate_imu(t)
        
        # Calculate velocity from position change
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
        
        # Create data point
        data_point = CollectedDataPoint(
            timestamp=t,
            true_gps_x=true_pos[0],
            true_gps_y=true_pos[1],
            true_gps_z=true_pos[2],
            spoofed_gps_x=spoofed_pos[0],
            spoofed_gps_y=spoofed_pos[1],
            spoofed_gps_z=spoofed_pos[2],
            imu_accel_x=imu_interp['accel_x'],
            imu_accel_y=imu_interp['accel_y'],
            imu_accel_z=imu_interp['accel_z'],
            imu_gyro_x=imu_interp['gyro_x'],
            imu_gyro_y=imu_interp['gyro_y'],
            imu_gyro_z=imu_interp['gyro_z'],
            imu_compass=imu_interp['compass'],
            kf_true_x=float(kf_true_pos[0]),
            kf_true_y=float(kf_true_pos[1]),
            kf_true_z=float(kf_true_pos[2]),
            kf_spoof_x=float(kf_spoof_pos[0]),
            kf_spoof_y=float(kf_spoof_pos[1]),
            kf_spoof_z=float(kf_spoof_pos[2]),
            innovation_true=float(innovation_true),
            innovation_spoof=float(innovation_spoof),
            position_error=float(position_error),
            kf_tracking_error=float(kf_tracking_error),
            velocity_magnitude=float(velocity_magnitude),
            is_attack_active=1 if self.attack_active else 0
        )
        self.data_points.append(data_point)
        
    def _gps_callback_spoof(self, data):
        """
        GPS callback for spoofed sensor.
        
        Note: Since both sensors are at the same location and fire simultaneously,
        the actual spoofing logic is handled in _gps_callback_true to ensure
        perfect synchronization. This callback is kept for potential future
        expansion where we might want independent spoofing logic.
        """
        # Spoofing is handled in _gps_callback_true for synchronization
        pass
        
    def _interpolate_imu(self, gps_timestamp: float) -> Dict[str, float]:
        """
        Interpolate IMU data to a GPS timestamp.
        
        Args:
            gps_timestamp: Target timestamp for interpolation
            
        Returns:
            Dictionary with interpolated IMU values
        """
        if len(self.imu_samples) < 2:
            return {
                'accel_x': 0.0, 'accel_y': 0.0, 'accel_z': 0.0,
                'gyro_x': 0.0, 'gyro_y': 0.0, 'gyro_z': 0.0,
                'compass': 0.0
            }
            
        # Extract IMU timestamps and values
        imu_times = np.array([s.timestamp for s in self.imu_samples])
        
        # Handle edge cases
        if gps_timestamp <= imu_times[0]:
            s = self.imu_samples[0]
            return {
                'accel_x': s.accel_x, 'accel_y': s.accel_y, 'accel_z': s.accel_z,
                'gyro_x': s.gyro_x, 'gyro_y': s.gyro_y, 'gyro_z': s.gyro_z,
                'compass': s.compass
            }
        if gps_timestamp >= imu_times[-1]:
            s = self.imu_samples[-1]
            return {
                'accel_x': s.accel_x, 'accel_y': s.accel_y, 'accel_z': s.accel_z,
                'gyro_x': s.gyro_x, 'gyro_y': s.gyro_y, 'gyro_z': s.gyro_z,
                'compass': s.compass
            }
            
        # Interpolate each IMU channel
        accel_x = np.interp(gps_timestamp, imu_times, [s.accel_x for s in self.imu_samples])
        accel_y = np.interp(gps_timestamp, imu_times, [s.accel_y for s in self.imu_samples])
        accel_z = np.interp(gps_timestamp, imu_times, [s.accel_z for s in self.imu_samples])
        gyro_x = np.interp(gps_timestamp, imu_times, [s.gyro_x for s in self.imu_samples])
        gyro_y = np.interp(gps_timestamp, imu_times, [s.gyro_y for s in self.imu_samples])
        gyro_z = np.interp(gps_timestamp, imu_times, [s.gyro_z for s in self.imu_samples])
        compass = np.interp(gps_timestamp, imu_times, [s.compass for s in self.imu_samples])
        
        return {
            'accel_x': float(accel_x),
            'accel_y': float(accel_y),
            'accel_z': float(accel_z),
            'gyro_x': float(gyro_x),
            'gyro_y': float(gyro_y),
            'gyro_z': float(gyro_z),
            'compass': float(compass)
        }
        
    def run_collection(self) -> int:
        """
        Run the data collection process.
        
        Returns:
            Number of data points collected
        """
        print(f"\n{'='*60}")
        print("ML Training Data Collection")
        print(f"{'='*60}")
        print(f"Duration: {self.duration}s")
        print(f"Warmup: {self.warmup_duration}s")
        if self.random_attacks:
            print(f"Attack mode: RANDOM (start/stop)")
            print(f"  Attack duration: {self.min_attack_duration}-{self.max_attack_duration}s")
            print(f"  Clean duration: {self.min_clean_duration}-{self.max_clean_duration}s")
            if self.attack_start_delay > 0:
                print(f"  Initial clean period: {self.attack_start_delay}s")
        else:
            print(f"Attack mode: FIXED")
            print(f"Attack starts after: {self.attack_start_delay}s")
            if self.attack_start_delay == 0:
                print("  (Attack starts immediately - all data points have true/spoofed pairs)")
        print(f"Attack type: INNOVATION_AWARE_GRADUAL_DRIFT")
        print(f"{'='*60}\n")
        
        # Setup traffic for continuous movement
        setup_continuous_traffic(self.world, self.vehicle)
        
        # Enable autopilot
        self.vehicle.set_autopilot(True)
        
        # Wait for warmup period
        print(f"Warming up for {self.warmup_duration}s...")
        time.sleep(self.warmup_duration)
        
        # Start collection
        print("Starting data collection...")
        self.is_collecting = True
        self.collection_start_time = time.time()
        
        # Track collection progress
        start_time = time.time()
        last_report = start_time
        
        while time.time() - start_time < self.duration:
            current_time = time.time()
            
            # Report progress every 10 seconds
            if current_time - last_report >= 10.0:
                elapsed = current_time - start_time
                points = len(self.data_points)
                rate = points / elapsed if elapsed > 0 else 0
                attack_status = "ACTIVE" if self.attack_active else "INACTIVE"
                if self.random_attacks and self.next_attack_transition_time is not None:
                    next_transition = self.next_attack_transition_time - (time.time() - start_time - self.warmup_duration)
                    print(f"  [{elapsed:.0f}s] Collected {points} points ({rate:.1f}/s) | Attack: {attack_status} | Next transition: {next_transition:.1f}s")
                else:
                    print(f"  [{elapsed:.0f}s] Collected {points} points ({rate:.1f}/s) | Attack: {attack_status}")
                last_report = current_time
                
            time.sleep(0.05)
            
        self.is_collecting = False
        print(f"\nCollection complete. Total data points: {len(self.data_points)}")
        
        return len(self.data_points)
        
    def calculate_derived_features(self) -> pd.DataFrame:
        """
        Calculate derived features for ML training.
        
        Returns:
            DataFrame with all features including derived ones
        """
        if len(self.data_points) == 0:
            print("No data points to process")
            return pd.DataFrame()
            
        # Convert to DataFrame
        df = pd.DataFrame([asdict(dp) for dp in self.data_points])
        
        # Calculate acceleration magnitude
        df['accel_magnitude'] = np.sqrt(
            df['imu_accel_x']**2 + 
            df['imu_accel_y']**2 + 
            df['imu_accel_z']**2
        )
        
        # Calculate gyroscope magnitude
        df['gyro_magnitude'] = np.sqrt(
            df['imu_gyro_x']**2 + 
            df['imu_gyro_y']**2 + 
            df['imu_gyro_z']**2
        )
        
        # Calculate position error rate (rate of change)
        df['position_error_rate'] = df['position_error'].diff() / df['timestamp'].diff()
        df['position_error_rate'] = df['position_error_rate'].fillna(0)
        
        # Calculate jerk (rate of change of acceleration)
        df['jerk_x'] = df['imu_accel_x'].diff() / df['timestamp'].diff()
        df['jerk_y'] = df['imu_accel_y'].diff() / df['timestamp'].diff()
        df['jerk_z'] = df['imu_accel_z'].diff() / df['timestamp'].diff()
        df['jerk_magnitude'] = np.sqrt(
            df['jerk_x'].fillna(0)**2 + 
            df['jerk_y'].fillna(0)**2 + 
            df['jerk_z'].fillna(0)**2
        )
        
        # Rolling window statistics (window=10 samples)
        window_size = min(10, len(df))
        
        # Innovation rolling statistics
        df['innovation_spoof_ma'] = df['innovation_spoof'].rolling(
            window=window_size, min_periods=1
        ).mean()
        df['innovation_spoof_std'] = df['innovation_spoof'].rolling(
            window=window_size, min_periods=1
        ).std().fillna(0)
        df['innovation_spoof_max'] = df['innovation_spoof'].rolling(
            window=window_size, min_periods=1
        ).max()
        
        # Position error rolling statistics
        df['position_error_ma'] = df['position_error'].rolling(
            window=window_size, min_periods=1
        ).mean()
        df['position_error_std'] = df['position_error'].rolling(
            window=window_size, min_periods=1
        ).std().fillna(0)
        
        # Tracking error rolling statistics
        df['kf_tracking_error_ma'] = df['kf_tracking_error'].rolling(
            window=window_size, min_periods=1
        ).mean()
        df['kf_tracking_error_std'] = df['kf_tracking_error'].rolling(
            window=window_size, min_periods=1
        ).std().fillna(0)
        
        # Acceleration rolling statistics
        df['accel_magnitude_ma'] = df['accel_magnitude'].rolling(
            window=window_size, min_periods=1
        ).mean()
        df['accel_magnitude_std'] = df['accel_magnitude'].rolling(
            window=window_size, min_periods=1
        ).std().fillna(0)
        
        # Innovation difference (true vs spoof)
        df['innovation_diff'] = df['innovation_spoof'] - df['innovation_true']
        
        # GPS drift components
        df['gps_drift_x'] = df['spoofed_gps_x'] - df['true_gps_x']
        df['gps_drift_y'] = df['spoofed_gps_y'] - df['true_gps_y']
        df['gps_drift_z'] = df['spoofed_gps_z'] - df['true_gps_z']
        
        # KF position difference
        df['kf_diff_x'] = df['kf_spoof_x'] - df['kf_true_x']
        df['kf_diff_y'] = df['kf_spoof_y'] - df['kf_true_y']
        df['kf_diff_z'] = df['kf_spoof_z'] - df['kf_true_z']
        df['kf_diff_magnitude'] = np.sqrt(
            df['kf_diff_x']**2 + df['kf_diff_y']**2 + df['kf_diff_z']**2
        )
        
        return df
        
    def calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate summary statistics for the collected data.
        
        Args:
            df: DataFrame with collected data
            
        Returns:
            Dictionary with statistics
        """
        # Separate attack and non-attack periods
        attack_df = df[df['is_attack_active'] == 1]
        clean_df = df[df['is_attack_active'] == 0]
        
        stats = {
            'total_samples': len(df),
            'clean_samples': len(clean_df),
            'attack_samples': len(attack_df),
            'duration_seconds': float(df['timestamp'].max() - df['timestamp'].min()),
            'sampling_rate_hz': len(df) / max(1, df['timestamp'].max() - df['timestamp'].min()),
            
            'position_error': {
                'min': float(df['position_error'].min()),
                'max': float(df['position_error'].max()),
                'mean': float(df['position_error'].mean()),
                'std': float(df['position_error'].std()),
                'median': float(df['position_error'].median())
            },
            
            'innovation_spoof': {
                'min': float(df['innovation_spoof'].min()),
                'max': float(df['innovation_spoof'].max()),
                'mean': float(df['innovation_spoof'].mean()),
                'std': float(df['innovation_spoof'].std()),
                'median': float(df['innovation_spoof'].median())
            },
            
            'kf_tracking_error': {
                'min': float(df['kf_tracking_error'].min()),
                'max': float(df['kf_tracking_error'].max()),
                'mean': float(df['kf_tracking_error'].mean()),
                'std': float(df['kf_tracking_error'].std()),
                'median': float(df['kf_tracking_error'].median())
            }
        }
        
        # Add attack-specific statistics if there are attack samples
        if len(attack_df) > 0:
            stats['attack_period'] = {
                'position_error_mean': float(attack_df['position_error'].mean()),
                'position_error_max': float(attack_df['position_error'].max()),
                'innovation_spoof_mean': float(attack_df['innovation_spoof'].mean()),
                'kf_tracking_error_mean': float(attack_df['kf_tracking_error'].mean())
            }
            
        # Add clean-period statistics if there are clean samples
        if len(clean_df) > 0:
            stats['clean_period'] = {
                'position_error_mean': float(clean_df['position_error'].mean()),
                'position_error_max': float(clean_df['position_error'].max()),
                'innovation_spoof_mean': float(clean_df['innovation_spoof'].mean()),
                'kf_tracking_error_mean': float(clean_df['kf_tracking_error'].mean())
            }
            
        # Calculate MSE and RMSE
        if len(df) > 0:
            mse = float(np.mean(df['position_error']**2))
            rmse = float(np.sqrt(mse))
            stats['mse'] = mse
            stats['rmse'] = rmse
            
        return stats
        
    def export_data(self, df: pd.DataFrame, stats: Dict) -> Tuple[str, str]:
        """
        Export collected data to JSON and CSV files.
        
        Args:
            df: DataFrame with all features
            stats: Statistics dictionary
            
        Returns:
            Tuple of (json_path, csv_path)
        """
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Build filename with optional label prefix
        if self.label:
            base_filename = f"{self.label}_ml_training_data_{timestamp_str}"
        else:
            base_filename = f"ml_training_data_{timestamp_str}"
        
        # Export CSV for ML training
        csv_filename = f"{base_filename}.csv"
        csv_path = os.path.join(self.output_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
        
        # Export JSON with metadata
        json_filename = f"{base_filename}.json"
        json_path = os.path.join(self.output_dir, json_filename)
        
        metadata = {
            'collection_timestamp': timestamp_str,
            'label': self.label if self.label else 'unlabeled',
            'duration_seconds': self.duration,
            'warmup_seconds': self.warmup_duration,
            'attack_start_delay_seconds': self.attack_start_delay,
            'attack_type': 'INNOVATION_AWARE_GRADUAL_DRIFT',
            'attack_mode': 'RANDOM' if self.random_attacks else 'FIXED',
            'gps_rate_hz': 10,
            'imu_rate_hz': 50,
            'total_samples': len(df)
        }
        
        if self.random_attacks:
            metadata['random_attack_params'] = {
                'min_attack_duration': self.min_attack_duration,
                'max_attack_duration': self.max_attack_duration,
                'min_clean_duration': self.min_clean_duration,
                'max_clean_duration': self.max_clean_duration,
                'random_seed': self.random_seed
            }
        
        json_data = {
            'metadata': metadata,
            'statistics': stats,
            'feature_columns': list(df.columns),
            'raw_data': {
                col: df[col].tolist() for col in df.columns
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved JSON: {json_path}")
        
        return json_path, csv_path
        
    def cleanup(self):
        """Cleanup sensors and resources."""
        sensors = [self.gps_sensor_true, self.gps_sensor_spoof, self.imu_sensor]
        for sensor in sensors:
            if sensor is not None:
                try:
                    sensor.stop()
                except Exception:
                    pass
                try:
                    sensor.destroy()
                except Exception:
                    pass
        print("[MLDataCollector] Sensors cleaned up")


def main():
    """Main entry point for ML data collection."""
    parser = argparse.ArgumentParser(
        description="Collect GPS/IMU data for ML-based spoofing detection"
    )
    parser.add_argument(
        '--duration', type=float, default=60.0,
        help='Collection duration in seconds (default: 60)'
    )
    parser.add_argument(
        '--warmup', type=float, default=5.0,
        help='Warmup duration before recording (default: 5)'
    )
    parser.add_argument(
        '--attack-delay', type=float, default=10.0,
        help='Delay before starting attack (default: 10)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='data',
        help='Output directory for data files (default: data)'
    )
    parser.add_argument(
        '--random-attacks', action='store_true',
        help='Enable random start/stop attacks (more realistic training data)'
    )
    parser.add_argument(
        '--min-attack-duration', type=float, default=5.0,
        help='Minimum attack duration in random mode (default: 5.0)'
    )
    parser.add_argument(
        '--max-attack-duration', type=float, default=15.0,
        help='Maximum attack duration in random mode (default: 15.0)'
    )
    parser.add_argument(
        '--min-clean-duration', type=float, default=5.0,
        help='Minimum clean period duration in random mode (default: 5.0)'
    )
    parser.add_argument(
        '--max-clean-duration', type=float, default=15.0,
        help='Maximum clean period duration in random mode (default: 15.0)'
    )
    parser.add_argument(
        '--label', type=str, default='',
        help='Label prefix for output files (e.g., "train_run01", "val_run05")'
    )
    
    args = parser.parse_args()
    
    print("ML Training Data Collector")
    print("="*60)
    
    # Connect to CARLA
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print(f"Connected to CARLA: {world.get_map().name}")
    except Exception as e:
        print(f"Failed to connect to CARLA: {e}")
        print("Make sure CarlaUE4.exe is running.")
        return 1
        
    # Spawn vehicle
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No spawn points available")
        return 1
        
    vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
    
    if vehicle is None:
        print("Failed to spawn vehicle, trying alternate spawn point...")
        vehicle = world.spawn_actor(vehicle_bp, spawn_points[1])
        
    if vehicle is None:
        print("Failed to spawn vehicle")
        return 1
        
    print(f"Spawned vehicle: {vehicle.type_id}")
    time.sleep(1.0)
    
    # Create collector
    collector = MLDataCollector(
        vehicle=vehicle,
        duration=args.duration,
        warmup_duration=args.warmup,
        attack_start_delay=args.attack_delay,
        output_dir=args.output_dir,
        random_attacks=args.random_attacks,
        min_attack_duration=args.min_attack_duration,
        max_attack_duration=args.max_attack_duration,
        min_clean_duration=args.min_clean_duration,
        max_clean_duration=args.max_clean_duration,
        label=args.label
    )
    
    try:
        # Setup sensors
        collector.setup_sensors()
        
        # Run collection
        num_points = collector.run_collection()
        
        if num_points > 0:
            # Calculate derived features
            print("\nCalculating derived features...")
            df = collector.calculate_derived_features()
            
            # Calculate statistics
            print("Calculating statistics...")
            stats = collector.calculate_statistics(df)
            
            # Print summary
            print(f"\n{'='*60}")
            print("Collection Summary")
            print(f"{'='*60}")
            print(f"Total samples: {stats['total_samples']}")
            print(f"Clean samples: {stats['clean_samples']}")
            print(f"Attack samples: {stats['attack_samples']}")
            print(f"Duration: {stats['duration_seconds']:.1f}s")
            print(f"Sampling rate: {stats['sampling_rate_hz']:.1f} Hz")
            print(f"\nPosition Error (meters):")
            print(f"  Mean: {stats['position_error']['mean']:.3f}")
            print(f"  Max: {stats['position_error']['max']:.3f}")
            print(f"  Std: {stats['position_error']['std']:.3f}")
            print(f"\nMSE: {stats.get('mse', 0):.6f}")
            print(f"RMSE: {stats.get('rmse', 0):.3f} meters")
            print(f"{'='*60}")
            
            # Export data
            print("\nExporting data...")
            json_path, csv_path = collector.export_data(df, stats)
            
            print(f"\nData collection complete!")
            print(f"  CSV: {csv_path}")
            print(f"  JSON: {json_path}")
        else:
            print("No data points collected")
            
    except KeyboardInterrupt:
        print("\nCollection interrupted by user")
    except Exception as e:
        print(f"Error during collection: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nCleaning up...")
        collector.cleanup()
        cleanup_traffic_setup(world, vehicle)
        try:
            vehicle.destroy()
        except Exception:
            pass
        print("Cleanup complete")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())

