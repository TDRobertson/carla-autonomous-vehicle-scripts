#!/usr/bin/env python3
"""
ML Training Data Collector for GPS+IMU+KF-only Spoofing Detection

Collects synchronized GPS and IMU data using ONLY victim-side sensor outputs:
- GNSS sensor (lat/lon/alt) converted to local ENU meters
- IMU accelerometer and gyroscope
- Kalman filter state and innovation statistics
- NO ground-truth vehicle position for features

Ground truth (vehicle.get_transform()) is used only for:
- Applying simulated spoofing attacks
- Logging is_attack_active for evaluation

Output:
- CSV file with GPS+IMU+KF-only features for ML training

Usage:
    # Collect clean training data (no attacks)
    python ml_data_collector_gps_imu_kf.py --duration 120 --output-dir data/training_gps_imu_kf
    
    # Collect validation data with attacks
    python ml_data_collector_gps_imu_kf.py --duration 120 --attack-delay 30 \
        --output-dir data/validation_gps_imu_kf
    
    # Collect with chaotic (random) attacks
    python ml_data_collector_gps_imu_kf.py --duration 180 --random-attacks \
        --output-dir data/validation_gps_imu_kf
"""

import sys
import os
import glob
import time
import json
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from collections import deque

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
from utils.geo_utils import GNSSToLocalConverter


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
class GPSIMUKFDataPoint:
    """
    Single data point with GPS+IMU+KF-only features.
    
    NO ground-truth position is stored as a feature.
    is_attack_active is only for evaluation, not for training.
    """
    # Timestamp
    timestamp: float
    
    # GPS position (local ENU meters from GNSS sensor - potentially spoofed)
    gps_x: float
    gps_y: float
    gps_z: float
    
    # Kalman filter updated state
    kf_x: float
    kf_y: float
    kf_z: float
    kf_vx: float
    kf_vy: float
    kf_vz: float
    
    # Innovation (measurement residual)
    innov_x: float
    innov_y: float
    innov_z: float
    innov_norm: float
    
    # Normalized Innovation Squared
    nis: float
    
    # Innovation covariance diagonal
    S_x: float
    S_y: float
    S_z: float
    
    # IMU data
    accel_magnitude: float
    gyro_magnitude: float
    
    # Position covariance diagonal (optional - for uncertainty tracking)
    P_x: float
    P_y: float
    P_z: float
    
    # Label (for evaluation only, NOT used for unsupervised training)
    is_attack_active: int


class MLDataCollectorGPSIMUKF:
    """
    Collects GPS+IMU+KF-only data for ML training.
    
    Key difference from standard collector:
    - Uses GNSS sensor output (lat/lon/alt) converted to local ENU
    - Does NOT store ground-truth vehicle position as features
    - Innovation/NIS statistics are the primary detection signals
    """
    
    def __init__(
        self,
        vehicle,
        duration: float = 60.0,
        warmup_duration: float = 5.0,
        attack_start_delay: float = 0.0,  # Default: no attacks (clean data)
        output_dir: str = "data/training_gps_imu_kf",
        random_attacks: bool = False,
        min_attack_duration: float = 5.0,
        max_attack_duration: float = 15.0,
        min_clean_duration: float = 5.0,
        max_clean_duration: float = 15.0,
        label: str = ""
    ):
        """
        Initialize the GPS+IMU+KF-only data collector.
        
        Args:
            vehicle: CARLA vehicle actor
            duration: Total collection duration in seconds
            warmup_duration: Initial warmup period before recording
            attack_start_delay: Delay before starting spoofing attack.
                               Set to 0 with no random_attacks for clean-only data.
                               Set > duration for fully clean data.
            output_dir: Directory to save output files
            random_attacks: If True, randomly start/stop attacks
            min_attack_duration: Min attack duration (random mode)
            max_attack_duration: Max attack duration (random mode)
            min_clean_duration: Min clean duration (random mode)
            max_clean_duration: Max clean duration (random mode)
            label: Label prefix for output files
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
        self.t0: Optional[float] = None
        self.collection_start_sim_time: Optional[float] = None
        
        # Sensors
        self.gnss_sensor: Optional[carla.Actor] = None
        self.imu_sensor: Optional[carla.Actor] = None
        
        # GNSS to local converter
        self.geo_converter = GNSSToLocalConverter()
        
        # GPS Spoofer
        self.spoofer = GPSSpoofer(
            [0, 0, 0],
            strategy=SpoofingStrategy.INNOVATION_AWARE_GRADUAL_DRIFT
        )
        self.current_innovation_norm = 0.0
        
        # Kalman filter (single - we only see the potentially spoofed GPS)
        self.kf = AdvancedKalmanFilter()
        self.kf_initialized = False
        
        # Raw data buffers
        self.imu_samples: List[IMUSample] = []
        
        # Collected data points
        self.data_points: List[GPSIMUKFDataPoint] = []
        
        # Rolling buffers for derived features
        self.innov_norm_buffer = deque(maxlen=10)
        self.nis_buffer = deque(maxlen=10)
        
        # State tracking
        self.is_collecting = False
        self.attack_active = False
        
        # Random attack state
        self.next_attack_transition_time: Optional[float] = None
        self.random_seed = int(time.time() * 1000) % 2**31
        np.random.seed(self.random_seed)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def setup_sensors(self):
        """Setup GNSS and IMU sensors."""
        bp_lib = self.world.get_blueprint_library()
        
        # GNSS Sensor (lat/lon/alt output)
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
        
        print("[MLDataCollectorGPSIMUKF] Sensors initialized:")
        print("  - GNSS: 10 Hz (lat/lon/alt -> local ENU)")
        print("  - IMU: 50 Hz")
        print("  - NO ground-truth position used for features")
        
    def _set_t0_if_needed(self, sensor_timestamp: float):
        """Set the time origin on first sensor reading."""
        if self.t0 is None:
            self.t0 = sensor_timestamp
            
    def _imu_callback(self, data):
        """IMU callback - collect raw IMU data."""
        self._set_t0_if_needed(data.timestamp)
        
        if not self.is_collecting:
            return
            
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
        
        # Update KF prediction
        if self.kf_initialized:
            try:
                self.kf.predict(data, timestamp=t)
            except Exception:
                pass
                
    def _gnss_callback(self, data):
        """GNSS callback - main data collection."""
        self._set_t0_if_needed(data.timestamp)
        
        if not self.is_collecting:
            return
            
        if self.collection_start_sim_time is None:
            self.collection_start_sim_time = data.timestamp - self.t0
            
        t = data.timestamp - self.t0
        
        # Get GNSS measurement and convert to local ENU
        lat = data.latitude
        lon = data.longitude
        alt = data.altitude
        gps_x, gps_y, gps_z = self.geo_converter.convert(lat, lon, alt)
        gps_pos = np.array([gps_x, gps_y, gps_z])
        
        # Determine attack state
        elapsed = t - self.collection_start_sim_time if self.collection_start_sim_time else 0
        
        if self.random_attacks:
            self._update_random_attack_state(elapsed)
        else:
            # Fixed attack mode
            self.attack_active = elapsed >= self.attack_start_delay
        
        # Apply spoofing if attack is active
        if self.attack_active:
            spoofed_gps = self.spoofer.spoof_position(gps_pos, self.current_innovation_norm)
        else:
            spoofed_gps = gps_pos.copy()
            
        # Initialize KF on first GPS
        if not self.kf_initialized:
            try:
                self.kf.position = spoofed_gps.copy()
                self.kf_initialized = True
            except Exception:
                pass
                
        # Update KF with (potentially spoofed) GPS and get detailed info
        kf_info = {
            'innovation_vector': np.zeros(3),
            'innovation_norm': 0.0,
            'S_diag': np.ones(3),
            'nis': 0.0,
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
        
        # Create data point
        data_point = GPSIMUKFDataPoint(
            timestamp=t,
            # GPS (potentially spoofed)
            gps_x=float(spoofed_gps[0]),
            gps_y=float(spoofed_gps[1]),
            gps_z=float(spoofed_gps[2]),
            # KF state
            kf_x=float(kf_info['position_upd'][0]),
            kf_y=float(kf_info['position_upd'][1]),
            kf_z=float(kf_info['position_upd'][2]),
            kf_vx=float(kf_info['velocity_upd'][0]),
            kf_vy=float(kf_info['velocity_upd'][1]),
            kf_vz=float(kf_info['velocity_upd'][2]),
            # Innovation
            innov_x=float(kf_info['innovation_vector'][0]),
            innov_y=float(kf_info['innovation_vector'][1]),
            innov_z=float(kf_info['innovation_vector'][2]),
            innov_norm=float(kf_info['innovation_norm']),
            # NIS
            nis=float(kf_info['nis']),
            # Innovation covariance
            S_x=float(kf_info['S_diag'][0]),
            S_y=float(kf_info['S_diag'][1]),
            S_z=float(kf_info['S_diag'][2]),
            # IMU
            accel_magnitude=float(accel_magnitude),
            gyro_magnitude=float(gyro_magnitude),
            # Position covariance
            P_x=float(kf_info['P_pos_diag'][0]),
            P_y=float(kf_info['P_pos_diag'][1]),
            P_z=float(kf_info['P_pos_diag'][2]),
            # Label (for evaluation only)
            is_attack_active=1 if self.attack_active else 0
        )
        self.data_points.append(data_point)
        
    def _update_random_attack_state(self, elapsed: float):
        """Update attack state for random attack mode."""
        # Initialize first transition
        if self.next_attack_transition_time is None:
            if self.attack_start_delay > 0 and elapsed < self.attack_start_delay:
                self.attack_active = False
                self.next_attack_transition_time = self.attack_start_delay
            else:
                self.attack_active = np.random.random() < 0.5
                if self.attack_active:
                    duration = np.random.uniform(self.min_attack_duration, self.max_attack_duration)
                else:
                    duration = np.random.uniform(self.min_clean_duration, self.max_clean_duration)
                self.next_attack_transition_time = elapsed + duration
                
        # Check for transition
        if elapsed >= self.next_attack_transition_time:
            self.attack_active = not self.attack_active
            if self.attack_active:
                duration = np.random.uniform(self.min_attack_duration, self.max_attack_duration)
            else:
                duration = np.random.uniform(self.min_clean_duration, self.max_clean_duration)
            self.next_attack_transition_time = elapsed + duration
            
    def _interpolate_imu(self, gps_timestamp: float) -> Dict[str, float]:
        """Interpolate IMU data to GPS timestamp."""
        if len(self.imu_samples) < 2:
            return {
                'accel_x': 0.0, 'accel_y': 0.0, 'accel_z': 0.0,
                'gyro_x': 0.0, 'gyro_y': 0.0, 'gyro_z': 0.0
            }
            
        imu_times = np.array([s.timestamp for s in self.imu_samples])
        
        if gps_timestamp <= imu_times[0]:
            s = self.imu_samples[0]
            return {
                'accel_x': s.accel_x, 'accel_y': s.accel_y, 'accel_z': s.accel_z,
                'gyro_x': s.gyro_x, 'gyro_y': s.gyro_y, 'gyro_z': s.gyro_z
            }
        if gps_timestamp >= imu_times[-1]:
            s = self.imu_samples[-1]
            return {
                'accel_x': s.accel_x, 'accel_y': s.accel_y, 'accel_z': s.accel_z,
                'gyro_x': s.gyro_x, 'gyro_y': s.gyro_y, 'gyro_z': s.gyro_z
            }
            
        # Interpolate
        accel_x = np.interp(gps_timestamp, imu_times, [s.accel_x for s in self.imu_samples])
        accel_y = np.interp(gps_timestamp, imu_times, [s.accel_y for s in self.imu_samples])
        accel_z = np.interp(gps_timestamp, imu_times, [s.accel_z for s in self.imu_samples])
        gyro_x = np.interp(gps_timestamp, imu_times, [s.gyro_x for s in self.imu_samples])
        gyro_y = np.interp(gps_timestamp, imu_times, [s.gyro_y for s in self.imu_samples])
        gyro_z = np.interp(gps_timestamp, imu_times, [s.gyro_z for s in self.imu_samples])
        
        return {
            'accel_x': float(accel_x),
            'accel_y': float(accel_y),
            'accel_z': float(accel_z),
            'gyro_x': float(gyro_x),
            'gyro_y': float(gyro_y),
            'gyro_z': float(gyro_z)
        }
        
    def run_collection(self) -> int:
        """Run the data collection process."""
        print(f"\n{'='*60}")
        print("GPS+IMU+KF-only ML Training Data Collection")
        print(f"{'='*60}")
        print(f"Duration: {self.duration}s")
        print(f"Warmup: {self.warmup_duration}s")
        
        if self.random_attacks:
            print(f"Attack mode: RANDOM")
            print(f"  Attack duration: {self.min_attack_duration}-{self.max_attack_duration}s")
            print(f"  Clean duration: {self.min_clean_duration}-{self.max_clean_duration}s")
        elif self.attack_start_delay >= self.duration:
            print(f"Attack mode: NONE (clean data only)")
        else:
            print(f"Attack mode: FIXED")
            print(f"  Attack starts after: {self.attack_start_delay}s")
            
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")
        
        # Setup traffic
        setup_continuous_traffic(self.world, self.vehicle)
        
        # Enable autopilot
        self.vehicle.set_autopilot(True)
        
        # Warmup
        print(f"Warming up for {self.warmup_duration}s...")
        time.sleep(self.warmup_duration)
        
        # Start collection
        print("Starting data collection...")
        self.is_collecting = True
        
        start_time = time.time()
        last_report = start_time
        
        while time.time() - start_time < self.duration:
            current_time = time.time()
            
            if current_time - last_report >= 10.0:
                elapsed = current_time - start_time
                points = len(self.data_points)
                rate = points / elapsed if elapsed > 0 else 0
                attack_status = "ACTIVE" if self.attack_active else "CLEAN"
                print(f"  [{elapsed:.0f}s] Collected {points} points ({rate:.1f}/s) | Attack: {attack_status}")
                last_report = current_time
                
            time.sleep(0.05)
            
        self.is_collecting = False
        print(f"\nCollection complete. Total data points: {len(self.data_points)}")
        
        return len(self.data_points)
        
    def calculate_derived_features(self) -> pd.DataFrame:
        """Calculate derived features (rolling stats) for ML."""
        if len(self.data_points) == 0:
            return pd.DataFrame()
            
        df = pd.DataFrame([asdict(dp) for dp in self.data_points])
        
        # Rolling window statistics
        window_size = min(10, len(df))
        
        # Innovation norm rolling stats
        df['innov_norm_ma'] = df['innov_norm'].rolling(window=window_size, min_periods=1).mean()
        df['innov_norm_std'] = df['innov_norm'].rolling(window=window_size, min_periods=1).std().fillna(0)
        
        # NIS rolling stats
        df['nis_ma'] = df['nis'].rolling(window=window_size, min_periods=1).mean()
        df['nis_std'] = df['nis'].rolling(window=window_size, min_periods=1).std().fillna(0)
        
        return df
        
    def calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics."""
        clean_df = df[df['is_attack_active'] == 0]
        attack_df = df[df['is_attack_active'] == 1]
        
        stats = {
            'total_samples': len(df),
            'clean_samples': len(clean_df),
            'attack_samples': len(attack_df),
            'duration_seconds': float(df['timestamp'].max() - df['timestamp'].min()) if len(df) > 0 else 0,
            'sampling_rate_hz': len(df) / max(1, df['timestamp'].max() - df['timestamp'].min()) if len(df) > 0 else 0,
            'innov_norm': {
                'min': float(df['innov_norm'].min()),
                'max': float(df['innov_norm'].max()),
                'mean': float(df['innov_norm'].mean()),
                'std': float(df['innov_norm'].std())
            },
            'nis': {
                'min': float(df['nis'].min()),
                'max': float(df['nis'].max()),
                'mean': float(df['nis'].mean()),
                'std': float(df['nis'].std())
            }
        }
        
        if len(attack_df) > 0:
            stats['attack_period'] = {
                'innov_norm_mean': float(attack_df['innov_norm'].mean()),
                'nis_mean': float(attack_df['nis'].mean())
            }
            
        if len(clean_df) > 0:
            stats['clean_period'] = {
                'innov_norm_mean': float(clean_df['innov_norm'].mean()),
                'nis_mean': float(clean_df['nis'].mean())
            }
            
        return stats
        
    def export_data(self, df: pd.DataFrame, stats: Dict) -> Tuple[str, str]:
        """Export collected data to CSV and JSON."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.label:
            base_filename = f"{self.label}_gps_imu_kf_data_{timestamp_str}"
        else:
            base_filename = f"gps_imu_kf_data_{timestamp_str}"
            
        # Export CSV
        csv_filename = f"{base_filename}.csv"
        csv_path = os.path.join(self.output_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")
        
        # Export JSON metadata
        json_filename = f"{base_filename}.json"
        json_path = os.path.join(self.output_dir, json_filename)
        
        metadata = {
            'collection_timestamp': timestamp_str,
            'label': self.label if self.label else 'unlabeled',
            'duration_seconds': self.duration,
            'warmup_seconds': self.warmup_duration,
            'attack_start_delay_seconds': self.attack_start_delay,
            'attack_mode': 'RANDOM' if self.random_attacks else ('FIXED' if self.attack_start_delay < self.duration else 'NONE'),
            'feature_type': 'GPS+IMU+KF-only (no ground truth)',
            'gnss_rate_hz': 10,
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
            'feature_columns': list(df.columns)
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved JSON: {json_path}")
        
        return json_path, csv_path
        
    def cleanup(self):
        """Cleanup sensors."""
        sensors = [self.gnss_sensor, self.imu_sensor]
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
        print("[MLDataCollectorGPSIMUKF] Sensors cleaned up")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Collect GPS+IMU+KF-only data for ML-based spoofing detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Collect clean training data (no attacks)
    python ml_data_collector_gps_imu_kf.py --duration 120 --output-dir data/training_gps_imu_kf
    
    # Collect validation data with attacks after 30s
    python ml_data_collector_gps_imu_kf.py --duration 120 --attack-delay 30 \\
        --output-dir data/validation_gps_imu_kf
    
    # Collect with random attacks
    python ml_data_collector_gps_imu_kf.py --duration 180 --random-attacks \\
        --output-dir data/validation_gps_imu_kf
        """
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
        '--attack-delay', type=float, default=99999.0,
        help='Delay before starting attack. Set > duration for clean-only data (default: 99999 = no attacks)'
    )
    parser.add_argument(
        '--output-dir', type=str, default='data/training_gps_imu_kf',
        help='Output directory (default: data/training_gps_imu_kf)'
    )
    parser.add_argument(
        '--random-attacks', action='store_true',
        help='Enable random start/stop attacks'
    )
    parser.add_argument(
        '--min-attack-duration', type=float, default=5.0,
        help='Min attack duration in random mode (default: 5.0)'
    )
    parser.add_argument(
        '--max-attack-duration', type=float, default=15.0,
        help='Max attack duration in random mode (default: 15.0)'
    )
    parser.add_argument(
        '--min-clean-duration', type=float, default=5.0,
        help='Min clean duration in random mode (default: 5.0)'
    )
    parser.add_argument(
        '--max-clean-duration', type=float, default=15.0,
        help='Max clean duration in random mode (default: 15.0)'
    )
    parser.add_argument(
        '--label', type=str, default='',
        help='Label prefix for output files'
    )
    
    args = parser.parse_args()
    
    print("GPS+IMU+KF-only ML Training Data Collector")
    print("="*60)
    print("This collector uses ONLY victim-side sensor data:")
    print("  - GNSS sensor (lat/lon/alt -> local ENU)")
    print("  - IMU accelerometer and gyroscope")
    print("  - Kalman filter state and innovation")
    print("  - NO ground-truth vehicle position for features")
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
    collector = MLDataCollectorGPSIMUKF(
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
            print("Collection Summary (GPS+IMU+KF-only)")
            print(f"{'='*60}")
            print(f"Total samples: {stats['total_samples']}")
            print(f"Clean samples: {stats['clean_samples']}")
            print(f"Attack samples: {stats['attack_samples']}")
            print(f"Duration: {stats['duration_seconds']:.1f}s")
            print(f"Sampling rate: {stats['sampling_rate_hz']:.1f} Hz")
            print(f"\nInnovation Norm:")
            print(f"  Mean: {stats['innov_norm']['mean']:.3f}")
            print(f"  Max: {stats['innov_norm']['max']:.3f}")
            print(f"  Std: {stats['innov_norm']['std']:.3f}")
            print(f"\nNIS:")
            print(f"  Mean: {stats['nis']['mean']:.3f}")
            print(f"  Max: {stats['nis']['max']:.3f}")
            print(f"{'='*60}")
            
            # Export data
            print("\nExporting data...")
            json_path, csv_path = collector.export_data(df, stats)
            
            print(f"\nData collection complete!")
            print(f"  CSV: {csv_path}")
            print(f"  JSON: {json_path}")
            
            if stats['attack_samples'] == 0:
                print("\n[INFO] No attack samples - this data is suitable for TRAINING")
            else:
                print("\n[INFO] Contains attack samples - this data is suitable for VALIDATION")
        else:
            print("No data points collected")
            
    except KeyboardInterrupt:
        print("\nCollection interrupted by user")
    except Exception as e:
        print(f"Error during collection: {e}")
        import traceback
        traceback.print_exc()
    finally:
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

