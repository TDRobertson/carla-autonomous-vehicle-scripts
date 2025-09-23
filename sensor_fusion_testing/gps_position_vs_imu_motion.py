#!/usr/bin/env python3
"""
GPS Position vs IMU Motion Comparison - Innovation-Aware Gradual Drift Attack
Conference-ready version.

- KF init on first GPS tick.
- Plot errors as RMSE (meters) instead of squared error.
- Trim the first few GPS ticks to avoid large initialization spikes.
- Professional figure naming for presentation.
- Includes:
  * GPS vs IMU baseline plots
  * Side-by-side detection overview (attack vs innovation)
  * Single-slide attack–defense summary overlay
"""

import sys
import os
import glob
import time
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

import carla

# Add the CARLA Python API to PYTHONPATH (best-effort)
try:
    sys.path.append(
        glob.glob(
            '../../carla/dist/carla-*%d.%d-%s.egg'
            % (sys.version_info.major, sys.version_info.minor,
               'win-amd64' if os.name == 'nt' else 'linux-x86_64')
        )[0]
    )
except IndexError:
    pass

# Project imports
from integration_files.gps_spoofer import GPSSpoofer, SpoofingStrategy
from integration_files.traffic_utils import setup_continuous_traffic, cleanup_traffic_setup
from integration_files.advanced_kalman_filter import AdvancedKalmanFilter


class GPSPositionvsIMUMotion:
    def __init__(self, world, vehicle, duration=30, warmup=5):
        self.world = world
        self.vehicle = vehicle
        self.duration = duration
        self.t0 = None  # simulation-time origin (first sensor tick)

        # Trim first N GPS ticks
        self.warmup = warmup

        # GPS (time base) + true/spoofed Cartesian positions (meters)
        self.gps_t, self.true_x, self.true_y, self.spoof_x, self.spoof_y = [], [], [], [], []

        # IMU raw streams
        self.imu_t = []
        self.ax, self.ay, self.az = [], [], []
        self.gx, self.gy, self.gz = [], [], []
        self.compass = []  # radians (North = 0)

        # Kalman filters (two runs in parallel)
        self.kf_true = AdvancedKalmanFilter()
        self.kf_spoof = AdvancedKalmanFilter()
        self._kf_inited = False

        # KF state logs (post-update)
        self.kf_true_x, self.kf_true_y = [], []
        self.kf_spoof_x, self.kf_spoof_y = [], []

        # Error series
        self.se_true, self.se_spoof = [], []
        self.se_gt_vs_gps_true, self.se_gt_vs_gps_spoof = [], []
        self.se_gps_true_vs_kfpred, self.se_gps_spoof_vs_kfpred = [], []
        self.innov_true2, self.innov_spoof2 = [], []

        # GPS spoofer
        self.spoofer = GPSSpoofer([0, 0, 0], strategy=SpoofingStrategy.INNOVATION_AWARE_GRADUAL_DRIFT)
        self.current_innovation = 0.0

        self.gps_sensor = None
        self.imu_sensor = None
        self.setup_sensors()

    # ---------------------- Helpers ----------------------

    def _rmse(self, arr):
        arr = np.asarray(arr, dtype=float)
        return np.sqrt(np.maximum(arr, 0.0))

    def _trim(self, N, *series):
        """Trim first self.warmup samples for clarity."""
        return (N[self.warmup:],) + tuple(s[self.warmup:] for s in series)

    # ---------------------- Sensors ----------------------

    def setup_sensors(self):
        bp_lib = self.world.get_blueprint_library()

        # GNSS
        gps_bp = bp_lib.find('sensor.other.gnss')
        gps_bp.set_attribute('sensor_tick', '0.10')  # 10 Hz
        self.gps_sensor = self.world.spawn_actor(
            gps_bp,
            carla.Transform(carla.Location(x=0.0, z=2.0)),
            attach_to=self.vehicle
        )
        self.gps_sensor.listen(self.gps_callback)

        # IMU
        imu_bp = bp_lib.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', '0.02')  # 50 Hz
        self.imu_sensor = self.world.spawn_actor(
            imu_bp,
            carla.Transform(carla.Location(x=0.0, z=2.0)),
            attach_to=self.vehicle
        )
        self.imu_sensor.listen(self.imu_callback)

    # ---------------------- Callbacks ----------------------

    def _set_t0_if_needed(self, sensor_timestamp: float):
        if self.t0 is None:
            self.t0 = sensor_timestamp

    def gps_callback(self, data):
        """Use vehicle world transform for 'true' Cartesian position (meters)."""
        self._set_t0_if_needed(data.timestamp)
        t = data.timestamp - self.t0

        tf = self.vehicle.get_transform()
        px, py = tf.location.x, tf.location.y

        spoofed = self.spoofer.spoof_position(np.array([px, py, 0.0]), self.current_innovation)
        sx, sy = float(spoofed[0]), float(spoofed[1])

        # Init KFs on first GPS tick
        if not self._kf_inited:
            try:
                self.kf_true.initialize(position=np.array([px, py, 0.0]))
                self.kf_spoof.initialize(position=np.array([px, py, 0.0]))
            except Exception:
                pass
            self._kf_inited = True

        # Log GPS timebase
        self.gps_t.append(t)
        self.true_x.append(px); self.true_y.append(py)
        self.spoof_x.append(sx); self.spoof_y.append(sy)

        # Predicted KF states (pre-update)
        try:
            st_true_pred = self.kf_true.get_state()
            kx_true_pred, ky_true_pred = float(st_true_pred['position'][0]), float(st_true_pred['position'][1])
        except Exception:
            kx_true_pred, ky_true_pred = px, py

        try:
            st_spoof_pred = self.kf_spoof.get_state()
            kx_spoof_pred, ky_spoof_pred = float(st_spoof_pred['position'][0]), float(st_spoof_pred['position'][1])
        except Exception:
            kx_spoof_pred, ky_spoof_pred = px, py

        # Innovation / professor series
        innov2_true = (px - kx_true_pred)**2 + (py - ky_true_pred)**2
        innov2_spoof = (sx - kx_spoof_pred)**2 + (sy - ky_spoof_pred)**2
        self.innov_true2.append(innov2_true)
        self.innov_spoof2.append(innov2_spoof)
        self.se_gps_true_vs_kfpred.append(innov2_true)
        self.se_gps_spoof_vs_kfpred.append(innov2_spoof)

        # KF updates
        try:
            self.kf_true.update_with_gps(np.array([px, py, 0.0]))
            st_true = self.kf_true.get_state()
            kx_true, ky_true = float(st_true['position'][0]), float(st_true['position'][1])
        except Exception:
            kx_true, ky_true = px, py
        self.kf_true_x.append(kx_true); self.kf_true_y.append(ky_true)

        try:
            self.kf_spoof.update_with_gps(np.array([sx, sy, 0.0]))
            st_spoof = self.kf_spoof.get_state()
            kx_spoof, ky_spoof = float(st_spoof['position'][0]), float(st_spoof['position'][1])
        except Exception:
            kx_spoof, ky_spoof = sx, sy
        self.kf_spoof_x.append(kx_spoof); self.kf_spoof_y.append(ky_spoof)

        # KF vs truth (post-update, optional)
        self.se_true.append((px - kx_true) ** 2 + (py - ky_true) ** 2)
        self.se_spoof.append((px - kx_spoof) ** 2 + (py - ky_spoof) ** 2)

        # Truth vs spoofed GPS
        self.se_gt_vs_gps_true.append(0.0)
        self.se_gt_vs_gps_spoof.append((px - sx) ** 2 + (py - sy) ** 2)

    def imu_callback(self, data):
        self._set_t0_if_needed(data.timestamp)
        t = data.timestamp - self.t0

        self.imu_t.append(t)
        self.ax.append(data.accelerometer.x)
        self.ay.append(data.accelerometer.y)
        self.az.append(data.accelerometer.z)
        self.gx.append(data.gyroscope.x)
        self.gy.append(data.gyroscope.y)
        self.gz.append(data.gyroscope.z)
        self.compass.append(data.compass)

        try:
            self.kf_true.predict(data, timestamp=t)
            self.kf_spoof.predict(data, timestamp=t)
        except Exception:
            pass

    # ---------------------- Run / Align / Smooth ----------------------

    def run_comparison(self):
        print(f"Running for {self.duration}s with {self.vehicle.type_id}")
        setup_continuous_traffic(self.world, self.vehicle)
        self.vehicle.set_autopilot(True)
        start = time.time()
        while time.time() - start < self.duration:
            time.sleep(0.05)
        print(f"Done. GPS points: {len(self.gps_t)} | IMU points: {len(self.imu_t)}")

    def _align_and_smooth(self):
        if len(self.gps_t) < 2 or len(self.imu_t) < 2:
            return None

        T = np.array(self.gps_t)
        ti = np.array(self.imu_t)

        def interp(series):
            return np.interp(T, ti, np.array(series, dtype=float))

        imu_ax, imu_ay, imu_az = interp(self.ax), interp(self.ay), interp(self.az)
        imu_compass = interp(self.compass)
        heading_deg = np.rad2deg(np.unwrap(imu_compass))

        def smooth(sig):
            if len(sig) > 31:
                return savgol_filter(sig, 31, 3)
            return sig

        return T, smooth(imu_ax), smooth(imu_ay), smooth(imu_az), smooth(heading_deg)

    # ---------------------- Plots ----------------------

    def plot_detection_overview(self):
        if len(self.se_gps_true_vs_kfpred) == 0 or len(self.se_gt_vs_gps_spoof) == 0:
            print("Not enough data for detection overview.")
            return

        N_pred = np.arange(1, len(self.se_gps_true_vs_kfpred) + 1)
        rmse_true_pred  = self._rmse(self.se_gps_true_vs_kfpred)
        rmse_spoof_pred = self._rmse(self.se_gps_spoof_vs_kfpred)
        N_spoof = np.arange(1, len(self.se_gt_vs_gps_spoof) + 1)
        rmse_spoof_div = self._rmse(self.se_gt_vs_gps_spoof)

        # Trim warmup
        N_pred, rmse_true_pred, rmse_spoof_pred = self._trim(N_pred, rmse_true_pred, rmse_spoof_pred)
        N_spoof, rmse_spoof_div = self._trim(N_spoof, rmse_spoof_div)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(N_spoof, rmse_spoof_div, '--', color='tab:red',
                     label='‖ Truth – GPS_spoof ‖ (RMSE)')
        axes[0].set_title('Ground Truth vs Spoofed GPS Divergence', fontsize=13)
        axes[0].set_xlabel('GPS Sample Index (n)'); axes[0].set_ylabel('Error (m)')
        axes[0].grid(True, alpha=0.3); axes[0].legend()

        axes[1].plot(N_pred, rmse_true_pred,  label='Innovation (True GPS)',   color='tab:blue')
        axes[1].plot(N_pred, rmse_spoof_pred, '--', label='Innovation (Spoofed GPS)', color='tab:orange')
        axes[1].set_title('GPS–IMU Innovation Residuals', fontsize=13)
        axes[1].set_xlabel('GPS Sample Index (n)'); axes[1].set_ylabel('Error (m)')
        axes[1].grid(True, alpha=0.3); axes[1].legend()

        plt.tight_layout()
        out = f"detection_overview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(out, dpi=300, bbox_inches='tight'); print(f"Saved: {out}")
        plt.show()

    def plot_attack_defense_summary(self):
        series = [
            self.se_gt_vs_gps_spoof,
            self.se_true, self.se_spoof,
            self.se_gps_true_vs_kfpred, self.se_gps_spoof_vs_kfpred
        ]
        if any(len(s) == 0 for s in series):
            print("Not enough data for attack/defense summary.")
            return

        Nmin = min(len(s) for s in series)
        A = self._rmse(self.se_gt_vs_gps_spoof[:Nmin])
        E_true  = self._rmse(self.se_true[:Nmin])
        E_spoof = self._rmse(self.se_spoof[:Nmin])
        R_true  = self._rmse(self.se_gps_true_vs_kfpred[:Nmin])
        R_spoof = self._rmse(self.se_gps_spoof_vs_kfpred[:Nmin])
        N = np.arange(1, Nmin + 1)

        # Trim warmup
        N, A, E_true, E_spoof, R_true, R_spoof = self._trim(N, A, E_true, E_spoof, R_true, R_spoof)

        plt.figure(figsize=(12, 5.5))
        plt.plot(N, A, linestyle='--', linewidth=2.0, label='Attack Magnitude: ‖Truth − GPS_spoof‖', color='tab:red')
        plt.plot(N, E_true,  linewidth=1.6, label='KF Tracking Error (True GPS): ‖Truth − KF_true‖', color='tab:blue', alpha=0.85)
        plt.plot(N, E_spoof, linewidth=1.6, label='KF Tracking Error (Spoofed GPS): ‖Truth − KF_spoof‖', color='tab:blue', linestyle=':', alpha=0.9)
        plt.plot(N, R_true,  linewidth=1.6, label='Innovation (True GPS): ‖GPS_true − KF_true(pred)‖', color='tab:green', alpha=0.9)
        plt.plot(N, R_spoof, linewidth=1.6, label='Innovation (Spoofed GPS): ‖GPS_spoof − KF_spoof(pred)‖', color='tab:orange', linestyle='--', alpha=0.9)

        plt.title('Attack–Defense Summary: Divergence, Tracking Error, and Innovation', fontsize=14)
        plt.xlabel('GPS Sample Index (n)'); plt.ylabel('Error (m)')
        plt.grid(True, alpha=0.3); plt.legend(loc='upper left', fontsize=9)

        out = f"attack_defense_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(out, dpi=300, bbox_inches='tight'); print(f"Saved: {out}")
        plt.show()

    # ---------------------- Cleanup ----------------------

    def cleanup(self):
        for s in (self.gps_sensor, self.imu_sensor):
            if s:
                try: s.stop()
                except Exception: pass
                try: s.destroy()
                except Exception: pass


def main():
    print("GPS vs IMU Motion with Conference-ready Detection Plots")
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print("Connected to", world.get_map().name)
    except Exception as e:
        print("CARLA connection failed:", e); return

    spawn_points = world.get_map().get_spawn_points()
    vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0]) or world.spawn_actor(vehicle_bp, spawn_points[1])
    time.sleep(1.0)

    analyzer = GPSPositionvsIMUMotion(world, vehicle, duration=30, warmup=50)
    try:
        analyzer.run_comparison()
        analyzer.plot_detection_overview()
        analyzer.plot_attack_defense_summary()
    finally:
        analyzer.cleanup()
        cleanup_traffic_setup(world, vehicle)
        try: vehicle.destroy()
        except Exception: pass
        print("Cleanup complete.")


if __name__ == "__main__":
    main()
