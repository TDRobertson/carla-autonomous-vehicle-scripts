#!/usr/bin/env python3
"""
GPS Position vs IMU Motion Comparison - Innovation-Aware Gradual Drift Attack (orientation + smoothing)

Changes:
- Plot IMU compass heading (degrees) instead of Z angular velocity.
- Smooth IMU signals (accel, gyro, compass) with Savitzky–Golay filter.
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

# Add the CARLA Python API to PYTHONPATH
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

# Your project imports
from integration_files.gps_spoofer import GPSSpoofer, SpoofingStrategy
from integration_files.traffic_utils import setup_continuous_traffic, cleanup_traffic_setup


class GPSPositionvsIMUMotion:
    def __init__(self, world, vehicle, duration=30):
        self.world = world
        self.vehicle = vehicle
        self.duration = duration
        self.t0 = None

        # GPS
        self.gps_t, self.true_x, self.true_y, self.spoof_x, self.spoof_y = [], [], [], [], []

        # IMU
        self.imu_t = []
        self.ax, self.ay, self.az = [], [], []
        self.gx, self.gy, self.gz = [], [], []
        self.compass = []

        self.spoofer = GPSSpoofer([0, 0, 0], strategy=SpoofingStrategy.INNOVATION_AWARE_GRADUAL_DRIFT)
        self.current_innovation = 0.0

        self.gps_sensor = None
        self.imu_sensor = None
        self.setup_sensors()

    def setup_sensors(self):
        bp_lib = self.world.get_blueprint_library()

        # GNSS
        gps_bp = bp_lib.find('sensor.other.gnss')
        gps_bp.set_attribute('sensor_tick', '0.10')  # 10 Hz
        self.gps_sensor = self.world.spawn_actor(
            gps_bp, carla.Transform(carla.Location(x=0.0, z=2.0)), attach_to=self.vehicle
        )
        self.gps_sensor.listen(self.gps_callback)

        # IMU
        imu_bp = bp_lib.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', '0.02')  # 50 Hz
        self.imu_sensor = self.world.spawn_actor(
            imu_bp, carla.Transform(carla.Location(x=0.0, z=2.0)), attach_to=self.vehicle
        )
        self.imu_sensor.listen(self.imu_callback)

    def _set_t0_if_needed(self, sensor_timestamp):
        if self.t0 is None:
            self.t0 = sensor_timestamp

    def gps_callback(self, data):
        self._set_t0_if_needed(data.timestamp)
        t = data.timestamp - self.t0
        tf = self.vehicle.get_transform()
        px, py = tf.location.x, tf.location.y
        spoofed = self.spoofer.spoof_position(np.array([px, py, 0.0]), self.current_innovation)
        self.gps_t.append(t)
        self.true_x.append(px)
        self.true_y.append(py)
        self.spoof_x.append(float(spoofed[0]))
        self.spoof_y.append(float(spoofed[1]))

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

    def run_comparison(self):
        print(f"Running for {self.duration}s with {self.vehicle.type_id}")
        setup_continuous_traffic(self.world, self.vehicle)
        self.vehicle.set_autopilot(True)
        start = time.time()
        while time.time() - start < self.duration:
            time.sleep(0.05)

    def _align_and_smooth(self):
        if len(self.gps_t) < 2 or len(self.imu_t) < 2:
            return None
        T = np.array(self.gps_t)
        ti = np.array(self.imu_t)

        def interp(series):
            return np.interp(T, ti, np.array(series, dtype=float))

        # interpolate
        imu_ax, imu_ay, imu_az = interp(self.ax), interp(self.ay), interp(self.az)
        imu_compass = interp(self.compass)

        # unwrap heading and convert to degrees
        heading_deg = np.rad2deg(np.unwrap(imu_compass))

        # smooth with Savitzky–Golay (skip if too few points)
        def smooth(sig):
            if len(sig) > 31:
                return savgol_filter(sig, 31, 3)
            return sig

        return (
            T,
            smooth(imu_ax),
            smooth(imu_ay),
            smooth(imu_az),
            smooth(heading_deg),
        )

    def create_comparison_plots(self):
        aligned = self._align_and_smooth()
        if aligned is None:
            print("Not enough data")
            return

        T, imu_ax, imu_ay, imu_az, heading_deg = aligned
        true_x = np.array(self.true_x[:len(T)])
        true_y = np.array(self.true_y[:len(T)])
        spoof_x = np.array(self.spoof_x[:len(T)])
        spoof_y = np.array(self.spoof_y[:len(T)])

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('GPS Position vs IMU Motion - Innovation-Aware Gradual Drift Attack',
                     fontsize=16, fontweight='bold')

        # GPS X
        axes[0, 0].plot(T, true_x, label='True GPS X')
        axes[0, 0].plot(T, spoof_x, '--', label='Spoofed GPS X')
        axes[0, 0].set_title('GPS X Position (Affected by Spoofing)')
        axes[0, 0].set_xlabel('Time (s)'); axes[0, 0].set_ylabel('X (m)')
        axes[0, 0].grid(True, alpha=0.3); axes[0, 0].legend()

        # GPS Y
        axes[0, 1].plot(T, true_y, label='True GPS Y')
        axes[0, 1].plot(T, spoof_y, '--', label='Spoofed GPS Y')
        axes[0, 1].set_title('GPS Y Position (Affected by Spoofing)')
        axes[0, 1].set_xlabel('Time (s)'); axes[0, 1].set_ylabel('Y (m)')
        axes[0, 1].grid(True, alpha=0.3); axes[0, 1].legend()

        # GPS Trajectory
        axes[0, 2].plot(true_x, true_y, label='True GPS')
        axes[0, 2].plot(spoof_x, spoof_y, '--', label='Spoofed GPS')
        axes[0, 2].set_title('GPS Position Trajectory')
        axes[0, 2].set_xlabel('X (m)'); axes[0, 2].set_ylabel('Y (m)')
        axes[0, 2].axis('equal'); axes[0, 2].grid(True, alpha=0.3); axes[0, 2].legend()

        # IMU Ax
        axes[1, 0].plot(T, imu_ax, label='X Accel')
        axes[1, 0].set_title('IMU X Acceleration (Unaffected)')
        axes[1, 0].set_xlabel('Time (s)'); axes[1, 0].set_ylabel('Accel (m/s²)')
        axes[1, 0].grid(True, alpha=0.3); axes[1, 0].legend()

        # IMU Ay
        axes[1, 1].plot(T, imu_ay, label='Y Accel')
        axes[1, 1].set_title('IMU Y Acceleration (Unaffected)')
        axes[1, 1].set_xlabel('Time (s)'); axes[1, 1].set_ylabel('Accel (m/s²)')
        axes[1, 1].grid(True, alpha=0.3); axes[1, 1].legend()

        # IMU Compass Heading
        axes[1, 2].plot(T, heading_deg, label='Heading (deg)')
        axes[1, 2].set_title('IMU Compass Orientation (Yaw)')
        axes[1, 2].set_xlabel('Time (s)'); axes[1, 2].set_ylabel('Heading (°)')
        axes[1, 2].grid(True, alpha=0.3); axes[1, 2].legend()

        plt.tight_layout()
        out = f"gps_vs_imu_orientation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(out, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {out}")
        plt.show()

    def cleanup(self):
        for s in (self.gps_sensor, self.imu_sensor):
            if s:
                try: s.stop()
                except: pass
                s.destroy()


def main():
    print("GPS vs IMU Motion with Orientation + Smoothing")
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print("Connected to", world.get_map().name)
    except Exception as e:
        print("CARLA connect failed:", e)
        return

    spawn_points = world.get_map().get_spawn_points()
    vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0]) or world.spawn_actor(vehicle_bp, spawn_points[1])
    time.sleep(1.0)

    analyzer = GPSPositionvsIMUMotion(world, vehicle, duration=30)
    try:
        analyzer.run_comparison()
        analyzer.create_comparison_plots()
    finally:
        analyzer.cleanup()
        cleanup_traffic_setup(world, vehicle)
        vehicle.destroy()


if __name__ == "__main__":
    main()
