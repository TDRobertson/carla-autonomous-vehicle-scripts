#!/usr/bin/env python3
"""
Innovation-Aware Gradual Drift Timeseries Analysis

This script captures and visualizes the raw GPS and IMU values before they're fed into the Kalman filter
during an innovation-aware gradual drift attack. It creates comprehensive timeseries graphs showing:

1. Raw GPS positions (true vs spoofed)
2. IMU accelerometer and gyroscope data
3. Innovation values over time
4. Position errors and drift patterns
5. Attack effectiveness metrics

Usage:
    python innovation_aware_timeseries_analysis.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import time
import glob
import carla
from datetime import datetime

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Add the CARLA Python API to PYTHONPATH
try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

from integration_files.gps_spoofer import GPSSpoofer, SpoofingStrategy
from integration_files.advanced_kalman_filter import AdvancedKalmanFilter
from integration_files.traffic_utils import setup_continuous_traffic, cleanup_traffic_setup

class InnovationAwareTimeseriesAnalyzer:
    def __init__(self, vehicle, duration=30):
        self.vehicle = vehicle
        self.duration = duration
        self.start_time = None
        
        # Data storage for timeseries analysis
        self.timestamps = []
        self.true_gps_positions = []
        self.spoofed_gps_positions = []
        self.imu_accelerometer = []
        self.imu_gyroscope = []
        self.innovation_values = []
        self.fused_positions = []
        self.position_errors = []
        self.velocity_errors = []
        
        # Initialize components
        self.kf = AdvancedKalmanFilter()
        self.spoofer = GPSSpoofer([0, 0, 0], strategy=SpoofingStrategy.INNOVATION_AWARE_GRADUAL_DRIFT)
        
        # Setup sensors
        self.setup_sensors()
        
    def setup_sensors(self):
        """Setup GPS and IMU sensors with data logging"""
        # Setup GPS
        gps_bp = self.vehicle.get_world().get_blueprint_library().find('sensor.other.gnss')
        self.gps_sensor = self.vehicle.get_world().spawn_actor(
            gps_bp,
            carla.Transform(carla.Location(x=0.0, z=0.0)),
            attach_to=self.vehicle
        )
        self.gps_sensor.listen(self.gps_callback)
        
        # Setup IMU
        imu_bp = self.vehicle.get_world().get_blueprint_library().find('sensor.other.imu')
        self.imu_sensor = self.vehicle.get_world().spawn_actor(
            imu_bp,
            carla.Transform(carla.Location(x=0.0, z=0.0)),
            attach_to=self.vehicle
        )
        self.imu_sensor.listen(self.imu_callback)
        
    def gps_callback(self, data):
        """GPS callback with comprehensive data logging"""
        if self.start_time is None:
            self.start_time = time.time()
            
        current_time = time.time() - self.start_time
        
        # Store true position
        true_position = np.array([
            data.transform.location.x,
            data.transform.location.y,
            data.transform.location.z
        ])
        
        # Apply spoofing
        innovation = getattr(self, 'current_innovation', 0.0)
        spoofed_position = self.spoofer.spoof_position(true_position, innovation)
        
        # Log data
        self.timestamps.append(current_time)
        self.true_gps_positions.append(true_position.copy())
        self.spoofed_gps_positions.append(spoofed_position.copy())
        
        # Update Kalman filter and get innovation
        innovation = self.kf.update_with_gps(spoofed_position)
        self.innovation_values.append(innovation)
        self.fused_positions.append(self.kf.position.copy())
        
        # Calculate position error
        position_error = np.linalg.norm(self.kf.position - true_position)
        self.position_errors.append(position_error)
        
        # Store current innovation for spoofer
        self.current_innovation = innovation
        
    def imu_callback(self, data):
        """IMU callback with comprehensive data logging"""
        if self.start_time is None:
            return
            
        current_time = time.time() - self.start_time
        
        # Extract IMU data
        accel = np.array([
            data.accelerometer.x,
            data.accelerometer.y,
            data.accelerometer.z
        ])
        
        gyro = np.array([
            data.gyroscope.x,
            data.gyroscope.y,
            data.gyroscope.z
        ])
        
        # Log IMU data
        self.imu_accelerometer.append(accel.copy())
        self.imu_gyroscope.append(gyro.copy())
        
        # Update Kalman filter prediction step
        timestamp = self.vehicle.get_world().get_snapshot().timestamp.elapsed_seconds
        self.kf.predict(data, timestamp)
        
    def run_analysis(self):
        """Run the complete timeseries analysis"""
        print(f"Starting Innovation-Aware Gradual Drift Timeseries Analysis")
        print(f"Duration: {self.duration} seconds")
        print(f"Vehicle: {self.vehicle.type_id}")
        print("="*60)
        
        # Setup continuous traffic (ignores traffic lights)
        setup_continuous_traffic(self.vehicle.get_world(), self.vehicle)
        
        # Enable autopilot for movement
        self.vehicle.set_autopilot(True)
        
        # Run for specified duration
        start_time = time.time()
        while time.time() - start_time < self.duration:
            time.sleep(0.1)
            
        print(f"Analysis complete. Collected {len(self.timestamps)} data points")
        
    def create_timeseries_plots(self):
        """Create comprehensive timeseries plots"""
        if not self.timestamps:
            print("No data collected for plotting")
            return
            
        # Ensure all arrays have the same length by using the minimum length
        min_length = min(
            len(self.timestamps),
            len(self.true_gps_positions),
            len(self.spoofed_gps_positions),
            len(self.fused_positions),
            len(self.innovation_values),
            len(self.position_errors)
        )
        
        # Truncate all arrays to the same length
        timestamps = np.array(self.timestamps[:min_length])
        true_gps = np.array(self.true_gps_positions[:min_length])
        spoofed_gps = np.array(self.spoofed_gps_positions[:min_length])
        fused_pos = np.array(self.fused_positions[:min_length])
        innovation = np.array(self.innovation_values[:min_length])
        position_errors = np.array(self.position_errors[:min_length])
        
        print(f"Using {min_length} data points for plotting (truncated from {len(self.timestamps)} total)")
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Innovation-Aware Gradual Drift Attack Timeseries Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Position trajectories (X-Y plane)
        ax1 = axes[0, 0]
        ax1.plot(true_gps[:, 0], true_gps[:, 1], 'g-', linewidth=2, label='True GPS Position', alpha=0.8)
        ax1.plot(spoofed_gps[:, 0], spoofed_gps[:, 1], 'r--', linewidth=2, label='Spoofed GPS Position', alpha=0.8)
        ax1.plot(fused_pos[:, 0], fused_pos[:, 1], 'b-', linewidth=2, label='Fused Position (Kalman)', alpha=0.8)
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title('Position Trajectories (X-Y Plane)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axis('equal')
        
        # Plot 2: Position error over time
        ax2 = axes[0, 1]
        ax2.plot(timestamps, position_errors, 'purple', linewidth=2, label='Position Error')
        ax2.axhline(y=5.0, color='red', linestyle='--', alpha=0.7, label='Innovation Threshold (5m)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position Error (m)')
        ax2.set_title('Position Error Over Time')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Innovation values over time
        ax3 = axes[1, 0]
        ax3.plot(timestamps, innovation, 'orange', linewidth=2, label='Innovation Magnitude')
        ax3.axhline(y=5.0, color='red', linestyle='--', alpha=0.7, label='Detection Threshold (5m)')
        ax3.axhline(y=3.5, color='yellow', linestyle='--', alpha=0.7, label='Safety Margin (3.5m)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Innovation Magnitude (m)')
        ax3.set_title('Innovation Values Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: GPS drift magnitude over time
        ax4 = axes[1, 1]
        gps_drift = np.linalg.norm(spoofed_gps - true_gps, axis=1)
        ax4.plot(timestamps, gps_drift, 'red', linewidth=2, label='GPS Drift Magnitude')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Drift Magnitude (m)')
        ax4.set_title('GPS Spoofing Drift Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: X position comparison
        ax5 = axes[2, 0]
        ax5.plot(timestamps, true_gps[:, 0], 'g-', linewidth=2, label='True X Position')
        ax5.plot(timestamps, spoofed_gps[:, 0], 'r--', linewidth=2, label='Spoofed X Position')
        ax5.plot(timestamps, fused_pos[:, 0], 'b-', linewidth=2, label='Fused X Position')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('X Position (m)')
        ax5.set_title('X Position Comparison Over Time')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Y position comparison
        ax6 = axes[2, 1]
        ax6.plot(timestamps, true_gps[:, 1], 'g-', linewidth=2, label='True Y Position')
        ax6.plot(timestamps, spoofed_gps[:, 1], 'r--', linewidth=2, label='Spoofed Y Position')
        ax6.plot(timestamps, fused_pos[:, 1], 'b-', linewidth=2, label='Fused Y Position')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Y Position (m)')
        ax6.set_title('Y Position Comparison Over Time')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"innovation_aware_timeseries_{timestamp_str}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Timeseries plot saved as: {filename}")
        
        plt.show()
        
    def create_imu_analysis_plots(self):
        """Create IMU data analysis plots"""
        if not self.imu_accelerometer:
            print("No IMU data collected for plotting")
            return
            
        # Ensure all arrays have the same length
        min_length = min(
            len(self.timestamps),
            len(self.imu_accelerometer),
            len(self.imu_gyroscope)
        )
        
        # Convert to numpy arrays with consistent length
        timestamps = np.array(self.timestamps[:min_length])
        accel_data = np.array(self.imu_accelerometer[:min_length])
        gyro_data = np.array(self.imu_gyroscope[:min_length])
        
        print(f"Using {min_length} data points for IMU plotting")
        
        # Create IMU analysis figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('IMU Data Analysis During Innovation-Aware Attack', fontsize=14, fontweight='bold')
        
        # Accelerometer data
        ax1 = axes[0, 0]
        ax1.plot(timestamps, accel_data[:, 0], 'r-', label='X Acceleration', alpha=0.8)
        ax1.plot(timestamps, accel_data[:, 1], 'g-', label='Y Acceleration', alpha=0.8)
        ax1.plot(timestamps, accel_data[:, 2], 'b-', label='Z Acceleration', alpha=0.8)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Acceleration (m/s²)')
        ax1.set_title('Accelerometer Data')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gyroscope data
        ax2 = axes[0, 1]
        ax2.plot(timestamps, gyro_data[:, 0], 'r-', label='X Angular Velocity', alpha=0.8)
        ax2.plot(timestamps, gyro_data[:, 1], 'g-', label='Y Angular Velocity', alpha=0.8)
        ax2.plot(timestamps, gyro_data[:, 2], 'b-', label='Z Angular Velocity', alpha=0.8)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Angular Velocity (rad/s)')
        ax2.set_title('Gyroscope Data')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Acceleration magnitude
        ax3 = axes[1, 0]
        accel_magnitude = np.linalg.norm(accel_data, axis=1)
        ax3.plot(timestamps, accel_magnitude, 'purple', linewidth=2, label='Acceleration Magnitude')
        ax3.axhline(y=9.81, color='red', linestyle='--', alpha=0.7, label='Gravity (9.81 m/s²)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Acceleration Magnitude (m/s²)')
        ax3.set_title('Acceleration Magnitude Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Angular velocity magnitude
        ax4 = axes[1, 1]
        gyro_magnitude = np.linalg.norm(gyro_data, axis=1)
        ax4.plot(timestamps, gyro_magnitude, 'orange', linewidth=2, label='Angular Velocity Magnitude')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Angular Velocity Magnitude (rad/s)')
        ax4.set_title('Angular Velocity Magnitude Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"imu_analysis_{timestamp_str}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"IMU analysis plot saved as: {filename}")
        
        plt.show()
        
    def print_analysis_summary(self):
        """Print comprehensive analysis summary"""
        if not self.timestamps:
            print("No data collected for analysis")
            return
            
        # Ensure all arrays have the same length
        min_length = min(
            len(self.timestamps),
            len(self.true_gps_positions),
            len(self.spoofed_gps_positions),
            len(self.fused_positions),
            len(self.innovation_values),
            len(self.position_errors)
        )
        
        # Convert to numpy arrays with consistent length
        timestamps = np.array(self.timestamps[:min_length])
        true_gps = np.array(self.true_gps_positions[:min_length])
        spoofed_gps = np.array(self.spoofed_gps_positions[:min_length])
        fused_pos = np.array(self.fused_positions[:min_length])
        innovation = np.array(self.innovation_values[:min_length])
        position_errors = np.array(self.position_errors[:min_length])
        
        print("\n" + "="*80)
        print("INNOVATION-AWARE GRADUAL DRIFT ATTACK ANALYSIS SUMMARY")
        print("="*80)
        
        # Basic statistics
        print(f"Analysis Duration: {timestamps[-1]:.2f} seconds")
        print(f"Data Points Collected: {len(timestamps)}")
        print(f"Average Update Rate: {len(timestamps)/timestamps[-1]:.2f} Hz")
        
        # Position error statistics
        print(f"\nPosition Error Statistics:")
        print(f"  Mean Position Error: {np.mean(position_errors):.3f} m")
        print(f"  Max Position Error: {np.max(position_errors):.3f} m")
        print(f"  Final Position Error: {position_errors[-1]:.3f} m")
        
        # Innovation statistics
        print(f"\nInnovation Statistics:")
        print(f"  Mean Innovation: {np.mean(innovation):.3f} m")
        print(f"  Max Innovation: {np.max(innovation):.3f} m")
        print(f"  Innovation > Threshold (5m): {np.sum(innovation > 5.0)} times")
        print(f"  Innovation > Safety Margin (3.5m): {np.sum(innovation > 3.5)} times")
        
        # GPS drift statistics
        gps_drift = np.linalg.norm(spoofed_gps - true_gps, axis=1)
        print(f"\nGPS Spoofing Statistics:")
        print(f"  Mean GPS Drift: {np.mean(gps_drift):.3f} m")
        print(f"  Max GPS Drift: {np.max(gps_drift):.3f} m")
        print(f"  Final GPS Drift: {gps_drift[-1]:.3f} m")
        
        # Attack effectiveness
        final_position_error = position_errors[-1]
        if final_position_error > 5.0:
            effectiveness = "SUCCESSFUL"
        elif final_position_error > 2.0:
            effectiveness = "PARTIALLY SUCCESSFUL"
        else:
            effectiveness = "MITIGATED"
            
        print(f"\nAttack Effectiveness: {effectiveness}")
        print(f"  Final Position Error: {final_position_error:.3f} m")
        
        # Innovation threshold analysis
        threshold_violations = np.sum(innovation > 5.0)
        safety_violations = np.sum(innovation > 3.5)
        print(f"\nDetection Analysis:")
        print(f"  Innovation Threshold Violations: {threshold_violations} ({threshold_violations/len(innovation)*100:.1f}%)")
        print(f"  Safety Margin Violations: {safety_violations} ({safety_violations/len(innovation)*100:.1f}%)")
        
        print("="*80)
        
    def cleanup(self):
        """Cleanup sensors and resources"""
        if hasattr(self, 'gps_sensor'):
            self.gps_sensor.destroy()
        if hasattr(self, 'imu_sensor'):
            self.imu_sensor.destroy()

def main():
    """Main function to run the timeseries analysis"""
    print("Innovation-Aware Gradual Drift Timeseries Analysis")
    print("="*60)
    
    # Connect to CARLA
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print(f"Connected to CARLA world: {world.get_map().name}")
    except Exception as e:
        print(f"Failed to connect to CARLA: {e}")
        return
    
    # Get spawn points and spawn vehicle
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
    time.sleep(2.0)
    
    try:
        # Create analyzer
        analyzer = InnovationAwareTimeseriesAnalyzer(vehicle, duration=30)
        
        # Run analysis
        analyzer.run_analysis()
        
        # Create plots
        analyzer.create_timeseries_plots()
        analyzer.create_imu_analysis_plots()
        
        # Print summary
        analyzer.print_analysis_summary()
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user")
    except Exception as e:
        print(f"Error during analysis: {e}")
    finally:
        # Cleanup
        print("Cleaning up...")
        analyzer.cleanup()
        cleanup_traffic_setup(world, vehicle)
        vehicle.destroy()
        print("Cleanup complete")

if __name__ == "__main__":
    main()
