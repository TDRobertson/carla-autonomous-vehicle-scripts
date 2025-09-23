#!/usr/bin/env python3
"""
Raw Sensor Data Logger for Innovation-Aware Gradual Drift Attack

This script specifically captures and logs the raw GPS and IMU sensor data
BEFORE it's fed into the Kalman filter, allowing analysis of the attack
at the sensor level.

Key Features:
- Captures raw GPS positions (true vs spoofed) before Kalman filtering
- Captures raw IMU accelerometer and gyroscope data
- Logs innovation values and attack parameters
- Creates focused timeseries graphs for sensor-level analysis
- Exports data to CSV for further analysis

Usage:
    python raw_sensor_data_logger.py
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
from integration_files.traffic_utils import setup_continuous_traffic, cleanup_traffic_setup

class RawSensorDataLogger:
    def __init__(self, vehicle, duration=30):
        self.vehicle = vehicle
        self.duration = duration
        self.start_time = None
        
        # Raw sensor data storage
        self.timestamps = []
        self.true_gps_x = []
        self.true_gps_y = []
        self.true_gps_z = []
        self.spoofed_gps_x = []
        self.spoofed_gps_y = []
        self.spoofed_gps_z = []
        self.imu_accel_x = []
        self.imu_accel_y = []
        self.imu_accel_z = []
        self.imu_gyro_x = []
        self.imu_gyro_y = []
        self.imu_gyro_z = []
        self.innovation_values = []
        self.attack_parameters = []
        
        # Initialize spoofer
        self.spoofer = GPSSpoofer([0, 0, 0], strategy=SpoofingStrategy.INNOVATION_AWARE_GRADUAL_DRIFT)
        self.current_innovation = 0.0
        
        # Setup sensors
        self.setup_sensors()
        
    def setup_sensors(self):
        """Setup GPS and IMU sensors for raw data logging"""
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
        """GPS callback - captures raw data before any processing"""
        if self.start_time is None:
            self.start_time = time.time()
            
        current_time = time.time() - self.start_time
        
        # Extract true GPS position
        true_position = np.array([
            data.transform.location.x,
            data.transform.location.y,
            data.transform.location.z
        ])
        
        # Apply spoofing (this is the raw spoofed data before Kalman filtering)
        spoofed_position = self.spoofer.spoof_position(true_position, self.current_innovation)
        
        # Log raw data
        self.timestamps.append(current_time)
        self.true_gps_x.append(true_position[0])
        self.true_gps_y.append(true_position[1])
        self.true_gps_z.append(true_position[2])
        self.spoofed_gps_x.append(spoofed_position[0])
        self.spoofed_gps_y.append(spoofed_position[1])
        self.spoofed_gps_z.append(spoofed_position[2])
        
        # Log innovation and attack parameters
        self.innovation_values.append(self.current_innovation)
        
        # Get attack parameters for analysis
        attack_stats = self.spoofer.get_innovation_stats()
        self.attack_parameters.append({
            'suspicious_counter': attack_stats['suspicious_counter'],
            'is_suspicious': attack_stats['is_suspicious'],
            'mean_innovation': attack_stats['mean_innovation']
        })
        
    def imu_callback(self, data):
        """IMU callback - captures raw IMU data"""
        if self.start_time is None:
            return
            
        current_time = time.time() - self.start_time
        
        # Extract raw IMU data
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
        
        # Log raw IMU data
        self.imu_accel_x.append(accel[0])
        self.imu_accel_y.append(accel[1])
        self.imu_accel_z.append(accel[2])
        self.imu_gyro_x.append(gyro[0])
        self.imu_gyro_y.append(gyro[1])
        self.imu_gyro_z.append(gyro[2])
        
    def run_logging(self):
        """Run the raw sensor data logging"""
        print(f"Starting Raw Sensor Data Logging")
        print(f"Duration: {self.duration} seconds")
        print(f"Vehicle: {self.vehicle.type_id}")
        print(f"Attack Strategy: {self.spoofer.strategy.name}")
        print("="*60)
        
        # Setup continuous traffic (ignores traffic lights)
        setup_continuous_traffic(self.vehicle.get_world(), self.vehicle)
        
        # Enable autopilot for movement
        self.vehicle.set_autopilot(True)
        
        # Run for specified duration
        start_time = time.time()
        while time.time() - start_time < self.duration:
            time.sleep(0.1)
            
        print(f"Logging complete. Collected {len(self.timestamps)} data points")
        
    def create_raw_sensor_plots(self):
        """Create focused plots of raw sensor data"""
        if not self.timestamps:
            print("No data collected for plotting")
            return
            
        # Ensure all arrays have the same length
        min_length = min(
            len(self.timestamps),
            len(self.true_gps_x),
            len(self.spoofed_gps_x),
            len(self.innovation_values),
            len(self.imu_accel_x),
            len(self.imu_gyro_x)
        )
        
        # Convert to numpy arrays with consistent length
        timestamps = np.array(self.timestamps[:min_length])
        true_gps = np.column_stack([
            self.true_gps_x[:min_length], 
            self.true_gps_y[:min_length], 
            self.true_gps_z[:min_length]
        ])
        spoofed_gps = np.column_stack([
            self.spoofed_gps_x[:min_length], 
            self.spoofed_gps_y[:min_length], 
            self.spoofed_gps_z[:min_length]
        ])
        innovation = np.array(self.innovation_values[:min_length])
        
        print(f"Using {min_length} data points for plotting (truncated from {len(self.timestamps)} total)")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Raw Sensor Data Analysis - Innovation-Aware Gradual Drift Attack', fontsize=16, fontweight='bold')
        
        # Plot 1: GPS X position over time
        ax1 = axes[0, 0]
        ax1.plot(timestamps, true_gps[:, 0], 'g-', linewidth=2, label='True GPS X', alpha=0.8)
        ax1.plot(timestamps, spoofed_gps[:, 0], 'r--', linewidth=2, label='Spoofed GPS X', alpha=0.8)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('X Position (m)')
        ax1.set_title('GPS X Position - Raw Sensor Data')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: GPS Y position over time
        ax2 = axes[0, 1]
        ax2.plot(timestamps, true_gps[:, 1], 'g-', linewidth=2, label='True GPS Y', alpha=0.8)
        ax2.plot(timestamps, spoofed_gps[:, 1], 'r--', linewidth=2, label='Spoofed GPS Y', alpha=0.8)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title('GPS Y Position - Raw Sensor Data')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: GPS drift magnitude
        ax3 = axes[0, 2]
        gps_drift = np.linalg.norm(spoofed_gps - true_gps, axis=1)
        ax3.plot(timestamps, gps_drift, 'purple', linewidth=2, label='GPS Drift Magnitude')
        ax3.axhline(y=5.0, color='red', linestyle='--', alpha=0.7, label='Innovation Threshold')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Drift Magnitude (m)')
        ax3.set_title('GPS Spoofing Drift - Raw Sensor Level')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Innovation values
        ax4 = axes[1, 0]
        ax4.plot(timestamps, innovation, 'orange', linewidth=2, label='Innovation Magnitude')
        ax4.axhline(y=5.0, color='red', linestyle='--', alpha=0.7, label='Detection Threshold')
        ax4.axhline(y=3.5, color='yellow', linestyle='--', alpha=0.7, label='Safety Margin')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Innovation (m)')
        ax4.set_title('Innovation Values Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Accelerometer data
        ax5 = axes[1, 1]
        ax5.plot(timestamps, self.imu_accel_x[:min_length], 'r-', label='X Acceleration', alpha=0.8)
        ax5.plot(timestamps, self.imu_accel_y[:min_length], 'g-', label='Y Acceleration', alpha=0.8)
        ax5.plot(timestamps, self.imu_accel_z[:min_length], 'b-', label='Z Acceleration', alpha=0.8)
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Acceleration (m/s²)')
        ax5.set_title('Raw Accelerometer Data')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Gyroscope data
        ax6 = axes[1, 2]
        ax6.plot(timestamps, self.imu_gyro_x[:min_length], 'r-', label='X Angular Velocity', alpha=0.8)
        ax6.plot(timestamps, self.imu_gyro_y[:min_length], 'g-', label='Y Angular Velocity', alpha=0.8)
        ax6.plot(timestamps, self.imu_gyro_z[:min_length], 'b-', label='Z Angular Velocity', alpha=0.8)
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Angular Velocity (rad/s)')
        ax6.set_title('Raw Gyroscope Data')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"raw_sensor_data_{timestamp_str}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Raw sensor data plot saved as: {filename}")
        
        plt.show()
        
    def export_to_csv(self):
        """Export raw sensor data to CSV for further analysis"""
        if not self.timestamps:
            print("No data to export")
            return
            
        # Ensure all arrays have the same length
        min_length = min(
            len(self.timestamps),
            len(self.true_gps_x),
            len(self.spoofed_gps_x),
            len(self.innovation_values),
            len(self.imu_accel_x),
            len(self.imu_gyro_x)
        )
        
        # Create DataFrame with consistent length
        data = {
            'timestamp': self.timestamps[:min_length],
            'true_gps_x': self.true_gps_x[:min_length],
            'true_gps_y': self.true_gps_y[:min_length],
            'true_gps_z': self.true_gps_z[:min_length],
            'spoofed_gps_x': self.spoofed_gps_x[:min_length],
            'spoofed_gps_y': self.spoofed_gps_y[:min_length],
            'spoofed_gps_z': self.spoofed_gps_z[:min_length],
            'imu_accel_x': self.imu_accel_x[:min_length],
            'imu_accel_y': self.imu_accel_y[:min_length],
            'imu_accel_z': self.imu_accel_z[:min_length],
            'imu_gyro_x': self.imu_gyro_x[:min_length],
            'imu_gyro_y': self.imu_gyro_y[:min_length],
            'imu_gyro_z': self.imu_gyro_z[:min_length],
            'innovation': self.innovation_values[:min_length]
        }
        
        # Add attack parameters
        for i, params in enumerate(self.attack_parameters[:min_length]):
            data[f'suspicious_counter_{i}'] = [params['suspicious_counter']] * min_length
            data[f'is_suspicious_{i}'] = [params['is_suspicious']] * min_length
            data[f'mean_innovation_{i}'] = [params['mean_innovation']] * min_length
        
        df = pd.DataFrame(data)
        
        # Save to CSV
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"raw_sensor_data_{timestamp_str}.csv"
        df.to_csv(filename, index=False)
        print(f"Raw sensor data exported to: {filename}")
        
    def print_raw_data_summary(self):
        """Print summary of raw sensor data"""
        if not self.timestamps:
            print("No data collected for analysis")
            return
            
        # Ensure all arrays have the same length
        min_length = min(
            len(self.timestamps),
            len(self.true_gps_x),
            len(self.spoofed_gps_x),
            len(self.innovation_values),
            len(self.imu_accel_x),
            len(self.imu_gyro_x)
        )
        
        # Convert to numpy arrays with consistent length
        timestamps = np.array(self.timestamps[:min_length])
        true_gps = np.column_stack([
            self.true_gps_x[:min_length], 
            self.true_gps_y[:min_length], 
            self.true_gps_z[:min_length]
        ])
        spoofed_gps = np.column_stack([
            self.spoofed_gps_x[:min_length], 
            self.spoofed_gps_y[:min_length], 
            self.spoofed_gps_z[:min_length]
        ])
        innovation = np.array(self.innovation_values[:min_length])
        
        print("\n" + "="*80)
        print("RAW SENSOR DATA ANALYSIS SUMMARY")
        print("="*80)
        
        # Basic statistics
        print(f"Data Collection Duration: {timestamps[-1]:.2f} seconds")
        print(f"Data Points Collected: {len(timestamps)}")
        print(f"Average Update Rate: {len(timestamps)/timestamps[-1]:.2f} Hz")
        
        # GPS spoofing statistics
        gps_drift = np.linalg.norm(spoofed_gps - true_gps, axis=1)
        print(f"\nGPS Spoofing Statistics (Raw Sensor Level):")
        print(f"  Mean GPS Drift: {np.mean(gps_drift):.3f} m")
        print(f"  Max GPS Drift: {np.max(gps_drift):.3f} m")
        print(f"  Final GPS Drift: {gps_drift[-1]:.3f} m")
        print(f"  Drift Rate: {np.mean(np.diff(gps_drift)):.3f} m/s")
        
        # Innovation statistics
        print(f"\nInnovation Statistics:")
        print(f"  Mean Innovation: {np.mean(innovation):.3f} m")
        print(f"  Max Innovation: {np.max(innovation):.3f} m")
        print(f"  Innovation > Threshold (5m): {np.sum(innovation > 5.0)} times")
        print(f"  Innovation > Safety Margin (3.5m): {np.sum(innovation > 3.5)} times")
        
        # IMU statistics
        accel_magnitude = np.sqrt(
            np.array(self.imu_accel_x[:min_length])**2 + 
            np.array(self.imu_accel_y[:min_length])**2 + 
            np.array(self.imu_accel_z[:min_length])**2
        )
        gyro_magnitude = np.sqrt(
            np.array(self.imu_gyro_x[:min_length])**2 + 
            np.array(self.imu_gyro_y[:min_length])**2 + 
            np.array(self.imu_gyro_z[:min_length])**2
        )
        
        print(f"\nIMU Statistics (Raw Sensor Level):")
        print(f"  Mean Acceleration Magnitude: {np.mean(accel_magnitude):.3f} m/s²")
        print(f"  Mean Gyroscope Magnitude: {np.mean(gyro_magnitude):.3f} rad/s")
        print(f"  Max Acceleration: {np.max(accel_magnitude):.3f} m/s²")
        print(f"  Max Gyroscope: {np.max(gyro_magnitude):.3f} rad/s")
        
        print("="*80)
        
    def cleanup(self):
        """Cleanup sensors and resources"""
        if hasattr(self, 'gps_sensor'):
            self.gps_sensor.destroy()
        if hasattr(self, 'imu_sensor'):
            self.imu_sensor.destroy()

def main():
    """Main function to run the raw sensor data logging"""
    print("Raw Sensor Data Logger - Innovation-Aware Gradual Drift Attack")
    print("="*70)
    
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
        # Create logger
        logger = RawSensorDataLogger(vehicle, duration=30)
        
        # Run logging
        logger.run_logging()
        
        # Create plots
        logger.create_raw_sensor_plots()
        
        # Export data
        logger.export_to_csv()
        
        # Print summary
        logger.print_raw_data_summary()
        
    except KeyboardInterrupt:
        print("\nLogging interrupted by user")
    except Exception as e:
        print(f"Error during logging: {e}")
    finally:
        # Cleanup
        print("Cleaning up...")
        logger.cleanup()
        cleanup_traffic_setup(world, vehicle)
        vehicle.destroy()
        print("Cleanup complete")

if __name__ == "__main__":
    main()
