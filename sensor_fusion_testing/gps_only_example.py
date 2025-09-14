"""
GPS-Only System Example
Simple example demonstrating GPS-only positioning with spoofing attacks.
"""
import numpy as np
import time
import sys
import glob
import os
from integration_files.gps_only_system import GPSOnlySystem
from integration_files.gps_spoofer import SpoofingStrategy
from integration_files.traffic_utils import setup_continuous_traffic, cleanup_traffic_setup

# Add the CARLA Python API to PYTHONPATH
try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

def main():
    """Simple example of GPS-only system usage"""
    print("GPS-Only System Example")
    print("======================")
    print("This example shows how to use the GPS-only positioning system")
    print("without IMU or Kalman filter correction.")
    print()
    
    # Connect to CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Get a valid spawn point
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No spawn points available")
        return
    spawn_point = spawn_points[0]
    
    # Spawn a vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    
    # Wait for the vehicle to spawn
    time.sleep(2.0)
    
    # Setup spectator camera
    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(
        transform.location + carla.Location(z=50),
        carla.Rotation(pitch=-90)
    ))
    
    # Setup continuous traffic flow (disable traffic lights)
    setup_continuous_traffic(world, vehicle)
    
    # Enable autopilot
    vehicle.set_autopilot(True)
    
    try:
        # Test without spoofing first
        print("1. Testing GPS-only system WITHOUT spoofing:")
        print("-" * 50)
        print("Watch the CARLA window for position visualization...")
        gps_system_clean = GPSOnlySystem(vehicle, enable_spoofing=False, enable_display=True)
        
        for i in range(10):
            gps_pos = gps_system_clean.get_fused_position()
            true_pos = gps_system_clean.get_true_position()
            
            if gps_pos is not None and true_pos is not None:
                error = np.linalg.norm(true_pos - gps_pos)
                print(f"True: [{true_pos[0]:.2f}, {true_pos[1]:.2f}, {true_pos[2]:.2f}] | "
                      f"GPS: [{gps_pos[0]:.2f}, {gps_pos[1]:.2f}, {gps_pos[2]:.2f}] | "
                      f"Error: {error:.3f}m")
            time.sleep(0.5)
        
        gps_system_clean.cleanup()
        
        # Test with gradual drift spoofing
        print("\n2. Testing GPS-only system WITH gradual drift spoofing:")
        print("-" * 50)
        print("Watch the CARLA window for position visualization...")
        gps_system_spoofed = GPSOnlySystem(vehicle, enable_spoofing=True, 
                                         spoofing_strategy=SpoofingStrategy.GRADUAL_DRIFT,
                                         enable_display=True)
        
        for i in range(20):
            gps_pos = gps_system_spoofed.get_fused_position()
            true_pos = gps_system_spoofed.get_true_position()
            
            if gps_pos is not None and true_pos is not None:
                error = np.linalg.norm(true_pos - gps_pos)
                print(f"True: [{true_pos[0]:.2f}, {true_pos[1]:.2f}, {true_pos[2]:.2f}] | "
                      f"GPS: [{gps_pos[0]:.2f}, {gps_pos[1]:.2f}, {gps_pos[2]:.2f}] | "
                      f"Error: {error:.3f}m")
            time.sleep(0.5)
        
        # Get final statistics
        stats = gps_system_spoofed.get_error_statistics()
        print(f"\nFinal Statistics:")
        print(f"  Mean Position Error: {stats['mean_position_error']:.3f}m")
        print(f"  Max Position Error: {stats['max_position_error']:.3f}m")
        print(f"  Mean Velocity Error: {stats['mean_velocity_error']:.3f}m/s")
        print(f"  Total Samples: {stats['num_samples']}")
        
        gps_system_spoofed.cleanup()
        
        print("\nExample completed successfully!")
        print("Notice how the GPS-only system shows the raw effects of spoofing")
        print("without any mitigation from sensor fusion.")
        
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    finally:
        print("Cleaning up...")
        cleanup_traffic_setup(world, vehicle)
        vehicle.destroy()

if __name__ == '__main__':
    main()
