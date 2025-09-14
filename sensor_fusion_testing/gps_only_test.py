"""
GPS-Only System Test
Tests GPS spoofing attacks without IMU or Kalman filter correction.
This script demonstrates the raw effects of GPS spoofing on a GPS-only positioning system.
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

def find_spawn_point(world):
    """Find a valid spawn point for the vehicle"""
    spawn_points = world.get_map().get_spawn_points()
    if spawn_points:
        return spawn_points[0]
    return carla.Transform(carla.Location(x=0.0, y=0.0, z=1.0))

def setup_spectator(world, vehicle):
    """Setup the spectator camera to view the vehicle"""
    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(
        transform.location + carla.Location(z=50),
        carla.Rotation(pitch=-90)
    ))

def main():
    """Main function for GPS-only testing"""
    print("GPS-Only System Test")
    print("===================")
    print("This test shows the raw effects of GPS spoofing without sensor fusion")
    print("Compare these results with the sensor fusion test to see the benefits")
    print()
    
    # Connect to CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Get a valid spawn point
    spawn_point = find_spawn_point(world)
    
    # Spawn a vehicle
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    
    # Wait for the vehicle to spawn
    time.sleep(2.0)
    
    # Setup spectator camera
    setup_spectator(world, vehicle)
    
    # Setup continuous traffic flow (disable traffic lights)
    setup_continuous_traffic(world, vehicle)
    
    # Enable autopilot
    vehicle.set_autopilot(True)
    
    # Test different spoofing strategies
    strategies = [
        (SpoofingStrategy.GRADUAL_DRIFT, "Gradual Drift Attack"),
        (SpoofingStrategy.SUDDEN_JUMP, "Sudden Jump Attack"),
        (SpoofingStrategy.RANDOM_WALK, "Random Walk Attack"),
        (SpoofingStrategy.REPLAY, "Replay Attack")
    ]
    
    try:
        for strategy, description in strategies:
            print(f"\nTesting: {description}")
            print("-" * 50)
            
            # Initialize GPS-only system with current strategy and display
            gps_system = GPSOnlySystem(vehicle, enable_spoofing=True, spoofing_strategy=strategy, enable_display=True)
            
            # Test for 30 seconds
            start_time = time.time()
            test_duration = 30.0
            
            while time.time() - start_time < test_duration:
                gps_pos = gps_system.get_fused_position()
                true_pos = gps_system.get_true_position()
                
                if gps_pos is not None and true_pos is not None:
                    position_error = gps_system.get_position_error()
                    velocity_error = gps_system.get_velocity_error()
                    
                    print(f"True: [{true_pos[0]:.2f}, {true_pos[1]:.2f}, {true_pos[2]:.2f}] | "
                          f"GPS: [{gps_pos[0]:.2f}, {gps_pos[1]:.2f}, {gps_pos[2]:.2f}] | "
                          f"Error: {position_error:.3f}m | Vel Error: {velocity_error:.3f}m/s")
                
                time.sleep(0.1)
            
            # Get final statistics
            stats = gps_system.get_error_statistics()
            print(f"\n{description} Results:")
            print(f"  Mean Position Error: {stats['mean_position_error']:.3f}m")
            print(f"  Max Position Error: {stats['max_position_error']:.3f}m")
            print(f"  Mean Velocity Error: {stats['mean_velocity_error']:.3f}m/s")
            print(f"  Max Velocity Error: {stats['max_velocity_error']:.3f}m/s")
            print(f"  Total Samples: {stats['num_samples']}")
            
            # Cleanup
            gps_system.cleanup()
            
            # Wait between tests
            time.sleep(2.0)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        print("Cleaning up...")
        cleanup_traffic_setup(world, vehicle)
        vehicle.destroy()

if __name__ == '__main__':
    main()
