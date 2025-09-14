"""
Position Display Demo Launcher
Runs the position display demo with proper import handling.
"""
import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now we can import from integration_files
from integration_files.gps_only_system import GPSOnlySystem
from integration_files.sensor_fusion import SensorFusion
from integration_files.gps_spoofer import SpoofingStrategy
from integration_files.traffic_utils import setup_continuous_traffic, cleanup_traffic_setup
from integration_files.camera_utils import setup_auto_follow_camera, setup_simple_overhead_camera

import numpy as np
import time
import glob
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
    """Main function for position display demonstration"""
    print("Position Display Demo")
    print("===================")
    print("This demo shows real-time position visualization:")
    print("- Green sphere: True vehicle position")
    print("- Red sphere: Sensor-estimated position")
    print("- Yellow line: Position error")
    print("- Console output: Detailed position information")
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
    
    # Setup overhead camera (looking down at vehicle)
    auto_camera = setup_simple_overhead_camera(world, vehicle, height=50.0)
    
    # Setup continuous traffic flow (disable traffic lights)
    setup_continuous_traffic(world, vehicle)
    
    # Enable autopilot
    vehicle.set_autopilot(True)
    
    try:
        # Test GPS-only system with display
        print("1. Testing GPS-Only System with Position Display")
        print("=" * 50)
        gps_system = GPSOnlySystem(vehicle, enable_spoofing=True, 
                                 spoofing_strategy=SpoofingStrategy.GRADUAL_DRIFT,
                                 enable_display=True)
        
        print("Watch the CARLA window for:")
        print("- Green sphere: True position")
        print("- Red sphere: GPS position")
        print("- Yellow line: Position error")
        print("- Console: Detailed position data")
        print()
        print("Press Ctrl+C to stop...")
        
        # Run for 30 seconds
        start_time = time.time()
        while time.time() - start_time < 30.0:
            time.sleep(0.1)
        
        gps_system.cleanup()
        
        # Wait between tests
        time.sleep(2.0)
        
        # Test Sensor Fusion system with display
        print("\n2. Testing Sensor Fusion System with Position Display")
        print("=" * 50)
        fusion_system = SensorFusion(vehicle, enable_spoofing=True, 
                                   spoofing_strategy=SpoofingStrategy.GRADUAL_DRIFT,
                                   enable_display=True)
        
        print("Watch the CARLA window for:")
        print("- Green sphere: True position")
        print("- Red sphere: Fused position")
        print("- Yellow line: Position error")
        print("- Console: Detailed position data with innovation values")
        print()
        print("Press Ctrl+C to stop...")
        
        # Run for 30 seconds
        start_time = time.time()
        while time.time() - start_time < 30.0:
            time.sleep(0.1)
        
        fusion_system.cleanup()
        
        print("\nDemo completed!")
        print("You should have seen:")
        print("- Visual markers in the CARLA world showing position differences")
        print("- Console output showing detailed position information")
        print("- Different behavior between GPS-only and sensor fusion systems")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    finally:
        print("Cleaning up...")
        auto_camera.cleanup()
        cleanup_traffic_setup(world, vehicle)
        vehicle.destroy()

if __name__ == '__main__':
    main()
