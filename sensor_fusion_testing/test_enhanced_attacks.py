#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced GPS spoofing attacks.
This script shows the difference between the old subtle attacks and the new aggressive ones.
"""

import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now we can import from integration_files
from integration_files.gps_spoofer import GPSSpoofer, SpoofingStrategy
from integration_files.gps_only_system import GPSOnlySystem
from integration_files.traffic_utils import setup_continuous_traffic, cleanup_traffic_setup
from integration_files.camera_utils import setup_simple_overhead_camera

import numpy as np
import time
import glob
import carla

def test_enhanced_attacks():
    """Test the enhanced GPS spoofing attacks"""
    
    print("Enhanced GPS Spoofing Attack Test")
    print("=" * 50)
    print("This test demonstrates the enhanced, more aggressive GPS spoofing attacks.")
    print("You should now see much more dramatic effects!")
    print()
    
    # Connect to CARLA
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        # Get the map
        map_name = world.get_map().name
        print(f"Connected to CARLA world: {map_name}")
        
    except Exception as e:
        print(f"Failed to connect to CARLA: {e}")
        print("Make sure CARLA is running on localhost:2000")
        return
    
    # Get spawn points
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        print("No spawn points found!")
        return
    
    # Spawn vehicle
    vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    spawn_point = spawn_points[0]
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
        # Test each attack strategy
        strategies = [
            (SpoofingStrategy.GRADUAL_DRIFT, "Enhanced Gradual Drift"),
            (SpoofingStrategy.SUDDEN_JUMP, "Enhanced Sudden Jump"),
            (SpoofingStrategy.RANDOM_WALK, "Enhanced Random Walk"),
            (SpoofingStrategy.REPLAY, "Enhanced Replay Attack")
        ]
        
        for strategy, name in strategies:
            print(f"\nTesting {name}...")
            print("-" * 30)
            
            # Create GPS-only system with enhanced attacks
            gps_system = GPSOnlySystem(
                world=world,
                vehicle=vehicle,
                strategy=strategy,
                enable_display=True
            )
            
            # Run test for 15 seconds
            start_time = time.time()
            while time.time() - start_time < 15.0:
                time.sleep(0.1)
            
            # Cleanup
            gps_system.cleanup()
            
            print(f"Completed {name} test")
            time.sleep(2.0)  # Brief pause between tests
        
        print("\nAll enhanced attack tests completed!")
        print("You should have seen much more dramatic effects:")
        print("- Gradual Drift: Figure-8 pattern with exponential growth")
        print("- Sudden Jump: Larger, more frequent jumps over time")
        print("- Random Walk: Increasing step size over time")
        print("- Replay: Faster replay with larger delays")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
    finally:
        # Cleanup
        print("Cleaning up...")
        auto_camera.cleanup()
        cleanup_traffic_setup(world, vehicle)
        vehicle.destroy()
        print("Cleanup complete")

if __name__ == "__main__":
    test_enhanced_attacks()
