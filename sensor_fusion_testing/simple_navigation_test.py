#!/usr/bin/env python3
"""
Simple Navigation Example

This script shows how to use the new waypoint navigation system
with sensor fusion in a simple, reusable way. It demonstrates
the basic usage pattern that can be integrated into your existing scripts.
"""

import numpy as np
import time
import sys
import glob
import os

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add the CARLA Python API to PYTHONPATH
try:
    # Add the egg file from the correct CARLA location
    carla_root = os.path.abspath('../../../CARLA_0.9.15/PythonAPI/carla')
    
    # Try to find the egg file with the correct pattern
    egg_pattern = os.path.join(carla_root, 'dist/carla-0.9.15-py3.7-win-amd64.egg')
    if os.path.exists(egg_pattern):
        sys.path.append(egg_pattern)
        print(f"Added CARLA egg path: {egg_pattern}")
    else:
        # Fallback to glob pattern
        egg_files = glob.glob(os.path.join(carla_root, 'dist/carla-*.egg'))
        if egg_files:
            sys.path.append(egg_files[0])
            print(f"Added CARLA egg path: {egg_files[0]}")
        else:
            print("Warning: No CARLA egg file found")
    
    # Add the carla directory for agents module
    carla_path = carla_root
    if os.path.exists(carla_path):
        sys.path.append(carla_path)
        print(f"Added CARLA path: {carla_path}")
    else:
        print(f"Warning: CARLA path not found: {carla_path}")
        
except Exception as e:
    print(f"Warning: Error setting up CARLA paths: {e}")
    print(f"Looked in: {os.path.join(carla_root, 'dist')}")

import carla
from sensor_fusion_testing.integration_files.waypoint_navigator import WaypointNavigator
from sensor_fusion_testing.integration_files.waypoint_generator import WaypointGenerator
from sensor_fusion_testing.integration_files.gps_spoofer import SpoofingStrategy

def setup_carla_and_vehicle():
    """Setup CARLA connection and spawn vehicle"""
    try:
        # Connect to CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        
        # Get spawn point
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0] if spawn_points else carla.Transform(carla.Location(x=0.0, y=0.0, z=1.0))
        
        # Spawn vehicle
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        
        # Wait for vehicle to spawn
        time.sleep(2.0)
        
        print(f"Vehicle spawned at {spawn_point.location}")
        return client, world, vehicle
        
    except Exception as e:
        print(f"Failed to setup CARLA: {e}")
        return None, None, None

def create_navigator(vehicle, enable_spoofing=False):
    """Create a waypoint navigator with sensor fusion"""
    try:
        navigator = WaypointNavigator(
            vehicle=vehicle,
            enable_spoofing=enable_spoofing,
            spoofing_strategy=SpoofingStrategy.GRADUAL_DRIFT,
            waypoint_reach_distance=3.0,
            max_speed=20.0,  # Conservative speed
            pid_config={
                'throttle': {'Kp': 0.3, 'Ki': 0.01, 'Kd': 0.05},
                'steering': {'Kp': 0.8, 'Ki': 0.01, 'Kd': 0.1}
            }
        )
        print("Navigator created successfully")
        return navigator
        
    except Exception as e:
        print(f"Failed to create navigator: {e}")
        return None

def generate_simple_route(world, start_location, end_location):
    """Generate a simple route between two points"""
    try:
        generator = WaypointGenerator(world)
        waypoints = generator.generate_route_waypoints(start_location, end_location)
        
        if waypoints:
            # Visualize the route
            generator.visualize_route(waypoints)
            print(f"Generated route with {len(waypoints)} waypoints")
            return waypoints
        else:
            print("Failed to generate route")
            return []
            
    except Exception as e:
        print(f"Error generating route: {e}")
        return []

def navigate_to_destination(vehicle, waypoints, duration=60):
    """Navigate to destination using sensor fusion"""
    try:
        # Create navigator
        navigator = create_navigator(vehicle)
        if not navigator:
            return False
        
        # Set waypoints and start navigation
        navigator.set_waypoints(waypoints)
        if not navigator.start_navigation():
            print("Failed to start navigation")
            return False
        
        print(f"Starting navigation with {len(waypoints)} waypoints")
        start_time = time.time()
        
        # Navigation loop
        while time.time() - start_time < duration:
            # Perform navigation step
            result = navigator.navigate_step(0.05)  # 20Hz update rate
            
            # Print progress
            if result['status'] == 'navigating':
                print(f"Distance to waypoint: {result['distance_to_waypoint']:.1f}m, "
                      f"Speed: {result['current_speed']:.1f}m/s")
            elif result['status'] == 'completed':
                print("Navigation completed!")
                break
            elif result['status'] == 'no_waypoint':
                print("No waypoint available")
                break
            
            time.sleep(0.05)
        
        # Get final statistics
        stats = navigator.get_navigation_stats()
        print(f"Navigation completed: {stats['waypoints_reached']} waypoints reached")
        
        # Cleanup
        navigator.cleanup()
        return True
        
    except Exception as e:
        print(f"Navigation failed: {e}")
        return False

def main():
    """Main function demonstrating simple navigation"""
    print("=== Simple Sensor Fusion Navigation Example ===")
    
    # Setup CARLA and vehicle
    client, world, vehicle = setup_carla_and_vehicle()
    if not vehicle:
        print("Failed to setup CARLA")
        return
    
    try:
        # Define start and end locations
        start_location = vehicle.get_location()
        end_location = carla.Location(x=start_location.x + 100, y=start_location.y + 100, z=start_location.z)
        
        print(f"Start: {start_location}")
        print(f"End: {end_location}")
        
        # Generate route
        waypoints = generate_simple_route(world, start_location, end_location)
        if not waypoints:
            print("Failed to generate route")
            return
        
        # Navigate to destination
        success = navigate_to_destination(vehicle, waypoints, duration=60)
        
        if success:
            print("✅ Navigation completed successfully!")
        else:
            print("❌ Navigation failed")
            
    except KeyboardInterrupt:
        print("Navigation interrupted by user")
    except Exception as e:
        print(f"Error during navigation: {e}")
    finally:
        # Cleanup
        if vehicle:
            vehicle.destroy()
        print("Cleanup complete")

if __name__ == '__main__':
    main() 