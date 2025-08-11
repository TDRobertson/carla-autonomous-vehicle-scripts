#!/usr/bin/env python3
"""
Navigation Warmup Test

Runs a simple waypoint navigation using WaypointNavigator and logs when the
controller is using true position (warmup) vs Kalman-fused position after the
warmup period. Confirms PID control and the position source swap after ~3s.
"""

import time
import sys
import os
import glob

# Ensure package imports work when run directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add the CARLA Python API to PYTHONPATH (Windows-friendly)
try:
    carla_root = os.path.abspath('../../../CARLA_0.9.15/PythonAPI/carla')

    egg_pattern = os.path.join(carla_root, 'dist/carla-0.9.15-py3.7-win-amd64.egg')
    if os.path.exists(egg_pattern):
        sys.path.append(egg_pattern)
        print(f"Added CARLA egg path: {egg_pattern}")
    else:
        egg_files = glob.glob(os.path.join(carla_root, 'dist/carla-*.egg'))
        if egg_files:
            sys.path.append(egg_files[0])
            print(f"Added CARLA egg path: {egg_files[0]}")
        else:
            print("Warning: No CARLA egg file found")

    carla_path = carla_root
    if os.path.exists(carla_path):
        sys.path.append(carla_path)
        print(f"Added CARLA path: {carla_path}")
    else:
        print(f"Warning: CARLA path not found: {carla_path}")
except Exception as e:
    print(f"Warning: Error setting up CARLA paths: {e}")

import carla  # noqa: E402
from sensor_fusion_testing.integration_files.waypoint_navigator import WaypointNavigator  # noqa: E402
from sensor_fusion_testing.integration_files.waypoint_generator import WaypointGenerator  # noqa: E402
from sensor_fusion_testing.integration_files.gps_spoofer import SpoofingStrategy  # noqa: E402


def setup_carla_and_vehicle():
    """Connect to CARLA and spawn a vehicle."""
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()

        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0] if spawn_points else carla.Transform(carla.Location(x=0.0, y=0.0, z=1.0))

        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)

        # Small delay to ensure actor is ready
        time.sleep(2.0)
        print(f"Vehicle spawned at {spawn_point.location}")
        return client, world, vehicle
    except Exception as e:
        print(f"Failed to setup CARLA: {e}")
        return None, None, None


def generate_simple_route(world: carla.World, vehicle: carla.Vehicle):
    """Generate a simple route from the current vehicle location to an offset destination."""
    try:
        start_location = vehicle.get_location()
        end_location = carla.Location(x=start_location.x + 150.0, y=start_location.y + 120.0, z=start_location.z)

        generator = WaypointGenerator(world)
        waypoints = generator.generate_route_waypoints(start_location, end_location)

        if waypoints:
            generator.visualize_route(waypoints, life_time=120.0)
            stats = generator.get_route_statistics(waypoints)
            print(f"Generated route with {len(waypoints)} waypoints, total distance ~{stats.get('total_distance', 0.0):.1f} m")
        else:
            print("Failed to generate route")
        return waypoints
    except Exception as e:
        print(f"Error generating route: {e}")
        return []


def run_navigation(vehicle: carla.Vehicle, waypoints, duration_seconds: float = 60.0, enable_spoofing: bool = False):
    """Navigate using WaypointNavigator and log warmup vs fused mode."""
    navigator = None
    try:
        navigator = WaypointNavigator(
            vehicle=vehicle,
            enable_spoofing=enable_spoofing,
            spoofing_strategy=SpoofingStrategy.GRADUAL_DRIFT,
            waypoint_reach_distance=3.0,
            max_speed=20.0,
            pid_config={
                'throttle': {'Kp': 0.3, 'Ki': 0.01, 'Kd': 0.05},
                'steering': {'Kp': 0.8, 'Ki': 0.01, 'Kd': 0.1}
            }
        )

        navigator.set_waypoints(waypoints)
        if not navigator.start_navigation():
            print("Failed to start navigation")
            return False

        print(f"Starting navigation with {len(waypoints)} waypoints")
        start_time = time.time()
        last_mode = None

        # 20 Hz loop
        dt = 0.05
        while time.time() - start_time < duration_seconds:
            result = navigator.navigate_step(dt)

            # Determine mode (warmup vs fused) based on navigator's own clock
            nav_start = navigator.get_navigation_stats().get('start_time')
            warmup_sec = getattr(navigator, 'position_warmup_duration', 3.0)
            now = time.time()
            in_warmup = (nav_start is not None) and ((now - nav_start) < warmup_sec)
            mode = 'warmup(true_position)' if in_warmup else 'fused(kalman)'

            if mode != last_mode:
                print(f"Mode switched to: {mode}")
                last_mode = mode

            if result['status'] == 'navigating':
                print(
                    f"Mode: {mode} | Dist: {result['distance_to_waypoint']:.1f} m | "
                    f"Speed: {result['current_speed']:.1f} m/s | Throttle: {result['throttle']:.2f} | "
                    f"Steer: {result['steering']:.2f}"
                )
            elif result['status'] == 'completed':
                print("Navigation completed!")
                break
            elif result['status'] == 'no_waypoint':
                print("No waypoint available")
                break

            time.sleep(dt)

        stats = navigator.get_navigation_stats()
        print(
            f"Run complete. Waypoints reached: {stats.get('waypoints_reached', 0)} | "
            f"Total distance: {stats.get('total_distance', 0.0):.1f} m | "
            f"Avg speed: {stats.get('average_speed', 0.0):.1f} m/s"
        )
        return True
    except Exception as e:
        print(f"Navigation failed: {e}")
        return False
    finally:
        if navigator is not None:
            navigator.cleanup()


def main():
    print("=== Navigation Warmup Test ===")
    client, world, vehicle = setup_carla_and_vehicle()
    if vehicle is None:
        print("Failed to initialize CARLA/vehicle")
        return

    try:
        waypoints = generate_simple_route(world, vehicle)
        if not waypoints:
            print("Could not generate waypoints; exiting")
            return

        success = run_navigation(vehicle, waypoints, duration_seconds=60.0, enable_spoofing=False)
        print("Success" if success else "Failed")
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        if vehicle is not None:
            vehicle.destroy()
        print("Cleanup complete")


if __name__ == '__main__':
    main()


