#!/usr/bin/env python3
"""
Navigation Spoofing Test

Runs waypoint navigation with PID control using WaypointNavigator while applying
GPS spoofing. Supports selecting spoofing strategy via CLI:

--1  GRADUAL_DRIFT
--2  SUDDEN_JUMP
--3  RANDOM_WALK
--4  REPLAY
--5  RUN ALL FOUR IN SEQUENCE (each for a fixed segment/time)
"""

import time
import sys
import os
import glob
import argparse

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


def parse_args():
    parser = argparse.ArgumentParser(description="Navigation spoofing test with PID and sensor fusion")
    parser.add_argument('--mode', type=int, choices=[1, 2, 3, 4, 5], required=True,
                        help='1=GRADUAL_DRIFT, 2=SUDDEN_JUMP, 3=RANDOM_WALK, 4=REPLAY, 5=ALL_IN_SEQUENCE')
    parser.add_argument('--duration', type=float, default=60.0, help='Total duration (seconds) for single-mode run')
    parser.add_argument('--speed', type=float, default=20.0, help='Max target speed (m/s)')
    return parser.parse_args()


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
        end_location = carla.Location(x=start_location.x + 200.0, y=start_location.y + 200.0, z=start_location.z)

        generator = WaypointGenerator(world)
        waypoints = generator.generate_route_waypoints(start_location, end_location)

        if waypoints:
            generator.visualize_route(waypoints, life_time=180.0)
            stats = generator.get_route_statistics(waypoints)
            print(f"Generated route with {len(waypoints)} waypoints, total distance ~{stats.get('total_distance', 0.0):.1f} m")
        else:
            print("Failed to generate route")
        return waypoints
    except Exception as e:
        print(f"Error generating route: {e}")
        return []


def run_single_mode(vehicle: carla.Vehicle, waypoints, duration_seconds: float, speed_mps: float, strategy: SpoofingStrategy):
    """Run navigation in a single spoofing mode for the specified duration."""
    navigator = None
    try:
        navigator = WaypointNavigator(
            vehicle=vehicle,
            enable_spoofing=True,
            spoofing_strategy=strategy,
            waypoint_reach_distance=3.0,
            max_speed=speed_mps,
            pid_config={
                'throttle': {'Kp': 0.3, 'Ki': 0.01, 'Kd': 0.05},
                'steering': {'Kp': 0.8, 'Ki': 0.01, 'Kd': 0.1}
            }
        )

        navigator.set_waypoints(waypoints)
        if not navigator.start_navigation():
            print("Failed to start navigation")
            return False

        print(f"Starting spoofed navigation: {strategy.name}")
        start_time = time.time()
        last_mode = None
        dt = 0.05

        while time.time() - start_time < duration_seconds:
            result = navigator.navigate_step(dt)

            # Warmup vs fused mode logging
            nav_start = navigator.get_navigation_stats().get('start_time')
            warmup_sec = getattr(navigator, 'position_warmup_duration', 3.0)
            in_warmup = (nav_start is not None) and ((time.time() - nav_start) < warmup_sec)
            mode = 'warmup(true_position)' if in_warmup else 'fused(kalman)'

            if mode != last_mode:
                print(f"Mode switched to: {mode}")
                last_mode = mode

            if result['status'] == 'navigating':
                print(
                    f"[{strategy.name}] Mode: {mode} | Dist: {result['distance_to_waypoint']:.1f} m | "
                    f"Speed: {result['current_speed']:.1f} m/s | Throttle: {result['throttle']:.2f} | "
                    f"Steer: {result['steering']:.2f}"
                )
            elif result['status'] == 'completed':
                print(f"[{strategy.name}] Route completed early")
                break
            elif result['status'] == 'no_waypoint':
                print(f"[{strategy.name}] No waypoint available; stopping")
                break

            time.sleep(dt)

        stats = navigator.get_navigation_stats()
        print(
            f"[{strategy.name}] Complete. Waypoints: {stats.get('waypoints_reached', 0)} | "
            f"Distance: {stats.get('total_distance', 0.0):.1f} m | Avg speed: {stats.get('average_speed', 0.0):.1f} m/s"
        )
        return True
    except Exception as e:
        print(f"Navigation failed ({strategy.name}): {e}")
        return False
    finally:
        if navigator is not None:
            navigator.cleanup()


def run_all_in_sequence(vehicle: carla.Vehicle, waypoints, speed_mps: float, segment_seconds: float = 45.0):
    """Run all four spoofing strategies sequentially over the same waypoint list."""
    order = [
        SpoofingStrategy.GRADUAL_DRIFT,
        SpoofingStrategy.SUDDEN_JUMP,
        SpoofingStrategy.RANDOM_WALK,
        SpoofingStrategy.REPLAY,
    ]
    for strategy in order:
        print(f"\n=== Running segment with {strategy.name} ===")
        ok = run_single_mode(vehicle, waypoints, duration_seconds=segment_seconds, speed_mps=speed_mps, strategy=strategy)
        if not ok:
            print(f"Segment failed: {strategy.name}")
    print("\nAll segments complete")
    return True


def main():
    args = parse_args()
    print("=== Navigation Spoofing Test ===")
    print(f"Selected mode: {args.mode}")

    client, world, vehicle = setup_carla_and_vehicle()
    if vehicle is None:
        print("Failed to initialize CARLA/vehicle")
        return

    try:
        waypoints = generate_simple_route(world, vehicle)
        if not waypoints:
            print("Could not generate waypoints; exiting")
            return

        if args.mode == 1:
            run_single_mode(vehicle, waypoints, args.duration, args.speed, SpoofingStrategy.GRADUAL_DRIFT)
        elif args.mode == 2:
            run_single_mode(vehicle, waypoints, args.duration, args.speed, SpoofingStrategy.SUDDEN_JUMP)
        elif args.mode == 3:
            run_single_mode(vehicle, waypoints, args.duration, args.speed, SpoofingStrategy.RANDOM_WALK)
        elif args.mode == 4:
            run_single_mode(vehicle, waypoints, args.duration, args.speed, SpoofingStrategy.REPLAY)
        elif args.mode == 5:
            run_all_in_sequence(vehicle, waypoints, speed_mps=args.speed, segment_seconds=max(30.0, args.duration / 4.0))

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        if vehicle is not None:
            vehicle.destroy()
        print("Cleanup complete")


if __name__ == '__main__':
    main()



