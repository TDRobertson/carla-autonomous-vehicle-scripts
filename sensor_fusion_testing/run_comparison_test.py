"""
GPS-Only vs Sensor Fusion Comparison Test Launcher
Runs the comparison test with proper import handling.
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
import json
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

def test_system(system, system_name, strategy, duration=30.0):
    """Test a positioning system with a specific spoofing strategy"""
    print(f"\nTesting {system_name} with {strategy.name}")
    print("-" * 60)
    
    # Set the spoofing strategy
    if hasattr(system, 'spoofer') and system.spoofer is not None:
        system.spoofer.set_strategy(strategy)
    
    # Collect data
    start_time = time.time()
    position_errors = []
    velocity_errors = []
    timestamps = []
    
    while time.time() - start_time < duration:
        true_pos = system.get_true_position()
        fused_pos = system.get_fused_position()
        true_vel = system.get_true_velocity()
        fused_vel = system.get_fused_velocity()
        
        if true_pos is not None and fused_pos is not None:
            position_error = np.linalg.norm(true_pos - fused_pos)
            position_errors.append(position_error)
            timestamps.append(time.time())
            
            if true_vel is not None and fused_vel is not None:
                velocity_error = np.linalg.norm(true_vel - fused_vel)
                velocity_errors.append(velocity_error)
            else:
                velocity_errors.append(0.0)
        
        time.sleep(0.1)
    
    # Calculate statistics
    if position_errors:
        stats = {
            'system': system_name,
            'strategy': strategy.name,
            'duration': duration,
            'samples': len(position_errors),
            'mean_position_error': np.mean(position_errors),
            'std_position_error': np.std(position_errors),
            'max_position_error': np.max(position_errors),
            'min_position_error': np.min(position_errors),
            'mean_velocity_error': np.mean(velocity_errors),
            'std_velocity_error': np.std(velocity_errors),
            'max_velocity_error': np.max(velocity_errors),
            'min_velocity_error': np.min(velocity_errors)
        }
        
        print(f"  Mean Position Error: {stats['mean_position_error']:.3f}m")
        print(f"  Max Position Error: {stats['max_position_error']:.3f}m")
        print(f"  Mean Velocity Error: {stats['mean_velocity_error']:.3f}m/s")
        print(f"  Max Velocity Error: {stats['max_velocity_error']:.3f}m/s")
        print(f"  Total Samples: {stats['samples']}")
        
        return stats
    else:
        print("  No data collected")
        return None

def main():
    """Main comparison function"""
    print("GPS-Only vs Sensor Fusion Comparison Test")
    print("=========================================")
    print("This test compares the effectiveness of GPS spoofing attacks")
    print("on GPS-only vs sensor fusion systems")
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
    
    # Test strategies
    strategies = [
        SpoofingStrategy.GRADUAL_DRIFT,
        SpoofingStrategy.SUDDEN_JUMP,
        SpoofingStrategy.RANDOM_WALK,
        SpoofingStrategy.REPLAY
    ]
    
    all_results = []
    
    try:
        for strategy in strategies:
            print(f"\n{'='*80}")
            print(f"Testing Strategy: {strategy.name}")
            print(f"{'='*80}")
            
            # Test GPS-only system
            print("\n1. GPS-Only System Test")
            print("Watch the CARLA window for position visualization...")
            gps_system = GPSOnlySystem(vehicle, enable_spoofing=True, spoofing_strategy=strategy, enable_display=True)
            gps_stats = test_system(gps_system, "GPS-Only", strategy, duration=30.0)
            if gps_stats:
                all_results.append(gps_stats)
            gps_system.cleanup()
            
            # Wait between tests
            time.sleep(2.0)
            
            # Test Sensor Fusion system
            print("\n2. Sensor Fusion System Test")
            print("Watch the CARLA window for position visualization...")
            fusion_system = SensorFusion(vehicle, enable_spoofing=True, spoofing_strategy=strategy, enable_display=True)
            fusion_stats = test_system(fusion_system, "Sensor Fusion", strategy, duration=30.0)
            if fusion_stats:
                all_results.append(fusion_stats)
            fusion_system.cleanup()
            
            # Wait between strategies
            time.sleep(2.0)
        
        # Analyze and display comparison results
        print(f"\n{'='*80}")
        print("COMPARISON RESULTS")
        print(f"{'='*80}")
        
        # Group results by strategy
        strategy_results = {}
        for result in all_results:
            strategy = result['strategy']
            if strategy not in strategy_results:
                strategy_results[strategy] = {}
            strategy_results[strategy][result['system']] = result
        
        # Display comparison for each strategy
        for strategy, systems in strategy_results.items():
            print(f"\n{strategy} Attack Results:")
            print("-" * 40)
            
            if 'GPS-Only' in systems and 'Sensor Fusion' in systems:
                gps = systems['GPS-Only']
                fusion = systems['Sensor Fusion']
                
                print(f"Position Error (Mean):")
                print(f"  GPS-Only:      {gps['mean_position_error']:.3f}m")
                print(f"  Sensor Fusion: {fusion['mean_position_error']:.3f}m")
                improvement = ((gps['mean_position_error'] - fusion['mean_position_error']) / gps['mean_position_error']) * 100
                print(f"  Improvement:   {improvement:.1f}%")
                
                print(f"Position Error (Max):")
                print(f"  GPS-Only:      {gps['max_position_error']:.3f}m")
                print(f"  Sensor Fusion: {fusion['max_position_error']:.3f}m")
                improvement = ((gps['max_position_error'] - fusion['max_position_error']) / gps['max_position_error']) * 100
                print(f"  Improvement:   {improvement:.1f}%")
                
                print(f"Velocity Error (Mean):")
                print(f"  GPS-Only:      {gps['mean_velocity_error']:.3f}m/s")
                print(f"  Sensor Fusion: {fusion['mean_velocity_error']:.3f}m/s")
                improvement = ((gps['mean_velocity_error'] - fusion['mean_velocity_error']) / gps['mean_velocity_error']) * 100
                print(f"  Improvement:   {improvement:.1f}%")
                
                # Determine effectiveness
                if fusion['mean_position_error'] < 1.0:  # 1 meter threshold
                    print(f"  Attack Status: MITIGATED by Sensor Fusion")
                else:
                    print(f"  Attack Status: SUCCESSFUL against both systems")
        
        # Save results
        output_file = "gps_vs_fusion_comparison.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {output_file}")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        print("Cleaning up...")
        auto_camera.cleanup()
        cleanup_traffic_setup(world, vehicle)
        vehicle.destroy()

if __name__ == '__main__':
    main()
