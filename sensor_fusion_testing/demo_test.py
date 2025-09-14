#!/usr/bin/env python3
"""
Comprehensive demonstration test for 5 showcase attacks.
This test showcases:
1. GPS-only vs Sensor Fusion comparison for each attack type
2. Clear demonstration of mitigation vs detection capabilities
3. Innovation-aware attack effectiveness
"""

import sys
import os

# Add the current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Now we can import from integration_files
from integration_files.gps_spoofer import GPSSpoofer, SpoofingStrategy
from integration_files.gps_only_system import GPSOnlySystem
from integration_files.sensor_fusion import SensorFusion
from integration_files.traffic_utils import setup_continuous_traffic, cleanup_traffic_setup
from integration_files.camera_utils import setup_simple_overhead_camera

import numpy as np
import time
import glob
import carla

def run_demo_section(title, description, duration=20):
    """Run a demo section with clear title and description"""
    print(f"\n{'='*60}")
    print(f"DEMO SECTION: {title}")
    print(f"{'='*60}")
    print(f"Description: {description}")
    print(f"Duration: {duration} seconds")
    print(f"{'='*60}")
    print("Starting in 1 second...")
    time.sleep(1)

def test_attack_comparison(world, vehicle, auto_camera, attack_strategy, attack_name, duration=20):
    """Test a specific attack on both GPS-only and Sensor Fusion systems"""
    
    print(f"\n{'='*80}")
    print(f"ATTACK COMPARISON: {attack_name}")
    print(f"{'='*80}")
    print("This demonstrates the difference between GPS-only and Sensor Fusion systems")
    print("for the same attack type.")
    print(f"{'='*80}")
    
    try:
        # Test GPS-only system first
        run_demo_section(f"GPS-Only: {attack_name}", 
                       f"Showing raw GPS spoofing effects with NO sensor fusion mitigation. "
                       f"Expected: Large position errors visible as red sphere deviates from green sphere. "
                       f"Replay attack replays previous GPS positions, creating position jumps.",
                       duration)
        
        gps_system = GPSOnlySystem(
            vehicle=vehicle,
            enable_spoofing=True,
            spoofing_strategy=attack_strategy,
            enable_display=True
        )
        
        start_time = time.time()
        while time.time() - start_time < duration:
            time.sleep(0.1)
        
        gps_system.cleanup()
        time.sleep(2.0)  # Brief pause between tests
        
        # Test Sensor Fusion system
        run_demo_section(f"Sensor Fusion: {attack_name}", 
                       f"Showing how GPS+IMU+Kalman filter handles this attack. "
                       f"Expected: Different behavior based on attack type - mitigation or detection. "
                       f"Replay attack should be mitigated by sensor fusion.",
                       duration)
        
        sensor_fusion = SensorFusion(
            vehicle=vehicle,
            enable_spoofing=True,
            spoofing_strategy=attack_strategy,
            enable_display=True
        )
        
        start_time = time.time()
        while time.time() - start_time < duration:
            time.sleep(0.1)
        
        sensor_fusion.cleanup()
        time.sleep(2.0)  # Brief pause between attack types
        
    except Exception as e:
        print(f"Error during {attack_name} test: {e}")

def test_innovation_aware_attack(world, vehicle, auto_camera):
    """Test innovation-aware attack - subtle and effective against sensor fusion"""
    
    print(f"\n{'='*80}")
    print("INNOVATION-AWARE ATTACK - Advanced Sophisticated Attack")
    print(f"{'='*80}")
    print("This demonstrates how innovation-aware attacks can be subtle yet effective")
    print("against sensor fusion systems by adapting to avoid detection thresholds.")
    print(f"{'='*80}")
    
    try:
        run_demo_section("Innovation-Aware Gradual Drift Attack", 
                       "Showing how innovation-aware attacks adapt to avoid detection. "
                       "Expected: Gradual position drift that adapts when approaching detection threshold.",
                       20)  # Standard duration
        
        sensor_fusion = SensorFusion(
            vehicle=vehicle,
            enable_spoofing=True,
            spoofing_strategy=SpoofingStrategy.INNOVATION_AWARE_GRADUAL_DRIFT,
            enable_display=True
        )
        
        start_time = time.time()
        while time.time() - start_time < 20:
            time.sleep(0.1)
        
        sensor_fusion.cleanup()
        
    except Exception as e:
        print(f"Error in innovation-aware test: {e}")

def main():
    """Run the complete professor demonstration with logical attack ordering"""
    
    print("PROFESSOR DEMONSTRATION: GPS Spoofing Attack Analysis")
    print("="*80)
    print("This demonstration shows GPS-only vs Sensor Fusion comparison for each attack type.")
    print("Logical ordering: Each attack is tested on both systems to show the differences.")
    print("="*80)
    
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
    
    # Setup camera and traffic
    auto_camera = setup_simple_overhead_camera(world, vehicle, height=50.0)
    setup_continuous_traffic(world, vehicle)
    vehicle.set_autopilot(True)
    
    try:
        # Test each attack type with logical ordering
        attack_tests = [
            (SpoofingStrategy.GRADUAL_DRIFT, "Gradual Drift Attack", 20),
            (SpoofingStrategy.SUDDEN_JUMP, "Sudden Jump Attack", 20),
            (SpoofingStrategy.RANDOM_WALK, "Random Walk Attack", 20),
            (SpoofingStrategy.REPLAY, "Replay Attack", 20)
        ]
        
        for attack_strategy, attack_name, duration in attack_tests:
            test_attack_comparison(world, vehicle, auto_camera, attack_strategy, attack_name, duration)
        
        # Test innovation-aware attack separately
        test_innovation_aware_attack(world, vehicle, auto_camera)
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETE!")
        print("="*80)
        print("Key takeaways from the logical attack ordering:")
        print("• Gradual Drift: GPS-only shows large errors, Sensor Fusion detects but doesn't fully mitigate")
        print("• Sudden Jump: GPS-only shows large errors, Sensor Fusion detects but doesn't fully mitigate")
        print("• Random Walk: GPS-only shows large errors, Sensor Fusion significantly mitigates")
        print("• Replay: GPS-only shows large errors, Sensor Fusion significantly mitigates")
        print("• Innovation-Aware: Sophisticated attack that adapts to avoid detection")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\nDemonstration interrupted by user")
    except Exception as e:
        print(f"Error during demonstration: {e}")
    finally:
        # Cleanup
        print("Cleaning up...")
        auto_camera.cleanup()
        cleanup_traffic_setup(world, vehicle)
        vehicle.destroy()
        print("Cleanup complete")

if __name__ == "__main__":
    main()