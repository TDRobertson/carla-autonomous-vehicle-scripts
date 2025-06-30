#!/usr/bin/env python3
"""
Simple test script for Innovation-Based GPS Spoofing Mitigation System

This script can be run from the project root directory to test the mitigation system.
"""

import sys
import os
import time
import numpy as np

# Add the sensor_fusion_testing directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sensor_fusion_dir = os.path.join(current_dir, 'sensor_fusion_testing')
integration_dir = os.path.join(sensor_fusion_dir, 'integration_files')
sys.path.insert(0, integration_dir)

from sensor_fusion import SensorFusion, find_spawn_point, setup_spectator
from gps_spoofer import SpoofingStrategy
import carla

def test_innovation_mitigation(attack_type="sudden_jump"):
    """Test the innovation-based mitigation system with different attack types."""
    
    # Map attack type string to enum
    attack_map = {
        "gradual_drift": SpoofingStrategy.GRADUAL_DRIFT,
        "sudden_jump": SpoofingStrategy.SUDDEN_JUMP,
        "random_walk": SpoofingStrategy.RANDOM_WALK,
        "replay": SpoofingStrategy.REPLAY
    }
    
    if attack_type not in attack_map:
        print(f"Unknown attack type: {attack_type}")
        print("Available types: gradual_drift, sudden_jump, random_walk, replay")
        return
    
    strategy = attack_map[attack_type]
    
    # Connect to CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(16.0)
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
    
    # Enable autopilot
    vehicle.set_autopilot(True)
    
    # Initialize sensor fusion with innovation-based mitigation
    fusion = SensorFusion(vehicle, enable_spoofing=True, spoofing_strategy=strategy)
    
    print(f"\n=== Testing Innovation-Based Mitigation: {attack_type.upper()} ===")
    print("Innovation threshold: 5.0 meters")
    print("Suspicious GPS count threshold: 3")
    print("Monitoring GPS acceptance/rejection and bias detection...")
    print("=" * 80)
    
    # Statistics tracking
    total_updates = 0
    gps_rejections = 0
    max_innovation = 0.0
    max_bias = 0.0
    
    try:
        start_time = time.time()
        
        while time.time() - start_time < 60:  # Run for 60 seconds
            # Get current data
            fused_pos = fusion.get_fused_position()
            true_pos = fusion.get_true_position()
            
            if fused_pos is not None and true_pos is not None:
                total_updates += 1
                position_error = np.linalg.norm(fused_pos - true_pos)
                
                # Get monitoring statistics
                innovation_stats = fusion.get_innovation_stats()
                bias_stats = fusion.get_bias_stats()
                gps_stats = fusion.get_gps_stats()
                
                # Update statistics
                if innovation_stats:
                    max_innovation = max(max_innovation, innovation_stats['current_innovation'])
                if bias_stats:
                    max_bias = max(max_bias, bias_stats['current_bias'])
                if gps_stats:
                    gps_rejections = gps_stats['rejected_count']
                
                # Display status every 5 seconds
                if total_updates % 50 == 0:  # Assuming ~10Hz updates
                    print(f"\n--- Status Update (t={time.time()-start_time:.1f}s) ---")
                    print(f"Position Error: {position_error:.3f}m")
                    
                    if innovation_stats:
                        print(f"Current Innovation: {innovation_stats['current_innovation']:.3f}m")
                        print(f"Max Innovation: {max_innovation:.3f}m")
                        print(f"Suspicious GPS Count: {innovation_stats['suspicious_count']}")
                    
                    if bias_stats:
                        print(f"GPS-IMU Bias: {bias_stats['current_bias']:.3f}m")
                        print(f"Max Bias: {max_bias:.3f}m")
                        print(f"Bias Std: {bias_stats['bias_std']:.3f}m")
                    
                    if gps_stats:
                        acceptance_rate = gps_stats['acceptance_rate']
                        print(f"GPS Acceptance Rate: {acceptance_rate:.2%}")
                        print(f"GPS Rejected: {gps_stats['rejected_count']} times")
                        
                        # Highlight when mitigation is working
                        if gps_stats['rejected_count'] > 0:
                            print("🛡️  MITIGATION ACTIVE: GPS data being rejected!")
                        if acceptance_rate < 0.5:
                            print("⚠️  HIGH REJECTION RATE: Attack may be detected!")
                
                time.sleep(0.1)  # 10Hz update rate
        
        # Final summary
        print(f"\n=== FINAL RESULTS: {attack_type.upper()} ===")
        print(f"Total Updates: {total_updates}")
        print(f"GPS Rejections: {gps_rejections}")
        print(f"Rejection Rate: {gps_rejections/total_updates:.2%}")
        print(f"Max Innovation: {max_innovation:.3f}m")
        print(f"Max GPS-IMU Bias: {max_bias:.3f}m")
        
        # Effectiveness assessment
        if gps_rejections > 0:
            print("MITIGATION EFFECTIVE: GPS spoofing detected and mitigated")
        else:
            print("MITIGATION NOT TRIGGERED: Attack may be too subtle")
        
        if max_innovation > 5.0:
            print("LARGE INNOVATIONS: Some GPS readings exceeded threshold")
        else:
            print("INNOVATION CONTROLLED: All GPS readings within threshold")
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        print("Cleaning up...")
        fusion.cleanup()
        vehicle.destroy()

def main():
    """Main function to run the test."""
    if len(sys.argv) > 1:
        attack_type = sys.argv[1]
    else:
        attack_type = "sudden_jump"
    
    test_innovation_mitigation(attack_type)

if __name__ == '__main__':
    main() 