"""
GPS-Only Sequential Attack Tester
Tests GPS spoofing attacks without IMU or Kalman filter correction.
Recommended: Use run_all.py or launch sync.py and fpv_ghost.py in separate terminals before running this script for proper visualization.
"""
import numpy as np
import time
import json
import os
import sys
import glob
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from .gps_spoofer import GPSSpoofer, SpoofingStrategy
from .gps_only_system import GPSOnlySystem
from .traffic_utils import setup_continuous_traffic, cleanup_traffic_setup

# Add the CARLA Python API to PYTHONPATH
try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

@dataclass
class AttackSequence:
    strategy: SpoofingStrategy
    duration: float  # Duration in seconds
    description: str

def find_spawn_point(world):
    """Find a valid spawn point for the vehicle."""
    spawn_points = world.get_map().get_spawn_points()
    if spawn_points:
        return spawn_points[0]  # Use the first spawn point
    return carla.Transform(carla.Location(x=0.0, y=0.0, z=1.0))  # Fallback spawn point

def setup_spectator(world, vehicle):
    """Setup the spectator camera to view the vehicle spawn point."""
    spectator = world.get_spectator()
    transform = vehicle.get_transform()
    spectator.set_transform(carla.Transform(
        transform.location + carla.Location(z=50),
        carla.Rotation(pitch=-90)
    ))

class GPSOnlySequentialAttackTester:
    def __init__(self, gps_system: GPSOnlySystem):
        self.gps_system = gps_system
        self.attack_sequences = []
        self.current_sequence_index = 0
        self.start_time = None
        self.sequence_start_time = None
        self.is_running = False
        self.results = {}
        
        # Enhanced analysis data for GPS-only testing
        self.attack_results: Dict[str, List] = {
            'true_positions': [],
            'gps_positions': [],
            'position_errors': [],
            'timestamps': [],
            'true_velocities': [],
            'gps_velocities': [],
            'velocity_errors': [],
            'attack_type': [],
            'attack_duration': [],
            'vehicle_speed': [],
            'vehicle_acceleration': [],
            'vehicle_angular_velocity': [],
            'time_since_last_update': [],
            'update_frequency': [],
            'position_error_rate': [],
            'velocity_error_rate': []
        }
        
    def add_attack_sequence(self, strategy: SpoofingStrategy, duration: float, description: str):
        self.attack_sequences.append(AttackSequence(strategy, duration, description))
        
    def start_test(self):
        if not self.attack_sequences:
            print("No attack sequences defined")
            return
        self.is_running = True
        self.current_sequence_index = 0
        self.start_time = time.time()
        self.sequence_start_time = self.start_time
        self.gps_system.spoofer.set_strategy(self.attack_sequences[0].strategy)
        print(f"Starting GPS-only attack sequence: {self.attack_sequences[0].description}")
        
    def update(self, timestamp: float):
        if not self.is_running:
            return
        current_attack = self.get_current_attack()
        if current_attack is None:
            self.is_running = False
            return
        elapsed_time = timestamp - self.sequence_start_time
        if elapsed_time >= current_attack.duration:
            print(f"Completed GPS-only attack sequence: {current_attack.description}")
            self.current_sequence_index += 1
            if self.current_sequence_index < len(self.attack_sequences):
                self.sequence_start_time = timestamp
                self.gps_system.spoofer.set_strategy(self.attack_sequences[self.current_sequence_index].strategy)
                print(f"Starting GPS-only attack sequence: {self.attack_sequences[self.current_sequence_index].description}")
            else:
                self.is_running = False
                print("All GPS-only attack sequences completed")
                return
        # Get current data
        true_position = self.gps_system.get_true_position()
        gps_position = self.gps_system.get_fused_position()
        true_velocity = self.gps_system.get_true_velocity()
        gps_velocity = self.gps_system.get_fused_velocity()
        
        if true_position is None or gps_position is None:
            return
            
        position_error = np.linalg.norm(true_position - gps_position)
        velocity_error = np.linalg.norm(true_velocity - gps_velocity) if true_velocity is not None and gps_velocity is not None else 0.0
        
        if current_attack.strategy not in self.results:
            self.results[current_attack.strategy] = {
                'true_positions': [],
                'gps_positions': [],
                'true_velocities': [],
                'gps_velocities': [],
                'position_errors': [],
                'velocity_errors': [],
                'timestamps': []
            }
        self.results[current_attack.strategy]['true_positions'].append(true_position)
        self.results[current_attack.strategy]['gps_positions'].append(gps_position)
        if true_velocity is not None:
            self.results[current_attack.strategy]['true_velocities'].append(true_velocity)
        if gps_velocity is not None:
            self.results[current_attack.strategy]['gps_velocities'].append(gps_velocity)
        self.results[current_attack.strategy]['position_errors'].append(position_error)
        self.results[current_attack.strategy]['velocity_errors'].append(velocity_error)
        self.results[current_attack.strategy]['timestamps'].append(timestamp)
    
    def get_current_attack(self) -> AttackSequence:
        if self.current_sequence_index < len(self.attack_sequences):
            return self.attack_sequences[self.current_sequence_index]
        return None
    
    def get_results(self) -> Dict:
        return self.results
    
    def save_results(self, output_dir: str = "gps_only_test_results"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for strategy, data in self.results.items():
            strategy_dir = os.path.join(output_dir, f"GPSOnly_{strategy.name}")
            if not os.path.exists(strategy_dir):
                os.makedirs(strategy_dir)
            with open(os.path.join(strategy_dir, 'raw_data.json'), 'w') as f:
                json.dump({k: np.array(v).tolist() for k, v in data.items()}, f, indent=2)
        print(f"GPS-only results saved to {output_dir}")
    
    def analyze_results(self):
        print("\nGPS-Only Attack Test Results Analysis:")
        print("=====================================")
        for strategy, data in self.results.items():
            print(f"\nAttack: {strategy.name} (GPS-Only)")
            errors = np.array(data['position_errors'])
            print(f"  Mean Position Error: {np.mean(errors):.3f}m")
            print(f"  Std Position Error: {np.std(errors):.3f}m")
            print(f"  Max Position Error: {np.max(errors):.3f}m")
            print(f"  Min Position Error: {np.min(errors):.3f}m")
            vels = np.array(data['velocity_errors'])
            print(f"  Mean Velocity Error: {np.mean(vels):.3f}m/s")
            print(f"  Std Velocity Error: {np.std(vels):.3f}m/s")
            print(f"  Max Velocity Error: {np.max(vels):.3f}m/s")
            print(f"  Min Velocity Error: {np.min(vels):.3f}m/s")
            
            # Calculate attack effectiveness
            if np.mean(errors) > 1.0:  # 1 meter threshold
                print(f"  Attack Effectiveness: SUCCESSFUL")
            else:
                print(f"  Attack Effectiveness: MITIGATED")

def main():
    """Main function for GPS-only sequential attack testing"""
    # Do NOT launch visualizations as subprocesses here.
    # Use run_all.py or launch sync.py and fpv_ghost.py in separate terminals for visualization.

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
    
    # Setup spectator camera once at the start
    setup_spectator(world, vehicle)
    
    # Setup continuous traffic flow (disable traffic lights)
    setup_continuous_traffic(world, vehicle)
    
    # Enable autopilot
    vehicle.set_autopilot(True)
    
    # Initialize GPS-only system with spoofing enabled
    gps_system = GPSOnlySystem(vehicle, enable_spoofing=True)
    
    # Create and configure the sequential attack tester
    tester = GPSOnlySequentialAttackTester(gps_system)
    
    # Add attack sequences
    tester.add_attack_sequence(SpoofingStrategy.GRADUAL_DRIFT, 30.0, "GPS-Only Gradual Drift Attack")
    tester.add_attack_sequence(SpoofingStrategy.SUDDEN_JUMP, 30.0, "GPS-Only Sudden Jump Attack")
    tester.add_attack_sequence(SpoofingStrategy.RANDOM_WALK, 30.0, "GPS-Only Random Walk Attack")
    tester.add_attack_sequence(SpoofingStrategy.REPLAY, 30.0, "GPS-Only Replay Attack")
    
    # Start the test
    tester.start_test()
    
    try:
        print("GPS-Only Sequential Attack Test Started")
        print("=====================================")
        print("This test shows the raw effects of GPS spoofing without sensor fusion")
        print("Compare these results with the sensor fusion test to see the benefits")
        print()
        
        # Run the test until all sequences are complete
        while tester.is_running:
            tester.update(time.time())
            time.sleep(0.1)
        
        # Analyze results after all sequences are complete
        print("\nGPS-Only Test Results Analysis:")
        print("==============================")
        tester.analyze_results()
        
        # Save results
        tester.save_results()
        
    except KeyboardInterrupt:
        print("\nGPS-only test interrupted by user")
        tester.analyze_results()
        tester.save_results()
    finally:
        print("Cleaning up...")
        gps_system.cleanup()
        cleanup_traffic_setup(world, vehicle)
        vehicle.destroy()

if __name__ == '__main__':
    main()
