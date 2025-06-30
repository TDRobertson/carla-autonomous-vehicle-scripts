"""
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
from gps_spoofer import GPSSpoofer, SpoofingStrategy
from sensor_fusion import SensorFusion

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

class SequentialAttackTester:
    def __init__(self, sensor_fusion: SensorFusion):
        self.sensor_fusion = sensor_fusion
        self.attack_sequences = []
        self.current_sequence_index = 0
        self.start_time = None
        self.sequence_start_time = None
        self.is_running = False
        self.results = {}
        # All visualization is now handled by real-time scripts
        # No DataVisualizer or static plotting here
        
        # Enhanced analysis data for ML training
        self.attack_results: Dict[str, List] = {
            'true_positions': [],
            'fused_positions': [],
            'errors': [],
            'timestamps': [],
            'true_velocities': [],
            'fused_velocities': [],
            'velocity_errors': [],
            'imu_accelerations': [],
            'imu_gyroscopes': [],
            'kalman_gains': [],
            'innovation_vectors': [],
            'innovation_covariances': [],
            'position_variance': [],
            'velocity_variance': [],
            'acceleration_variance': [],
            'attack_type': [],
            'attack_confidence': [],
            'attack_duration': [],
            'vehicle_speed': [],
            'vehicle_acceleration': [],
            'vehicle_angular_velocity': [],
            'time_since_last_update': [],
            'update_frequency': [],
            'position_error_rate': [],
            'velocity_error_rate': [],
            'acceleration_error_rate': [],
            # innovation-based mitigation data
            'innovation_magnitudes': [],
            'gps_accepted': [],
            'gps_rejected': [],
            'suspicious_gps_count': [],
            'gps_imu_bias': [],
            'bias_std': [],
            'mitigation_triggered': [],
            'fallback_to_imu': []
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
        self.sensor_fusion.spoofer.set_strategy(self.attack_sequences[0].strategy)
        print(f"Starting attack sequence: {self.attack_sequences[0].description}")
        
    def update(self, timestamp: float):
        if not self.is_running:
            return
        current_attack = self.get_current_attack()
        if current_attack is None:
            self.is_running = False
            return
        elapsed_time = timestamp - self.sequence_start_time
        if elapsed_time >= current_attack.duration:
            print(f"Completed attack sequence: {current_attack.description}")
            self.current_sequence_index += 1
            if self.current_sequence_index < len(self.attack_sequences):
                self.sequence_start_time = timestamp
                self.sensor_fusion.spoofer.set_strategy(self.attack_sequences[self.current_sequence_index].strategy)
                print(f"Starting attack sequence: {self.attack_sequences[self.current_sequence_index].description}")
            else:
                self.is_running = False
                print("All attack sequences completed")
                return
        # Get current data
        true_position = self.sensor_fusion.get_true_position()
        fused_position = self.sensor_fusion.get_fused_position()
        true_velocity = self.sensor_fusion.get_true_velocity()
        fused_velocity = self.sensor_fusion.get_fused_velocity()
        imu_data = self.sensor_fusion.get_imu_data()
        kalman_metrics = self.sensor_fusion.get_kalman_metrics()
        
        # Get innovation-based mitigation data
        innovation_stats = self.sensor_fusion.get_innovation_stats()
        bias_stats = self.sensor_fusion.get_bias_stats()
        gps_stats = self.sensor_fusion.get_gps_stats()
        
        if true_position is None or fused_position is None:
            return
        position_error = np.linalg.norm(true_position - fused_position)
        velocity_error = np.linalg.norm(true_velocity - fused_velocity) if true_velocity is not None and fused_velocity is not None else 0.0
        if current_attack.strategy not in self.results:
            self.results[current_attack.strategy] = {
                'true_positions': [],
                'fused_positions': [],
                'true_velocities': [],
                'fused_velocities': [],
                'imu_data': [],
                'kalman_metrics': [],
                'position_errors': [],
                'velocity_errors': [],
                'timestamps': [],
                # New innovation-based mitigation data
                'innovation_magnitudes': [],
                'gps_accepted': [],
                'gps_rejected': [],
                'suspicious_gps_count': [],
                'gps_imu_bias': [],
                'bias_std': [],
                'mitigation_triggered': [],
                'fallback_to_imu': []
            }
        self.results[current_attack.strategy]['true_positions'].append(true_position)
        self.results[current_attack.strategy]['fused_positions'].append(fused_position)
        if true_velocity is not None:
            self.results[current_attack.strategy]['true_velocities'].append(true_velocity)
        if fused_velocity is not None:
            self.results[current_attack.strategy]['fused_velocities'].append(fused_velocity)
        if imu_data is not None:
            self.results[current_attack.strategy]['imu_data'].append(imu_data)
        if kalman_metrics is not None:
            self.results[current_attack.strategy]['kalman_metrics'].append(kalman_metrics)
        self.results[current_attack.strategy]['position_errors'].append(position_error)
        self.results[current_attack.strategy]['velocity_errors'].append(velocity_error)
        self.results[current_attack.strategy]['timestamps'].append(timestamp)
        
        # Store innovation-based mitigation data
        if innovation_stats:
            self.results[current_attack.strategy]['innovation_magnitudes'].append(innovation_stats['current_innovation'])
            self.results[current_attack.strategy]['suspicious_gps_count'].append(innovation_stats['suspicious_count'])
        else:
            self.results[current_attack.strategy]['innovation_magnitudes'].append(0.0)
            self.results[current_attack.strategy]['suspicious_gps_count'].append(0)
            
        if bias_stats:
            self.results[current_attack.strategy]['gps_imu_bias'].append(bias_stats['current_bias'])
            self.results[current_attack.strategy]['bias_std'].append(bias_stats['bias_std'])
        else:
            self.results[current_attack.strategy]['gps_imu_bias'].append(0.0)
            self.results[current_attack.strategy]['bias_std'].append(0.0)
            
        if gps_stats:
            self.results[current_attack.strategy]['gps_accepted'].append(gps_stats['accepted_count'])
            self.results[current_attack.strategy]['gps_rejected'].append(gps_stats['rejected_count'])
            # Track if mitigation was triggered (GPS rejection rate > 50%)
            mitigation_triggered = gps_stats['acceptance_rate'] < 0.5 if gps_stats['accepted_count'] + gps_stats['rejected_count'] > 0 else False
            self.results[current_attack.strategy]['mitigation_triggered'].append(mitigation_triggered)
            # Track fallback to IMU (when GPS is rejected)
            fallback_to_imu = gps_stats['rejected_count'] > 0
            self.results[current_attack.strategy]['fallback_to_imu'].append(fallback_to_imu)
        else:
            self.results[current_attack.strategy]['gps_accepted'].append(0)
            self.results[current_attack.strategy]['gps_rejected'].append(0)
            self.results[current_attack.strategy]['mitigation_triggered'].append(False)
            self.results[current_attack.strategy]['fallback_to_imu'].append(False)
        # No static plotting or DataVisualizer calls
    
    def get_current_attack(self) -> AttackSequence:
        if self.current_sequence_index < len(self.attack_sequences):
            return self.attack_sequences[self.current_sequence_index]
        return None
    
    def get_results(self) -> Dict:
        return self.results
    
    def save_results(self, output_dir: str = "test_results"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for strategy, data in self.results.items():
            strategy_dir = os.path.join(output_dir, f"SpoofingStrategy.{strategy.name}")
            if not os.path.exists(strategy_dir):
                os.makedirs(strategy_dir)
            with open(os.path.join(strategy_dir, 'raw_data.json'), 'w') as f:
                json.dump({k: np.array(v).tolist() for k, v in data.items()}, f, indent=2)
        print(f"Results saved to {output_dir}")
    
    def analyze_results(self):
        for strategy, data in self.results.items():
            print(f"\nAttack: {strategy.name}")
            errors = np.array(data['position_errors'])
            print(f"  Mean Position Error: {np.mean(errors):.3f}")
            print(f"  Std Position Error: {np.std(errors):.3f}")
            print(f"  Max Position Error: {np.max(errors):.3f}")
            print(f"  Min Position Error: {np.min(errors):.3f}")
            vels = np.array(data['velocity_errors'])
            print(f"  Mean Velocity Error: {np.mean(vels):.3f}")
            print(f"  Std Velocity Error: {np.std(vels):.3f}")
            print(f"  Max Velocity Error: {np.max(vels):.3f}")
            print(f"  Min Velocity Error: {np.min(vels):.3f}")
            
            # Innovation-based mitigation analysis
            if 'innovation_magnitudes' in data and data['innovation_magnitudes']:
                innovations = np.array(data['innovation_magnitudes'])
                print(f"  Mean Innovation: {np.mean(innovations):.3f}")
                print(f"  Max Innovation: {np.max(innovations):.3f}")
                print(f"  Innovation > 5m: {np.sum(innovations > 5.0)} times")
            
            if 'gps_accepted' in data and data['gps_accepted']:
                total_gps = data['gps_accepted'][-1] + data['gps_rejected'][-1]
                if total_gps > 0:
                    acceptance_rate = data['gps_accepted'][-1] / total_gps
                    print(f"  GPS Acceptance Rate: {acceptance_rate:.2%}")
                    print(f"  GPS Rejected: {data['gps_rejected'][-1]} times")
                    print(f"  GPS Accepted: {data['gps_accepted'][-1]} times")
            
            if 'gps_imu_bias' in data and data['gps_imu_bias']:
                biases = np.array(data['gps_imu_bias'])
                print(f"  Mean GPS-IMU Bias: {np.mean(biases):.3f}")
                print(f"  Max GPS-IMU Bias: {np.max(biases):.3f}")
            
            if 'mitigation_triggered' in data and data['mitigation_triggered']:
                mitigation_count = sum(data['mitigation_triggered'])
                print(f"  Mitigation Triggered: {mitigation_count} times")
            
            if 'fallback_to_imu' in data and data['fallback_to_imu']:
                fallback_count = sum(data['fallback_to_imu'])
                print(f"  Fallback to IMU: {fallback_count} times")

def main():
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
    
    # Enable autopilot
    vehicle.set_autopilot(True)
    
    # Initialize sensor fusion with spoofing enabled
    fusion = SensorFusion(vehicle, enable_spoofing=True)
    
    # Create and configure the sequential attack tester
    tester = SequentialAttackTester(fusion)
    
    # Add attack sequences
    tester.add_attack_sequence(SpoofingStrategy.GRADUAL_DRIFT, 30.0, "Gradual Drift Attack")
    tester.add_attack_sequence(SpoofingStrategy.SUDDEN_JUMP, 30.0, "Sudden Jump Attack")
    tester.add_attack_sequence(SpoofingStrategy.RANDOM_WALK, 30.0, "Random Walk Attack")
    tester.add_attack_sequence(SpoofingStrategy.REPLAY, 30.0, "Replay Attack")
    
    # Start the test
    tester.start_test()
    
    try:
        # Run the test until all sequences are complete
        while tester.is_running:
            tester.update(time.time())
            time.sleep(0.1)
        
        # Analyze results after all sequences are complete
        print("\nTest Results Analysis:")
        print("=====================")
        tester.analyze_results()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        tester.analyze_results()
    finally:
        print("Cleaning up...")
        fusion.cleanup()
        vehicle.destroy()

if __name__ == '__main__':
    main() 