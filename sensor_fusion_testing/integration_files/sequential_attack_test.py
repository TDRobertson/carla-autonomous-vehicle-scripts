import carla
import numpy as np
import time
import json
import os
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from gps_spoofer import GPSSpoofer, SpoofingStrategy
from sensor_fusion import SensorFusion
from visualization import DataVisualizer

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
        self.visualizer = DataVisualizer()
        
        # Enhanced analysis data for ML training
        self.attack_results: Dict[str, List] = {
            # Position data
            'true_positions': [],
            'fused_positions': [],
            'errors': [],
            'timestamps': [],
            
            # Velocity data
            'true_velocities': [],
            'fused_velocities': [],
            'velocity_errors': [],
            
            # IMU data
            'imu_accelerations': [],
            'imu_gyroscopes': [],
            
            # Kalman filter metrics
            'kalman_gains': [],
            'innovation_vectors': [],
            'innovation_covariances': [],
            
            # Statistical features
            'position_variance': [],
            'velocity_variance': [],
            'acceleration_variance': [],
            
            # Attack-specific features
            'attack_type': [],
            'attack_confidence': [],
            'attack_duration': [],
            
            # Environmental features
            'vehicle_speed': [],
            'vehicle_acceleration': [],
            'vehicle_angular_velocity': [],
            
            # Temporal features
            'time_since_last_update': [],
            'update_frequency': [],
            
            # Error metrics
            'position_error_rate': [],
            'velocity_error_rate': [],
            'acceleration_error_rate': []
        }
        
    def add_attack_sequence(self, strategy: SpoofingStrategy, duration: float, description: str):
        """Add an attack sequence to the test."""
        self.attack_sequences.append(AttackSequence(strategy, duration, description))
        
    def start_test(self):
        """Start the sequential attack test."""
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
        """Update test state and collect data."""
        if not self.is_running:
            return
            
        current_attack = self.get_current_attack()
        if current_attack is None:
            self.is_running = False
            return
            
        # Check if current attack sequence is complete
        elapsed_time = timestamp - self.sequence_start_time
        if elapsed_time >= current_attack.duration:
            print(f"Completed attack sequence: {current_attack.description}")
            self.current_sequence_index += 1
            
            if self.current_sequence_index < len(self.attack_sequences):
                # Start next attack sequence
                self.sequence_start_time = timestamp
                self.sensor_fusion.spoofer.set_strategy(self.attack_sequences[self.current_sequence_index].strategy)
                print(f"Starting attack sequence: {self.attack_sequences[self.current_sequence_index].description}")
            else:
                self.is_running = False
                print("All attack sequences completed")
                return
                
        # Update sensor fusion
        self.sensor_fusion.update()
        
        # Get current data
        true_position = self.sensor_fusion.get_true_position()
        fused_position = self.sensor_fusion.get_fused_position()
        true_velocity = self.sensor_fusion.get_true_velocity()
        fused_velocity = self.sensor_fusion.get_fused_velocity()
        imu_data = self.sensor_fusion.get_imu_data()
        kalman_metrics = self.sensor_fusion.get_kalman_metrics()
        
        # Skip if we don't have position data
        if true_position is None or fused_position is None:
            return
            
        # Calculate error
        position_error = np.linalg.norm(true_position - fused_position)
        velocity_error = np.linalg.norm(true_velocity - fused_velocity) if true_velocity is not None and fused_velocity is not None else 0.0
        
        # Store results
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
                'timestamps': []
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
        
        # Update visualizations periodically (every 10 updates)
        if len(self.results[current_attack.strategy]['timestamps']) % 10 == 0:
            self._update_visualizations(current_attack.strategy)
            
    def _update_visualizations(self, current_strategy: SpoofingStrategy):
        """Update visualizations for the current attack strategy."""
        if current_strategy not in self.results:
            return
            
        results = self.results[current_strategy]
        
        # Position tracking plot
        fig = self.visualizer.plot_position_tracking(
            results['true_positions'],
            results['fused_positions'],
            results['timestamps']
        )
        if fig is not None:
            self.visualizer.figures[f'position_tracking_{current_strategy}'] = fig
            
        # Error evolution plot
        fig = self.visualizer.plot_error_evolution(
            results['position_errors'],
            results['velocity_errors'],
            results['timestamps']
        )
        if fig is not None:
            self.visualizer.figures[f'error_evolution_{current_strategy}'] = fig
            
        # Velocity profiles plot
        if len(results['true_velocities']) > 0 and len(results['fused_velocities']) > 0:
            fig = self.visualizer.plot_velocity_profiles(
                results['true_velocities'],
                results['fused_velocities'],
                results['timestamps']
            )
            if fig is not None:
                self.visualizer.figures[f'velocity_profiles_{current_strategy}'] = fig
                
        # Error distribution plot
        fig = self.visualizer.plot_error_distribution(
            results['position_errors']
        )
        if fig is not None:
            self.visualizer.figures[f'error_distribution_{current_strategy}'] = fig
            
        # Position error heatmap
        fig = self.visualizer.plot_position_error_heatmap(
            results['true_positions'],
            results['fused_positions']
        )
        if fig is not None:
            self.visualizer.figures[f'error_heatmap_{current_strategy}'] = fig
            
        # Correlation matrix
        metrics = {}
        
        # Add position error
        if len(results['position_errors']) > 0:
            metrics['position_error'] = results['position_errors']
            
        # Add velocity error
        if len(results['velocity_errors']) > 0:
            metrics['velocity_error'] = results['velocity_errors']
            
        # Add velocity components if available
        if len(results['true_velocities']) > 0:
            true_vel = np.array(results['true_velocities'])
            metrics['true_velocity_x'] = true_vel[:, 0]
            metrics['true_velocity_y'] = true_vel[:, 1]
            
        if len(results['fused_velocities']) > 0:
            fused_vel = np.array(results['fused_velocities'])
            metrics['fused_velocity_x'] = fused_vel[:, 0]
            metrics['fused_velocity_y'] = fused_vel[:, 1]
            
        # Only create correlation matrix if we have metrics
        if metrics:
            fig = self.visualizer.plot_correlation_matrix(metrics)
            if fig is not None:
                self.visualizer.figures[f'correlation_matrix_{current_strategy}'] = fig
            
    def get_current_attack(self) -> AttackSequence:
        """Get the current attack sequence."""
        if not self.is_running:
            return None
        return self.attack_sequences[self.current_sequence_index]
        
    def get_results(self) -> Dict:
        """Get the test results."""
        return self.results
        
    def save_results(self, output_dir: str = "test_results"):
        """Save test results to files for future analysis."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save raw data
        for strategy, data in self.results.items():
            strategy_dir = os.path.join(output_dir, str(strategy))
            if not os.path.exists(strategy_dir):
                os.makedirs(strategy_dir)
                
            # Convert numpy arrays to lists for JSON serialization
            serializable_data = {
                'true_positions': [pos.tolist() for pos in data['true_positions']],
                'fused_positions': [pos.tolist() for pos in data['fused_positions']],
                'true_velocities': [vel.tolist() for vel in data['true_velocities']],
                'fused_velocities': [vel.tolist() for vel in data['fused_velocities']],
                'position_errors': data['position_errors'],
                'velocity_errors': data['velocity_errors'],
                'timestamps': data['timestamps']
            }
            
            # Save as JSON
            with open(os.path.join(strategy_dir, 'raw_data.json'), 'w') as f:
                json.dump(serializable_data, f, indent=2)
                
            # Save statistics
            stats = {
                'position_error_stats': {
                    'mean': float(np.mean(data['position_errors'])),
                    'std': float(np.std(data['position_errors'])),
                    'max': float(np.max(data['position_errors'])),
                    'min': float(np.min(data['position_errors']))
                },
                'velocity_error_stats': {
                    'mean': float(np.mean(data['velocity_errors'])),
                    'std': float(np.std(data['velocity_errors'])),
                    'max': float(np.max(data['velocity_errors'])),
                    'min': float(np.min(data['velocity_errors']))
                },
                'error_rate': float(np.mean(np.array(data['position_errors']) > 1.0) * 100)
            }
            
            with open(os.path.join(strategy_dir, 'statistics.json'), 'w') as f:
                json.dump(stats, f, indent=2)
                
    def analyze_results(self):
        """Analyze and print statistics of the test results."""
        print("\nTest Results Analysis:")
        print("=====================")
        
        for strategy, data in self.results.items():
            if not data['position_errors']:  # Skip if no data collected
                print(f"\nStrategy: {strategy}")
                print("-----------------")
                print("No data collected for this strategy")
                continue
                
            print(f"\nStrategy: {strategy}")
            print("-----------------")
            
            # Position error statistics
            position_errors = np.array(data['position_errors'])
            print(f"Position Error Statistics:")
            print(f"  Average Error: {np.mean(position_errors):.3f} m")
            print(f"  Max Error: {np.max(position_errors):.3f} m")
            print(f"  Min Error: {np.min(position_errors):.3f} m")
            print(f"  Error Std Dev: {np.std(position_errors):.3f} m")
            
            # Velocity error statistics
            velocity_errors = np.array(data['velocity_errors'])
            if len(velocity_errors) > 0:
                print(f"\nVelocity Error Statistics:")
                print(f"  Average Error: {np.mean(velocity_errors):.3f} m/s")
                print(f"  Max Error: {np.max(velocity_errors):.3f} m/s")
                print(f"  Min Error: {np.min(velocity_errors):.3f} m/s")
                print(f"  Error Std Dev: {np.std(velocity_errors):.3f} m/s")
            
            # Error rate analysis
            error_threshold = 1.0  # meters
            error_rate = np.mean(position_errors > error_threshold) * 100
            print(f"\nError Rate Analysis:")
            print(f"  Error Rate (> {error_threshold}m): {error_rate:.1f}%")
            
        # Generate final visualizations
        for strategy in self.results.keys():
            self._update_visualizations(strategy)
            
        # Save all visualizations
        self.visualizer.save_all_plots("test_results")
        
        # Save raw data and statistics
        self.save_results()
        
        # Generate correlation matrix for all metrics
        all_metrics = {}
        for strategy, data in self.results.items():
            if not data['position_errors']:  # Skip if no data collected
                continue
            for key, values in data.items():
                if isinstance(values[0], (int, float, np.ndarray)):
                    if isinstance(values[0], np.ndarray):
                        all_metrics[f"{strategy}_{key}"] = np.linalg.norm(values, axis=1)
                    else:
                        all_metrics[f"{strategy}_{key}"] = values
                        
        if all_metrics:  # Only generate correlation matrix if we have data
            self.visualizer.figures['correlation_matrix'] = \
                self.visualizer.plot_correlation_matrix(all_metrics)
            self.visualizer.save_all_plots("test_results")
            
        # Clean up
        self.visualizer.clear_plots()

def main():
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