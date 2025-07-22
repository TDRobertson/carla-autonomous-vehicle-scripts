import numpy as np
import time
import sys
import glob
import os
import json
import matplotlib.pyplot as plt

# Add current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add the CARLA Python API to PYTHONPATH
try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from sensor_fusion_testing.integration_files.sensor_fusion import SensorFusion
from sensor_fusion_testing.integration_files.gps_spoofer import SpoofingStrategy

class InnovationAwareAttackTest:
    def __init__(self):
        self.client = None
        self.world = None
        self.vehicle = None
        self.fusion = None
        self.data = {
            'time': [],
            'true_position': [],
            'fused_position': [],
            'gps_position': [],
            'innovation': [],
            'position_error': [],
            'attack_active': []
        }
        
    def setup_carla(self):
        """Setup CARLA connection and spawn vehicle"""
        try:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            
            # Get spawn point
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = spawn_points[0] if spawn_points else carla.Transform(carla.Location(x=0.0, y=0.0, z=1.0))
            
            # Spawn vehicle
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            
            # Wait for vehicle to spawn
            time.sleep(2.0)
            
            # Enable autopilot
            self.vehicle.set_autopilot(True)
            
            print("CARLA setup complete")
            return True
            
        except Exception as e:
            print(f"Failed to setup CARLA: {e}")
            return False
    
    def setup_sensor_fusion(self):
        """Setup sensor fusion with innovation-aware attack"""
        try:
            # Initialize sensor fusion with innovation-aware gradual drift attack
            self.fusion = SensorFusion(
                self.vehicle, 
                enable_spoofing=True, 
                spoofing_strategy=SpoofingStrategy.INNOVATION_AWARE_GRADUAL_DRIFT
            )
            print("Sensor fusion setup complete")
            return True
            
        except Exception as e:
            print(f"Failed to setup sensor fusion: {e}")
            return False
    
    def collect_data(self, duration=60):
        """Collect data during the attack test with real-time trajectory plotting"""
        print(f"Starting data collection for {duration} seconds...")
        start_time = time.time()
        
        # Real-time plotting setup
        plt.ion()
        fig, ax = plt.subplots()
        true_xs, true_ys = [], []
        fused_xs, fused_ys = [] , []
        true_line, = ax.plot([], [], 'b-', label='True Position')
        fused_line, = ax.plot([], [], 'r-', label='Spoofed Position')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Vehicle Trajectory (Real-Time)')
        ax.legend()
        plt.show(block=False)
        
        while time.time() - start_time < duration:
            try:
                # Get current data
                true_pos = self.fusion.get_true_position()
                fused_pos = self.fusion.get_fused_position()
                innovation_stats = self.fusion.get_innovation_stats()
                
                if true_pos is not None and fused_pos is not None:
                    current_time = time.time() - start_time
                    
                    # Calculate position error
                    position_error = np.linalg.norm(fused_pos - true_pos)
                    
                    # Store data
                    self.data['time'].append(current_time)
                    self.data['true_position'].append(true_pos.tolist())
                    self.data['fused_position'].append(fused_pos.tolist())
                    self.data['innovation'].append(innovation_stats['current_innovation'])
                    self.data['position_error'].append(position_error)
                    self.data['attack_active'].append(1)  # Attack is always active in this test
                    
                    # Real-time plot update
                    true_xs.append(true_pos[0])
                    true_ys.append(true_pos[1])
                    fused_xs.append(fused_pos[0])
                    fused_ys.append(fused_pos[1])
                    true_line.set_data(true_xs, true_ys)
                    fused_line.set_data(fused_xs, fused_ys)
                    ax.relim()
                    ax.autoscale_view()
                    plt.pause(0.001)
                    
                    # Print progress every 10 seconds
                    if int(current_time) % 10 == 0 and current_time > 0:
                        print(f"Time: {current_time:.1f}s, Error: {position_error:.2f}m, Innovation: {innovation_stats['current_innovation']:.2f}m")
                
                time.sleep(0.1)  # 10Hz data collection
                
            except KeyboardInterrupt:
                print("Data collection interrupted")
                break
            except Exception as e:
                print(f"Error during data collection: {e}")
                break
        
        plt.ioff()
        plt.close(fig)
        print("Data collection complete")
    
    def analyze_results(self):
        """Analyze the attack results"""
        if not self.data['time']:
            print("No data collected")
            return
        
        # Calculate statistics
        position_errors = np.array(self.data['position_error'])
        innovations = np.array(self.data['innovation'])
        
        print("\n=== ATTACK RESULTS ===")
        print(f"Test Duration: {self.data['time'][-1]:.1f} seconds")
        print(f"Mean Position Error: {np.mean(position_errors):.2f} meters")
        print(f"Max Position Error: {np.max(position_errors):.2f} meters")
        print(f"Mean Innovation: {np.mean(innovations):.2f} meters")
        print(f"Max Innovation: {np.max(innovations):.2f} meters")
        
        # Check if attack was successful (position error > 5m for significant time)
        significant_errors = position_errors > 5.0
        success_percentage = np.mean(significant_errors) * 100
        print(f"Attack Success Rate: {success_percentage:.1f}% (time with error > 5m)")
        
        # Check if innovation stayed below threshold
        innovation_threshold = 5.0
        low_innovation_percentage = np.mean(innovations < innovation_threshold) * 100
        print(f"Stealth Rate: {low_innovation_percentage:.1f}% (time below {innovation_threshold}m threshold)")
        
        # Additional analysis
        print(f"\n=== DETAILED ANALYSIS ===")
        print(f"Position Error Statistics:")
        print(f"  - Min: {np.min(position_errors):.2f}m")
        print(f"  - 25th percentile: {np.percentile(position_errors, 25):.2f}m")
        print(f"  - Median: {np.median(position_errors):.2f}m")
        print(f"  - 75th percentile: {np.percentile(position_errors, 75):.2f}m")
        print(f"  - Std Dev: {np.std(position_errors):.2f}m")
        
        print(f"\nInnovation Statistics:")
        print(f"  - Min: {np.min(innovations):.2f}m")
        print(f"  - 25th percentile: {np.percentile(innovations, 25):.2f}m")
        print(f"  - Median: {np.median(innovations):.2f}m")
        print(f"  - 75th percentile: {np.percentile(innovations, 75):.2f}m")
        print(f"  - Std Dev: {np.std(innovations):.2f}m")
        
        return {
            'mean_error': np.mean(position_errors),
            'max_error': np.max(position_errors),
            'success_rate': success_percentage,
            'stealth_rate': low_innovation_percentage,
            'error_stats': {
                'min': np.min(position_errors),
                'max': np.max(position_errors),
                'mean': np.mean(position_errors),
                'median': np.median(position_errors),
                'std': np.std(position_errors)
            },
            'innovation_stats': {
                'min': np.min(innovations),
                'max': np.max(innovations),
                'mean': np.mean(innovations),
                'median': np.median(innovations),
                'std': np.std(innovations)
            }
        }
    
    def save_results(self, results):
        """Save results to JSON file"""
        output_data = {
            'test_info': {
                'test_type': 'innovation_aware_gradual_drift',
                'duration': self.data['time'][-1] if self.data['time'] else 0,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'results': results,
            'raw_data': self.data
        }
        
        filename = f"innovation_aware_attack_results_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
    
    def save_trajectory_plot(self):
        """Save a static trajectory plot as PNG in plotmaps directory."""
        true_positions = np.array(self.data['true_position'])
        fused_positions = np.array(self.data['fused_position'])
        if true_positions.size == 0 or fused_positions.size == 0:
            print("No trajectory data to plot.")
            return
        plt.figure()
        plt.plot(true_positions[:, 0], true_positions[:, 1], 'b-', label='True Position')
        plt.plot(fused_positions[:, 0], fused_positions[:, 1], 'r-', label='Spoofed Position')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Vehicle Trajectory')
        plt.legend()
        plt.tight_layout()
        os.makedirs('sensor_fusion_testing/plotmaps', exist_ok=True)
        filename = f"sensor_fusion_testing/plotmaps/trajectory_{int(time.time())}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Trajectory plot saved to: {filename}")
    
    def print_summary(self, results):
        """Print a summary of the attack effectiveness"""
        print("\n" + "="*60)
        print("ATTACK EFFECTIVENESS SUMMARY")
        print("="*60)
        
        # Attack success assessment
        if results['success_rate'] > 50:
            print("✅ ATTACK SUCCESS: Position error > 5m for majority of time")
        else:
            print("❌ ATTACK FAILURE: Position error not consistently high")
        
        # Stealth assessment
        if results['stealth_rate'] > 80:
            print("✅ STEALTH SUCCESS: Innovation mostly below detection threshold")
        else:
            print("❌ STEALTH FAILURE: Innovation frequently above detection threshold")
        
        # Overall assessment
        if results['success_rate'] > 50 and results['stealth_rate'] > 80:
            print("\n🎯 OVERALL SUCCESS: Attack overcame Kalman filter while remaining stealthy")
            print("   This demonstrates the need for additional detection mechanisms.")
        elif results['success_rate'] > 50:
            print("\n⚠️  PARTIAL SUCCESS: Attack was effective but not stealthy")
            print("   Innovation-based detection would catch this attack.")
        elif results['stealth_rate'] > 80:
            print("\n⚠️  PARTIAL SUCCESS: Attack was stealthy but not effective")
            print("   Kalman filter provided adequate protection.")
        else:
            print("\n❌ COMPLETE FAILURE: Attack was neither effective nor stealthy")
            print("   Kalman filter provided robust protection.")
        
        print("\n" + "="*60)
    
    def cleanup(self):
        """Cleanup resources"""
        if self.fusion:
            self.fusion.cleanup()
        if self.vehicle:
            self.vehicle.destroy()
        if self.client:
            self.client = None
        print("Cleanup complete")
    
    def run_test(self, duration=60):
        """Run the complete attack test"""
        try:
            # Setup
            if not self.setup_carla():
                return False
            
            if not self.setup_sensor_fusion():
                return False
            
            # Collect data
            self.collect_data(duration)
            
            # Analyze results
            results = self.analyze_results()
            
            # Save results
            self.save_results(results)
            
            # Save trajectory plot
            self.save_trajectory_plot()
            
            # Print summary
            self.print_summary(results)
            
            return results
            
        except Exception as e:
            print(f"Test failed: {e}")
            return False
        finally:
            self.cleanup()

def main():
    """Main function to run the innovation-aware attack test"""
    print("=== Innovation-Aware Gradual Drift Attack Test ===")
    print("This test demonstrates how a sophisticated gradual drift attack")
    print("can overcome a Kalman filter by monitoring innovation values.")
    print()
    
    test = InnovationAwareAttackTest()
    results = test.run_test(duration=60)  # 60 second test
    
    if results:
        print("\nTest completed successfully!")
        print("Check the generated JSON file for detailed results.")
    else:
        print("\nTest failed. Check the error messages above.")

if __name__ == '__main__':
    main() 