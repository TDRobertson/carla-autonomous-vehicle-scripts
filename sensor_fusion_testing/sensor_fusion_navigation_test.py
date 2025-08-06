#!/usr/bin/env python3
"""
Sensor Fusion Navigation Example

This script demonstrates how to use the new waypoint navigation system
with sensor fusion for autonomous vehicle navigation. It shows various
navigation modes and integrates with your current research setup.

Features:
- Waypoint navigation using sensor fusion for position tracking
- Integration with CARLA's traffic manager
- Support for GPS spoofing attacks during navigation
- Real-time navigation statistics and visualization
- Multiple route generation strategies
"""

import numpy as np
import time
import sys
import glob
import os
import json
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any

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
from sensor_fusion_testing.integration_files.waypoint_navigator import WaypointNavigator
from sensor_fusion_testing.integration_files.waypoint_generator import WaypointGenerator
from sensor_fusion_testing.integration_files.sensor_fusion import SensorFusion
from sensor_fusion_testing.integration_files.gps_spoofer import SpoofingStrategy

class SensorFusionNavigationExample:
    """
    Comprehensive example demonstrating sensor fusion-based navigation.
    """
    
    def __init__(self):
        self.client = None
        self.world = None
        self.vehicle = None
        self.navigator = None
        self.waypoint_generator = None
        self.traffic_manager = None
        
        # Navigation data collection
        self.navigation_data = {
            'time': [],
            'true_position': [],
            'fused_position': [],
            'target_waypoint': [],
            'distance_to_waypoint': [],
            'heading_error': [],
            'speed': [],
            'throttle': [],
            'steering': [],
            'innovation': []
        }
        
    def setup_carla(self):
        """Setup CARLA connection and spawn vehicle"""
        try:
            self.client = carla.Client('localhost', 2000)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            
            # Set synchronous mode for better control
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05  # 20 FPS
            self.world.apply_settings(settings)
            
            # Get spawn point
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = spawn_points[0] if spawn_points else carla.Transform(carla.Location(x=0.0, y=0.0, z=1.0))
            
            # Spawn vehicle
            blueprint_library = self.world.get_blueprint_library()
            vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            
            # Wait for vehicle to spawn
            for _ in range(10):
                self.world.tick()
                time.sleep(0.1)
            
            # Setup traffic manager - use client.get_trafficmanager() instead of world.get_traffic_manager()
            self.traffic_manager = self.client.get_trafficmanager(8000)
            self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
            self.traffic_manager.ignore_lights_percentage(self.vehicle, 100)
            
            print("CARLA setup complete")
            return True
            
        except Exception as e:
            print(f"Failed to setup CARLA: {e}")
            return False
    
    def setup_navigation_system(self, enable_spoofing: bool = False, 
                               spoofing_strategy: SpoofingStrategy = SpoofingStrategy.GRADUAL_DRIFT):
        """Setup the navigation system with sensor fusion"""
        try:
            # Initialize waypoint generator
            self.waypoint_generator = WaypointGenerator(self.world)
            
            # Initialize waypoint navigator with sensor fusion
            self.navigator = WaypointNavigator(
                vehicle=self.vehicle,
                sensor_fusion=None,  # Will be created by navigator
                enable_spoofing=enable_spoofing,
                spoofing_strategy=spoofing_strategy,
                waypoint_reach_distance=3.0,
                waypoint_lookahead_distance=10.0,
                max_speed=25.0,  # Reduced for safety
                pid_config={
                    'throttle': {'Kp': 0.4, 'Ki': 0.02, 'Kd': 0.08},
                    'steering': {'Kp': 1.0, 'Ki': 0.01, 'Kd': 0.15},
                    'speed': {'Kp': 0.6, 'Ki': 0.03, 'Kd': 0.12}
                }
            )
            
            print("Navigation system setup complete")
            return True
            
        except Exception as e:
            print(f"Failed to setup navigation system: {e}")
            return False
    
    def generate_test_route(self, route_type: str = "random") -> List[carla.Waypoint]:
        """Generate a test route based on the specified type"""
        try:
            if route_type == "random":
                waypoints = self.waypoint_generator.generate_random_route(
                    route_length=15,
                    max_distance=500.0
                )
            elif route_type == "circular":
                waypoints = self.waypoint_generator.generate_circular_route(
                    radius=80.0,
                    num_waypoints=20
                )
            elif route_type == "spawn_points":
                waypoints = self.waypoint_generator.generate_route_from_spawn_points(
                    start_spawn_index=0,
                    end_spawn_index=5
                )
            elif route_type == "custom":
                # Custom route from current location to a distant point
                start_location = self.vehicle.get_location()
                end_location = carla.Location(x=start_location.x + 200, y=start_location.y + 200, z=start_location.z)
                waypoints = self.waypoint_generator.generate_route_waypoints(start_location, end_location)
            else:
                print(f"Unknown route type: {route_type}")
                return []
            
            if waypoints:
                # Visualize the route
                self.waypoint_generator.visualize_route(waypoints)
                
                # Print route statistics
                stats = self.waypoint_generator.get_route_statistics(waypoints)
                print(f"Route statistics: {stats}")
                
            return waypoints
            
        except Exception as e:
            print(f"Error generating test route: {e}")
            return []
    
    def collect_navigation_data(self, duration: int = 120):
        """Collect navigation data during the test"""
        print(f"Starting navigation data collection for {duration} seconds...")
        start_time = time.time()
        
        # Real-time plotting setup
        plt.ion()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Initialize plot data
        times, true_xs, true_ys, fused_xs, fused_ys = [], [], [], [], []
        true_line, = ax1.plot([], [], 'b-', label='True Position', linewidth=2)
        fused_line, = ax1.plot([], [], 'r-', label='Fused Position', linewidth=2)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_title('Vehicle Trajectory')
        ax1.legend()
        ax1.grid(True)
        
        # Speed plot
        speed_line, = ax2.plot([], [], 'g-', label='Speed')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Speed (m/s)')
        ax2.set_title('Vehicle Speed')
        ax2.legend()
        ax2.grid(True)
        
        # Distance to waypoint plot
        distance_line, = ax3.plot([], [], 'm-', label='Distance to Waypoint')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Distance (m)')
        ax3.set_title('Distance to Current Waypoint')
        ax3.legend()
        ax3.grid(True)
        
        # Innovation plot
        innovation_line, = ax4.plot([], [], 'c-', label='Innovation')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Innovation (m)')
        ax4.set_title('Kalman Filter Innovation')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show(block=False)
        
        last_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                current_time = time.time() - start_time
                dt = time.time() - last_time
                last_time = time.time()
                
                # Perform navigation step
                navigation_result = self.navigator.navigate_step(dt)
                
                # Get sensor fusion data
                sensor_stats = self.navigator.get_sensor_fusion_stats()
                
                # Collect data
                if sensor_stats.get('true_position') is not None:
                    self.navigation_data['time'].append(current_time)
                    self.navigation_data['true_position'].append(sensor_stats['true_position'].tolist())
                    
                    if sensor_stats.get('fused_position') is not None:
                        self.navigation_data['fused_position'].append(sensor_stats['fused_position'].tolist())
                    else:
                        self.navigation_data['fused_position'].append([0, 0, 0])
                    
                    # Navigation metrics
                    self.navigation_data['distance_to_waypoint'].append(navigation_result.get('distance_to_waypoint', 0))
                    self.navigation_data['heading_error'].append(navigation_result.get('heading_error', 0))
                    self.navigation_data['speed'].append(navigation_result.get('current_speed', 0))
                    self.navigation_data['throttle'].append(navigation_result.get('throttle', 0))
                    self.navigation_data['steering'].append(navigation_result.get('steering', 0))
                    
                    # Innovation data
                    innovation_stats = sensor_stats.get('innovation_stats', {})
                    self.navigation_data['innovation'].append(innovation_stats.get('current_innovation', 0))
                    
                    # Update real-time plots
                    if len(self.navigation_data['true_position']) > 1:
                        # Trajectory plot
                        true_pos = np.array(self.navigation_data['true_position'])
                        fused_pos = np.array(self.navigation_data['fused_position'])
                        
                        true_xs = true_pos[:, 0]
                        true_ys = true_pos[:, 1]
                        fused_xs = fused_pos[:, 0]
                        fused_ys = fused_pos[:, 1]
                        
                        true_line.set_data(true_xs, true_ys)
                        fused_line.set_data(fused_xs, fused_ys)
                        ax1.relim()
                        ax1.autoscale_view()
                        
                        # Speed plot
                        times = self.navigation_data['time']
                        speeds = self.navigation_data['speed']
                        speed_line.set_data(times, speeds)
                        ax2.relim()
                        ax2.autoscale_view()
                        
                        # Distance plot
                        distances = self.navigation_data['distance_to_waypoint']
                        distance_line.set_data(times, distances)
                        ax3.relim()
                        ax3.autoscale_view()
                        
                        # Innovation plot
                        innovations = self.navigation_data['innovation']
                        innovation_line.set_data(times, innovations)
                        ax4.relim()
                        ax4.autoscale_view()
                        
                        plt.pause(0.001)
                
                # Check if navigation completed
                if navigation_result['status'] == 'completed':
                    print("Navigation completed!")
                    break
                
                # Print progress every 10 seconds
                if int(current_time) % 10 == 0 and current_time > 0:
                    nav_stats = self.navigator.get_navigation_stats()
                    print(f"Time: {current_time:.1f}s, "
                          f"Waypoints: {nav_stats['waypoints_reached']}, "
                          f"Speed: {navigation_result.get('current_speed', 0):.1f}m/s, "
                          f"Distance: {navigation_result.get('distance_to_waypoint', 0):.1f}m")
                
                # Tick the world
                self.world.tick()
                
            except KeyboardInterrupt:
                print("Navigation interrupted by user")
                break
            except Exception as e:
                print(f"Error during navigation: {e}")
                break
        
        plt.ioff()
        plt.close(fig)
        print("Navigation data collection complete")
    
    def analyze_navigation_results(self) -> Dict[str, Any]:
        """Analyze the navigation results"""
        if not self.navigation_data['time']:
            print("No navigation data collected")
            return {}
        
        # Calculate statistics
        nav_stats = self.navigator.get_navigation_stats()
        sensor_stats = self.navigator.get_sensor_fusion_stats()
        
        # Position error analysis
        true_positions = np.array(self.navigation_data['true_position'])
        fused_positions = np.array(self.navigation_data['fused_position'])
        
        if len(true_positions) > 0 and len(fused_positions) > 0:
            position_errors = np.linalg.norm(fused_positions - true_positions, axis=1)
            mean_position_error = np.mean(position_errors)
            max_position_error = np.max(position_errors)
        else:
            mean_position_error = 0.0
            max_position_error = 0.0
        
        # Navigation performance metrics
        speeds = np.array(self.navigation_data['speed'])
        distances = np.array(self.navigation_data['distance_to_waypoint'])
        innovations = np.array(self.navigation_data['innovation'])
        
        results = {
            'navigation_stats': nav_stats,
            'sensor_fusion_stats': sensor_stats,
            'performance_metrics': {
                'mean_position_error': mean_position_error,
                'max_position_error': max_position_error,
                'mean_speed': np.mean(speeds) if len(speeds) > 0 else 0.0,
                'max_speed': np.max(speeds) if len(speeds) > 0 else 0.0,
                'mean_distance_to_waypoint': np.mean(distances) if len(distances) > 0 else 0.0,
                'mean_innovation': np.mean(innovations) if len(innovations) > 0 else 0.0,
                'max_innovation': np.max(innovations) if len(innovations) > 0 else 0.0
            },
            'navigation_success': nav_stats['waypoints_reached'] > 0,
            'total_navigation_time': self.navigation_data['time'][-1] if self.navigation_data['time'] else 0.0
        }
        
        print("\n=== NAVIGATION RESULTS ===")
        print(f"Navigation Time: {results['total_navigation_time']:.1f} seconds")
        print(f"Waypoints Reached: {nav_stats['waypoints_reached']}")
        print(f"Total Distance: {nav_stats['total_distance']:.1f} meters")
        print(f"Average Speed: {results['performance_metrics']['mean_speed']:.1f} m/s")
        print(f"Max Speed: {results['performance_metrics']['max_speed']:.1f} m/s")
        print(f"Mean Position Error: {results['performance_metrics']['mean_position_error']:.2f} meters")
        print(f"Mean Innovation: {results['performance_metrics']['mean_innovation']:.2f} meters")
        
        return results
    
    def save_results(self, results: Dict[str, Any], route_type: str):
        """Save navigation results to JSON file"""
        output_data = {
            'test_info': {
                'test_type': 'sensor_fusion_navigation',
                'route_type': route_type,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'duration': results['total_navigation_time']
            },
            'results': results,
            'raw_data': self.navigation_data
        }
        
        filename = f"sensor_fusion_navigation_{route_type}_{int(time.time())}.json"
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
    
    def save_trajectory_plot(self, route_type: str):
        """Save trajectory plot"""
        if not self.navigation_data['true_position']:
            print("No trajectory data to plot")
            return
        
        true_positions = np.array(self.navigation_data['true_position'])
        fused_positions = np.array(self.navigation_data['fused_position'])
        
        plt.figure(figsize=(10, 8))
        plt.plot(true_positions[:, 0], true_positions[:, 1], 'b-', label='True Position', linewidth=2)
        plt.plot(fused_positions[:, 0], fused_positions[:, 1], 'r-', label='Fused Position', linewidth=2)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(f'Sensor Fusion Navigation Trajectory ({route_type})')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        
        # Save plot
        os.makedirs('sensor_fusion_testing/plotmaps', exist_ok=True)
        filename = f"sensor_fusion_testing/plotmaps/navigation_trajectory_{route_type}_{int(time.time())}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Trajectory plot saved to: {filename}")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.navigator:
            self.navigator.cleanup()
        if self.vehicle:
            self.vehicle.destroy()
        if self.client:
            # Reset world settings
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            self.world.apply_settings(settings)
        print("Cleanup complete")
    
    def run_navigation_test(self, route_type: str = "random", duration: int = 120, 
                           enable_spoofing: bool = False, 
                           spoofing_strategy: SpoofingStrategy = SpoofingStrategy.GRADUAL_DRIFT):
        """Run a complete navigation test"""
        try:
            print(f"\n=== SENSOR FUSION NAVIGATION TEST ===")
            print(f"Route Type: {route_type}")
            print(f"Duration: {duration} seconds")
            print(f"GPS Spoofing: {enable_spoofing}")
            if enable_spoofing:
                print(f"Spoofing Strategy: {spoofing_strategy}")
            print("=" * 50)
            
            # Setup
            if not self.setup_carla():
                return False
            
            if not self.setup_navigation_system(enable_spoofing, spoofing_strategy):
                return False
            
            # Generate route
            waypoints = self.generate_test_route(route_type)
            if not waypoints:
                print("Failed to generate route")
                return False
            
            # Set waypoints and start navigation
            self.navigator.set_waypoints(waypoints)
            if not self.navigator.start_navigation():
                print("Failed to start navigation")
                return False
            
            # Collect data
            self.collect_navigation_data(duration)
            
            # Analyze results
            results = self.analyze_navigation_results()
            
            # Save results
            self.save_results(results, route_type)
            self.save_trajectory_plot(route_type)
            
            return results
            
        except Exception as e:
            print(f"Navigation test failed: {e}")
            return False
        finally:
            self.cleanup()

def main():
    """Main function to run navigation examples"""
    print("=== Sensor Fusion Navigation Example ===")
    print("This example demonstrates waypoint navigation using sensor fusion")
    print("for position tracking, with support for GPS spoofing attacks.")
    print()
    
    # Create example instance
    example = SensorFusionNavigationExample()
    
    # Test different route types
    route_types = ["random", "circular", "spawn_points"]
    
    for route_type in route_types:
        print(f"\nTesting {route_type} route...")
        
        # Test without spoofing
        results = example.run_navigation_test(
            route_type=route_type,
            duration=60,  # 60 seconds per test
            enable_spoofing=False
        )
        
        if results:
            print(f"✅ {route_type} route test completed successfully")
        else:
            print(f"❌ {route_type} route test failed")
        
        # Wait between tests
        time.sleep(2)
    
    # Test with GPS spoofing
    print(f"\nTesting with GPS spoofing attack...")
    results = example.run_navigation_test(
        route_type="random",
        duration=60,
        enable_spoofing=True,
        spoofing_strategy=SpoofingStrategy.INNOVATION_AWARE_GRADUAL_DRIFT
    )
    
    if results:
        print("✅ GPS spoofing test completed successfully")
    else:
        print("❌ GPS spoofing test failed")
    
    print("\nAll tests completed!")

if __name__ == '__main__':
    main() 