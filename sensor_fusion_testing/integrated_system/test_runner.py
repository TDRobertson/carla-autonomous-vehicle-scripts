import carla
import numpy as np
import time
import json
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from gps_spoofer import GPSSpoofer, SpoofingStrategy
from integrated_system.sensor_fusion import IntegratedSensorFusion
from data_processor import DataProcessor

@dataclass
class TestConfig:
    strategy: SpoofingStrategy
    duration: float
    description: str
    weather: str = "ClearNoon"
    map_name: str = "Town01"

class TestRunner:
    def __init__(self, output_dir: str = "test_results"):
        self.output_dir = output_dir
        self.data_processor = DataProcessor(output_dir)
        self.current_test = None
        self.results = {}
        self.start_time = None
        
    def setup_carla(self, map_name: str = "Town01") -> tuple:
        """Setup CARLA client and world."""
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        
        # Load map
        world = client.get_world()
        if world.get_map().name != map_name:
            world = client.load_world(map_name)
            
        return client, world
        
    def setup_vehicle(self, world: carla.World) -> carla.Actor:
        """Setup test vehicle."""
        # Get spawn point
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0]
        
        # Spawn vehicle
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        
        # Wait for spawn
        time.sleep(2.0)
        
        # Enable autopilot
        vehicle.set_autopilot(True)
        
        return vehicle
        
    def setup_weather(self, world: carla.World, weather: str):
        """Setup weather conditions."""
        weather_presets = {
            "ClearNoon": carla.WeatherParameters.ClearNoon,
            "CloudyNoon": carla.WeatherParameters.CloudyNoon,
            "WetNoon": carla.WeatherParameters.WetNoon,
            "WetCloudyNoon": carla.WeatherParameters.WetCloudyNoon,
            "MidRainyNoon": carla.WeatherParameters.MidRainyNoon,
            "HardRainNoon": carla.WeatherParameters.HardRainNoon,
            "SoftRainNoon": carla.WeatherParameters.SoftRainNoon,
            "ClearSunset": carla.WeatherParameters.ClearSunset,
            "CloudySunset": carla.WeatherParameters.CloudySunset,
            "WetSunset": carla.WeatherParameters.WetSunset,
            "WetCloudySunset": carla.WeatherParameters.WetCloudySunset,
            "MidRainSunset": carla.WeatherParameters.MidRainSunset,
            "HardRainSunset": carla.WeatherParameters.HardRainSunset,
            "SoftRainSunset": carla.WeatherParameters.SoftRainSunset
        }
        
        if weather in weather_presets:
            world.set_weather(weather_presets[weather])
            
    def run_test(self, config: TestConfig):
        """Run a single test with the given configuration."""
        print(f"\nRunning test: {config.description}")
        print(f"Strategy: {config.strategy}")
        print(f"Duration: {config.duration}s")
        print(f"Weather: {config.weather}")
        print(f"Map: {config.map_name}")
        
        # Setup CARLA
        client, world = self.setup_carla(config.map_name)
        self.setup_weather(world, config.weather)
        
        # Setup vehicle
        vehicle = self.setup_vehicle(world)
        
        # Initialize sensor fusion
        fusion = IntegratedSensorFusion(
            vehicle,
            enable_spoofing=True,
            spoofing_strategy=config.strategy
        )
        
        # Initialize results storage
        self.current_test = {
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
        
        # Run test
        self.start_time = time.time()
        try:
            while time.time() - self.start_time < config.duration:
                # Update sensor fusion
                fusion.update()
                
                # Get current data
                true_position = fusion.get_true_position()
                fused_position = fusion.get_fused_position()
                true_velocity = fusion.get_velocity()
                fused_velocity = fusion.get_velocity()  # Using true velocity for now
                imu_data = fusion.get_imu_data()
                kalman_metrics = fusion.get_kalman_metrics()
                
                if true_position is not None and fused_position is not None:
                    # Calculate errors
                    position_error = np.linalg.norm(true_position - fused_position)
                    velocity_error = np.linalg.norm(true_velocity - fused_velocity) if true_velocity is not None and fused_velocity is not None else 0.0
                    
                    # Store results
                    self.current_test['true_positions'].append(true_position.tolist())
                    self.current_test['fused_positions'].append(fused_position.tolist())
                    if true_velocity is not None:
                        self.current_test['true_velocities'].append(true_velocity.tolist())
                    if fused_velocity is not None:
                        self.current_test['fused_velocities'].append(fused_velocity.tolist())
                    if imu_data is not None:
                        self.current_test['imu_data'].append(imu_data)
                    if kalman_metrics is not None:
                        self.current_test['kalman_metrics'].append(kalman_metrics)
                    self.current_test['position_errors'].append(float(position_error))
                    self.current_test['velocity_errors'].append(float(velocity_error))
                    self.current_test['timestamps'].append(time.time() - self.start_time)
                    
                time.sleep(0.1)  # 10Hz update rate
                
        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        finally:
            # Cleanup
            fusion.cleanup()
            vehicle.destroy()
            
            # Save results
            self.save_results(config)
            
    def save_results(self, config: TestConfig):
        """Save test results to file."""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # Create strategy directory
        strategy_dir = os.path.join(self.output_dir, str(config.strategy))
        if not os.path.exists(strategy_dir):
            os.makedirs(strategy_dir)
            
        # Save raw data
        with open(os.path.join(strategy_dir, 'raw_data.json'), 'w') as f:
            json.dump(self.current_test, f, indent=2)
            
        # Calculate and save statistics
        stats = {
            'position_error_stats': {
                'mean': float(np.mean(self.current_test['position_errors'])),
                'std': float(np.std(self.current_test['position_errors'])),
                'max': float(np.max(self.current_test['position_errors'])),
                'min': float(np.min(self.current_test['position_errors']))
            },
            'velocity_error_stats': {
                'mean': float(np.mean(self.current_test['velocity_errors'])),
                'std': float(np.std(self.current_test['velocity_errors'])),
                'max': float(np.max(self.current_test['velocity_errors'])),
                'min': float(np.min(self.current_test['velocity_errors']))
            },
            'test_config': {
                'strategy': str(config.strategy),
                'duration': config.duration,
                'description': config.description,
                'weather': config.weather,
                'map_name': config.map_name
            }
        }
        
        with open(os.path.join(strategy_dir, 'statistics.json'), 'w') as f:
            json.dump(stats, f, indent=2)
            
        print(f"\nResults saved to {strategy_dir}")
        
def main():
    # Create test runner
    runner = TestRunner()
    
    # Define test configurations
    tests = [
        TestConfig(
            strategy=SpoofingStrategy.GRADUAL_DRIFT,
            duration=30.0,
            description="Gradual Drift Attack",
            weather="ClearNoon"
        ),
        TestConfig(
            strategy=SpoofingStrategy.SUDDEN_JUMP,
            duration=30.0,
            description="Sudden Jump Attack",
            weather="ClearNoon"
        ),
        TestConfig(
            strategy=SpoofingStrategy.RANDOM_WALK,
            duration=30.0,
            description="Random Walk Attack",
            weather="ClearNoon"
        ),
        TestConfig(
            strategy=SpoofingStrategy.REPLAY,
            duration=30.0,
            description="Replay Attack",
            weather="ClearNoon"
        )
    ]
    
    # Run tests
    for test in tests:
        runner.run_test(test)
        
    # Process results
    runner.data_processor.save_processed_data()
    importance_df = runner.data_processor.generate_feature_importance()
    print("\nFeature Importance Analysis:")
    print(importance_df)
    
if __name__ == '__main__':
    main() 