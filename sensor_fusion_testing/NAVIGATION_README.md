# Sensor Fusion Waypoint Navigation System

This directory contains a modular waypoint navigation system that integrates with your existing sensor fusion research. The system allows autonomous vehicles to navigate through waypoints using sensor fusion for position tracking, with support for GPS spoofing attacks during navigation.

## Overview

The navigation system consists of three main components:

1. **WaypointNavigator** - Handles vehicle control and navigation using sensor fusion
2. **WaypointGenerator** - Generates routes and waypoints using CARLA's traffic manager
3. **Integration Examples** - Demonstrates how to use the system with your research

## Key Features

- **Sensor Fusion Integration**: Uses your existing sensor fusion system for position tracking
- **GPS Spoofing Support**: Can test navigation under various GPS spoofing attacks
- **Multiple Route Types**: Random routes, circular routes, spawn point routes, and custom routes
- **Real-time Visualization**: Live plotting of vehicle trajectory and navigation metrics
- **Comprehensive Statistics**: Detailed navigation performance and sensor fusion metrics
- **Modular Design**: Easy to integrate with your existing scripts
- **Path Resolution**: Automatic CARLA path detection and configuration for Windows environments

## Files Overview

### Core Modules

- `integration_files/waypoint_navigator.py` - Main navigation controller
- `integration_files/waypoint_generator.py` - Route and waypoint generation with CARLA path resolution
- `simple_navigation_test.py` - Simple usage example (renamed from simple_navigation_example.py)

### Integration with Your Research

The system integrates seamlessly with your existing sensor fusion research:

- Uses your `SensorFusion` class for position tracking
- Supports all your GPS spoofing strategies
- Maintains compatibility with your innovation-aware attack testing
- Provides additional metrics for attack effectiveness analysis

## Quick Start

### Basic Usage

```python
import carla
from sensor_fusion_testing.integration_files.waypoint_navigator import WaypointNavigator
from sensor_fusion_testing.integration_files.waypoint_generator import WaypointGenerator

# Setup CARLA and vehicle
client = carla.Client('localhost', 2000)
world = client.get_world()
vehicle = world.spawn_actor(vehicle_bp, spawn_point)

# Create navigator with sensor fusion
navigator = WaypointNavigator(
    vehicle=vehicle,
    enable_spoofing=False,  # Set to True for attack testing
    max_speed=25.0
)

# Generate route
generator = WaypointGenerator(world)
waypoints = generator.generate_route_waypoints(start_location, end_location)

# Navigate
navigator.set_waypoints(waypoints)
navigator.start_navigation()

# Navigation loop
while True:
    result = navigator.navigate_step(0.05)
    if result['status'] == 'completed':
        break
    time.sleep(0.05)

# Get statistics
stats = navigator.get_navigation_stats()
print(f"Reached {stats['waypoints_reached']} waypoints")
```

### With GPS Spoofing Attack

```python
from sensor_fusion_testing.integration_files.gps_spoofer import SpoofingStrategy

# Create navigator with spoofing
navigator = WaypointNavigator(
    vehicle=vehicle,
    enable_spoofing=True,
    spoofing_strategy=SpoofingStrategy.INNOVATION_AWARE_GRADUAL_DRIFT
)

# Navigation proceeds normally, but with spoofed GPS data
# You can analyze the impact on navigation performance
```

## Route Generation Options

### 1. Random Route

```python
waypoints = generator.generate_random_route(
    route_length=15,
    max_distance=500.0
)
```

### 2. Circular Route

```python
waypoints = generator.generate_circular_route(
    radius=80.0,
    num_waypoints=20
)
```

### 3. Spawn Point Route

```python
waypoints = generator.generate_route_from_spawn_points(
    start_spawn_index=0,
    end_spawn_index=5
)
```

### 4. Custom Route

```python
waypoints = generator.generate_route_waypoints(
    start_location,
    end_location,
    route_id="my_route"
)
```

## Navigation Configuration

### PID Controller Tuning

```python
navigator = WaypointNavigator(
    vehicle=vehicle,
    pid_config={
        'throttle': {'Kp': 0.4, 'Ki': 0.02, 'Kd': 0.08},
        'steering': {'Kp': 1.0, 'Ki': 0.01, 'Kd': 0.15},
        'speed': {'Kp': 0.6, 'Ki': 0.03, 'Kd': 0.12}
    }
)
```

### Navigation Parameters

```python
navigator = WaypointNavigator(
    vehicle=vehicle,
    waypoint_reach_distance=3.0,  # Distance to consider waypoint reached
    waypoint_lookahead_distance=10.0,  # Look ahead distance
    max_speed=25.0,  # Maximum vehicle speed
    enable_spoofing=False,
    spoofing_strategy=SpoofingStrategy.GRADUAL_DRIFT
)
```

## Data Collection and Analysis

### Real-time Data Collection

The system automatically collects:

- True and fused position data
- Navigation metrics (speed, distance to waypoint, heading error)
- Sensor fusion statistics (innovation, Kalman filter metrics)
- Control outputs (throttle, steering)

### Analysis Example

```python
# Get navigation statistics
nav_stats = navigator.get_navigation_stats()
print(f"Total distance: {nav_stats['total_distance']:.1f}m")
print(f"Average speed: {nav_stats['average_speed']:.1f}m/s")
print(f"Waypoints reached: {nav_stats['waypoints_reached']}")

# Get sensor fusion statistics
sensor_stats = navigator.get_sensor_fusion_stats()
print(f"Position error: {sensor_stats['position_error']:.2f}m")
print(f"Innovation: {sensor_stats['innovation_stats']['current_innovation']:.2f}m")
```

## Integration with Your Research

### 1. Attack Testing

Use the navigation system to test how GPS spoofing attacks affect autonomous navigation:

```python
# Test different spoofing strategies
strategies = [
    SpoofingStrategy.GRADUAL_DRIFT,
    SpoofingStrategy.INNOVATION_AWARE_GRADUAL_DRIFT,
    SpoofingStrategy.RANDOM_WALK,
    SpoofingStrategy.SUDDEN_JUMP
]

for strategy in strategies:
    navigator = WaypointNavigator(
        vehicle=vehicle,
        enable_spoofing=True,
        spoofing_strategy=strategy
    )
    # Run navigation test and collect data
```

### 2. Performance Comparison

Compare navigation performance with and without attacks:

```python
# Baseline test (no spoofing)
baseline_stats = run_navigation_test(enable_spoofing=False)

# Attack test
attack_stats = run_navigation_test(
    enable_spoofing=True,
    spoofing_strategy=SpoofingStrategy.INNOVATION_AWARE_GRADUAL_DRIFT
)

# Compare results
compare_navigation_performance(baseline_stats, attack_stats)
```

### 3. Innovation Analysis

Analyze how innovation values change during navigation under attack:

```python
sensor_stats = navigator.get_sensor_fusion_stats()
innovation_history = sensor_stats['innovation_stats']['history']

# Analyze innovation patterns during navigation
analyze_innovation_patterns(innovation_history)
```

## Running Examples

### Simple Example

```bash
cd sensor_fusion_testing
python simple_navigation_test.py
```

## Path Resolution and CARLA Integration

### Automatic Path Detection

The system automatically detects and configures CARLA paths for Windows environments:

- **Egg File Detection**: Automatically finds CARLA egg files in `C:\CARLA_0.9.15\PythonAPI\carla\dist\`
- **Agents Module**: Adds the correct path for `agents.navigation.global_route_planner`
- **Fallback Support**: Uses glob patterns if exact file names aren't found

### Path Configuration

The system handles the following paths automatically:

```python
# CARLA root directory
carla_root = "C:/CARLA_0.9.15/PythonAPI/carla"

# Egg file location
egg_file = "carla-0.9.15-py3.7-win-amd64.egg"

# Agents module location
agents_path = "C:/CARLA_0.9.15/PythonAPI/carla"
```

### Troubleshooting Path Issues

If you encounter path-related errors:

1. **Check CARLA Installation**: Ensure CARLA 0.9.15 is installed in the expected location
2. **Verify Egg File**: Check that `carla-0.9.15-py3.7-win-amd64.egg` exists in the dist folder
3. **Python Version**: Ensure you're using Python 3.7+ for compatibility
4. **Manual Path Setup**: If automatic detection fails, you can manually set paths in the scripts

## Output Files

The system generates several output files:

1. **JSON Results**: Detailed navigation and sensor fusion data
2. **Trajectory Plots**: Visual representation of vehicle paths
3. **Real-time Plots**: Live visualization during navigation
4. **Statistics**: Performance metrics and analysis

## Troubleshooting

### Common Issues

1. **Vehicle not moving**: Check PID parameters and waypoint reach distance
2. **Poor navigation**: Adjust PID controller gains
3. **Route generation fails**: Check spawn points and map connectivity
4. **Sensor fusion errors**: Verify sensor setup and Kalman filter parameters
5. **Path resolution errors**: Check CARLA installation and Python version compatibility

### Performance Tuning

1. **For smoother navigation**: Increase PID integral gains
2. **For faster response**: Increase PID proportional gains
3. **For stability**: Increase PID derivative gains
4. **For safety**: Reduce maximum speed and increase waypoint reach distance

## Advanced Usage

### Custom Route Planning

```python
# Create custom waypoint chain
start_waypoint = world.get_map().get_waypoint(start_location)
waypoints = generator.generate_waypoint_chain(
    start_waypoint,
    num_waypoints=10,
    distance_between=50.0
)
```

### Traffic Manager Integration

```python
# Use traffic manager for route generation
traffic_manager = world.get_traffic_manager()
waypoints = generator.generate_traffic_manager_route(
    vehicle,
    destination,
    traffic_manager
)
```

### Multiple Vehicles

```python
# Create multiple navigators for different vehicles
navigators = []
for vehicle in vehicles:
    navigator = WaypointNavigator(vehicle)
    navigators.append(navigator)

# Coordinate navigation
for navigator in navigators:
    navigator.navigate_step(0.05)
```

## Contributing

To extend the navigation system:

1. Add new route generation methods to `WaypointGenerator`
2. Implement custom PID controllers in `WaypointNavigator`
3. Create new attack testing scenarios
4. Add additional sensor fusion metrics

## Dependencies

- CARLA 0.9.15+
- NumPy
- Matplotlib (for visualization)
- Your existing sensor fusion modules

The system is designed to be modular and easily extensible for your research needs.

## Recent Updates

### Path Resolution Fixes (Latest)

- **Automatic CARLA Detection**: System now automatically detects CARLA installation paths
- **Windows Compatibility**: Fixed path resolution for Windows environments
- **Egg File Support**: Added support for both `.egg` and `.whl` CARLA packages
- **Fallback Mechanisms**: Error handling for path detection failures

### File Renames and Organization

- `simple_navigation_example.py` → `simple_navigation_test.py`
- Removed `sensor_fusion_navigation_example.py` (was too complex for initial testing)
- Updated all import paths and references

### Integration Improvements

- Seamless integration with your existing sensor fusion research
- Support for all GPS spoofing strategies from your research
- Maintains compatibility with innovation-aware attack testing
- Provides additional metrics for attack effectiveness analysis
