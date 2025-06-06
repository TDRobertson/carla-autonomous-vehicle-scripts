# Sensor Fusion Vehicle Script

This script implements a sensor fusion system for autonomous vehicles in CARLA, combining camera, LiDAR, and radar sensors with the CARLA traffic manager.

## Features

- Configurable sensor setup (Camera, LiDAR, Radar)
- Integration with CARLA's traffic manager for autopilot
- Radar-based emergency braking system
- Sensor data collection and processing
- Configurable traffic vehicle spawning
- Clean actor cleanup on exit

## Requirements

- CARLA 0.9.15
- Python 3.7+
- NumPy
- OpenCV
- CARLA Python API

## Usage

1. Start the CARLA simulator
2. Run the script with desired sensor configuration:

```bash
# Run with all sensors enabled (default) and 20 traffic vehicles
python multi_sensor_vehicle.py

# Run with specific sensors disabled
python multi_sensor_vehicle.py --no-camera  # Disable camera
python multi_sensor_vehicle.py --no-lidar   # Disable LiDAR
python multi_sensor_vehicle.py --no-radar   # Disable radar

# Run with multiple sensors disabled
python multi_sensor_vehicle.py --no-camera --no-lidar  # Only radar enabled

# Adjust number of traffic vehicles
python multi_sensor_vehicle.py --num-traffic 10  # Spawn 10 traffic vehicles
python multi_sensor_vehicle.py --num-traffic 0   # No traffic vehicles

# Enable spectator camera to follow the ego vehicle
python multi_sensor_vehicle.py --spectator

# Manual vehicle control
# In the toggle window, press 'm' to enable or disable manual mode. When manual mode is enabled, use WASD to drive the vehicle. The toggle window must be focused for keypresses to register.
```

## Sensor Configuration

### Camera

- Resolution: 800x600
- FOV: 90 degrees
- Position: (1.5, 0, 2.4) relative to vehicle

### LiDAR

- Channels: 32
- Points per second: 100,000
- Rotation frequency: 20 Hz
- Range: 20 meters
- Position: (0, 0, 2.4) relative to vehicle

### Radar

- Horizontal FOV: 30 degrees
- Vertical FOV: 10 degrees
- Range: 20 meters
- Position: (2.0, 0, 1.0) relative to vehicle

## Traffic Configuration

- Default number of vehicles: 20 (configurable via --num-traffic)
- Vehicles are spawned at random available spawn points
- Each vehicle uses CARLA's traffic manager for navigation
- Random speed variations (-20% to +10% of speed limit)
- Random following distances (1.0 to 4.0 meters)
- Deterministic behavior with fixed random seed (42)

## Behavior

- The ego vehicle operates in autopilot mode using CARLA's traffic manager
- Radar detection triggers emergency braking when objects are within 5 meters
- Emergency braking intensity is proportional to object proximity
- Normal operation resumes after emergency stop if path is clear
- Traffic vehicles follow traffic rules and avoid collisions

## Notes

- The script uses a Tesla Model 3 as the ego vehicle
- The traffic manager is configured with a 2.0 meter following distance for the ego vehicle
- Emergency braking is triggered only when vehicle speed is above 5 km/h
- Traffic vehicles are automatically cleaned up when the script exits
- The script uses synchronous mode for more deterministic behavior
