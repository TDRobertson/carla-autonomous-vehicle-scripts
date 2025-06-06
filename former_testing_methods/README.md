# Former Testing Methods

This directory contains the historical development phases of the autonomous vehicle control system. These methods were used during the initial development and testing phases of the project, before the current sensor fusion implementation.

## Directory Structure

### 1. data_collection/

Contains scripts used for collecting training and testing data from the CARLA simulator. This includes:

- Camera image capture
- Vehicle state data (steering, throttle, brake, speed)
- Data storage in CSV format
- Data preprocessing utilities

### 2. navigation_scripts/

Contains the core vehicle control implementation, including:

- PID controller for vehicle control
- Waypoint following algorithms
- Coordinate system conversions (xyz to geolocation)
- Basic path planning

### 3. utility_scripts/

Contains helper functions and utilities used across the project:

- Waypoint visualization tools
- Map analysis utilities
- Data processing helpers
- Debug visualization tools

### 4. deprecated/

Contains older versions of scripts and experimental features that were replaced by more robust implementations. This serves as a historical record of the development process and may contain useful reference code.

## Usage Notes

These scripts represent the earlier stages of development and may not be actively maintained. They are kept for reference and historical purposes. For the current implementation, please refer to the `sensor_fusion_testing` directory in the root of the repository.

## Dependencies

The same dependencies as the main project apply:

- Carla 0.9.15
- Python 3.10+
- Numpy
- OpenCV
- Pandas
- Carla library

## Running the Scripts

Each subdirectory may contain its own README with specific instructions. In general:

1. Ensure CARLA server is running
2. Navigate to the specific script directory
3. Run the desired script with Python

Note: These scripts may require specific CARLA map settings or vehicle configurations that are documented in their respective directories.
