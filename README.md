# carla-autonomous-vehicle-scripts

## This repository contains the code used to control the autonomous vehicle in the Carla simulator. The code is written in Python and uses the Carla Python API to interact with the simulator.

## Current Work Phase:

### sensor_fusion_testing/

This is the current active development directory containing the latest implementation of GPS-IMU sensor fusion with spoofing detection capabilities. See the [sensor_fusion_testing README](sensor_fusion_testing/README.md) for detailed documentation.

## Historical Development:

### former_testing_methods/

This directory contains the previous development phases of the project, organized into:

1. data_collection: Contains the code used to collect data from the simulator. The data collected includes images from the front camera, steering angle, throttle, brake, and speed. The data is saved in a CSV file.
2. navigation_scripts: Contains the code used to control the vehicle in the simulator. The code uses a simple PID controller to control the steering angle, throttle, and brake of the vehicle. The code also includes a simple waypoint following algorithm to navigate the vehicle to a given destination. Certain files use the built-in xyz coordinate system, others use a workaround formula to convert actor xyz location outputs into their relevant geolocation data to be used for navigation.
3. utility_scripts: Contains utility functions used by the data collection and navigation scripts. As of right now just includes scripts to show waypoint locations throughout the map.
4. deprecated: Contains old code that is no longer used. This code was generally used to slowly build new features to add to the main scripts. Essentially serves as a history of the development process.

## Dependencies:

- Carla 0.9.15
- Python 3.10+
- Numpy
- OpenCV
- Pandas
- Carla library

## Analysis and Results

See [analysis.md](./analysis.md) for a detailed summary of the research goals, file integrations, test methodology, results, and interpretation. This document is suitable for presentation and includes a summary table of results for all spoofing attacks tested.

## How to run (Sensor Fusion Testing):

1. Start the CARLA simulator (CarlaUE4)
2. Activate your virtual environment
3. From your project root, run:
   ```sh
   python run_all.py
   ```
   - This opens three terminals for scene setup, Kalman/trajectory visualization, and camera visualization.
4. In a fourth terminal, run:
   ```sh
   python sensor_fusion_testing/integration_files/sequential_attack_test.py
   ```
   - This runs the attack test and prints results.
5. For post-run analysis, run:
   ```sh
   python sensor_fusion_testing/integration_files/data_processor.py
   ```

For historical methods, refer to the specific READMEs in each subdirectory

## Notes:

- The code is still a work in progress and may contain bugs
- The code is written for Carla 0.9.15 and may not work with other versions of Carla
- The code is written for Python 3.10+ and may not work with older versions of Python
- The code is written for a windows host machine, and may not work on other operating systems.
  - only a few lines of code is needed to convert several scripts to their linux counterparts, but the majority of the code is platform agnostic.

## Future work:

- Add more advanced control algorithms
- Add more advanced navigation algorithms
- Conversion of all scripts to be ros compatible
- Improve CARLA-ROS-BRIDGE setup instructions for WSL and Linux systems

### Linux conversion notes:

- The only scripts that need to be converted are the navigation scripts, as they are the only scripts that use the os library to interact with the host machine.
  Replace the following lines:

```python
# windows
# Ensure the CARLA Python API path is correctly added
carla_path = 'C:/CARLA_0.9.15/PythonAPI/carla'
if carla_path not in sys.path:
    sys.path.append(carla_path)

# linux
# Ensure the CARLA Python API path is correctly added
carla_path = '/home/your_username/CARLA_0.9.15/PythonAPI/carla'
if carla_path not in sys.path:
    sys.path.append(carla_path)
```

Optionally, set up the carla root in your environment variables and use:

```python
carla_path = os.path.join(os.environ.get('CARLA_ROOT', ''), 'PythonAPI', 'carla')
```
