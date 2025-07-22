# Sensor Fusion Testing Framework

This directory contains a comprehensive testing framework for sensor fusion algorithms in autonomous vehicles using CARLA simulator. The framework includes visualization tools, sensor fusion implementations, GPS spoofing detection, and machine learning analysis capabilities.

## Overview

The sensor fusion testing framework is designed to:

- Test and validate sensor fusion algorithms using GPS and IMU data
- Simulate various GPS spoofing attacks
- Visualize vehicle trajectories and sensor data in real-time
- Analyze the effectiveness of spoofing detection algorithms
- Generate datasets for machine learning-based attack detection

## Analysis and Results

See [../analysis.md](../analysis.md) for a detailed summary of the research goals, file integrations, test methodology, results, and interpretation. This document includes a comprehensive analysis of all spoofing attacks tested and is suitable for presentation.

## File Structure

### Core Testing Files

#### `fpv_ghost.py`

**Purpose**: Dual-camera visualization system for vehicle monitoring

**Features**:

- **First-Person View (FPV)**: Camera positioned at driver's eye level
- **Ghost/Overhead View**: Camera positioned above and behind the vehicle
- **Resizable Window**: Dynamic window resizing with automatic camera resolution adjustment
- **Real-time Rendering**: Continuous display of both camera feeds side-by-side

**Use Case**: Ideal for testing and debugging autonomous vehicle behavior, providing both driver perspective and overhead view simultaneously.

#### `scene.py`

**Purpose**: Autonomous vehicle navigation setup and testing environment

**Features**:

- **Fixed Route Navigation**: Spawns Tesla Model 3 at specific starting point and navigates to predefined destination
- **Traffic Management**: Uses CARLA's traffic manager with aggressive settings (ignores lights/signs, close following distance)
- **Visual Markers**: Draws destination markers and arrival zones in the world
- **Stuck Detection**: Monitors vehicle progress and attempts to "nudge" stuck vehicles
- **Automatic Stopping**: Stops vehicle when it reaches the destination zone

**Use Case**: Creates a controlled testing environment for autonomous navigation algorithms.

#### `sync.py`

**Purpose**: Advanced sensor fusion system with Extended Kalman Filter (EKF)

**Features**:

- **Extended Kalman Filter**: 9-state EKF for position, velocity, and orientation estimation
- **Multi-Sensor Integration**: Combines GPS (20 Hz) and IMU (100 Hz) data
- **Real-time Visualization**: Pygame-based display showing three trajectory types
- **Error Analysis**: Calculates and displays positioning errors between different methods

**Components**:

- `IMUIntegrator`: Implements quaternion-based orientation updates and state prediction
- `VehicleMonitor`: Manages sensor setup, data collection, and visualization

**Use Case**: Comprehensive sensor fusion testing and validation tool for evaluating positioning accuracy and sensor fusion algorithms.

### Integration Files (`integration_files/`)

#### `data_processor.py`

**Purpose**: Data processing and machine learning preparation

**Features**:

- **Data Loading**: Loads test results from JSON files for all spoofing strategies
- **Feature Engineering**: Creates derived features including moving averages, standard deviations, and 3D error metrics
- **ML Data Preparation**: Prepares training and test datasets with feature scaling
- **Statistical Analysis**: Generates correlation matrices and feature importance analysis
- **Data Validation**: Handles missing data and ensures consistent array lengths

**Use Case**: Processes raw sensor fusion test data into machine learning-ready datasets.

#### `gps_spoofer.py`

**Purpose**: GPS spoofing attack simulation

**Features**:

- **Multiple Attack Strategies**: Implements four different spoofing techniques
- **Gradual Drift**: Slowly drifts away from true position over time
- **Sudden Jump**: Creates random position jumps at intervals
- **Random Walk**: Generates random movement patterns
- **Replay Attack**: Records and replays previous position data

**Attack Types**:

- `GRADUAL_DRIFT`: Subtle position drift using sine/cosine functions
- `SUDDEN_JUMP`: Random position jumps with 1% probability
- `RANDOM_WALK`: Random step-based movement
- `REPLAY`: Records positions and replays them after a delay

**Use Case**: Simulates realistic GPS spoofing attacks for testing detection algorithms.

#### `sensor_fusion.py`

**Purpose**: Enhanced sensor fusion with spoofing integration

**Features**:

- **Kalman Filter Integration**: Uses custom Kalman filter for state estimation
- **Spoofing Support**: Integrates with GPS spoofer for attack simulation
- **Multi-Sensor Setup**: Configures GPS and IMU sensors with proper sampling rates
- **State Management**: Tracks true position, fused position, and velocity estimates
- **Metrics Collection**: Provides Kalman filter metrics and error calculations

**Use Case**: Core sensor fusion implementation that can be used with or without spoofing attacks.

#### `sequential_attack_test.py`

**Purpose**: Automated testing framework for sequential spoofing attacks

**Features**:

- **Sequential Testing**: Runs multiple attack strategies in sequence
- **Comprehensive Data Collection**: Collects position, velocity, IMU, and Kalman filter data
- **Real-time Visualization**: Updates plots during testing
- **Statistical Analysis**: Calculates error statistics for each attack type
- **Data Export**: Saves raw data and statistics to JSON files

**Test Sequence**:

1. Gradual Drift Attack (30 seconds)
2. Sudden Jump Attack (30 seconds)
3. Random Walk Attack (30 seconds)
4. Replay Attack (30 seconds)

**Use Case**: Automated testing of sensor fusion systems against various attack scenarios.

#### `spoofing_detector.py`

**Purpose**: Machine learning-based spoofing detection system

**Features**:

- **Multi-Strategy Detection**: Detects all four types of spoofing attacks
- **Statistical Analysis**: Uses position history and velocity patterns
- **Confidence Scoring**: Provides confidence levels for each detection
- **Position Correction**: Attempts to correct spoofed positions
- **Pattern Recognition**: Identifies repeating patterns and temporal inconsistencies

**Detection Methods**:

- **Gradual Drift**: Analyzes position trends and velocity consistency
- **Sudden Jump**: Detects position discontinuities and velocity anomalies
- **Random Walk**: Analyzes movement statistics and variance
- **Replay**: Identifies temporal patterns and repeating sequences

**Use Case**: Real-time detection and correction of GPS spoofing attacks.

## Usage Examples

### Complete Pipeline Testing

```bash
# From project root, launch all visualizations and scene setup
python run_all.py

# In a separate terminal, run the attack testing
python sensor_fusion_testing/integration_files/sequential_attack_test.py

# For post-run analysis
python sensor_fusion_testing/integration_files/data_processor.py
```

### Individual Component Testing

```bash
# Run the main sensor fusion visualization
python sync.py

# Set up autonomous navigation scenario
python scene.py

# View dual camera feeds
python fpv_ghost.py

# Test sensor fusion with spoofing
python integration_files/sensor_fusion.py
```

### Advanced Attack Testing

```bash
# Run sequential attack testing
python integration_files/sequential_attack_test.py

# Process test results for ML
python integration_files/data_processor.py
```

## Dependencies

- **CARLA**: Autonomous driving simulator
- **NumPy**: Numerical computing
- **Pygame**: Real-time visualization
- **SciPy**: Scientific computing (for quaternion operations)
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning utilities
- **Matplotlib**: Plotting and visualization

## Output Files

The framework generates several types of output:

### Test Results (`test_results/`)

- `raw_data.json`: Raw sensor data for each attack strategy
- `statistics.json`: Statistical analysis of errors and performance
- Various visualization plots (PNG format)

### Machine Learning Data (`ml_data/`)

- `X_train.npy`, `X_test.npy`: Feature matrices
- `y_train.npy`, `y_test.npy`: Label matrices
- `scaler_params.json`: Feature scaling parameters
- `feature_importance.csv`: Feature importance analysis

## Key Features

1. **Real-time Visualization**: Live monitoring of vehicle trajectories and sensor data
2. **Comprehensive Testing**: Automated testing across multiple attack scenarios
3. **Data Analysis**: Statistical analysis and machine learning preparation
4. **Modular Design**: Components can be used independently or together
5. **Extensible Framework**: Easy to add new attack types or detection methods

## Contributing

When adding new features:

1. Follow the existing code structure and naming conventions
2. Add appropriate documentation and comments
3. Include error handling for robustness
4. Test with multiple attack scenarios
5. Update this README with new functionality

## Notes

- Ensure CARLA server is running on localhost:2000 before running any scripts
- Some scripts require a vehicle with `role_name='hero'` to be present in the simulation
- The framework is designed for research and testing purposes
- All position data is in CARLA's coordinate system (meters)
- Time measurements are in seconds from simulation start

## Matplotlib Windows ImportError Note

**Important:**

If you encounter an error like:

```
ImportError: DLL load failed while importing _c_internal_utils: The specified module could not be found.
```

This is due to a known bug with the matplotlib 3.9.1 wheels on Windows (see [matplotlib issue #28551](https://github.com/matplotlib/matplotlib/issues/28551)).

**How to fix:**

- Upgrade to matplotlib version 3.10.3 or later:
  ```sh
  pip install matplotlib>=3.10.3
  ```
- Alternatively, you can force pip to only use binary wheels (never build from source):
  ```sh
  pip install --only-binary "matplotlib" matplotlib
  ```

**Do not use matplotlib 3.9.1 on Windows.**

For more details, see the [official comment from the maintainer](https://github.com/matplotlib/matplotlib/issues/28551#issuecomment-2249649647).
