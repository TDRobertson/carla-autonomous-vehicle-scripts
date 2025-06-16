# GPS-IMU Sensor Fusion with Spoofing Detection

This project implements a sensor fusion system using GPS and IMU data in the CARLA simulator, with integrated GPS spoofing capabilities and Kalman filter-based state estimation. The system is designed to study the robustness of sensor fusion algorithms against various GPS spoofing attacks.

## System Architecture

### 1. Sensor Fusion System

The sensor fusion system combines data from two primary sensors:

- **GPS (GNSS)**: Provides absolute position measurements
- **IMU**: Provides acceleration and angular velocity measurements

The system uses an Extended Kalman Filter (EKF) to fuse these measurements and maintain a continuous state estimate of the vehicle's position and velocity.

### 2. Kalman Filter Implementation

The Kalman filter is implemented as a 6-state system:

- State vector: [x, y, z, vx, vy, vz]
  - Position (x, y, z)
  - Velocity (vx, vy, vz)

Key components:

- **State Transition Matrix (A)**: Implements a constant velocity model
- **Measurement Matrix (H)**: Maps state to GPS measurements
- **Process Noise Covariance (Q)**: Models system uncertainty
- **Measurement Noise Covariance (R)**: Models sensor noise

The filter operates in two steps:

1. **Prediction**: Projects state forward using the constant velocity model
2. **Update**: Incorporates GPS measurements to correct the state estimate

### 3. GPS Spoofing Module

The spoofing module implements four distinct attack strategies:

1. **Gradual Drift**

   - Implements a sinusoidal drift pattern
   - Parameters:
     - Drift rate: 0.1 m/s
     - Pattern: Sinusoidal in x and y coordinates
   - Designed to mimic natural GPS drift

2. **Sudden Jump**

   - Creates random position jumps
   - Parameters:
     - Jump magnitude: 5.0 meters
     - Probability: 1% per update
   - Simulates signal hijacking attacks

3. **Random Walk**

   - Implements a random walk pattern
   - Parameters:
     - Step size: 0.5 meters
     - Independent in x and y coordinates
   - Models random interference

4. **Replay Attack**
   - Records and replays previous positions
   - Parameters:
     - Buffer size: 100 positions
     - Replay delay: 2.0 seconds
   - Simulates signal replay attacks

## Machine Learning Integration (In Progress)

The system is currently being enhanced with machine learning capabilities:

1. **Attack Detection Model**

   - Supervised learning for known attack patterns
   - Feature extraction from sensor data
   - Real-time classification of attack types
   - Confidence scoring for detections

2. **Zero-Day Attack Detection**
   - One-class classification for anomaly detection
   - Unsupervised learning approach
   - Pattern recognition for unknown attacks
   - Continuous learning system

## Analysis and Results

Detailed analysis of the system's performance and test results can be found in:

- `analysis.md`: Comprehensive analysis in markdown format
- `analysis.pdf`: Detailed technical report with visualizations

Key metrics tracked:

- Detection accuracy
- False positive rates
- System performance
- Response times
- Recovery capabilities

## Sequential Attack Testing Environment

The sequential attack testing environment provides a framework for testing the system's response to multiple spoofing attacks in sequence. This is crucial for evaluating the system's ability to detect and adapt to changing attack patterns.

### 1. Test Structure

The testing environment is implemented in `sequential_attack_test.py` and provides:

- **Automated Attack Sequencing**: Automatically switches between different attack types
- **Configurable Test Parameters**: Adjustable duration for each attack type
- **Comprehensive Data Collection**: Records position data, error metrics, and timestamps
- **Real-time Analysis**: Provides immediate feedback during testing
- **Detailed Post-test Analysis**: Generates statistics for each attack type

### 2. Data Collection for ML Training

The system collects comprehensive data for machine learning model training:

#### 2.1 Position and Velocity Data

- True and fused positions
- True and fused velocities
- Position and velocity errors
- Error rates and variances

#### 2.2 IMU Data

- Accelerometer readings
- Gyroscope readings
- Angular velocity measurements

#### 2.3 Kalman Filter Metrics

- Kalman gains
- Innovation vectors
- Innovation covariances

#### 2.4 Statistical Features

- Position variance
- Velocity variance
- Acceleration variance

#### 2.5 Attack-specific Features

- Attack type
- Attack confidence
- Attack duration

#### 2.6 Environmental Features

- Vehicle speed
- Vehicle acceleration
- Vehicle angular velocity

#### 2.7 Temporal Features

- Time since last update
- Update frequency

#### 2.8 Error Metrics

- Position error rate
- Velocity error rate
- Acceleration error rate

### 3. Data Visualization

The system supports various visualization capabilities for analyzing the collected data:

#### 3.1 Real-time Visualization

- Position tracking plot
- Error evolution over time
- Velocity profiles
- Attack type indicators

#### 3.2 Post-test Analysis Plots

- Error distribution histograms
- Position error heatmaps
- Velocity error patterns
- Attack transition analysis

#### 3.3 Statistical Visualizations

- Box plots for error distributions
- Scatter plots for position errors
- Time series analysis
- Correlation matrices

#### 3.4 Attack-specific Visualizations

- Attack signature plots
- Transition analysis
- Error pattern recognition
- Detection confidence plots

### 4. Running Tests

To run a sequential attack test:

1. Ensure CARLA server is running
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Execute the test script:
   ```bash
   python sequential_attack_test.py
   ```
4. The test will:
   - Spawn a vehicle in CARLA
   - Enable autopilot
   - Run through each attack type (default 30 seconds each)
   - Collect and analyze data
   - Display detailed statistics

### 5. Test Configuration

The test sequence can be customized by modifying the attack sequences in the main function:

```python
tester.add_attack_sequence(
    SpoofingStrategy.GRADUAL_DRIFT,
    30.0,  # Duration in seconds
    "Gradual Drift Attack"
)
```

Each attack sequence requires:

- Attack strategy type
- Duration
- Description

### 6. Analysis Output

The test provides detailed analysis for each attack type:

- Average error
- Maximum error
- Minimum error
- Error standard deviation
- Velocity error statistics
- Error rate analysis

Example output:

```
Attack: Gradual Drift Attack
Average Error: 0.123 meters
Max Error: 0.145 meters
Min Error: 0.098 meters
Error Std Dev: 0.012 meters
Average Velocity Error: 0.045 m/s
Max Velocity Error: 0.067 m/s
Average Error Rate: 0.023 m/s
Max Error Rate: 0.034 m/s
```

### 7. Data Export

The collected data can be exported in various formats for ML training:

- CSV format for tabular data
- JSON format for structured data
- NumPy arrays for numerical analysis
- Pandas DataFrames for data manipulation

### 8. ML Training Preparation

The collected data is structured to support:

- Supervised learning (attack classification)
- Unsupervised learning (anomaly detection)
- Time series analysis
- Pattern recognition
- Feature engineering

## Implementation Details

### Sensor Fusion

The sensor fusion system is implemented in `sensor_fusion.py` and uses an Extended Kalman Filter to combine GPS and IMU measurements. The system maintains a continuous state estimate of the vehicle's position and velocity, with error correction based on sensor measurements.

### Spoofing Detection

The spoofing detection system is implemented in `spoofing_detector.py` and uses a combination of statistical analysis and machine learning to identify potential GPS spoofing attacks. The system can detect both known attack patterns and anomalies that may indicate zero-day attacks.

### Data Processing

The data processing system is implemented in `data_processor.py` and handles:

- Raw sensor data preprocessing
- Feature extraction
- Data normalization
- Statistical analysis
- ML feature preparation

### Visualization

The visualization system is implemented in `visualization.py` and provides:

- Real-time data plotting
- Error visualization
- Attack pattern visualization
- Statistical analysis plots
- Performance metrics display

## Future Work

1. **Enhanced ML Integration**

   - Implement deep learning models
   - Add transfer learning capabilities
   - Improve zero-day detection
   - Optimize model performance

2. **System Optimization**

   - Reduce computational overhead
   - Improve real-time performance
   - Enhance detection accuracy
   - Minimize false positives

3. **Extended Testing**
   - More attack scenarios
   - Edge case handling
   - System stress testing
   - Long-term reliability testing

## Visualization Capabilities

The testing environment includes comprehensive visualization capabilities to help analyze the performance of the sensor fusion system under different attack scenarios. These visualizations are automatically generated during testing and saved to the `test_results` directory.

### Real-time Visualizations

During test execution, the following visualizations are updated in real-time:

1. **Position Tracking**

   - Plots true vs. fused positions over time
   - Shows the trajectory of the vehicle and how well the fusion system tracks it
   - Helps identify when and where the system loses track of the true position

2. **Error Evolution**

   - Shows how the position error changes over time
   - Helps identify patterns in error behavior during different attack types
   - Useful for understanding the system's response to attacks

3. **Velocity Profiles**
   - Compares true and fused velocity estimates
   - Helps identify velocity estimation errors
   - Useful for understanding how attacks affect velocity estimation

### Post-test Analysis Plots

After test completion, additional visualizations are generated:

1. **Error Distribution**

   - Histogram of position errors
   - Shows the statistical distribution of errors
   - Helps identify if errors follow a particular pattern

2. **Position Error Heatmap**

   - 2D heatmap showing error magnitude across the vehicle's path
   - Helps identify areas where the system performs poorly
   - Useful for understanding spatial patterns in errors

3. **Attack Transitions**

   - Shows when different attack types were active
   - Helps correlate system performance with attack types
   - Useful for understanding system behavior during attack switches

4. **Metric Correlations**
   - Correlation matrix of all collected metrics
   - Helps identify relationships between different measurements
   - Useful for feature selection in machine learning models

### Using the Visualizations

The visualizations are automatically saved in the `test_results` directory with the following naming convention:

- `position_tracking_{attack_type}.png`
- `error_evolution_{attack_type}.png`
- `velocity_profiles_{attack_type}.png`
- `error_distribution_{attack_type}.png`
- `error_heatmap_{attack_type}.png`
- `correlation_matrix.png`

These visualizations can be used to:

1. Compare performance across different attack types
2. Identify patterns in system behavior
3. Validate the effectiveness of the sensor fusion system
4. Guide improvements to the system
5. Support machine learning model development

# Sensor Fusion Testing Environment

This environment provides tools for testing and analyzing sensor fusion systems under various GPS spoofing attacks.

## Features

- Real-time sensor fusion testing in CARLA simulator
- Multiple GPS spoofing attack strategies
- Comprehensive data collection and analysis
- Visualization tools for attack detection
- Sequential attack testing capabilities

## Components

### 1. Sensor Fusion System

- Integrates GPS and IMU data
- Implements Kalman filtering
- Handles sensor data fusion
- Manages spoofing detection

### 2. GPS Spoofer

- Implements multiple spoofing strategies:
  - Gradual Drift
  - Sudden Jump
  - Random Walk
  - Replay Attack
- Configurable attack parameters
- Real-time position manipulation

### 3. Spoofing Detector

- Detects various types of spoofing attacks
- Implements multiple detection methods
- Provides confidence scores
- Real-time attack classification

### 4. Visualization Tools

- Real-time position tracking
- Error evolution plots
- Velocity profile analysis
- Error distribution visualization
- Correlation matrix analysis
- Position error heatmaps

### 5. Sequential Attack Testing

- Automated attack sequence execution
- Configurable test parameters
- Comprehensive data collection
- Real-time analysis
- Detailed post-test analysis

## Data Collection

The system collects the following data during testing:

1. Position Data:

   - True vehicle positions
   - Fused position estimates
   - Position errors
   - Error rates

2. Velocity Data:

   - True vehicle velocities
   - Fused velocity estimates
   - Velocity errors
   - Error statistics

3. Sensor Data:

   - IMU measurements
   - GPS readings
   - Kalman filter metrics

4. Analysis Metrics:
   - Error distributions
   - Correlation matrices
   - Attack detection rates
   - Performance statistics

## Running Tests

### Basic Testing

```bash
python sensor_fusion.py
```

### Sequential Attack Testing

```bash
python sequential_attack_test.py
```

### Test Configuration

- Modify attack sequences in `sequential_attack_test.py`
- Adjust test parameters in the main function
- Configure visualization options in `visualization.py`

## Analysis Output

The system provides comprehensive analysis including:

1. Position Error Statistics:

   - Average error
   - Maximum error
   - Minimum error
   - Error standard deviation

2. Velocity Error Statistics:

   - Average velocity error
   - Maximum velocity error
   - Minimum velocity error
   - Velocity error standard deviation

3. Error Rate Analysis:

   - Error rates above thresholds
   - Attack detection rates
   - False positive rates

4. Visualization Outputs:
   - Position tracking plots
   - Error evolution graphs
   - Velocity profiles
   - Error distributions
   - Correlation matrices
   - Error heatmaps

## Data Storage

Test results are stored in:

- Memory during test execution
- Visualization files in the output directory
- Analysis reports in the documentation

## Requirements

- CARLA Simulator
- Python 3.7+
- Required Python packages:
  - numpy
  - matplotlib
  - carla
  - dataclasses
  - typing

## Directory Structure

```
sensor_fusion_testing/
├── sensor_fusion.py
├── gps_spoofer.py
├── spoofing_detector.py
├── visualization.py
├── sequential_attack_test.py
├── kalman_filter.py
├── README.md
└── analysis.md
```
