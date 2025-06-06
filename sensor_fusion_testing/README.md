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

## Implementation Details

### Sensor Fusion

The `SensorFusion` class manages:

- Sensor initialization and data collection
- Kalman filter integration
- Spoofing injection
- State estimation

### Kalman Filter

The `KalmanFilter` class implements:

- State prediction using constant velocity model
- Measurement update using GPS data
- Error covariance management
- State estimation

### GPS Spoofer

The `GPSSpoofer` class provides:

- Multiple spoofing strategies
- Configurable attack parameters
- Real-time position manipulation
- Strategy switching capability

## Results

### Gradual Drift Spoofing Analysis

#### Position Error Characteristics

- **Average Error**: ~0.1 meters (10 cm)
- **Error Stability**: Very stable with minimal fluctuation
- **Error Range**: 0.100009 to 0.100010 meters

#### Kalman Filter Performance

1. **State Estimation**

   - Maintains stable position estimates
   - Effectively tracks vehicle motion
   - Minimal divergence from true position

2. **Spoofing Response**

   - Successfully maintains consistent offset
   - Preserves relative motion characteristics
   - Demonstrates filter robustness

3. **Height Estimation**
   - Z-axis accuracy: ~0.002 meters
   - Maintains ground-level tracking
   - Minimal vertical drift

#### Key Observations

1. The Kalman filter effectively maintains tracking despite spoofing
2. The gradual drift strategy creates a consistent, predictable offset
3. The system demonstrates good stability and reliability
4. The small error magnitude suggests effective sensor fusion

## Usage

1. Ensure CARLA simulator is running
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the sensor fusion system:
   ```bash
   python sensor_fusion.py
   ```

## Configuration

The system can be configured through several parameters:

### Kalman Filter

- Process noise covariance (Q)
- Measurement noise covariance (R)
- State transition matrix (A)
- Measurement matrix (H)

### Spoofing Module

- Drift rate
- Jump magnitude
- Random walk step size
- Replay buffer size

## Future Work

1. **Enhanced Detection**

   - Implement statistical anomaly detection
   - Add machine learning-based spoofing detection
   - Develop real-time attack classification

2. **Improved Fusion**

   - Incorporate additional sensors
   - Implement more sophisticated motion models
   - Add adaptive noise estimation

3. **Advanced Spoofing**
   - Develop more sophisticated attack patterns
   - Implement coordinated multi-sensor attacks
   - Add timing-based spoofing strategies
