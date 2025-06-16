# Integrated Sensor Fusion System

This project implements an integrated sensor fusion system for autonomous vehicles using CARLA simulator. The system combines data from multiple sensors (GNSS, IMU, camera) to provide robust state estimation and GPS spoofing detection.

## Features

- Extended Kalman Filter (EKF) for state estimation
- GPS spoofing detection and correction
- Multiple spoofing attack strategies
- Comprehensive test suite
- CARLA simulator integration

## Directory Structure

```
carla-autonomous-vehicle-scripts/
├── core/
│   ├── car.py
│   ├── kalman_filter.py
│   ├── spoofing_detector.py
│   └── gps_spoofer.py
├── utils/
│   └── rotations.py
├── tests/
│   ├── test_car.py
│   ├── test_kalman_filter.py
│   ├── test_spoofing_detection.py
│   ├── test_gps_spoofer.py
│   ├── test_rotations.py
│   └── test_integrated_system.py
├── run_tests.py
├── requirements.txt
└── README.md
```

## Installation

1. Install CARLA simulator (version 0.9.15 or later)
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start CARLA server:

   ```bash
   ./CarlaUE4.sh -quality-level=Low
   ```

2. Run tests:
   ```bash
   python run_tests.py
   ```

## Components

### Extended Kalman Filter

The EKF implementation provides state estimation using:

- Position (x, y, z)
- Velocity (vx, vy, vz)
- Orientation (quaternion)

### GPS Spoofing Detection

The spoofing detector monitors:

- Position jumps
- Velocity inconsistencies
- Historical position drift
- Sequential attack patterns

### Spoofing Strategies

The system supports multiple spoofing attack strategies:

- Jump attacks
- Drift attacks
- Random noise
- Sequential steps

## Testing

The test suite includes:

- Unit tests for each component
- Integration tests for the complete system
- CARLA simulator integration tests

Run tests with:

```bash
python run_tests.py
```


