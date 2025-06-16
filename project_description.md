# Autonomous Vehicle Sensor Fusion and Security Research Project

## Project Overview

This research project focuses on developing robust sensor fusion techniques and security measures for autonomous vehicles, with particular emphasis on GPS spoofing detection and mitigation using multi-sensor fusion approaches.

## Core Research Areas

### 1. Sensor Fusion and Integration

- Investigation of multi-sensor fusion combining:
  - Camera (object detection and depth perception)
  - LiDAR (distance detection + AI-based object detection)
  - Radar (distance detection calculations)
  - GPS (absolute positioning)
  - IMU (relative positioning and motion tracking)
- Analysis of sensor redundancy and complementary use cases
- Development of fallback mechanisms where secondary sensors activate if primary sensors fail

### 2. GPS Spoofing Detection and Mitigation

#### Current Implementation

- Integration of GPS and IMU with Kalman filtering
- Development of machine learning-based detection algorithms
- Creation of test scenarios for various spoofing attacks

#### Research Goals

1. **Phase 1: Basic Implementation**

   - Establish working GPS-IMU-Kalman filter setup
   - Verify proper navigation functionality
   - Document sensor fusion behavior

2. **Phase 2: Attack Development**

   - Create sophisticated spoofing methods that can bypass sensor fusion
   - Implement sequential and parallel attack vectors
   - Test against Kalman filter detection capabilities

3. **Phase 3: Defense Mechanisms**
   - Develop detection methods for spoofing attacks
   - Implement correction mechanisms
   - Create machine learning models for attack classification

### 3. Machine Learning Implementation

- Dataset creation for spoofed and non-spoofed scenarios
- Development of ML models for:
  - Attack detection
  - Attack classification
  - Zero-day attack detection
- Performance analysis of detection speed and accuracy
- Implementation of one-class classifiers for unknown attack detection

### 4. Testing and Validation

- Integration with CARLA simulation environment
- Testing under various conditions:
  - Different weather conditions
  - Various traffic scenarios
  - Multiple map environments
- Evaluation of sensor performance in mixed traffic scenarios

## Technical Implementation Details

### Sensor Fusion Architecture

- GPS provides absolute positioning data
- IMU tracks relative movement from initial position
- Kalman filter implementation for error correction
- Feedback loop between sensors for error mitigation

### Current Focus

1. Understanding IMU-Kalman filter interaction
2. Documenting error accumulation and correction mechanisms
3. Testing existing attacks against sensor fusion
4. Developing new attack vectors if current ones are mitigated


