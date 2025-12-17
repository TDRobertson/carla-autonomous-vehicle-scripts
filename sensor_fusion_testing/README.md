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

#### `ml_data_collector.py`

**Purpose**: Synchronized data collection for training machine learning models to detect GPS spoofing attacks.

**Features**:

- **Dual GPS Sensor Architecture**: Spawns two GPS sensors at the same vehicle location
  - Sensor 1: Provides true (unspoofed) GPS readings
  - Sensor 2: Applies INNOVATION_AWARE_GRADUAL_DRIFT spoofing
  - Both sensors fire simultaneously, ensuring perfect timestamp synchronization
- **IMU Interpolation**: IMU data (50 Hz) is interpolated to GPS timestamps (10 Hz) using linear interpolation
- **Dual Kalman Filters**: Two parallel Kalman filter instances track true and spoofed paths independently
- **Derived Features**: Calculates ML-ready features including rolling statistics, jerk, and innovation metrics
- **Configurable Attack Timing**: Separate warmup and attack-delay periods for clean/spoofed data segments

**Design Decisions**:

1. **Why Dual Sensors?** Using two GPS sensors at the same physical location guarantees that true and spoofed readings have identical timestamps. This eliminates synchronization issues that would arise from applying spoofing in post-processing.

2. **Why INNOVATION_AWARE_GRADUAL_DRIFT?** This is the most sophisticated (stealthy) attack strategy. It adapts its drift rate based on Kalman filter innovation values to avoid detection thresholds, making it the hardest attack to detect and thus the most valuable for ML training.

3. **Why Interpolate IMU to GPS?** GPS is the slower sensor (10 Hz vs 50 Hz for IMU). Interpolating IMU data to GPS timestamps ensures every data point has complete sensor information without upsampling GPS data (which would introduce artifacts).

4. **Why Dual Kalman Filters?** Running separate KF instances for true and spoofed paths allows direct comparison of filter behavior. The innovation values from the spoofed KF path are key features for anomaly detection.

5. **Why Separate Clean/Attack Periods?** The configurable `attack_start_delay` allows collection of clean baseline data before spoofing begins. This is essential for one-class classifiers that train only on clean data. Setting `attack_delay=0` starts the attack immediately, ensuring every data point has a corresponding true/spoofed pair (useful for testing and supervised learning).

6. **Why Random Attack Mode?** Real-world attacks don't start at predictable times. Random start/stop attacks create more realistic training data and help ML models learn to detect attacks regardless of when they occur, improving generalization.

**Data Schema** (per timestamp):

| Field                | Description                               |
| -------------------- | ----------------------------------------- |
| `timestamp`          | CARLA simulation time (seconds)           |
| `true_gps_x/y/z`     | True GPS position (meters)                |
| `spoofed_gps_x/y/z`  | Spoofed GPS position (meters)             |
| `imu_accel_x/y/z`    | Accelerometer (m/s^2), interpolated       |
| `imu_gyro_x/y/z`     | Gyroscope (rad/s), interpolated           |
| `imu_compass`        | Heading (radians)                         |
| `kf_true_x/y/z`      | Kalman filter estimate (true GPS path)    |
| `kf_spoof_x/y/z`     | Kalman filter estimate (spoofed GPS path) |
| `innovation_true`    | Innovation magnitude (true path)          |
| `innovation_spoof`   | Innovation magnitude (spoofed path)       |
| `position_error`     | Euclidean distance: true vs spoofed GPS   |
| `kf_tracking_error`  | Euclidean distance: true GPS vs KF_spoof  |
| `velocity_magnitude` | Derived from position change rate         |
| `is_attack_active`   | Label: 1 if spoofing applied, 0 otherwise |

**Derived Features** (calculated post-collection):

- `accel_magnitude`, `gyro_magnitude` - Sensor magnitudes
- `jerk_x/y/z`, `jerk_magnitude` - Rate of change of acceleration
- `position_error_rate` - How fast position error is growing
- `innovation_spoof_ma/std/max` - Rolling window statistics (window=10)
- `position_error_ma/std` - Rolling error statistics
- `gps_drift_x/y/z` - Component-wise drift values
- `kf_diff_magnitude` - Difference between true and spoofed KF estimates

**MSE Formula**:

The script calculates Mean Squared Error as:

```
MSE = (1/n) * sum((true_position - spoofed_position)^2)
RMSE = sqrt(MSE)  # Same units as measurement (meters)
```

For 3D position error:

```
SE_i = (x_true - x_spoof)^2 + (y_true - y_spoof)^2 + (z_true - z_spoof)^2
```

**Output Files**:

- `data/ml_training_data_{timestamp}.csv` - Flat CSV for sklearn/pandas
- `data/ml_training_data_{timestamp}.json` - Full metadata + raw data

**Use Case**: Generates high-quality, synchronized datasets for training one-class classifiers (Isolation Forest, One-Class SVM, Local Outlier Factor) to detect GPS spoofing attacks.

## Usage Examples

### ML Training Data Collection

**Manual Collection:**

```bash
# Collect synchronized GPS/IMU data for ML training (default: 60s, outputs to data/)
python ml_data_collector.py

# Start attack immediately (attack_delay=0) - useful for testing
# Every data point will have true/spoofed pairs from the start
python ml_data_collector.py --attack-delay 0

# Custom duration and timing
python ml_data_collector.py --duration 120 --warmup 10 --attack-delay 20

# With custom label for organized file naming
python ml_data_collector.py --duration 120 --attack-delay 30 --label train_run01

# Random attack mode - attacks randomly start/stop
python ml_data_collector.py --random-attacks --duration 120

# Random attacks with custom intervals
python ml_data_collector.py --random-attacks \
  --min-attack-duration 5 --max-attack-duration 20 \
  --min-clean-duration 5 --max-clean-duration 15

# Specify output directory and label
python ml_data_collector.py --duration 90 --output-dir data/custom --label exp01_run05
```

**Automated Collection (Recommended):**

Three automation options are available for batch data collection:

**Option 1: Windows Batch Script**

```cmd
cd sensor_fusion_testing
collect_ml_datasets.bat
```

Menu-driven interface with options:

- Quick Test (5 runs × 60s)
- One-Class Training (25 runs × 120s)
- One-Class Validation (5 runs × 180s, random)
- Supervised Training (20 runs × 120s)
- Supervised Validation (10 runs × 150s, random)
- Custom Parameters

**Option 2: Linux/Mac Shell Script**

```bash
cd sensor_fusion_testing
chmod +x collect_ml_datasets.sh  # First time only
./collect_ml_datasets.sh
```

Same menu-driven interface as Windows version.

**Option 3: Python Collection Manager** (Cross-platform, with progress bars)

```bash
cd sensor_fusion_testing

# Interactive mode (menu-driven)
python automated_data_collection.py

# Non-interactive mode (specify preset)
python automated_data_collection.py --preset quick_test
python automated_data_collection.py --preset one_class_training
python automated_data_collection.py --preset supervised_training

# With custom retry settings
python automated_data_collection.py --preset one_class_training --max-retries 5
```

**Features of Automation Scripts:**

- Automatic progress tracking
- Error handling and retry logic
- CARLA connection health checks
- Dataset statistics after collection
- Organized file naming (e.g., `train_run01_ml_training_data_20251207_120001.csv`)
- Resume capability (skips existing runs)

**Manual Collection Parameters**:

- `--duration`: Total collection time in seconds (default: 60)
- `--warmup`: Initial warmup before recording starts (default: 5)
- `--attack-delay`: Seconds of clean data before spoofing begins (default: 10)
  - Set to `0` to start attack immediately - ensures every data point has true/spoofed pairs
  - Useful for testing and supervised learning scenarios
- `--output-dir`: Directory for output files (default: data)
- `--label`: Label prefix for output files (e.g., "train_run01", "val_run05")
  - Output: `{label}_ml_training_data_{timestamp}.csv`
  - Helps organize datasets by experiment or run number
- `--random-attacks`: Enable random start/stop attacks (default: False)
  - Creates more realistic training data with unpredictable attack timing
  - Attacks randomly start and stop throughout collection period
- `--min-attack-duration`: Minimum attack duration in random mode (default: 5.0)
- `--max-attack-duration`: Maximum attack duration in random mode (default: 15.0)
- `--min-clean-duration`: Minimum clean period duration in random mode (default: 5.0)
- `--max-clean-duration`: Maximum clean period duration in random mode (default: 15.0)

**Automation Script Parameters** (Python manager):

- `--preset`: Choose preset configuration (quick_test, one_class_training, etc.)
- `--max-retries`: Maximum retry attempts per run if collection fails (default: 3)

**Example workflows for ML training**:

**Recommended: Automated Collection**

```bash
# 1. Start CARLA simulator (CarlaUE4.exe)

# 2. Run automated collection (Windows)
cd sensor_fusion_testing
collect_ml_datasets.bat

# OR (Linux/Mac)
./collect_ml_datasets.sh

# OR (Python, any platform)
python automated_data_collection.py --preset one_class_training
```

**File Organization:**

Automated scripts produce organized outputs:

```
data/
├── training/
│   ├── train_run01_ml_training_data_20251207_120001.csv
│   ├── train_run01_ml_training_data_20251207_120001.json
│   ├── train_run02_ml_training_data_20251207_122015.csv
│   └── ... (25 runs total)
├── validation/
│   ├── val_run01_ml_training_data_20251207_130045.csv
│   └── ... (5 runs total)
└── test/
    ├── test_run01_ml_training_data_20251207_140001.csv
    └── ... (5 runs total)
```

**Manual Collection Examples:**

**For One-Class Classifiers** (train on clean data only):

```bash
# 1. Start CARLA simulator (CarlaUE4.exe)

# 2. Collect data with clean baseline period
python ml_data_collector.py --duration 120 --attack-delay 30 --label train_run01

# 3. Load and filter to clean data only
import pandas as pd
df = pd.read_csv('data/train_run01_ml_training_data_YYYYMMDD_HHMMSS.csv')
clean_data = df[df['is_attack_active'] == 0]  # Use for training
attack_data = df[df['is_attack_active'] == 1]  # Use for testing
```

**For Testing/Validation** (all data points have true/spoofed pairs):

```bash
# Start attack immediately - every point has both true and spoofed values
python ml_data_collector.py --attack-delay 0 --duration 60 --label test_run01
```

**For Realistic Training** (random attack timing):

```bash
# Random attacks create unpredictable timing - better generalization
python ml_data_collector.py --random-attacks --duration 180 \
  --min-attack-duration 5 --max-attack-duration 20 \
  --min-clean-duration 5 --max-clean-duration 15 \
  --label realistic_run01
```

### ML Model Training and Testing

After collecting data, train machine learning models to detect GPS spoofing attacks.

**Two Training Approaches:**

| Approach                       | Script                           | Description                                                           |
| ------------------------------ | -------------------------------- | --------------------------------------------------------------------- |
| **Unsupervised (Recommended)** | `train_unsupervised_ensemble.py` | Pure unsupervised - no attack labels used for training or calibration |
| Supervised                     | `train_models.py`                | Original approach - uses validation attack labels for weighting       |

#### Unsupervised Ensemble

The unsupervised approach trains only on clean data and calibrates thresholds from clean score percentiles no attack labels are used.

```bash
# 1. Collect training data
cd sensor_fusion_testing
collect_ml_datasets.bat  # Select Option 2: One-Class Training

# 2. Collect validation data
collect_ml_datasets.bat  # Select Option 3: One-Class Validation

# 3. Train unsupervised ensemble (IsolationForest + PCA + LOF)
python train_unsupervised_ensemble.py --train-dir data/training --val-dir data/validation

# 4. Test with live detection in CARLA
python detect_spoofing_live.py --mode unsupervised --duration 120
```

**Unsupervised Training Output:**

```
trained_models_unsupervised/
|-- iforest.pkl           # IsolationForest with calibrated threshold
|-- pca.pkl               # PCA reconstruction error model
|-- lof.pkl               # Local Outlier Factor model
|-- unsup_ensemble.pkl    # Ensemble configuration (voting + smoothing)
|-- scaler.pkl            # Feature normalization
|-- config.json           # Training config and evaluation metrics
```

**Key Parameters:**

```bash
# Control false positive rate (default: 2%)
python train_unsupervised_ensemble.py --target-fpr 1.0  # Stricter (fewer FPs)
python train_unsupervised_ensemble.py --target-fpr 5.0  # More sensitive

# Temporal smoothing (reduces transient false positives)
python train_unsupervised_ensemble.py --smoothing-window 5 --smoothing-required 3
```

#### Supervised Ensemble (Original)

**Quick Start:**

```bash
# 1. Collect training data (25 runs, 120s each, clean baseline)
cd sensor_fusion_testing
collect_ml_datasets.bat  # Select Option 2: One-Class Training

# 2. Collect validation data (5 runs, 180s each, random attacks)
collect_ml_datasets.bat  # Select Option 3: One-Class Validation

# 3. Train all models (Isolation Forest, One-Class SVM, LOF, Elliptic Envelope)
python train_models.py --train-dir data/training --val-dir data/validation

# 4. Test with live detection in CARLA
python detect_spoofing_live.py --mode supervised --model-dir trained_models --duration 600
```

**Training Output:**

```
trained_models/
├── isolation_forest.pkl      # Tree-based isolation
├── one_class_svm.pkl         # Kernel-based boundary
├── lof.pkl                   # Local density comparison
├── elliptic_envelope.pkl     # Gaussian distribution
├── ensemble.pkl              # Combined voting system
└── scaler.pkl               # Feature normalization

results/
├── model_comparison.png         # Performance comparison
├── confusion_matrices.png       # TP/FP/TN/FN visualization
├── roc_curves.png              # ROC curves
├── precision_recall_curves.png  # PR curves
└── performance_report.json     # Detailed metrics
```

**Testing Options:**

```bash
# Option 1: Live real-time detection with unsupervised ensemble (recommended)
python detect_spoofing_live.py --mode unsupervised --duration 120

# Option 2: Live detection with supervised ensemble
python detect_spoofing_live.py --mode supervised --model-dir trained_models --duration 60

# Option 3: Evaluate on new test data
python ml_data_collector.py --attack-delay 0 --duration 60 --label new_test
# Then load models and evaluate in Python

# Option 4: Quick evaluation on existing data
python train_models.py --train-dir data/training --val-dir data/test --skip-save
```

**Live Detection Options:**

| Flag                  | Description                                    |
| --------------------- | ---------------------------------------------- |
| `--mode unsupervised` | Use unsupervised ensemble (default)            |
| `--mode supervised`   | Use original supervised ensemble               |
| `--duration N`        | Run for N seconds                              |
| `--attack-delay N`    | Seconds of clean baseline before attack        |
| `--no-smoothing`      | Disable temporal smoothing (unsupervised only) |
| `--no-display`        | Console output only (no pygame window)         |

**Model Selection Guide:**

| Use Case                | Recommended Model            | Why                                           |
| ----------------------- | ---------------------------- | --------------------------------------------- |
| Production deployment   | Ensemble (confidence voting) | Best balance: high F1, lowest false positives |
| Fast inference          | Isolation Forest             | Fastest predictions (~0.001s per sample)      |
| Best overall accuracy   | One-Class SVM                | Highest F1-score, configurable boundary       |
| Detecting gradual drift | LOF                          | Compares local density, catches slow changes  |
| Research/prototyping    | Train all, compare           | Understand strengths/weaknesses               |

**For detailed information, see:**

- `ML_TRAINING_GUIDE.md` - Complete training workflow and model details
- `ml_models/README.md` - Package documentation and API reference
- `AUTOMATION_GUIDE.md` - Data collection automation strategies

### Complete Pipeline Testing

```bash
# From project root, launch all visualizations and scene setup
python run_all.py

# In a separate terminal, run the attack testing
python sensor_fusion_testing/integration_files/sequential_attack_test.py

# For post-run analysis
python sensor_fusion_testing/integration_files/data_processor.py
```

### Navigation Spoofing Test (Waypoint + Fusion + Attacks)

Run a waypoint navigation demo that uses sensor fusion and applies GPS spoofing attacks. Select the attack mode and optionally override attack parameters from the CLI.

Modes:

- 1 = Gradual drift
- 2 = Sudden jump
- 3 = Random walk
- 4 = Replay
- 5 = Run all four in sequence

Basic usage:

```bash
python navigation_spoofing_test.py --mode 1 --duration 60 --speed 20
```

CLI overrides (examples):

```bash
# Gradual drift (mild vs aggressive)
python navigation_spoofing_test.py --mode 1 --drift-rate 0.05
python navigation_spoofing_test.py --mode 1 --drift-rate 0.25

# Innovation-aware drift
python navigation_spoofing_test.py --mode 1 \
  --adaptive-drift-rate 0.22 --drift-amp 0.08 --drift-freq 0.05

# Sudden jump (bigger, more frequent)
python navigation_spoofing_test.py --mode 2 --jump-magnitude 12 --jump-prob 0.05

# Random walk (noisier)
python navigation_spoofing_test.py --mode 3 --random-walk-step 1.0

# Replay with ~5 s delay
python navigation_spoofing_test.py --mode 4 --replay-delay 5

# Keep attacker’s awareness aligned with mitigation assumptions
python navigation_spoofing_test.py --mode 1 --innovation-threshold 5.0
```

Parameters:

- `--drift-rate` (m): amplitude of sinusoidal bias for basic gradual drift.
- `--adaptive-drift-rate` (m/s): base growth rate for innovation-aware drift.
- `--drift-amp` (m): oscillation amplitude added to innovation-aware drift.
- `--drift-freq` (Hz): oscillation frequency for innovation-aware drift.
- `--jump-magnitude` (m): size of sudden jumps.
- `--jump-prob` (0–1): probability of a jump each control step.
- `--random-walk-step` (m): max per-axis random-walk step size.
- `--innovation-threshold` (m): attacker’s awareness threshold used to modulate aggression.
- `--replay-delay` (s): approximate lag before replay starts; buffer gate assumes ~10 Hz GPS.

Notes:

- Gradual drift direction traces a small circle in the XY plane over time (world frame).
- Innovation-aware drift starts along +X in world coordinates and adds slow directional changes and noise to remain stealthy.
- Sudden jumps can be attenuated by a small Kalman gain, but frequent/large jumps still displace the fused state.
- Replay reuses earlier true positions after the delay, appearing locally smooth but wrong in absolute terms.

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

### ML Training Data (`data/`)

Generated by `ml_data_collector.py`:

- `ml_training_data_{timestamp}.csv`: Flat CSV with all features for sklearn/pandas
- `ml_training_data_{timestamp}.json`: Full JSON with metadata, statistics, and raw data

CSV columns include:

- Raw sensor data: `true_gps_x/y/z`, `spoofed_gps_x/y/z`, `imu_accel_x/y/z`, `imu_gyro_x/y/z`
- Kalman filter states: `kf_true_x/y/z`, `kf_spoof_x/y/z`, `innovation_true`, `innovation_spoof`
- Derived features: `position_error`, `kf_tracking_error`, `velocity_magnitude`
- Rolling statistics: `innovation_spoof_ma`, `position_error_std`, etc.
- Labels: `is_attack_active` (0 = clean, 1 = spoofed)

### Processed ML Data (`ml_data/`)

Generated by `data_processor.py`:

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
