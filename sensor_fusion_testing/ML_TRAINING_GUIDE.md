# ML Training Guide - GPS Spoofing Detection

Complete guide for training and using ML models to detect GPS spoofing attacks.

## Overview

The ML system uses **anomaly detection** trained on clean (non-spoofed) data to identify GPS spoofing attacks. Two approaches are available:

1. **Supervised Ensemble** (original): Four one-class classifiers with validation-based weighting
2. **Unsupervised Ensemble** (new): Pure unsupervised approach with threshold calibration from clean data only

The **unsupervised approach** is recommended for research because:
- No attack labels are used for training or calibration
- More realistic threat model (can detect unknown attacks)
- Thresholds derived purely from clean data statistics

## Prerequisites

```bash
# Install required packages
pip install numpy pandas scikit-learn joblib matplotlib seaborn
```

## Step-by-Step Workflow

### Step 1: Collect Training Data

First, collect data using the automated collection system:

```cmd
cd sensor_fusion_testing
collect_ml_datasets.bat
```

Select **Option 2: One-Class Training** (25 runs × 120s)

This creates:
- `data/training/` with 25 CSV files
- Each file has ~1,200 samples
- Total: ~30,000 samples

### Step 2: Collect Validation Data

```cmd
collect_ml_datasets.bat
```

Select **Option 3: One-Class Validation** (5 runs × 180s, random attacks)

This creates:
- `data/validation/` with 5 CSV files
- Random attack timing for realistic testing

### Step 3: Train ML Models

```bash
python train_models.py --train-dir data/training --val-dir data/validation
```

This will:
1. Load and preprocess data (clean for training, clean+attack for validation)
2. Train 4 one-class classifiers:
   - Isolation Forest
   - One-Class SVM
   - Local Outlier Factor (LOF)
   - Elliptic Envelope
3. Evaluate each model
4. Create weighted ensemble
5. Save trained models to `trained_models/`
6. Save results to `results/`
7. Generate visualizations

**Expected Output:**
```
trained_models/
├── isolation_forest.pkl
├── one_class_svm.pkl
├── lof.pkl
├── elliptic_envelope.pkl
├── ensemble.pkl
└── scaler.pkl

results/
├── model_comparison.png
├── confusion_matrices.png
├── roc_curves.png
├── precision_recall_curves.png
└── performance_report.json
```

### Step 4: Review Results

Check the visualizations in `results/`:

1. **model_comparison.png** - Compare Precision, Recall, F1, AUC across models
2. **confusion_matrices.png** - See TP/FP/TN/FN for each model
3. **roc_curves.png** - ROC curves showing detection vs false positive tradeoff
4. **performance_report.json** - Detailed metrics in JSON format

### Step 5: Test Live Detection (Optional)

```bash
python detect_spoofing_live.py --model-dir trained_models --duration 60
```

Demonstrates real-time detection using trained models in CARLA.

## Understanding the Models

### Isolation Forest
- **How it works**: Isolates anomalies using random decision trees
- **Strengths**: Fast, handles high dimensions, no assumptions about data distribution
- **Best for**: General-purpose anomaly detection

### One-Class SVM
- **How it works**: Learns a boundary around normal data using kernel tricks
- **Strengths**: Good with complex boundaries, mathematically principled
- **Best for**: When you need a clear decision boundary

### Local Outlier Factor (LOF)
- **How it works**: Compares local density of each point to neighbors
- **Strengths**: Detects local anomalies, good for gradual drift
- **Best for**: Density-based anomalies

### Elliptic Envelope
- **How it works**: Assumes Gaussian distribution, fits ellipse around data
- **Strengths**: Very fast inference, robust to outliers in training
- **Best for**: When features are roughly Gaussian

### Ensemble (Weighted Voting)
- **How it works**: Combines all models, weighted by validation F1-scores
- **Strengths**: Reduces false positives, more robust than any single model
- **Best for**: Production deployment

## Feature Importance

The models use these key features:

| Feature | Why It's Important |
|---------|-------------------|
| `innovation_spoof` | Kalman filter residual - spikes during attacks |
| `innovation_spoof_ma` | Rolling average - detects sustained deviations |
| `innovation_spoof_std` | Rolling std dev - detects instability |
| `position_error` | Direct measure of GPS drift |
| `kf_tracking_error` | Kalman filter's tracking performance |
| `accel_magnitude` | IMU confirms vehicle motion |
| `jerk_magnitude` | Sudden changes in acceleration |

## Expected Performance

Based on your research findings:

| Attack Type | Expected Detection Rate | Notes |
|-------------|------------------------|-------|
| Gradual Drift | 80-95% | High effectiveness |
| Sudden Jump | 85-98% | Easy to detect (large innovations) |
| Random Walk | 40-60% | Already mitigated by Kalman filter |
| Replay | 30-50% | Locally smooth, hard to detect |

**False Positive Rate Target**: < 5%

## Interpreting Results

### Good Performance
- F1-Score > 0.80
- AUC > 0.90
- False Positive Rate < 0.05
- Time-to-Detection < 5 seconds

### Poor Performance (Need More Data)
- F1-Score < 0.60
- AUC < 0.75
- High false positives (> 10%)

If performance is poor:
1. Collect more training data (50+ runs)
2. Try different feature combinations
3. Adjust model hyperparameters
4. Check data quality (are attacks actually happening?)

## Advanced Usage

### Custom Feature Selection

```python
from ml_models.data_loader import DataLoader

# Use only specific features
custom_features = [
    'innovation_spoof',
    'innovation_spoof_ma',
    'position_error',
    'kf_tracking_error'
]

loader = DataLoader(feature_list=custom_features)
data = loader.load_and_prepare('data/training', 'data/validation')
```

### Custom Model Configuration

```python
from ml_models.one_class_trainer import create_all_models

config = {
    'if_n_estimators': 200,  # More trees
    'contamination': 0.15,    # Expect 15% anomalies
    'svm_kernel': 'poly',     # Polynomial kernel
    'lof_n_neighbors': 30     # More neighbors
}

models = create_all_models(config)
```

### Load and Use Trained Models

```python
import joblib
from ml_models.ensemble import EnsembleVoting

# Load ensemble
ensemble = EnsembleVoting.load('trained_models/ensemble.pkl')

# Load scaler
scaler = joblib.load('trained_models/scaler.pkl')

# Predict on new data
X_new = scaler.transform(new_features)
predictions = ensemble.predict(X_new)

# -1 = attack detected, 1 = normal
is_attack = predictions == -1
```

## Troubleshooting

**ImportError: No module named 'ml_models'**
- Make sure you're in the `sensor_fusion_testing/` directory
- Or add to PYTHONPATH: `export PYTHONPATH=$PYTHONPATH:$(pwd)`

**ValueError: No CSV files found**
- Make sure you've collected data first using `collect_ml_datasets.bat`
- Check the directory paths in the training command

**Poor model performance**
- Verify spoofing is actually happening (check `position_error` column in CSVs)
- Collect more training data (50-100 runs)
- Try different feature combinations
- Check for data quality issues

**Models train but predict everything as normal**
- Check if attack samples exist in validation data
- Verify `is_attack_active` column has both 0 and 1 values
- Adjust `contamination` parameter (increase if needed)

## Next Steps After Training

1. **Review Performance**: Check visualizations and metrics
2. **Iterate if Needed**: Collect more data or tune hyperparameters
3. **Deploy**: Integrate models into your sensor fusion pipeline
4. **Monitor**: Track false positive rates in production

## Research Integration

These ML models complement your existing research:
- **Kalman Filter**: Good at mitigating random walk and replay attacks
- **ML Models**: Good at detecting gradual drift and sudden jump attacks
- **Combined System**: Provides defense-in-depth against all attack types

The goal is to demonstrate that **Kalman filter + ML detection** provides more robust security than Kalman filter alone.

---

# Unsupervised Ensemble Training (Recommended)

The unsupervised approach trains models using **only clean data** and calibrates thresholds from clean score percentiles. No attack labels are used for training or threshold selection.

## Why Unsupervised?

| Aspect | Supervised | Unsupervised |
|--------|------------|--------------|
| Training data | Clean only | Clean only |
| Threshold calibration | Uses attack labels | Uses clean percentiles |
| Unknown attacks | May miss novel attacks | Detects any anomaly |
| False positive control | Via contamination param | Via target FPR percentile |
| Research validity | Requires known attacks | Stronger threat model |

## Unsupervised Models

The unsupervised ensemble uses three complementary models:

### IsolationForest
- **Scoring**: Tree-based isolation depth
- **Threshold**: Set at Nth percentile of clean scores
- **Best for**: High-dimensional data, fast training

### PCA Reconstruction Error
- **Scoring**: Reconstruction MSE after projection
- **Threshold**: Set at Nth percentile of clean errors
- **Best for**: Detecting deviations from normal patterns

### Local Outlier Factor (LOF)
- **Scoring**: Local density comparison
- **Threshold**: Set at Nth percentile of clean scores
- **Best for**: Gradual drift detection, local anomalies

## Training the Unsupervised Ensemble

### Step 1: Collect Clean Training Data

Use the same data collection as before:

```cmd
cd sensor_fusion_testing
collect_ml_datasets.bat
```

Select **Option 2: One-Class Training** (25 runs x 120s)

This creates `data/training/` with clean and attack samples. The unsupervised trainer will extract only clean samples.

### Step 2: Collect Validation Data

```cmd
collect_ml_datasets.bat
```

Select **Option 3: One-Class Validation** (5 runs x 180s, random attacks)

### Step 3: Train Unsupervised Ensemble

```bash
python train_unsupervised_ensemble.py --train-dir data/training --val-dir data/validation
```

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--target-fpr` | 2.0 | Target false positive rate as percentile (2.0 = 2% FPR) |
| `--calibration-split` | 0.2 | Fraction of clean data held out for threshold calibration |
| `--voting-threshold` | 0.5 | Fraction of models required for anomaly vote (0.5 = majority) |
| `--smoothing-window` | 3 | Window size for temporal smoothing |
| `--smoothing-required` | 2 | Required detections in window to trigger alert |

**Example with custom FPR:**

```bash
python train_unsupervised_ensemble.py \
    --train-dir data/training \
    --val-dir data/validation \
    --target-fpr 1.0 \
    --output-dir trained_models_unsupervised
```

### Step 4: Review Output

```
trained_models_unsupervised/
|-- iforest.pkl           # IsolationForest model + threshold
|-- pca.pkl               # PCA model + threshold
|-- lof.pkl               # LOF model + threshold
|-- unsup_ensemble.pkl    # Ensemble configuration
|-- scaler.pkl            # Feature scaler
|-- config.json           # Training configuration and metrics
```

## Live Testing with Unsupervised Ensemble

### Basic Test

```bash
python detect_spoofing_live.py --mode unsupervised --duration 120
```

This will:
1. Load models from `trained_models_unsupervised/`
2. Run clean baseline for 10 seconds
3. Apply INNOVATION_AWARE_GRADUAL_DRIFT attack
4. Run detection every 1 second
5. Report TP/FP/TN/FN counts

### Advanced Options

```bash
python detect_spoofing_live.py \
    --mode unsupervised \
    --model-dir trained_models_unsupervised \
    --duration 300 \
    --attack-delay 30 \
    --no-smoothing
```

| Flag | Description |
|------|-------------|
| `--mode unsupervised` | Use unsupervised ensemble (default) |
| `--mode supervised` | Use original supervised ensemble |
| `--no-smoothing` | Disable temporal smoothing |
| `--no-display` | Console output only (no pygame) |
| `--attack-delay N` | Seconds of clean baseline before attack |

## Understanding Threshold Calibration

The key innovation of the unsupervised approach is threshold calibration:

```
1. Train models on clean data only
2. Score held-out clean samples
3. Set threshold at target percentile (e.g., 2nd percentile)
4. Samples with scores below threshold = anomaly
```

**Example:**

If `--target-fpr 2.0`:
- Score all clean calibration samples
- Find the 2nd percentile of scores
- Set this as the threshold
- Expected FPR on clean data: ~2%

## Ensemble Voting

The unsupervised ensemble uses **majority voting**:

```
For each sample:
  1. Get score from each model
  2. Compare score to model's threshold
  3. If score < threshold, model votes "anomaly"
  4. If 2+ models vote anomaly, ensemble predicts anomaly
```

**Temporal Smoothing** (optional):
- Tracks last N detection decisions
- Only triggers alert if M+ anomalies in window
- Reduces false positives from transient noise

## Comparing Supervised vs Unsupervised

Run both and compare:

```bash
# Supervised (original)
python detect_spoofing_live.py --mode supervised --duration 120

# Unsupervised (new)
python detect_spoofing_live.py --mode unsupervised --duration 120
```

Expected differences:
- Unsupervised may have slightly higher FPR (no attack label tuning)
- Unsupervised may detect attacks faster (no contamination parameter)
- Unsupervised is more scientifically defensible for research

## Programmatic Usage

```python
import joblib
from ml_models.unsupervised_ensemble import UnsupervisedEnsemble

# Load ensemble
ensemble = UnsupervisedEnsemble.load('trained_models_unsupervised/unsup_ensemble.pkl')
scaler = joblib.load('trained_models_unsupervised/scaler.pkl')

# Scale features
X_scaled = scaler.transform(new_features)

# Batch prediction (no smoothing)
is_anomaly = ensemble.predict(X_scaled)  # Boolean array

# Single-sample with smoothing (for real-time)
is_anomaly, info = ensemble.predict_single(X_scaled[0], use_smoothing=True)

# Info contains:
# - info['scores']: per-model scores
# - info['votes']: per-model votes
# - info['thresholds']: per-model thresholds
# - info['final_detection']: ensemble decision
```

## Tuning Recommendations

### For Lower False Positives
- Decrease `--target-fpr` (e.g., 1.0 instead of 2.0)
- Increase `--smoothing-required` (e.g., 3 instead of 2)
- Increase `--smoothing-window` (e.g., 5 instead of 3)

### For Higher Detection Rate
- Increase `--target-fpr` (e.g., 5.0 instead of 2.0)
- Decrease `--voting-threshold` (e.g., 0.33 for any-model-votes)
- Disable smoothing (`--no-smoothing`)

### For Research Papers
- Use `--target-fpr 2.0` (standard 2% FPR target)
- Use majority voting (`--voting-threshold 0.5`)
- Report both supervised and unsupervised results
- Emphasize that unsupervised uses no attack labels

