# ML Models Package for GPS Spoofing Detection

This package contains machine learning models and utilities for detecting GPS spoofing attacks using one-class anomaly detection.

## Package Structure

```
ml_models/
├── __init__.py              - Package initialization
├── data_loader.py           - Data loading and preprocessing
├── one_class_trainer.py     - One-class classifier implementations
├── metrics.py               - Evaluation metrics
├── ensemble.py              - Ensemble voting system
└── visualizer.py            - Results visualization
```

## Quick Start

```python
from ml_models import DataLoader, create_all_models, train_all_models

# Load data
loader = DataLoader()
data = loader.load_and_prepare('data/training', 'data/validation')

# Train models
models = create_all_models()
models = train_all_models(models, data['X_train_clean'])

# Evaluate
from ml_models import MetricsCalculator
metrics = MetricsCalculator()
for name, model in models.items():
    metrics.evaluate_model(model, data['X_val'], data['y_val'], name)

metrics.print_summary()
```

## Modules

### data_loader.py

**Purpose**: Load and preprocess GPS/IMU data for ML training

**Key Classes**:

- `DataLoader` - Main data loading and preprocessing class

**Key Methods**:

- `load_dataset(directory)` - Load all CSVs from directory
- `prepare_one_class_data(df)` - Extract clean samples
- `get_feature_matrix(df)` - Extract and normalize features
- `load_and_prepare(train_dir, val_dir)` - Complete data pipeline

**Features Used**:

- innovation_spoof, innovation_spoof_ma/std/max
- position_error, kf_tracking_error
- accel_magnitude, gyro_magnitude, jerk_magnitude
- position_error_ma/std, kf_tracking_error_ma/std
- innovation_diff, kf_diff_magnitude

### one_class_trainer.py

**Purpose**: Implement one-class anomaly detection models

**Models**:

- `IsolationForestModel` - Tree-based isolation
- `OneClassSVMModel` - Kernel-based boundary
- `LOFModel` - Local density comparison
- `EllipticEnvelopeModel` - Gaussian distribution assumption

**Key Methods**:

- `fit(X_clean)` - Train on clean data only
- `predict(X)` - Predict labels (-1=anomaly, 1=normal)
- `score_samples(X)` - Get anomaly scores
- `save_model(path)` / `load_model(path)` - Persistence

### metrics.py

**Purpose**: Calculate evaluation metrics for anomaly detection

**Key Class**:

- `MetricsCalculator` - Comprehensive metrics calculation

**Metrics Calculated**:

- Precision, Recall, F1-Score
- ROC-AUC, ROC curve data
- Confusion matrix (TP/TN/FP/FN)
- Detection rate, false positive rate
- Time-to-detection after attack starts

### ensemble.py

**Purpose**: Combine multiple models for robust detection

**Voting Strategies**:

- `majority` - Simple majority vote (3/4 models)
- `weighted` - Weight by validation F1-scores
- `confidence` - Use anomaly scores with thresholds

**Key Class**:

- `EnsembleVoting` - Ensemble prediction system

**Key Functions**:

- `create_weighted_ensemble(models, metrics)` - Create optimal ensemble
- `evaluate_ensemble_strategies(...)` - Compare voting methods

### visualizer.py

**Purpose**: Generate performance visualizations

**Plots Created**:

- Model comparison bar charts
- Confusion matrices (heatmap)
- ROC curves
- Precision-Recall curves
- Feature importance (for tree models)

**Key Class**:

- `ResultsVisualizer` - Visualization generator

## Usage Examples

### Basic Training

```python
from ml_models.data_loader import DataLoader
from ml_models.one_class_trainer import IsolationForestModel

# Load data
loader = DataLoader()
data = loader.load_and_prepare('data/training')

# Train single model
model = IsolationForestModel()
model.fit(data['X_train_clean'])

# Evaluate
y_pred = model.predict(data['X_val'])
```

### Using Ensemble

```python
from ml_models.ensemble import EnsembleVoting
import joblib

# Load trained ensemble
ensemble = EnsembleVoting.load('trained_models/ensemble.pkl')
scaler = joblib.load('trained_models/scaler.pkl')

# Predict on new data
X_new_scaled = scaler.transform(X_new)
predictions = ensemble.predict(X_new_scaled)

# Get individual model predictions
individual = ensemble.get_individual_predictions(X_new_scaled)
print(f"Isolation Forest: {individual['isolation_forest']}")
print(f"One-Class SVM: {individual['one_class_svm']}")
```

### Custom Metrics

```python
from ml_models.metrics import MetricsCalculator

calc = MetricsCalculator()

# Evaluate model
metrics = calc.evaluate_model(
    model=my_model,
    X_val=X_test,
    y_val=y_test,
    model_name='My Model'
)

print(f"F1-Score: {metrics['f1_score']}")
print(f"AUC: {metrics['auc']}")
```

## Model Selection Guide

Choose based on your priorities:

| Priority                 | Recommended Model   |
| ------------------------ | ------------------- |
| Best overall performance | Ensemble (weighted) |
| Fastest training         | Elliptic Envelope   |
| Fastest inference        | Isolation Forest    |
| Best for gradual drift   | LOF                 |
| Most configurable        | One-Class SVM       |
| Production deployment    | Ensemble            |

## Hyperparameter Tuning

### Isolation Forest

```python
model = IsolationForestModel(
    n_estimators=200,      # More trees = better, slower
    contamination=0.1,     # Expected anomaly rate in test
    max_features=0.8       # Features per tree
)
```

### One-Class SVM

```python
model = OneClassSVMModel(
    kernel='rbf',          # 'rbf', 'linear', 'poly'
    gamma='scale',         # Kernel coefficient
    nu=0.1                 # Training error bound
)
```

### LOF

```python
model = LOFModel(
    n_neighbors=20,        # More = smoother boundary
    contamination=0.1      # Expected anomaly rate
)
```

## Integration with Existing Research

This ML system integrates with your sensor fusion research:

1. **Data Collection**: Uses `ml_data_collector.py` with dual GPS sensors
2. **Feature Engineering**: Uses Kalman filter innovations and IMU data
3. **Attack Testing**: Trained on INNOVATION_AWARE_GRADUAL_DRIFT attacks
4. **Research Goal**: Demonstrate that ML is necessary beyond Kalman filtering

## Performance Optimization

### For Large Datasets

```python
# Use joblib for parallel processing
from ml_models.one_class_trainer import IsolationForestModel

model = IsolationForestModel(n_estimators=100)
model.model.n_jobs = -1  # Use all CPU cores
```

### For Real-Time Detection

Use Isolation Forest (fastest inference):

- Training: ~1 second per 10k samples
- Prediction: ~0.001 seconds per sample
- Good for streaming data

## References

See also:

- `ML_TRAINING_GUIDE.md` - Complete training workflow
- `AUTOMATION_GUIDE.md` - Data collection automation
- `README.md` - General sensor fusion testing overview
