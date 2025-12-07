# ML Data Collection Automation Guide

This guide explains the automated data collection system for training machine learning models to detect GPS spoofing attacks.

## Quick Start

### Windows
```cmd
cd sensor_fusion_testing
collect_ml_datasets.bat
```

### Linux/Mac
```bash
cd sensor_fusion_testing
chmod +x collect_ml_datasets.sh
./collect_ml_datasets.sh
```

### Cross-Platform (Python)
```bash
cd sensor_fusion_testing
python automated_data_collection.py
```

## What Was Implemented

### 1. Enhanced ML Data Collector (`ml_data_collector.py`)

**New Feature: `--label` Parameter**

Allows custom labeling of output files for organized dataset management.

**Usage:**
```bash
python ml_data_collector.py --duration 120 --attack-delay 30 --label train_run01
```

**Output:**
- `train_run01_ml_training_data_20251207_120001.csv`
- `train_run01_ml_training_data_20251207_120001.json`

Without `--label`, files are named: `ml_training_data_{timestamp}.csv`

### 2. Windows Batch Script (`collect_ml_datasets.bat`)

**Features:**
- Menu-driven interface
- 5 preset collection modes
- Custom parameter option
- Automatic error handling
- Progress reporting

**Preset Modes:**
1. **Quick Test** - 5 runs × 60s (testing/validation)
2. **One-Class Training** - 25 runs × 120s (~1 hour)
3. **One-Class Validation** - 5 runs × 180s with random attacks
4. **Supervised Training** - 20 runs × 120s (~45 minutes)
5. **Supervised Validation** - 10 runs × 150s with random attacks

**File Organization:**
```
data/
├── training/        (train_run01...train_run25)
├── validation/      (val_run01...val_run05)
├── test/            (test_run01...test_run05)
├── supervised_training/      (sup_train_run01...sup_train_run20)
└── supervised_validation/    (sup_val_run01...sup_val_run10)
```

### 3. Linux/Mac Shell Script (`collect_ml_datasets.sh`)

Same functionality as Windows batch script with:
- Colored terminal output
- Unix-style progress indicators
- Error handling and retry logic
- Bash-compatible syntax

**Permissions:**
```bash
chmod +x collect_ml_datasets.sh
```

### 4. Python Collection Manager (`automated_data_collection.py`)

**Advanced Features:**
- Cross-platform compatibility
- Progress bars (requires `tqdm`)
- Automatic CARLA connection checking
- Retry logic with configurable attempts
- Dataset statistics after collection
- Non-interactive mode for scripting

**Interactive Mode:**
```bash
python automated_data_collection.py
```

**Non-Interactive Mode:**
```bash
# Run specific preset
python automated_data_collection.py --preset quick_test
python automated_data_collection.py --preset one_class_training

# Custom retry settings
python automated_data_collection.py --preset one_class_training --max-retries 5
```

**Requirements:**
- Python 3.7+
- Optional: `tqdm` for progress bars (`pip install tqdm`)
- Optional: `pandas` for statistics (`pip install pandas`)

## File Naming Convention

All automation scripts follow this naming pattern:

```
{label}_ml_training_data_{timestamp}.{ext}
```

**Examples:**
- `train_run01_ml_training_data_20251207_120001.csv`
- `val_run05_ml_training_data_20251207_130045.json`
- `test_run03_ml_training_data_20251207_140020.csv`

**Label Prefixes:**
- `train_run` - Training data
- `val_run` - Validation data
- `test_run` - Test data
- `sup_train_run` - Supervised training data
- `sup_val_run` - Supervised validation data

## Collection Strategies

### For One-Class Classifiers

**Training Set:**
- Use: **One-Class Training** preset
- Runs: 25 × 120s
- Attack delay: 30s (clean baseline)
- Goal: 70-80% clean, 20-30% attack data

**Validation Set:**
- Use: **One-Class Validation** preset
- Runs: 5 × 180s
- Random attacks enabled
- Goal: Test generalization

### For Supervised Learning

**Training Set:**
- Use: **Supervised Training** preset
- Runs: 20 × 120s
- Attack delay: 0s (immediate)
- Goal: Maximum labeled pairs

**Validation Set:**
- Use: **Supervised Validation** preset
- Runs: 10 × 150s
- Random attacks + immediate start
- Goal: Timing diversity

## Time Estimates

| Preset | Duration | Estimated Time |
|--------|----------|----------------|
| Quick Test | 5 × 60s | ~10 minutes |
| One-Class Training | 25 × 120s | ~1 hour |
| One-Class Validation | 5 × 180s | ~20 minutes |
| Supervised Training | 20 × 120s | ~45 minutes |
| Supervised Validation | 10 × 150s | ~30 minutes |

**Total for Complete Dataset:** ~2-3 hours

## Error Handling

All scripts include:
- Automatic retry on collection failures
- Error logging
- Graceful exit on user cancellation
- Progress preservation (can resume interrupted collections)

**Python Manager:**
- Configurable retry attempts (`--max-retries`)
- CARLA connection health checks
- Detailed error reporting

## Dataset Statistics

After collection, the Python manager automatically displays:
- Total files collected
- Total samples
- Clean vs attack sample distribution
- Percentage breakdown

**Manual Statistics:**
```python
import pandas as pd
import glob

files = glob.glob('data/training/*.csv')
dfs = [pd.read_csv(f) for f in files]
combined = pd.concat(dfs, ignore_index=True)

print(f"Total samples: {len(combined)}")
print(f"Clean: {len(combined[combined.is_attack_active==0])}")
print(f"Attack: {len(combined[combined.is_attack_active==1])}")
```

## Troubleshooting

**CARLA Not Connecting:**
- Ensure `CarlaUE4.exe` is running
- Check port 2000 is accessible
- Python manager will warn about connection issues

**Collection Failures:**
- Scripts automatically retry (3 attempts by default)
- Check CARLA logs for errors
- Verify sufficient disk space

**File Not Found:**
- Ensure scripts are run from `sensor_fusion_testing/` directory
- Check Python is in PATH
- Verify `ml_data_collector.py` exists

**Progress Bars Not Showing (Python):**
- Install tqdm: `pip install tqdm`
- Scripts work without it, just no progress bars

## Best Practices

1. **Run collections overnight** for large datasets
2. **Monitor first few runs** to ensure data quality
3. **Keep metadata JSON files** for experiment tracking
4. **Version control** your data collection configurations
5. **Separate directories** for different experiments
6. **Back up data** regularly during long collections

## Advanced Usage

**Custom Batch Collection (Windows):**
```cmd
REM Run multiple presets in sequence
collect_ml_datasets.bat
REM Select option 2 (One-Class Training)
REM After completion, run again and select option 3 (One-Class Validation)
```

**Custom Script Integration (Python):**
```python
from automated_data_collection import DataCollectionManager, CollectionConfig

manager = DataCollectionManager(max_retries=5)

custom_config = CollectionConfig(
    name="My Experiment",
    num_runs=10,
    duration=90,
    attack_delay=15,
    output_dir="data/my_experiment",
    label_prefix="exp01_run",
    random_attacks=False,
    description="Custom experimental setup"
)

manager.collect_dataset(custom_config)
```

## Summary

The automated data collection system provides:
- **3 script options** (Windows .bat, Linux .sh, Python .py)
- **5 preset modes** for common scenarios
- **Organized file naming** with labels
- **Error handling and retries**
- **Progress tracking**
- **Dataset statistics**

Choose the script that best fits your platform and workflow. All produce identical, properly formatted datasets for ML training.

