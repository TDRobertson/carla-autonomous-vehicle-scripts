# GPS Spoofing Test Launchers

This directory contains launcher scripts that handle import paths correctly when running from the root directory.

## Available Launchers

### 1. Position Display Demo

```bash
python sensor_fusion_testing/run_position_demo.py
```

- Shows real-time position visualization
- Tests both GPS-only and sensor fusion systems
- Perfect for demonstrations and presentations

### 2. GPS-Only Test

```bash
python sensor_fusion_testing/run_gps_only_test.py
```

- Tests GPS-only system with all spoofing strategies
- Shows raw effects of GPS spoofing without sensor fusion
- Includes position visualization

### 3. Comparison Test

```bash
python sensor_fusion_testing/run_comparison_test.py
```

- Direct comparison between GPS-only and sensor fusion
- Tests all spoofing strategies on both systems
- Shows improvement percentages
- Saves results to JSON file

## What You'll See

### Visual Elements (in CARLA window):

- 🟢 **Green sphere**: True vehicle position
- 🔴 **Red sphere**: Sensor-estimated position
- 🟡 **Yellow line**: Position error between true and sensor positions
- **Floating text**: Coordinates and error information
- **Color-coded errors**: Green (low), Yellow (medium), Red (high)

### Console Output:

```
TRUE: [123.45, 67.89, 1.23] | SENSOR: [125.67, 69.12, 1.25] | ERROR: 2.34m | ATTACK: GRADUAL_DRIFT | INNOVATION: 1.23
```

## Prerequisites

1. **CARLA Simulator**: Make sure CarlaUE4.exe is running
2. **Python Environment**: Activate your virtual environment
3. **Dependencies**: All required packages should be installed

## Usage Tips

1. **Start CARLA first**: Launch CarlaUE4.exe before running any tests
2. **Watch the CARLA window**: The visual elements appear in the 3D world
3. **Monitor console output**: Detailed position data is printed to console
4. **Compare results**: Use the comparison test to see sensor fusion benefits

## Troubleshooting

### Import Errors

If you get import errors, make sure you're running from the root directory:

```bash
# Correct - run from root directory
python sensor_fusion_testing/run_position_demo.py

# Incorrect - don't run from sensor_fusion_testing directory
cd sensor_fusion_testing
python run_position_demo.py  # This will cause import errors
```

### CARLA Connection Issues

- Make sure CarlaUE4.exe is running
- Check that CARLA is listening on localhost:2000
- Wait a few seconds after starting CARLA before running tests

### No Visual Elements

- Make sure you're looking at the CARLA 3D window
- The spheres and lines appear in the world, not on the UI
- Try moving the camera view to see the vehicle

## Expected Results

### GPS-Only System

- Large separation between green and red spheres during attacks
- High position errors in console output
- Shows vulnerability to GPS spoofing

### Sensor Fusion System

- Smaller separation between green and red spheres
- Lower position errors in console output
- Shows some protection against GPS spoofing

### Comparison

- Clear improvement percentages
- Evidence that sensor fusion provides better protection
- Data saved for further analysis
