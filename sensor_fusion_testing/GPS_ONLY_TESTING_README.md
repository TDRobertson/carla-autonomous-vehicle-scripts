# GPS-Only Testing System

This directory contains modified files that allow testing GPS spoofing attacks on a GPS-only positioning system without IMU or Kalman filter correction. This enables comparison between GPS-only and sensor fusion systems to demonstrate the benefits of sensor fusion for GPS spoofing mitigation.

## Files Created

### Core GPS-Only System

- `integration_files/gps_only_system.py` - GPS-only positioning system without sensor fusion
- `integration_files/gps_only_sequential_test.py` - Sequential attack testing for GPS-only system
- `integration_files/gps_only_waypoint_navigator.py` - GPS-only waypoint navigation system

### Test Scripts

- `gps_only_test.py` - Simple test script for GPS-only system
- `compare_gps_vs_fusion.py` - Comprehensive comparison between GPS-only and sensor fusion systems

## Key Differences from Sensor Fusion System

### GPS-Only System (`gps_only_system.py`)

- **No IMU sensor**: Only uses GPS for positioning
- **No Kalman filter**: Direct GPS position as "fused" position
- **No innovation tracking**: No spoofing detection capabilities
- **Simple velocity estimation**: Calculated from GPS position differences
- **Raw GPS effects**: Shows unmitigated effects of GPS spoofing

### Sensor Fusion System (existing)

- **IMU + GPS**: Uses both sensors for positioning
- **Kalman filter**: Fuses sensor data with error correction
- **Innovation tracking**: Monitors for spoofing detection
- **Advanced velocity estimation**: Uses IMU data for better velocity
- **Spoofing mitigation**: Attempts to detect and mitigate attacks

## Usage

### 1. Test GPS-Only System

```bash
# Test individual GPS-only system
python sensor_fusion_testing/gps_only_test.py

# Test GPS-only sequential attacks
python sensor_fusion_testing/integration_files/gps_only_sequential_test.py
```

### 2. Compare GPS-Only vs Sensor Fusion

```bash
# Run comprehensive comparison
python sensor_fusion_testing/compare_gps_vs_fusion.py
```

### 3. GPS-Only Waypoint Navigation

```python
from integration_files.gps_only_waypoint_navigator import GPSOnlyWaypointNavigator
from integration_files.waypoint_generator import WaypointGenerator

# Create GPS-only navigator
navigator = GPSOnlyWaypointNavigator(vehicle, enable_spoofing=True)

# Generate waypoints
generator = WaypointGenerator(world)
waypoints = generator.generate_random_route()

# Set waypoints and start navigation
navigator.set_waypoints(waypoints)
navigator.start_navigation()
```

## Expected Results

### GPS-Only System

- **High position errors**: GPS spoofing attacks will cause significant positioning errors
- **No mitigation**: Attacks will be successful without sensor fusion protection
- **Raw attack effects**: Shows the true impact of GPS spoofing on navigation

### Sensor Fusion System

- **Lower position errors**: Kalman filter provides some protection against attacks
- **Partial mitigation**: Some attacks may be detected and mitigated
- **Better resilience**: Shows the benefits of sensor fusion for security

## Attack Strategies Tested

1. **Gradual Drift**: Slowly drifting GPS position over time
2. **Sudden Jump**: Abrupt changes in GPS position
3. **Random Walk**: Random GPS position variations
4. **Replay**: Replaying previous GPS positions

## Output Files

- `gps_only_test_results/` - Results from GPS-only sequential testing
- `gps_vs_fusion_comparison.json` - Comparison results between systems

## Integration with Existing Research

These GPS-only testing files integrate seamlessly with your existing sensor fusion research:

- Uses the same GPS spoofing strategies (`gps_spoofer.py`)
- Compatible with existing visualization systems
- Maintains the same data collection format
- Enables direct comparison with sensor fusion results

## Research Applications

1. **Baseline Comparison**: Establish GPS-only performance as baseline
2. **Sensor Fusion Benefits**: Demonstrate the value of sensor fusion
3. **Attack Effectiveness**: Show which attacks are most effective
4. **Mitigation Analysis**: Analyze which attacks can be mitigated
5. **Navigation Impact**: Study the effect on autonomous navigation

## Notes

- The GPS-only system shows the raw, unmitigated effects of GPS spoofing
- This provides a baseline for comparing sensor fusion effectiveness
- Results can be used to justify the need for advanced spoofing detection
- The system maintains the same interface as the sensor fusion system for easy comparison
