# Development Plan: Advanced Kalman Filter Integration for GPS Spoofing Pipeline

## Research Goal

- Test the limits of Kalman filter-based GPS+IMU sensor fusion against 4 spoofing attacks (Gradual Drift, Sudden Jump, Random Walk, Replay).
- Evaluate if sensor fusion alone (no explicit spoofing detection/correction) can mitigate these attacks.
- Use CARLA's simulation, with new scripts for vehicle control and visualization.

---

## Integration Steps

### 1. Refactor Advanced Kalman Filter

- Extract the advanced Kalman filter logic from `sync.py` into a new reusable module (e.g., `advanced_kalman_filter.py`).
- Ensure this module is importable and can be used by both `sync.py` and `sensor_fusion.py`.
- Rationale: Guarantees that all sensor fusion and attack tests use the most advanced filter implementation.

### 2. Update Sensor Fusion

- Update `sensor_fusion.py` to use the new advanced Kalman filter module for GPS+IMU fusion.
- Ensure the `SensorFusion` class can accept GPS data (optionally spoofed) and IMU data.
- Rationale: Centralizes sensor fusion logic and ensures consistency across all tests.

### 3. Connect the Spoofer

- Ensure the `SensorFusion` class in `sensor_fusion.py` can accept GPS data that may be spoofed by the spoofer module.
- Confirm that the attack type can be switched at runtime.
- Rationale: Allows for flexible testing of all spoofing strategies.

### 4. Update Sequential Attack Test

- Update `sequential_attack_test.py` to use the updated `SensorFusion` class with the advanced Kalman filter.
- Ensure all four attacks are run in sequence and results are logged.
- Rationale: Provides a unified test harness for all spoofing scenarios.

### 5. Visualization Automation

- Add hooks in `sequential_attack_test.py` to automatically launch the visualizations from `sync.py` and `fpv_ghost.py`.
- If necessary, refactor the visualization components from `sync.py` into a new reusable module (e.g., `fusion_visualization.py`).
- Rationale: Enables real-time monitoring of vehicle state and sensor fusion performance during tests.

### 6. Data Collection & Analysis

- Use the existing data collection in `sequential_attack_test.py` and `data_processor.py` for post-run analysis.
- Focus on error metrics, position/velocity drift, and Kalman filter performance under attack.

---

## Checklist

- [ ] Advanced Kalman filter refactored into a reusable module.
- [ ] `sensor_fusion.py` updated to use the new filter and accept spoofed GPS data.
- [ ] `sequential_attack_test.py` uses the updated SensorFusion and runs all attacks.
- [ ] Visualization scripts are launched automatically during tests.
- [ ] Data collection and analysis scripts are ready for post-experiment review.

---

## Next Steps

1. Refactor the advanced Kalman filter from `sync.py` into a new module.
2. Update `sensor_fusion.py` to use the new filter and accept spoofed GPS data.
3. Update `sequential_attack_test.py` to use the new SensorFusion and automate visualization.
4. Test the full pipeline and analyze results.
