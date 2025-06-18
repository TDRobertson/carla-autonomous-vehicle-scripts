@echo off
start cmd /k "venv\Scripts\activate && python sensor_fusion_testing\scene.py"
start cmd /k "venv\Scripts\activate && python sensor_fusion_testing\fpv_ghost.py"
start cmd /k "venv\Scripts\activate && python sensor_fusion_testing\sync.py" 