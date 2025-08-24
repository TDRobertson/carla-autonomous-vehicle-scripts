#!/bin/bash
# Launch all three scripts in new terminals with the virtual environment activated and window placement

# Top left
(gnome-terminal --geometry=80x24+0+0 -- bash -c "echo '=== Running scene.py ==='; source venv/bin/activate && python sensor_fusion_testing/scene.py; exec bash") &
# Top right
(gnome-terminal --geometry=80x24+960+0 -- bash -c "echo '=== Running fpv_ghost.py ==='; source venv/bin/activate && python sensor_fusion_testing/fpv_ghost.py; exec bash") &
# Bottom left
(gnome-terminal --geometry=80x24+0+540 -- bash -c "echo '=== Running sync.py ==='; source venv/bin/activate && python sensor_fusion_testing/sync.py; exec bash") & 