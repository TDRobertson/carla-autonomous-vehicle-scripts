# Description: This script shows the waypoints on the map with their XYZ coordinates.
# This was used to find starting points for the vehicles in the simulation.

import glob
import os
import sys
import carla

# Add the CARLA Python API to PYTHONPATH
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# Ensure the CARLA Python API path is correctly added
carla_path = 'C:/CARLA_0.9.15/PythonAPI/carla'
if carla_path not in sys.path:
    sys.path.append(carla_path)

import carla

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
map = world.get_map()
distance = 5

waypoints = map.generate_waypoints(distance)
for w in waypoints:
    location = w.transform.location
    coordinates_text = f"X: {location.x:.2f}, Y: {location.y:.2f}, Z: {location.z:.2f}"
    world.debug.draw_string(location, coordinates_text, draw_shadow=False,
                            color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                            persistent_lines=True)

print("Waypoints with XYZ coordinates have been drawn on the map.")
