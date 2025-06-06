# Description: This script shows all waypoints in the map and their corresponding GNSS locations.
# The waypoints are drawn as red points and the GNSS locations are drawn as green text at the waypoint locations.

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

# Get all waypoints in the map
waypoints = map.generate_waypoints(5.0)  # Change the distance between waypoints as needed

for waypoint in waypoints:
    # Get the GNSS location of the waypoint
    gnss_location = map.transform_to_geolocation(waypoint.transform.location)
    
    # Draw the waypoint location
    world.debug.draw_point(waypoint.transform.location, size=0.1, color=carla.Color(r=255, g=0, b=0), life_time=60.0)
    
    # Draw the GNSS location as text at the waypoint location
    gnss_text = f"Lat: {gnss_location.latitude:.6f}, Lon: {gnss_location.longitude:.6f}, Alt: {gnss_location.altitude:.2f}"
    world.debug.draw_string(waypoint.transform.location, gnss_text, draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=60.0)

print("Waypoints and GNSS locations have been drawn on the map.")
