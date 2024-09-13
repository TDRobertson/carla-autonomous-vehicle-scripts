import glob
import os
import sys
import math
import time
import numpy as np
import open3d as o3d
from matplotlib import cm

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
from agents.navigation.global_route_planner import GlobalRoutePlanner

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
town_map = world.get_map()  # Add this line to get the map
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.*')[0]
actor_list = []
# Get the spawn points from the map
spawn_points = world.get_map().get_spawn_points()
start_point = spawn_points[0]
# Store world variable into local variable
town_map = world.get_map()
# all waypoint pairs defining roads (tuples)
roads = town_map.get_topology()

# vehicle = world.spawn_actor(vehicle_bp, start_point)
# actor_list.append(vehicle)
# vehicle.set_autopilot(True)

# Trace route from start to end
start_location = carla.Location(start_point.location)
end_location = carla.Location(x=44, y=139, z=0)

sampling_resolution = 2.0
grp = GlobalRoutePlanner(town_map, sampling_resolution)
route = grp.trace_route(start_location, end_location)
for waypoint in route:
    # world.debug.draw_string(waypoint[0].transform.location, 'O', draw_shadow=False,
    #                         color=carla.Color(r=255, g=0, b=0), life_time=120.0,
    #                         persistent_lines=True)
    world.debug.draw_point(waypoint[0].transform.location, size=0.1, color=carla.Color(r=0, g=255, b=0), life_time=120.0,
                           persistent_lines=True)
    
# Cleanup function to remove active actors
def cleanUp():
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")

# Define a function to calculate the distance between two points
def calculate_distance(location1, location2):
    return location1.distance(location2)

# If the script is interrupted, cleanup actors
try:
        # Spawn the vehicle at the start point
        vehicle = world.spawn_actor(vehicle_bp, start_point)
        actor_list.append(vehicle)
        if vehicle is not None:
            print(f"Vehicle spawned at location: {start_point.location}")
        else:
            print("Failed to spawn the vehicle.")

        # Control the vehicle to drive forward
        vehicle.apply_control(carla.VehicleControl(throttle=0.7))
        print("Vehicle is moving forward.")

        # Main Loop to drive vehicle along the route
        reached_target = False
        try:
            while True:
                # drive the vehicle to the next waypoint in the route
                for waypoint in route:
                    # calculate distance to target
                    distance = calculate_distance(vehicle.get_location(), waypoint[0].transform.location)
                    print(f"Distance to target: {distance:.2f} m")

                    # Check if vehicle has reached the target waypoint
                    if distance < 0.5:
                        print("Vehicle reached the target waypoint.")
                        
                    # stop once vehicle reaches final target
                    if waypoint == route[-1]:
                        reached_target = True
                        break

                    else:
                        vehicle.apply_control(carla.VehicleControl(throttle=0.7))

                time.sleep(1)
                
        finally:
            if reached_target:
                print("Vehicle reached the final target.")
            else:
                print("Vehicle did not reach the final target.")
            cleanUp()
except KeyboardInterrupt:
    cleanUp()
    print("Script interrupted by user.")
except Exception as e:
    print(f"An error occurred: {e}")
    cleanUp()
    raise e