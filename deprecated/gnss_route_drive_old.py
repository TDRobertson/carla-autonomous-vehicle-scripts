# Deprecated file originally used to test transforming geolocation data. 
# This file is no longer used in the project.

import glob
import os
import sys
import math
import time
import numpy as np
import cv2
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

# Get the spawn points from the map
spawn_points = world.get_map().get_spawn_points()
start_point = spawn_points[0]
town_map = world.get_map()

# Define the spawn point
initial_spawn_point = carla.Transform(carla.Location(x=-25.19, y=139, z=0), carla.Rotation(yaw=0))
final_target_location = carla.Location(x=44, y=139, z=0)

# Correctly transform target location to geolocation
gnss_target_location = map.transform_to_geolocation(final_target_location)

# Cleanup function to remove active actors
def cleanUp():
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")

actor_list = []

# Define a function to calculate the distance between two points
def calculate_distance(location1, location2):
    return location1.distance(location2)

# Draw a circle using multiple points
def draw_circle(world, location, radius=1.0, color=carla.Color(r=255, g=0, b=0), life_time=60.0, num_points=36):
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x_offset = radius * math.cos(angle)
        y_offset = radius * math.sin(angle)
        point_location = carla.Location(location.x + x_offset, location.y + y_offset, location.z)
        world.debug.draw_point(point_location, size=0.1, color=color, life_time=life_time)


# Create route planner for vehicle to follow
grp = GlobalRoutePlanner(map)

try:
    # Draw circles at the start and target locations
    draw_circle(world, initial_spawn_point.location)
    draw_circle(world, final_target_location, color=carla.Color(r=0, g=255, b=0))

    # Spawn the vehicle
    vehicle = world.spawn_actor(vehicle_bp, initial_spawn_point)
    if vehicle is not None:
        print(f"Vehicle spawned at location: {initial_spawn_point.location}")
    else:
        print("Failed to spawn the vehicle.")

    # Add a GNSS sensor to the vehicle
    gnss_bp = blueprint_library.find('sensor.other.gnss')
    gnss_transform = carla.Transform(carla.Location(x=0, y=0, z=2))  # Position the GNSS sensor on the vehicle
    gnss_sensor = world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)

    # Initialize GNSS data storage
    gnss_data = {"latitude": None, "longitude": None, "altitude": None}

    # Define a callback to store GNSS data
    def gnss_callback(gnss):
        gnss_data["latitude"] = gnss.latitude
        gnss_data["longitude"] = gnss.longitude
        gnss_data["altitude"] = gnss.altitude

    # Subscribe to GNSS sensor data
    gnss_sensor.listen(gnss_callback)
    # gnss_sensor.listen(lambda data: gnss_callback(data))


    # Control the vehicle to drive forward
    vehicle.apply_control(carla.VehicleControl(throttle=0.7))
    print("Vehicle is moving forward.")

    # Main loop to drive the vehicle
    reached_target = False
    try:
        while True:
            if gnss_data["latitude"] is not None and gnss_data["longitude"] is not None:
                # Convert GNSS data to CARLA coordinates using an approximation
                ref_location = carla.Location(x=-25.19, y=139, z=0)
                ref_geolocation = town_map.transform_to_geolocation(ref_location)
                
                lat_diff = gnss_data["latitude"] - ref_geolocation.latitude
                lon_diff = gnss_data["longitude"] - ref_geolocation.longitude

                # Assuming 1 degree latitude ≈ 111.32 km and 1 degree longitude ≈ 111.32 km * cos(latitude)
                # longitude decreases moving towards poles, so must take cosine of latitude to convert correctly.
                approx_x = ref_location.x + lon_diff * 111320 * math.cos(ref_geolocation.latitude * math.pi / 180)
                approx_y = ref_location.y + lat_diff * 111320

                current_location = carla.Location(x=approx_x, y=approx_y)

                # Calculate the distance to the target location
                distance = calculate_distance(current_location, final_target_location)
                # print(f"Distance to target (GNSS): {distance} meters")

                # Check if the vehicle has reached the target location
                if distance < 10.0 and not reached_target:  # Start slowing down 10 meters from the target
                    reached_target = True
                    print("Approaching target! Slowing down.")

                # Gradually reduce throttle and apply brake
                if reached_target:
                    throttle = max(0.0, 0.7 * (distance / 10.0))
                    brake = min(1.0, 1.0 - throttle)
                    vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake))
                    if distance < 0.5:  # Stop when within 0.5 meters of the target
                        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                        print("Vehicle stopped.")
                        break
                else:
                    vehicle.apply_control(carla.VehicleControl(throttle=0.7))

            time.sleep(0.1)

    finally:
        # Clean up
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))  # Ensure vehicle is fully stopped
        vehicle.destroy()
        gnss_sensor.destroy()

except Exception as e:
    print(f"An error occurred: {e}")