import glob
import os
import sys
import math
import time
import numpy as np
import random

# Add the CARLA Python API to PYTHONPATH
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

carla_path = 'C:/CARLA_0.9.15/PythonAPI/carla'
if carla_path not in sys.path:
    sys.path.append(carla_path)

import carla
from agents.navigation.global_route_planner import GlobalRoutePlanner


class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0
        self.previous_error = 0

    def compute(self, error, dt):
        if dt == 0:  # Prevent division by zero
            return 0
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        return output


# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.*')[0]
gnss_bp = blueprint_library.find('sensor.other.gnss')
actor_list = []

# Define spawn points and target
spawn_point = carla.Transform(carla.Location(x=-20.19, y=137.09, z=0.00), carla.Rotation(yaw=0))
target_location = carla.Location(x=59.35, y=137.68, z=0.00)
sampling_resolution = 3.0
town_map = world.get_map()
grp = GlobalRoutePlanner(town_map, sampling_resolution)
route = grp.trace_route(spawn_point.location, target_location)

# PID Controllers
throttle_pid = PIDController(Kp=0.8, Ki=0.01, Kd=0.1)
steering_pid = PIDController(Kp=1.0, Ki=0.01, Kd=0.1)

# GNSS and true GPS data
gnss_data = {"latitude": None, "longitude": None, "altitude": None}
true_gps_data = {"latitude": None, "longitude": None, "altitude": None}

# Cleanup function
def cleanUp():
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")


# GNSS callback
def gnss_callback(gnss):
    current_time = time.time()
    elapsed_time = current_time - start_time

    if elapsed_time > startup:
        noise_std_dev = ((elapsed_time - startup) / 20.0) * max_noise_std_dev
        bias = ((elapsed_time - startup) / 20.0) * max_bias
    else:
        noise_std_dev = 0.0
        bias = 0.0

    gnss_data["latitude"] = gnss.latitude + random.gauss(0, noise_std_dev) + bias
    gnss_data["longitude"] = gnss.longitude + random.gauss(0, noise_std_dev)
    gnss_data["altitude"] = gnss.altitude


# Reverse GPS spoofing
def reverse_gps_spoofing(spoofed_gps, elapsed_time, startup_time, max_noise_std_dev, max_bias):
    if elapsed_time < startup_time:
        return spoofed_gps

    bias = ((elapsed_time - startup_time) / 20.0) * max_bias
    estimated_latitude = spoofed_gps["latitude"] - bias
    estimated_longitude = spoofed_gps["longitude"]  # Assuming no bias on longitude
    estimated_altitude = spoofed_gps["altitude"]
    return {"latitude": estimated_latitude, "longitude": estimated_longitude, "altitude": estimated_altitude}


# Convert CARLA location to GPS
def true_location_to_gps(location, reference_geolocation):
    ref_location = carla.Location(x=-20.19, y=137.09, z=0.00)
    lat_diff = (location.y - ref_location.y) / 111320
    lon_diff = (location.x - ref_location.x) / (111320 * math.cos(reference_geolocation.latitude * math.pi / 180))
    latitude = reference_geolocation.latitude + lat_diff
    longitude = reference_geolocation.longitude + lon_diff
    altitude = location.z
    return {"latitude": latitude, "longitude": longitude, "altitude": altitude}


# GPS to CARLA coordinates
def gps_to_carla_coords(gnss, ref_geolocation, ref_location):
    lat_diff = gnss["latitude"] - ref_geolocation.latitude
    lon_diff = gnss["longitude"] - ref_geolocation.longitude
    x = ref_location.x + lon_diff * 111320 * math.cos(ref_geolocation.latitude * math.pi / 180)
    y = ref_location.y + lat_diff * 111320
    return carla.Location(x=x, y=y, z=ref_location.z)


# Draw route
for waypoint in route:
    world.debug.draw_point(waypoint[0].transform.location, size=0.1, color=carla.Color(r=0, g=255, b=0), life_time=120.0, persistent_lines=True)

try:
    # Vehicle and GNSS initialization
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    actor_list.append(vehicle)
    gnss_sensor = world.spawn_actor(gnss_bp, carla.Transform(carla.Location(x=0, y=0, z=2)), attach_to=vehicle)
    actor_list.append(gnss_sensor)
    gnss_sensor.listen(gnss_callback)

    ref_geolocation = town_map.transform_to_geolocation(spawn_point.location)

    # Variables for spoofing
    start_time = time.time()
    max_noise_std_dev = 0.00001
    max_bias = -0.00005
    startup = 3
    target_index = 0
    reached_target = False

    while not reached_target:
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Wait for GNSS data
        if gnss_data["latitude"] is None or gnss_data["longitude"] is None:
            print("Waiting for GNSS data...")
            time.sleep(0.1)
            continue

        # True GPS from CARLA
        vehicle_true_location = vehicle.get_location()
        true_gps_data.update(true_location_to_gps(vehicle_true_location, ref_geolocation))

        # Reverse-engineered GPS
        reversed_gps_data = reverse_gps_spoofing(gnss_data, elapsed_time, startup, max_noise_std_dev, max_bias)

        # Log data
        print(f"True GPS:      Lat: {true_gps_data['latitude']:.6f}, Lon: {true_gps_data['longitude']:.6f}")
        print(f"Spoofed GPS:   Lat: {gnss_data['latitude']:.6f}, Lon: {gnss_data['longitude']:.6f}")
        print(f"Reversed GPS:  Lat: {reversed_gps_data['latitude']:.6f}, Lon: {reversed_gps_data['longitude']:.6f}")

        # Convert spoofed GPS to CARLA coordinates
        current_location = gps_to_carla_coords(gnss_data, ref_geolocation, spawn_point.location)

        # Visualize current vehicle position and spoofed data
        world.debug.draw_point(current_location, size=0.2, color=carla.Color(r=255, g=0, b=0), life_time=0.1)
        world.debug.draw_point(vehicle_true_location, size=0.2, color=carla.Color(r=0, g=0, b=255), life_time=0.1)

        # Target waypoint logic
        target_waypoint = route[target_index][0]
        target_location = target_waypoint.transform.location
        if vehicle_true_location.distance(target_location) < 2.0:
            target_index += 1
            if target_index >= len(route):
                reached_target = True

        time.sleep(0.1)

    print("Vehicle reached the final target.")
    cleanUp()

except KeyboardInterrupt:
    cleanUp()
    print("Script interrupted by user.")
except Exception as e:
    print(f"An error occurred: {e}")
    cleanUp()
    raise e
