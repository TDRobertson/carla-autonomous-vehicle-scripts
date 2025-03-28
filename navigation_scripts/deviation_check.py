'''
The purpose of this file is to demonstrate how the addition of gaussian noise or bias can trick the vehicles gps navigation system into incorrectly determining its location.
This results in the vehicles PID controller incorrectly determining the vehicles location and thus the vehicle will drive off the road. For example, the addition of a left leaning bias
causes the controller to believe the vehicle is further to the right than it is in reality. This causes the vehicle to steer left to correct its position, causing it to cross the median 
into potential oncoming traffic. This is a dangerous scenario that can be exploited by malicious actors to cause accidents.
'''

import glob
import os
import sys
import math
import time
import numpy as np
import cv2
import threading
import random

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

# Get the spawn points from the map and define our route
spawn_points = world.get_map().get_spawn_points()
town_map = world.get_map()
spawn_point = carla.Transform(carla.Location(x=-20.19, y=137.09, z=0.00), carla.Rotation(yaw=0))
target_location = carla.Location(x=59.35, y=137.68, z=0.00)

sampling_resolution = 3.0   # distance between each waypoint (in meters)
grp = GlobalRoutePlanner(town_map, sampling_resolution)
route = grp.trace_route(spawn_point.location, target_location)
for waypoint in route:
    world.debug.draw_point(waypoint[0].transform.location, size=0.1, color=carla.Color(r=0, g=255, b=0), life_time=120.0, persistent_lines=True)

# PID Controllers for throttle and steering
throttle_pid = PIDController(Kp=0.8, Ki=0.01, Kd=0.1)
steering_pid = PIDController(Kp=1.0, Ki=0.01, Kd=0.1)

# Cleanup function to remove active actors
def cleanUp():
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")

# Draw a circle using multiple points
def draw_circle(world, location, radius=1.0, color=carla.Color(r=255, g=0, b=0), life_time=60.0, num_points=36):
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        x_offset = radius * math.cos(angle)
        y_offset = radius * math.sin(angle)
        point_location = carla.Location(location.x + x_offset, location.y + y_offset, location.z)
        world.debug.draw_point(point_location, size=0.1, color=color, life_time=life_time)

# Define a function to calculate the x and y distances between two points
def calculate_distances(location1, location2):
    dx = location2.x - location1.x
    dy = location2.y - location1.y
    return dx, dy

# Print the route waypoints for debugging
for i, waypoint in enumerate(route):
    location = waypoint[0].transform.location
    yaw = waypoint[0].transform.rotation.yaw
    print(f"Waypoint {i}: Location({location.x}, {location.y}, {location.z}), Yaw: {yaw}")

# Define a callback to store camera data
def camera_callback(image, data_dict):
    data_dict['image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

# Function to display camera feed in a separate thread
def display_camera_feed(camera_data):
    cv2.namedWindow("Camera Feed", cv2.WINDOW_AUTOSIZE)
    while True:
        cv2.imshow("Camera Feed", camera_data["image"])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

# Compute steering angle normalization
def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

try:
    # Draw circles at the start and target locations
    draw_circle(world, spawn_point.location)
    draw_circle(world, target_location, color=carla.Color(r=0, g=200, b=125))

    # Spawn the vehicle at the start point
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    actor_list.append(vehicle)
    if vehicle is not None:
        print(f"Vehicle spawned at location: {spawn_point.location}")
    else:
        print("Failed to spawn the vehicle.")

    # Add a GNSS sensor to the vehicle
    gnss_bp = blueprint_library.find('sensor.other.gnss')
    gnss_transform = carla.Transform(carla.Location(x=0, y=0, z=2))
    gnss_sensor = world.spawn_actor(gnss_bp, gnss_transform, attach_to=vehicle)
    actor_list.append(gnss_sensor)

    # Initialize GNSS data storage
    gnss_data = {"latitude": None, "longitude": None, "altitude": None}

    # Variables to track elapsed time and maximum noise/bias values
    start_time = time.time()
    max_noise_std_dev = 0.00001  # Maximum standard deviation for noise (approx. 1.11 m error per step)
    max_bias = -0.00002         # Maximum bias to introduce a left swerve (approx. 1.11 m error)
    startup = 3

    # GNSS sensor callback with added bias (and optional noise)
    def gnss_callback(gnss):
        current_time = time.time()
        elapsed_time = current_time - start_time
        if elapsed_time > startup:
            noise_std_dev = ((elapsed_time - startup) / 20.0) * max_noise_std_dev
            bias = ((elapsed_time - startup) / 20.0) * max_bias
        else:
            noise_std_dev = 0.0
            bias = 0.0

        # Apply bias (and noise if needed)
        noisy_latitude = gnss.latitude + bias
        noisy_longitude = gnss.longitude  # No noise applied to longitude here
        gnss_data["latitude"] = noisy_latitude
        gnss_data["longitude"] = noisy_longitude
        gnss_data["altitude"] = gnss.altitude

    # Subscribe to GNSS sensor data
    gnss_sensor.listen(gnss_callback)

    # Initialize camera sensor
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_transform = carla.Transform(carla.Location(x=1.5, y=0, z=2.4), carla.Rotation(pitch=-15))
    camera_sensor = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    actor_list.append(camera_sensor)

    image_width = camera_bp.get_attribute('image_size_x').as_int()
    image_height = camera_bp.get_attribute('image_size_y').as_int()
    camera_data = {"image": np.zeros((image_height, image_width, 4), dtype=np.uint8)}
    camera_sensor.listen(lambda image: camera_callback(image, camera_data))

    camera_thread = threading.Thread(target=display_camera_feed, args=(camera_data,))
    camera_thread.start()

    # Initialize variables to track the min, max, and average deviation (after PID correction)
    min_deviation = float('inf')
    max_deviation = 0.0
    sum_deviation = 0.0
    count_deviation = 0

    # Main loop to drive the vehicle along the route
    reached_target = False
    last_time = time.time()
    target_index = 0

    while not reached_target:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        # Fetch current true vehicle location from CARLA
        vehicle_true_location = vehicle.get_location()
        print(f"True Location from CARLA: (X: {vehicle_true_location.x:.2f}, Y: {vehicle_true_location.y:.2f})")

        target_waypoint = route[target_index][0]
        target_location = target_waypoint.transform.location
        world.debug.draw_point(target_location, size=0.2, color=carla.Color(r=255, g=0, b=0), life_time=0.1)

        # Convert GNSS data to CARLA coordinates (approximation)
        ref_location = carla.Location(x=-20.19, y=137.09, z=0.00)
        ref_geolocation = town_map.transform_to_geolocation(ref_location)
        lat_diff = gnss_data["latitude"] - ref_geolocation.latitude
        lon_diff = gnss_data["longitude"] - ref_geolocation.longitude

        approx_x = ref_location.x + lon_diff * 111320 * math.cos(ref_geolocation.latitude * math.pi / 180)
        approx_y = ref_location.y + lat_diff * 111320
        current_location = carla.Location(x=approx_x, y=approx_y)

        print(f"Modified GPS Location: (X: {approx_x:.2f}, Y: {approx_y:.2f})")

        # Print deviation BEFORE applying PID corrections
        pre_diff_x = vehicle_true_location.x - approx_x
        pre_diff_y = vehicle_true_location.y - approx_y
        print(f"Deviation BEFORE PID correction: ΔX: {pre_diff_x:.2f}, ΔY: {pre_diff_y:.2f}")

        # Also print the difference (True - Modified) for context
        diff_x = vehicle_true_location.x - approx_x
        diff_y = vehicle_true_location.y - approx_y
        print(f"Difference (True - Modified): ΔX: {diff_x:.2f}, ΔY: {diff_y:.2f}\n")

        # Calculate distances to target using the modified GNSS location
        dx, dy = calculate_distances(current_location, target_location)
        print(f"X Distance to target (GNSS): {dx:.2f} meters, Y Distance to target (GNSS): {dy:.2f} meters")
        print(f"Current GNSS Location: ({approx_x:.2f}, {approx_y:.2f}), Target GNSS Location: ({target_location.x:.2f}, {target_location.y:.2f})")

        # Check if the current waypoint has been reached
        if abs(dx) < 1.5 and abs(dy) < 0.5:
            print(f"Reached waypoint {target_index} at location {target_location}")
            target_index += 1
            if target_index >= len(route):
                reached_target = True
                continue
            target_waypoint = route[target_index][0]
            target_location = target_waypoint.transform.location
            print(f"Moving to waypoint {target_index} at location {target_location}")

        # Compute control signals using PID controllers
        error_x = dx
        error_y = dy
        throttle = throttle_pid.compute(math.sqrt(error_x**2 + error_y**2), dt)
        throttle = np.clip(throttle, 0.0, 0.75)

        target_yaw = math.atan2(-dy, dx)  # Negative dy for coordinate conversion
        target_yaw = normalize_angle(target_yaw)
        vehicle_transform = vehicle.get_transform()
        vehicle_yaw = vehicle_transform.rotation.yaw * (math.pi / 180.0)
        yaw_error = normalize_angle(target_yaw - vehicle_yaw)
        steering = steering_pid.compute(yaw_error, dt)
        steering = np.clip(steering, -1.0, 1.0)

        # Apply the control commands to the vehicle
        vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steering))

        # Allow a short delay for the control to take effect and then fetch the updated location
        time.sleep(0.05)
        updated_location = vehicle.get_location()
        post_diff_x = updated_location.x - approx_x
        post_diff_y = updated_location.y - approx_y
        print(f"Deviation AFTER PID correction: ΔX: {post_diff_x:.2f}, ΔY: {post_diff_y:.2f}\n")

        # Calculate the current deviation magnitude and update the min, max, and average trackers
        current_deviation = math.sqrt(post_diff_x**2 + post_diff_y**2)
        min_deviation = min(min_deviation, current_deviation)
        max_deviation = max(max_deviation, current_deviation)
        sum_deviation += current_deviation
        count_deviation += 1

        time.sleep(0.05)

    # After the loop, compute and print the average deviation
    if count_deviation > 0:
        avg_deviation = sum_deviation / count_deviation
    else:
        avg_deviation = 0.0

    print(f"Minimum deviation during run: {min_deviation:.2f} meters")
    print(f"Maximum deviation during run: {max_deviation:.2f} meters")
    print(f"Average deviation during run: {avg_deviation:.2f} meters")
    print("Vehicle reached the final target.")
    cleanUp()

except KeyboardInterrupt:
    cleanUp()
    print("Script interrupted by user.")
except Exception as e:
    print(f"An error occurred: {e}")
    cleanUp()
    raise e
