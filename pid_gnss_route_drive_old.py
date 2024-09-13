# Description: This script utilizes a PID controller to integrate with the carla waypoint system
# Plan a route from the start point to the end point using the GlobalRoutePlanner to generate a list
# of waypoints from start to destination positions and drives the vehicle along the waypoints.
# until it reaches the destination.
import glob
import os
import sys
import math
import time
import numpy as np

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
actor_list = []

# Get the spawn points from the map
spawn_points = world.get_map().get_spawn_points()
start_point = spawn_points[0]
town_map = world.get_map()

# Trace route from start to end
start_location = carla.Location(start_point.location)
end_location = carla.Location(x=44, y=139, z=0)
# Transform end location to its geoLocation (GNSS) equivalent
gnss_end_location = town_map.transform_to_geolocation(end_location)

# Use the GlobalRoutePlanner to trace a route from start to end
sampling_resolution = 2.0
grp = GlobalRoutePlanner(town_map, sampling_resolution)
route = grp.trace_route(start_location, end_location)
for waypoint in route:
    world.debug.draw_point(waypoint[0].transform.location, size=0.1, color=carla.Color(r=0, g=255, b=0), life_time=120.0, persistent_lines=True)

# PID Controllers for throttle and steering
# Default: Kp=1.0, Ki=0.0, Kd=0.05
throttle_pid = PIDController(Kp=0.8, Ki=0.01, Kd=0.1)
steering_pid = PIDController(Kp=1.2, Ki=0.01, Kd=0.1)

# Cleanup function to remove active actors
def cleanUp():
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")

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



# Print all of the route waypoints to the terminal for debugging
# for i, waypoint in enumerate(route):
#             location = waypoint[0].transform.location
#             yaw = waypoint[0].transform.rotation.yaw
#             print(f"Waypoint {i}: Location({location.x}, {location.y}, {location.z}), Yaw: {yaw}")

# Main loop to drive the vehicle along the route
try:
    # Draw circles around start and end points
    draw_circle(world, start_point.location, color=carla.Color(r=70, g=255, b=100))
    draw_circle(world, end_location, color=carla.Color(r=255, g=0, b=255))

    # Spawn the vehicle at the start point
    vehicle = world.spawn_actor(vehicle_bp, start_point)
    actor_list.append(vehicle)
    if vehicle is not None:
        print(f"Vehicle spawned at location: {start_point.location}")
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

    # Reference location variables
    ref_location = start_point.location
    ref_geolocation = town_map.transform_to_geolocation(ref_location)

    # Print start and end GNSS locations for debugging
    print(f"Start GNSS location: {ref_geolocation}")
    print(f"End GNSS location: {gnss_end_location}")

    # Main loop to drive the vehicle along the route
    reached_target = False
    last_time = time.time()
    target_index = 0

    while not reached_target:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time

        if gnss_data["latitude"] is not None and gnss_data["longitude"] is not None:
            # Convert GNSS data to CARLA coordinates using an approximation
            latitude_diff = gnss_data["latitude"] - ref_geolocation.latitude
            longitude_diff = gnss_data["longitude"] - ref_geolocation.longitude

            # Assuming 1 degree latitude ≈ 111.32 km and 1 degree longitude ≈ 111.32 km * cos(latitude)
            approx_x = ref_location.x + longitude_diff * 111320 * math.cos(ref_geolocation.latitude * math.pi / 180)
            approx_y = ref_location.y + latitude_diff * 111320

            vehicle_location = carla.Location(x=approx_x, y=approx_y, z=0)

            print(f"GNSS data - Latitude: {gnss_data['latitude']}, Longitude: {gnss_data['longitude']}")
            print(f"Calculated location - x: {approx_x}, y: {approx_y}")

            target_waypoint = route[target_index][0]
            target_location = target_waypoint.transform.location

            # Visualize the target waypoint
            world.debug.draw_point(target_location, size=0.2, color=carla.Color(r=255, g=0, b=0), life_time=0.1)

            distance = calculate_distance(vehicle_location, target_location)
            print(f"Distance to waypoint {target_index}: {distance:.2f} m")

            if distance < 0.5:  # Reduced threshold for better accuracy
                print(f"Reached waypoint {target_index} at location {target_location}")
                target_index += 1
                if target_index >= len(route):
                    reached_target = True
                    break
                target_waypoint = route[target_index][0]
                target_location = target_waypoint.transform.location
                print(f"Moving to waypoint {target_index} at location {target_location}")

            # Compute control signals
            error = calculate_distance(vehicle_location, target_location)
            # yaw_error = normalize_angle(math.atan2(target_location.y - current_location.y, target_location.x - current_location.x) - vehicle_yaw)

            throttle = throttle_pid.compute(error, dt)
            throttle = np.clip(throttle, 0.0, 1.0)  # Ensure throttle is between 0 and 1

            # Compute steering angle
            def normalize_angle(angle):
                while angle > math.pi:
                    angle -= 2 * math.pi
                while angle < -math.pi:
                    angle += 2 * math.pi
                return angle

            # Update yaw error calculation
            vehicle_transform = vehicle.get_transform()
            vehicle_yaw = vehicle_transform.rotation.yaw * (math.pi / 180.0)
            target_yaw = target_waypoint.transform.rotation.yaw * (math.pi / 180.0)
            yaw_error = normalize_angle(target_yaw - vehicle_yaw)

            steering = steering_pid.compute(yaw_error, dt)
            steering = np.clip(steering, -1.0, 1.0)  # Ensure steering is between -1 and 1

            # Apply vehicle control
            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steering))

            print(f"Distance to target: {distance:.2f} m, Throttle: {throttle:.2f}, Steering: {steering:.2f}")

            # Default 0.05 seconds sleep
            time.sleep(0.025)

    print("Vehicle reached the final target.")
    cleanUp()

except KeyboardInterrupt:
    cleanUp()
    print("Script interrupted by user.")
except Exception as e:
    print(f"An error occurred: {e}")
    cleanUp()
    raise e

