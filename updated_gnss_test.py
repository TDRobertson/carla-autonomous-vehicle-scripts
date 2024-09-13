import glob
import os
import sys
import carla
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

# Add the path to the agents module
agents_path = 'C:/CARLA_0.9.15/PythonAPI/examples'
if agents_path not in sys.path:
    sys.path.append(agents_path)

from agents.navigation.global_route_planner import GlobalRoutePlanner

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()
town_map = world.get_map()

# GNSS coordinates for spawn and destination
spawn_gnss = carla.GeoLocation(latitude=-0.00123, longitude=-0.000226, altitude=0.0)
destination_gnss = carla.GeoLocation(latitude=-0.001236, longitude=0.000309, altitude=0.0)

# Convert GNSS to CARLA locations
spawn_location = town_map.get_waypoint(carla.Location(x=spawn_gnss.latitude, y=spawn_gnss.longitude, z=spawn_gnss.altitude)).transform.location
destination_location = town_map.get_waypoint(carla.Location(x=destination_gnss.latitude, y=destination_gnss.longitude, z=destination_gnss.altitude)).transform.location

# Spawn vehicle at GNSS location
vehicle_bp = blueprint_library.filter('vehicle.*')[0]
vehicle = world.spawn_actor(vehicle_bp, carla.Transform(spawn_location))
actor_list = [vehicle]

# Create sensors and attach them to the vehicle
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_transform = carla.Transform(carla.Location(z=2.0))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
actor_list.append(camera)

gnss_bp = blueprint_library.find('sensor.other.gnss')
gnss = world.spawn_actor(gnss_bp, carla.Transform(), attach_to=vehicle)
actor_list.append(gnss)

imu_bp = blueprint_library.find('sensor.other.imu')
imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=vehicle)
actor_list.append(imu)

collision_bp = blueprint_library.find('sensor.other.collision')
collision = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
actor_list.append(collision)

lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
lane_invasion = world.spawn_actor(lane_invasion_bp, carla.Transform(), attach_to=vehicle)
actor_list.append(lane_invasion)

obstacle_bp = blueprint_library.find('sensor.other.obstacle')
obstacle_bp.set_attribute('distance', '10')
obstacle_bp.set_attribute('hit_radius', '0.5')
obstacle = world.spawn_actor(obstacle_bp, carla.Transform(), attach_to=vehicle)
actor_list.append(obstacle)

# Setup sensor callbacks
sensor_data = {'rgb_image': None, 'gnss': None, 'imu': None, 'collision': False, 'lane_invasion': False, 'obstacle': []}

def rgb_callback(image):
    sensor_data['rgb_image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

def gnss_callback(data):
    sensor_data['gnss'] = [data.latitude, data.longitude, data.altitude]

def imu_callback(data):
    sensor_data['imu'] = [data.accelerometer, data.gyroscope, data.compass]

def collision_callback(event):
    sensor_data['collision'] = True

def lane_invasion_callback(event):
    sensor_data['lane_invasion'] = True

def obstacle_callback(event):
    if 'static' not in event.other_actor.type_id:
        sensor_data['obstacle'].append({'transform': event.other_actor.type_id, 'frame': event.frame})

camera.listen(rgb_callback)
gnss.listen(gnss_callback)
imu.listen(imu_callback)
collision.listen(collision_callback)
lane_invasion.listen(lane_invasion_callback)
obstacle.listen(obstacle_callback)

# Plan route from start to destination
grp = GlobalRoutePlanner(town_map, sampling_resolution=2)
route = grp.trace_route(spawn_location, destination_location)

# Draw the route
for waypoint, _ in route:
    world.debug.draw_string(waypoint.transform.location, 'O', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=120.0, persistent_lines=True)

# Control the vehicle to follow the route
class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, target, current):
        error = target - current
        self.integral += error
        derivative = error - self.prev_error
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative

pid_controller = PIDController(Kp=1.0, Ki=0.01, Kd=0.1)

def move_vehicle_to_next_waypoint(vehicle, waypoint):
    control = carla.VehicleControl()
    vehicle_transform = vehicle.get_transform()
    vehicle_location = vehicle_transform.location
    vehicle_forward_vector = vehicle_transform.get_forward_vector()

    waypoint_location = waypoint.transform.location
    direction_vector = waypoint_location - vehicle_location
    direction_vector = np.array([direction_vector.x, direction_vector.y, direction_vector.z])
    forward_vector = np.array([vehicle_forward_vector.x, vehicle_forward_vector.y, vehicle_forward_vector.z])

    angle = np.arctan2(direction_vector[1], direction_vector[0]) - np.arctan2(forward_vector[1], forward_vector[0])
    distance = np.linalg.norm(direction_vector[:2])

    steer = np.clip(pid_controller.compute(0, angle), -1.0, 1.0)
    control.steer = steer
    control.throttle = 0.5 if distance > 2.0 else 0.0
    control.brake = 1.0 if distance < 2.0 else 0.0
    vehicle.apply_control(control)

current_waypoint_index = 0
total_waypoints = len(route)

while current_waypoint_index < total_waypoints:
    move_vehicle_to_next_waypoint(vehicle, route[current_waypoint_index][0])
    vehicle_location = vehicle.get_transform().location
    waypoint_location = route[current_waypoint_index][0].transform.location
    distance = np.linalg.norm([vehicle_location.x - waypoint_location.x, vehicle_location.y - waypoint_location.y])
    if distance < 2.0:
        current_waypoint_index += 1

# Cleanup after 20 seconds
import time
time.sleep(20)
camera.stop()
gnss.stop()
imu.stop()
collision.stop()
lane_invasion.stop()
obstacle.stop()

for actor in actor_list:
    actor.destroy()

print("All cleaned up!")
