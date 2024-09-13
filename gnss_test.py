# attempting to create a route manager in carla using gnss for navigation, currently doesn't work as intended

import glob
import os
import sys
import random
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

client = carla.Client('localhost', 2000)
world = client.get_world()
blueprint_library = world.get_blueprint_library()
spawn_points = world.get_map().get_spawn_points()
start_point = spawn_points[0]

def cleanUp():
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")

actor_list = []

vehicle_bp = blueprint_library.filter('vehicle.*')[0]
vehicle = world.spawn_actor(vehicle_bp, start_point)
actor_list.append(vehicle)

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

def rgb_callback(image, data_dict):
    data_dict['rgb_image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

def gnss_callback(data, data_dict):
    data_dict['gnss'] = [data.latitude, data.longitude, data.altitude]

def imu_callback(data, data_dict):
    data_dict['imu'] = [data.accelerometer, data.gyroscope, data.compass]

def collision_callback(event, data_dict):
    data_dict['collision'] = True

def lane_invasion_callback(event, data_dict):
    data_dict['lane_invasion'] = True

def obstacle_callback(event, data_dict, camera, k_mat):
    if 'static' not in event.other_actor.type_id:
        data_dict['obstacle'].append({'transform': event.other_actor.type_id, 'frame': event.frame})

    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
    image_point = get_image_point(event.other_actor.get_transform().location, k_mat, world_2_camera)
    if 0 < image_point[0] < image_width and 0 < image_point[1] < image_height:
        cv2.circle(data_dict['rgb_image'], tuple(image_point), 10, (0, 0, 255), 3)

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def get_image_point(location, k_mat, world_2_camera):
    point = np.array([location.x, location.y, location.z, 1])
    point_camera = np.dot(world_2_camera, point)
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]
    point_image = np.dot(k_mat, point_camera)
    point_image[0] /= point_image[2]
    point_image[1] /= point_image[2]
    return tuple(map(int, point_image[0:2]))

world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
image_width = camera_bp.get_attribute('image_size_x').as_int()
image_height = camera_bp.get_attribute('image_size_y').as_int()
fov = camera_bp.get_attribute('fov').as_float()
k_mat = build_projection_matrix(image_width, image_height, fov)

town_map = world.get_map()
roads = town_map.get_topology()
gnss_data = []
for road in roads:
    for waypoint in road:
        gnss_data.append([waypoint.transform.location.x, waypoint.transform.location.y, waypoint.transform.location.z])
# print(gnss_data)

sampling_resolution = 2
grp = GlobalRoutePlanner(town_map, sampling_resolution)
start_route = carla.Location(x=gnss_data[0][0], y=gnss_data[0][1], z=gnss_data[0][2])
end_route = carla.Location(x=gnss_data[-1][0], y=gnss_data[-1][1], z=gnss_data[-1][2])
route = grp.trace_route(start_route, end_route)

for i in range(len(route)):
    world.debug.draw_string(route[i][0].transform.location, 'O', draw_shadow=False,
                            color=carla.Color(r=255, g=0, b=0), life_time=120.0,
                            persistent_lines=True)

image_w = camera_bp.get_attribute('image_size_x').as_int()
image_h = camera_bp.get_attribute('image_size_y').as_int()

collision_counter = 20
lane_invasion_counter = 20

sensor_data = {'rgb_image': np.zeros((image_h, image_w, 4)),
               'collision': False,
               'lane_invasion': False,
               'obstacle': [],
               'gnss': [0, 0, 0],
               'imu': {
                   'gyro': carla.Vector3D(),
                   'accel': carla.Vector3D(),
                   'compass': 0
               }
               }

cv2.namedWindow('Camera RGB', cv2.WINDOW_AUTOSIZE)
cv2.imshow('Camera RGB', sensor_data['rgb_image'])
cv2.waitKey(1)

camera.listen(lambda image: rgb_callback(image, sensor_data))
gnss.listen(lambda data: gnss_callback(data, sensor_data))
imu.listen(lambda data: imu_callback(data, sensor_data))
collision.listen(lambda event: collision_callback(event, sensor_data))
lane_invasion.listen(lambda event: lane_invasion_callback(event, sensor_data))
obstacle.listen(lambda event: obstacle_callback(event, sensor_data, camera, k_mat))

font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 50)
fontScale = 0.5
fontColor = (255, 255, 255)
lineType = 2
thickness = 2

def draw_compass(img, theta):
    compass_center = (700, 100)
    compass_size = 50

    cardinal_directions = [
        ('N', [0, -1]),
        ('E', [1, 0]),
        ('S', [0, 1]),
        ('W', [-1, 0])
    ]

    for car_dir in cardinal_directions:
        cv2.putText(sensor_data['rgb_image'], car_dir[0],
                    (int(compass_center[0] + 1.2 * compass_size * car_dir[1][0]),
                     int(compass_center[1] + 1.2 * compass_size * car_dir[1][1])),
                    font, fontScale, fontColor, lineType, thickness)

    compass_point = (int(compass_center[0] + compass_size * np.sin(theta)), int(compass_center[1] - compass_size * np.cos(theta)))
    cv2.line(img, compass_center, compass_point, (255, 255, 255), 3)

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

while True:
    if current_waypoint_index < total_waypoints:
        move_vehicle_to_next_waypoint(vehicle, route[current_waypoint_index][0])
        vehicle_location = vehicle.get_transform().location
        waypoint_location = route[current_waypoint_index][0].transform.location
        distance = np.linalg.norm([vehicle_location.x - waypoint_location.x, vehicle_location.y - waypoint_location.y])
        if distance < 2.0:
            current_waypoint_index += 1
    else:
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 1.0
        vehicle.apply_control(control)

    cv2.putText(sensor_data['rgb_image'], 'Latitude: ' + str(sensor_data['gnss'][0]),
                (10, 30), font, fontScale, fontColor, lineType, thickness)

    cv2.putText(sensor_data['rgb_image'], 'Longitude: ' + str(sensor_data['gnss'][1]),
                (10, 50), font, fontScale, fontColor, lineType, thickness)

    cv2.putText(sensor_data['rgb_image'], 'Altitude: ' + str(sensor_data['gnss'][2]),
                (10, 70), font, fontScale, fontColor, lineType, thickness)

    if sensor_data['collision']:
        collision_counter -= 1
        if collision_counter <= 1:
            sensor_data['collision'] = False
        cv2.putText(sensor_data['rgb_image'], 'Collision Detected!',
                    (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, 2)
    else:
        collision_counter = 20

    if sensor_data['lane_invasion']:
        lane_invasion_counter -= 1
        if lane_invasion_counter <= 1:
            sensor_data['lane_invasion'] = False
        cv2.putText(sensor_data['rgb_image'], 'Lane Invasion Detected!',
                    (250, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, 2)

    cv2.imshow('Camera RGB', sensor_data['rgb_image'])
    if cv2.waitKey(1) == ord('q'):
        break

camera.stop()
gnss.stop()
imu.stop()
collision.stop()
lane_invasion.stop()
obstacle.stop()
cv2.destroyAllWindows()
cleanUp()
