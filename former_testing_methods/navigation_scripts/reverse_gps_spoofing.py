import glob
import os
import sys
import math
import time
import random
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

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.*')[0]
gnss_bp = blueprint_library.find('sensor.other.gnss')
actor_list = []

# Define the spawn point and target location
spawn_point = carla.Transform(carla.Location(x=-25.19, y=139, z=0), carla.Rotation(yaw=0))
target_location = carla.Location(x=44, y=139, z=0)

# Cleanup function to remove active actors
def cleanUp():
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")

# Define a function to calculate the distance between two points
def calculate_distance(location1, location2):
    return location1.distance(location2)

try:
    # Spawn the vehicle
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    actor_list.append(vehicle)
    print(f"Vehicle spawned at location: {spawn_point.location}")

    # Add a GNSS sensor to the vehicle (biased)
    biased_gnss_transform = carla.Transform(carla.Location(x=0, y=0, z=2))
    biased_gnss_sensor = world.spawn_actor(gnss_bp, biased_gnss_transform, attach_to=vehicle)
    actor_list.append(biased_gnss_sensor)

    # Add a second GNSS sensor for unbiased data
    unbiased_gnss_transform = carla.Transform(carla.Location(x=0, y=0, z=2))
    unbiased_gnss_sensor = world.spawn_actor(gnss_bp, unbiased_gnss_transform, attach_to=vehicle)
    actor_list.append(unbiased_gnss_sensor)

    # Initialize GNSS data storage
    gnss_data = {"latitude": None, "longitude": None, "altitude": None}
    unbiased_gnss_data = {"latitude": None, "longitude": None, "altitude": None}

    # Bias parameters
    start_time = time.time()
    max_noise_std_dev = 0.00001
    max_bias = -0.00005
    startup = 3

    # Callback for biased GNSS data
    def biased_gnss_callback(gnss):
        current_time = time.time()
        elapsed_time = current_time - start_time

        # Add Gaussian noise and bias
        if elapsed_time > startup:
            noise_std_dev = ((elapsed_time - startup) / 20.0) * max_noise_std_dev
            bias = ((elapsed_time - startup) / 20.0) * max_bias
        else:
            noise_std_dev = 0.0
            bias = 0.0

        gnss_data["latitude"] = gnss.latitude + bias
        gnss_data["longitude"] = gnss.longitude
        gnss_data["altitude"] = gnss.altitude

    # Callback for unbiased GNSS data
    def unbiased_gnss_callback(gnss):
        unbiased_gnss_data["latitude"] = gnss.latitude
        unbiased_gnss_data["longitude"] = gnss.longitude
        unbiased_gnss_data["altitude"] = gnss.altitude

    # Subscribe to GNSS data
    biased_gnss_sensor.listen(biased_gnss_callback)
    unbiased_gnss_sensor.listen(unbiased_gnss_callback)

    # Initialize reverse engineering data
    reverse_engineered_path = []
    gps_bias_detected = []

    reached_target = False
    while True:
        if (gnss_data["latitude"] is not None and 
            gnss_data["longitude"] is not None and 
            unbiased_gnss_data["latitude"] is not None and 
            unbiased_gnss_data["longitude"] is not None):
            
            # Calculate bias
            lat_bias = gnss_data["latitude"] - unbiased_gnss_data["latitude"]
            lon_bias = gnss_data["longitude"] - unbiased_gnss_data["longitude"]
            gps_bias_detected.append((lat_bias, lon_bias))
            
            # Reverse engineer the true location
            reverse_engineered_latitude = gnss_data["latitude"] - lat_bias
            reverse_engineered_longitude = gnss_data["longitude"] - lon_bias
            reverse_engineered_path.append((reverse_engineered_latitude, reverse_engineered_longitude))

            # Validation
            true_latitude = unbiased_gnss_data["latitude"]
            true_longitude = unbiased_gnss_data["longitude"]
            reverse_engineering_error = math.sqrt(
                (reverse_engineered_latitude - true_latitude)**2 + 
                (reverse_engineered_longitude - true_longitude)**2
            )

            print(f"Bias Detected -> Latitude: {lat_bias}, Longitude: {lon_bias}")
            print(f"Reverse Engineered Location: Lat={reverse_engineered_latitude}, Lon={reverse_engineered_longitude}")
            print(f"Reverse Engineering Error: {reverse_engineering_error} meters")

            # Vehicle movement and stopping logic
            ref_location = carla.Location(x=-25.19, y=139, z=0)
            ref_geolocation = world.get_map().transform_to_geolocation(ref_location)
            lat_diff = gnss_data["latitude"] - ref_geolocation.latitude
            lon_diff = gnss_data["longitude"] - ref_geolocation.longitude
            approx_x = ref_location.x + lon_diff * 111320 * math.cos(ref_geolocation.latitude * math.pi / 180)
            approx_y = ref_location.y + lat_diff * 111320
            current_location = carla.Location(x=approx_x, y=approx_y)
            distance = calculate_distance(current_location, target_location)
            print(f"Distance to target: {distance} meters")

            if distance < 10.0 and not reached_target:
                reached_target = True
                print("Approaching target! Slowing down.")

            if reached_target:
                throttle = max(0.0, 0.7 * (distance / 10.0))
                brake = min(1.0, 1.0 - throttle)
                vehicle.apply_control(carla.VehicleControl(throttle=throttle, brake=brake))
                if distance < 2:
                    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                    print("Vehicle stopped.")
                    break
            else:
                vehicle.apply_control(carla.VehicleControl(throttle=0.7))

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
