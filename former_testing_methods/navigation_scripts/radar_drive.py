import carla
import time
import math

# Function to calculate the distance between two points
def calculate_distance(location1, location2):
    return location1.distance(location2)


# Similar spawn function that should make the Chevy Impala
def spawn_obstacle_vehicle(world, blueprint_library, location):
    vehicle_bp = blueprint_library.find('vehicle.chevrolet.impala')  # Specify Chevy Impala
    spawn_transform = carla.Transform(location, carla.Rotation(yaw=0))

    # Visualize the intended spawn point
    world.debug.draw_point(location, size=0.5, color=carla.Color(r=255, g=0, b=0), life_time=10.0)

    obstacle_vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
    if obstacle_vehicle:
        print(f"Obstacle vehicle (Chevy Impala) spawned at {location}")
    else:
        print("Failed to spawn obstacle vehicle. Check spawn location or blueprint.")
    return obstacle_vehicle

# # Function to spawn an additional vehicle THIS FUNCTION WORKS
# def spawn_obstacle_vehicle(world, blueprint_library, location):
#     vehicle_bp = blueprint_library.filter('vehicle.*')[1]  # Use a different blueprint for the obstacle vehicle
#     spawn_transform = carla.Transform(location, carla.Rotation(yaw=0))
#     obstacle_vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
#     if obstacle_vehicle:
#         print(f"Obstacle vehicle spawned at {location}")
#     return obstacle_vehicle

# Function to add a radar sensor
def attach_radar(vehicle, world):
    radar_bp = world.get_blueprint_library().find('sensor.other.radar')

    # Modify radar attributes
    radar_bp.set_attribute('horizontal_fov', '8')  # Set horizontal FOV
    radar_bp.set_attribute('vertical_fov', '0.5')    # Set Vertical FOV 
    radar_bp.set_attribute('range', '15')           # Set radar range in meters

    radar_location = carla.Location(x=2.5, z=1.0)  # Attach radar at the front of the vehicle
    radar_transform = carla.Transform(radar_location)
    radar = world.spawn_actor(radar_bp, radar_transform, attach_to=vehicle)

    # Listen to radar data
    radar.listen(lambda data: process_radar_data(data, vehicle))
    return radar


# # Process radar data to detect objects
# def process_radar_data(data, vehicle):
#     for detection in data:
#         if detection.depth < 10.0:  # Object within 10 meters
#             vehicle.set_autopilot(False)
#             vehicle.apply_control(carla.VehicleControl(throttle=0.2, brake=0.8))
#             return
        
def process_radar_data(data, vehicle):
    world = vehicle.get_world()

    # Iterate over each detection
    for detection in data:
        # Calculate the relative position using azimuth and altitude
        relative_x = detection.depth * math.cos(detection.azimuth) * math.cos(detection.altitude)
        relative_y = detection.depth * math.sin(detection.azimuth) * math.cos(detection.altitude)
        relative_z = detection.depth * math.sin(detection.altitude)

        # Get the world position relative to the vehicle
        start_location = vehicle.get_transform().location
        end_location = start_location + carla.Location(x=relative_x, y=relative_y, z=relative_z)

        # Draw a line from the vehicle to the detected point
        world.debug.draw_line(
            start_location,
            end_location,
            thickness=0.1,
            color=carla.Color(r=0, g=255, b=0),
            life_time=0.1
        )

        # Optionally draw points at the detected location
        world.debug.draw_point(
            end_location,
            size=0.2,
            color=carla.Color(r=255, g=0, b=0),
            life_time=0.1
        )
        print(f"Object detected at {detection.depth:.2f} meters.")

        # if detection.depth < 10.0:  # Object within 10 meters
        #     vehicle.set_autopilot(False)
        #     vehicle.apply_control(carla.VehicleControl(throttle=0.2, brake=1.0))
        # return



# Connect to CARLA
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# Set up the blueprint library
blueprint_library = world.get_blueprint_library()

# Main vehicle spawn setup
vehicle_bp = blueprint_library.filter('vehicle.*')[0]
spawn_point = carla.Transform(carla.Location(x=-25.19, y=139, z=0), carla.Rotation(yaw=0))
target_location = carla.Location(x=44, y=139, z=0)

# Spawn additional vehicle 8 meters in front of target location
obstacle_location = carla.Location(x=target_location.x - 8, y=target_location.y, z=0)
obstacle_vehicle = spawn_obstacle_vehicle(world, blueprint_library, obstacle_location)

try:
    # Spawn the vehicle
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    radar = attach_radar(vehicle, world)  # Attach radar to vehicle

    vehicle.apply_control(carla.VehicleControl(throttle=0.6))
    print("Vehicle is moving forward.")

    start_time = None
    stopped = False

    while True:
        current_location = vehicle.get_location()
        distance_to_target = calculate_distance(current_location, target_location)

        if not stopped and obstacle_vehicle:  # Ensure obstacle vehicle exists
            obstacle_current_location = obstacle_vehicle.get_location()  # Dynamic position
            distance_to_obstacle = calculate_distance(current_location, obstacle_current_location)

            if distance_to_obstacle < 10.0:
                print(f"Approaching obstacle! Distance: {distance_to_obstacle:.2f} meters. Slowing down.")
                vehicle.apply_control(carla.VehicleControl(throttle=0.1, brake=1.0))

        # Logic for stopping at the target
        if vehicle.get_velocity().length() < 0.1:
            if start_time is None:
                start_time = time.time()
            elif time.time() - start_time > 5:  # Wait for 5 seconds after stopping
                print("Vehicle has stopped for 5 seconds. Despawning...")
                break

        time.sleep(0.1)


finally:
    vehicle.destroy()
    if obstacle_vehicle:
        obstacle_vehicle.destroy()
    if radar:
        radar.destroy()
