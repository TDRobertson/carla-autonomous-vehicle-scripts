import carla
import time
import math
import random

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

# Function to add Gaussian noise to GNSS readings
def add_gaussian_noise(location, stddev=4.0):   # Default of 1 meter standard deviation doesn't go outside target bounds, demonstration purposes
    noisy_location = carla.Location(
        x=location.x + random.gauss(0, stddev),
        y=location.y + random.gauss(0, stddev),
        z=location.z  # Assuming no noise in the z-axis for simplicity
    )
    return noisy_location

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

world = client.get_world()

# Set up the blueprint library
blueprint_library = world.get_blueprint_library()

# Choose a vehicle blueprint
vehicle_bp = blueprint_library.filter('vehicle.*')[0]

# Define the spawn point
spawn_point = carla.Transform(carla.Location(x=-25.19, y=139, z=0), carla.Rotation(yaw=0))
target_location = carla.Location(x=44, y=139, z=0)

try:
    # Draw circles at the start and target locations
    draw_circle(world, spawn_point.location)
    draw_circle(world, target_location, color=carla.Color(r=0, g=255, b=0))

    # Spawn the vehicle
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)
    if vehicle is not None:
        print(f"Vehicle spawned at location: {spawn_point.location}")
    else:
        print("Failed to spawn the vehicle.")

    # Control the vehicle to drive forward
    vehicle.apply_control(carla.VehicleControl(throttle=0.7))
    print("Vehicle is moving forward.")

    # Main loop to drive the vehicle
    try:
        while True:
            # Get the current location of the vehicle
            current_location = vehicle.get_location()
            print(f"Current vehicle location: {current_location}")

            # Add Gaussian noise to the current GNSS location
            noisy_location = add_gaussian_noise(current_location)
            print(f"Noisy vehicle location: {noisy_location}")

            # Calculate the distance to the target location using noisy GNSS data
            distance = calculate_distance(noisy_location, target_location)
            print(f"Distance to target (with noise): {distance} meters")

            # Check if the vehicle has reached the target location
            if distance < 1.0:  # Distance threshold in meters
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                print("Target reached! Stopping and despawning the vehicle.")
                time.sleep(1)  # Give some time for the vehicle to fully stop
                break

            time.sleep(0.1)

    finally:
        # Clean up
        vehicle.destroy()

except Exception as e:
    print(f"An error occurred: {e}")
