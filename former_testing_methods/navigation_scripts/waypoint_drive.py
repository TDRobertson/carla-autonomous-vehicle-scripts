'''
A simple script designed to spawn a vehicle and drive it to a target location.
The vehicle will gradually slow down as it approaches the target location, and the vehicle
will despawn once it reaches the target destination.

The purpose of the script was to quickly set up an environment for the gnss_drive.py script
using carlas built in xyz coordinate system to test vehicle positioning before adding
in a gps approximation system.
'''
import carla
import time
import math

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
    reached_target = False
    try:
        while True:
            # Get the current location of the vehicle
            current_location = vehicle.get_location()
            print(f"Current vehicle location: {current_location}")

            # Calculate the distance to the target location
            distance = calculate_distance(current_location, target_location)
            print(f"Distance to target: {distance} meters")

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

except Exception as e:
    print(f"An error occurred: {e}")
