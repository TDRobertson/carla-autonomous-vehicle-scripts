# This file serves as a starting template for other scripts.
import glob
import os
import sys
import random
import time
import numpy as np
import cv2

# Add the carla egg file to the python path
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Set the image width and height for use with sensor.camera.rgb
im_width = 640
im_height = 480

# Set the carla client and world
# # Set Render Mode for the World
# settings = carla.WorldSettings()
# settings = carla.get_settings()
# # Do not render the world
# #settings.no_rendering_mode = False
# # Render the world offscreen
# settings.offscreen = True
# # Set the fixed delta seconds for the world
# settings.fixed_delta_seconds = 0.05
# # Set the synchronous mode for the world
# settings.synchronous_mode = True
# # Apply the settings to the world
# carla.WorldSettings.apply_settings(settings)


# Function to get the host starting with command-line argument, then environment variable, then default to localhost
def get_host():
    # Check for command-line argument
    if len(sys.argv) > 1:
        return sys.argv[1]
    # Check for windows_host environment variable
    elif 'windows_host' in os.environ:
        return os.environ['windows_host']
    # Default to localhost
    else:
        return "localhost"
    
# Function to process the raw image data for the neural network
def process_img(image):
    i = np.array(image.raw_data)
    # Convert the image raw data to a numpy iterable
    # i1 = np.fromiter(image.raw_data, dtype=np.dtype('uint8'))
    # reshape the image to im_height x im_width x 4 using 4 for rgba
    i2 = i.reshape((im_height, im_width, 4))
    # remove the alpha value (basically, remove the 4th element)
    # Caputes all width, height, and rgb values
    i3 = i2[:, :, :3]
    # Display the image
    cv2.imshow("", i3)
    # Wait for 1ms
    cv2.waitKey(1)
    # Normalize the image and return it, neural networks want input values ranging from -1 to 1
    return i3/255.0

# Function to clean up all actors
def cleanUp():
    for actor in actor_list:
        actor.destroy()
    print("All cleaned up!")

actor_list = []

try:
    host = get_host()
    client = carla.Client(host, 2000)
    client.set_timeout(2.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Loads the Tesla Model 3 blueprint
    ego_bp = blueprint_library.filter("model3")[0]
    ego_bp.set_attribute("role_name", "hero")
    

    # Sets the Tesla Model 3 spawn point at the origin
    # spawn_point = carla.Transform(carla.Location(x=0, y=0, z=0), carla.Rotation())

    # Sets the Tesla Model 3 spawn point at a random location
    spawn_point = random.choice(world.get_map().get_spawn_points())
    
    # Spawn the Tesla Model 3 as the ego vehicle
    ego_vehicle = world.spawn_actor(ego_bp, spawn_point)
    # apply vehicle control to go in a straight line
    ego_vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    actor_list.append(ego_vehicle)


    # Create a camera sensor
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    # Set the image size and field of view for the camera sensor
    camera_bp.set_attribute("image_size_x", f"{im_width}")
    camera_bp.set_attribute("image_size_y", f"{im_height}")
    camera_bp.set_attribute("fov", "110")
    # Attach camera sensor to the ego vehicle
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))
    camera_sensor = world.spawn_actor(camera_bp, spawn_point, attach_to=ego_vehicle)
    actor_list.append(camera_sensor)
    # camera_sensor.listen(lambda data: data.save_to_disk("output/%.6d.png" % data.frame))
    camera_sensor.listen(lambda image: image.save_to_disk('output/%06d.png' % image.frame))

    # # Spawn other vehicle actors
    # for _ in range(10):
    #     bp = random.choice(blueprint_library.filter("vehicle"))
    #     spawn_point = random.choice(world.get_map().get_spawn_points())
    #     actor = world.spawn_actor(bp, spawn_point)
    #     actor_list.append(actor)
    #     # set autopilot for the actor, note its rule based environment
    #     actor.set_autopilot(True)

    # Run for 10 seconds
    time.sleep(10)
finally:
    # Clean up all actors
    cleanUp()