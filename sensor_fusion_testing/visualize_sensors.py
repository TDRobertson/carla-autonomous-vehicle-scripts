# Author: Shing-Yan Loo (yan99033@gmail.com) - Modified by me
# This script spawns a car at a random spawn point in the map
# and then visualize the sensor signals
# The list of sensors used are as follows:
# - camera (only for visualization, not being used for localization)
# - IMU
# - GNSS

import glob
import os
import sys
import signal
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import random
import cv2
import time

from multiprocessing import Queue, Value, Process
from ctypes import c_bool

from car import Car
from visualizer import visualizer

from util import destroy_queue

def signal_handler(signum, frame):
    print("\nReceived interrupt signal. Cleaning up...")
    if 'quit' in globals():
        quit.value = True
    sys.exit(0)

def main():
    # Initialize quit variable before try block
    global quit
    quit = Value(c_bool, False)
    visual_msg_queue = None
    proc = None
    car = None

    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)

    try:
        print("Attempting to connect to CARLA server...")
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)  # Increased timeout to 10 seconds for debugging

        # Try to connect to the server
        try:
            print("Getting world...")
            world = client.get_world()
            print("Successfully connected to CARLA server!")
            print(f"Current map: {world.get_map().name}")
        except RuntimeError as e:
            print("\nError: Could not connect to CARLA server.")
            print("Please check the following:")
            print("1. Make sure CARLAUE4.exe is running")
            print("2. Check if you can see the CARLA window with the 3D environment")
            print("3. Try restarting CARLAUE4.exe")
            print("4. Check if port 2000 is not blocked by firewall")
            print(f"\nDetailed error: {str(e)}")
            return
        except Exception as e:
            print(f"\nUnexpected error while connecting to CARLA: {str(e)}")
            return

        spawn_point = random.choice(world.get_map().get_spawn_points())

        # Create a car object
        car = Car(world, client, spawn_point)
        print('created a car object')

        # Visualizer
        visual_msg_queue = Queue()
        proc = Process(target=visualizer, args=(visual_msg_queue, quit))
        proc.daemon = True
        proc.start()

        # In case Matplotlib is not able to keep up the pace of the growing queue,
        # we have to limit the rate of the items being pushed into the queue
        visual_fps = 3
        last_ts = time.time()

        try:
            # Drive the car around and get sensor readings
            while True:
                world.tick()
                frame = world.get_snapshot().frame

                # Get sensor readings
                sensors = car.get_sensor_readings(frame)

                # get image
                if sensors['image'] is not None: 
                    image = sensors['image']
                    cv2.imshow('image', image)
                    cv2.waitKey(1)  
                
                # Limit the frame-rate
                if time.time() - last_ts < 1. / visual_fps:
                    continue
                
                # timestamp for inserting a new item into the queue
                last_ts = time.time()

                # visual message
                visual_msg = dict()

                # Get ground truth vehicle 
                gt_location = car.get_location()
                visual_msg['gt_traj'] = [gt_location.x, gt_location.y, gt_location.z]

                # Get imu reading
                if sensors['imu'] is not None:
                    imu = sensors['imu']
                    accelero = imu.accelerometer
                    gyroscop = imu.gyroscope
                    visual_msg['imu'] = [accelero.x, accelero.y, accelero.z, 
                                         gyroscop.x, gyroscop.y, gyroscop.z]

                # Get gps reading
                if sensors['gnss'] is not None:
                    gnss = sensors['gnss']
                    visual_msg['gnss'] = [gnss.x, gnss.y, gnss.z]            

                visual_msg_queue.put(visual_msg)

        except KeyboardInterrupt:
            print("\nReceived keyboard interrupt. Cleaning up...")

    finally:
        print('Exiting visualizer')
        if quit is not None:
            quit.value = True
        if visual_msg_queue is not None:
            destroy_queue(visual_msg_queue)
        if proc is not None:
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.terminate()

        if car is not None:
            print('destroying the car object')
            car.destroy()

        cv2.destroyAllWindows()
        print('done')

if __name__ == '__main__':
    main()