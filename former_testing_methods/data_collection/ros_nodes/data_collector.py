#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, NavSatFix
import cv2
import numpy as np
import os
import csv
import struct
from datetime import datetime
import time
from rosgraph_msgs.msg import Clock


class DataCollector(Node):
    def __init__(self):
        super().__init__('data_collector')

        self.clock_subscription = self.create_subscription(Clock, "/clock", self.clock_callback, 10)
        self.spawn_time = 20.0  # Time in seconds when the vehicle should be spawned
        self.spawned = False  # Flag to ensure the vehicle is spawned only once
        


    def clock_callback(self, msg):
        current_time = msg.clock.sec + msg.clock.nanosec * 1e-9
        if current_time >= self.spawn_time and not self.spawned:
            # Set capture frequencies
            self.capture_image_every_n = 20  # Capture an image every N messages
            self.capture_lidar_every_n = 20   # Capture LiDAR data every N messages
            self.capture_gnss_every_n = 4    # Capture GNSS data every N messages

            # Create a timestamped directory to store recordings
            timestamp = datetime.now().strftime("D%m-%d-%Y_T%H-%M-%S")
            self.directory = f'recordings/carla_recordings_{timestamp}_C{self.capture_image_every_n}n_L{self.capture_lidar_every_n}n_G{self.capture_gnss_every_n}n'
            os.makedirs(self.directory, exist_ok=True)

            # Create directories to store data
            self.image_dir = os.path.join(self.directory, 'camera_images')
            os.makedirs(self.image_dir, exist_ok=True)

            # CSV files to store metadata
            self.camera_csv_file = open(os.path.join(self.directory, 'camera_data.csv'), 'w', newline='')
            self.camera_csv_writer = csv.writer(self.camera_csv_file)
            self.camera_csv_writer.writerow(['timestamp_sec', 'image_path'])

            self.lidar_csv_file = open(os.path.join(self.directory, 'lidar_data.csv'), 'w', newline='')
            self.lidar_csv_writer = csv.writer(self.lidar_csv_file)
            self.lidar_csv_writer.writerow(['timestamp_sec', 'x', 'y', 'z', 'intensity'])

            self.gnss_csv_file = open(os.path.join(self.directory, 'gnss_data.csv'), 'w', newline='')
            self.gnss_csv_writer = csv.writer(self.gnss_csv_file)
            self.gnss_csv_writer.writerow([
                'timestamp_sec', 'latitude', 'longitude', 'altitude'])

            # Counters to control data capture frequency
            self.image_counter = 0
            self.lidar_counter = 0
            self.gnss_counter = 0

            # Create subscriptions for Camera, LiDAR, and GNSS topics
            self.camera_sub = self.create_subscription(
                Image, '/carla/ego_vehicle/rgb_front/image', self.camera_callback, 10)
            self.lidar_sub = self.create_subscription(
                PointCloud2, '/carla/ego_vehicle/lidar', self.lidar_callback, 10)
            self.gnss_sub = self.create_subscription(
                NavSatFix, '/carla/ego_vehicle/gnss', self.gnss_callback, 10)
            
            # Create README file to explain capture rates
            self.create_readme()
            self.spawned = True

    def create_readme(self):
        readme_path = os.path.join(self.directory, 'README.txt')
        with open(readme_path, 'w') as readme:
            readme.write(f'Capture rate:\n')
            readme.write(f'  - Image: Every {self.capture_image_every_n} messages\n')
            readme.write(f'  - LiDAR: Every {self.capture_lidar_every_n} messages\n')
            readme.write(f'  - GNSS: Every {self.capture_gnss_every_n} messages\n\n')
            readme.write('Data is captured every N messages, where N is the capture rate specified above.\n')
            readme.write(f'For example:\n')
            readme.write(f'  - If the capture rate is {self.capture_image_every_n} messages for images, an image is captured every {self.capture_image_every_n}/20 seconds = {self.capture_image_every_n / 20:.2f} seconds (at 20 FPS).\n')
            readme.write(f'  - If the capture rate is {self.capture_lidar_every_n} messages for LiDAR, data is captured every {self.capture_lidar_every_n}/20 seconds = {self.capture_lidar_every_n / 20:.2f} seconds (at 20 FPS).\n')
            readme.write(f'  - If the capture rate is {self.capture_gnss_every_n} messages for GNSS, data is captured every {self.capture_gnss_every_n}/20 seconds = {self.capture_gnss_every_n / 20:.2f} seconds (at 20 FPS).\n')

    def camera_callback(self, msg):
        self.image_counter += 1
        if self.image_counter % self.capture_image_every_n != 0:
            return

        # Convert ROS Image message to OpenCV image
        if msg.encoding == 'rgb8':
            cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        elif msg.encoding == 'bgr8':
            cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        elif msg.encoding == 'bgra8':
            cv_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 4)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGRA2BGR)
        else:
            self.get_logger().error(f'Unsupported encoding: {msg.encoding}')
            return

        # Generate image filename
        timestamp_sec = msg.header.stamp.sec
        image_filename = f'image_{timestamp_sec}.png'
        image_path = os.path.join(self.image_dir, image_filename)

        # Save image using OpenCV
        cv2.imwrite(image_path, cv_image)

        # Log metadata in CSV file
        self.camera_csv_writer.writerow([timestamp_sec, image_path])
        self.get_logger().info(f'Collected and saved image: {image_filename}')

    def lidar_callback(self, msg):
        self.lidar_counter += 1
        if self.lidar_counter % self.capture_lidar_every_n != 0:
            return

        # Extract data from the PointCloud2 message
        data = self.parse_pointcloud2(msg)
        timestamp_sec = msg.header.stamp.sec
        for point in data:
            self.lidar_csv_writer.writerow([
                timestamp_sec, point['x'], point['y'], point['z'], point['intensity']
            ])
        self.get_logger().info(f'Collected {len(data)} LiDAR points')

    def parse_pointcloud2(self, msg):
        fmt = 'ffff'  # Format for each point (x, y, z, intensity)
        width, height = msg.width, msg.height
        point_step = msg.point_step
        row_step = msg.row_step
        points = []

        for i in range(height):
            for j in range(width):
                offset = i * row_step + j * point_step
                x, y, z, intensity = struct.unpack_from(fmt, msg.data, offset)
                points.append({'x': x, 'y':y, 'z': z, 'intensity': intensity})

        return points

    def gnss_callback(self, msg):
        self.gnss_counter += 1
        if self.gnss_counter % self.capture_gnss_every_n != 0:
            return

        # Extract data from the GNSS message
        data = {
            'timestamp_sec': msg.header.stamp.sec,
            'latitude': msg.latitude,
            'longitude': msg.longitude,
            'altitude': msg.altitude,
        }

        # Log metadata in CSV file
        self.gnss_csv_writer.writerow([
            data['timestamp_sec'], data['latitude'],
            data['longitude'], data['altitude']
        ])
        self.get_logger().info(f'Collected GNSS data point: {data}')

    def save_data(self):
        self.camera_csv_file.close()
        self.lidar_csv_file.close()
        self.gnss_csv_file.close()

def main(args=None):
    rclpy.init(args=args)
    node = DataCollector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_data()
        node.get_logger().info('Data saved to CSV files')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
