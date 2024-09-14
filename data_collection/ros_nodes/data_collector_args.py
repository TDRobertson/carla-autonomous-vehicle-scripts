#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, NavSatFix
import cv2
import numpy as np
import os
import csv
import struct
import argparse

class DataCollector(Node):
    def __init__(self, capture_camera, capture_lidar, capture_gnss):
        super().__init__('data_collector')

        self.capture_camera = capture_camera
        self.capture_lidar = capture_lidar
        self.capture_gnss = capture_gnss

        # Create subscriptions for Camera, LiDAR, and GNSS topics
        if self.capture_camera:
            self.camera_sub = self.create_subscription(
                Image, '/carla/ego_vehicle/rgb_front/image', self.camera_callback, 10)
            self.image_dir = 'camera_images'
            os.makedirs(self.image_dir, exist_ok=True)
            self.camera_csv_file = open('camera_data.csv', 'w', newline='')
            self.camera_csv_writer = csv.writer(self.camera_csv_file)
            self.camera_csv_writer.writerow(['timestamp_sec', 'timestamp_nanosec', 'image_path'])
            self.image_counter = 0
            self.capture_image_every_n = 10  # Capture an image every N messages

        if self.capture_lidar:
            self.lidar_sub = self.create_subscription(
                PointCloud2, '/carla/ego_vehicle/lidar', self.lidar_callback, 10)
            self.lidar_csv_file = open('lidar_data.csv', 'w', newline='')
            self.lidar_csv_writer = csv.writer(self.lidar_csv_file)
            self.lidar_csv_writer.writerow(['timestamp_sec', 'timestamp_nanosec', 'x', 'y', 'z'])
            self.lidar_counter = 0
            self.capture_lidar_every_n = 5   # Capture LiDAR data every N messages

        if self.capture_gnss:
            self.gnss_sub = self.create_subscription(
                NavSatFix, '/carla/ego_vehicle/gnss', self.gnss_callback, 10)
            self.gnss_csv_file = open('gnss_data.csv', 'w', newline='')
            self.gnss_csv_writer = csv.writer(self.gnss_csv_file)
            self.gnss_csv_writer.writerow([
                'timestamp_sec', 'timestamp_nanosec', 'latitude', 'longitude', 'altitude',
                'position_covariance', 'position_covariance_type'
            ])
            self.gnss_counter = 0
            self.capture_gnss_every_n = 2    # Capture GNSS data every N messages

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
        timestamp_nanosec = msg.header.stamp.nanosec
        image_filename = f'image_{timestamp_sec}_{timestamp_nanosec}.png'
        image_path = os.path.join(self.image_dir, image_filename)

        # Save image using OpenCV
        cv2.imwrite(image_path, cv_image)

        # Log metadata in CSV file
        self.camera_csv_writer.writerow([timestamp_sec, timestamp_nanosec, image_path])
        self.get_logger().info(f'Collected and saved image: {image_filename}')

    def lidar_callback(self, msg):
        self.lidar_counter += 1
        if self.lidar_counter % self.capture_lidar_every_n != 0:
            return

        # Extract data from the PointCloud2 message
        data = self.parse_pointcloud2(msg)
        timestamp_sec = msg.header.stamp.sec
        timestamp_nanosec = msg.header.stamp.nanosec
        for point in data:
            self.lidar_csv_writer.writerow([
                timestamp_sec, timestamp_nanosec, point['x'], point['y'], point['z']
            ])
        self.get_logger().info(f'Collected {len(data)} LiDAR points')

    def parse_pointcloud2(self, msg):
        fmt = 'fff'  # Format for each point (x, y, z)
        width, height = msg.width, msg.height
        point_step = msg.point_step
        row_step = msg.row_step
        points = []

        for i in range(height):
            for j in range(width):
                offset = i * row_step + j * point_step
                x, y, z = struct.unpack_from(fmt, msg.data, offset)
                points.append({'x': x, 'y': y, 'z': z})

        return points

    def gnss_callback(self, msg):
        self.gnss_counter += 1
        if self.gnss_counter % self.capture_gnss_every_n != 0:
            return

        # Extract data from the GNSS message
        data = {
            'timestamp_sec': msg.header.stamp.sec,
            'timestamp_nanosec': msg.header.stamp.nanosec,
            'latitude': msg.latitude,
            'longitude': msg.longitude,
            'altitude': msg.altitude,
            'position_covariance': msg.position_covariance,
            'position_covariance_type': msg.position_covariance_type
        }

        # Log metadata in CSV file
        self.gnss_csv_writer.writerow([
            data['timestamp_sec'], data['timestamp_nanosec'], data['latitude'],
            data['longitude'], data['altitude'], data['position_covariance'],
            data['position_covariance_type']
        ])
        self.get_logger().info(f'Collected GNSS data point: {data}')

    def save_data(self):
        if self.capture_camera:
            self.camera_csv_file.close()
        if self.capture_lidar:
            self.lidar_csv_file.close()
        if self.capture_gnss:
            self.gnss_csv_file.close()

def main(args=None):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Capture data from CARLA simulator')
    parser.add_argument('--camera', action='store_true', help='Capture camera data')
    parser.add_argument('--lidar', action='store_true', help='Capture LiDAR data')
    parser.add_argument('--gnss', action='store_true', help='Capture GNSS data')
    args = parser.parse_args()

    # Default to capturing all data if no arguments are given
    capture_camera = args.camera or not (args.camera or args.lidar or args.gnss)
    capture_lidar = args.lidar or not (args.camera or args.lidar or args.gnss)
    capture_gnss = args.gnss or not (args.camera or args.lidar or args.gnss)

    rclpy.init(args=None)
    node = DataCollector(capture_camera, capture_lidar, capture_gnss)

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
