# Description: This script subscribes to the camera, LiDAR, and GNSS topics in the Carla simulator and saves the data to CSV files while applying voxel grid downscaling to reduce
# the number of LiDAR points. The script saves images from the camera topic, LiDAR points, and GNSS data to separate CSV files. The script also creates a README file to explain the
# capture rates of the data. The voxel size can be adjusted to control the downsampling rate.

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
import open3d as o3d

class DataCollector(Node):
    def __init__(self, image_capture_rate=10, gnss_capture_rate=2, lidar_capture_rate=50, voxel_size=0.1):
        super().__init__('data_collector')

        # Set capture frequencies
        self.capture_image_every_n = image_capture_rate  # Capture an image every N messages
        self.capture_lidar_every_n = lidar_capture_rate   # Capture LiDAR data every N messages
        self.capture_gnss_every_n = gnss_capture_rate    # Capture GNSS data every N messages
        self.voxel_size = voxel_size

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
        self.camera_csv_writer.writerow(['timestamp_sec', 'timestamp_nanosec', 'image_path'])

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
            readme.write(f'  - Voxel size for LiDAR downsampling: {self.voxel_size}\n')


    # Downsample point cloud using Open3D voxel grid downsampling
    def voxel_grid_downsample(self, points, voxel_size=0.05):
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points[:, :3]) # Extract only x, y, z coordinates for downscaled point cloud
        downsampled_pc = point_cloud.voxel_down_sample(voxel_size=voxel_size)
        downsampled_points = np.asarray(downsampled_pc.points)

        # Add intensity values to the downsampled point cloud
        intensity_values = []
        # Find the points in the original point cloud that fall into the same voxel
        for p in downsampled_points:
            indices = np.all(np.isclose(points[:, :3], p), atol=voxel_size/2, axis=1)
            if np.any(indices):
                intensities = points[indices, 3] # Intensity values of all points in the voxel
                avg_intensity = np.mean(intensities) # Average intensity of all points in the voxel
                intensity_values.append([p[0], p[1], p[2], avg_intensity])

        return intensity_values
    

    # Callback functions to capture data
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
        #self.get_logger().info(f'Collected and saved image: {image_filename}')

    def lidar_callback(self, msg):
        self.lidar_counter += 1
        if self.lidar_counter % self.capture_lidar_every_n != 0:
            return

        # Extract data from the PointCloud2 message
        data = self.parse_pointcloud2(msg)

        # Convert data to NumPy array for processing
        points = np.array([[p['x'], p['y'], p['z'], p['intensity']] for p in data])

        # Downsample the point cloud
        downsampled_points = self.voxel_grid_downsample(points, self.voxel_size)

        timestamp_sec = msg.header.stamp.sec
        for point in downsampled_points:
            self.lidar_csv_writer.writerow([
                timestamp_sec, point[0], point[1], point[2], point[3]])
        # self.get_logger().info(f'Collected {len(downsampled_points)} LiDAR points')

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
            data['timestamp_sec'], data['latitude'], data['longitude'], data['altitude']])
        # self.get_logger().info(f'Collected GNSS data point: {data}')

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
