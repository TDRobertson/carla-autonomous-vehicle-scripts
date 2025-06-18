import pygame
import numpy as np
import math
import time
import sys
import glob
import os
from pygame.locals import RESIZABLE, VIDEORESIZE
from scipy.spatial.transform import Rotation

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

class IMUIntegrator:
    def __init__(self):
        # State vector: [position, velocity, orientation]
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.orientation = np.array([0, 0, 0, 1])  # quaternion [x,y,z,w]
        
        # Covariance matrix
        self.P = np.eye(9) * 0.1
        
        # Process noise
        self.Q = np.eye(9) * 0.01
        
        # Measurement noise
        self.R = np.eye(3) * 0.1
        
        # Last timestamp
        self.last_timestamp = None
        
        # Gravity vector
        self.g = np.array([0, 0, -9.81])

    def predict(self, imu_data, timestamp):
        if self.last_timestamp is None:
            self.last_timestamp = timestamp
            return
            
        # Time delta
        dt = timestamp - self.last_timestamp
        self.last_timestamp = timestamp
        
        # Get IMU measurements
        acc = np.array([imu_data.accelerometer.x, 
                       imu_data.accelerometer.y, 
                       imu_data.accelerometer.z])
        gyro = np.array([imu_data.gyroscope.x,
                        imu_data.gyroscope.y,
                        imu_data.gyroscope.z])
        
        # Convert quaternion to rotation matrix
        R = Rotation.from_quat(self.orientation).as_matrix()
        
        # Rotate acceleration to world frame and remove gravity
        acc_world = R @ acc + self.g
        
        # Update state
        self.position += self.velocity * dt + 0.5 * acc_world * dt**2
        self.velocity += acc_world * dt
        
        # Update orientation using gyroscope
        angle = gyro * dt
        dq = Rotation.from_rotvec(angle).as_quat()
        self.orientation = Rotation.from_quat(self.orientation).as_quat()
        self.orientation = (Rotation.from_quat(dq) * Rotation.from_quat(self.orientation)).as_quat()
        
        # Predict covariance
        F = self._get_state_transition_matrix(dt, R, acc)
        self.P = F @ self.P @ F.T + self.Q

    def update_with_gps(self, gps_pos):
        """Correct state using GPS measurement"""
        # Measurement matrix (we only observe position)
        H = np.zeros((3, 9))
        H[:3, :3] = np.eye(3)
        
        # Kalman gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Innovation
        innovation = gps_pos - self.position
        
        # Update state
        state_update = K @ innovation
        self.position += state_update[:3]
        self.velocity += state_update[3:6]
        
        # Update orientation using small angle approximation
        angle_update = state_update[6:]
        dq = Rotation.from_rotvec(angle_update).as_quat()
        self.orientation = (Rotation.from_quat(dq) * Rotation.from_quat(self.orientation)).as_quat()
        
        # Update covariance
        self.P = (np.eye(9) - K @ H) @ self.P

    def _get_state_transition_matrix(self, dt, R, acc):
        """Calculate the state transition matrix F"""
        F = np.eye(9)
        F[:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:] = -dt * self._skew_symmetric(R @ acc)
        return F

    def _skew_symmetric(self, v):
        """Convert vector to skew symmetric matrix"""
        return np.array([[0, -v[2], v[1]],
                        [v[2], 0, -v[0]],
                        [-v[1], v[0], 0]])

    def get_state(self):
        return {
            'position': self.position,
            'velocity': self.velocity,
            'orientation': self.orientation
        }

class VehicleMonitor:
    def __init__(self, width=1200, height=800):
        pygame.init()
        self.display = pygame.display.set_mode((width, height), RESIZABLE)
        pygame.display.set_caption("Vehicle Position Monitor")
        self.font = pygame.font.Font(None, 36)
        
        # Connect to CARLA
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        
        # Find our vehicle
        self.vehicle = None
        while self.vehicle is None:
            vehicles = self.world.get_actors().filter('vehicle.*')
            for v in vehicles:
                if v.attributes.get('role_name') == 'hero':
                    self.vehicle = v
                    break
            time.sleep(0.1)
        
        # Initialize sensors with proper noise profiles
        self.setup_sensors()
        
        # Wait for initial GPS reading
        self.initial_gps = None
        while self.initial_gps is None:
            self.initial_gps = self.last_gps
            time.sleep(0.1)
        
        # Initialize data storage
        self.true_positions = []
        self.gps_positions = []
        self.imu_positions = []
        
        # Initialize IMU integrator
        self.imu_integrator = IMUIntegrator()
        
        # Store initial state
        self.initial_position = self.vehicle.get_location()
        
        # PROPER EKF INITIALIZATION
        # Convert initial GPS to cartesian coordinates
        initial_lat = self.initial_gps.latitude
        initial_lon = self.initial_gps.longitude
        initial_alt = self.initial_gps.altitude
        
        SCALE_LAT = 111000
        SCALE_LON = 111000 * math.cos(math.radians(initial_lat))
        
        # Calculate initial position from GPS
        initial_gps_pos = np.array([
            initial_lon * SCALE_LON,
            -initial_lat * SCALE_LAT,
            initial_alt
        ])
        
        # Set IMU integrator initial position
        self.imu_integrator.position = initial_gps_pos
        
        # Set initial orientation from vehicle transform
        initial_transform = self.vehicle.get_transform()
        roll = math.radians(initial_transform.rotation.roll)
        pitch = math.radians(initial_transform.rotation.pitch)
        yaw = math.radians(initial_transform.rotation.yaw)
        
        # Convert Euler angles to quaternion for initial orientation
        self.imu_integrator.orientation = Rotation.from_euler('xyz', [roll, pitch, yaw]).as_quat()
        
        # Set initial velocity if available
        initial_velocity = self.vehicle.get_velocity()
        if initial_velocity:
            self.imu_integrator.velocity = np.array([
                initial_velocity.x,
                initial_velocity.y,
                initial_velocity.z
            ])
            
        # Colors
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        
    def setup_sensors(self):
        # GPS setup
        gps_bp = self.world.get_blueprint_library().find('sensor.other.gnss')
        
        # IMU setup without noise
        imu_bp = self.world.get_blueprint_library().find('sensor.other.imu')
        
        # Sensor sampling frequency
        IMU_FREQ = 100  # Hz
        GNSS_FREQ = 20  # Hz
        imu_bp.set_attribute('sensor_tick', str(1.0 / IMU_FREQ))
        gps_bp.set_attribute('sensor_tick', str(1.0 / GNSS_FREQ))
        
        # Spawn sensors
        self.gps = self.world.spawn_actor(gps_bp, carla.Transform(carla.Location()), attach_to=self.vehicle)
        self.imu = self.world.spawn_actor(imu_bp, carla.Transform(carla.Location()), attach_to=self.vehicle)
        
        self.last_gps = None
        self.last_imu = None
        
        # Set up sensor callbacks
        self.gps.listen(self.gps_callback)
        self.imu.listen(self.imu_callback)
    
    def gps_callback(self, data):
        self.last_gps = data
    
    def imu_callback(self, data):
        self.last_imu = data

    def update_positions(self):
        current_loc = self.vehicle.get_location()
        true_pos = (current_loc.x, current_loc.y, current_loc.z)
        self.true_positions.append(true_pos)
        
        # GPS position processing
        if self.last_gps:
            lat = self.last_gps.latitude
            lon = self.last_gps.longitude
            alt = self.last_gps.altitude
            
            SCALE_LAT = 111000
            SCALE_LON = 111000 * math.cos(math.radians(lat))
            
            gps_pos = (
                lon * SCALE_LON,
                -lat * SCALE_LAT,
                alt
            )
            self.gps_positions.append(gps_pos)
        
        # IMU integration
        if self.last_imu:
            current_time = self.world.get_snapshot().timestamp.elapsed_seconds
            self.imu_integrator.predict(self.last_imu, current_time)
            
            if self.last_gps:  # Use GPS for corrections
                gps_pos = np.array(gps_pos)
                self.imu_integrator.update_with_gps(gps_pos)
                
            state = self.imu_integrator.get_state()
            self.imu_positions.append(tuple(state['position']))
    
    def draw_text(self, text, pos, color):
        text_surface = self.font.render(text, True, color)
        self.display.blit(text_surface, pos)
    
    def draw_positions(self):
        width, height = self.display.get_size()
        self.display.fill((0, 0, 0))
        
        # Calculate bounds for all recorded positions
        all_positions = self.true_positions + self.gps_positions + self.imu_positions
        if not all_positions:
            return
        
        min_x = min(pos[0] for pos in all_positions)
        max_x = max(pos[0] for pos in all_positions)
        min_y = min(pos[1] for pos in all_positions)
        max_y = max(pos[1] for pos in all_positions)
        
        # Determine the center and scale
        center_x, center_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        max_extent = max(max_x - min_x, max_y - min_y)
        scale = min(width, height) / (max_extent * 1.2)  # Add some padding
        
        def transform_position(pos):
            screen_x = (pos[0] - center_x) * scale + width // 2
            screen_y = (pos[1] - center_y) * scale + height // 2
            return screen_x, screen_y
        
        # Draw current positions and trajectories
        if self.true_positions:
            true_pos = self.true_positions[-1]
            self.draw_text(f"True Pos: ({true_pos[0]:.2f}, {true_pos[1]:.2f}, {true_pos[2]:.2f})", 
                        (10, 10), self.WHITE)
        
        if self.gps_positions:
            gps_pos = self.gps_positions[-1]
            self.draw_text(f"GPS Pos: ({gps_pos[0]:.2f}, {gps_pos[1]:.2f}, {gps_pos[2]:.2f})", 
                        (10, 50), self.GREEN)
            
            if self.last_gps:
                self.draw_text(f"Raw GPS: ({self.last_gps.latitude:.6f}, {self.last_gps.longitude:.6f}, {self.last_gps.altitude:.2f})", 
                            (10, 90), self.GREEN)
        
        if self.imu_positions:
            imu_pos = self.imu_positions[-1]
            self.draw_text(f"IMU Pos: ({imu_pos[0]:.2f}, {imu_pos[1]:.2f}, {imu_pos[2]:.2f})", 
                        (10, 130), self.BLUE)
            
            if self.last_imu:
                acc = self.last_imu.accelerometer
                gyro = self.last_imu.gyroscope
                self.draw_text(f"Raw IMU Acc: ({acc.x:.2f}, {acc.y:.2f}, {acc.z:.2f})", 
                            (10, 170), self.BLUE)
                self.draw_text(f"Raw IMU Gyro: ({gyro.x:.2f}, {gyro.y:.2f}, {gyro.z:.2f})", 
                            (10, 210), self.BLUE)
        
        # Draw error metrics
        if self.true_positions and self.gps_positions:
            true_pos = np.array(self.true_positions[-1])
            gps_pos = np.array(self.gps_positions[-1])
            gps_error = np.linalg.norm(true_pos - gps_pos)
            self.draw_text(f"GPS Error: {gps_error:.2f} meters", 
                        (10, 250), self.GREEN)

        if self.true_positions and self.imu_positions:
            true_pos = np.array(self.true_positions[-1])
            imu_pos = np.array(self.imu_positions[-1])
            imu_error = np.linalg.norm(true_pos - imu_pos)
            self.draw_text(f"IMU Error: {imu_error:.2f} meters", 
                        (10, 290), self.BLUE)

        # Draw trajectories
        def plot_trajectory(positions, color):
            if len(positions) < 2:
                return
            points = [transform_position(pos) for pos in positions]
            pygame.draw.lines(self.display, color, False, points, 2)
        
        plot_trajectory(self.true_positions, self.WHITE)
        plot_trajectory(self.gps_positions, self.GREEN)
        plot_trajectory(self.imu_positions, self.BLUE)
        
        # Draw legend
        self.draw_text("Legend:", (width - 200, 10), self.WHITE)
        self.draw_text("True", (width - 200, 50), self.WHITE)
        self.draw_text("GPS", (width - 200, 90), self.GREEN)
        self.draw_text("IMU", (width - 200, 130), self.BLUE)

    def run(self):
        try:
            while True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == VIDEORESIZE:
                        width, height = event.size
                        self.display = pygame.display.set_mode((width, height), RESIZABLE)
                
                self.update_positions()
                self.draw_positions()
                pygame.display.flip()
                    
        finally:
            pygame.quit()
            self.gps.destroy()
            self.imu.destroy()

if __name__ == '__main__':
    try:
        monitor = VehicleMonitor()
        monitor.run()
    except KeyboardInterrupt:
        pass