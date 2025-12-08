#!/usr/bin/env python3
"""
Real-Time GPS Spoofing Detection Demo

Demonstrates trained ML models detecting GPS spoofing attacks in real-time
within the CARLA simulator.

Usage:
    python detect_spoofing_live.py [--model-dir trained_models] [--duration 60]
"""

import sys
import os
import glob
import time
import argparse
import numpy as np
from datetime import datetime

# Add CARLA to path
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

try:
    sys.path.append(glob.glob('../../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from ml_models.data_loader import DataLoader
from ml_models.ensemble import EnsembleVoting
import joblib

# Try to import pygame for visualization
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("Warning: pygame not available. No visual display.")


class LiveSpoofingDetector:
    """
    Real-time GPS spoofing detection using trained ML models.
    """
    
    def __init__(
        self,
        vehicle,
        model_dir: str = "trained_models",
        enable_display: bool = True
    ):
        """
        Initialize live detector.
        
        Args:
            vehicle: CARLA vehicle actor
            model_dir: Directory containing trained models
            enable_display: Enable pygame visualization
        """
        self.vehicle = vehicle
        self.world = vehicle.get_world()
        self.model_dir = model_dir
        self.enable_display = enable_display and HAS_PYGAME
        
        # Load trained models and scaler
        self.load_models()
        
        # Sensor data buffers (for feature calculation)
        self.data_buffer = []
        self.max_buffer_size = 20  # Keep last 20 samples for rolling stats
        
        # Sensors
        self.gps_sensor = None
        self.imu_sensor = None
        
        # Detection results
        self.detection_history = []
        self.alert_active = False
        
        # Timing
        self.t0 = None
        self.kf_initialized = False
        
        # Kalman filters for feature calculation
        from integration_files.advanced_kalman_filter import AdvancedKalmanFilter
        self.kf_true = AdvancedKalmanFilter()
        self.kf_spoof = AdvancedKalmanFilter()
        
        # Initialize pygame display
        if self.enable_display:
            self.init_display()
            
    def load_models(self):
        """Load trained models from disk."""
        print(f"Loading models from {self.model_dir}...")
        
        try:
            # Load scaler
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            self.scaler = joblib.load(scaler_path)
            print(f"  ✓ Scaler loaded")
            
            # Load ensemble
            ensemble_path = os.path.join(self.model_dir, 'ensemble.pkl')
            self.ensemble = EnsembleVoting.load(ensemble_path)
            print(f"  ✓ Ensemble loaded ({len(self.ensemble.models)} models)")
            
            # Load feature names from data loader
            self.loader = DataLoader()
            self.feature_names = self.loader.PRIMARY_FEATURES
            
            print(f"Models loaded successfully")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            print("Make sure to run train_models.py first")
            raise
            
    def setup_sensors(self):
        """Setup GPS and IMU sensors."""
        bp_lib = self.world.get_blueprint_library()
        
        # GPS sensor
        gps_bp = bp_lib.find('sensor.other.gnss')
        gps_bp.set_attribute('sensor_tick', '0.10')  # 10 Hz
        self.gps_sensor = self.world.spawn_actor(
            gps_bp,
            carla.Transform(carla.Location(x=0.0, z=2.0)),
            attach_to=self.vehicle
        )
        self.gps_sensor.listen(self.gps_callback)
        
        # IMU sensor
        imu_bp = bp_lib.find('sensor.other.imu')
        imu_bp.set_attribute('sensor_tick', '0.02')  # 50 Hz
        self.imu_sensor = self.world.spawn_actor(
            imu_bp,
            carla.Transform(carla.Location(x=0.0, z=2.0)),
            attach_to=self.vehicle
        )
        self.imu_sensor.listen(self.imu_callback)
        
        print("Sensors initialized")
        
    def init_display(self):
        """Initialize pygame display."""
        pygame.init()
        self.screen_width = 800
        self.screen_height = 400
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Live GPS Spoofing Detection")
        self.font = pygame.font.Font(None, 24)
        self.small_font = pygame.font.Font(None, 18)
        self.clock = pygame.time.Clock()
        
    def gps_callback(self, data):
        """GPS callback - collect data and run detection."""
        if self.t0 is None:
            self.t0 = data.timestamp
            
        # Get features (this is simplified - in practice you'd collect proper features)
        # For demo purposes, we'll show the concept
        pass
        
    def imu_callback(self, data):
        """IMU callback."""
        pass
        
    def calculate_features(self) -> Optional[np.ndarray]:
        """Calculate features from buffered sensor data."""
        if len(self.data_buffer) < 10:
            return None  # Need minimum samples for rolling statistics
            
        # Extract features from buffer
        # This would use the same features as training
        # For now, return None (demo purposes)
        return None
        
    def detect_attack(self, features: np.ndarray) -> Dict:
        """
        Run detection on current features.
        
        Args:
            features: Feature vector
            
        Returns:
            Detection result dictionary
        """
        # Normalize features
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Get predictions from ensemble
        prediction = self.ensemble.predict(features_scaled)[0]
        
        # Get individual model predictions
        individual_preds = self.ensemble.get_individual_predictions(features_scaled)
        
        # Convert to binary
        is_attack = prediction == -1
        
        return {
            'is_attack': is_attack,
            'ensemble_prediction': prediction,
            'individual_predictions': individual_preds,
            'timestamp': time.time()
        }
        
    def update_display(self):
        """Update pygame display with detection status."""
        if not self.enable_display:
            return
            
        self.screen.fill((20, 20, 30))  # Dark background
        
        # Title
        title = self.font.render("GPS Spoofing Detection - Live Demo", True, (255, 255, 255))
        self.screen.blit(title, (20, 20))
        
        # Status
        y_offset = 80
        if self.alert_active:
            status_text = "STATUS: ATTACK DETECTED"
            status_color = (255, 50, 50)  # Red
        else:
            status_text = "STATUS: NORMAL"
            status_color = (50, 255, 50)  # Green
            
        status = self.font.render(status_text, True, status_color)
        self.screen.blit(status, (20, y_offset))
        
        # Detection history (last 10)
        y_offset += 60
        history_text = self.small_font.render("Recent Detections:", True, (200, 200, 200))
        self.screen.blit(history_text, (20, y_offset))
        
        for i, detection in enumerate(self.detection_history[-10:]):
            y_offset += 25
            is_attack = detection.get('is_attack', False)
            timestamp = detection.get('timestamp', 0)
            
            if is_attack:
                text = f"  [{i+1}] ATTACK - {timestamp:.1f}s"
                color = (255, 100, 100)
            else:
                text = f"  [{i+1}] Normal - {timestamp:.1f}s"
                color = (100, 255, 100)
                
            detection_text = self.small_font.render(text, True, color)
            self.screen.blit(detection_text, (40, y_offset))
            
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS
        
    def run_demo(self, duration: float = 60.0):
        """
        Run live detection demo.
        
        Args:
            duration: Demo duration in seconds
        """
        print("\n" + "="*60)
        print("LIVE GPS SPOOFING DETECTION DEMO")
        print("="*60)
        print(f"Duration: {duration}s")
        print(f"Loaded models: {len(self.ensemble.models)}")
        print("="*60 + "\n")
        
        # Setup sensors
        self.setup_sensors()
        
        # Enable autopilot
        self.vehicle.set_autopilot(True)
        
        print("Running detection...")
        print("(Note: This is a demo - full feature extraction not implemented)")
        print("For full implementation, integrate with ml_data_collector.py feature pipeline")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # Update display
                if self.enable_display:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("Demo stopped by user")
                            return
                    self.update_display()
                    
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
            
        print(f"\nDemo complete. {len(self.detection_history)} detections")
        
    def cleanup(self):
        """Cleanup sensors and display."""
        if self.gps_sensor:
            self.gps_sensor.destroy()
        if self.imu_sensor:
            self.imu_sensor.destroy()
        if self.enable_display:
            pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="Live GPS spoofing detection demo"
    )
    parser.add_argument(
        '--model-dir', type=str, default='trained_models',
        help='Directory with trained models (default: trained_models)'
    )
    parser.add_argument(
        '--duration', type=float, default=60.0,
        help='Demo duration in seconds (default: 60)'
    )
    parser.add_argument(
        '--no-display', action='store_true',
        help='Disable pygame visualization'
    )
    
    args = parser.parse_args()
    
    print("Live GPS Spoofing Detection")
    print("="*60)
    
    # Connect to CARLA
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print(f"Connected to CARLA: {world.get_map().name}")
    except Exception as e:
        print(f"Failed to connect to CARLA: {e}")
        print("Make sure CarlaUE4.exe is running.")
        return 1
        
    # Spawn vehicle
    spawn_points = world.get_map().get_spawn_points()
    vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[0])
    
    if vehicle is None:
        print("Failed to spawn vehicle")
        return 1
        
    print(f"Spawned vehicle: {vehicle.type_id}")
    time.sleep(1.0)
    
    # Create detector
    detector = LiveSpoofingDetector(
        vehicle=vehicle,
        model_dir=args.model_dir,
        enable_display=not args.no_display
    )
    
    try:
        # Run demo
        detector.run_demo(duration=args.duration)
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        print("\nCleaning up...")
        detector.cleanup()
        try:
            vehicle.destroy()
        except Exception:
            pass
        print("Cleanup complete")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())

