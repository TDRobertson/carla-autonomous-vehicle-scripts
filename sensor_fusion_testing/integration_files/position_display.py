"""
Position Display Utility for CARLA
Shows true position and sensor-estimated position on screen for GPS spoofing visualization.
"""
import carla
import numpy as np
import time
from typing import Optional, Tuple

class PositionDisplay:
    """
    Displays position information on screen using CARLA's debug drawing.
    Shows true position and sensor-estimated position for attack visualization.
    """
    
    def __init__(self, world: carla.World, vehicle: carla.Vehicle, enable_console_output=True):
        self.world = world
        self.vehicle = vehicle
        self.display_enabled = True
        self.last_update_time = 0.0
        self.update_interval = 0.1  # Update every 100ms
        self.enable_console_output = enable_console_output
        
        # Position data
        self.true_position = None
        self.sensor_position = None
        self.position_error = 0.0
        self.velocity_error = 0.0
        self.attack_type = "None"
        self.innovation_value = 0.0
        
        # Display settings
        self.text_size = 0.8
        self.text_color = carla.Color(255, 255, 255, 255)  # White
        self.error_color = carla.Color(255, 100, 100, 255)  # Red for errors
        self.success_color = carla.Color(100, 255, 100, 255)  # Green for success
        
    def update_positions(self, true_pos: Optional[np.ndarray], 
                        sensor_pos: Optional[np.ndarray],
                        attack_type: str = "None",
                        innovation: float = 0.0):
        """Update the position data for display"""
        self.true_position = true_pos
        self.sensor_position = sensor_pos
        self.attack_type = attack_type
        self.innovation_value = innovation
        
        # Calculate errors
        if true_pos is not None and sensor_pos is not None:
            self.position_error = np.linalg.norm(true_pos - sensor_pos)
        else:
            self.position_error = 0.0
            
    def update_velocity_error(self, velocity_error: float):
        """Update velocity error for display"""
        self.velocity_error = velocity_error
        
    def draw_position_info(self):
        """Draw position information on screen"""
        if not self.display_enabled:
            return
            
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            return
            
        self.last_update_time = current_time
        
        # Get camera viewport for text positioning
        try:
            # Draw 3D position markers in the world
            self._draw_3d_markers()
            
            # Draw text overlay above vehicle
            self._draw_text_overlay()
            
        except Exception as e:
            # Silently handle any drawing errors
            pass
            
    def _draw_text_overlay(self):
        """Draw text overlay on screen using 3D text positioned above vehicle"""
        if self.true_position is None or self.sensor_position is None:
            return
            
        try:
            # Get vehicle position for text placement
            vehicle_transform = self.vehicle.get_transform()
            vehicle_location = vehicle_transform.location
            
            # Position text above vehicle
            text_height = 15.0  # Height above vehicle
            text_location = carla.Location(
                x=vehicle_location.x,
                y=vehicle_location.y,
                z=vehicle_location.z + text_height
            )
            
            # Draw main info text
            info_text = f"TRUE: [{self.true_position[0]:.1f}, {self.true_position[1]:.1f}, {self.true_position[2]:.1f}]"
            self.world.debug.draw_string(
                text_location,
                info_text,
                draw_shadow=True,
                color=carla.Color(0, 255, 0, 255),  # Green for true position
                life_time=0.1,
                persistent_lines=False
            )
            
            # Draw sensor position text below
            sensor_text_location = carla.Location(
                x=vehicle_location.x,
                y=vehicle_location.y,
                z=vehicle_location.z + text_height - 2.0
            )
            
            sensor_text = f"SENSOR: [{self.sensor_position[0]:.1f}, {self.sensor_position[1]:.1f}, {self.sensor_position[2]:.1f}]"
            self.world.debug.draw_string(
                sensor_text_location,
                sensor_text,
                draw_shadow=True,
                color=carla.Color(255, 0, 0, 255),  # Red for sensor position
                life_time=0.1,
                persistent_lines=False
            )
            
            # Draw error info below
            error_text_location = carla.Location(
                x=vehicle_location.x,
                y=vehicle_location.y,
                z=vehicle_location.z + text_height - 4.0
            )
            
            # Choose color based on error magnitude
            if self.position_error > 5.0:  # High error - red
                error_color = carla.Color(255, 0, 0, 255)
            elif self.position_error > 1.0:  # Medium error - yellow
                error_color = carla.Color(255, 255, 0, 255)
            else:  # Low error - green
                error_color = carla.Color(0, 255, 0, 255)
                
            error_text = f"ERROR: {self.position_error:.2f}m | ATTACK: {self.attack_type} | INNOVATION: {self.innovation_value:.2f}"
            self.world.debug.draw_string(
                error_text_location,
                error_text,
                draw_shadow=True,
                color=error_color,
                life_time=0.1,
                persistent_lines=False
            )
            
        except Exception as e:
            # Silently handle any drawing errors
            pass
        
    def _draw_3d_markers(self):
        """Draw 3D markers and text in the world"""
        if self.true_position is None or self.sensor_position is None:
            return
            
        # Draw true position marker (green sphere)
        true_location = carla.Location(
            x=float(self.true_position[0]),
            y=float(self.true_position[1]),
            z=float(self.true_position[2]) + 2.0  # Offset above ground
        )
        self.world.debug.draw_point(
            true_location,
            size=0.3,
            color=carla.Color(0, 255, 0, 255),  # Green
            life_time=0.1
        )
        
        # Draw sensor position marker (red sphere)
        sensor_location = carla.Location(
            x=float(self.sensor_position[0]),
            y=float(self.sensor_position[1]),
            z=float(self.sensor_position[2]) + 2.0  # Offset above ground
        )
        self.world.debug.draw_point(
            sensor_location,
            size=0.3,
            color=carla.Color(255, 0, 0, 255),  # Red
            life_time=0.1
        )
        
        # Draw line between true and sensor positions
        self.world.debug.draw_line(
            true_location,
            sensor_location,
            thickness=0.05,
            color=carla.Color(255, 255, 0, 255),  # Yellow line
            life_time=0.1
        )
        
        # Draw text labels
        self._draw_position_labels(true_location, sensor_location)
        
    def _draw_position_labels(self, true_location: carla.Location, 
                            sensor_location: carla.Location):
        """Draw text labels for positions"""
        # True position label
        true_text_location = carla.Location(
            x=true_location.x,
            y=true_location.y,
            z=true_location.z + 1.0
        )
        self.world.debug.draw_string(
            true_text_location,
            f"TRUE: [{self.true_position[0]:.1f}, {self.true_position[1]:.1f}, {self.true_position[2]:.1f}]",
            draw_shadow=True,
            color=carla.Color(0, 255, 0, 255),
            life_time=0.1,
            persistent_lines=False
        )
        
        # Sensor position label
        sensor_text_location = carla.Location(
            x=sensor_location.x,
            y=sensor_location.y,
            z=sensor_location.z + 1.0
        )
        self.world.debug.draw_string(
            sensor_text_location,
            f"SENSOR: [{self.sensor_position[0]:.1f}, {self.sensor_position[1]:.1f}, {self.sensor_position[2]:.1f}]",
            draw_shadow=True,
            color=carla.Color(255, 0, 0, 255),
            life_time=0.1,
            persistent_lines=False
        )
        
        # Error information
        error_location = carla.Location(
            x=(true_location.x + sensor_location.x) / 2,
            y=(true_location.y + sensor_location.y) / 2,
            z=max(true_location.z, sensor_location.z) + 2.0
        )
        
        # Choose color based on error magnitude
        if self.position_error > 5.0:  # High error - red
            error_color = carla.Color(255, 0, 0, 255)
        elif self.position_error > 1.0:  # Medium error - yellow
            error_color = carla.Color(255, 255, 0, 255)
        else:  # Low error - green
            error_color = carla.Color(0, 255, 0, 255)
            
        self.world.debug.draw_string(
            error_location,
            f"ERROR: {self.position_error:.2f}m | ATTACK: {self.attack_type} | INNOVATION: {self.innovation_value:.2f}",
            draw_shadow=True,
            color=error_color,
            life_time=0.1,
            persistent_lines=False
        )
        
    def draw_console_output(self):
        """Draw position information to console"""
        if not self.enable_console_output:
            return
            
        if self.true_position is not None and self.sensor_position is not None:
            print(f"\rTRUE: [{self.true_position[0]:.2f}, {self.true_position[1]:.2f}, {self.true_position[2]:.2f}] | "
                  f"SENSOR: [{self.sensor_position[0]:.2f}, {self.sensor_position[1]:.2f}, {self.sensor_position[2]:.2f}] | "
                  f"ERROR: {self.position_error:.3f}m | ATTACK: {self.attack_type} | INNOVATION: {self.innovation_value:.2f}", 
                  end="", flush=True)
        else:
            print(f"\rWaiting for position data... | ATTACK: {self.attack_type}", end="", flush=True)
            
    def enable_display(self):
        """Enable position display"""
        self.display_enabled = True
        
    def disable_display(self):
        """Disable position display"""
        self.display_enabled = False
        
    def set_update_interval(self, interval: float):
        """Set the update interval for display"""
        self.update_interval = max(0.01, interval)  # Minimum 10ms
        
    def cleanup(self):
        """Cleanup display resources"""
        self.display_enabled = False
