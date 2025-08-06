import numpy as np
import time
import sys
import glob
import os
import random
from typing import List, Optional, Tuple, Dict, Any

# Add the CARLA Python API to PYTHONPATH
try:
    # Add the egg file from the correct CARLA location
    carla_root = os.path.abspath('../../../CARLA_0.9.15/PythonAPI/carla')
    
    # Try to find the egg file with the correct pattern
    egg_pattern = os.path.join(carla_root, 'dist/carla-0.9.15-py3.7-win-amd64.egg')
    if os.path.exists(egg_pattern):
        sys.path.append(egg_pattern)
        print(f"Added CARLA egg path: {egg_pattern}")
    else:
        # Fallback to glob pattern
        egg_files = glob.glob(os.path.join(carla_root, 'dist/carla-*.egg'))
        if egg_files:
            sys.path.append(egg_files[0])
            print(f"Added CARLA egg path: {egg_files[0]}")
        else:
            print("Warning: No CARLA egg file found")
    
    # Add the carla directory for agents module
    carla_path = carla_root
    if os.path.exists(carla_path):
        sys.path.append(carla_path)
        print(f"Added CARLA path: {carla_path}")
    else:
        print(f"Warning: CARLA path not found: {carla_path}")
        
except Exception as e:
    print(f"Warning: Error setting up CARLA paths: {e}")
    print(f"Looked in: {os.path.join(carla_root, 'dist')}")

import carla

# Try to import GlobalRoutePlanner
try:
    from agents.navigation.global_route_planner import GlobalRoutePlanner
    print("Successfully imported GlobalRoutePlanner")
except ImportError as e:
    print(f"Error importing GlobalRoutePlanner: {e}")
    print("Please ensure CARLA PythonAPI is in your PYTHONPATH")
    raise

class WaypointGenerator:
    """
    Waypoint generator that integrates with CARLA's traffic manager and route planner.
    Supports various waypoint generation strategies for autonomous navigation.
    """
    
    def __init__(self, world: carla.World, sampling_resolution: float = 2.0):
        """
        Initialize the waypoint generator.
        
        Args:
            world: CARLA world instance
            sampling_resolution: Resolution for route planning (meters)
        """
        self.world = world
        self.map = world.get_map()
        self.sampling_resolution = sampling_resolution
        
        # Initialize route planner
        self.route_planner = GlobalRoutePlanner(self.map, sampling_resolution)
        
        # Cache for generated routes
        self.route_cache = {}
        
    def generate_route_waypoints(self, 
                                start_location: carla.Location,
                                end_location: carla.Location,
                                route_id: Optional[str] = None) -> List[carla.Waypoint]:
        """
        Generate waypoints for a route from start to end location.
        
        Args:
            start_location: Starting location
            end_location: Destination location
            route_id: Optional route identifier for caching
            
        Returns:
            List of waypoints for the route
        """
        try:
            # Check cache first
            if route_id and route_id in self.route_cache:
                print(f"Using cached route: {route_id}")
                return self.route_cache[route_id]
            
            # Generate route using CARLA's route planner
            route = self.route_planner.trace_route(start_location, end_location)
            
            # Extract waypoints from route
            waypoints = []
            for waypoint, road_option in route:
                waypoints.append(waypoint)
                
            print(f"Generated route with {len(waypoints)} waypoints")
            print(f"Start: {start_location}, End: {end_location}")
            
            # Cache the route if route_id provided
            if route_id:
                self.route_cache[route_id] = waypoints
                
            return waypoints
            
        except Exception as e:
            print(f"Error generating route: {e}")
            return []
            
    def generate_random_route(self, 
                            start_location: Optional[carla.Location] = None,
                            route_length: int = 10,
                            max_distance: float = 1000.0) -> List[carla.Waypoint]:
        """
        Generate a random route with specified number of waypoints.
        
        Args:
            start_location: Starting location (uses random spawn point if None)
            route_length: Number of waypoints in the route
            max_distance: Maximum distance for route generation
            
        Returns:
            List of waypoints for the random route
        """
        try:
            # Get start location
            if start_location is None:
                spawn_points = self.map.get_spawn_points()
                if not spawn_points:
                    raise ValueError("No spawn points available")
                start_location = spawn_points[random.randint(0, len(spawn_points) - 1)].location
                
            # Generate random waypoints
            waypoints = []
            current_location = start_location
            
            for i in range(route_length):
                # Get a random waypoint near current location
                current_waypoint = self.map.get_waypoint(current_location)
                if current_waypoint is None:
                    print(f"Could not find waypoint at location {i}")
                    break
                    
                # Get next waypoints and choose a random one
                next_waypoints = current_waypoint.next(50.0)  # Look ahead 50 meters
                if not next_waypoints:
                    print(f"No next waypoints found at step {i}")
                    break
                    
                # Choose a random next waypoint
                next_waypoint = random.choice(next_waypoints)
                waypoints.append(next_waypoint)
                
                # Update current location
                current_location = next_waypoint.transform.location
                
                # Check if we've gone too far
                distance_from_start = current_location.distance(start_location)
                if distance_from_start > max_distance:
                    print(f"Route reached maximum distance at step {i}")
                    break
                    
            print(f"Generated random route with {len(waypoints)} waypoints")
            return waypoints
            
        except Exception as e:
            print(f"Error generating random route: {e}")
            return []
            
    def generate_circular_route(self, 
                              center_location: Optional[carla.Location] = None,
                              radius: float = 100.0,
                              num_waypoints: int = 20) -> List[carla.Waypoint]:
        """
        Generate a circular route around a center point.
        
        Args:
            center_location: Center of the circle (uses random spawn point if None)
            radius: Radius of the circle in meters
            num_waypoints: Number of waypoints in the circle
            
        Returns:
            List of waypoints forming a circular route
        """
        try:
            # Get center location
            if center_location is None:
                spawn_points = self.map.get_spawn_points()
                if not spawn_points:
                    raise ValueError("No spawn points available")
                center_location = spawn_points[random.randint(0, len(spawn_points) - 1)].location
                
            # Generate circular waypoints
            waypoints = []
            angle_step = 2 * np.pi / num_waypoints
            
            for i in range(num_waypoints):
                angle = i * angle_step
                
                # Calculate position on circle
                x = center_location.x + radius * np.cos(angle)
                y = center_location.y + radius * np.sin(angle)
                z = center_location.z
                
                # Create location
                circle_location = carla.Location(x=x, y=y, z=z)
                
                # Get nearest waypoint
                waypoint = self.map.get_waypoint(circle_location)
                if waypoint is not None:
                    waypoints.append(waypoint)
                    
            print(f"Generated circular route with {len(waypoints)} waypoints")
            print(f"Center: {center_location}, Radius: {radius}m")
            return waypoints
            
        except Exception as e:
            print(f"Error generating circular route: {e}")
            return []
            
    def generate_traffic_manager_route(self, 
                                     vehicle: carla.Vehicle,
                                     destination: carla.Location,
                                     traffic_manager: Optional[carla.TrafficManager] = None) -> List[carla.Waypoint]:
        """
        Generate a route using CARLA's traffic manager.
        
        Args:
            vehicle: Vehicle to generate route for
            destination: Destination location
            traffic_manager: Traffic manager instance (will create one if None)
            
        Returns:
            List of waypoints for the traffic manager route
        """
        try:
            # Create traffic manager if not provided
            if traffic_manager is None:
                traffic_manager = self.world.get_traffic_manager()
                
            # Set destination for traffic manager
            traffic_manager.set_destination(vehicle, destination)
            
            # Get the route from traffic manager
            # Note: This is a simplified approach. In practice, you might need to
            # extract waypoints from the traffic manager's internal route
            start_location = vehicle.get_location()
            waypoints = self.generate_route_waypoints(start_location, destination)
            
            print(f"Generated traffic manager route with {len(waypoints)} waypoints")
            return waypoints
            
        except Exception as e:
            print(f"Error generating traffic manager route: {e}")
            return []
            
    def generate_waypoint_chain(self, 
                               start_waypoint: carla.Waypoint,
                               num_waypoints: int = 10,
                               distance_between: float = 50.0) -> List[carla.Waypoint]:
        """
        Generate a chain of waypoints starting from a given waypoint.
        
        Args:
            start_waypoint: Starting waypoint
            num_waypoints: Number of waypoints to generate
            distance_between: Distance between consecutive waypoints
            
        Returns:
            List of waypoints in the chain
        """
        try:
            waypoints = [start_waypoint]
            current_waypoint = start_waypoint
            
            for i in range(num_waypoints - 1):
                # Get next waypoints at specified distance
                next_waypoints = current_waypoint.next(distance_between)
                if not next_waypoints:
                    print(f"No next waypoints found at step {i}")
                    break
                    
                # Choose the first next waypoint (or could be random)
                next_waypoint = next_waypoints[0]
                waypoints.append(next_waypoint)
                current_waypoint = next_waypoint
                
            print(f"Generated waypoint chain with {len(waypoints)} waypoints")
            return waypoints
            
        except Exception as e:
            print(f"Error generating waypoint chain: {e}")
            return []
            
    def generate_route_from_spawn_points(self, 
                                       start_spawn_index: int = 0,
                                       end_spawn_index: Optional[int] = None) -> List[carla.Waypoint]:
        """
        Generate a route between two spawn points.
        
        Args:
            start_spawn_index: Index of starting spawn point
            end_spawn_index: Index of ending spawn point (random if None)
            
        Returns:
            List of waypoints for the route
        """
        try:
            spawn_points = self.map.get_spawn_points()
            if not spawn_points:
                raise ValueError("No spawn points available")
                
            # Validate start index
            if start_spawn_index >= len(spawn_points):
                start_spawn_index = 0
                
            # Get end index
            if end_spawn_index is None:
                end_spawn_index = random.randint(0, len(spawn_points) - 1)
            elif end_spawn_index >= len(spawn_points):
                end_spawn_index = len(spawn_points) - 1
                
            # Ensure start and end are different
            if start_spawn_index == end_spawn_index:
                end_spawn_index = (end_spawn_index + 1) % len(spawn_points)
                
            start_location = spawn_points[start_spawn_index].location
            end_location = spawn_points[end_spawn_index].location
            
            route_id = f"spawn_{start_spawn_index}_to_{end_spawn_index}"
            return self.generate_route_waypoints(start_location, end_location, route_id)
            
        except Exception as e:
            print(f"Error generating route from spawn points: {e}")
            return []
            
    def visualize_route(self, waypoints: List[carla.Waypoint], 
                       color: carla.Color = carla.Color(r=0, g=255, b=0),
                       life_time: float = 120.0):
        """
        Visualize a route by drawing waypoints in the CARLA world.
        
        Args:
            waypoints: List of waypoints to visualize
            color: Color for the waypoint markers
            life_time: How long to show the markers
        """
        try:
            for i, waypoint in enumerate(waypoints):
                location = waypoint.transform.location
                self.world.debug.draw_point(
                    location, 
                    size=0.2, 
                    color=color, 
                    life_time=life_time, 
                    persistent_lines=True
                )
                
                # Draw line to next waypoint
                if i < len(waypoints) - 1:
                    next_location = waypoints[i + 1].transform.location
                    self.world.debug.draw_line(
                        location, 
                        next_location, 
                        thickness=0.1, 
                        color=color, 
                        life_time=life_time, 
                        persistent_lines=True
                    )
                    
            print(f"Visualized route with {len(waypoints)} waypoints")
            
        except Exception as e:
            print(f"Error visualizing route: {e}")
            
    def get_route_statistics(self, waypoints: List[carla.Waypoint]) -> Dict[str, Any]:
        """
        Calculate statistics for a route.
        
        Args:
            waypoints: List of waypoints to analyze
            
        Returns:
            Dictionary containing route statistics
        """
        if not waypoints:
            return {}
            
        try:
            total_distance = 0.0
            distances = []
            
            for i in range(len(waypoints) - 1):
                current_location = waypoints[i].transform.location
                next_location = waypoints[i + 1].transform.location
                distance = current_location.distance(next_location)
                total_distance += distance
                distances.append(distance)
                
            # Calculate statistics
            stats = {
                'num_waypoints': len(waypoints),
                'total_distance': total_distance,
                'average_distance_between_waypoints': np.mean(distances) if distances else 0.0,
                'min_distance_between_waypoints': np.min(distances) if distances else 0.0,
                'max_distance_between_waypoints': np.max(distances) if distances else 0.0,
                'start_location': waypoints[0].transform.location if waypoints else None,
                'end_location': waypoints[-1].transform.location if waypoints else None
            }
            
            return stats
            
        except Exception as e:
            print(f"Error calculating route statistics: {e}")
            return {}
            
    def clear_cache(self):
        """Clear the route cache"""
        self.route_cache.clear()
        print("Route cache cleared")
        
    def get_available_spawn_points(self) -> List[carla.Transform]:
        """Get all available spawn points"""
        return self.map.get_spawn_points()
        
    def get_random_spawn_point(self) -> Optional[carla.Transform]:
        """Get a random spawn point"""
        spawn_points = self.get_available_spawn_points()
        if spawn_points:
            return random.choice(spawn_points)
        return None 