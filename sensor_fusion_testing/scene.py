# import carla
# import random
# import time

# def setup_carla_scene():
#     # Connect to client
#     client = carla.Client('localhost', 2000)
#     client.set_timeout(10.0)
#     world = client.get_world()
#     print(client.get_available_maps())
#     print(client.get_server_version())
#     # Load default map if needed
#     if world.get_map().name != 'Town03':
#         world = client.load_world('Town03')
    
#     # Get blueprint library
#     blueprint_library = world.get_blueprint_library()
    
#     # Spawn points
#     spawn_points = world.get_map().get_spawn_points()
    
#     # Get vehicle blueprint (Tesla Model 3)
#     vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
#     vehicle_bp.set_attribute('role_name', 'hero')
    
#     # Try to spawn vehicle
#     vehicle = None
#     while vehicle is None:
#         try:
#             vehicle = world.spawn_actor(vehicle_bp, random.choice(spawn_points))
#         except:
#             continue
    
#     # Set up traffic manager and disable traffic lights for this vehicle
#     traffic_manager = client.get_trafficmanager()
#     vehicle.set_autopilot(True, traffic_manager.get_port())
#     traffic_manager.ignore_lights_percentage(vehicle, 100)  # Vehicle will ignore all traffic lights
    
#     return client, vehicle

# if __name__ == '__main__':
#     try:
#         client, vehicle = setup_carla_scene()
#         print("Scene setup complete. Vehicle is spawned.")
#         print("Traffic lights are disabled for the vehicle.")
#         print("Press Ctrl+C to exit...")
        
#         while True:
#             time.sleep(1.0)
            
#     except KeyboardInterrupt:
#         print("Cleaning up...")
#         if 'vehicle' in locals():
#             vehicle.destroy()

import carla
import time
import math
import random

def setup_carla_scene():
    # Connect to client
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    print(client.get_available_maps())
    print(client.get_server_version())
    
    # Load default map if needed
    if world.get_map().name != 'Town03':
        world = client.load_world('Town03')
    
    # Get blueprint library and map
    blueprint_library = world.get_blueprint_library()
    carla_map = world.get_map()
    
    # Get spawn points
    spawn_points = carla_map.get_spawn_points()
    
    # Try different spawn point combinations that are closer together
    # This increases the chances of successful navigation
    start_index = 0
    destination_index = 5  # Try a closer destination first
    
    start_point = spawn_points[start_index]
    destination_point = spawn_points[destination_index]
    
    print(f"Starting point: {start_point.location}")
    print(f"Destination point: {destination_point.location}")
    
    # Get vehicle blueprint (Tesla Model 3)
    vehicle_bp = blueprint_library.find('vehicle.tesla.model3')
    vehicle_bp.set_attribute('role_name', 'hero')
    
    # Spawn vehicle at fixed starting point
    vehicle = world.spawn_actor(vehicle_bp, start_point)
    
    # Set up traffic manager
    traffic_manager = client.get_trafficmanager(8000)
    
    # Configure traffic manager more aggressively
    vehicle.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.ignore_lights_percentage(vehicle, 100)
    traffic_manager.ignore_signs_percentage(vehicle, 100)
    traffic_manager.random_left_lanechange_percentage(vehicle, 0)
    traffic_manager.random_right_lanechange_percentage(vehicle, 0)
    traffic_manager.distance_to_leading_vehicle(vehicle, 0.5)  
    traffic_manager.vehicle_percentage_speed_difference(vehicle, -50)  
    
    # Create destination marker for visualization (make it bigger)
    world.debug.draw_point(destination_point.location, size=1.0, color=carla.Color(255, 0, 0), life_time=0)
    world.debug.draw_string(destination_point.location, "DESTINATION", draw_shadow=True,
                           color=carla.Color(255, 0, 0), life_time=0)
    
    # Draw a circle around the destination to represent the arrival zone
    arrival_radius = 8.0  # meters
    for angle in range(0, 360, 10):
        rad = math.radians(angle)
        point = carla.Location(
            x=destination_point.location.x + arrival_radius * math.cos(rad),
            y=destination_point.location.y + arrival_radius * math.sin(rad),
            z=destination_point.location.z + 0.5  # Raise slightly above ground
        )
        world.debug.draw_point(point, size=0.2, color=carla.Color(0, 255, 0), life_time=0)
    
    return client, vehicle, destination_point, world, traffic_manager

def main():
    try:
        client, vehicle, destination, world, traffic_manager = setup_carla_scene()
        print("Scene setup complete. Vehicle is spawned.")
        print("Vehicle is now driving in autopilot mode...")
        print("A green circle has been drawn around the destination.")
        print("Vehicle will stop when it enters this circle.")
        print("You can now control the camera freely in the CarlaUE4 window.")
        
        # Monitor progress variables
        timeout = 300  # 5 minutes timeout
        start_time = time.time()
        arrival_threshold = 8.0  # meters
        arrived = False
        
        # Progress monitoring 
        last_location = vehicle.get_location()
        stuck_counter = 0
        progress_check_interval = 5  # seconds
        last_progress_check = time.time()
        
        while time.time() - start_time < timeout and not arrived:
            current_time = time.time()
            current_location = vehicle.get_location()
            
            # Calculate distance to destination
            distance = current_location.distance(destination.location)
            print(f"Distance to destination: {distance:.2f} meters")
            
            # Check if vehicle is stuck (every 5 seconds)
            if current_time - last_progress_check > progress_check_interval:
                # Calculate movement since last check
                movement = current_location.distance(last_location)
                
                print(f"Movement in last {progress_check_interval} seconds: {movement:.2f} meters")
                
                # If very little movement, increment stuck counter
                if movement < 1.0:  # Less than 1 meter in 5 seconds
                    stuck_counter += 1
                    print(f"Vehicle might be stuck! Stuck counter: {stuck_counter}")
                    
                    # If stuck for too long, try nudging the vehicle
                    if stuck_counter >= 3:
                        print("Vehicle is stuck! Attempting to nudge it...")
                        
                        # Briefly disable autopilot and apply throttle
                        vehicle.set_autopilot(False)
                        vehicle.apply_control(carla.VehicleControl(throttle=0.7, steer=0.0))
                        time.sleep(1.0)
                        
                        # Re-enable autopilot
                        vehicle.set_autopilot(True, traffic_manager.get_port())
                        stuck_counter = 0
                else:
                    stuck_counter = 0  # Reset counter if moving well
                
                # Update for next check
                last_location = current_location
                last_progress_check = current_time
            
            # Check if arrived at destination
            if distance < arrival_threshold:
                # Calculate vehicle speed
                velocity = vehicle.get_velocity()
                speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
                
                print(f"Inside destination zone! Distance: {distance:.2f}m, Speed: {speed:.2f} km/h")
                
                # Stop the vehicle immediately
                vehicle.set_autopilot(False)
                
                # Apply strong braking
                for i in range(20):  # Apply brake multiple times to ensure stopping
                    vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                    time.sleep(0.1)
                
                print("Vehicle stopped at destination!")
                arrived = True
                break
            
            time.sleep(1.0)
            
        if arrived:
            print("Success! Vehicle reached the destination.")
            
            # Keep the vehicle stopped for a while
            for i in range(10):
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                time.sleep(0.5)
        else:
            print("Timeout reached before destination")
            
    except KeyboardInterrupt:
        print("Cleaning up...")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'vehicle' in locals():
            vehicle.destroy()

if __name__ == '__main__':
    main()