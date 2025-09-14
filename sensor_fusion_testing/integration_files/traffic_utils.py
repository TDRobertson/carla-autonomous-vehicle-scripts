"""
Traffic management utilities for CARLA testing.
Includes functions to disable traffic lights and manage traffic flow.
"""
import carla

def disable_traffic_lights(world):
    """
    Disable all traffic lights in the world to prevent vehicles from stopping.
    This ensures continuous movement during GPS spoofing tests.
    """
    try:
        # Get all traffic lights
        traffic_lights = world.get_actors().filter('traffic.traffic_light')
        
        print(f"Found {len(traffic_lights)} traffic lights")
        
        # Disable each traffic light
        for traffic_light in traffic_lights:
            traffic_light.set_state(carla.TrafficLightState.Green)
            traffic_light.freeze(True)  # Freeze the traffic light state
        
        print("All traffic lights disabled and set to green")
        return True
        
    except Exception as e:
        print(f"Error disabling traffic lights: {e}")
        return False

def enable_traffic_lights(world):
    """
    Re-enable traffic lights in the world.
    """
    try:
        # Get all traffic lights
        traffic_lights = world.get_actors().filter('traffic.traffic_light')
        
        # Re-enable each traffic light
        for traffic_light in traffic_lights:
            traffic_light.freeze(False)  # Unfreeze the traffic light
        
        print("Traffic lights re-enabled")
        return True
        
    except Exception as e:
        print(f"Error enabling traffic lights: {e}")
        return False

def set_vehicle_autopilot_aggressive(vehicle, aggressive=True):
    """
    Set vehicle autopilot to aggressive mode to ignore traffic lights.
    """
    try:
        # Get the traffic manager
        traffic_manager = vehicle.get_world().get_traffic_manager()
        
        # Set aggressive driving
        traffic_manager.vehicle_percentage_speed_difference(vehicle, -20.0)  # 20% faster
        traffic_manager.ignore_lights_percentage(vehicle, 100.0)  # Ignore all traffic lights
        traffic_manager.ignore_signs_percentage(vehicle, 100.0)  # Ignore all stop signs
        traffic_manager.ignore_walkers_percentage(vehicle, 100.0)  # Ignore all walkers
        
        if aggressive:
            print("Vehicle set to aggressive autopilot mode (ignores traffic lights)")
        else:
            print("Vehicle set to normal autopilot mode")
        
        return True
        
    except Exception as e:
        print(f"Error setting vehicle autopilot: {e}")
        return False

def setup_continuous_traffic(world, vehicle):
    """
    Setup continuous traffic flow for testing.
    Disables traffic lights and sets aggressive autopilot.
    """
    print("Setting up continuous traffic flow for testing...")
    
    # Disable traffic lights
    disable_traffic_lights(world)
    
    # Set aggressive autopilot
    set_vehicle_autopilot_aggressive(vehicle, aggressive=True)
    
    print("Continuous traffic flow setup complete")

def cleanup_traffic_setup(world, vehicle):
    """
    Cleanup traffic setup and restore normal behavior.
    """
    print("Cleaning up traffic setup...")
    
    # Re-enable traffic lights
    enable_traffic_lights(world)
    
    # Set normal autopilot
    set_vehicle_autopilot_aggressive(vehicle, aggressive=False)
    
    print("Traffic setup cleanup complete")
