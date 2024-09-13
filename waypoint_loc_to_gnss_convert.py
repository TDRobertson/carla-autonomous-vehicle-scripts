import carla

def main():
    # Connect to the CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Get the world and the map
    world = client.get_world()
    map = world.get_map()

    # Define the location (x, y, z) to get the waypoint
    location = carla.Location(x=-25.19, y=139, z=0)

    # Get the waypoint from the location
    waypoint = map.get_waypoint(location)

    # Convert the waypoint location to geolocation
    geolocation = map.transform_to_geolocation(waypoint.transform.location)

    # Print the geolocation data
    print(f"Latitude: {geolocation.latitude}, Longitude: {geolocation.longitude}, Altitude: {geolocation.altitude}")

if __name__ == '__main__':
    main()
