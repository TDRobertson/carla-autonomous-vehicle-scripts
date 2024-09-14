import rclpy
from rclpy.node import Node
from sensor_msgs.msg import NavSatFix
from carla_msgs.msg import CarlaEgoVehicleControl

# Define the target GNSS location where the vehicle should stop
TARGET_LATITUDE = -0.000947
TARGET_LONGITUDE = 0.000263

# Define a threshold distance to determine if the vehicle should stop
THRESHOLD_DISTANCE = 0.0001

class GNSSController(Node):

    def __init__(self):
        super().__init__('gnss_controller')
        
        # Create a subscriber to the GNSS topic
        self.subscription = self.create_subscription(
            NavSatFix,
            '/carla/ego_vehicle/gnss',
            self.gnss_callback,
            10)
        
        # Create a publisher to control the vehicle
        self.publisher = self.create_publisher(CarlaEgoVehicleControl, '/carla/ego_vehicle/vehicle_control_cmd', 10)
        
        # Vehicle control message
        self.control_msg = CarlaEgoVehicleControl()

    def gnss_callback(self, gnss_data):
        # Get the current GNSS readings
        current_latitude = gnss_data.latitude
        current_longitude = gnss_data.longitude
        
        self.get_logger().info(f"Current Latitude: {current_latitude}, Current Longitude: {current_longitude}, Altitude: {gnss_data.altitude}")
        
        # Calculate the distance to the target location
        distance_to_target = ((current_latitude - TARGET_LATITUDE) ** 2 + 
                              (current_longitude - TARGET_LONGITUDE) ** 2) ** 0.5
        
        self.get_logger().info(f"Distance to target: {distance_to_target}")

        # If the vehicle is within the threshold distance of the target location, stop the vehicle
        if distance_to_target < THRESHOLD_DISTANCE:
            self.control_msg.throttle = 0.0
            self.control_msg.brake = 1.0
            self.get_logger().info("Stopping the vehicle near the target GNSS values.")
        else:
            self.control_msg.throttle = 0.5
            self.control_msg.brake = 0.0
        
        # Publish the vehicle control message
        self.publisher.publish(self.control_msg)

def main(args=None):
    rclpy.init(args=args)

    gnss_controller = GNSSController()

    rclpy.spin(gnss_controller)

    gnss_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
