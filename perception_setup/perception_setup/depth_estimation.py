#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import CameraInfo
import numpy as np

class AreaDepthEstimator(Node):
    def __init__(self):
        super().__init__('area_depth_estimator')

        # --- UNIFIED PARAMETERS (Match C++ names) ---
        self.declare_parameter('target_depth', 0.20)  # Meters
        self.declare_parameter('marker_size', 0.05)   # Meters
        self.declare_parameter('smoothing_factor', 0.1)

        self.current_depth_estimate = self.get_parameter('target_depth').value
        self.latest_feature_array = None
        self.focal_length_x = None 
        self.calculated_desired_area = None

        # Subscribers
        self.create_subscription(CameraInfo, '/cameraAR4/camera_info', self.cb_info, 10)
        self.create_subscription(Float64MultiArray, '/ee/feature_coordinates_6D', self.cb_features, 10)
        
        self.depth_pub = self.create_publisher(Float64, '/camera_to_marker_depth', 10)
        self.create_timer(0.05, self.timer_callback)
        self.get_logger().info('Depth Estimator (Unified Params) started.')

    def cb_info(self, msg):
        if self.focal_length_x is None:
            self.focal_length_x = msg.k[0]
            
            # Use unified parameters
            Z = self.get_parameter('target_depth').value
            L = self.get_parameter('marker_size').value # Already in meters
            
            # Area = (L * f / Z)^2
            side_pixels = (L * self.focal_length_x) / Z
            self.calculated_desired_area = side_pixels * side_pixels
            
            self.get_logger().info(f"Target Area: {self.calculated_desired_area:.1f} px^2")

    def cb_features(self, msg):
        arr = np.array(msg.data)
        if len(arr) % 2 != 0: return
        self.latest_feature_array = arr.reshape(-1, 2)

    def calculate_polygon_area(self, points):
        x = points[:, 0]; y = points[:, 1]
        x_shift = np.roll(x, -1); y_shift = np.roll(y, -1)
        return 0.5 * np.abs(np.dot(x, y_shift) - np.dot(y, x_shift))

    def timer_callback(self):
        if self.latest_feature_array is None or self.calculated_desired_area is None: return
        
        # Calculate Area (Square or Triangle logic handled by polygon formula)
        if len(self.latest_feature_array) < 3:
            pt1 = self.latest_feature_array[0]; pt2 = self.latest_feature_array[1]
            dist = np.linalg.norm(pt1 - pt2)
            current_area = dist * dist 
        else:
            current_area = self.calculate_polygon_area(self.latest_feature_array)

        if current_area < 1.0: return

        Z_des = self.get_parameter('target_depth').value
        Area_des = self.calculated_desired_area
        alpha = self.get_parameter('smoothing_factor').value

        raw_depth = Z_des * np.sqrt(Area_des / current_area)
        self.current_depth_estimate = (alpha * raw_depth) + ((1 - alpha) * self.current_depth_estimate)

        msg = Float64()
        msg.data = self.current_depth_estimate
        self.depth_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(AreaDepthEstimator())
    rclpy.shutdown()

if __name__ == '__main__':
    main()