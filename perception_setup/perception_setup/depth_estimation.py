#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray
import numpy as np

class AreaDepthEstimator(Node):
    def __init__(self):
        super().__init__('area_depth_estimator')

        # --- Parameters ---
        # Z_star: The actual physical distance when the robot is at the desired position
        self.declare_parameter('desired_depth', 0.2093) 
        
        # Area_star: The area (in pixels^2) of the feature polygon when at the desired position
        # YOU MUST CALIBRATE THIS: Move robot to goal, read the log, and update this value.
        self.declare_parameter('desired_feature_area', 10011.0 ) 

        # Smoothing factor (0.0 to 1.0). 1.0 = No smoothing, 0.1 = Heavy smoothing
        self.declare_parameter('smoothing_factor', 0.2)

        # --- State Variables ---
        self.current_depth_estimate = self.get_parameter('desired_depth').value
        self.latest_feature_array = None
        
        # --- Subscribers ---
        # Assumes data is [u1, v1, u2, v2, u3, v3, ...]
        self.feature_sub = self.create_subscription(
            Float64MultiArray, '/feature_coordinates_6D', self.feature_callback, 10)
        
        # --- Publisher ---
        self.depth_pub = self.create_publisher(Float64, '/camera_to_marker_depth', 10)

        # --- Timer ---
        self.timer = self.create_timer(0.05, self.timer_callback) # 20 Hz
        
        self.get_logger().info('Area-Based Depth Estimator started.')
        self.get_logger().info('REMINDER: Set "desired_feature_area" based on the logs when at goal position.')

    def feature_callback(self, msg):
        """Stores and reshapes the feature vector."""
        # Reshape flat array into (N, 2) where N is number of points
        arr = np.array(msg.data)
        if len(arr) % 2 != 0:
            self.get_logger().error("Feature array length is not even!")
            return
        self.latest_feature_array = arr.reshape(-1, 2)

    def calculate_polygon_area(self, points):
        """
        Calculates area using the Shoelace Formula (Surveyor's Formula).
        This connects the dots: 1->2->3->4->1 and calculates the area inside.
        Assumes points are ordered (e.g., clockwise or counter-clockwise).
        """
        x = points[:, 0]
        y = points[:, 1]
        
        # Shift arrays to do (x_i * y_i+1)
        x_shift = np.roll(x, -1)
        y_shift = np.roll(y, -1)
        
        # Area = 0.5 * | sum(x*y_shift - y*x_shift) |
        area = 0.5 * np.abs(np.dot(x, y_shift) - np.dot(y, x_shift))
        return area

    def timer_callback(self):
        if self.latest_feature_array is None:
            return
        
        # 1. Calculate Current Area (in pixels^2)
        # Check if we have enough points to make a shape (need at least 3)
        if len(self.latest_feature_array) < 3:
            # Fallback for 2 points: use bounding box diagonal squared
            pt1 = self.latest_feature_array[0]
            pt2 = self.latest_feature_array[1]
            dist = np.linalg.norm(pt1 - pt2)
            current_area = dist * dist # heuristic approximation
        else:
            current_area = self.calculate_polygon_area(self.latest_feature_array)

        # Safety check for zero area (features collapsed or lost)
        if current_area < 1.0:
            self.get_logger().warn(f"Area too small ({current_area:.1f}). Features might be lost or collinear.")
            return

        # 2. Get Parameters
        Z_des = self.get_parameter('desired_depth').value
        Area_des = self.get_parameter('desired_feature_area').value
        alpha = self.get_parameter('smoothing_factor').value

        # 3. The Area-Depth Law
        # Z_current = Z_desired * sqrt(Area_desired / Area_current)
        # Because Area scales with 1/Z^2, Sqrt(Area) scales with 1/Z.
        
        raw_depth_estimate = Z_des * np.sqrt(Area_des / current_area)

        # 4. Low Pass Filter (Smoothing)
        # smooth_val = (alpha * new) + ((1-alpha) * old)
        self.current_depth_estimate = (alpha * raw_depth_estimate) + ((1 - alpha) * self.current_depth_estimate)

        # 5. Publish
        msg = Float64()
        msg.data = self.current_depth_estimate
        self.depth_pub.publish(msg)

        # Logging for calibration
        # When your robot is at the goal, copy this 'Area' value into your parameters!
        self.get_logger().info(f"Area: {current_area:.1f} px | Est Depth: {self.current_depth_estimate:.3f} m", throttle_duration_sec=0.5)

def main(args=None):
    rclpy.init(args=args)
    node = AreaDepthEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()