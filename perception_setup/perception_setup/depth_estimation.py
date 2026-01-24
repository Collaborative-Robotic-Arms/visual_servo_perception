#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray
import numpy as np
import time

class AreaDepthEstimator(Node):
    def __init__(self):
        super().__init__('area_depth_estimator')

        # --- Parameters ---
        self.declare_parameter('desired_depth', 0.2) 
        self.declare_parameter('desired_feature_area', 7056.9) 
        self.declare_parameter('smoothing_factor', 0.2)

        # --- State ---
        self.current_depth_estimate = self.get_parameter('desired_depth').value
        self.dynamic_desired_area = self.get_parameter('desired_feature_area').value
        
        # Logging State
        self.last_logged_desired_area = 0.0

        # --- Subscribers ---
        self.feature_sub = self.create_subscription(
            Float64MultiArray, '/feature_coordinates_6D', self.feature_callback, 10)
        
        self.desired_features_sub = self.create_subscription(
            Float64MultiArray, '/visual_servo/desired_features', self.desired_features_callback, 10)
        
        # --- Publisher ---
        self.depth_pub = self.create_publisher(Float64, '/camera_to_brick_depth', 10)
        
        # Initial Log
        self.get_logger().info('Depth Estimator Online.')

    def desired_features_callback(self, msg):
        """Calculates area of the DESIRED polygon to set the scale"""
        if not msg.data: return
        
        area = self.calculate_area_from_flat_list(msg.data)
        
        if area > 100.0:
            self.dynamic_desired_area = area
            
            # --- LOGGING ---
            # Only log if the desired area has changed significantly (avoid spamming)
            if abs(self.dynamic_desired_area - self.last_logged_desired_area) > 1.0:
                self.get_logger().info(f"TARGET UPDATED: Desired Area set to {area:.1f} pixels²")
                self.last_logged_desired_area = self.dynamic_desired_area

    def feature_callback(self, msg):
        # Start Clock
        t_start = time.perf_counter()

        data = msg.data 
        if not data:
            self.get_logger().info("WAITING FOR FEATURES...", throttle_duration_sec=2.0)
            return

        # --- OPTIMIZED AREA CALCULATION ---
        area = self.calculate_area_from_flat_list(data)

        # Safety Check
        if area < 100.0: 
            return

        # --- DEPTH ESTIMATION ---
        Z_des = self.get_parameter('desired_depth').value
        Area_des = self.dynamic_desired_area
        alpha = self.get_parameter('smoothing_factor').value

        # Depth Law: Z = Z* sqrt(A* / A)
        raw_depth_estimate = Z_des * (Area_des / area)**0.5

        # Low Pass Filter
        self.current_depth_estimate = (alpha * raw_depth_estimate) + ((1 - alpha) * self.current_depth_estimate)

        # --- PUBLISH ---
        msg_out = Float64()
        msg_out.data = self.current_depth_estimate
        self.depth_pub.publish(msg_out)

        # Stop Clock
        t_end = time.perf_counter()
        dt_ms = (t_end - t_start) * 1000.0

        # --- LOGGING ---
        # Throttled log to show performance without flooding
        self.get_logger().info(
            f"ESTIMATION | Area: {area:.0f}/{Area_des:.0f} | Depth: {self.current_depth_estimate:.3f}m | Calc: {dt_ms:.2f}ms", 
            throttle_duration_sec=0.5
        )

    def calculate_area_from_flat_list(self, data):
        """Helper to calculate area from [x0, y0, x1, y1...]"""
        area = 0.0
        if len(data) == 8:
            # Fast Shoelace for 4 points
            x0, y0 = data[0], data[1]
            x1, y1 = data[2], data[3]
            x2, y2 = data[4], data[5]
            x3, y3 = data[6], data[7]
            term1 = x0*y1 + x1*y2 + x2*y3 + x3*y0
            term2 = y0*x1 + y1*x2 + y2*x3 + y3*x0
            area = 0.5 * abs(term1 - term2)
        else:
            # Generic
            if len(data) % 2 != 0 or len(data) < 6: return 0.0
            points = np.array(data).reshape(-1, 2)
            x = points[:, 0]
            y = points[:, 1]
            x_shift = np.roll(x, -1)
            y_shift = np.roll(y, -1)
            area = 0.5 * np.abs(np.dot(x, y_shift) - np.dot(y, x_shift))
        return area

def main(args=None):
    rclpy.init(args=args)
    node = AreaDepthEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()