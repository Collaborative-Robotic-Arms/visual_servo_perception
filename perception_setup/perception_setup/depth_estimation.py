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
        self.declare_parameter('desired_depth', 0.2114) 
        self.declare_parameter('desired_feature_area', 7056.9) 
        self.declare_parameter('smoothing_factor', 0.2)

        # --- State ---
        self.current_depth_estimate = self.get_parameter('desired_depth').value
        
        # --- Subscribers ---
        # Trigger calculation immediately upon receiving data (Event Driven)
        self.feature_sub = self.create_subscription(
            Float64MultiArray, '/feature_coordinates_6D', self.feature_callback, 10)
        
        # --- Publisher ---
        self.depth_pub = self.create_publisher(Float64, '/camera_to_marker_depth', 10)
        
        self.get_logger().info('Depth Estimator Started. Optimized event-driven mode.')

    def feature_callback(self, msg):
        # Start Clock
        t_start = time.perf_counter()

        data = msg.data # Access raw list/array from ROS message
        if not data: return

        # --- OPTIMIZED AREA CALCULATION ---
        area = 0.0
        
        # FAST PATH: If we have exactly 4 points (8 numbers), skip NumPy overhead entirely.
        # This is significantly faster for small arrays.
        if len(data) == 8:
            # Unrolled Shoelace Formula for 4 points
            # Points: (x0,y0), (x1,y1), (x2,y2), (x3,y3)
            # data indices: 0,1,   2,3,   4,5,   6,7
            x0, y0 = data[0], data[1]
            x1, y1 = data[2], data[3]
            x2, y2 = data[4], data[5]
            x3, y3 = data[6], data[7]

            # Shoelace: 0.5 * |(x0y1 + x1y2 + x2y3 + x3y0) - (y0x1 + y1x2 + y2x3 + y3x0)|
            term1 = x0*y1 + x1*y2 + x2*y3 + x3*y0
            term2 = y0*x1 + y1*x2 + y2*x3 + y3*x0
            area = 0.5 * abs(term1 - term2)

        # SLOW PATH: Use NumPy for generic N-point polygons
        else:
            if len(data) % 2 != 0 or len(data) < 6: return
            points = np.array(data).reshape(-1, 2)
            area = self.calculate_polygon_area_numpy(points)

        # Safety Check
        if area < 100.0: return

        # --- DEPTH ESTIMATION ---
        Z_des = self.get_parameter('desired_depth').value
        Area_des = self.get_parameter('desired_feature_area').value
        alpha = self.get_parameter('smoothing_factor').value

        # Depth Law: Z = Z* sqrt(A* / A)
        # Using pure python math.sqrt is slightly faster for scalars than np.sqrt
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
        # Log area, depth, and compute time
        self.get_logger().info(
            f"Time: {dt_ms:.3f}ms | Area: {area:.1f} | Depth: {self.current_depth_estimate:.3f} m", 
            throttle_duration_sec=0.5
        )

    def calculate_polygon_area_numpy(self, points):
        """ Fallback for non-4-point polygons """
        x = points[:, 0]
        y = points[:, 1]
        x_shift = np.roll(x, -1)
        y_shift = np.roll(y, -1)
        return 0.5 * np.abs(np.dot(x, y_shift) - np.dot(y, x_shift))

def main(args=None):
    rclpy.init(args=args)
    node = AreaDepthEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()