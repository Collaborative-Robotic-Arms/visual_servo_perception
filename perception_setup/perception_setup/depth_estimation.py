#! /usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import CameraInfo
import numpy as np

class AreaDepthEstimator(Node):
    def __init__(self):
        super().__init__('area_depth_estimator')

        # --- PARAMETERS (Physics only, no magic numbers) ---
        self.declare_parameter('desired_depth', 0.20)  # 20 cm
        self.declare_parameter('marker_size_cm', 5.0)  # 5 cm
        self.declare_parameter('smoothing_factor', 0.1)

        # State
        self.current_depth_estimate = self.get_parameter('desired_depth').value
        self.latest_feature_array = None
        self.focal_length_x = None  # Will be filled by CameraInfo
        self.calculated_desired_area = None

        # --- Subscribers ---
        self.create_subscription(
            CameraInfo, 
            '/cameraAR4/camera_info', 
            self.camera_info_callback, 
            10
        )
        self.create_subscription(
            Float64MultiArray, 
            '/ee/feature_coordinates_6D', 
            self.feature_callback, 
            10
        )
        
        # --- Publisher ---
        self.depth_pub = self.create_publisher(Float64, '/camera_to_marker_depth', 10)
        self.create_timer(0.05, self.timer_callback)
        
        self.get_logger().info('Dynamic Depth Estimator started. Waiting for Camera Info...')

    def camera_info_callback(self, msg):
        if self.focal_length_x is None:
            # Get Focal Length (fx) from K matrix (index 0)
            self.focal_length_x = msg.k[0]
            
            # CALCULATE DESIRED AREA DYNAMICALLY
            # Area = (L * f / Z)^2
            Z = self.get_parameter('desired_depth').value
            L_cm = self.get_parameter('marker_size_cm').value
            # Convert cm to same unit as Z (meters) -> L_m
            L_m = L_cm / 100.0
            
            # Side length in pixels
            side_pixels = (L_m * self.focal_length_x) / Z
            self.calculated_desired_area = side_pixels * side_pixels
            
            self.get_logger().info(f"Calibration Complete:")
            self.get_logger().info(f"  Focal Length: {self.focal_length_x:.1f} px")
            self.get_logger().info(f"  Target Depth: {Z} m")
            self.get_logger().info(f"  Calculated Target Area: {self.calculated_desired_area:.1f} px^2")

    def feature_callback(self, msg):
        arr = np.array(msg.data)
        if len(arr) % 2 != 0: return
        self.latest_feature_array = arr.reshape(-1, 2)

    def calculate_polygon_area(self, points):
        x = points[:, 0]
        y = points[:, 1]
        x_shift = np.roll(x, -1)
        y_shift = np.roll(y, -1)
        area = 0.5 * np.abs(np.dot(x, y_shift) - np.dot(y, x_shift))
        return area

    def timer_callback(self):
        if self.latest_feature_array is None or self.calculated_desired_area is None:
            return
        
        # 1. Calculate Current Area
        if len(self.latest_feature_array) < 3:
            # Fallback for 2 points (diagonal squared)
            pt1 = self.latest_feature_array[0]
            pt2 = self.latest_feature_array[1]
            dist = np.linalg.norm(pt1 - pt2)
            current_area = dist * dist 
        else:
            current_area = self.calculate_polygon_area(self.latest_feature_array)

        if current_area < 1.0: return

        # 2. Estimate Depth using the Calculated Area
        Z_des = self.get_parameter('desired_depth').value
        Area_des = self.calculated_desired_area
        alpha = self.get_parameter('smoothing_factor').value

        # Z_current = Z_desired * sqrt(Area_desired / Area_current)
        raw_depth = Z_des * np.sqrt(Area_des / current_area)

        # Filter
        self.current_depth_estimate = (alpha * raw_depth) + ((1 - alpha) * self.current_depth_estimate)

        # Publish
        msg = Float64()
        msg.data = self.current_depth_estimate
        self.depth_pub.publish(msg)
        
        # Debug Log
        # self.get_logger().info(f"Depth: {self.current_depth_estimate:.3f} m", throttle_duration_sec=0.5)

def main(args=None):
    rclpy.init(args=args)
    node = AreaDepthEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()