#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray  # Used for [u1, v1, u2, v2, u3, v3]
from cv_bridge import CvBridge, CvBridgeError
import numpy as np  

class FeatureDetector(Node):
    def __init__(self):
        super().__init__('feature_detector_node')
        self.bridge = CvBridge()
        # --- Parameters ---    
        self.declare_parameter('marker_size_cm', 8.0)
        self.marker_size = self.get_parameter('marker_size_cm').get_parameter_value().double_value

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.get_logger().info("ArUco detector initialized.")

        # Publisher for the 3x2=6 feature coordinates (u1, v1, u2, v2, u3, v3)
        self.publisher_ = self.create_publisher(
            Float64MultiArray, 
            '/feature_coordinates_6D', # <-- Updated Topic Name to reflect 6D data
            10)
        
        self.image_publisher_ = self.create_publisher(Image, '/processed_image', 10)
        
        self.image_subscription = self.create_subscription(
            Image,
            '/cameraAR4/image_raw',
            self.image_callback,
            10)
   
        self.get_logger().info("Node started, subscribing to /cameraAR4/image_raw")
        self.get_logger().info("Publishing 6D feature coordinates to /feature_coordinates_6D")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return
        
        corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None:
            # Only process the first detected marker
            marker_corners = corners[0][0] # Shape (4, 2)
            
            # Draw the detected markers for debugging
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            
            # --- MODIFICATION: Select the first 3 corners (3x2 array) ---
            # Indices [0], [1], and [2] are selected.
            selected_corners = marker_corners[:3] 
            
            # --- Debugging Output ---
            self.get_logger().info("--- Selected Corners ---", throttle_duration_sec=0.5)
            # Print statements to visualize the data being sent
            self.get_logger().info(f"Corner 1 (u, v): ({selected_corners[0][0]:.1f}, {selected_corners[0][1]:.1f})", throttle_duration_sec=0.5)
            self.get_logger().info(f"Corner 2 (u, v): ({selected_corners[1][0]:.1f}, {selected_corners[1][1]:.1f})", throttle_duration_sec=0.5)
            self.get_logger().info(f"Corner 3 (u, v): ({selected_corners[2][0]:.1f}, {selected_corners[2][1]:.1f})", throttle_duration_sec=0.5)
            
            # Flatten the (3, 2) array into a 6-element list: [u1, v1, u2, v2, u3, v3]
            flat_corners = selected_corners.flatten().tolist()
            
            multiarray_msg = Float64MultiArray()
            multiarray_msg.data = flat_corners
            
            self.publisher_.publish(multiarray_msg)
            
        # Publish the processed image (with 2D markers drawn)
        try:
            processed_image_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.image_publisher_.publish(processed_image_msg)
        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert processed image: {e}')

def main(args=None):
    rclpy.init(args=args)
    feature_detector = FeatureDetector()
    try:
        rclpy.spin(feature_detector)
    except KeyboardInterrupt:
        pass
    finally:
        feature_detector.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()