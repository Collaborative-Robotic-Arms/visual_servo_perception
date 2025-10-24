#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped  
from cv_bridge import CvBridge, CvBridgeError
import numpy as np  

class FeatureDetector(Node):
    def __init__(self):
        super().__init__('feature_detector_node')
        self.bridge = CvBridge()
        
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.get_logger().info("ArUco detector initialized.")

        # Publisher for the feature coordinates
        self.publisher_ = self.create_publisher(PointStamped, '/feature_coordinates', 10)
        
        # --- 1. ADD A PUBLISHER FOR THE DEBUG IMAGE ---
        self.image_publisher_ = self.create_publisher(Image, '/processed_image', 10)
        
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.get_logger().info("Node started, subscribing to /camera/image_raw")

    def image_callback(self, msg):
        print("dakhlt hena")

        try:
            print("dakhlt el try")
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            print("3adet awel satr")
        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return
        print("3adet el execpt")
        corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            marker_corners = corners[0][0]
            
            center_x = int(np.mean(marker_corners[:, 0]))
            center_y = int(np.mean(marker_corners[:, 1]))

            point_msg = PointStamped()
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.header.frame_id = msg.header.frame_id
            
            point_msg.point.x = float(center_x)
            point_msg.point.y = float(center_y)
            point_msg.point.z = 0.0
            
            self.publisher_.publish(point_msg)
            
            # Draw visualizations on the image
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            cv2.circle(cv_image, (center_x, center_y), 7, (0, 0, 255), -1)

        # --- 2. REPLACE cv2.imshow() WITH THE PUBLISHER ---
        # Instead of trying to display a window, publish the processed image
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