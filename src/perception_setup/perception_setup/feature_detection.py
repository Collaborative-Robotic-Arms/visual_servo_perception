import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped  # Import PointStamped message
from cv_bridge import CvBridge, CvBridgeError
import numpy as np  # Import numpy

class FeatureDetector(Node):
    def __init__(self):
        super().__init__('feature_detector_node')
        self.bridge = CvBridge()
        
        # 1. --- SETUP ARUCO DETECTION ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.get_logger().info("ArUco detector initialized.")

        # 2. --- CREATE PUBLISHER ---
        # This publisher will send the coordinates of the detected feature
        self.publisher_ = self.create_publisher(PointStamped, '/feature_coordinates', 10)
        
        # 3. --- CREATE SUBSCRIBER ---
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.get_logger().info("Node started, subscribing to /camera/image_raw")

    def image_callback(self, msg):
        """Callback function to process incoming image messages."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        corners, ids, rejected = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)

        if ids is not None:
            marker_corners = corners[0][0]
            
            center_x = int(np.mean(marker_corners[:, 0]))
            center_y = int(np.mean(marker_corners[:, 1]))

            point_msg = PointStamped()
            point_msg.header.stamp = self.get_clock().now().to_msg()
            point_msg.header.frame_id = msg.header.frame_id
            
            # (u, v) are the pixel coordinates. z is no longer needed here.
            point_msg.point.x = float(center_x)  # u-coordinate
            point_msg.point.y = float(center_y)  # v-coordinate
            point_msg.point.z = 0.0
            
            self.publisher_.publish(point_msg)
            
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
            cv2.circle(cv_image, (center_x, center_y), 7, (0, 0, 255), -1)

        cv2.imshow("Detection Window", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    feature_detector = FeatureDetector()
    try:
        rclpy.spin(feature_detector)
    except KeyboardInterrupt:
        pass
    finally:
        feature_detector.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()