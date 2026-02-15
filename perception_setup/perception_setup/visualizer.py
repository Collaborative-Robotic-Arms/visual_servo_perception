#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray

class VSOverlayNode(Node):
    def __init__(self):
        super().__init__('vs_overlay_node')
        self.bridge = CvBridge()
        
        # Internal State for Features
        self.current_features = []
        self.desired_features = []
        
        # Subscribers
        self.create_subscription(Image, 'cameraAR4/image_raw', self.image_callback, 10)
        self.create_subscription(Float64MultiArray, '/visual_servo/shifted_features', self.current_cb, 10)
        self.create_subscription(Float64MultiArray, '/visual_servo/desired_features', self.desired_cb, 10)
        
        # Publisher for the annotated feed
        self.image_pub = self.create_publisher(Image, '/visual_servo/annotated_feed', 10)
        
        self.get_logger().info("Visual Servo Overlay Node Online.")

    def current_cb(self, msg):
        self.current_features = list(msg.data)

    def desired_cb(self, msg):
        self.desired_features = list(msg.data)

    def draw_features(self, img, features, color, label):
        if not features:
            return img
        
        points = []
        # Features are likely [u1, v1, u2, v2...]
        for i in range(0, len(features), 2):
            u, v = int(features[i]), int(features[i+1])
            points.append((u, v))
            # Draw point and ID
            cv2.circle(img, (u, v), 4, color, -1)
            cv2.putText(img, f"{label}_{i//2}", (u+5, v-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw connecting lines (Polygon)
        if len(points) > 1:
            for j in range(len(points)):
                cv2.line(img, points[j], points[(j+1)%len(points)], color, 2)
        return img

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Draw Desired Features (Yellow)
        cv_img = self.draw_features(cv_img, self.desired_features, (0, 255, 255), "goal")
        
        # Draw Current Features (Green)
        cv_img = self.draw_features(cv_img, self.current_features, (0, 255, 0), "curr")

        # Add a legend/status
        cv2.putText(cv_img, "Green: Current | Yellow: Goal", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Convert back to ROS and publish
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_img, encoding='bgr8'))

def main():
    rclpy.init()
    node = VSOverlayNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()