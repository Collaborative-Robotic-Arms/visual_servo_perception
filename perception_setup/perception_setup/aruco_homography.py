#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np

class ArucoHomography(Node):
    def __init__(self):
        super().__init__('aruco_homography_node')
        self.bridge = CvBridge()
        self.img_fixed = None
        self.img_ee = None
        self.feat_fixed = None 
        self.feat_ee = None    

        # Subscribers
        self.create_subscription(Image, '/environment_camera/image_raw', self.cb_img_fixed, 10)
        self.create_subscription(Image, '/cameraAR4/image_raw', self.cb_img_ee, 10)
        self.create_subscription(Float64MultiArray, '/fixed/feature_coordinates_6D', self.cb_feat_fixed, 10)
        self.create_subscription(Float64MultiArray, '/ee/feature_coordinates_6D', self.cb_feat_ee, 10)
        self.create_timer(0.1, self.process_homography)
        self.get_logger().info("Homography Matcher Initialized.")

    def cb_img_fixed(self, msg): self.img_fixed = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    def cb_img_ee(self, msg): self.img_ee = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    def cb_feat_fixed(self, msg): self.feat_fixed = np.array(msg.data).reshape(-1, 2)
    def cb_feat_ee(self, msg): self.feat_ee = np.array(msg.data).reshape(-1, 2)

    def process_homography(self):
        if self.feat_fixed is None or self.feat_ee is None: return

        src_pts = self.feat_fixed.astype(np.float32)
        dst_pts = self.feat_ee.astype(np.float32)

        # --- SAFETY CHECK: Need 4 points ---
        if len(src_pts) < 4 or len(dst_pts) < 4:
            self.get_logger().warn(f"Waiting for full visibility... Fixed: {len(src_pts)}, Arm: {len(dst_pts)}", throttle_duration_sec=1.0)
            return

        # Ensure equal size
        min_points = min(len(src_pts), len(dst_pts))
        src_pts = src_pts[:min_points]
        dst_pts = dst_pts[:min_points]

        try:
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                self.get_logger().info("MATCH FOUND! Homography Calculated.", throttle_duration_sec=2.0)
                self.visualize_matches(src_pts, dst_pts)
        except cv2.error as e:
            self.get_logger().error(f"OpenCV Error: {e}")

    def visualize_matches(self, src, dst):
        if self.img_fixed is None or self.img_ee is None: return
        h1, w1 = self.img_fixed.shape[:2]
        h2, w2 = self.img_ee.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis[:h1, :w1] = self.img_fixed
        vis[:h2, w1:w1+w2] = self.img_ee

        for i in range(len(src)):
            pt1 = (int(src[i][0]), int(src[i][1]))
            pt2 = (int(dst[i][0]) + w1, int(dst[i][1])) 
            cv2.line(vis, pt1, pt2, (0, 255, 0), 2)
            cv2.circle(vis, pt1, 5, (0, 0, 255), -1)
            cv2.circle(vis, pt2, 5, (255, 0, 0), -1)
        cv2.imshow("Homography Matches", vis)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoHomography()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()