#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from dual_arms_msgs.msg import Brick
import cv2
import numpy as np
import math

class BricksHomography(Node):
    def __init__(self):
        super().__init__('bricks_homography_node')
        self.bridge = CvBridge()
        self.feat_env = None  
        self.feat_arm = None  
        self.H = None 
        
        # Subscribers
        self.create_subscription(Detection2DArray, '/yolo/detections', self.cb_dets_env, 10)
        self.create_subscription(Point, '/grasp/pixel_coords', self.cb_grasp_env, 10)
        self.create_subscription(Float64MultiArray, '/feature_coordinates_6D', self.cb_feat_arm, 10)
        
        # Debug Visualization
        self.last_img_env = None
        self.last_img_arm = None
        self.create_subscription(Image, '/environment_camera/image_raw', self.cb_img_env, 10)
        self.create_subscription(Image, '/cameraAR4/image_raw', self.cb_img_arm, 10)

        # Publishers
        self.grasp_trans_pub = self.create_publisher(Point, '/grasp/pixel_coords_transformed', 10)
        self.debug_pub = self.create_publisher(Image, '/homography/debug_view', 10)
        
        self.get_logger().info("Bricks Homography Bridge (Rotation & Geometry Aware) Online.")

    def cb_img_env(self, msg): self.last_img_env = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    def cb_img_arm(self, msg): self.last_img_arm = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def cb_feat_arm(self, msg): 
        if len(msg.data) >= 8:
            self.feat_arm = np.array(msg.data).reshape(-1, 2)

    def cb_dets_env(self, msg):
        """ Calculate H using the first Global Brick that matches the Local Brick's shape """
        if not msg.detections or self.feat_arm is None: return

        candidates = []
        for det in msg.detections:
            corners = self.get_corners_from_det(det)
            if self.is_geometric_match(corners, self.feat_arm):
                candidates.append(corners)

        if not candidates: return

        # Use the first valid candidate to update H
        self.feat_env = candidates[0]
        try:
            new_H, _ = cv2.findHomography(self.feat_env, self.feat_arm, cv2.RANSAC, 5.0)
            if new_H is not None: 
                self.H = new_H
                self.visualize_homography()
        except: pass

    def cb_grasp_env(self, msg):
        """ Transform X/Y using H, but Pass-Through the Angle (Z) """
        if self.H is not None:
            # 1. Transform the Point (Geometry)
            src = np.array([[[msg.x, msg.y]]], dtype=np.float32)
            try:
                dst = cv2.perspectiveTransform(src, self.H)
                
                out_msg = Point()
                out_msg.x = float(dst[0][0][0])
                out_msg.y = float(dst[0][0][1])
                
                # 2. Pass the Angle Through Directly
                #    The Manager will use this to rotate the Goal Box.
                #    The Controller will then rotate the wrist to match this angle.
                out_msg.z = msg.z 
                
                self.grasp_trans_pub.publish(out_msg)
            except Exception as e:
                self.get_logger().warn(f"Transformation Error: {e}")
                
    # --- Helpers ---
    def get_corners_from_det(self, det):
        # 1. Read Rotation
        cx = det.bbox.center.position.x
        cy = det.bbox.center.position.y
        w = det.bbox.size_x
        h = det.bbox.size_y
        theta = det.bbox.center.theta 

        # 2. Create Rotated Corners
        corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rotated = np.dot(corners, R.T) + np.array([cx, cy])
        
        # 3. Sort Top-Left, Top-Right...
        center = np.mean(rotated, axis=0)
        return np.array(sorted(rotated, key=lambda p: math.atan2(p[1]-center[1], p[0]-center[0])), dtype=np.float32)

    def is_geometric_match(self, c1, c2):
        def get_ratio(pts):
            dists = sorted([np.linalg.norm(pts[i]-pts[(i+1)%4]) for i in range(4)])
            return (dists[0]+dists[1]) / (dists[2]+dists[3]) if dists[3] > 0 else 0
        return abs(get_ratio(c1) - get_ratio(c2)) < 0.2

    def visualize_homography(self):
        if self.last_img_env is None or self.last_img_arm is None: return
        h1, w1 = self.last_img_env.shape[:2]
        h2, w2 = self.last_img_arm.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        vis[:h1, :w1] = self.last_img_env
        vis[:h2, w1:w1+w2] = self.last_img_arm
        
        for i in range(4):
            p1 = (int(self.feat_env[i][0]), int(self.feat_env[i][1]))
            p2 = (int(self.feat_arm[i][0]) + w1, int(self.feat_arm[i][1]))
            cv2.line(vis, p1, p2, (0, 255, 0), 2)
        
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(BricksHomography())
    rclpy.shutdown()

if __name__ == '__main__': main()