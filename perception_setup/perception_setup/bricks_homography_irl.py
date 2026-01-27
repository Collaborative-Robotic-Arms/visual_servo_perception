#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point
from vision_msgs.msg import Detection2DArray
from sensor_msgs.msg import Image, CameraInfo
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

        # --- CAMERA INTRINSICS STATE ---
        self.camera_info_received = False
        self.K = np.eye(3)
        self.D = np.zeros(4)
        
        # --- SUBSCRIBERS ---
        self.create_subscription(Detection2DArray, '/yolo/detections', self.cb_dets_env, 10)
        self.create_subscription(Point, '/grasp/pixel_coords', self.cb_grasp_env, 10)
        
        # Note: We expect these features to be UNDISTORTED (Linear) from the upstream node
        self.create_subscription(Float64MultiArray, '/feature_coordinates_6D', self.cb_feat_arm, 10)
        
        self.create_subscription(Brick, '/mission/target_brick', self.cb_mission, 10)
        
        # Debug Visualization
        self.last_img_env = None
        self.last_img_arm = None
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.cb_img_env, 10)
        self.create_subscription(Image, '/cameraAR4/image_raw', self.cb_img_arm, 10)

        # NEW: Camera Info Subscriber
        self.info_sub = self.create_subscription(CameraInfo, '/cameraAR4/camera_info', self.camera_info_callback, 10)

        # --- PUBLISHERS ---
        self.grasp_trans_pub = self.create_publisher(Point, '/grasp/pixel_coords_transformed', 10)
        self.debug_pub = self.create_publisher(Image, '/homography/debug_view', 10)
        
        self.get_logger().info("Bricks Homography: Fisheye-Aware Bridge Active.")

    def camera_info_callback(self, msg):
        """ Updates Intrinsics from Topic """
        if not self.camera_info_received:
            self.K = np.array(msg.k).reshape(3, 3)
            self.D = np.array(msg.d)[:4]
            self.camera_info_received = True
            self.get_logger().info(f"Camera Info Received. Homography undistortion ready.")

    def cb_mission(self, msg):
        self.get_logger().info(f"New Mission Received (Type: {msg.type}). Resetting Homography search.")
        self.feat_env = None
        self.H = None

    def cb_img_env(self, msg): 
        # Environment camera is assumed to be standard/linear
        self.last_img_env = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def cb_img_arm(self, msg): 
        # --- CRITICAL FIX ---
        # The features we receive are Linear (Undistorted).
        # We must undistort the background image so the visualization lines up.
        if not self.camera_info_received: return

        raw_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.last_img_arm = cv2.fisheye.undistortImage(raw_img, self.K, self.D, Knew=self.K)

    def cb_feat_arm(self, msg): 
        if len(msg.data) >= 8:
            self.feat_arm = np.array(msg.data).reshape(-1, 2)

    def cb_dets_env(self, msg):
        if not msg.detections or self.feat_arm is None: return

        best_candidate = None
        
        if self.feat_env is not None:
            # Sticky Tracking
            prev_center = np.mean(self.feat_env, axis=0)
            min_dist = float('inf')
            for det in msg.detections:
                corners = self.get_corners_from_det(det)
                if self.is_geometric_match(corners, self.feat_arm):
                    dist = np.linalg.norm(np.mean(corners, axis=0) - prev_center)
                    if dist < min_dist:
                        min_dist = dist
                        best_candidate = corners
        else:
            # Search mode
            for det in msg.detections:
                corners = self.get_corners_from_det(det)
                if self.is_geometric_match(corners, self.feat_arm):
                    best_candidate = corners
                    break

        if best_candidate is not None:
            self.feat_env = best_candidate
            try:
                # 
                # Both sets of points (env and arm) must be in Linear Space for this to work
                new_H, _ = cv2.findHomography(self.feat_env, self.feat_arm, cv2.RANSAC, 5.0)
                if new_H is not None: 
                    self.H = new_H
                    self.visualize_homography()
            except: pass

    def cb_grasp_env(self, msg):
        if self.H is not None:
            src = np.array([[[msg.x, msg.y]]], dtype=np.float32)
            try:
                # Transform: Linear Env -> Linear Arm
                dst = cv2.perspectiveTransform(src, self.H)
                out_msg = Point()
                out_msg.x = float(dst[0][0][0])
                out_msg.y = float(dst[0][0][1])
                out_msg.z = msg.z 
                self.grasp_trans_pub.publish(out_msg)
            except: pass
                
    def get_corners_from_det(self, det):
        cx, cy = det.bbox.center.position.x, det.bbox.center.position.y
        w, h = det.bbox.size_x, det.bbox.size_y
        theta = det.bbox.center.theta 
        corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        rotated = np.dot(corners, R.T) + np.array([cx, cy])
        center = np.mean(rotated, axis=0)
        return np.array(sorted(rotated, key=lambda p: math.atan2(p[1]-center[1], p[0]-center[0])), dtype=np.float32)

    def is_geometric_match(self, c1, c2):
        # Compares aspect ratios
        def get_ratio(pts):
            dists = sorted([np.linalg.norm(pts[i]-pts[(i+1)%4]) for i in range(4)])
            return (dists[0]+dists[1]) / (dists[2]+dists[3]) if dists[3] > 0 else 0
        return abs(get_ratio(c1) - get_ratio(c2)) < 0.2

    def visualize_homography(self):
        if self.last_img_env is None or self.last_img_arm is None: return
        h1, w1 = self.last_img_env.shape[:2]
        h2, w2 = self.last_img_arm.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        
        # Place Environment Image (Left)
        vis[:h1, :w1] = self.last_img_env
        
        # Place Arm Image (Right) - Now Undistorted!
        vis[:h2, w1:w1+w2] = self.last_img_arm
        
        for i in range(4):
            p1 = (int(self.feat_env[i][0]), int(self.feat_env[i][1]))
            # p2 coordinates match the UNDISTORTED arm image
            p2 = (int(self.feat_arm[i][0]) + w1, int(self.feat_arm[i][1]))
            cv2.line(vis, p1, p2, (0, 255, 0), 2)
            
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(vis, "bgr8"))

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(BricksHomography())
    rclpy.shutdown()

if __name__ == '__main__': main()