#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque

class IBVSVisualizer(Node):
    def __init__(self):
        super().__init__('ibvs_visualizer')
        
        # --- PARAMETERS ---
        self.declare_parameter('target_depth', 0.20)  # 20cm
        self.declare_parameter('marker_size_cm', 5.0) # 5cm
        
        # --- SUBSCRIBERS ---
        self.create_subscription(CameraInfo, '/cameraAR4/camera_info', self.cb_info, 10)
        self.create_subscription(Image, '/cameraAR4/image_raw', self.cb_image, 10)
        self.create_subscription(Float64MultiArray, '/ee/feature_coordinates_6D', self.cb_features, 10)
        
        # --- PUBLISHER ---
        self.pub_debug = self.create_publisher(Image, '/ibvs/debug_view', 10)
        
        self.bridge = CvBridge()
        self.K = None 
        self.current_features = None 
        
        # --- TRAJECTORY HISTORY (New!) ---
        # Stores the last 50 positions for up to 4 points
        self.history = [deque(maxlen=50) for _ in range(4)]
        
        self.get_logger().info("IBVS Visualizer with Tracking started.")

    def cb_info(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)

    def cb_features(self, msg):
        # 1. Parse current features
        feats = np.array(msg.data).reshape(-1, 2)
        self.current_features = feats
        
        # 2. Update History for Trajectory Tracking
        for i, point in enumerate(feats):
            if i < 4: # Only track up to 4 corners
                self.history[i].appendleft(point.astype(int))

    def cb_image(self, msg):
        if self.K is None: return
        
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # --- 1. DRAW TRAJECTORIES (The "Tracking" part) ---
        for i, track in enumerate(self.history):
            if len(track) > 1:
                # Draw a line connecting all past points
                pts = np.array(track, dtype=np.int32)
                # Faint Blue Lines
                cv2.polylines(cv_img, [pts], False, (255, 200, 0), 1) 

        # --- 2. DRAW CURRENT MARKER (Blue) ---
        if self.current_features is not None:
            pts = self.current_features.astype(np.int32)
            cv2.polylines(cv_img, [pts], True, (255, 0, 0), 2) 
            for p in pts:
                cv2.circle(cv_img, tuple(p), 4, (255, 0, 0), -1)

        # --- 3. DRAW GOAL MARKER (Green) ---
        goal_pixels = self.calculate_goal_pixels()
        if goal_pixels is not None:
            pts_goal = goal_pixels.astype(np.int32)
            cv2.polylines(cv_img, [pts_goal], True, (0, 255, 0), 2)
            for p in pts_goal:
                cv2.circle(cv_img, tuple(p), 4, (0, 255, 0), -1)

            # --- 4. DRAW ERROR LINES (Yellow) ---
            if self.current_features is not None:
                n = min(len(self.current_features), len(goal_pixels))
                for i in range(n):
                    pt_curr = tuple(self.current_features[i].astype(int))
                    pt_goal = tuple(goal_pixels[i].astype(int))
                    cv2.line(cv_img, pt_curr, pt_goal, (0, 255, 255), 1)

        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(cv_img, "bgr8"))

    def calculate_goal_pixels(self):
        if self.K is None: return None
        
        Z = self.get_parameter('target_depth').value
        L = self.get_parameter('marker_size_cm').value / 100.0 
        h = L / 2.0
        
        # 3 Points (Triangle) - Matching your Controller Mode
        # If you switched controller back to 4 points, add the 4th line back here!
        X_3D = np.array([
            [-h, -h, Z], # Top-Left
            [ h, -h, Z], # Top-Right
            [ h,  h, Z]  # Bottom-Right
        ])
        
        pixels = []
        for point in X_3D:
            projected = self.K @ point 
            u = projected[0] / projected[2]
            v = projected[1] / projected[2]
            pixels.append([u, v])
            
        return np.array(pixels)

def main(args=None):
    rclpy.init(args=args)
    node = IBVSVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()