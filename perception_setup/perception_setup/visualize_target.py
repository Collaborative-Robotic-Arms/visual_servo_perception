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
        
        # --- UNIFIED PARAMETERS ---
        self.declare_parameter('target_depth', 0.20)
        self.declare_parameter('marker_size', 0.05) 
        
        self.create_subscription(CameraInfo, '/cameraAR4/camera_info', self.cb_info, 10)
        self.create_subscription(Image, '/cameraAR4/image_raw', self.cb_image, 10)
        self.create_subscription(Float64MultiArray, '/ee/feature_coordinates_6D', self.cb_features, 10)
        self.pub_debug = self.create_publisher(Image, '/ibvs/debug_view', 10)
        
        self.bridge = CvBridge(); self.K = None; self.current_features = None 
        self.history = [deque(maxlen=50) for _ in range(4)]
        self.get_logger().info("Visualizer (Unified Params) started.")

    def cb_info(self, msg):
        if self.K is None: self.K = np.array(msg.k).reshape(3, 3)

    def cb_features(self, msg):
        feats = np.array(msg.data).reshape(-1, 2)
        self.current_features = feats
        for i, point in enumerate(feats):
            if i < 4: self.history[i].appendleft(point.astype(int))

    def cb_image(self, msg):
        if self.K is None: return
        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        
        # Draw History
        for track in self.history:
            if len(track) > 1:
                cv2.polylines(cv_img, [np.array(track, dtype=np.int32)], False, (255, 200, 0), 1) 

        # Draw Current
        if self.current_features is not None:
            pts = self.current_features.astype(np.int32)
            cv2.polylines(cv_img, [pts], True, (255, 0, 0), 2) 

        # Draw Goal
        goal_pixels = self.calculate_goal_pixels()
        if goal_pixels is not None:
            pts_goal = goal_pixels.astype(np.int32)
            cv2.polylines(cv_img, [pts_goal], True, (0, 255, 0), 2)
            
            # Draw Error Lines
            if self.current_features is not None:
                n = min(len(self.current_features), len(goal_pixels))
                for i in range(n):
                    cv2.line(cv_img, tuple(self.current_features[i].astype(int)), tuple(goal_pixels[i].astype(int)), (0, 255, 255), 1)

        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(cv_img, "bgr8"))

    def calculate_goal_pixels(self):
        if self.K is None: return None
        
        # Use Unified Parameters
        Z = self.get_parameter('target_depth').value
        L = self.get_parameter('marker_size').value
        h = L / 2.0
        
        # 3 Points (Triangle) - Adjust if using 4 points
        X_3D = np.array([ [-h, -h, Z], [ h, -h, Z], [ h,  h, Z] ])
        
        pixels = []
        for point in X_3D:
            projected = self.K @ point 
            pixels.append([projected[0]/projected[2], projected[1]/projected[2]])
            
        return np.array(pixels)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(IBVSVisualizer())
    rclpy.shutdown()

if __name__ == '__main__':
    main()