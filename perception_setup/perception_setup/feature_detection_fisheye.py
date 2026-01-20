#! /usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point
import math
import yaml

class BrickTrackerROI(Node):
    def __init__(self):
        super().__init__('brick_tracker_roi')
        self.bridge = CvBridge()
        
        # Params
        self.declare_parameter('h_min', 100); self.declare_parameter('h_max', 140)
        self.declare_parameter('s_min', 100); self.declare_parameter('s_max', 255)
        self.declare_parameter('v_min', 50);  self.declare_parameter('v_max', 255)
        self.declare_parameter('min_area', 50)
        self.declare_parameter('padding_factor', 1.0) 
        self.declare_parameter('smoothing_alpha', 0.4)
        
        self.load_calibration('/home/omar-magdy/gp_ws/sim_fisheye.yaml')
        
        # State
        self.roi_center_seed = None 
        self.prev_corners = None
        self.dynamic_roi_size = 200 
        self.desired_features = np.zeros((4,2))

        # Subs/Pubs
        self.image_sub = self.create_subscription(Image, '/cameraAR4/image_raw', self.image_callback, 10)
        self.desired_features_sub = self.create_subscription(Float64MultiArray, '/visual_servo/desired_features', self.desired_features_callback, 10)
        self.seed_sub = self.create_subscription(Point, '/visual_servo/tracker_seed', self.seed_callback, 10)
        
        self.feature_pub = self.create_publisher(Float64MultiArray, '/feature_coordinates_6D', 10)
        self.debug_pub = self.create_publisher(Image, '/brick/debug_image', 10)

        self.get_logger().info("Tracker Ready (Center-Bias). Waiting for lock...")

    def load_calibration(self, path):
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
                self.K = np.array(data['camera_matrix']['data']).reshape(3,3)
                self.D = np.array(data['distortion_coefficients']['data'])
        except: self.K = np.eye(3); self.D = np.zeros(4)

    def desired_features_callback(self, msg):
        if len(msg.data) >= 8: self.desired_features = np.array(msg.data).reshape(-1, 2)

    def seed_callback(self, msg):
        self.roi_center_seed = (msg.x, msg.y)
        self.prev_corners = None 
        self.dynamic_roi_size = 200 
        self.get_logger().info(f"LOCKED ON TRIGGER at ({msg.x:.0f}, {msg.y:.0f})")

    def image_callback(self, msg):
        try: cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: return

        # 1. Determine ROI
        roi_offset = (0, 0); search_img = cv_image
        
        if self.prev_corners is not None or self.roi_center_seed is not None:
            current_roi_size = int(self.dynamic_roi_size)
            if self.prev_corners is not None: cx, cy = np.mean(self.prev_corners, axis=0)
            else: cx, cy = self.roi_center_seed
            
            x_min = max(0, int(cx - current_roi_size/2)); y_min = max(0, int(cy - current_roi_size/2))
            x_max = min(cv_image.shape[1], int(cx + current_roi_size/2)); y_max = min(cv_image.shape[0], int(cy + current_roi_size/2))
            search_img = cv_image[y_min:y_max, x_min:x_max]
            roi_offset = (x_min, y_min)
            
            found, raw_corners = self.detect_blob_rect(search_img, roi_offset)
            if not found: found, raw_corners = self.detect_blob_rect(cv_image, (0,0)) # Retry Full
        else:
            found, raw_corners = self.detect_blob_rect(cv_image, (0,0)) # Auto Search

        # 2. Process
        if found:
            if self.prev_corners is None: sorted_corners = self.sort_corners_geometric(raw_corners)
            else: sorted_corners = self.match_closest_configuration(raw_corners, self.prev_corners)

            alpha = self.get_parameter('smoothing_alpha').value
            if self.prev_corners is not None: smoothed_corners = (alpha * sorted_corners) + ((1 - alpha) * self.prev_corners)
            else: smoothed_corners = sorted_corners
            
            self.prev_corners = smoothed_corners
            final_corners = self.apply_padding(smoothed_corners)

            # ROI Update
            xs = smoothed_corners[:, 0]; ys = smoothed_corners[:, 1]
            max_dim = max(np.max(xs) - np.min(xs), np.max(ys) - np.min(ys))
            self.dynamic_roi_size = max(150, min(max_dim * 2.5, 640))

            msg_pub = Float64MultiArray(); msg_pub.data = final_corners.flatten().tolist()
            self.feature_pub.publish(msg_pub)
            self.draw_debug_image(cv_image, final_corners, roi_offset, self.dynamic_roi_size)
        
        else:
            if self.roi_center_seed: self.dynamic_roi_size = min(640, self.dynamic_roi_size * 1.1)
            cv2.putText(cv_image, "SCANNING...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

    def detect_blob_rect(self, image, offset):
        if image.size == 0: return False, []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, 
                           np.array([self.get_parameter('h_min').value, self.get_parameter('s_min').value, self.get_parameter('v_min').value]), 
                           np.array([self.get_parameter('h_max').value, self.get_parameter('s_max').value, self.get_parameter('v_max').value]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.get_parameter('min_area').value]
        if not valid_contours: return False, []

        # --- FIX: Select Center-Most Blob ---
        img_center = np.array([image.shape[1]/2.0, image.shape[0]/2.0])
        best_cnt = min(valid_contours, key=lambda c: np.linalg.norm(np.array(self.get_centroid(c)) - img_center))

        cnt_reshaped = best_cnt.astype(np.float32).reshape(-1, 2) + np.array(offset, dtype=np.float32)
        try: cnt_undistorted = cv2.fisheye.undistortPoints(cnt_reshaped.reshape(-1, 1, 2), self.K, self.D, P=self.K)
        except: cnt_undistorted = cnt_reshaped.reshape(-1, 1, 2)
        return True, np.float32(cv2.boxPoints(cv2.minAreaRect(cnt_undistorted)))

    def get_centroid(self, cnt):
        M = cv2.moments(cnt)
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (0,0)

    def match_closest_configuration(self, current, prev):
        base = self.sort_corners_geometric(current)
        shifts = [np.roll(base, i, axis=0) for i in range(4)]
        return min(shifts, key=lambda s: np.sum(np.linalg.norm(s - prev, axis=1)))

    def sort_corners_geometric(self, corners):
        center = np.mean(corners, axis=0)
        return np.array(sorted(corners, key=lambda p: math.atan2(p[1]-center[1], p[0]-center[0])), dtype=np.float32)

    def apply_padding(self, corners):
        center = np.mean(corners, axis=0)
        return center + ((corners - center) * self.get_parameter('padding_factor').value)

    def draw_debug_image(self, img, corners, roi_offset, roi_size):
        if roi_offset != (0, 0): cv2.rectangle(img, roi_offset, (int(roi_offset[0]+roi_size), int(roi_offset[1]+roi_size)), (0,255,255), 2)
        for i, pt in enumerate(corners):
            cv2.line(img, tuple(pt.astype(int)), tuple(corners[(i+1)%4].astype(int)), (0, 255, 0), 2)
            cv2.putText(img, str(i), tuple(pt.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        for pt in self.desired_features: cv2.circle(img, tuple(pt.astype(int)), 5, (255, 0, 0), 2)

def main(args=None):
    rclpy.init(args=args); rclpy.spin(BrickTrackerROI()); rclpy.shutdown()

if __name__ == '__main__': main()