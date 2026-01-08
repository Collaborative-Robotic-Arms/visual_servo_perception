#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import math
import time
import yaml
from collections import deque

class BrickTrackerROI(Node):
    def __init__(self):
        super().__init__('brick_tracker_roi')
        
        self.bridge = CvBridge()
        
        # --- Parameters ---
        self.declare_parameter('h_min', 100); self.declare_parameter('h_max', 140)
        self.declare_parameter('s_min', 100); self.declare_parameter('s_max', 255)
        self.declare_parameter('v_min', 50);  self.declare_parameter('v_max', 255)
        self.declare_parameter('min_area', 50)
        
        self.declare_parameter('padding_factor', 1.0) 
        self.declare_parameter('smoothing_alpha', 0.4)
        self.declare_parameter('glitch_threshold', 40.0)
        
        # --- Calibration ---
        calib_path = '/home/omar-magdy/gp_ws/sim_fisheye.yaml'
        self.load_calibration(calib_path)
        
        # --- Desired Features ---
        self.desired_features = np.array([
            [312.8, 309.4], [384.5, 310.2], 
            [383.5, 408.6], [311.8, 407.8]
        ], dtype=np.float32)

        # --- MEMORY ---
        self.prev_corners = None
        self.trail_history = deque(maxlen=50)
        self.dynamic_roi_size = 200 
        
        # --- Communication ---
        self.image_sub = self.create_subscription(Image, '/cameraAR4/image_raw', self.image_callback, 10)
        self.feature_pub = self.create_publisher(Float64MultiArray, '/feature_coordinates_6D', 10)
        
        # Debug Publishers
        self.debug_pub = self.create_publisher(Image, '/brick/debug_image', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/brick/feature_markers', 10)

        self.get_logger().info("Brick Tracker Started (Fast ROI + MinAreaRect)")

    def load_calibration(self, path):
        try:
            with open(path, 'r') as f:
                data = yaml.safe_load(f)
                if 'camera_matrix' in data and 'data' in data['camera_matrix']:
                    self.K = np.array(data['camera_matrix']['data']).reshape(3,3)
                    self.D = np.array(data['distortion_coefficients']['data'])
                else:
                    self.K = np.array(data['camera_matrix']).reshape(3,3)
                    self.D = np.array(data['distortion_coefficients'])
        except Exception as e:
            self.get_logger().error(f"Calibration Error: {e}")
            self.K = np.eye(3); self.D = np.zeros(4)

    def image_callback(self, msg):
        start_time = time.perf_counter()

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError: return

        # --- 1. DYNAMIC WINDOW (Fast ROI) ---
        roi_offset = (0, 0)
        search_img = cv_image
        current_roi_size = int(self.dynamic_roi_size)
        
        if self.prev_corners is not None:
            center = np.mean(self.prev_corners, axis=0)
            x_min = max(0, int(center[0] - current_roi_size/2))
            y_min = max(0, int(center[1] - current_roi_size/2))
            x_max = min(cv_image.shape[1], int(center[0] + current_roi_size/2))
            y_max = min(cv_image.shape[0], int(center[1] + current_roi_size/2))
            
            search_img = cv_image[y_min:y_max, x_min:x_max]
            roi_offset = (x_min, y_min)

        # --- 2. DETECTION ---
        found, raw_corners = self.detect_blob_rect(search_img, roi_offset)

        if not found and self.prev_corners is not None:
            search_img = cv_image
            roi_offset = (0, 0)
            found, raw_corners = self.detect_blob_rect(search_img, roi_offset)

        if found:
            if np.max(np.abs(raw_corners)) > 10000 or np.isnan(raw_corners).any():
                return

            # --- 3. STABILIZATION ---
            if self.prev_corners is None:
                sorted_corners = self.sort_corners_geometric(raw_corners)
                glitch_detected = False
            else:
                aligned_corners = self.match_closest_configuration(raw_corners, self.prev_corners)
                diffs = np.linalg.norm(aligned_corners - self.prev_corners, axis=1)
                
                if np.mean(diffs) > self.get_parameter('glitch_threshold').value:
                    glitch_detected = True
                    old_center = np.mean(self.prev_corners, axis=0)
                    new_center = np.mean(aligned_corners, axis=0)
                    sorted_corners = self.prev_corners + (new_center - old_center)
                else:
                    glitch_detected = False
                    sorted_corners = aligned_corners

            # --- 4. SMOOTHING ---
            alpha = self.get_parameter('smoothing_alpha').value
            if self.prev_corners is not None:
                smoothed_corners = (alpha * sorted_corners) + ((1 - alpha) * self.prev_corners)
            else:
                smoothed_corners = sorted_corners
            
            self.prev_corners = smoothed_corners

            # --- 5. ADAPTIVE ROI ---
            xs = smoothed_corners[:, 0]; ys = smoothed_corners[:, 1]
            max_dim = max(np.max(xs) - np.min(xs), np.max(ys) - np.min(ys))
            self.dynamic_roi_size = max(150, min(max_dim * 2.0, 640))

            # --- 6. OUTPUT ---
            final_corners = self.apply_padding(smoothed_corners)

            end_time = time.perf_counter()
            dt_ms = (end_time - start_time) * 1000.0
            
            status_tag = "[GLITCH LOCK]" if glitch_detected else "[RECT]"
            
            # --- UPDATED PRINT: Shows all 4 corners + Time ---
            print(f"⏱️ {dt_ms:.1f}ms {status_tag}")
            print(f"   TL: {final_corners[0].astype(int)}  TR: {final_corners[1].astype(int)}")
            print(f"   BR: {final_corners[2].astype(int)}  BL: {final_corners[3].astype(int)}")
            print("------------------------------------------------")

            msg_pub = Float64MultiArray()
            msg_pub.data = final_corners.flatten().tolist()
            self.feature_pub.publish(msg_pub)
            
            self.draw_debug_image(cv_image, final_corners, roi_offset, current_roi_size, glitch_detected)
            self.publish_rviz_markers(final_corners, msg.header.frame_id)
        else:
            self.prev_corners = None
            self.dynamic_roi_size = 200
        
        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

    def detect_blob_rect(self, image, offset):
        """
        Uses MinAreaRect on RAW image (Fast).
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([self.get_parameter('h_min').value, self.get_parameter('s_min').value, self.get_parameter('v_min').value])
        upper = np.array([self.get_parameter('h_max').value, self.get_parameter('s_max').value, self.get_parameter('v_max').value])
        
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours: return False, []
        best_cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(best_cnt) < self.get_parameter('min_area').value: return False, []

        cnt_reshaped = best_cnt.astype(np.float32).reshape(-1, 2)
        cnt_global = cnt_reshaped + np.array(offset, dtype=np.float32)
        cnt_global = cnt_global.reshape(-1, 1, 2)

        try: 
            cnt_undistorted = cv2.fisheye.undistortPoints(cnt_global, self.K, self.D, P=self.K)
        except: 
            cnt_undistorted = cnt_global

        rect = cv2.minAreaRect(cnt_undistorted)
        box = cv2.boxPoints(rect)
        
        return True, np.float32(box)

    def match_closest_configuration(self, current_raw, previous):
        base_sort = self.sort_corners_geometric(current_raw)
        min_dist = float('inf'); best_shift = 0
        for shift in range(4):
            shifted = np.roll(base_sort, shift, axis=0)
            dist = np.sum(np.linalg.norm(shifted - previous, axis=1))
            if dist < min_dist: min_dist = dist; best_shift = shift
        return np.roll(base_sort, best_shift, axis=0)

    def sort_corners_geometric(self, corners):
        center = np.mean(corners, axis=0)
        return np.array(sorted(corners, key=lambda pt: math.atan2(pt[1] - center[1], pt[0] - center[0])), dtype=np.float32)

    def apply_padding(self, corners):
        padding = self.get_parameter('padding_factor').value
        center = np.mean(corners, axis=0)
        return center + ((corners - center) * padding)

    def project_pixel_to_3d(self, u, v, Z=0.5):
        fx = self.K[0,0]; fy = self.K[1,1]
        cx = self.K[0,2]; cy = self.K[1,2]
        x = (u - cx) * Z / fx; y = (v - cy) * Z / fy
        return x, y, Z

    def draw_debug_image(self, img, corners, roi_offset, roi_size, glitch):
        if roi_offset != (0, 0):
            top_left = roi_offset
            bottom_right = (roi_offset[0] + roi_size, roi_offset[1] + roi_size)
            tl_x = max(0, min(top_left[0], 640)); tl_y = max(0, min(top_left[1], 480))
            br_x = max(0, min(bottom_right[0], 640)); br_y = max(0, min(bottom_right[1], 480))
            color = (0, 0, 255) if glitch else (0, 255, 255)
            cv2.rectangle(img, (tl_x, tl_y), (br_x, br_y), color, 2)

        for i, pt in enumerate(corners):
            pt_curr = (int(np.clip(pt[0], -5000, 5000)), int(np.clip(pt[1], -5000, 5000)))
            pt_des = (int(self.desired_features[i][0]), int(self.desired_features[i][1]))
            
            # --- CONVERGING LINES (Already here) ---
            # Draws line from Current Corner -> Desired Feature
            cv2.line(img, pt_curr, pt_des, (255, 0, 0), 1) 
            
            cv2.putText(img, str(i), pt_curr, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            next_pt = corners[(i+1)%4]
            pt_next = (int(np.clip(next_pt[0], -5000, 5000)), int(np.clip(next_pt[1], -5000, 5000)))
            cv2.line(img, pt_curr, pt_next, (0, 255, 0), 2)

        for pt in self.desired_features:
            cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (255, 0, 0), 2)

    def publish_rviz_markers(self, corners, frame_id):
        marker_array = MarkerArray()
        timestamp = self.get_clock().now().to_msg()
        for i, pt in enumerate(corners):
            m = Marker(); m.header.frame_id = frame_id; m.header.stamp = timestamp
            m.ns = "current_features"; m.id = i; m.type = Marker.SPHERE; m.action = Marker.ADD
            x, y, z = self.project_pixel_to_3d(pt[0], pt[1], 0.5)
            m.pose.position.x = x; m.pose.position.y = y; m.pose.position.z = z
            m.scale.x = 0.03; m.scale.y = 0.03; m.scale.z = 0.03
            m.color.r = 1.0; m.color.a = 1.0 
            marker_array.markers.append(m)
        center = np.mean(corners, axis=0)
        cx, cy, cz = self.project_pixel_to_3d(center[0], center[1], 0.5)
        self.trail_history.append(Point(x=cx, y=cy, z=cz))
        line = Marker(); line.header.frame_id = frame_id; line.header.stamp = timestamp
        line.ns = "trail"; line.id = 200; line.type = Marker.LINE_STRIP; line.action = Marker.ADD
        line.scale.x = 0.005; line.color.r = 1.0; line.color.g = 1.0; line.color.a = 1.0 
        line.points = list(self.trail_history)
        marker_array.markers.append(line)
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(BrickTrackerROI())
    rclpy.shutdown()

if __name__ == '__main__':
    main()