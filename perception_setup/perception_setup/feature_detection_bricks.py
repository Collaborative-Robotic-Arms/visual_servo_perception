#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker, MarkerArray
import math
import os
import time

class BrickTrackerBlob(Node):
    def __init__(self):
        super().__init__('brick_tracker_blob')
        
        # --- Parameters ---
        self.declare_parameter('h_min', 100)
        self.declare_parameter('h_max', 140)
        self.declare_parameter('s_min', 50)
        self.declare_parameter('s_max', 255)
        self.declare_parameter('v_min', 50)
        self.declare_parameter('v_max', 255)
        self.declare_parameter('min_area', 100)
        
        # --- Desired Pixel Coordinates (4 POINTS) ---
        self.desired_features = [
            (265.0, 185.0),
            (375.0, 185.0),
            (375.0, 295.0),
            (265.0, 295.0) 
        ]

        # --- State Variables ---
        self.prev_corners = None
        self.first_lock = True
        
        # --- CSV Logging Setup ---
        self.log_path = os.path.join(os.path.expanduser("~"), "servo_starts.csv")
        file_exists = os.path.isfile(self.log_path)
        
        # Keep file open for appending
        self.log_file = open(self.log_path, "a")
        if not file_exists:
            self.log_file.write("timestamp,center_u,center_v\n")
            self.log_file.flush()

        self.get_logger().info(f"Tracker Started. Saving starts to: {self.log_path}")
        
        # --- Setup ---
        self.bridge = CvBridge()
        
        self.image_sub = self.create_subscription(
            Image, '/cameraAR4/image_raw', self.image_callback, 10)
            
        self.feature_pub = self.create_publisher(
            Float64MultiArray, '/feature_coordinates_6D', 10)
            
        self.debug_pub = self.create_publisher(
            Image, '/brick/debug_image', 10)
            
        self.marker_pub = self.create_publisher(
            MarkerArray, '/brick/feature_markers', 10)

    def __del__(self):
        if hasattr(self, 'log_file') and not self.log_file.closed:
            self.log_file.close()

    def image_callback(self, msg):
        start_time = time.time() # Start timer

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError:
            return

        # 1. Detect Brick (Blob Method)
        found, corners = self.detect_brick_blob(cv_image)

        if found:
            # 2. Match Corners (Temporal Consistency)
            if self.first_lock:
                sorted_corners = self.sort_corners_geometric(corners)
                self.prev_corners = sorted_corners
                
                # --- LOGGING START POSITION ---
                self.get_logger().info(">> FIRST LOCK ACQUIRED - LOGGING POSITION <<")
                self.log_start_position(sorted_corners)
                self.first_lock = False
            else:
                sorted_corners = self.match_corners(corners, self.prev_corners)
                self.prev_corners = sorted_corners

            # 3. Publish Features
            feature_msg = Float64MultiArray()
            flat_list = []
            coord_str = ""
            
            for i in range(4): 
                flat_list.append(sorted_corners[i][0])
                flat_list.append(sorted_corners[i][1])
                # Format string for terminal print
                coord_str += f"[{int(sorted_corners[i][0])},{int(sorted_corners[i][1])}] "
                
            feature_msg.data = flat_list
            self.feature_pub.publish(feature_msg)

            # --- TIMING & PRINTS ---
            dt_ms = (time.time() - start_time) * 1000.0
            self.get_logger().info(f"Time: {dt_ms:.2f} ms | Features: {coord_str}")

            # 4. Visualization
            self.publish_markers(sorted_corners, msg.header.frame_id)
            self.draw_debug(cv_image, sorted_corners)
            
            # Add timing text to image
            cv2.putText(cv_image, "LOCKED", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(cv_image, f"Compute: {dt_ms:.1f} ms", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        else:
            cv2.putText(cv_image, "SEARCHING...", (30, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not self.first_lock:
                self.get_logger().warn("Tracking Lost!", throttle_duration_sec=1.0)
                self.first_lock = True # Reset lock if lost

        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
        except CvBridgeError:
            pass

    def log_start_position(self, corners):
        """Calculates center of the 4 points and saves to CSV."""
        if self.log_file.closed: return
        
        # Calculate Centroid
        cx = np.mean(corners[:, 0])
        cy = np.mean(corners[:, 1])
        
        now = int(time.time())
        self.log_file.write(f"{now},{cx:.1f},{cy:.1f}\n")
        self.log_file.flush()
        
        self.get_logger().info(f"Saved Start Position to CSV: ({cx:.1f}, {cy:.1f})")

    def detect_brick_blob(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, 
                           (self.get_parameter('h_min').value, self.get_parameter('s_min').value, self.get_parameter('v_min').value), 
                           (self.get_parameter('h_max').value, 255, 255))
        
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        max_area = -1
        found = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.get_parameter('min_area').value: continue
            
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0: continue
            
            solidity = area / hull_area
            if solidity > 0.60:
                if area > max_area:
                    max_area = area
                    best_candidate = cnt
                    found = True
        
        if not found: return False, []

        rect = cv2.minAreaRect(best_candidate)
        box = cv2.boxPoints(rect)
        box = np.float32(box)
        return True, box

    def match_corners(self, current_raw, previous):
        current_geo = self.sort_corners_geometric(current_raw)
        min_dist = float('inf')
        best_shift = 0
        for shift in range(4):
            shifted = np.roll(current_geo, shift, axis=0)
            dist = np.sum(np.linalg.norm(shifted - previous, axis=1))
            if dist < min_dist:
                min_dist = dist
                best_shift = shift
        
        if best_shift != 0:
            # Matches C++ logging style
            self.get_logger().info(f"Corrected Rotation Shift: {best_shift}")
            
        return np.roll(current_geo, best_shift, axis=0)

    def sort_corners_geometric(self, corners):
        center = np.mean(corners, axis=0)
        sorted_pts = sorted(corners, key=lambda pt: math.atan2(pt[1] - center[1], pt[0] - center[0]))
        min_sum = float('inf')
        start_idx = 0
        for i, pt in enumerate(sorted_pts):
            if (pt[0] + pt[1]) < min_sum:
                min_sum = pt[0] + pt[1]
                start_idx = i
        return np.array(sorted_pts[start_idx:] + sorted_pts[:start_idx])

    def draw_debug(self, img, corners):
        for i in range(4):
            pt1 = (int(corners[i][0]), int(corners[i][1]))
            pt2 = (int(corners[(i+1)%4][0]), int(corners[(i+1)%4][1]))
            cv2.line(img, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(img, str(i), pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        for i, pt in enumerate(self.desired_features):
            pt_int = (int(pt[0]), int(pt[1]))
            cv2.circle(img, pt_int, 8, (0, 255, 0), 2)
            if i < len(corners):
                curr = (int(corners[i][0]), int(corners[i][1]))
                cv2.line(img, curr, pt_int, (200, 200, 200), 1)

    def publish_markers(self, corners, frame_id):
        marker_array = MarkerArray()
        for i, pt in enumerate(corners):
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "curr"; m.id = i; m.type = Marker.SPHERE; m.action = Marker.ADD
            m.pose.position.x = (pt[0] - 320.0) / 600.0
            m.pose.position.y = (pt[1] - 240.0) / 600.0
            m.pose.position.z = 1.0
            m.scale.x = 0.05; m.scale.y = 0.05; m.scale.z = 0.05
            m.color.r = 1.0; m.color.g = 1.0; m.color.a = 1.0
            marker_array.markers.append(m)
            
        for i, pt in enumerate(self.desired_features):
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "des"; m.id = i+10; m.type = Marker.SPHERE; m.action = Marker.ADD
            m.pose.position.x = (pt[0] - 320.0) / 600.0
            m.pose.position.y = (pt[1] - 240.0) / 600.0
            m.pose.position.z = 1.0
            m.scale.x = 0.05; m.scale.y = 0.05; m.scale.z = 0.05
            m.color.r = 0.0; m.color.g = 1.0; m.color.a = 1.0
            marker_array.markers.append(m)
            
        self.marker_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(BrickTrackerBlob())
    rclpy.shutdown()

if __name__ == '__main__':
    main()