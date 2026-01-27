#! /usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64MultiArray, Bool
from geometry_msgs.msg import Point
from dual_arms_msgs.msg import Brick
import math

class BrickTrackerROI(Node):
    def __init__(self):
        super().__init__('brick_tracker_roi')
        self.bridge = CvBridge()
        
        # --- CAMERA INTRINSICS STATE ---
        self.camera_info_received = False
        self.K = np.eye(3)
        self.D = np.zeros(4)

        # --- PARAMS ---
        self.declare_parameter('h_min', 100); self.declare_parameter('h_max', 140)
        self.declare_parameter('s_min', 100); self.declare_parameter('s_max', 255)
        self.declare_parameter('v_min', 50);  self.declare_parameter('v_max', 255)
        self.declare_parameter('min_area', 50)
        self.declare_parameter('padding_factor', 1.0) 
        self.declare_parameter('smoothing_alpha', 0.4)
        
        # Morph Params (Ported from hardcoded file)
        # Increase this to bridge larger gaps between the L-legs
        self.declare_parameter('morph_kernel_size', 20) 

        # Stability
        self.declare_parameter('max_jump_dist', 30.0)
        self.declare_parameter('max_area_change', 0.4)
        self.declare_parameter('target_switch_threshold', 10)

        # State
        self.roi_center_seed = None 
        self.prev_corners = None
        self.prev_area = None
        self.dynamic_roi_size = 200 
        self.desired_features = np.zeros((4,2))
        self.rejection_count = 0
        self.current_brick_type = None

        # --- SUBSCRIBERS ---
        self.image_sub = self.create_subscription(Image, '/cameraAR4/image_raw', self.image_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/cameraAR4/camera_info', self.camera_info_callback, 10)
        self.desired_features_sub = self.create_subscription(Float64MultiArray, '/visual_servo/desired_features', self.desired_features_callback, 10)
        self.seed_sub = self.create_subscription(Point, '/visual_servo/tracker_seed', self.seed_callback, 10)
        self.reset_sub = self.create_subscription(Bool, '/visual_servo/reset_roi', self.reset_callback, 10)
        self.mission_sub = self.create_subscription(Brick, '/mission/target_brick', self.cb_mission, 10)
        
        # --- PUBLISHERS ---
        self.feature_pub = self.create_publisher(Float64MultiArray, '/feature_coordinates_6D', 10)
        self.debug_pub = self.create_publisher(Image, '/feature_detection/debug', 10) 

        self.get_logger().info("Brick Tracker ROI (Dynamic Intrinsics + Morph Closing) Ready.")

    def camera_info_callback(self, msg):
        """ Updates Intrinsics from Topic """
        if not self.camera_info_received:
            self.K = np.array(msg.k).reshape(3, 3)
            self.D = np.array(msg.d)[:4]
            self.camera_info_received = True
            self.get_logger().info(f"Camera Info Received. K loaded.")

    def cb_mission(self, msg):
        """ Auto-Reset tracking when brick type changes """
        if msg.type != self.current_brick_type:
            self.current_brick_type = msg.type
            self.force_reset("New Mission")

    def reset_callback(self, msg):
        if msg.data:
            self.force_reset("Manual Trigger")

    def force_reset(self, reason):
        self.prev_corners = None
        self.roi_center_seed = None
        self.prev_area = None
        self.rejection_count = 0
        self.get_logger().warn(f"RESET TRIGGERED ({reason}): ROI cleared. Scanning full image.")

    def desired_features_callback(self, msg):
        if len(msg.data) >= 8: self.desired_features = np.array(msg.data).reshape(-1, 2)

    def seed_callback(self, msg):
        self.roi_center_seed = (msg.x, msg.y)
        self.prev_corners = None 
        self.prev_area = None
        self.rejection_count = 0
        self.get_logger().info(f"SEED TRIGGER: Locking onto ({msg.x:.0f}, {msg.y:.0f})")

    def image_callback(self, msg):
        if not self.camera_info_received:
            self.get_logger().warn("Waiting for Camera Info...", throttle_duration_sec=2.0)
            return

        try: cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: return

        # 1. Logic: Use ROI only if we are tracking something valid
        found = False
        raw_corners = []
        roi_offset = (0, 0)
        
        # Step A: Try Tracking inside ROI (Fast & Stable)
        if self.prev_corners is not None or self.roi_center_seed is not None:
            
            current_roi_size = int(self.dynamic_roi_size)
            if self.prev_corners is not None: cx, cy = np.mean(self.prev_corners, axis=0)
            else: cx, cy = self.roi_center_seed
            
            x_min = max(0, int(cx - current_roi_size/2)); y_min = max(0, int(cy - current_roi_size/2))
            x_max = min(cv_image.shape[1], int(cx + current_roi_size/2)); y_max = min(cv_image.shape[0], int(cy + current_roi_size/2))
            
            search_img = cv_image[y_min:y_max, x_min:x_max]
            roi_offset = (x_min, y_min)
            found, raw_corners = self.detect_blob_closing(search_img, roi_offset)
        
        # Step B: If ROI failed, OR if we have no history, Scan FULL IMAGE
        if not found:
            roi_offset = (0, 0)
            found, raw_corners = self.detect_blob_closing(cv_image, (0,0))
            if found:
                self.prev_corners = None 
                self.rejection_count = 0

        # 2. Process Features
        if found:
            if self.prev_corners is None: sorted_corners = self.sort_corners_geometric(raw_corners)
            else: sorted_corners = self.match_closest_configuration(raw_corners, self.prev_corners)

            accepted = True
            curr_area = self.poly_area(sorted_corners)
            
            # Stability Checks (Jump & Area)
            if self.prev_corners is not None and self.prev_area is not None:
                curr_center = np.mean(sorted_corners, axis=0)
                prev_center = np.mean(self.prev_corners, axis=0)
                dist = np.linalg.norm(curr_center - prev_center)
                area_ratio = abs(curr_area - self.prev_area) / (self.prev_area + 1e-5)

                if dist > self.get_parameter('max_jump_dist').value or \
                   area_ratio > self.get_parameter('max_area_change').value:
                    accepted = False
                    self.rejection_count += 1
                else:
                    self.rejection_count = 0

            # Force Switch if "Let Go" threshold reached
            if self.rejection_count > self.get_parameter('target_switch_threshold').value:
                self.get_logger().warn("AUTO-SWITCH: Snapping to new object.")
                accepted = True
                self.rejection_count = 0
                self.prev_corners = sorted_corners 
            
            if accepted:
                alpha = self.get_parameter('smoothing_alpha').value
                if self.prev_corners is not None and self.rejection_count == 0: 
                    smoothed_corners = (alpha * sorted_corners) + ((1 - alpha) * self.prev_corners)
                else: 
                    smoothed_corners = sorted_corners
                
                self.prev_corners = smoothed_corners
                self.prev_area = curr_area 
                
                final_corners = self.apply_padding(smoothed_corners)

                # Update ROI size dynamically
                xs = smoothed_corners[:, 0]; ys = smoothed_corners[:, 1]
                max_dim = max(np.max(xs) - np.min(xs), np.max(ys) - np.min(ys))
                self.dynamic_roi_size = max(150, min(max_dim * 2.5, 640))

                msg_pub = Float64MultiArray(); msg_pub.data = final_corners.flatten().tolist()
                self.feature_pub.publish(msg_pub)
                self.draw_debug_image(cv_image, final_corners, roi_offset, self.dynamic_roi_size, True)
            
            else:
                if self.prev_corners is not None:
                    final_corners = self.apply_padding(self.prev_corners)
                    msg_pub = Float64MultiArray(); msg_pub.data = final_corners.flatten().tolist()
                    self.feature_pub.publish(msg_pub)
                    self.draw_debug_image(cv_image, final_corners, roi_offset, self.dynamic_roi_size, False)
        
        else:
            if self.roi_center_seed: self.dynamic_roi_size = min(640, self.dynamic_roi_size * 1.1)
            cv2.putText(cv_image, "SCANNING...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            self.draw_debug_image(cv_image, [], roi_offset, self.dynamic_roi_size, False)

        self.debug_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

    # --- MORPHOLOGICAL CLOSING ---
    def detect_blob_closing(self, image, offset):
        if image.size == 0: return False, []
        
        # 1. Masking
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, 
                           np.array([self.get_parameter('h_min').value, self.get_parameter('s_min').value, self.get_parameter('v_min').value]), 
                           np.array([self.get_parameter('h_max').value, self.get_parameter('s_max').value, self.get_parameter('v_max').value]))
        
        # 2. MORPH CLOSING (Connect neighbors without growing size)
        k_size = self.get_parameter('morph_kernel_size').value
        kernel = np.ones((k_size, k_size), np.uint8)
        
        # This fills the gap between the L-legs but keeps outer edges intact
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # 3. Find Contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.get_parameter('min_area').value]
        if not valid_contours: return False, []

        img_center = np.array([image.shape[1]/2.0, image.shape[0]/2.0])
        best_cnt = min(valid_contours, key=lambda c: np.linalg.norm(np.array(self.get_centroid(c)) - img_center))

        # 4. Undistort Points (USING DYNAMIC INTRINSICS)
        cnt_reshaped = best_cnt.astype(np.float32).reshape(-1, 2) + np.array(offset, dtype=np.float32)
        try: 
            # Note: self.K and self.D are updated via camera_info_callback
            cnt_undistorted = cv2.fisheye.undistortPoints(cnt_reshaped.reshape(-1, 1, 2), self.K, self.D, P=self.K)
            pts = cnt_undistorted.reshape(-1, 2) 
        except: 
            pts = cnt_reshaped

        # 5. MinAreaRect
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        
        return True, np.float32(box)

    def poly_area(self, corners):
        x = corners[:, 0]; y = corners[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

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

    def draw_debug_image(self, img, corners, roi_offset, roi_size, accepted):
        if roi_offset != (0, 0): 
            cv2.rectangle(img, roi_offset, (int(roi_offset[0]+roi_size), int(roi_offset[1]+roi_size)), (0,255,255), 2)
            cv2.putText(img, "ROI LOCKED", (roi_offset[0], roi_offset[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        
        if len(corners) > 0:
            color = (0, 255, 0) if accepted else (0, 0, 255) 
            for i, pt in enumerate(corners):
                cv2.line(img, tuple(pt.astype(int)), tuple(corners[(i+1)%4].astype(int)), color, 2)
                cv2.putText(img, str(i), tuple(pt.astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            
            if not accepted and self.rejection_count > 0:
                status = f"SWITCHING? ({self.rejection_count}/10)"
            else:
                status = "TRACKING" if accepted else "HOLDING"
            
            cv2.putText(img, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        for pt in self.desired_features: cv2.circle(img, tuple(pt.astype(int)), 5, (255, 0, 0), 2)

def main(args=None):
    rclpy.init(args=args); rclpy.spin(BrickTrackerROI()); rclpy.shutdown()

if __name__ == '__main__': main()