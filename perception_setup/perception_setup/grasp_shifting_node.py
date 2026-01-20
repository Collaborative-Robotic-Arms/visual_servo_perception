#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CameraInfo
from dual_arms_msgs.msg import Brick
from cv_bridge import CvBridge
import numpy as np
import cv2

class GraspShiftingNode(Node):
    def __init__(self):
        super().__init__('grasp_shifting_node')
        self.bridge = CvBridge()
        self.ref_corners = None       
        self.ref_grasp_pixel = None   
        self.current_corners_rect = None 
        self.pending_grasp_msg = None 
        
        # --- TRUE SIMULATION INTRINSICS (Linear Space) ---
        self.f_sim = 190.68
        self.K_sim = np.array([
            [self.f_sim,        0.0, 320.0],
            [       0.0, self.f_sim, 240.0],
            [       0.0,        0.0,   1.0]
        ], dtype=np.float64)
        
        self.D_sim = np.zeros(4) 

        # --- CAMERA FEED DATA ---
        self.K_real = None
        self.D_real = None
        self.latest_image = None

        # --- SUBSCRIBERS ---
        self.create_subscription(Float64MultiArray, '/feature_coordinates_6D', self.cb_current_corners, 10)
        self.create_subscription(Point, '/grasp/pixel_coords_transformed', self.cb_transformed_grasp, 10)
        self.create_subscription(Brick, '/mission/target_brick', self.cb_mission, 10)
        self.create_subscription(Image, '/cameraAR4/image_raw', self.cb_image, 10)
        self.create_subscription(CameraInfo, '/cameraAR4/camera_info', self.cb_cam_info, 10)
        
        # --- PUBLISHERS ---
        self.shifted_pub = self.create_publisher(Float64MultiArray, '/visual_servo/shifted_features', 10)
        self.debug_pub = self.create_publisher(Image, '/visual_servo/debug_shift', 10)
        self.seed_pub = self.create_publisher(Point, '/visual_servo/tracker_seed', 10)
        
        self.get_logger().info(f"Grasp Shifter Online. Camera Feed Visualization Active.")

    def cb_cam_info(self, msg):
        if self.K_real is None:
            self.K_real = np.array(msg.k).reshape(3, 3)
            self.D_real = np.array(msg.d)

    def cb_image(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            pass

    def cb_mission(self, msg):
        self.ref_corners = None
        self.pending_grasp_msg = None

    def cb_transformed_grasp(self, msg): 
        self.pending_grasp_msg = msg
        self.try_latch()

    def cb_current_corners(self, msg): 
        if len(msg.data) >= 8:
            self.current_corners_rect = np.array(msg.data).reshape(-1, 2)
            self.try_latch() 
            self.process_and_publish()
        
    def try_latch(self):
        if self.ref_corners is not None: return
        if self.pending_grasp_msg is None or self.current_corners_rect is None: return
        
        self.ref_corners = self.current_corners_rect.copy()
        
        # Undistort the incoming transformed point to match our Linear space
        raw_click = np.array([[[self.pending_grasp_msg.x, self.pending_grasp_msg.y]]], dtype=np.float32)
        self.ref_grasp_pixel = cv2.fisheye.undistortPoints(
            raw_click, self.K_sim, self.D_sim, P=self.K_sim
        ).reshape(-1, 2)

        # Trigger the local tracker seed
        seed_msg = Point()
        seed_msg.x = float(self.pending_grasp_msg.x)
        seed_msg.y = float(self.pending_grasp_msg.y)
        self.seed_pub.publish(seed_msg)
        
        self.get_logger().info(f"LATCHED! Seed sent to tracker.")

    def process_and_publish(self):
        if self.ref_corners is not None and self.current_corners_rect is not None:
            try:
                ref_rect = self.ref_corners
                curr_rect = self.current_corners_rect
                grasp_rect = self.ref_grasp_pixel

                H, _ = cv2.findHomography(ref_rect, curr_rect, cv2.RANSAC, 5.0)
                
                if H is not None:
                    grasp_shifted_rect = cv2.perspectiveTransform(grasp_rect.reshape(-1,1,2), H).reshape(-1,2)
                    current_center_rect = np.mean(curr_rect, axis=0)
                    shift_vector = grasp_shifted_rect[0] - current_center_rect
                    final_features_rect = curr_rect + shift_vector
                    
                    msg = Float64MultiArray()
                    msg.data = final_features_rect.flatten().tolist()
                    self.shifted_pub.publish(msg)

                    self.publish_debug_image(curr_rect, ref_rect, grasp_shifted_rect[0])
                    
            except Exception: pass

    def publish_debug_image(self, curr, ref, grasp_target):
        # 1. Prepare Background Image (Camera Feed instead of Black Screen)
        if self.latest_image is not None and self.K_real is not None and self.D_real is not None:
            try:
                # Rectify the fisheye image to match our Linear math
                debug_img = cv2.fisheye.undistortImage(
                    self.latest_image, self.K_real, self.D_real, Knew=self.K_sim, new_size=(640, 480)
                )
            except:
                debug_img = self.latest_image.copy()
        else:
            debug_img = np.zeros((480, 640, 3), dtype=np.uint8)

        def draw_poly(pts, color, thickness=2):
            pts = pts.astype(int).reshape((-1, 1, 2))
            cv2.polylines(debug_img, [pts], True, color, thickness)

        # Draw geometries over the camera feed
        draw_poly(ref, (100, 100, 100), 1)  # Ref (Gray)
        draw_poly(curr, (255, 255, 0), 2)   # Current (Cyan)
        
        g_pt = tuple(grasp_target.astype(int))
        cv2.circle(debug_img, g_pt, 6, (0, 255, 0), -1) 
        
        center = np.mean(curr, axis=0).astype(int)
        cv2.drawMarker(debug_img, tuple(center), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.arrowedLine(debug_img, tuple(center), g_pt, (0, 255, 255), 2)

        msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
        self.debug_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(GraspShiftingNode())
    rclpy.shutdown()

if __name__ == '__main__': main()