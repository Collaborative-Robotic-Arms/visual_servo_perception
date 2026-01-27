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
        
        # --- STATE VARIABLES ---
        self.ref_corners = None       
        self.ref_grasp_pixel = None   
        self.current_corners_rect = None 
        self.pending_grasp_msg = None 
        
        # --- CAMERA INTRINSICS STATE ---
        self.camera_info_received = False
        self.K = np.eye(3)
        self.D = np.zeros(4)

        self.latest_image = None

        # --- SUBSCRIBERS ---
        self.create_subscription(Float64MultiArray, '/feature_coordinates_6D', self.cb_current_corners, 10)
        self.create_subscription(Point, '/grasp/pixel_coords_transformed', self.cb_transformed_grasp, 10)
        self.create_subscription(Brick, '/mission/target_brick', self.cb_mission, 10)
        self.create_subscription(Image, '/cameraAR4/image_raw', self.cb_image, 10)
        self.info_sub = self.create_subscription(CameraInfo, '/cameraAR4/camera_info', self.camera_info_callback, 10)
        
        # --- PUBLISHERS ---
        self.shifted_pub = self.create_publisher(Float64MultiArray, '/visual_servo/shifted_features', 10)
        self.debug_pub = self.create_publisher(Image, '/visual_servo/debug_shift', 10)
        self.seed_pub = self.create_publisher(Point, '/visual_servo/tracker_seed', 10)
        
        self.get_logger().info(f"Grasp Shifter Online (Dynamic Intrinsics).")

    def camera_info_callback(self, msg):
        """ Updates Intrinsics from Topic """
        if not self.camera_info_received:
            self.K = np.array(msg.k).reshape(3, 3)
            self.D = np.array(msg.d)[:4]
            self.camera_info_received = True
            self.get_logger().info(f"Camera Info Received. Debug visualization ready.")

    def cb_image(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except:
            pass

    def cb_mission(self, msg):
        """ Reset logic when a new brick type is selected """
        self.ref_corners = None
        self.pending_grasp_msg = None
        self.ref_grasp_pixel = None

    def cb_transformed_grasp(self, msg): 
        self.pending_grasp_msg = msg
        self.try_latch()

    def cb_current_corners(self, msg): 
        if len(msg.data) >= 8:
            self.current_corners_rect = np.array(msg.data).reshape(-1, 2)
            self.try_latch() 
            self.process_and_publish()
        
    def try_latch(self):
        # We only latch if we have:
        # 1. A pending click from the user/homography
        # 2. Valid current features from the camera
        # 3. We haven't latched already
        if self.ref_corners is not None: return
        if self.pending_grasp_msg is None or self.current_corners_rect is None: return
        
        # 1. Store the Reference Features (The "Shape" at the moment of the click)
        self.ref_corners = self.current_corners_rect.copy()
        
        # 2. Store the Reference Click Point
        # NOTE: The incoming point is ALREADY in Linear Space (from Homography Node)
        # So we just store it directly.
        self.ref_grasp_pixel = np.array([
            [self.pending_grasp_msg.x, self.pending_grasp_msg.y]
        ], dtype=np.float32)

        # 3. Publish Seed (Optional, for other trackers)
        seed_msg = Point()
        seed_msg.x = float(self.pending_grasp_msg.x)
        seed_msg.y = float(self.pending_grasp_msg.y)
        self.seed_pub.publish(seed_msg)
        
        self.get_logger().info(f"LATCHED! Tracking point: {self.ref_grasp_pixel[0]}")

    def process_and_publish(self):
        """ 
        The Core Logic:
        1. Calculate Homography between Ref Corners and Current Corners.
        2. Use that H to move the Grasp Point to its new position.
        3. Calculate the Shift Vector (New Grasp Point - Current Center).
        4. Shift the features sent to the controller.
        """
        if self.ref_corners is not None and self.current_corners_rect is not None:
            try:
                ref_rect = self.ref_corners
                curr_rect = self.current_corners_rect
                grasp_ref = self.ref_grasp_pixel

                # Find H mapping Old Shape -> New Shape
                H, _ = cv2.findHomography(ref_rect, curr_rect, cv2.RANSAC, 5.0)
                
                if H is not None:
                    # Move the grasp point according to how the object moved
                    grasp_shifted = cv2.perspectiveTransform(grasp_ref.reshape(-1,1,2), H).reshape(-1,2)
                    
                    # Calculate Center of the detected object
                    current_center_rect = np.mean(curr_rect, axis=0)
                    
                    # Vector from Center -> Grasp Point
                    shift_vector = grasp_shifted[0] - current_center_rect
                    
                    # Apply this shift to the Features
                    # (This tricks the Visual Servo into aligning the Center with the Grasp Point)
                    final_features_rect = curr_rect + shift_vector
                    
                    msg = Float64MultiArray()
                    msg.data = final_features_rect.flatten().tolist()
                    self.shifted_pub.publish(msg)

                    self.publish_debug_image(curr_rect, ref_rect, grasp_shifted[0])
                    
            except Exception: pass

    def publish_debug_image(self, curr, ref, grasp_target):
        # 1. Prepare Background Image
        if self.latest_image is not None and self.camera_info_received:
            try:
                # CRITICAL: Undistort the background so it matches our Linear features
                # using the dynamically loaded K matrix.
                debug_img = cv2.fisheye.undistortImage(
                    self.latest_image, self.K, self.D, Knew=self.K
                )
            except:
                debug_img = self.latest_image.copy()
        else:
            if self.latest_image is not None:
                debug_img = self.latest_image.copy() # Fallback if info not yet received
            else:
                debug_img = np.zeros((480, 640, 3), dtype=np.uint8)

        def draw_poly(pts, color, thickness=2):
            pts = pts.astype(int).reshape((-1, 1, 2))
            cv2.polylines(debug_img, [pts], True, color, thickness)

        # 2. Draw Geometries
        draw_poly(ref, (100, 100, 100), 1)  # Reference Shape (Gray)
        draw_poly(curr, (255, 255, 0), 2)   # Current Shape (Cyan)
        
        # Grasp Target (Green Dot)
        g_pt = tuple(grasp_target.astype(int))
        cv2.circle(debug_img, g_pt, 6, (0, 255, 0), -1) 
        
        # Center of Object
        center = np.mean(curr, axis=0).astype(int)
        cv2.drawMarker(debug_img, tuple(center), (0, 255, 255), cv2.MARKER_CROSS, 20, 2)
        
        # Arrow from Center -> Grasp
        cv2.arrowedLine(debug_img, tuple(center), g_pt, (0, 255, 255), 2)

        # 3. Publish
        msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
        self.debug_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(GraspShiftingNode())
    rclpy.shutdown()

if __name__ == '__main__': main()