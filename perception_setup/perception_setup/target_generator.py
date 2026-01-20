#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import math
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Float64MultiArray, Int32
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CameraInfo
from dual_arms_msgs.msg import Brick

class VisualServoManager(Node):
    def __init__(self):
        super().__init__('visual_servo_manager')
        self.bridge = CvBridge()
        
        # --- CONFIGURATION ---
        self.declare_parameter('desired_depth', 0.2)
        self.declare_parameter('box_scale_multiplier', 1.0)
        self.declare_parameter('snap_to_90_deg', True) 
        
        # --- INTRINSICS (Linear Model) ---
        self.f_sim = 190.68
        self.K_sim = np.array([[self.f_sim, 0.0, 320.0], [0.0, self.f_sim, 240.0], [0.0, 0.0, 1.0]])
        self.D_zero = np.zeros(4)

        # --- REAL CAMERA INTRINSICS ---
        self.K_real = None
        self.D_real = None

        self.brick_config = {
            Brick.I_BRICK: (0.058, 0.030),
            Brick.L_BRICK: (0.060, 0.060),
            Brick.T_BRICK: (0.090, 0.030),
            Brick.Z_BRICK: (0.120, 0.090),
            255: (0.050, 0.020)
        }
        
        self.current_brick_dims = (0.050, 0.020)
        self.locked_angle = None      
        self.is_angle_locked = False
        self.grasping_enabled = False
        self.latest_image = None

        # --- SUBSCRIBERS ---
        self.create_subscription(Brick, '/mission/target_brick', self.cb_mission, 10)
        self.create_subscription(Int32, '/grasp/target_index', self.cb_grasp_trigger, 10)
        self.create_subscription(Point, '/grasp/pixel_coords', self.cb_grasp_update, 10)
        self.create_subscription(Image, '/cameraAR4/image_raw', self.cb_image, 10)
        self.create_subscription(CameraInfo, '/cameraAR4/camera_info', self.cb_cam_info, 10)
        
        # --- PUBLISHERS ---
        self.desired_pub = self.create_publisher(Float64MultiArray, '/visual_servo/desired_features', 10)
        self.debug_pub = self.create_publisher(Image, '/visual_servo/debug_goal', 10)
        
        self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("Target Generator Online. Visualizing IDs on Debug Feed.")

    def cb_cam_info(self, msg):
        if self.K_real is None:
            self.K_real = np.array(msg.k).reshape(3, 3)
            self.D_real = np.array(msg.d)

    def cb_image(self, msg):
        try: self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: pass

    def cb_mission(self, msg):
        if msg.type in self.brick_config:
            self.current_brick_dims = self.brick_config[msg.type]
            self.is_angle_locked = False

    def cb_grasp_trigger(self, msg):
        self.grasping_enabled = (msg.data >= 0)

    def cb_grasp_update(self, msg):
        if not self.grasping_enabled or self.is_angle_locked: return
        final_angle = msg.z
        if self.get_parameter('snap_to_90_deg').value:
            step = math.pi / 2.0
            final_angle = round(msg.z / step) * step
        self.locked_angle = final_angle
        self.is_angle_locked = True

    def get_linear_features(self, L, W, angle_rad):
        Z = self.get_parameter('desired_depth').value
        scale = self.get_parameter('box_scale_multiplier').value
        off_x = 0.0; off_y = 0.090556
        rot_offset = 1.5708 # 90-degree grasp offset
        
        total_angle = angle_rad + rot_offset
        L *= scale; W *= scale
        
        # 1. Define base corners
        raw_corners = np.array([[-L/2, -W/2, 0], [ L/2, -W/2, 0], [ L/2,  W/2, 0], [-L/2,  W/2, 0]], dtype=np.float32)
        
        # 2. Apply Rotation
        c, s = np.cos(total_angle), np.sin(total_angle)
        R = np.array([[c, -s, 0], [s,  c, 0], [0,  0, 1]])
        rotated_corners = np.dot(raw_corners, R.T)
        final_points_3d = rotated_corners + np.array([off_x, off_y, Z])

        # 3. Project to pixels
        pixel_coords, _ = cv2.projectPoints(final_points_3d, np.zeros(3), np.zeros(3), self.K_sim, self.D_zero)
        feats_pix = pixel_coords.reshape(-1, 2)

        # --- THE KEY FIX: INDEX SHIFTING ---
        # We 'roll' the features so that physical corner 0 is now 90 degrees offset.
        # This forces the ViSP controller to rotate the wrist.
        feats_pix = np.roll(feats_pix, shift=1, axis=0)
        # -----------------------------------
        
        return feats_pix, final_points_3d

    def publish_debug_image(self, feats_pix, debug_img):
        pixels = feats_pix.astype(int)
        
        for i in range(4):
            p1 = tuple(pixels[i])
            p2 = tuple(pixels[(i+1)%4])
            # Draw the Goal Box
            cv2.line(debug_img, p1, p2, (0, 255, 255), 2)
            # DRAW THE CORNER ID
            cv2.putText(debug_img, str(i), p1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
        msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
        self.debug_pub.publish(msg)

    def timer_callback(self):
        # Prepare Background
        if self.latest_image is not None and self.K_real is not None and self.D_real is not None:
            try:
                debug_img = cv2.fisheye.undistortImage(
                    self.latest_image, self.K_real, self.D_real, Knew=self.K_sim, new_size=(640, 480)
                )
            except: debug_img = self.latest_image.copy()
        else:
            debug_img = np.zeros((480, 640, 3), dtype=np.uint8)

        if not self.is_angle_locked or self.current_brick_dims is None:
            cv2.putText(debug_img, "WAITING...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8"))
            return

        # Generate, Shift, and Publish
        feats_pix, _ = self.get_linear_features(self.current_brick_dims[0], self.current_brick_dims[1], self.locked_angle)
        
        out_msg = Float64MultiArray()
        out_msg.data = feats_pix.flatten().tolist()
        self.desired_pub.publish(out_msg)
        
        self.publish_debug_image(feats_pix, debug_img)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(VisualServoManager())
    rclpy.shutdown()

if __name__ == '__main__': main()