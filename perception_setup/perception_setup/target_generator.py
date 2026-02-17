#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
import numpy as np
import math
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Float64MultiArray, Int32, Bool
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
from dual_arms_msgs.msg import Brick

class VisualServoManager(Node):
    def __init__(self):
        super().__init__('visual_servo_manager')
        self.bridge = CvBridge()
        
        # --- CONFIGURATION ---
        self.declare_parameter('desired_depth', 0.3)
        self.declare_parameter('box_scale_multiplier', 1.0)
        self.declare_parameter('smart_orientation_check', True)
        self.declare_parameter('hand_eye_offset', 0.090556) # The 0.085 offset
        
        # --- HARDCODED CAMERA INTRINSICS ---
        self.fx = 190.68; self.fy = 190.68
        self.cx = 320.0;  self.cy = 240.0
        self.K = np.array([[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]])
        self.D = np.zeros(4) 

        # Brick Dimensions
        self.brick_config = {
            Brick.I_BRICK: (0.058, 0.030),
            Brick.L_BRICK: (0.060, 0.060),
            Brick.T_BRICK: (0.090, 0.030),
            Brick.Z_BRICK: (0.120, 0.090),
            255: (0.050, 0.020)
        }
        self.current_brick_dims = (0.050, 0.020)
        
        # --- STATE ---
        self.locked_angle = 0.0
        self.latest_image = None
        self.grasping_enabled = False
        self.brick_centroid = None
        self.grasp_centroid = None

        # --- CACHING ---
        self.target_computed = False
        self.cached_feats = None
        self.cached_offset = 0.0
        self.cached_sign = 0.0

        # --- TESTING ---
        self.test_mode_active = False
        self.fake_grasp_delta = None 

        # --- SUBSCRIBERS ---
        self.create_subscription(Brick, '/mission/target_brick', self.cb_mission, 10)
        self.create_subscription(Int32, '/grasp/target_index', self.cb_grasp_trigger, 10)
        self.create_subscription(Image, '/cameraAR4/image_raw', self.cb_image, qos_profile_sensor_data)
        self.create_subscription(Float64MultiArray, '/feature_coordinates_6D', self.cb_raw_features, 10)
        self.create_subscription(Float64MultiArray, '/visual_servo/shifted_features', self.cb_shifted_features, 10)
        self.create_subscription(Point, '/test/fake_grasp', self.cb_test_grasp, 10)
        
        # --- PUBLISHERS ---
        self.desired_pub = self.create_publisher(Float64MultiArray, '/visual_servo/desired_features', 10)
        self.debug_pub = self.create_publisher(Image, '/visual_servo/debug_goal', 10)
        self.reset_roi_pub = self.create_publisher(Bool, '/visual_servo/reset_roi', 10)
        
        self.create_timer(0.1, self.timer_callback)
        self.get_logger().info("Target Generator Online. Logic: Fixed Rotation + Dynamic Offset.")

    def cb_image(self, msg):
        try: self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: pass

    def cb_mission(self, msg):
        if msg.type in self.brick_config:
            self.current_brick_dims = self.brick_config[msg.type]
            self.reset_roi_pub.publish(Bool(data=True))
            self.target_computed = False
            self.get_logger().info("New Brick Type received. Resetting Target.")

    def cb_grasp_trigger(self, msg):
        self.grasping_enabled = (msg.data >= 0)
        self.target_computed = False
        self.get_logger().info("New Grasp Trigger received. Resetting Target.")

    def cb_raw_features(self, msg):
        if len(msg.data) >= 8:
            pts = np.array(msg.data).reshape(-1, 2)
            self.brick_centroid = np.mean(pts, axis=0)

    def cb_shifted_features(self, msg):
        if len(msg.data) >= 8:
            pts = np.array(msg.data).reshape(-1, 2)
            self.grasp_centroid = np.mean(pts, axis=0)

    def cb_test_grasp(self, msg):
        self.get_logger().info(f"TEST TRIGGERED: Simulating Delta dx={msg.x:.0f}, dy={msg.y:.0f}")
        self.fake_grasp_delta = np.array([msg.x, msg.y])
        self.test_mode_active = True
        self.target_computed = False

    def determine_pose_config(self):
        """ 
        Returns: (Rotation_Angle, Offset_Sign, Success)
        """
        if not self.get_parameter('smart_orientation_check').value:
            return 1.5708, 1.0, True 

        dx, dy = 0.0, 0.0
        valid_data = False

        if self.test_mode_active and self.fake_grasp_delta is not None:
            dx, dy = self.fake_grasp_delta
            valid_data = True
        elif self.brick_centroid is not None and self.grasp_centroid is not None:
            delta = self.grasp_centroid - self.brick_centroid
            dx = delta[0]
            dy = delta[1]
            valid_data = True
        
        if not valid_data:
            return 1.5708, 1.0, False 

        # --- UPDATED LOGIC ---
        if abs(dx) > abs(dy):
            # Horizontal (Left or Right) -> Always +90 deg
            rotation =  1.5708
            if dx < 0:
                # LEFT (matches "90" case) -> ADD offset
                sign = -1.0
            else:
                # RIGHT (matches "-90" case) -> SUBTRACT offset
                sign =  1.0
        else:
            # Vertical (Top or Bottom) -> Always 0 deg
            rotation =  0.0
            if dy < 0:
                # UP (matches "0" case) -> ADD offset
                sign = -1.0
            else:
                # DOWN (matches "180" case) -> SUBTRACT offset
                sign =  1.0
                
        return rotation, sign, True

    def get_linear_features(self, L, W, angle_rad):
        Z = self.get_parameter('desired_depth').value
        scale = self.get_parameter('box_scale_multiplier').value
        base_offset = self.get_parameter('hand_eye_offset').value
        
        # Get rotation and sign from logic
        rot_offset, offset_sign, success = self.determine_pose_config()
        if not success and not self.test_mode_active:
            return None, None, None, 0.0, False

        # 1. Apply Rotation
        total_angle = angle_rad + rot_offset
        L *= scale; W *= scale
        
        raw_corners = np.array([[-L/2, -W/2, 0], [ L/2, -W/2, 0], [ L/2,  W/2, 0], [-L/2,  W/2, 0]], dtype=np.float32)
        c, s = np.cos(total_angle), np.sin(total_angle)
        R = np.array([[c, -s, 0], [s,  c, 0], [0,  0, 1]])
        rotated_corners = np.dot(raw_corners, R.T)
        
        # 2. Apply Offset (Sign * Base_Value)
        final_off_y = offset_sign * base_offset
        
        final_points_3d = rotated_corners + np.array([0.0, final_off_y, Z])
        
        pixel_coords, _ = cv2.projectPoints(final_points_3d, np.zeros(3), np.zeros(3), self.K, self.D)
        feats_pix = pixel_coords.reshape(-1, 2)
        
        return feats_pix, final_points_3d, rot_offset, final_off_y, True

    def publish_debug_image(self, feats_pix, debug_img, rot_offset, applied_offset):
        if feats_pix is None: return

        pixels = feats_pix.astype(int)
        for i in range(4):
            cv2.line(debug_img, tuple(pixels[i]), tuple(pixels[(i+1)%4]), (0, 255, 255), 2)
            cv2.putText(debug_img, str(i), tuple(pixels[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        mode = "LOCKED" if self.target_computed else "WAITING"
        info = f"{mode} | Rot: {math.degrees(rot_offset):.0f} | Off: {applied_offset:.3f}"
        cv2.putText(debug_img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
        self.debug_pub.publish(msg)

    def timer_callback(self):
        if self.latest_image is not None:
            try: debug_img = cv2.fisheye.undistortImage(self.latest_image, self.K, self.D, Knew=self.K, new_size=(640, 480))
            except: debug_img = self.latest_image.copy()
        else:
            debug_img = np.zeros((480, 640, 3), dtype=np.uint8)

        if not self.target_computed:
            feats, _, rot, off, success = self.get_linear_features(
                self.current_brick_dims[0], 
                self.current_brick_dims[1], 
                self.locked_angle
            )
            
            if success:
                self.cached_feats = feats
                self.cached_offset = rot
                self.cached_sign = off
                self.target_computed = True
                self.get_logger().info(f"Target Computed: Rot {math.degrees(rot):.0f}, Offset {off:.3f}")
        
        if self.target_computed and self.cached_feats is not None:
            out_msg = Float64MultiArray()
            out_msg.data = self.cached_feats.flatten().tolist()
            self.desired_pub.publish(out_msg)
            self.publish_debug_image(self.cached_feats, debug_img, self.cached_offset, self.cached_sign)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(VisualServoManager())
    rclpy.shutdown()

if __name__ == '__main__': main()