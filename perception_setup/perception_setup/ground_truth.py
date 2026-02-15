#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float64MultiArray, Float64
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs 
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
import math

class VisualServoDebugger(Node):
    def __init__(self):
        super().__init__('ground_truth_lookat_debug')
        
        # --- CONFIGURATION ---
        self.declare_parameter('gt_topic', '/model/L_brick_1/pose') 
        self.declare_parameter('camera_frame', 'ar4_camera_optical_link')
        
        # Manual Offset (World -> Table)
        self.table_offset_x = 0.3325
        self.table_offset_y = 0.0
        self.table_offset_z = 1.1
        
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        # INTRINSICS
        self.fx = 190.68; self.fy = 190.68
        self.cx = 320.0;  self.cy = 240.0
        
        # Brick Geometry
        self.brick_L = 0.060; self.brick_W = 0.060; self.brick_H = 0.056 
        z_surf = self.brick_H
        self.model_corners_3d = np.array([
            [0.0,          0.0,          z_surf], 
            [self.brick_L, 0.0,          z_surf],
            [self.brick_L, self.brick_W, z_surf],
            [0.0,          self.brick_W, z_surf]
        ], dtype=np.float32)

        self.gt_pose_raw = None 
        self.latest_image = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()

        self.create_subscription(Pose, self.get_parameter('gt_topic').value, self.cb_gt_pose, qos_profile_sensor_data)
        self.create_subscription(Image, '/cameraAR4/image_raw', self.cb_image, 10)
        self.create_subscription(Float64, '/camera_to_brick_depth', self.cb_depth, 10)
        self.debug_pub = self.create_publisher(Image, '/ground_truth/debug_view', 10)

        self.create_timer(0.5, self.validate_loop)
        self.get_logger().info("Auto-LookAt DEBUG Online. Ignoring TF Orientation.")

    def cb_gt_pose(self, msg): self.gt_pose_raw = msg
    def cb_depth(self, msg): self.latest_est_depth = msg.data
    def cb_image(self, msg): 
        try: self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except: pass

    def get_rpy_from_matrix(self, matrix):
        r = R.from_matrix(matrix)
        return r.as_euler('xyz', degrees=True)

    def get_rpy_from_quat(self, q):
        # Safety check for zero quaternion
        if abs(q.w) < 1e-6 and abs(q.x) < 1e-6 and abs(q.y) < 1e-6 and abs(q.z) < 1e-6:
            return [0.0, 0.0, 0.0]
        try:
            r = R.from_quat([q.x, q.y, q.z, q.w])
            return r.as_euler('xyz', degrees=True)
        except:
            return [0.0, 0.0, 0.0]

    def project_fisheye_stereographic(self, corners_cam_frame):
        pixels = []
        for pt in corners_cam_frame:
            x, y, z = pt
            if z <= 0.001: pixels.append([-1000, -1000]); continue
            xn = x/z; yn = y/z
            r_linear = np.sqrt(xn**2 + yn**2)
            theta = np.arctan(r_linear)
            r_fish = 2.0 * np.tan(theta/2.0)
            scale = 1.0 if r_linear < 1e-6 else (r_fish / r_linear)
            u = self.fx * (xn * scale) + self.cx
            v = self.fy * (yn * scale) + self.cy
            pixels.append([u, v])
        return np.array(pixels)

    def validate_loop(self):
        if self.gt_pose_raw is None: return

        print("\n--- DEBUG CYCLE START ---")

        # 1. Get Brick Position (Table Frame)
        raw = self.gt_pose_raw
        print(f"[1] GAZEBO RAW: X={raw.position.x:.3f}, Y={raw.position.y:.3f}, Z={raw.position.z:.3f}")

        bx = self.gt_pose_raw.position.x - self.table_offset_x
        by = self.gt_pose_raw.position.y - self.table_offset_y
        bz = self.gt_pose_raw.position.z - self.table_offset_z
        P_brick = np.array([bx, by, bz])
        print(f"[2] BRICK (Table Frame): X={bx:.3f}, Y={by:.3f}, Z={bz:.3f}")

        # 2. Get Camera Position (Table Frame) from TF
        tgt = self.get_parameter('camera_frame').value
        try:
            if not self.tf_buffer.can_transform(tgt, 'abb_table', rclpy.time.Time()): 
                print("[ERROR] No TF Transform found.")
                return
            
            # Create a zero-point in Camera Frame
            p_cam_origin = PoseStamped()
            p_cam_origin.header.frame_id = tgt
            p_cam_origin.header.stamp = rclpy.time.Time().to_msg()
            
            # CRITICAL FIX: Initialize Quaternion to Identity (w=1) to prevent crash!
            p_cam_origin.pose.orientation.w = 1.0 
            
            # Transform Camera Origin -> Table Frame
            cam_in_table = self.tf_buffer.transform(p_cam_origin, 'abb_table', timeout=rclpy.duration.Duration(seconds=0.1))
            cx = cam_in_table.pose.position.x
            cy = cam_in_table.pose.position.y
            cz = cam_in_table.pose.position.z
            P_cam = np.array([cx, cy, cz])
            
            # Debug Print for TF Orientation
            tf_rpy = self.get_rpy_from_quat(cam_in_table.pose.orientation)
            print(f"[3] CAM (Table Frame): X={cx:.3f}, Y={cy:.3f}, Z={cz:.3f} | TF_RPY={tf_rpy}")
            
        except Exception as e: 
            print(f"[ERROR] TF Exception: {e}")
            return

        # 3. AUTO-LOOKAT LOGIC
        # Vector from Cam to Brick
        V_forward = P_brick - P_cam
        dist = np.linalg.norm(V_forward)
        
        # Construct Rotation Matrix (Z-forward pointing at brick)
        z_axis = V_forward / dist
        
        # Heuristic: Global Y (0,1,0) as "Up"
        x_axis = np.cross(z_axis, np.array([0, 1, 0])) 
        if np.linalg.norm(x_axis) < 0.1: 
             x_axis = np.cross(z_axis, np.array([1, 0, 0]))
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # R_world_to_cam = [X.T, Y.T, Z.T]
        R_lookat = np.vstack([x_axis, y_axis, z_axis]) # 3x3
        
        lookat_rpy = self.get_rpy_from_matrix(R_lookat.T) 
        print(f"[4] LOOKAT CALC: Dist={dist:.4f}m | Synced RPY={lookat_rpy}")

        # 4. PROJECT POINTS
        gt_pixels = []
        for pt_local in self.model_corners_3d:
            # Point in Table Frame
            P_pt_table = P_brick + pt_local 
            
            # Vector Cam -> Pt
            V_pt = P_pt_table - P_cam
            
            # Rotate into Camera Frame
            P_pt_cam = R_lookat @ V_pt
            
            # Project
            pixels = self.project_fisheye_stereographic([P_pt_cam])
            gt_pixels.append(pixels[0])

        # 5. VISUALIZE
        if self.latest_image is not None:
            debug_img = self.latest_image.copy()
            for i in range(4):
                p1 = tuple(np.array(gt_pixels[i]).astype(int))
                p2 = tuple(np.array(gt_pixels[(i+1)%4]).astype(int))
                cv2.line(debug_img, p1, p2, (0, 255, 0), 2)
                cv2.circle(debug_img, p1, 4, (0, 255, 0), -1)

            cv2.putText(debug_img, f"True Dist: {dist:.3f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            try:
                est_d = getattr(self, 'latest_est_depth', 0.0)
                cv2.putText(debug_img, f"Est Depth: {est_d:.3f}m", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                print(f"[5] VALIDATION: True={dist:.3f} | Est={est_d:.3f} | Diff={abs(dist-est_d):.3f}")
            except: pass
            
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(debug_img, "bgr8"))

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(VisualServoDebugger())
    rclpy.shutdown()

if __name__ == '__main__': main()