#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Float64
from dual_arms_msgs.msg import Brick 
import numpy as np
import cv2

class PnPDepthEstimator(Node):
    def __init__(self):
        super().__init__('depth_estimation_pnp')

        # --- 1. CONFIGURATION: INTRINSICS ---
        # Validated f=190.68 from simulation
        self.fx = 190.68
        self.fy = 190.68
        self.cx = 320.0
        self.cy = 240.0
        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # --- 2. CONFIGURATION: BRICK GEOMETRY ---
        def make_rect(dims):
            L, W = dims
            return np.array([
                [-L/2, -W/2, 0], [ L/2, -W/2, 0],
                [ L/2,  W/2, 0], [-L/2,  W/2, 0]
            ], dtype=np.float32)

        # Updated to match your confirmation (0.06, 0.06 for 255)
        self.brick_geometries = {
            Brick.I_BRICK: make_rect((0.058, 0.030)),
            Brick.L_BRICK: make_rect((0.060, 0.060)),
            Brick.T_BRICK: make_rect((0.090, 0.030)),
            Brick.Z_BRICK: make_rect((0.120, 0.090)),
            255:           make_rect((0.060, 0.060)) 
        }

        # Default State
        self.current_obj_points = self.brick_geometries[Brick.L_BRICK]
        self.current_brick_id = Brick.L_BRICK

        # --- SUBSCRIBERS & PUBLISHERS ---
        self.create_subscription(Brick, '/mission/target_brick', self.cb_mission, 10)
        self.create_subscription(Float64MultiArray, '/feature_coordinates_6D', self.cb_features, 10)
        self.depth_pub = self.create_publisher(Float64, '/camera_to_brick_depth', 10)

        self.get_logger().info("PnP Depth Online. Double-Undistortion Logic Removed.")

    def cb_mission(self, msg):
        if msg.type in self.brick_geometries:
            self.current_obj_points = self.brick_geometries[msg.type]
            self.current_brick_id = msg.type
        else:
            self.get_logger().warn(f"Unknown Brick ID: {msg.type}.")

    def sort_corners(self, pts):
        """Sorts 2D points to match 3D order: TL, TR, BR, BL"""
        s = pts.sum(axis=1)
        tl = pts[np.argmin(s)]
        br = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        tr = pts[np.argmin(diff)]
        bl = pts[np.argmax(diff)]
        return np.array([tl, tr, br, bl], dtype=np.float32)

    def cb_features(self, msg):
        if len(msg.data) < 8: return 

        # Incoming points are ALREADY LINEAR from feature_detection_fisheye.py
        # DO NOT UNDISTORT AGAIN.
        raw_pixels = np.array(msg.data).reshape(-1, 2)
        if len(raw_pixels) != 4: return

        # 1. Sort
        sorted_pixels = self.sort_corners(raw_pixels)

        # 2. Solve PnP
        success, rvec, tvec = cv2.solvePnP(
            self.current_obj_points, 
            sorted_pixels, 
            self.camera_matrix, 
            np.zeros(4),
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            z_depth = float(tvec[2][0]) # Access the scalar value explicitly
            out = Float64()
            out.data = abs(z_depth)
            self.depth_pub.publish(out)

def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(PnPDepthEstimator())
    rclpy.shutdown()

if __name__ == '__main__': main()