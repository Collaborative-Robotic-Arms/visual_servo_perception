#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float64MultiArray, Float64
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

class VSVisualizer(Node):
    def __init__(self):
        super().__init__('vs_visualizer')
        
        # Camera Intrinsics (Must match your controller)
        self.f_sim = 190.68
        self.cx, self.cy = 320.0, 240.0
        
        # State
        self.current_depth = 0.3
        self.target_depth = 0.2
        
        # Subscribers
        self.create_subscription(Float64MultiArray, '/visual_servo/desired_features', self.cb_desired, 10)
        self.create_subscription(Float64MultiArray, '/visual_servo/shifted_features', self.cb_current, 10)
        self.create_subscription(Float64, '/camera_to_brick_depth', self.cb_depth, 10)
        
        # Publisher
        self.marker_pub = self.create_publisher(MarkerArray, '/visual_servo/rviz_markers', 10)
        
        self.get_logger().info("RViz Visualizer Online. Add /visual_servo/rviz_markers (MarkerArray) in RViz.")

    def cb_depth(self, msg):
        self.current_depth = msg.data

    def project_to_3d(self, pixels, z):
        """Projects pixels back to ar4_ee_link 3D space"""
        pts_3d = []
        for i in range(0, len(pixels), 2):
            u, v = pixels[i], pixels[i+1]
            x = (u - self.cx) * z / self.f_sim
            y = (v - self.cy) * z / self.f_sim
            pts_3d.append([x, y, z])
        return pts_3d

    def create_markers(self, pts, color, ns, is_line=True):
        markers = []
        # Points/IDs
        for i, pt in enumerate(pts):
            m = Marker()
            m.header.frame_id = "ar4_ee_link"
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = f"{ns}_ids"
            m.id = i
            m.type = Marker.TEXT_VIEW_FACING
            m.action = Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = pt[0], pt[1], pt[2]
            m.scale.z = 0.02
            m.color.r, m.color.g, m.color.b, m.color.a = color
            m.text = str(i)
            markers.append(m)

        # Polygon Lines
        if is_line:
            line = Marker()
            line.header.frame_id = "ar4_ee_link"
            line.type = Marker.LINE_STRIP
            line.action = Marker.ADD
            line.ns = f"{ns}_lines"
            line.id = 100
            line.scale.x = 0.005
            line.color.r, line.color.g, line.color.b, line.color.a = color
            for pt in pts:
                p = Point()
                p.x, p.y, p.z = pt[0], pt[1], pt[2]
                line.points.append(p)
            # Close the loop
            p_start = Point()
            p_start.x, p_start.y, p_start.z = pts[0][0], pts[0][1], pts[0][2]
            line.points.append(p_start)
            markers.append(line)
        return markers

    def cb_desired(self, msg):
        pts = self.project_to_3d(msg.data, self.target_depth)
        ma = MarkerArray()
        ma.markers = self.create_markers(pts, (1.0, 1.0, 0.0, 1.0), "goal") # Yellow
        self.marker_pub.publish(ma)

    def cb_current(self, msg):
        pts = self.project_to_3d(msg.data, self.current_depth)
        ma = MarkerArray()
        ma.markers = self.create_markers(pts, (0.0, 1.0, 0.0, 1.0), "current") # Green
        self.marker_pub.publish(ma)

def main():
    rclpy.init()
    rclpy.spin(VSVisualizer())
    rclpy.shutdown()

if __name__ == '__main__': main()