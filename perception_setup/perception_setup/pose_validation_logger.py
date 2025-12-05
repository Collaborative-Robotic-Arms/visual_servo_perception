#! /usr/bin/env python3
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
import csv
import time
import math

# --- CONFIGURATION ---
CAMERA_FRAME = "ar4_camera_link"
MARKER_FRAME = "aruco_target_frame" 
# ---------------------

def euler_from_quaternion(x, y, z, w):
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return roll_x, pitch_y, yaw_z

class PoseValidationLogger(Node):
    def __init__(self):
        super().__init__('pose_validator')
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        timestamp = time.strftime("%H%M%S")
        self.filename = f'pose_validation_{timestamp}.csv'
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        
        # Header
        self.writer.writerow(['time', 'x', 'y', 'z', 'roll', 'pitch', 'yaw'])
        
        self.start_time = self.get_clock().now().nanoseconds
        self.timer = self.create_timer(0.05, self.log_pose) # 20Hz
        
        self.get_logger().info(f"Recording Camera -> Marker Pose to {self.filename}...")

    def log_pose(self):
        try:
            # CORRECTED: Look up Marker in Camera Frame (Depth = +X)
            t = self.tf_buffer.lookup_transform(
                CAMERA_FRAME, 
                MARKER_FRAME, 
                rclpy.time.Time())
            
            curr_time = (self.get_clock().now().nanoseconds - self.start_time) / 1e9
            
            trans = t.transform.translation
            rot = t.transform.rotation
            
            # Convert Quat to Euler
            r, p, y = euler_from_quaternion(rot.x, rot.y, rot.z, rot.w)
            
            self.writer.writerow([curr_time, trans.x, trans.y, trans.z, r, p, y])
            
        except Exception as e:
            pass

    def destroy_node(self):
        self.file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = PoseValidationLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()