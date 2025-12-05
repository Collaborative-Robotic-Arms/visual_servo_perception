#! /usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import JointState
import csv
import time
import math

class SafeDataLogger(Node):
    def __init__(self):
        super().__init__('safe_data_logger')
        
        # --- CONFIGURATION ---
        # Update this to match your robot's command topic if you want 
        # to see "commands" instead of "actuals". 
        # But /joint_states is the best "truth" for safety checks.
        self.joint_topic = '/joint_states' 
        self.num_joints = 6 # AR4 has 6 joints
        # ---------------------

        timestamp = time.strftime("%H%M%S")
        self.filename = f'ibvs_hardware_test_{timestamp}.csv'
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        
        # Header
        header = [
            'time', 
            'error_norm', 
            'fuzzy_depth',
            'cart_vel_x', 'cart_vel_y', 'cart_vel_z', # Cartesian Command
            'cart_ang_x', 'cart_ang_y', 'cart_ang_z'
        ]
        
        # Add dynamic headers for joints
        for i in range(1, self.num_joints + 1):
            header.append(f'joint_{i}_vel')
            
        self.writer.writerow(header)
        
        self.start_time = self.get_clock().now().nanoseconds
        
        # Buffers
        self.latest_error = 0.0
        self.latest_depth = 0.0
        self.latest_twist_linear = [0.0, 0.0, 0.0]
        self.latest_twist_angular = [0.0, 0.0, 0.0]
        self.latest_joint_vels = [0.0] * self.num_joints
        
        # Subscribers
        self.create_subscription(Float64, '/vs/error_norm', self.cb_error, 10)
        self.create_subscription(Float64, '/camera_to_marker_depth', self.cb_depth, 10)
        self.create_subscription(TwistStamped, '/servo_node/delta_twist_cmds', self.cb_twist, 10)
        self.create_subscription(JointState, self.joint_topic, self.cb_joints, 10)
        
        # 20 Hz Logging
        self.timer = self.create_timer(0.05, self.log_data)
        
        self.get_logger().info(f"Logging Safety Data to {self.filename}...")
        self.get_logger().info(f"Monitoring Joint Velocities on {self.joint_topic}")

    def cb_error(self, msg): self.latest_error = msg.data
    def cb_depth(self, msg): self.latest_depth = msg.data
    
    def cb_twist(self, msg):
        self.latest_twist_linear = [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z]
        self.latest_twist_angular = [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z]
        
    def cb_joints(self, msg):
        # Map joint info. Note: msg.velocity might be empty in some sims if not enabled!
        if len(msg.velocity) > 0:
            # We assume the order is J1...J6. 
            # If your robot driver randomizes order, we would need to map by 'name'.
            # For AR4/MoveIt, standard index order usually works.
            count = min(len(msg.velocity), self.num_joints)
            self.latest_joint_vels = list(msg.velocity[:count])
            
            # Pad if fewer joints found
            while len(self.latest_joint_vels) < self.num_joints:
                self.latest_joint_vels.append(0.0)

    def log_data(self):
        current_time = (self.get_clock().now().nanoseconds - self.start_time) / 1e9
        
        row = [current_time, self.latest_error, self.latest_depth] + \
              self.latest_twist_linear + \
              self.latest_twist_angular + \
              self.latest_joint_vels
              
        self.writer.writerow(row)

    def destroy_node(self):
        self.file.close()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = SafeDataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()