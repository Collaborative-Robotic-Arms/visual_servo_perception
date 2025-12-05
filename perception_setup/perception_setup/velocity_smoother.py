#! /usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
import numpy as np

class VelocitySmoother(Node):
    def __init__(self):
        super().__init__('velocity_smoother')
        
        # --- CONFIGURATION ---
        self.input_topic = '/servo_node/delta_twist_cmds_raw'  # From IBVS
        self.output_topic = '/servo_node/delta_twist_cmds' # To Robot
        
        # Acceleration Limit (m/s^2 or rad/s^2)
        # Lower = Smoother but more lag
        # 0.1 is very smooth, 0.5 is snappy, 10.0 is raw
        self.accel_limit_linear = 0.05 
        self.accel_limit_angular = 0.1
        
        # Frequency
        self.rate = 50.0 # Hz (Run faster than the camera!)
        self.dt = 1.0 / self.rate
        # ---------------------

        self.sub = self.create_subscription(
            TwistStamped, self.input_topic, self.cb_cmd, 10)
        
        self.pub = self.create_publisher(
            TwistStamped, self.output_topic, 10)
        
        self.timer = self.create_timer(self.dt, self.control_loop)
        
        # State
        self.target_twist = np.zeros(6) # [vx, vy, vz, wx, wy, wz]
        self.current_twist = np.zeros(6)
        self.header = None # Keep frame_id
        
        self.get_logger().info(f"Smoother Started: {self.input_topic} -> {self.output_topic}")

    def cb_cmd(self, msg):
        # Update the target we WANT to reach
        self.header = msg.header
        self.target_twist = np.array([
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
            msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z
        ])

    def control_loop(self):
        if self.header is None: return

        # Calculate difference between Current and Target
        error = self.target_twist - self.current_twist
        
        # Calculate max step allowed in this timestep
        # step = acceleration * dt
        max_step_lin = self.accel_limit_linear * self.dt
        max_step_ang = self.accel_limit_angular * self.dt
        
        max_steps = np.array([
            max_step_lin, max_step_lin, max_step_lin,
            max_step_ang, max_step_ang, max_step_ang
        ])
        
        # Clamp the change to the max step (Ramp logic)
        delta = np.clip(error, -max_steps, max_steps)
        
        # Update current twist
        self.current_twist += delta
        
        # Publish
        out_msg = TwistStamped()
        out_msg.header = self.header
        out_msg.header.stamp = self.get_clock().now().to_msg()
        
        out_msg.twist.linear.x = self.current_twist[0]
        out_msg.twist.linear.y = self.current_twist[1]
        out_msg.twist.linear.z = self.current_twist[2]
        out_msg.twist.angular.x = self.current_twist[3]
        out_msg.twist.angular.y = self.current_twist[4]
        out_msg.twist.angular.z = self.current_twist[5]
        
        self.pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = VelocitySmoother()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()