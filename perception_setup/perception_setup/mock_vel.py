#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import numpy as np
import time

class MockVelocityPublisher(Node):
    def __init__(self):
        super().__init__('mock_velocity_publisher_node')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Publish at 20Hz to match your camera framerate roughly
        timer_period = 0.05 
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.start_time = time.time()
        
        self.get_logger().info("Mock Velocity Publisher started. Publishing smooth sine-wave velocities.")

    def timer_callback(self):
        msg = Twist()
        
        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        
        # --- GENERATE SMOOTH TEST MOTION ---
        # Random noise (0-100) breaks estimators. 
        # We use sine waves to simulate a robot driving forward/back and turning.
        
        # Linear X: Oscillates between -0.5 m/s and 0.5 m/s
        msg.linear.x = 0.5 * np.sin(elapsed)
        
        # Linear Y: Usually 0 for non-holonomic robots, but we can add a tiny bit of noise
        msg.linear.y = 0.0
        
        # Linear Z: Robots don't usually fly up/down, keep 0
        msg.linear.z = 0.0
        
        # Angular Z: Oscillates to simulate turning
        msg.angular.z = 0.2 * np.cos(elapsed)
        
        self.publisher_.publish(msg)
        
        # Log periodically, not every single time (to reduce spam)
        # self.get_logger().info(f'Vel: LinX={msg.linear.x:.2f}, AngZ={msg.angular.z:.2f}')

def main(args=None):
    rclpy.init(args=args)
    node = MockVelocityPublisher()
    try:
        # rclpy.spin() is what makes this loop infinitely
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()