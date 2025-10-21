import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Twist
from std_msgs.msg import Float64
import numpy as np

class DepthEstimator(Node):
    def __init__(self):
        super().__init__('depth_estimator_node')

        # --- Parameters ---
        self.declare_parameter('observer_gain', 1.0) # Gain lambda
        self.declare_parameter('initial_depth', 1.0) # Initial guess for Z
        # Camera intrinsic parameters (replace with your camera's values)
        self.declare_parameter('camera.fx', 381.36) # Focal length in x
        self.declare_parameter('camera.fy', 381.36) # Focal length in y
        self.declare_parameter('camera.cx', 320.5)  # Principal point x
        self.declare_parameter('camera.cy', 240.5)  # Principal point y

        # --- State Variables ---
        self.latest_feature = None
        self.previous_feature = None
        self.latest_velocity_cmd = None
        self.depth_estimate = self.get_parameter('initial_depth').value
        
        # --- Subscribers ---
        self.feature_sub = self.create_subscription(
            PointStamped, '/feature_coordinates', self.feature_callback, 10)
        self.velocity_sub = self.create_subscription(
            Twist, '/cmd_vel', self.velocity_callback, 10)
        
        # --- Publisher ---
        self.depth_pub = self.create_publisher(Float64, '/estimated_depth', 10)

        # --- Timer for the main observer loop ---
        timer_period = 0.05  # seconds (20 Hz)
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.dt = timer_period
        self.get_logger().info('Depth Estimator node has been started.')

    def feature_callback(self, msg):
        """Stores the latest feature coordinates and timestamp."""
        # Store the current message, and shift the previous one
        self.previous_feature = self.latest_feature
        self.latest_feature = msg

    def velocity_callback(self, msg):
        """Stores the latest velocity command."""
        self.latest_velocity_cmd = msg

    def timer_callback(self):
        """Main logic for the dynamic observer."""
        if self.latest_feature is None or self.previous_feature is None or self.latest_velocity_cmd is None:
            # Not enough data to compute yet
            return

        # --- 1. Calculate Measured Feature Velocity (s_dot) ---
        # Time difference
        t_now = self.latest_feature.header.stamp.sec + self.latest_feature.header.stamp.nanosec * 1e-9
        t_prev = self.previous_feature.header.stamp.sec + self.previous_feature.header.stamp.nanosec * 1e-9
        dt_feature = t_now - t_prev
        
        if dt_feature <= 1e-6: # Avoid division by zero
            return
            
        u_dot = (self.latest_feature.point.x - self.previous_feature.point.x) / dt_feature
        v_dot = (self.latest_feature.point.y - self.previous_feature.point.y) / dt_feature
        s_dot_measured = np.array([u_dot, v_dot])
        
        # --- 2. Get Normalized Coordinates and Inputs ---
        fx = self.get_parameter('camera.fx').value
        fy = self.get_parameter('camera.fy').value
        cx = self.get_parameter('camera.cx').value
        cy = self.get_parameter('camera.cy').value
        
        # Convert pixel coordinates to normalized image coordinates
        u = (self.latest_feature.point.x - cx) / fx
        v = (self.latest_feature.point.y - cy) / fy

        # Get the commanded velocity
        vc = self.latest_velocity_cmd
        camera_velocity = np.array([vc.linear.x, vc.linear.y, vc.linear.z, 
                                    vc.angular.x, vc.angular.y, vc.angular.z])

        # --- 3. Calculate Predicted Feature Velocity (s_hat_dot) ---
        Z = self.depth_estimate
        L_s = np.array([
            [-1/Z, 0,    u/Z, u*v,       -(1 + u**2), v],
            [0,    -1/Z, v/Z, 1 + v**2,  -u*v,       -u]
        ])
        
        # Here we scale the jacobian by focal lengths for pixel velocity
        L_s_pixel = np.array([[fx, 0], [0, fy]]) @ L_s
        s_dot_predicted = L_s_pixel @ camera_velocity

        # --- 4. Observer Update Law ---
        error = s_dot_measured - s_dot_predicted
        gain = self.get_parameter('observer_gain').value
        
        # This is the simplified observer update term (from the paper by Espiau et al.)
        Lz = L_s_pixel[:, 2] # The 3rd column of the Jacobian relates to Vz
        z_dot = gain * Z * np.dot(Lz, error)

        # --- 5. Integrate and Publish ---
        self.depth_estimate += z_dot * self.dt
        
        # Publish the new estimate
        depth_msg = Float64()
        depth_msg.data = self.depth_estimate
        self.depth_pub.publish(depth_msg)

def main(args=None):
    rclpy.init(args=args)
    depth_estimator = DepthEstimator()
    rclpy.spin(depth_estimator)
    depth_estimator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()