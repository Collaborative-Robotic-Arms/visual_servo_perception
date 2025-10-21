import faulthandler
faulthandler.enable()

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class Step2(Node):
    def __init__(self):
        super().__init__('step2')
        self.bridge = CvBridge()
        self.create_subscription(Image, '/camera/image_raw', self.cb, 10)
        self.get_logger().info('Step2 subscriber ready')

    def cb(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imshow("test", img)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main():
    rclpy.init()
    node = Step2()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()