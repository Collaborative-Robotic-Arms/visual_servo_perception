import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ImageProcessor(Node):
    def __init__(self):
        super().__init__('image_processor')
        self.bridge = CvBridge()
        
        # Subscribe to the raw image topic published by the camera driver
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.get_logger().info("Image processor node started, subscribing to /camera/image_raw")

    def image_callback(self, msg):
        """Callback function to process incoming image messages."""
        try:
            # Convert the ROS Image message to an OpenCV image (numpy array)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # --- YOUR OBJECT DETECTION LOGIC GOES HERE ---
        # For example, you could detect features, run a model, etc.
        # For now, we'll just draw a simple rectangle as a placeholder.
        
        h, w, _ = cv_image.shape
        cv2.rectangle(cv_image, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
        cv2.putText(cv_image, "Processing Placeholder", (w//4 + 5, h//4 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # --- Display the processed image (for debugging) ---
        # Note: cv2.imshow() can cause issues with ROS 2's threading model.
        # It's better to publish the processed image to a new topic for visualization.
        # However, for simple debugging, it's often acceptable.
        cv2.imshow("Processed Image", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    image_processor = ImageProcessor()
    try:
        rclpy.spin(image_processor)
    except KeyboardInterrupt:
        pass
    finally:
        image_processor.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()