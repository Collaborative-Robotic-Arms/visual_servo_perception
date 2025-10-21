import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import threading
import time
import queue

class VideoStream:
    # ... (Your VideoStream class code is perfect, no changes needed here) ...
    def __init__(self, rtsp_url):
        self.RTSP_URL = rtsp_url
        self.RECONNECT_DELAY = 3
        self.MAX_RETRIES = 5
        self.FRAME_TIMEOUT = 2.0
        self.FRAME_WIDTH = 320
        self.FRAME_HEIGHT = 240
        self.TARGET_FPS = 20
        self.cap = None
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.thread = None
        self.connect()

    def connect(self):
        for attempt in range(self.MAX_RETRIES):
            try:
                if self.cap is not None:
                    self.cap.release()
                print(f"Connecting to RTSP stream (attempt {attempt + 1})...")
                self.cap = cv2.VideoCapture(f"{self.RTSP_URL}?tcp", cv2.CAP_FFMPEG)
                if not self.cap.isOpened():
                    raise RuntimeError("Failed to open stream")
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.FRAME_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.FRAME_HEIGHT)
                self.cap.set(cv2.CAP_PROP_FPS, self.TARGET_FPS)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                print("RTSP stream connected successfully")
                return True
            except Exception as e:
                print(f"Connection failed: {str(e)}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RECONNECT_DELAY)
        return False

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._read_frames, daemon=True)
            self.thread.start()
            return True
        return False

    def stop(self):
        self.running = False
        if self.thread is not None:
            self.thread.join()
        if self.cap is not None:
            self.cap.release()
        print("Video stream stopped.")

    def _read_frames(self):
        last_frame_time = time.time()
        while self.running:
            if not self.cap.isOpened():
                print("Stream is not open, attempting to reconnect...")
                if not self.connect():
                    time.sleep(self.RECONNECT_DELAY)
                    continue
            ret, frame = self.cap.read()
            if not ret or frame is None:
                if time.time() - last_frame_time > self.FRAME_TIMEOUT:
                    print("⚠️ Frame timeout reached, attempting reconnect...")
                    self.cap.release()
                continue
            last_frame_time = time.time()
            if self.frame_queue.full():
                self.frame_queue.get_nowait()
            self.frame_queue.put(frame)
            time.sleep(1 / (self.TARGET_FPS * 2))

    def get_frame(self, timeout=0.5):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None


# --- The Simplified ROS 2 Node ---
class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        
        self.declare_parameter('rtsp_url', 'rtsp://10.42.0.202:554')
        self.declare_parameter('topic_name', '/camera/image_raw')
        self.declare_parameter('publish_rate_hz', 20.0)

        rtsp_url = self.get_parameter('rtsp_url').get_parameter_value().string_value
        topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        publish_rate = self.get_parameter('publish_rate_hz').get_parameter_value().double_value

        self.get_logger().info(f"Connecting to RTSP stream at: {rtsp_url}")
        
        self.stream = VideoStream(rtsp_url)
        self.stream.start()

        # Publisher now uses the standard Image message type
        self.publisher_ = self.create_publisher(Image, topic_name, 10)
        
        self.timer = self.create_timer(1.0 / publish_rate, self.timer_callback)
        self.bridge = CvBridge()
        
        self.get_logger().info(f"Publishing raw images to '{topic_name}' at {publish_rate} Hz")

    def timer_callback(self):
        frame = self.stream.get_frame()
        
        if frame is not None:
            try:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                # Simply convert the OpenCV frame to a ROS Image and publish
                ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                ros_image.header.stamp = self.get_clock().now().to_msg()
                ros_image.header.frame_id = "camera_frame"
                
                self.publisher_.publish(ros_image)
            except Exception as e:
                self.get_logger().error(f"Failed to publish frame: {e}")
        else:
            self.get_logger().warn("No frame received from stream.")

    def on_shutdown(self):
        self.get_logger().info("Shutting down node and stopping video stream.")
        self.stream.stop()

def main(args=None):
    rclpy.init(args=args)
    camera_publisher = CameraPublisher()
    try:
        rclpy.spin(camera_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        camera_publisher.on_shutdown()
        camera_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()