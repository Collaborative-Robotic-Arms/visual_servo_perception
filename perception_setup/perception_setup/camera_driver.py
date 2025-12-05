#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.time import Time
import cv2
from sensor_msgs.msg import Image, CameraInfo  
from cv_bridge import CvBridge
import threading
import time
import queue
from camera_info_manager import CameraInfoManager

class VideoStream:
    def __init__(self, rtsp_url):
        self.RTSP_URL = rtsp_url
        self.RECONNECT_DELAY = 3
        self.MAX_RETRIES = 5
        self.FRAME_TIMEOUT = 2.0
        self.TARGET_FPS = 30 
        self.cap = None
        # Queue stores tuple: (frame, timestamp_nanoseconds)
        # maxsize=1 ensures we drop old frames immediately (Low Latency)
        self.frame_queue = queue.Queue(maxsize=1) 
        self.running = False
        self.thread = None
        self.connect()

    def connect(self):
        for attempt in range(self.MAX_RETRIES):
            try:
                if self.cap is not None:
                    self.cap.release()
                print(f"Connecting to RTSP stream: {self.RTSP_URL} (attempt {attempt + 1})...")
                
                # Force TCP to prevent artifacting on WiFi
                self.cap = cv2.VideoCapture(f"{self.RTSP_URL}?tcp", cv2.CAP_FFMPEG)
                
                if not self.cap.isOpened():
                    raise RuntimeError("Failed to open stream")
                
                self.cap.set(cv2.CAP_PROP_FPS, self.TARGET_FPS)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Critical for Visual Servoing
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
                print("Stream closed, reconnecting...")
                if not self.connect():
                    time.sleep(self.RECONNECT_DELAY)
                    continue

            # Capture frame
            ret, frame = self.cap.read()
            
            # --- CRITICAL FIX: Capture Time Here ---
            # We capture the time immediately after the frame arrives.
            # If we wait for the ROS timer callback, the time will be wrong (delayed).
            capture_time_ns = int(time.time() * 1e9)
            
            if not ret or frame is None:
                if time.time() - last_frame_time > self.FRAME_TIMEOUT:
                    print("⚠️ Frame timeout, reconnecting...")
                    self.cap.release()
                continue

            last_frame_time = time.time()

            # Buffer Management:
            # If the queue is full, throw away the old frame to make room for the new one.
            # This ensures the robot always processes the *newest* possible data.
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            
            self.frame_queue.put((frame, capture_time_ns))
            
            # Tiny sleep to prevent 100% CPU usage in this thread
            time.sleep(0.001)

    def get_frame(self, timeout=0.5):
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None, None


class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        
        # --- RESTORED YOUR ORIGINAL IP ---
        self.declare_parameter('rtsp_url', 'rtsp://10.42.0.202:554')
        self.declare_parameter('topic_name', '/camera/image_raw')
        self.declare_parameter('publish_rate_hz', 20.0) # Matched your original 20Hz
        self.declare_parameter('frame_id', 'camera_optical_frame')
        
        # Calibration Parameters
        self.declare_parameter('camera_info_url', 'file:///home/omar-magdy/visp_ws/esp32_calibration0.yaml')
        self.declare_parameter('camera_name', 'esp32_cam')

        # Get Parameters
        rtsp_url = self.get_parameter('rtsp_url').get_parameter_value().string_value
        topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        self.publish_rate = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        self.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
        
        camera_info_url = self.get_parameter('camera_info_url').get_parameter_value().string_value
        camera_name = self.get_parameter('camera_name').get_parameter_value().string_value

        self.get_logger().info(f"Starting Camera Driver for: {rtsp_url}")
        
        # Start Stream
        self.stream = VideoStream(rtsp_url)
        self.stream.start()

        # Publishers
        self.publisher_ = self.create_publisher(Image, topic_name, 10)
        info_topic_name = topic_name.replace('image_raw', 'camera_info')
        self.info_publisher_ = self.create_publisher(CameraInfo, info_topic_name, 10)

        # Camera Info Manager
        self.get_logger().info(f"Loading calibration: {camera_info_url}")
        self.cam_info_manager = CameraInfoManager(self, camera_name, camera_info_url)
        if not self.cam_info_manager.loadCameraInfo():
             self.get_logger().warn("Could not load calibration file! Publishing uncalibrated data.")
        
        # Timer
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)
        self.bridge = CvBridge()
        
    def timer_callback(self):
        # Retrieve frame AND the timestamp from when it was captured
        frame, timestamp_ns = self.stream.get_frame()
        
        if frame is not None:
            try:
                # Rotate if needed (ESP32-CAM is often upside down)
                frame = cv2.rotate(frame, cv2.ROTATE_180)
                
                # Create ROS Time object from the capture timestamp
                # This ensures the message time matches Reality, not "Publish Time"
                ros_time = Time(nanoseconds=timestamp_ns).to_msg()
                
                # 1. Image Message
                ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                ros_image.header.stamp = ros_time
                ros_image.header.frame_id = self.frame_id
                
                self.publisher_.publish(ros_image)
                
                # 2. Camera Info Message
                cam_info_msg = self.cam_info_manager.getCameraInfo()
                cam_info_msg.header.stamp = ros_time # MUST match image timestamp
                cam_info_msg.header.frame_id = self.frame_id
                
                self.info_publisher_.publish(cam_info_msg)
                
            except Exception as e:
                self.get_logger().error(f"Publishing error: {e}")

    def on_shutdown(self):
        self.get_logger().info("Stopping camera stream...")
        self.stream.stop()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()