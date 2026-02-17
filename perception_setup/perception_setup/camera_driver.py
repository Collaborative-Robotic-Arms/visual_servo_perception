#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import cv2
import os
import yaml
import threading
import time
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

class CameraBufferCleaner(threading.Thread):
    def __init__(self, url, logger):
        super().__init__()
        self.url = url
        self.logger = logger
        self.lock = threading.Lock()
        self.running = True
        self.latest_frame = None
        
        # --- PERFORMANCE TRACKING ---
        self.frame_count = 0
        self.start_time = time.time()
        self.last_frame_time = time.time()
        self.total_lag_duration = 0.0
        self.total_smooth_duration = 0.0
        self.LAG_THRESHOLD = 0.1  # 100ms threshold

    def run(self):
        # OPTIMIZED FOR HOTSPOT (TCP)
        # rtsp_transport;tcp: Guarantees frame delivery (no grey screens on noisy WiFi).
        # fflags;nobuffer: Tells FFMPEG "don't wait, just decode".
        # reorder_queue_size;100: Small buffer for packet jitter.
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|"
            "strict;experimental|reorder_queue_size;100"
        )
        
        cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            self.logger.error(f"❌ Connection Failed: {self.url}")
            return

        self.logger.info("⚡ Camera Thread Started (TCP Mode + Analytics).")
        self.start_time = time.time()
        self.last_frame_time = time.time()
        
        while self.running:
            ret, frame = cap.read()
            now = time.time()
            delta = now - self.last_frame_time
            
            # --- STATS LOGIC ---
            if delta > self.LAG_THRESHOLD:
                self.total_lag_duration += delta
            else:
                self.total_smooth_duration += delta
            
            self.last_frame_time = now
            self.frame_count += 1

            if ret:
                with self.lock:
                    self.latest_frame = frame
            else:
                # If TCP drops, it usually means a full disconnect/reconnect cycle
                pass 
                
        cap.release()

    def get_frame(self):
        with self.lock:
            return self.latest_frame

    def get_report(self):
        total_time = time.time() - self.start_time
        fps = self.frame_count / total_time if total_time > 0 else 0
        health = 100 - (self.total_lag_duration / total_time * 100) if total_time > 0 else 0
        return total_time, fps, self.total_lag_duration, health

    def stop(self):
        self.running = False


class VisualServoingDriver(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        
        # PARAMETERS
        self.declare_parameter('rtsp_url', 'rtsp://10.42.0.202:554')
        self.declare_parameter('camera_info_url', 'file:///home/omar-magdy/gp_ws/fisheye.yaml')
        self.declare_parameter('frame_id', 'cameraAR4_optical_frame')
        
        self.rtsp_url = self.get_parameter('rtsp_url').value
        self.info_url = self.get_parameter('camera_info_url').value
        self.frame_id = self.get_parameter('frame_id').value

        # QoS: BEST EFFORT (Mandatory for real-time servoing)
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.publisher = self.create_publisher(Image, '/cameraAR4/image_raw', qos_profile)
        self.info_publisher = self.create_publisher(CameraInfo, '/cameraAR4/camera_info', qos_profile)
        self.bridge = CvBridge()
        self.camera_info_msg = self.load_calibration(self.info_url)

        # START THREAD
        self.cam_thread = CameraBufferCleaner(self.rtsp_url, self.get_logger())
        self.cam_thread.start()

        # TIMER (60Hz Target)
        self.create_timer(0.016, self.publish_loop)
        
        # WATCHDOG (10 Minutes) - Generates report automatically
        self.start_time = time.time()
        self.MAX_RUN_TIME = 1200.0
        self.create_timer(1.0, self.watchdog)

        self.get_logger().info("🚀 Driver Online: TCP / Best Effort / Analytics Active.")

    def publish_loop(self):
        frame = self.cam_thread.get_frame()
        if frame is None: return

        try:
            now = self.get_clock().now().to_msg()
            
            msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            msg.header.stamp = now
            msg.header.frame_id = self.frame_id
            
            self.camera_info_msg.header.stamp = now
            self.camera_info_msg.header.frame_id = self.frame_id
            
            self.publisher.publish(msg)
            self.info_publisher.publish(self.camera_info_msg)
        except Exception: pass

    def watchdog(self):
        if time.time() - self.start_time > self.MAX_RUN_TIME:
            self.generate_final_report()
            rclpy.shutdown()

    def generate_final_report(self):
        t, fps, lag, health = self.cam_thread.get_report()
        print("\n" + "="*45)
        print("📊 STREAM DIAGNOSTICS (TCP MODE)")
        print(f"Total Time:      {t:.2f}s")
        print(f"Avg FPS:         {fps:.2f}")
        print(f"Lag Duration:    {lag:.2f}s (Time > 100ms)")
        print(f"Health Score:    {health:.1f}%")
        print("="*45 + "\n")

    def load_calibration(self, url):
        msg = CameraInfo()
        path = url.replace("file://", "")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    cal = yaml.safe_load(f)
                    msg.width = cal['image_width']
                    msg.height = cal['image_height']
                    msg.distortion_model = cal['distortion_model']
                    msg.d = list(cal['distortion_coefficients']['data'])
                    msg.k = list(cal['camera_matrix']['data'])
                    msg.r = list(cal['rectification_matrix']['data'])
                    msg.p = list(cal['projection_matrix']['data'])
            except: pass
        return msg

    def destroy_node(self):
        self.cam_thread.stop()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = VisualServoingDriver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        # Report is generated on Ctrl+C as well
        node.generate_final_report()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__': main()