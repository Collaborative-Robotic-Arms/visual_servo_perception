import cv2
import time
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
# Match the URL from your camera_driver.py
STREAM_URL = 'rtsp://10.42.0.202:554/mjpeg/1' 
# ==========================================

def main():
    print(f"Connecting to: {STREAM_URL} using FFMPEG/TCP...")
    
    # Use the exact same backend and flags as your ROS driver
    # ?tcp forces TCP transport (more stable than UDP for high-res images)
    cap = cv2.VideoCapture(f"{STREAM_URL}?tcp", cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("Error: Could not connect to stream. Check IP and WiFi.")
        return

    # Match the driver's low-latency buffer setting
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Latency/FPS Calculation Variables
    prev_time = time.time()
    frame_times = [] # Store last 30 frame durations for averaging
    avg_fps = 0
    
    print("Stream started. Press 'q' to quit.")

    try:
        while True:
            # 1. Capture Frame
            # The time it takes to get from 'read()' call to return is roughly 
            # the time passing between frame arrivals (inter-frame delay).
            ret, frame = cap.read()
            
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            
            if not ret:
                print("Error: Failed to grab frame.")
                break

            # 2. Calculate Stats
            fps = 1.0 / dt if dt > 0 else 0
            
            # Smoothing: Keep rolling average of last 30 frames
            frame_times.append(dt)
            if len(frame_times) > 30:
                frame_times.pop(0)
            
            avg_dt = sum(frame_times) / len(frame_times)
            avg_fps = 1.0 / avg_dt if avg_dt > 0 else 0
            
            # Jitter: Standard deviation of frame times (ms)
            jitter = np.std(frame_times) * 1000 

            # 3. Visualization Overlay
            height, width, _ = frame.shape
            
            # Rotate to match your driver (ESP32 is usually inverted)
            frame = cv2.rotate(frame, cv2.ROTATE_180)

            # Draw FPS Stats
            text_color = (0, 255, 0) # Green
            if avg_fps < 10: text_color = (0, 0, 255) # Red if too slow
            
            cv2.putText(frame, f"Res: {width}x{height}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            cv2.putText(frame, f"Jitter: {jitter:.1f} ms", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Latency Test Helper: Draw current PC time
            # Point your camera at this screen to see the time difference!
            current_ms = int(time.time() * 1000) % 10000
            cv2.putText(frame, f"PC Time: {current_ms}", (10, height - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('ESP32-CAM RTSP Speed Test', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\nTest finished. Final Average FPS: {avg_fps:.2f}")

if __name__ == '__main__':
    main()