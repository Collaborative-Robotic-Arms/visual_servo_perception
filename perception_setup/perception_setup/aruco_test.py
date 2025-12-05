import cv2
import time
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
STREAM_URL = 'rtsp://10.42.0.202:554/mjpeg/1'
# ==========================================

def main():
    print(f"Connecting to: {STREAM_URL} ...")
    cap = cv2.VideoCapture(f"{STREAM_URL}?tcp", cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: Could not connect.")
        return

    # ArUco Setup
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

    print("Stream started. Press 'q' to quit.")

    try:
        while True:
            # 1. Receive Frame (Network Latency happens here)
            ret, frame = cap.read()
            if not ret: break
            
            # Rotate if needed
            frame = cv2.rotate(frame, cv2.ROTATE_180)

            # 2. START TIMER (Processing Only)
            start_proc = time.perf_counter()

            # --- THE HEAVY LIFTING ---
            corners, ids, rejected = detector.detectMarkers(frame)
            
            # Draw markers (optional, adds a tiny bit of time)
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            # -------------------------

            # 3. STOP TIMER
            end_proc = time.perf_counter()
            
            # Calculate Processing Time
            proc_time_ms = (end_proc - start_proc) * 1000
            
            # Display Stats
            text = f"Processing Time: {proc_time_ms:.2f} ms"
            color = (0, 255, 0) if proc_time_ms < 30 else (0, 0, 255)
            
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, color, 2)
            
            # Show resolution for context
            h, w = frame.shape[:2]
            cv2.putText(frame, f"Res: {w}x{h}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (255, 255, 0), 2)

            cv2.imshow('ArUco Speed Test', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()