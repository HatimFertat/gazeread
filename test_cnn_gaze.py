#!/usr/bin/env python
"""
Simple test script to verify the CNN gaze estimator works without the GUI.
"""
import sys
import os
import time
import logging
import cv2
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Set OpenCV debug level
os.environ['OPENCV_LOG_LEVEL'] = 'DEBUG'

# Import the CNN gaze estimator
from readgaze.eye_tracking.tracker import CNNGazeEstimator

def main():
    """Run a simple test of the CNN gaze estimator."""
    logger = logging.getLogger(__name__)
    logger.info("Starting CNN gaze estimator test")
    
    # Initialize the CNN gaze estimator
    cnn_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnn")
    model_path = os.path.join(cnn_dir, "p00.ckpt")
    calibration_path = os.path.join(cnn_dir, "calibration_matrix.yaml")
    
    logger.info(f"Model path: {model_path}")
    logger.info(f"Calibration path: {calibration_path}")
    
    # Create the estimator
    estimator = CNNGazeEstimator(model_path, calibration_path)
    
    # Initialize webcam
    logger.info("Opening webcam")
    cap = cv2.VideoCapture(0)

    # Try to improve webcam settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Check webcam properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Webcam properties: {width}x{height} @ {fps}fps")
    
    if not cap.isOpened():
        logger.error("Failed to open webcam")
        return
    
    try:
        # Process frames
        frame_count = 0
        last_point = None
        start_time = time.time()
        fps_counter = 0
        fps_display = 0
        
        # Create a separate window for the face mesh visualization
        cv2.namedWindow("Face Mesh", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face Mesh", 640, 480)
        
        # Import MediaPipe for visualization
        import mediapipe as mp
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh
        
        # Set drawing specifications
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            # Process frame
            frame_count += 1
            fps_counter += 1
            
            # Calculate FPS every second
            current_time = time.time()
            if current_time - start_time >= 1.0:
                fps_display = fps_counter
                fps_counter = 0
                start_time = current_time
            
            # Get face landmarks and gaze point
            process_start = time.time()
            features, is_blinking = estimator.extract_features(frame)
            process_time = time.time() - process_start
            
            # Create a debug view
            debug_frame = frame.copy()
            cv2.putText(debug_frame, f"FPS: {fps_display}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(debug_frame, f"Process time: {process_time*1000:.1f}ms", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Create face mesh visualization
            face_mesh_frame = frame.copy()
            image_rgb = cv2.cvtColor(face_mesh_frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = estimator.face_mesh.process(image_rgb)
            image_rgb.flags.writeable = True
            
            if results.multi_face_landmarks:
                # Draw the face mesh annotations on the image
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=face_mesh_frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    mp_drawing.draw_landmarks(
                        image=face_mesh_frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    
                    # Highlight the specific landmarks we use
                    height, width = face_mesh_frame.shape[:2]
                    landmarks_used = [33, 133, 362, 263, 61, 291, 1]  # Same as in tracker.py
                    for idx in landmarks_used:
                        try:
                            landmark = face_landmarks.landmark[idx]
                            pos = (int(landmark.x * width), int(landmark.y * height))
                            cv2.circle(face_mesh_frame, pos, 5, (0, 255, 0), -1)
                            cv2.putText(face_mesh_frame, str(idx), pos, 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        except IndexError:
                            logger.error(f"Landmark {idx} not found")
                
                # Show the face mesh window
                cv2.imshow("Face Mesh", face_mesh_frame)
            else:
                cv2.putText(face_mesh_frame, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Face Mesh", face_mesh_frame)
            
            if features is None:
                logger.debug("No features detected")
                cv2.putText(debug_frame, "No face detected", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("CNN Gaze Test", debug_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Get gaze point
            predict_start = time.time()
            gaze_point = estimator.predict([features])[0]
            predict_time = time.time() - predict_start
            
            x, y = map(int, gaze_point)
            
            # Display movement info
            if last_point is not None:
                dx = x - last_point[0]
                dy = y - last_point[1]
                distance = np.sqrt(dx*dx + dy*dy)
                if distance > 5:  # Only log significant movements
                    logger.info(f"Gaze moved: ({last_point[0]}, {last_point[1]}) -> "
                               f"({x}, {y}), distance: {distance:.1f}px")
                
                # Draw movement arrow
                screen_w, screen_h = estimator.monitor_pixels
                frame_h, frame_w = debug_frame.shape[:2]
                
                # Scale from screen coordinates to frame coordinates
                scaled_x1 = int(last_point[0] * frame_w / screen_w)
                scaled_y1 = int(last_point[1] * frame_h / screen_h)
                scaled_x2 = int(x * frame_w / screen_w)
                scaled_y2 = int(y * frame_h / screen_h)
                
                # Draw arrow showing movement
                if distance > 10:  # Only show significant movements
                    cv2.arrowedLine(debug_frame, (scaled_x1, scaled_y1), (scaled_x2, scaled_y2), 
                                   (255, 0, 0), 2)
            
            last_point = (x, y)
            
            # Display the prediction time
            cv2.putText(debug_frame, f"Predict time: {predict_time*1000:.1f}ms", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the gaze coordinates
            cv2.putText(debug_frame, f"Gaze: ({x}, {y})", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw circle at gaze position (scaled to frame size)
            frame_h, frame_w = debug_frame.shape[:2]
            screen_w, screen_h = estimator.monitor_pixels
            scaled_x = int(x * frame_w / screen_w)
            scaled_y = int(y * frame_h / screen_h)
            
            cv2.circle(debug_frame, (scaled_x, scaled_y), 10, (0, 0, 255), -1)
            
            # Show the debug frame
            cv2.imshow("CNN Gaze Test", debug_frame)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        logger.exception(f"Error in gaze estimation: {e}")
    finally:
        # Release webcam and close windows
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 