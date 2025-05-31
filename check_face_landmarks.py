#!/usr/bin/env python
"""
Simple script to visualize MediaPipe face mesh landmarks and verify the IDs.
"""
import cv2
import mediapipe as mp
import numpy as np
import sys
import os
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Run a simple test to visualize face mesh landmarks."""
    logger.info("Starting face landmark visualization")
    
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # Improved face mesh settings
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Define the landmarks we want to highlight
    important_landmarks = {
        33: "Right eye outer",
        133: "Left eye outer",
        362: "Right eye inner",
        263: "Left eye inner",
        61: "Right mouth corner",
        291: "Left mouth corner",
        1: "Nose tip"
    }
    
    # Initialize webcam
    logger.info("Opening webcam")
    cap = cv2.VideoCapture(0)
    
    # Try to set better resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    if not cap.isOpened():
        logger.error("Failed to open webcam")
        return
    
    # Create windows
    cv2.namedWindow("Face Mesh", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Important Landmarks", cv2.WINDOW_NORMAL)
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to read frame")
                break
            
            # Convert to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            
            # Process with face mesh
            results = face_mesh.process(image_rgb)
            
            # Draw the face mesh
            image_rgb.flags.writeable = True
            mesh_frame = frame.copy()
            landmark_frame = frame.copy()
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw the full mesh
                    mp_drawing.draw_landmarks(
                        image=mesh_frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    
                    # Draw face contours
                    mp_drawing.draw_landmarks(
                        image=mesh_frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                    
                    # Draw important landmarks
                    height, width = frame.shape[:2]
                    for idx, name in important_landmarks.items():
                        try:
                            landmark = face_landmarks.landmark[idx]
                            x, y = int(landmark.x * width), int(landmark.y * height)
                            
                            # Draw on the landmark frame
                            cv2.circle(landmark_frame, (x, y), 5, (0, 255, 0), -1)
                            cv2.putText(landmark_frame, f"{idx}: {name}", (x + 10, y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            
                            # Log the landmark position
                            if cv2.waitKey(1) & 0xFF == ord('p'):  # Press 'p' to print landmarks
                                logger.info(f"Landmark {idx} ({name}): ({x}, {y})")
                                
                        except IndexError:
                            logger.error(f"Landmark {idx} not found")
            else:
                cv2.putText(mesh_frame, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(landmark_frame, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Show the frames
            cv2.imshow("Face Mesh", mesh_frame)
            cv2.imshow("Important Landmarks", landmark_frame)
            
            # Add instructions
            help_text = "Press 'p' to print landmark coordinates, 'q' to quit"
            cv2.putText(landmark_frame, help_text, (10, landmark_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        logger.exception(f"Error in face mesh detection: {e}")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 