# CNN Integration Notes

## Overview
This document summarizes the integration of a CNN-based gaze estimation model into the ReadGaze application and the fixes implemented to make it work reliably.

## Key Implementation Components

### 1. Face Detection and Landmark Extraction
- Used MediaPipe Face Mesh for robust facial landmark detection
- Configured the face mesh detector with higher minimum detection confidence (0.7)
- Used specific landmarks for eye corners, mouth corners, and nose tip

### 2. Head Pose Estimation
- Implemented a fallback mechanism for head pose estimation:
  - First attempt: EPNP method (faster, more accurate when it works)
  - Fallback: ITERATIVE method (more robust but potentially less accurate)
- Added tolerance parameters to the ITERATIVE solver (reprojectionError=12.0)
- Increased iterations (150) for better convergence

### 3. Face Model
- Ensured the 3D face model coordinates match the detected 2D landmarks
- Used a simplified 7-point face model (eyes, mouth, nose) for stability
- Properly aligned coordinate systems between the face model and camera

### 4. Error Handling
- Added robust error handling throughout the pipeline
- Implemented logging at different levels for debugging
- Created visualization tools for verifying landmark detection

### 5. CNN Model Integration
- Loaded pre-trained CNN model (p00.ckpt)
- Used camera calibration matrix for proper coordinate transformation
- Implemented appropriate preprocessing for the CNN input

## Testing
- Created a standalone test script (test_cnn_gaze.py) for testing the CNN estimator
- Added visualization of detected landmarks and gaze vectors
- Added metrics for movement distance and coordinates

## User Experience
- Added user guidance (look directly at camera with head facing forward)
- Disabled calibration requirement when using CNN mode
- Added command-line arguments for model path and debug mode

## Known Limitations
- Performance depends on lighting conditions and camera quality
- Head pose estimation can still occasionally fail in extreme poses
- Some latency compared to the traditional method 