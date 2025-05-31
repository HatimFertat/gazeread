import sys
import cv2
import time
import numpy as np
import warnings
import logging
import os
import re
from typing import Optional, Tuple, Literal, List, Dict, Any
warnings.filterwarnings("ignore", category=RuntimeWarning, module="eyetrax.filters.kde")

from eyetrax.gaze import GazeEstimator, CNNGazeEstimator
from eyetrax.calibration import run_9_point_calibration, run_9_point_calibration_cnn
from eyetrax.filters import KalmanSmoother, KDESmoother, NoSmoother, make_kalman
from eyetrax.utils.screen import get_screen_size

FilterType = Literal["none", "kalman", "kde"]

class EyeTrackerCNN:
    """CNN-based eye tracker that directly uses CNNGazeEstimator like in the test script."""
    
    def __init__(self, model_path=None, calibration_path=None, wrapper_model_path=None, wrapper_model_name="ridge"):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing CNN-based eye tracker")
        
        # Initialize CNN estimator with direct camera access
        try:
            self.estimator = CNNGazeEstimator(
                model_path, 
                calibration_path, 
                use_direct_camera=True,
                wrapper_model_name=wrapper_model_name
            )
            self.calibrated = False  # Start uncalibrated by default
            self.connected = True
            self.filter_type: FilterType = "none"
            self.screen_width, self.screen_height = get_screen_size()
            self.logger.info("CNN estimator initialized successfully")

            # Try to load wrapper model if path provided
            if wrapper_model_path and os.path.exists(wrapper_model_path):
                if self.estimator.load_wrapper_model(wrapper_model_path):
                    self.calibrated = True
                    self.logger.info(f"Loaded wrapper model from {wrapper_model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CNN estimator: {e}")
            self.estimator = None
            self.calibrated = False
            self.connected = False
    
    def connect(self) -> bool:
        """Connection is handled directly by CNNGazeEstimator."""
        if self.estimator is not None:
            self.connected = True
            return True
        return False
    
    def is_connected(self) -> bool:
        """Check if the estimator has a working camera."""
        return self.connected and self.estimator is not None
    
    def set_filter(self, filter_type: FilterType):
        """Set the filter type for gaze smoothing."""
        self.filter_type = filter_type
        
        if filter_type == "kalman":
            kalman = make_kalman()
            self.smoother = KalmanSmoother(kalman)
            if self.calibrated:
                # Tune the Kalman filter with the current estimator
                self.smoother.tune(self.estimator, camera_index=0)
        elif filter_type == "kde":
            self.smoother = KDESmoother(self.screen_width, self.screen_height, confidence=0.8)
        else:
            self.smoother = NoSmoother()
            
        return True
    
    def calibrate(self) -> bool:
        """Run CNN calibration with wrapper model."""
        if not self.is_connected():
            self.logger.error("Cannot calibrate: eye tracker not connected")
            return False
            
        try:
            self.logger.info("Starting CNN calibration to train wrapper model...")
            
            # Verify that estimator has necessary attributes
            if not hasattr(self.estimator, 'wrapper_model'):
                self.logger.error("Estimator has no 'wrapper_model' attribute")
                return False
                
            success = run_9_point_calibration_cnn(self.estimator)
            if success:
                self.calibrated = True
                self.logger.info("CNN calibration completed successfully")
                return True
            else:
                self.logger.error("CNN calibration failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Error during CNN calibration: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def get_gaze_point(self) -> Optional[Tuple[Tuple[float, float], bool]]:
        """Get the current gaze point coordinates and blink status."""
        if not self.is_connected():
            self.logger.debug("Eye tracker not connected")
            return None
        
        try:
            # Get frame directly from the estimator
            frame = self.estimator.get_frame()
            if frame is None:
                self.logger.debug("Failed to get frame from camera")
                return None
            
            # Extract features and check for blinks
            features, is_blinking = self.estimator.extract_features(frame)
            
            if features is None:
                return None
            
            # Predict screen coordinates using CNN with wrapper model adjustment
            gaze_point = self.estimator.predict([features])[0]
            self.logger.debug(f"Get gaze point: {gaze_point}")
            
            # Use raw values without additional smoothing
            x, y = map(int, gaze_point)
            
            return ((x, y), is_blinking)
            
        except Exception as e:
            self.logger.error(f"Error getting gaze data: {e}")
            return None
    
    def save_model(self, path: str) -> bool:
        """Save wrapper model if calibrated."""
        if not self.calibrated:
            self.logger.error("Cannot save model: not calibrated")
            return False
            
        try:
            # Save the wrapper model
            if hasattr(self.estimator, "save_wrapper_model"):
                success = self.estimator.save_wrapper_model(path)
                if success:
                    self.logger.info(f"Saved wrapper model to {path}")
                    return True
                else:
                    self.logger.warning("Failed to save wrapper model")
                    return False
            else:
                self.logger.warning("save_wrapper_model method not found in estimator")
                return False
        except Exception as e:
            self.logger.error(f"Error saving wrapper model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load wrapper model."""
        try:
            if hasattr(self.estimator, "load_wrapper_model"):
                success = self.estimator.load_wrapper_model(path)
                if success:
                    self.calibrated = True
                    self.logger.info(f"Loaded wrapper model from {path}")
                    
                    # If using Kalman filter, tune it after loading
                    if self.filter_type == "kalman":
                        self.smoother.tune(self.estimator, camera_index=0)

                    return True
                else:
                    self.logger.warning("Failed to load wrapper model")
                    return False
            else:
                self.logger.warning("load_wrapper_model method not found in estimator")
                return False
        except Exception as e:
            self.logger.error(f"Error loading wrapper model: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from the camera."""
        if self.estimator is not None:
            try:
                self.estimator.release_camera()
                self.connected = False
                self.logger.info("Disconnected from camera")
            except Exception as e:
                self.logger.error(f"Error disconnecting from camera: {e}")

class EyeTracker:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.estimator = GazeEstimator(model_name="ridge", use_robust_features=True)
        self.camera = None
        self.connected = False
        self.calibrated = False
        self.filter_type: FilterType = "none"
        self.smoother = NoSmoother()
        self.screen_width, self.screen_height = get_screen_size()
        self.connect()
        
    def connect(self) -> bool:
        """Attempt to connect to the webcam and initialize eye tracking."""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                self.logger.error("Failed to open webcam")
                return False
                
            self.connected = True
            self.logger.info("Connected to webcam")
            return True
        except Exception as e:
            self.logger.error(f"Error connecting to webcam: {e}")
            return False
            
    def is_connected(self) -> bool:
        """Check if the camera is connected."""
        return self.connected and self.camera is not None
        
    def set_filter(self, filter_type: FilterType):
        """Set the filter type for gaze smoothing."""
        self.filter_type = filter_type
        
        if filter_type == "kalman":
            kalman = make_kalman()
            self.smoother = KalmanSmoother(kalman)
            if self.calibrated:
                # Tune the Kalman filter with the current estimator
                self.smoother.tune(self.estimator, camera_index=0)
        elif filter_type == "kde":
            self.smoother = KDESmoother(self.screen_width, self.screen_height, confidence=0.5)
        else:
            self.smoother = NoSmoother()
            
        return True
        
    def calibrate(self) -> bool:
        """Perform eye tracker calibration."""
        if not self.is_connected():
            self.logger.error("Cannot calibrate: eye tracker not connected")
            return False
            
        try:
            self.logger.info("Starting calibration...")
            run_9_point_calibration(self.estimator)
            self.calibrated = True
            self.logger.info("Calibration completed successfully")
            
            # If using Kalman filter, tune it after calibration
            if self.filter_type == "kalman":
                self.smoother.tune(self.estimator, camera_index=0)
                
            return True
        except Exception as e:
            self.logger.error(f"Error during calibration: {e}")
            return False
            
    def get_gaze_point(self) -> Optional[Tuple[Tuple[float, float], bool]]:
        """Get the current gaze point coordinates and blink status."""
        if not self.is_connected():
            self.logger.debug("Eye tracker not connected")
            return None
            
        if not self.calibrated:
            self.logger.debug("Eye tracker not calibrated")
            return None
            
        try:
            ret, frame = self.camera.read()
            if not ret:
                self.logger.debug("Failed to read frame from camera")
                return None
                
            # Extract features and check for blinks
            features, is_blinking = self.estimator.extract_features(frame)
            
            if features is None:
                return None
                
            # Predict screen coordinates
            gaze_point = self.estimator.predict(np.array([features]))[0]
            x, y = map(int, gaze_point)
            
            # Apply smoothing
            x_smooth, y_smooth = self.smoother.step(x, y)
            # self.logger.debug(f"Raw gaze: ({x}, {y}), Smoothed: ({x_smooth}, {y_smooth}), Blinking: {is_blinking}")
            return ((x_smooth, y_smooth), is_blinking)
            
        except Exception as e:
            self.logger.error(f"Error getting gaze data: {e}")
            return None
            
    def save_model(self, path: str) -> bool:
        """Save the trained gaze estimator model."""
        if not self.calibrated:
            self.logger.error("Cannot save model: not calibrated")
            return False
            
        try:
            self.estimator.save_model(path)
            self.logger.info(f"Saved calibration model to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False
            
    def load_model(self, path: str) -> bool:
        """Load a trained gaze estimator model."""
        try:
            self.estimator.load_model(path)
            self.calibrated = True
            self.logger.info(f"Loaded calibration model from {path}")
            
            # If using Kalman filter, tune it after loading
            if self.filter_type == "kalman":
                self.smoother.tune(self.estimator, camera_index=0)
                
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
            
    def disconnect(self):
        """Disconnect from the camera."""
        if self.camera:
            try:
                self.camera.release()
                self.connected = False
                self.camera = None
                self.logger.info("Disconnected from camera")
            except Exception as e:
                self.logger.error(f"Error disconnecting from camera: {e}") 
                print(f"Error disconnecting from eye tracker: {e}") 