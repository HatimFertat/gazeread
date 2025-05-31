from __future__ import annotations

from collections import deque
from pathlib import Path

import logging
from typing import Tuple, List
import sys
import os
import re

import cv2
import mediapipe as mp
import numpy as np

from eyetrax.constants import LEFT_EYE_INDICES, MUTUAL_INDICES, RIGHT_EYE_INDICES
from eyetrax.models import BaseModel, create_model


# Add CNN directory to path
cnn_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "cnn")
sys.path.append(cnn_dir)

# Import CNN modules directly with explicit imports
import cnn.utils as cnn_utils
import cnn.mpii_face_gaze_preprocessing as cnn_preprocessing
from cnn.model import Model
from .face_model import create_face_model

# Set up logger
logger = logging.getLogger(__name__)

class GazeEstimator:
    def __init__(
        self,
        model_name: str = "ridge",
        model_kwargs: dict | None = None,
        ear_history_len: int = 50,
        blink_threshold_ratio: float = 0.8,
        min_history: int = 15,
        use_robust_features: bool = False,
    ):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        )
        self.model: BaseModel = create_model(model_name, **(model_kwargs or {}))

        self._ear_history = deque(maxlen=ear_history_len)
        self._blink_ratio = blink_threshold_ratio
        self._min_history = min_history
        self.use_robust_features = use_robust_features
        
        # Initialize camera matrix and distortion coefficients with defaults
        # These will be overridden if a calibration file is available
        self.camera_matrix = np.array([
            [1000, 0, 640],
            [0, 1000, 360],
            [0, 0, 1]
        ])
        self.dist_coefficients = np.zeros((1, 5))
        
        # Try to load camera calibration if available
        self._try_load_camera_calibration()
        
        # Initialize buffers for pose estimation and smoothing
        self.rvec_buffer = deque(maxlen=3)
        self.tvec_buffer = deque(maxlen=3)
        self.rvec = None
        self.tvec = None
        
    def _try_load_camera_calibration(self):
        """Attempt to load camera calibration from standard locations"""
        cnn_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "cnn")
        calibration_path = os.path.join(cnn_dir, "calibration_matrix.yaml")
        
        try:
            if os.path.exists(calibration_path):
                from cnn.utils import get_camera_matrix
                self.camera_matrix, self.dist_coefficients = get_camera_matrix(calibration_path)
                logger.info(f"Loaded camera calibration from {calibration_path}")
            else:
                logger.warning("Camera calibration file not found, using defaults")
        except Exception as e:
            logger.warning(f"Failed to load camera calibration: {e}")
    
    def extract_features_robust(self, image):
        """
        Enhanced feature extraction using the complex face model and camera calibration
        
        Args:
            image: Input frame for feature extraction
            
        Returns:
            features: Feature vector for gaze estimation or None if extraction failed
            blink_detected: Boolean indicating if a blink was detected
        """
        # Convert image to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, _ = image.shape
        
        # Process with face mesh
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            return None, False
            
        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark
        
        # Extract all points as numpy array (for PnP later)
        all_points = np.array(
            [(lm.x * width, lm.y * height) for lm in landmarks], dtype=np.float32
        )
        
        # ---- BLINK DETECTION (from simple method) ----
        left_eye_inner = np.array([landmarks[133].x, landmarks[133].y])
        left_eye_outer = np.array([landmarks[33].x, landmarks[33].y])
        left_eye_top = np.array([landmarks[159].x, landmarks[159].y])
        left_eye_bottom = np.array([landmarks[145].x, landmarks[145].y])

        right_eye_inner = np.array([landmarks[362].x, landmarks[362].y])
        right_eye_outer = np.array([landmarks[263].x, landmarks[263].y])
        right_eye_top = np.array([landmarks[386].x, landmarks[386].y])
        right_eye_bottom = np.array([landmarks[374].x, landmarks[374].y])

        left_eye_width = np.linalg.norm(left_eye_outer - left_eye_inner)
        left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
        left_EAR = left_eye_height / (left_eye_width + 1e-9)

        right_eye_width = np.linalg.norm(right_eye_outer - right_eye_inner)
        right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
        right_EAR = right_eye_height / (right_eye_width + 1e-9)

        EAR = (left_EAR + right_EAR) / 2

        self._ear_history.append(EAR)
        if len(self._ear_history) >= self._min_history:
            thr = float(np.mean(self._ear_history)) * self._blink_ratio
        else:
            thr = 0.2
        blink_detected = EAR < thr
        
        # ---- ADVANCED POSE ESTIMATION using PnP ----
        # Use landmarks_ids from face_model to extract specific landmarks for PnP
        try:
            face_landmarks_subset = np.asarray([all_points[i] for i in landmarks_ids])
            
            # Solve PnP to get head pose
            success, rvec, tvec = cv2.solvePnP(
                face_model, face_landmarks_subset,
                self.camera_matrix, self.dist_coefficients,
                flags=cv2.SOLVEPNP_EPNP,
                useExtrinsicGuess=bool(self.rvec is not None)
            )
            
            if not success:
                # Try iterative method as fallback
                success, rvec, tvec = cv2.solvePnP(
                    face_model, face_landmarks_subset,
                    self.camera_matrix, self.dist_coefficients,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
                if not success:
                    # Fall back to simple method if both attempts fail
                    return self.extract_features_simple(image)
                    
            # Refine pose estimation
            for _ in range(3):
                success, rvec, tvec = cv2.solvePnP(
                    face_model, face_landmarks_subset,
                    self.camera_matrix, self.dist_coefficients,
                    rvec=rvec, tvec=tvec,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                
            # Smooth pose parameters
            self.rvec_buffer.append(rvec)
            if len(self.rvec_buffer) > 0:
                rvec = np.mean(self.rvec_buffer, axis=0)
                
            self.tvec_buffer.append(tvec)
            if len(self.tvec_buffer) > 0:
                tvec = np.mean(self.tvec_buffer, axis=0)
                
            self.rvec, self.tvec = rvec, tvec
            
            # Get rotation matrix from rodrigues
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # ---- COMBINE FEATURES FROM BOTH METHODS ----
            # Get key eye landmarks as in simple method
            subset_indices = LEFT_EYE_INDICES + RIGHT_EYE_INDICES + MUTUAL_INDICES
            
            # Transform eye landmarks to a normalized space (as in simple method)
            nose_anchor = all_points[4]  # Nose tip
            
            # Use rotation matrix from PnP for better normalization
            # Convert pixel coordinates to camera coordinates first
            fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
            cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
            
            # Extract and normalize eye landmarks
            eye_landmarks = []
            for idx in subset_indices:
                x, y = all_points[idx]
                # Convert to normalized camera coordinates
                x_norm = (x - cx) / fx
                y_norm = (y - cy) / fy
                z_norm = 1.0  # Assume points are on a plane at z=1
                
                # Apply inverse rotation to normalize orientation
                point_cam = np.array([x_norm, y_norm, z_norm])
                normalized_point = rotation_matrix.T @ point_cam
                
                eye_landmarks.append(normalized_point)
                
            eye_landmarks = np.array(eye_landmarks)
            
            # Extract Euler angles from rotation matrix for additional features
            # Convert rotation matrix to Euler angles
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            pitch = np.arctan2(-rotation_matrix[2, 0], 
                              np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            
            # Flatten eye landmarks and combine with head pose
            flattened_landmarks = eye_landmarks.flatten()
            features = np.concatenate([flattened_landmarks, [yaw, pitch, roll]])
            
            return features, blink_detected
            
        except Exception as e:
            logger.warning(f"Error in robust feature extraction: {e}")
            # Fall back to simple method if anything fails
            return self.extract_features_simple(image)

    def extract_features_simple(self, image):
        """
        Takes in image and returns landmarks around the eye region
        Normalization with nose tip as anchor
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return None, False

        face_landmarks = results.multi_face_landmarks[0]
        landmarks = face_landmarks.landmark

        all_points = np.array(
            [(lm.x, lm.y, lm.z) for lm in landmarks], dtype=np.float32
        )
        nose_anchor = all_points[4]
        left_corner = all_points[33]
        right_corner = all_points[263]
        top_of_head = all_points[10]

        shifted_points = all_points - nose_anchor
        x_axis = right_corner - left_corner
        x_axis /= np.linalg.norm(x_axis) + 1e-9
        y_approx = top_of_head - nose_anchor
        y_approx /= np.linalg.norm(y_approx) + 1e-9
        y_axis = y_approx - np.dot(y_approx, x_axis) * x_axis
        y_axis /= np.linalg.norm(y_axis) + 1e-9
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis) + 1e-9
        R = np.column_stack((x_axis, y_axis, z_axis))
        rotated_points = (R.T @ shifted_points.T).T

        left_corner_rot = R.T @ (left_corner - nose_anchor)
        right_corner_rot = R.T @ (right_corner - nose_anchor)
        inter_eye_dist = np.linalg.norm(right_corner_rot - left_corner_rot)
        if inter_eye_dist > 1e-7:
            rotated_points /= inter_eye_dist

        subset_indices = LEFT_EYE_INDICES + RIGHT_EYE_INDICES + MUTUAL_INDICES
        eye_landmarks = rotated_points[subset_indices]
        features = eye_landmarks.flatten()

        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1] ** 2 + R[2, 2] ** 2))
        roll = np.arctan2(R[2, 1], R[2, 2])
        features = np.concatenate([features, [yaw, pitch, roll]])

        # Blink detection
        left_eye_inner = np.array([landmarks[133].x, landmarks[133].y])
        left_eye_outer = np.array([landmarks[33].x, landmarks[33].y])
        left_eye_top = np.array([landmarks[159].x, landmarks[159].y])
        left_eye_bottom = np.array([landmarks[145].x, landmarks[145].y])

        right_eye_inner = np.array([landmarks[362].x, landmarks[362].y])
        right_eye_outer = np.array([landmarks[263].x, landmarks[263].y])
        right_eye_top = np.array([landmarks[386].x, landmarks[386].y])
        right_eye_bottom = np.array([landmarks[374].x, landmarks[374].y])

        left_eye_width = np.linalg.norm(left_eye_outer - left_eye_inner)
        left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
        left_EAR = left_eye_height / (left_eye_width + 1e-9)

        right_eye_width = np.linalg.norm(right_eye_outer - right_eye_inner)
        right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
        right_EAR = right_eye_height / (right_eye_width + 1e-9)

        EAR = (left_EAR + right_EAR) / 2

        self._ear_history.append(EAR)
        if len(self._ear_history) >= self._min_history:
            thr = float(np.mean(self._ear_history)) * self._blink_ratio
        else:
            thr = 0.2
        blink_detected = EAR < thr

        return features, blink_detected
    
    def extract_features(self, image):
        if self.use_robust_features:
            return self.extract_features_robust(image)
        else:
            return self.extract_features_simple(image)

    def save_model(self, path: str | Path):
        """
        Pickle model
        """
        self.model.save(path)

    def load_model(self, path: str | Path):
        self.model = BaseModel.load(path)

    def train(self, X, y, variable_scaling=None):
        """
        Trains gaze prediction model
        """
        self.model.train(X, y, variable_scaling)

    def predict(self, X):
        """
        Predicts gaze location
        """
        return self.model.predict(X)


# Define the landmarks IDs used by the model
# These correspond to specific points on the face model:
# 33: right eye outer corner
# 133: left eye outer corner
# 362: right eye inner corner (MediaPipe notation)
# 263: left eye inner corner (MediaPipe notation)
# 61: right mouth corner
# 291: left mouth corner
# 1: nose tip
landmarks_ids = [33, 133, 362, 263, 61, 291, 1]  # right eye, left eye, mouth, nose

def load_face_model_from_file() -> Tuple[np.ndarray, np.ndarray, List[int]]:
    """
    Load the face model directly from the main.py file.
    
    Returns:
        Tuple of (face_model_all, face_model, landmarks_ids)
    """

    logger.warning("Using backup face model from face_model.py")
        
    # Use our simplified face model as a fallback
    face_model_all, face_model = create_face_model(landmarks_ids)
    return face_model_all, face_model, landmarks_ids

# Load the face model
face_model_all, face_model, landmarks_ids = load_face_model_from_file()

class CNNGazeEstimator:
    """Wrapper for the CNN gaze estimation model that implements the interface needed by EyeTracker."""
    
    def __init__(self, model_path: str = None, calibration_matrix_path: str = None, use_direct_camera: bool = False, wrapper_model_name: str = "ridge"):
        """Initialize the CNN gaze estimator.
        
        Args:
            model_path: Path to the CNN model checkpoint
            calibration_matrix_path: Path to the camera calibration matrix
            use_direct_camera: Whether to use direct camera access like in the test script
            wrapper_model_name: Name of the base model to use for adjusting CNN output (e.g., "ridge", "svr")
        """
        import torch
        import mediapipe as mp
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        from eyetrax.models import create_model
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing CNN gaze estimator")
        self.logger.info("USAGE HINT: For best results, start by looking directly at the camera with your head facing forward")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        # Setup face mesh detector with improved settings
        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.logger.info("Face mesh detector initialized")
        
        # Setup transform for CNN input
        self.transform = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])
        
        # Load model
        if model_path is None:
            model_path = os.path.join(cnn_dir, "p00.ckpt")
            
        self.model = Model()
        self.model.to(self.device)
        
        if os.path.exists(model_path):
            self.logger.info(f"Loading CNN model from {model_path}")
            try:
                ckpt = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(ckpt["state_dict"])
                self.model.eval()
                self.logger.info("CNN model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load CNN model: {e}")
                raise
        else:
            self.logger.error(f"Model file not found: {model_path}")
            
        # Load camera calibration
        if calibration_matrix_path is None:
            calibration_matrix_path = os.path.join(cnn_dir, "calibration_matrix.yaml")
            
        if os.path.exists(calibration_matrix_path):
            self.camera_matrix, self.dist_coefficients = cnn_utils.get_camera_matrix(calibration_matrix_path)
            self.logger.info(f"Loaded camera calibration from {calibration_matrix_path}")
        else:
            self.logger.error(f"Calibration file not found: {calibration_matrix_path}")
            self.camera_matrix, self.dist_coefficients = None, None
            
        # Setup landmarks and face model - use the ones defined at module level
        self.landmarks_ids = landmarks_ids  # Use the one defined at module level
        self.logger.info(f"Using landmarks IDs: {self.landmarks_ids}")
        
        # Initialize buffers for smoothing
        self.face_landmarks_buffer = []
        self.rvec_buffer = []
        self.tvec_buffer = []
        self.gaze_vector_buffer = []
        self.rvec = None
        self.tvec = None
        
        # Get monitor dimensions
        self.monitor_mm, self.monitor_pixels = cnn_utils.get_monitor_dimensions()
        self.logger.info(f"Monitor dimensions: {self.monitor_mm}mm, {self.monitor_pixels}px")
        
        # Initialize the face model - using the ones defined at module level
        self.face_model_all = face_model_all
        self.face_model = face_model
        
        self.logger.info(f"Face model shape: {self.face_model.shape}")
        
        # Initialize the plane equation
        self.plane = cnn_utils.plane_equation(np.eye(3), np.asarray([[0], [0], [0]]))
        
        # Setup direct camera access if enabled
        self.use_direct_camera = use_direct_camera
        self.camera = None
        if use_direct_camera:
            self.setup_direct_camera()
        
        # Create wrapper model for CNN output adjustment
        self.wrapper_model_name = wrapper_model_name
        self.logger.info(f"Creating {wrapper_model_name} wrapper model for CNN output adjustment")
        self.wrapper_model = create_model("cnn_wrapper", base_model=wrapper_model_name)
        self.is_calibrated = False
        
        # Debug variables
        self.last_gaze_point = None
        self.frame_count = 0
        
    def setup_direct_camera(self):
        """Setup direct camera access like in the test script."""
        try:
            self.logger.info("Setting up direct camera access")
            self.camera = cv2.VideoCapture(0)
            
            # Use identical settings as the test script
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Check webcam properties
            width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = self.camera.get(cv2.CAP_PROP_FPS)
            self.logger.info(f"Direct camera properties: {width}x{height} @ {fps}fps")
            
            if not self.camera.isOpened():
                self.logger.error("Failed to open direct camera")
                self.use_direct_camera = False
                
        except Exception as e:
            self.logger.error(f"Error setting up direct camera: {e}")
            self.use_direct_camera = False
            
    def get_frame(self):
        """Get a frame from the direct camera if enabled."""
        if not self.use_direct_camera or self.camera is None:
            return None
            
        ret, frame = self.camera.read()
        if not ret:
            self.logger.error("Failed to read frame from direct camera")
            return None
            
        return frame
        
    def release_camera(self):
        """Release the direct camera if it was used."""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        
    def extract_features(self, frame):
        """Extract features from a frame for gaze prediction.
        
        Args:
            frame: Input video frame
            
        Returns:
            features: Dictionary of features or None if extraction failed
            is_blinking: Boolean indicating if eyes are closed
        """
        if self.camera_matrix is None or self.dist_coefficients is None:
            self.logger.error("Camera not calibrated")
            return None, False
            
        
        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with face mesh - improve detection parameters
        image_rgb.flags.writeable = False
        results = self.face_mesh.process(image_rgb)
        
        if not results.multi_face_landmarks:
            self.logger.debug("No face detected")
            return None, False  # No face detected
            
        # Extract face landmarks
        face_landmarks = np.asarray([[landmark.x * width, landmark.y * height] 
                                    for landmark in results.multi_face_landmarks[0].landmark])
        
        # Check if we have enough landmarks
        if len(face_landmarks) < max(self.landmarks_ids) + 1:
            self.logger.error(f"Not enough landmarks detected: {len(face_landmarks)} < {max(self.landmarks_ids) + 1}")
            return None, False
            
        try:
            face_landmarks_subset = np.asarray([face_landmarks[i] for i in self.landmarks_ids])
        except IndexError as e:
            self.logger.error(f"IndexError accessing landmarks: {e}")
            return None, False
        
        # Debug output to check landmark detection
        # self.logger.debug(f"Landmarks shape: {face_landmarks_subset.shape}")
        
        # Smooth face landmarks
        self.face_landmarks_buffer.append(face_landmarks_subset)
        if len(self.face_landmarks_buffer) > 3:
            self.face_landmarks_buffer.pop(0)
        face_landmarks_smooth = np.mean(self.face_landmarks_buffer, axis=0)
        
        # Check for NaN values in landmarks
        if np.isnan(face_landmarks_smooth).any():
            self.logger.error("NaN values detected in landmarks")
            return None, False
        
        # Ensure face landmarks and model are correct shape before PnP
        if face_landmarks_smooth.shape[0] != self.face_model.shape[0]:
            self.logger.error(f"Landmark count mismatch: {face_landmarks_smooth.shape[0]} != {self.face_model.shape[0]}")
            return None, False
            
        # Debug output of face model and landmarks
        # self.logger.debug(f"Face model shape: {self.face_model.shape}, landmarks shape: {face_landmarks_smooth.shape}")
        
        # First try with EPNP method
        try:
            success, rvec, tvec = cv2.solvePnP(
                self.face_model, face_landmarks_smooth, 
                self.camera_matrix, self.dist_coefficients,
                flags=cv2.SOLVEPNP_EPNP,
                useExtrinsicGuess=bool(self.rvec is not None)
            )
        except cv2.error as e:
            self.logger.error(f"OpenCV error during EPNP: {e}")
            success = False
            
        # If EPNP failed, try ITERATIVE
        if not success:
            self.logger.debug("Failed to estimate head pose with EPNP, trying ITERATIVE")
            try:
                # Fall back to simpler method
                success, rvec, tvec = cv2.solvePnP(
                    self.face_model, face_landmarks_smooth, 
                    self.camera_matrix, self.dist_coefficients,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    useExtrinsicGuess=False  # Don't use previous guess if EPNP failed
                )
                if not success:
                    self.logger.debug("Failed to estimate head pose")
                    return None, False
            except cv2.error as e:
                self.logger.error(f"OpenCV error during solvePnP: {e}")
                return None, False
                
        # Refine pose estimation
        try:
            for _ in range(3):  # Only do a few iterations for refinement
                success, rvec, tvec = cv2.solvePnP(
                    self.face_model, face_landmarks_smooth, 
                    self.camera_matrix, self.dist_coefficients, 
                    rvec=rvec, tvec=tvec, 
                    useExtrinsicGuess=True, 
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
        except cv2.error as e:
            self.logger.error(f"OpenCV error during refinement: {e}")
            # Continue with the last successful rvec/tvec
            
        # Smooth pose parameters
        self.rvec_buffer.append(rvec)
        if len(self.rvec_buffer) > 3:
            self.rvec_buffer.pop(0)
        rvec = np.mean(self.rvec_buffer, axis=0)
        
        self.tvec_buffer.append(tvec)
        if len(self.tvec_buffer) > 3:
            self.tvec_buffer.pop(0)
        tvec = np.mean(self.tvec_buffer, axis=0)
        
        self.rvec, self.tvec = rvec, tvec
        
        try:
            # Get face landmarks in camera coordinate system
            face_model_transformed, face_model_all_transformed = cnn_utils.get_face_landmarks_in_ccs(
                self.camera_matrix, self.dist_coefficients, 
                frame.shape, results, self.face_model, 
                self.face_model_all, self.landmarks_ids
            )
            
            # Calculate eye centers
            left_eye_center = 0.5 * (face_model_transformed[:, 2] + face_model_transformed[:, 3]).reshape((3, 1))
            right_eye_center = 0.5 * (face_model_transformed[:, 0] + face_model_transformed[:, 1]).reshape((3, 1))
            face_center = face_model_transformed.mean(axis=1).reshape((3, 1))
            
            # Normalize images for CNN
            img_warped_left_eye, _, _ = cnn_preprocessing.normalize_single_image(
                image_rgb, rvec, None, left_eye_center, self.camera_matrix
            )
            img_warped_right_eye, _, _ = cnn_preprocessing.normalize_single_image(
                image_rgb, rvec, None, right_eye_center, self.camera_matrix
            )
            img_warped_face, _, rotation_matrix = cnn_preprocessing.normalize_single_image(
                image_rgb, rvec, None, face_center, self.camera_matrix, is_eye=False
            )
            
            # Return features needed for prediction
            features = {
                'face_center': face_center,
                'rotation_matrix': rotation_matrix,
                'full_face': img_warped_face,
                'right_eye': img_warped_right_eye,
                'left_eye': img_warped_left_eye
            }
            
            # CNN model doesn't have explicit blink detection
            is_blinking = False
            
            return features, is_blinking
            
        except Exception as e:
            self.logger.error(f"Error in feature extraction: {e}")
            return None, False
        
    def predict_(self, features_batch):
        """Predict gaze points from features.
        
        Args:
            features_batch: List of feature dictionaries from extract_features
            
        Returns:
            List of screen coordinates (x, y)
        """
        import torch
        
        results = []
        self.frame_count += 1
        
        for features in features_batch:
            if features is None:
                results.append((0, 0))
                continue
                
            face_center = features['face_center']
            rotation_matrix = features['rotation_matrix']
            
            # Prepare inputs for CNN
            person_idx = torch.Tensor([0]).unsqueeze(0).long().to(self.device)
            full_face_image = self.transform(image=features['full_face'])["image"].unsqueeze(0).float().to(self.device)
            left_eye_image = self.transform(image=features['left_eye'])["image"].unsqueeze(0).float().to(self.device)
            right_eye_image = self.transform(image=features['right_eye'])["image"].unsqueeze(0).float().to(self.device)
            
            # Get prediction from model
            with torch.no_grad():
                output = self.model(person_idx, full_face_image, right_eye_image, left_eye_image).squeeze(0).detach().cpu().numpy()
                
            # Convert 2D gaze to 3D vector
            gaze_vector_3d_normalized = cnn_utils.gaze_2d_to_3d(output)
            gaze_vector = np.dot(np.linalg.inv(rotation_matrix), gaze_vector_3d_normalized)
            
            # Smooth gaze vector
            self.gaze_vector_buffer.append(gaze_vector)
            if len(self.gaze_vector_buffer) > 10:
                self.gaze_vector_buffer.pop(0)
            gaze_vector = np.mean(self.gaze_vector_buffer, axis=0)
            
            # # Get plane parameters
            # plane_w = self.plane[0:3]
            # plane_b = self.plane[3]
            
            # # Get intersection with screen (raw CNN prediction)
            # result = cnn_utils.ray_plane_intersection(face_center.reshape(3), gaze_vector, plane_w, plane_b)
            # raw_point_on_screen = cnn_utils.get_point_on_screen(self.monitor_mm, self.monitor_pixels, result)
            
            # Apply wrapper model adjustment if calibrated
            if self.is_calibrated:
                adjusted_point = self.wrapper_model.predict_adjusted(gaze_vector)
                final_point = adjusted_point
            else:
                final_point = gaze_vector
            
            # # Log movement every few frames
            # if self.last_gaze_point is not None and self.frame_count % 30 == 0:
            #     dx = final_point[0] - self.last_gaze_point[0]
            #     dy = final_point[1] - self.last_gaze_point[1]
            #     distance = np.sqrt(dx*dx + dy*dy)
            #     if distance > 5:  # Only log significant movements
            #         self.logger.info(f"Gaze moved: ({self.last_gaze_point[0]}, {self.last_gaze_point[1]}) -> "
            #                         f"({final_point[0]}, {final_point[1]}), distance: {distance:.1f}px")
            
            # self.last_gaze_point = final_point
            results.append(final_point)
            
        return results
        
        
    def predict(self, features_batch):
        """Predict gaze points from features.
        
        Args:
            features_batch: List of feature dictionaries from extract_features
            
        Returns:
            List of screen coordinates (x, y)
        """
        import torch
        
        results = []
        self.frame_count += 1
        
        for features in features_batch:
            if features is None:
                results.append((0, 0))
                continue
                
            face_center = features['face_center']
            rotation_matrix = features['rotation_matrix']
            
            # Prepare inputs for CNN
            person_idx = torch.Tensor([0]).unsqueeze(0).long().to(self.device)
            full_face_image = self.transform(image=features['full_face'])["image"].unsqueeze(0).float().to(self.device)
            left_eye_image = self.transform(image=features['left_eye'])["image"].unsqueeze(0).float().to(self.device)
            right_eye_image = self.transform(image=features['right_eye'])["image"].unsqueeze(0).float().to(self.device)
            
            # Get prediction from model
            with torch.no_grad():
                output = self.model(person_idx, full_face_image, right_eye_image, left_eye_image).squeeze(0).detach().cpu().numpy()
                
            # Convert 2D gaze to 3D vector
            gaze_vector_3d_normalized = cnn_utils.gaze_2d_to_3d(output)
            gaze_vector = np.dot(np.linalg.inv(rotation_matrix), gaze_vector_3d_normalized)
            
            # Smooth gaze vector
            self.gaze_vector_buffer.append(gaze_vector)
            if len(self.gaze_vector_buffer) > 10:
                self.gaze_vector_buffer.pop(0)
            gaze_vector = np.mean(self.gaze_vector_buffer, axis=0)
            
            # Get plane parameters
            plane_w = self.plane[0:3]
            plane_b = self.plane[3]
            
            # Get intersection with screen (raw CNN prediction)
            result = cnn_utils.ray_plane_intersection(face_center.reshape(3), gaze_vector, plane_w, plane_b)
            raw_point_on_screen = cnn_utils.get_point_on_screen(self.monitor_mm, self.monitor_pixels, result)
            
            # Apply wrapper model adjustment if calibrated
            if self.is_calibrated:
                adjusted_point = self.wrapper_model.predict_adjusted(raw_point_on_screen)
                final_point = adjusted_point
            else:
                final_point = raw_point_on_screen
            
            # Log movement every few frames
            if self.last_gaze_point is not None and self.frame_count % 30 == 0:
                dx = final_point[0] - self.last_gaze_point[0]
                dy = final_point[1] - self.last_gaze_point[1]
                distance = np.sqrt(dx*dx + dy*dy)
                if distance > 5:  # Only log significant movements
                    self.logger.info(f"Gaze moved: ({self.last_gaze_point[0]}, {self.last_gaze_point[1]}) -> "
                                    f"({final_point[0]}, {final_point[1]}), distance: {distance:.1f}px")
            
            self.last_gaze_point = final_point
            results.append(final_point)
            
        return results
        
    def save_model(self, path: str):
        """Save model is not implemented for CNN."""
        self.logger.warning("save_model is not implemented for CNNGazeEstimator")
        return False
        
    def prepare_cnn_input(self, features):
        """
        Prepare CNN input tensors from extracted features.
        
        Args:
            features: Features dictionary from extract_features
            
        Returns:
            Tuple of (face_tensor, right_eye_tensor, left_eye_tensor) or (None, None, None) if failed
        """
        import torch
        
        if features is None:
            return None, None, None
            
        try:
            # Convert images to tensors using transform
            full_face_tensor = self.transform(image=features['full_face'])["image"].unsqueeze(0).float().to(self.device)
            right_eye_tensor = self.transform(image=features['right_eye'])["image"].unsqueeze(0).float().to(self.device)
            left_eye_tensor = self.transform(image=features['left_eye'])["image"].unsqueeze(0).float().to(self.device)
            
            return full_face_tensor, right_eye_tensor, left_eye_tensor
        except Exception as e:
            self.logger.error(f"Error preparing CNN input: {e}")
            return None, None, None
    
    def save_subject_bias(self, path: str):
        """
        Save the subject bias tensor to file.
        
        Args:
            path: Path to save the subject bias tensor
            
        Returns:
            True if successful, False otherwise
        """
        import torch
        import os
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Get the subject bias from the model
            subject_bias = self.model.subject_biases.detach().cpu()
            
            # Save to file
            torch.save({"subject_bias": subject_bias}, path)
            self.logger.info(f"Saved subject bias to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving subject bias: {e}")
            return False
    
    def load_subject_bias(self, path: str):
        """
        Load the subject bias tensor from file.
        
        Args:
            path: Path to the subject bias file
            
        Returns:
            True if successful, False otherwise
        """
        import torch
        import os
        
        if not os.path.exists(path):
            self.logger.warning(f"Subject bias file not found: {path}")
            return False
            
        try:
            # Load from file
            checkpoint = torch.load(path, map_location=self.device)
            
            # Set the subject bias in the model
            if "subject_bias" in checkpoint:
                self.model.subject_biases.data = checkpoint["subject_bias"].to(self.device)
                self.logger.info(f"Loaded subject bias from {path}")
                return True
            else:
                self.logger.error("Invalid subject bias file format")
                return False
        except Exception as e:
            self.logger.error(f"Error loading subject bias: {e}")
            return False
        
    def load_model(self, path: str):
        """Load a trained model.
        
        Args:
            path: Path to model checkpoint
            
        Returns:
            True if successful, False otherwise
        """
        import torch
        
        try:
            self.logger.info(f"Loading CNN model from {path}")
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False

    def calibrate_wrapper_model(self, features, targets):
        """
        Calibrate the wrapper model using collected samples.
        
        Args:
            features: List of raw CNN predictions (point_on_screen coordinates)
            targets: List of target coordinates (ground truth)
            
        Returns:
            True if calibration was successful, False otherwise
        """
        if not features or not targets:
            self.logger.warning("No features or targets provided for calibration")
            return False
            
        self.logger.info(f"Calibrating wrapper model with {len(features)} samples")
        
        # The features are already raw CNN predictions (point_on_screen), 
        # so we can directly train the wrapper model
        for raw_pred, target in zip(features, targets):
            self.wrapper_model.collect_training_sample(raw_pred, target)
            
        # Train the model from collected samples
        success = self.wrapper_model.train_from_collected()
        if success:
            self.is_calibrated = True
            self.logger.info("Wrapper model calibration successful")
        
        return success
        
    def save_wrapper_model(self, path):
        """
        Save the wrapper model to a file.
        
        Args:
            path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save the model
            self.wrapper_model.save(path)
            self.logger.info(f"Saved wrapper model to {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving wrapper model: {e}")
            return False
            
    def load_wrapper_model(self, path):
        """
        Load the wrapper model from a file.
        
        Args:
            path: Path to the model file
            
        Returns:
            True if successful, False otherwise
        """
        from eyetrax.models import BaseModel
        
        if not os.path.exists(path):
            self.logger.warning(f"Wrapper model file not found: {path}")
            return False
            
        try:
            self.wrapper_model = BaseModel.load(path)
            self.is_calibrated = True
            self.logger.info(f"Loaded wrapper model from {path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading wrapper model: {e}")
            return False

    def get_raw_prediction(self, features):
        """
        Get raw CNN prediction without wrapper adjustment.
        Used for calibration purposes.
        
        Args:
            features: Feature dictionary from extract_features
            
        Returns:
            Raw screen coordinates prediction
        """
        # Temporarily disable calibration
        original_is_calibrated = self.is_calibrated
        self.is_calibrated = False
        
        # Get raw prediction
        raw_prediction = self.predict([features])[0]
        
        # Restore calibration state
        self.is_calibrated = original_is_calibrated
        
        return raw_prediction

