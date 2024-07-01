import collections
import cv2
import numpy as np
import torch
import mediapipe as mp
from model import Model
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
from mpii_face_gaze_preprocessing import normalize_single_image
from utils import get_camera_matrix, get_face_landmarks_in_ccs, gaze_2d_to_3d, ray_plane_intersection,get_point_on_screen, get_monitor_dimensions,get_plane, calculate_bias_vector_and_plane
import os


class GazeTracker:
    def __init__(self, calibration_matrix_path, model_path):
        print("Initializing GazeTracker...")  # Debugging print statement
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.monitor_mm, self.monitor_pixels = get_monitor_dimensions()
        self.camera_matrix, self.dist_coefficients = get_camera_matrix(calibration_matrix_path)
        self.model = Model().to(self.device)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        print("Model loaded and set to evaluation mode.")  # Debugging print statement

        gpu_options = {
            "model_complexity": 1,
            "refine_landmarks": True,
            "max_num_faces": 1,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
            "use_gpu": True
        }
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, 
            max_num_faces=gpu_options['max_num_faces'],
            refine_landmarks=gpu_options['refine_landmarks'],
            min_detection_confidence=gpu_options['min_detection_confidence'],
            min_tracking_confidence=gpu_options['min_tracking_confidence']
        )
        self.bias, self.plane = calculate_bias_vector_and_plane("../gaze-data-collection/data/p00/data.csv",self.camera_matrix,self.dist_coefficients, self.model, self.face_mesh)


        self.smoothing_buffer = collections.deque(maxlen=3)
        self.rvec_buffer = collections.deque(maxlen=3)
        self.tvec_buffer = collections.deque(maxlen=3)
        self.gaze_vector_buffer = collections.deque(maxlen=10)
        self.gaze_points = collections.deque(maxlen=64)
        self.rvec, self.tvec = None, None
        print("GazeTracker initialized.")  # Debugging print statement

    def process_frame(self, frame, face_model, face_model_all, landmarks_ids):
        # print(f"Processing frame of shape: {frame.shape} and type: {frame.dtype}")  # Debugging print statement
        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = self.face_mesh.process(image_rgb)
        # print(f"Face mesh results: {results.multi_face_landmarks}")  # Debugging print statement

        if results.multi_face_landmarks:
            # print("Face landmarks detected")  # Debugging print statement
            face_landmarks = np.asarray([[landmark.x * width, landmark.y * height] for landmark in results.multi_face_landmarks[0].landmark])
            face_landmarks = np.asarray([face_landmarks[i] for i in landmarks_ids])
            self.smoothing_buffer.append(face_landmarks)
            face_landmarks = np.asarray(self.smoothing_buffer).mean(axis=0)

            success, self.rvec, self.tvec, inliers = cv2.solvePnPRansac(face_model, face_landmarks, self.camera_matrix, self.dist_coefficients, rvec=self.rvec, tvec=self.tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP)
            for _ in range(10):
                success, self.rvec, self.tvec = cv2.solvePnP(face_model, face_landmarks, self.camera_matrix, self.dist_coefficients, rvec=self.rvec, tvec=self.tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)

            self.rvec_buffer.append(self.rvec)
            self.rvec = np.asarray(self.rvec_buffer).mean(axis=0)
            self.tvec_buffer.append(self.tvec)
            self.tvec = np.asarray(self.tvec_buffer).mean(axis=0)

            face_model_transformed, face_model_all_transformed = get_face_landmarks_in_ccs(self.camera_matrix, self.dist_coefficients, frame.shape, results, face_model, face_model_all, landmarks_ids)
            left_eye_center = 0.5 * (face_model_transformed[:, 2] + face_model_transformed[:, 3]).reshape((3, 1))
            right_eye_center = 0.5 * (face_model_transformed[:, 0] + face_model_transformed[:, 1]).reshape((3, 1))
            face_center = face_model_transformed.mean(axis=1).reshape((3, 1))

            img_warped_left_eye, _, _ = normalize_single_image(image_rgb, self.rvec, None, left_eye_center, self.camera_matrix)
            img_warped_right_eye, _, _ = normalize_single_image(image_rgb, self.rvec, None, right_eye_center, self.camera_matrix)
            img_warped_face, _, rotation_matrix = normalize_single_image(image_rgb, self.rvec, None, face_center, self.camera_matrix, is_eye=False)

            transform = Compose([Normalize(), ToTensorV2()])
            person_idx = torch.Tensor([0]).unsqueeze(0).long().to(self.device)
            full_face_image = transform(image=img_warped_face)["image"].unsqueeze(0).float().to(self.device)
            left_eye_image = transform(image=img_warped_left_eye)["image"].unsqueeze(0).float().to(self.device)
            right_eye_image = transform(image=img_warped_right_eye)["image"].unsqueeze(0).float().to(self.device)

            output = self.model(person_idx, full_face_image, right_eye_image, left_eye_image).squeeze(0).detach().cpu().numpy()
            gaze_vector_3d_normalized = gaze_2d_to_3d(output)
            gaze_vector = np.dot(np.linalg.inv(rotation_matrix), gaze_vector_3d_normalized)

            self.gaze_vector_buffer.append(gaze_vector)
            gaze_vector = np.asarray(self.gaze_vector_buffer).mean(axis=0)

            plane_w = self.plane[:3]
            plane_b = self.plane[3]
            result = ray_plane_intersection(face_center.reshape(3), gaze_vector, plane_w, plane_b)
            point_on_screen = get_point_on_screen(self.monitor_mm, self.monitor_pixels, result)

            return point_on_screen
        
        print("No face landmarks detected")  # Debugging print statement
        return None
