import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
from typing import Tuple
import yaml

face_model_all = np.load("face_model.npy")
face_model_all -= face_model_all[1]
face_model_all *= np.array([1, -1, -1])
face_model_all *= 10

landmarks_ids = [33, 133, 362, 263, 61, 291, 1]
face_model = np.asarray([face_model_all[i] for i in landmarks_ids])

def get_plane_parameters(csv_path: str, calibration_matrix_path: str) -> np.ndarray:
    """
    Computes the plane parameters from calibration data.
    
    :param csv_path: Path to the CSV file containing calibration data.
    :param calibration_matrix_path: Path to the camera calibration matrix file.
    :param face_model: 3D face model landmarks.
    :param face_model_all: 3D face model landmarks for all points.
    :param landmarks_ids: List of landmark IDs used for gaze tracking.
    :return: Plane parameters (a, b, c, d) where the equation of plane is ax + by + cz = d.
    """
    def load_calibration_data(csv_path):
        calibration_data = pd.read_csv(csv_path)
        return calibration_data

    def detect_landmarks_and_estimate_gaze(image_path, face_mesh, camera_matrix, dist_coefficients):
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            face_landmarks = np.asarray([[landmark.x * width, landmark.y * height] for landmark in results.multi_face_landmarks[0].landmark])
            face_landmarks = np.asarray([face_landmarks[i] for i in landmarks_ids])

            success, rvec, tvec, inliers = cv2.solvePnPRansac(face_model, face_landmarks, camera_matrix, dist_coefficients, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP)
            head_rotation_matrix, _ = cv2.Rodrigues(rvec.reshape(-1))
            return head_rotation_matrix, tvec
        return None, None

    def fit_plane(points):
        points = np.array(points)
        centroid = np.mean(points, axis=0)
        centered_points = points - centroid
        _, _, vh = np.linalg.svd(centered_points)
        normal = vh[2, :]
        d = -centroid.dot(normal)
        return np.append(normal, d)

    # Load camera matrix and distortion coefficients
    with open(calibration_matrix_path, 'r') as file:
        calibration_matrix = yaml.safe_load(file)
    camera_matrix = np.asarray(calibration_matrix['camera_matrix']).reshape(3, 3)
    dist_coefficients = np.asarray(calibration_matrix['dist_coeff'])

    # Initialize MediaPipe FaceMesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    # Load calibration data
    calibration_data = load_calibration_data(csv_path)

    calibration_points_3d = []
    calibration_points_2d = []

    for index, row in calibration_data.iterrows():
        image_path = row['file_name']
        point_on_screen = eval(row['point_on_screen'])  # Convert string to tuple
        head_rotation_matrix, tvec = detect_landmarks_and_estimate_gaze(image_path, face_mesh, camera_matrix, dist_coefficients)
        if head_rotation_matrix is not None and tvec is not None:
            calibration_points_3d.append((head_rotation_matrix @ tvec).reshape(-1))
            calibration_points_2d.append(point_on_screen)

    # Fit the plane
    plane_parameters = fit_plane(calibration_points_3d)
    
    return plane_parameters

