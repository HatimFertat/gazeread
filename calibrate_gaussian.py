from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import collections
import time
from argparse import ArgumentParser

from albumentations import Compose, Normalize
import cv2
import pandas as pd
import mediapipe as mp
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from model import Model
from mpii_face_gaze_preprocessing import normalize_single_image
from utils import get_camera_matrix, get_face_landmarks_in_ccs, gaze_2d_to_3d, ray_plane_intersection, plane_equation, get_monitor_dimensions, get_point_on_screen
from visualization import Plot3DScene
from webcam import WebcamSource

from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.utils.optimize import _check_optimize_result
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import scipy.optimize
from scipy.optimize import minimize
from utils import calculate_calibration

def get_true_gaze_vector(true_screen_point,eye_center, monitor_mm, monitor_pixels, plane_w, plane_b):
    """
    Convert a true screen point to a true gaze vector.

    :param true_screen_point: (x, y) in pixel coordinates on the screen
    :param monitor_mm: (width, height) in millimeters of the monitor
    :param monitor_pixels: (width, height) in pixels of the monitor
    :param plane_w: normal vector of the plane
    :param plane_b: plane offset
    :param camera_matrix: Camera matrix for the transformation
    :param dist_coefficients: Distortion coefficients for the camera
    :param face_landmarks: Landmarks of the face used for pose estimation
    :return: True gaze vector in 3D
    """
    # Convert screen point to 3D point on the plane
    def screen_to_3d_point(screen_point, monitor_mm, monitor_pixels, plane_w, plane_b):
        # Convert screen coordinates to normalized coordinates in the plane
        normalized_x = screen_point[0] / monitor_pixels[0]
        normalized_y = screen_point[1] / monitor_pixels[1]

        # Convert normalized coordinates to millimeters
        point_mm = np.array([
            normalized_x * monitor_mm[0],
            normalized_y * monitor_mm[1],
            0
        ])

        # Calculate the 3D point in the plane
        d = plane_b
        point_3d = point_mm - d * plane_w

        return point_3d

    # Compute the gaze vector from the eye center to the 3D point on the plane
    def compute_gaze_vector(eye_center, point_3d):
        gaze_vector = point_3d - eye_center
        gaze_vector /= np.linalg.norm(gaze_vector)
        return gaze_vector

    # Process the true screen point to get the corresponding 3D point
    true_point_3d = screen_to_3d_point(true_screen_point, monitor_mm, monitor_pixels, plane_w, plane_b)
    # Compute the true gaze vector
    true_gaze_vector = compute_gaze_vector(eye_center.reshape(-1), true_point_3d.reshape(-1))

    return true_gaze_vector

def fit_gaze_vector_ridge(gaze_vectors, true_gaze_vectors):
    # Standardize the data
    scaler_x = StandardScaler().fit(gaze_vectors)
    scaler_y = StandardScaler().fit(true_gaze_vectors)
    gaze_vectors_scaled = scaler_x.transform(gaze_vectors)
    true_gaze_vectors_scaled = scaler_y.transform(true_gaze_vectors)

    # Fit Ridge regression model
    ridge = Ridge(alpha=1.0)
    ridge.fit(gaze_vectors_scaled, true_gaze_vectors_scaled)
    
    return ridge, scaler_x, scaler_y

def predict_gaze_vector_ridge(ridge, scaler_x, scaler_y, gaze_vector):
    gaze_vector_scaled = scaler_x.transform(gaze_vector)
    predicted_scaled = ridge.predict(gaze_vector_scaled)
    return scaler_y.inverse_transform(predicted_scaled)

def fit_screen_point_ridge(screen_points_raw, screen_points_true):
    scaler_x = StandardScaler().fit(screen_points_raw)
    scaler_y = StandardScaler().fit(screen_points_true)
    screen_points_raw_scaled = scaler_x.transform(screen_points_raw)
    screen_points_true_scaled = scaler_y.transform(screen_points_true)

    # Fit Ridge regression model
    ridge = Ridge(alpha=1.0)
    ridge.fit(screen_points_raw_scaled, screen_points_true_scaled)
    
    return ridge, scaler_x, scaler_y

def predict_screen_point_ridge(screen_point_calibrater, screen_point):
    [ridge, scaler_x, scaler_y] = screen_point_calibrater
    screen_point_scaled = scaler_x.transform(screen_point)
    predicted_scaled = ridge.predict(screen_point_scaled)
    return scaler_y.inverse_transform(predicted_scaled)


def process_image_for_calibration(image_path, GT_point_on_screen, camera_matrix, dist_coefficients,model, face_mesh=None):
    landmarks_ids = [33, 133, 362, 263, 61, 291, 1]  # reye, leye, mouth
    face_model_all = np.load("face_model.npy")
    face_model_all -= face_model_all[1]
    face_model_all *= np.array([1, -1, -1])  # fix axis
    face_model_all *= 10   
    face_model = np.asarray([face_model_all[i] for i in landmarks_ids])
    
    if face_mesh is None:
        face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    monitor_mm, monitor_pixels = get_monitor_dimensions()
    plane = plane_equation(np.eye(3), np.asarray([[0], [0], [0]]))
    plane_w = plane[0:3]
    plane_b = plane[3]

    image = cv2.imread(image_path)
    height, width, _ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)
    rvec,tvec = None,None

    if results.multi_face_landmarks:
        face_landmarks = np.asarray([[landmark.x * width, landmark.y * height] for landmark in results.multi_face_landmarks[0].landmark])
        face_landmarks = np.asarray([face_landmarks[i] for i in landmarks_ids])

        success, rvec, tvec, inliers = cv2.solvePnPRansac(face_model, face_landmarks, camera_matrix, dist_coefficients, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP)  # Initial fit

        # Estimate gaze vector using the model
        face_model_transformed, face_model_all_transformed = get_face_landmarks_in_ccs(camera_matrix, dist_coefficients, image.shape, results, face_model, face_model_all, landmarks_ids)
        left_eye_center = 0.5 * (face_model_transformed[:, 2] + face_model_transformed[:, 3]).reshape((3, 1))
        right_eye_center = 0.5 * (face_model_transformed[:, 0] + face_model_transformed[:, 1]).reshape((3, 1))
        face_center = face_model_transformed.mean(axis=1).reshape((3, 1))
        eye_center = (left_eye_center + right_eye_center) / 2.0  # average to get the overall eye center

        GT_gaze_vector = get_true_gaze_vector(GT_point_on_screen,eye_center, monitor_mm, monitor_pixels, plane_w, plane_b)

        img_warped_left_eye, _, _ = normalize_single_image(image_rgb, rvec, None, left_eye_center, camera_matrix)
        img_warped_right_eye, _, _ = normalize_single_image(image_rgb, rvec, None, right_eye_center, camera_matrix)
        img_warped_face, _, rotation_matrix = normalize_single_image(image_rgb, rvec, None, face_center, camera_matrix, is_eye=False)

        transform = Compose([Normalize(), ToTensorV2()])
        person_idx = torch.Tensor([0]).unsqueeze(0).long()
        full_face_image = transform(image=img_warped_face)["image"].unsqueeze(0).float().to(model.device)
        left_eye_image = transform(image=img_warped_left_eye)["image"].unsqueeze(0).float().to(model.device)
        right_eye_image = transform(image=img_warped_right_eye)["image"].unsqueeze(0).float().to(model.device)

        output = model(person_idx, full_face_image, right_eye_image, left_eye_image).squeeze(0).detach().cpu().numpy()
        gaze_vector_3d_normalized = gaze_2d_to_3d(output)
        gaze_vector = np.dot(np.linalg.inv(rotation_matrix), gaze_vector_3d_normalized)

        plane_w = plane[:3]
        plane_b = plane[3]
        result = ray_plane_intersection(face_center.reshape(3), gaze_vector, plane_w, plane_b)
        point_on_screen = get_point_on_screen(monitor_mm, monitor_pixels, result)

        GTresult = ray_plane_intersection(face_center.reshape(3), GT_gaze_vector, plane_w, plane_b)
        GTpoint_on_screen = get_point_on_screen(monitor_mm, monitor_pixels, GTresult)

    # Ensure the bias tensor is on the same device as the model
    return gaze_vector,GT_gaze_vector, point_on_screen,GTpoint_on_screen


def calibrate_gaze_tracker(csvpath,calibration_matrix_path):
    gaze_vectors = []
    true_vectors = []
    points_raw = []
    points_true = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model().to(device)
    checkpoint = torch.load("p00.ckpt")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    base_path = "../gaze-data-collection/data/p00/"
    calibration_data = pd.read_csv(csvpath)
    camera_matrix, dist_coefficients = get_camera_matrix(calibration_matrix_path)
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    
    for index, row in calibration_data.iterrows():
        img_path = base_path + row['file_name']
        true_point = eval(row['point_on_screen'])  # Convert string to tuple

        # Process the image to get the raw gaze vector and screen point
        raw_gaze_vector, true_gaze_vector, raw_screen_point, GTscreenpoint = process_image_for_calibration(img_path,true_point,camera_matrix,dist_coefficients,model,face_mesh)
        

        gaze_vectors.append(raw_gaze_vector)
        true_vectors.append(true_gaze_vector)
        
        points_raw.append(raw_screen_point)
        points_true.append(true_point)

    gaze_vector_gp, gaze_vector_scaler_x, gaze_vector_scaler_y = fit_gaze_vector_ridge(gaze_vectors, true_vectors)
    screen_point_gp, screen_point_scaler_x, screen_point_scaler_y = fit_screen_point_ridge(points_raw, points_true)

    return gaze_vector_gp, gaze_vector_scaler_x, gaze_vector_scaler_y, screen_point_gp, screen_point_scaler_x, screen_point_scaler_y


if __name__ == '__main__':
    # Example usage
    calibration_matrix_path = './calibration_matrix.yaml'
    csvpath = "../gaze-data-collection/data/p00/data.csv"
    # gaze_vector_gp, gaze_vector_scaler_x, gaze_vector_scaler_y, screen_point_gp, screen_point_scaler_x, screen_point_scaler_y = calibrate_gaze_tracker(csvpath,calibration_matrix_path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_path = "../gaze-data-collection/data/p00/"
    csvpath = "../gaze-data-collection/data/p00/data.csv"
    
    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    camera_matrix, dist_coeffs = get_camera_matrix('./calibration_matrix.yaml')
    model = Model().to(device)
    checkpoint = torch.load('p00.ckpt')
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    known_points,estimated_points = calculate_calibration(csvpath,calibration_matrix_path,model,face_mesh)
    screen_point_gp, screen_point_scaler_x, screen_point_scaler_y = fit_screen_point_ridge(estimated_points, known_points)    

    # print(gaze_vector_gp,screen_point_gp)

