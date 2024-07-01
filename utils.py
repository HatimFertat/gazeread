from enum import Enum
from typing import Tuple, Union

import cv2
import numpy as np
import yaml
import pandas as pd
import mediapipe as mp
from face_model_array import face_model_all
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
from mpii_face_gaze_preprocessing import normalize_single_image
import torch
from model import Model

face_model_all -= face_model_all[1]
face_model_all *= np.array([1, -1, -1])
face_model_all *= 10

landmarks_ids = [33, 133, 362, 263, 61, 291, 1]
face_model = np.asarray([face_model_all[i] for i in landmarks_ids])

def get_monitor_dimensions() -> Union[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[None, None]]:
    """
    Get monitor dimensions from Gdk.
    from on https://github.com/NVlabs/few_shot_gaze/blob/master/demo/monitor.py
    :return: tuple of monitor width and height in mm and pixels or None
    """
    try:
        from screeninfo import get_monitors
        
        default_screen = get_monitors()[0].__dict__

        h_mm = default_screen['height_mm']
        w_mm = default_screen['width_mm']

        h_pixels = default_screen['height']
        w_pixels = default_screen['width']

        return (w_mm, h_mm), (w_pixels, h_pixels)

    except ModuleNotFoundError:
        return None, None


FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.5
TEXT_THICKNESS = 2


class TargetOrientation(Enum):
    UP = 82
    DOWN = 84
    LEFT = 81
    RIGHT = 83


def get_camera_matrix(calibration_matrix_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load camera_matrix and dist_coefficients from `calibration_matrix_path`.

    :param base_path: base path of data
    :return: camera intrinsic matrix and dist_coefficients
    """
    with open(calibration_matrix_path, 'r') as file:
        calibration_matrix = yaml.safe_load(file)
    camera_matrix = np.asarray(calibration_matrix['camera_matrix']).reshape(3, 3)
    dist_coefficients = np.asarray(calibration_matrix['dist_coeff'])
    return camera_matrix, dist_coefficients


def get_face_landmarks_in_ccs(camera_matrix, dist_coefficients, shape, results, face_model, face_model_all, landmarks_ids):
    """
    Fit `face_model` onto `face_landmarks` using `solvePnP`.

    :param camera_matrix: camera intrinsic matrix
    :param dist_coefficients: distortion coefficients
    :param shape: image shape
    :param results: output of MediaPipe FaceMesh
    :return: full face model in the camera coordinate system
    """
    height, width, _ = shape
    face_landmarks = np.asarray([[landmark.x * width, landmark.y * height] for landmark in results.multi_face_landmarks[0].landmark])
    face_landmarks = np.asarray([face_landmarks[i] for i in landmarks_ids])

    rvec, tvec = None, None
    success, rvec, tvec, inliers = cv2.solvePnPRansac(face_model, face_landmarks, camera_matrix, dist_coefficients, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP)  # Initial fit
    for _ in range(10):
        success, rvec, tvec = cv2.solvePnP(face_model, face_landmarks, camera_matrix, dist_coefficients, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)  # Second fit for higher accuracy

    head_rotation_matrix, _ = cv2.Rodrigues(rvec.reshape(-1))
    return np.dot(head_rotation_matrix, face_model.T) + tvec.reshape((3, 1)), np.dot(head_rotation_matrix, face_model_all.T) + tvec.reshape((3, 1))  # 3D positions of facial landmarks


def gaze_2d_to_3d(gaze: np.ndarray) -> np.ndarray:
    """
    pitch and gaze to 3d vector

    :param gaze: pitch and gaze vector
    :return: 3d vector
    """
    x = -np.cos(gaze[0]) * np.sin(gaze[1])
    y = -np.sin(gaze[0])
    z = -np.cos(gaze[0]) * np.cos(gaze[1])
    return np.array([x, y, z])


def ray_plane_intersection(support_vector: np.ndarray, direction_vector: np.ndarray, plane_normal: np.ndarray, plane_d: np.ndarray) -> np.ndarray:
    """
    Calulate the intersection between the gaze ray and the plane that represents the monitor.

    :param support_vector: support vector of the gaze
    :param direction_vector: direction vector of the gaze
    :param plane_normal: normal of the plane
    :param plane_d: d of the plane
    :return: point in 3D where the the person is looking at on the screen
    """
    a11 = direction_vector[1]
    a12 = -direction_vector[0]
    b1 = direction_vector[1] * support_vector[0] - direction_vector[0] * support_vector[1]

    a22 = direction_vector[2]
    a23 = -direction_vector[1]
    b2 = direction_vector[2] * support_vector[1] - direction_vector[1] * support_vector[2]

    line_w = np.array([[a11, a12, 0], [0, a22, a23]])
    line_b = np.array([[b1], [b2]])

    matrix = np.insert(line_w, 2, plane_normal, axis=0)
    bias = np.insert(line_b, 2, plane_d, axis=0)

    return np.linalg.solve(matrix, bias).reshape(3)


def plane_equation(rmat: np.ndarray, tmat: np.ndarray) -> np.ndarray:
    """
    Computes the equation of x-y plane.
    The normal vector of the plane is z-axis in rotation matrix. And tmat provide on point in the plane.

    :param rmat: rotation matrix
    :param tmat: translation matrix
    :return: (a, b, c, d), where the equation of plane is ax + by + cz = d
    """

    assert type(rmat) == type(np.zeros(0)) and rmat.shape == (3, 3), "There is an error about rmat."
    assert type(tmat) == type(np.zeros(0)) and tmat.size == 3, "There is an error about tmat."

    n = rmat[:, 2]
    origin = np.reshape(tmat, (3))

    a = n[0]
    b = n[1]
    c = n[2]

    d = origin[0] * n[0] + origin[1] * n[1] + origin[2] * n[2]
    return np.array([a, b, c, d])


def get_point_on_screen(monitor_mm: Tuple[float, float], monitor_pixels: Tuple[float, float], result: np.ndarray) -> Tuple[int, int]:
    """
    Calculate point in screen in pixels.

    :param monitor_mm: dimensions of the monitor in mm
    :param monitor_pixels: dimensions of the monitor in pixels
    :param result: predicted point on the screen in mm
    :return: point in screen in pixels
    """
    result_x = result[0]
    result_x = -result_x + monitor_mm[0] / 2
    result_x = result_x * (monitor_pixels[0] / monitor_mm[0])

    result_y = result[1]
    result_y = result_y - 20  # 20 mm offset
    result_y = min(result_y, monitor_mm[1])
    result_y = result_y * (monitor_pixels[1] / monitor_mm[1])

    return tuple(np.asarray([result_x, result_y]).round().astype(int))

def fit_plane(points):
    points = np.array(points)
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    _, _, vh = np.linalg.svd(centered_points)
    normal = vh[2, :]
    d = -centroid.dot(normal)
    return np.append(normal, d)

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

def get_plane(csvpath,camera_matrix,dist_coefficients,face_mesh=None):
    base_path = "../gaze-data-collection/data/p00/"
    calibration_data = pd.read_csv(csvpath)

    if face_mesh == None:
        face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    calibration_points_3d = []
    calibration_points_2d = []

    for index, row in calibration_data.iterrows():
        image_path = base_path + row['file_name']
        point_on_screen = eval(row['point_on_screen'])  # Convert string to tuple
        head_rotation_matrix, tvec = detect_landmarks_and_estimate_gaze(image_path, face_mesh, camera_matrix, dist_coefficients)
        if head_rotation_matrix is not None and tvec is not None:
            calibration_points_3d.append((head_rotation_matrix @ tvec).reshape(-1))
            calibration_points_2d.append(point_on_screen)

    plane_parameters = fit_plane(calibration_points_3d)
    return plane_parameters

def calculate_bias_vector_and_plane(csvpath, camera_matrix, dist_coefficients,model, face_mesh=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_path = "../gaze-data-collection/data/p00/"

    try:
        calibration_data = pd.read_csv(csvpath)
    except:
        print("EXCEPT \n\n\n\n\n\n\n\n")
        return torch.tensor([0,0])

    if face_mesh is None:
        face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    estimated_gaze_vectors = []
    known_points = []

    monitor_mm, monitor_pixels = get_monitor_dimensions()
    plane = get_plane("../gaze-data-collection/data/p00/data.csv",camera_matrix,dist_coefficients,face_mesh)


    for index, row in calibration_data.iterrows():
        image_path = base_path + row['file_name']
        point_on_screen = eval(row['point_on_screen'])  # Convert string to tuple

        image = cv2.imread(image_path)
        height, width, _ = image.shape
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            face_landmarks = np.asarray([[landmark.x * width, landmark.y * height] for landmark in results.multi_face_landmarks[0].landmark])
            face_landmarks = np.asarray([face_landmarks[i] for i in landmarks_ids])

            success, rvec, tvec, inliers = cv2.solvePnPRansac(face_model, face_landmarks, camera_matrix, dist_coefficients, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP)
            head_rotation_matrix, _ = cv2.Rodrigues(rvec.reshape(-1))

            # Estimate gaze vector using the model
            face_model_transformed, face_model_all_transformed = get_face_landmarks_in_ccs(camera_matrix, dist_coefficients, image.shape, results, face_model, face_model_all, landmarks_ids)
            left_eye_center = 0.5 * (face_model_transformed[:, 2] + face_model_transformed[:, 3]).reshape((3, 1))
            right_eye_center = 0.5 * (face_model_transformed[:, 0] + face_model_transformed[:, 1]).reshape((3, 1))
            face_center = face_model_transformed.mean(axis=1).reshape((3, 1))

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

            known_points.append(point_on_screen)

            plane_w = plane[:3]
            plane_b = plane[3]
            result = ray_plane_intersection(face_center.reshape(3), gaze_vector, plane_w, plane_b)
            point_on_screen = get_point_on_screen(monitor_mm, monitor_pixels, result)

            estimated_gaze_vectors.append(point_on_screen)

    estimated_gaze_vectors = np.array(estimated_gaze_vectors)
    known_points = np.array(known_points)

    # Calculate the bias vector
    bias = np.mean(known_points - estimated_gaze_vectors, axis=0)

    # Ensure the bias tensor is on the same device as the model
    return torch.tensor(bias, dtype=torch.float32, device=model.device), plane
