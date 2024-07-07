from enum import Enum
from typing import Tuple, Union

import cv2
import numpy as np
import yaml
import pandas as pd
import mediapipe as mp
from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2
from mpii_face_gaze_preprocessing import normalize_single_image
import torch
from model import Model
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler



face_model_all = np.load("face_model.npy")
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

def calculate_calibration(csvpath, camera_matrix_path, face_mesh=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_path = "../gaze-data-collection/data/p00/"

    model = Model().to(device)
    checkpoint = torch.load("p00.ckpt")
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    camera_matrix,dist_coeffs = get_camera_matrix(camera_matrix_path)

    face_model_all = np.load("face_model.npy")
    face_model_all -= face_model_all[1]
    face_model_all *= np.array([1, -1, -1])
    face_model_all *= 10

    landmarks_ids = [33, 133, 362, 263, 61, 291, 1]
    face_model = np.asarray([face_model_all[i] for i in landmarks_ids])

    try:
        calibration_data = pd.read_csv(csvpath)
    except:
        print("EXCEPT \n\n\n\n\n\n\n\n")
        return torch.tensor([0,0])

    if face_mesh is None:
        face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    estimated_points = []
    known_points = []

    monitor_mm, monitor_pixels = get_monitor_dimensions()
    # plane = get_plane("../gaze-data-collection/data/p00/data.csv",camera_matrix,dist_coefficients,face_mesh)
    plane = plane_equation(np.eye(3), np.asarray([[0], [0], [0]]))
    plane_w = plane[:3]
    plane_b = plane[3]
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

            success, rvec, tvec, inliers = cv2.solvePnPRansac(face_model, face_landmarks, camera_matrix, dist_coeffs, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP)
            head_rotation_matrix, _ = cv2.Rodrigues(rvec.reshape(-1))

            # Estimate gaze vector using the model
            face_model_transformed, face_model_all_transformed = get_face_landmarks_in_ccs(camera_matrix, dist_coeffs, image.shape, results, face_model, face_model_all, landmarks_ids)
            left_eye_center = 0.5 * (face_model_transformed[:, 2] + face_model_transformed[:, 3]).reshape((3, 1))
            right_eye_center = 0.5 * (face_model_transformed[:, 0] + face_model_transformed[:, 1]).reshape((3, 1))
            face_center = face_model_transformed.mean(axis=1).reshape((3, 1))

            img_warped_left_eye, _, _ = normalize_single_image(image_rgb, rvec, None, left_eye_center, camera_matrix)
            img_warped_right_eye, _, _ = normalize_single_image(image_rgb, rvec, None, right_eye_center, camera_matrix)
            img_warped_face, _, rotation_matrix = normalize_single_image(image_rgb, rvec, None, face_center, camera_matrix, is_eye=False)

            plane = plane_equation(rotation_matrix, np.asarray([[0], [0], [0]]))
            plane_w = plane[0:3]
            plane_b = plane[3]
            
            transform = Compose([Normalize(), ToTensorV2()])
            person_idx = torch.Tensor([0]).unsqueeze(0).long()
            full_face_image = transform(image=img_warped_face)["image"].unsqueeze(0).float().to(model.device)
            left_eye_image = transform(image=img_warped_left_eye)["image"].unsqueeze(0).float().to(model.device)
            right_eye_image = transform(image=img_warped_right_eye)["image"].unsqueeze(0).float().to(model.device)

            output = model(person_idx, full_face_image, right_eye_image, left_eye_image).squeeze(0).detach().cpu().numpy()
            gaze_vector_3d_normalized = gaze_2d_to_3d(output)
            gaze_vector = np.dot(np.linalg.inv(rotation_matrix), gaze_vector_3d_normalized)

            known_points.append(point_on_screen)

            result = ray_plane_intersection(face_center.reshape(3), gaze_vector, plane_w, plane_b)
            point_on_screen = get_point_on_screen(monitor_mm, monitor_pixels, result)

            estimated_points.append(point_on_screen)

    estimated_points = np.array(estimated_points)
    known_points = np.array(known_points)

    # Calculate the bias vector
    # bias = np.mean(known_points - estimated_points, axis=0)

    # Ensure the bias tensor is on the same device as the model
    return known_points,estimated_points

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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_path = "../gaze-data-collection/data/p00/"
    csvpath = "../gaze-data-collection/data/p00/data.csv"
    
    gpu_options = {
            "model_complexity": 1,
            "refine_landmarks": True,
            "max_num_faces": 1,
            "min_detection_confidence": 0.5,
            "min_tracking_confidence": 0.5,
            "use_gpu": True
        }
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, 
        max_num_faces=gpu_options['max_num_faces'],
        refine_landmarks=gpu_options['refine_landmarks'],
        min_detection_confidence=gpu_options['min_detection_confidence'],
        min_tracking_confidence=gpu_options['min_tracking_confidence']
    )
    camera_matrix, dist_coeffs = get_camera_matrix('./calibration_matrix.yaml')
    model = Model().to(device)
    checkpoint = torch.load('p00.ckpt')
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    known_points,estimated_points = calculate_calibration(csvpath,camera_matrix,dist_coeffs,model,face_mesh)

    print("done")

