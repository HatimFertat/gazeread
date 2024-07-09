import collections
import time
from argparse import ArgumentParser

import albumentations as A
import cv2
import mediapipe as mp
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from model import Model
from mpii_face_gaze_preprocessing import normalize_single_image
from utils import get_camera_matrix, get_face_landmarks_in_ccs, gaze_2d_to_3d, ray_plane_intersection, plane_equation, get_monitor_dimensions, get_point_on_screen
from visualization import Plot3DScene
from webcam import WebcamSource
from utils import calculate_calibration,fit_screen_point, predict_screen_point

face_model_all = np.load("face_model.npy")

# face model from https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj

face_model_all -= face_model_all[1]
face_model_all *= np.array([1, -1, -1])  # fix axis
face_model_all *= 10

landmarks_ids = [33, 133, 362, 263, 61, 291, 1]  # reye, leye, mouth
face_model = np.asarray([face_model_all[i] for i in landmarks_ids])

WINDOW_NAME = 'laser pointer preview'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(calibration_matrix_path: str, monitor_mm=None, monitor_pixels=None, model=None, visualize_preprocessing=False, visualize_laser_pointer=True, visualize_3d=False, screen_point_calibrater=None, method='ridge'):
    # setup webcam
    source = WebcamSource(width=1280, height=720, fps=60, buffer_size=10)
    camera_matrix, dist_coefficients = get_camera_matrix(calibration_matrix_path)

    # setup monitor
    if monitor_mm is None or monitor_pixels is None:
        monitor_mm, monitor_pixels = get_monitor_dimensions()
        if monitor_mm is None or monitor_pixels is None:
            raise ValueError('Please supply monitor dimensions manually as they could not be retrieved.')
    print(f'Found default monitor of size {monitor_mm[0]}x{monitor_mm[1]}mm and {monitor_pixels[0]}x{monitor_pixels[1]}px.')


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
    # face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    # TODO load calibrated screen position
    # plane = get_plane("../gaze-data-collection/data/p00/data.csv",camera_matrix,dist_coefficients,face_mesh)

    plane = plane_equation(np.eye(3), np.asarray([[0], [0], [0]]))
    plane_w = plane[0:3]
    plane_b = plane[3]
    
    fps_deque = collections.deque(maxlen=60)  # to measure the FPS
    prev_frame_time = 0

    smoothing_buffer = collections.deque(maxlen=3)
    rvec_buffer = collections.deque(maxlen=3)
    tvec_buffer = collections.deque(maxlen=3)
    gaze_vector_buffer = collections.deque(maxlen=10)
    rvec, tvec = None, None
    gaze_points = collections.deque(maxlen=64)

    # bias = np.array([1409.76165803, 1134.15025907])

    transform = A.Compose([
        A.Normalize(),
        ToTensorV2()
    ])

    if visualize_laser_pointer:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    plot_3d_scene = Plot3DScene(face_model, monitor_mm[0], monitor_mm[1], 20) if visualize_3d else None

    for frame_idx, frame in enumerate(source):
        height, width, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image_rgb.flags.writeable = False
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            # head pose estimation
            face_landmarks = np.asarray([[landmark.x * width, landmark.y * height] for landmark in results.multi_face_landmarks[0].landmark])
            face_landmarks = np.asarray([face_landmarks[i] for i in landmarks_ids])
            smoothing_buffer.append(face_landmarks)
            face_landmarks = np.asarray(smoothing_buffer).mean(axis=0)

            success, rvec, tvec, inliers = cv2.solvePnPRansac(face_model, face_landmarks, camera_matrix, dist_coefficients, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP)  # Initial fit
            for _ in range(10):
                success, rvec, tvec = cv2.solvePnP(face_model, face_landmarks, camera_matrix, dist_coefficients, rvec=rvec, tvec=tvec, useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)  # Second fit for higher accuracy

            rvec_buffer.append(rvec)
            rvec = np.asarray(rvec_buffer).mean(axis=0)
            tvec_buffer.append(tvec)
            tvec = np.asarray(tvec_buffer).mean(axis=0)

            # data preprocessing
            face_model_transformed, face_model_all_transformed = get_face_landmarks_in_ccs(camera_matrix, dist_coefficients, frame.shape, results, face_model, face_model_all, landmarks_ids)
            left_eye_center = 0.5 * (face_model_transformed[:, 2] + face_model_transformed[:, 3]).reshape((3, 1))  # center eye
            right_eye_center = 0.5 * (face_model_transformed[:, 0] + face_model_transformed[:, 1]).reshape((3, 1))  # center eye
            face_center = face_model_transformed.mean(axis=1).reshape((3, 1))

            img_warped_left_eye, _, _ = normalize_single_image(image_rgb, rvec, None, left_eye_center, camera_matrix)
            img_warped_right_eye, _, _ = normalize_single_image(image_rgb, rvec, None, right_eye_center, camera_matrix)
            img_warped_face, _, rotation_matrix = normalize_single_image(image_rgb, rvec, None, face_center, camera_matrix, is_eye=False)

            plane = plane_equation(rotation_matrix, np.asarray([[0], [0], [0]]))
            plane_w = plane[0:3]
            plane_b = plane[3]

            if visualize_preprocessing:
                cv2.imshow('img_warped_left_eye', cv2.cvtColor(img_warped_left_eye, cv2.COLOR_RGB2BGR))
                cv2.imshow('img_warped_right_eye', cv2.cvtColor(img_warped_right_eye, cv2.COLOR_RGB2BGR))
                cv2.imshow('img_warped_face', cv2.cvtColor(img_warped_face, cv2.COLOR_RGB2BGR))

            person_idx = torch.Tensor([0]).unsqueeze(0).long().to(device)  # TODO adapt this depending on the loaded model
            full_face_image = transform(image=img_warped_face)["image"].unsqueeze(0).float().to(device)
            left_eye_image = transform(image=img_warped_left_eye)["image"].unsqueeze(0).float().to(device)
            right_eye_image = transform(image=img_warped_right_eye)["image"].unsqueeze(0).float().to(device)

            # prediction
            output = model(person_idx, full_face_image, right_eye_image, left_eye_image).squeeze(0).detach().cpu().numpy()
            gaze_vector_3d_normalized = gaze_2d_to_3d(output)
            gaze_vector = np.dot(np.linalg.inv(rotation_matrix), gaze_vector_3d_normalized)

            gaze_vector_buffer.append(gaze_vector)
            gaze_vector = np.asarray(gaze_vector_buffer).mean(axis=0)

            # if gaze_vector_gp is not None: gaze_vector = predict_gaze_vector_gp(gaze_vector_gp, [gaze_vector])[0]

            # gaze vector to screen
            result = ray_plane_intersection(face_center.reshape(3), gaze_vector, plane_w, plane_b)
            point_on_screen = get_point_on_screen(monitor_mm, monitor_pixels, result)

            oldpt = point_on_screen
            if screen_point_calibrater is not None: point_on_screen = predict_screen_point(screen_point_calibrater, [point_on_screen], method)[0]
            point_on_screen = (int(point_on_screen[0]),int(point_on_screen[1]))
            # print(oldpt, point_on_screen)
            
            # point_on_screen = (point_on_screen[0] + bias[0], point_on_screen[1] + bias[1])
            
            if visualize_laser_pointer:
                display = np.ones((monitor_pixels[1], monitor_pixels[0], 3), np.float32)

                gaze_points.appendleft(point_on_screen)

                for idx in range(1, len(gaze_points)):
                    thickness = round((len(gaze_points) - idx) / len(gaze_points) * 5) + 1
                    cv2.line(display, gaze_points[idx - 1], gaze_points[idx], (0, 0, 255), thickness)
                if frame_idx % 2 == 0:
                    cv2.imshow(WINDOW_NAME, display)

            if visualize_3d:
                plot_3d_scene.plot_face_landmarks(face_model_all_transformed)
                plot_3d_scene.plot_center_point(face_center, gaze_vector)
                plot_3d_scene.plot_point_on_screen(result)
                plot_3d_scene.update_canvas()

        new_frame_time = time.time()
        fps_deque.append(1 / (new_frame_time - prev_frame_time))
        prev_frame_time = new_frame_time
        if frame_idx % 60 == 0:
            print(f'FPS: {np.mean(fps_deque):5.2f}')

        # Add waitKey to handle window events
        if (cv2.waitKey(1) & 0xFF == 27) or (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    # Cleanup
    source.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--calibration_matrix_path", type=str, default='./calibration_matrix.yaml')
    parser.add_argument("--model_path", type=str, default='./p00.ckpt')
    parser.add_argument("--monitor_mm", type=str, default=None)
    parser.add_argument("--monitor_pixels", type=str, default=None)
    parser.add_argument("--visualize_preprocessing", type=bool, default=False)
    parser.add_argument("--visualize_laser_pointer", type=bool, default=True)
    parser.add_argument("--visualize_3d", type=bool, default=False)
    parser.add_argument("--calibrate_gaze", action='store_true', default=False)
    parser.add_argument("--calibrate_screen_point", action='store_true', default=False)
    parser.add_argument("--method", type=str, default='poly')
    args = parser.parse_args()

    if args.monitor_mm is not None:
        args.monitor_mm = tuple(map(int, args.monitor_mm.split(',')))
    if args.monitor_pixels is not None:
        args.monitor_pixels = tuple(map(int, args.monitor_pixels.split(',')))

    model = Model().to(device)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    screen_point_calibrater = None  
    if args.calibrate_screen_point:
        start = time.time()
        csvpath = "../gaze-data-collection/data/p00/data.csv"
        face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

        known_points,estimated_points = calculate_calibration(csvpath,args.calibration_matrix_path,face_mesh)
        if args.method == 'poly' or args.method == 'tps':
            poly_reg, poly, screen_point_scaler_x, screen_point_scaler_y = fit_screen_point(estimated_points, known_points, method=args.method) 
            screen_point_calibrater = [poly_reg, poly, screen_point_scaler_x, screen_point_scaler_y]
        else:
            screen_point_gp, screen_point_scaler_x, screen_point_scaler_y = fit_screen_point(estimated_points, known_points, method=args.method)   
            screen_point_calibrater = [screen_point_gp, screen_point_scaler_x, screen_point_scaler_y]
        calib_time = time.time()-start
        print(f'Calibration done in {calib_time:.2f}')

    try:
        main(args.calibration_matrix_path, args.monitor_mm, args.monitor_pixels, model, args.visualize_preprocessing, args.visualize_laser_pointer, args.visualize_3d, screen_point_calibrater, args.method)

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cv2.destroyAllWindows()
