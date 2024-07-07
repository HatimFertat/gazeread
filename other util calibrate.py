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

    with open('output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['image_path', 'True Point on Screen', 'Point on Screen', 'GT Gaze Vector', 'Gaze Vector'])
        
        monitor_mm, monitor_pixels = get_monitor_dimensions()
        # plane = get_plane("../gaze-data-collection/data/p00/data.csv",camera_matrix,dist_coefficients,face_mesh)
        plane = plane_equation(np.eye(3), np.asarray([[0], [0], [0]]))
        plane_w = plane[:3]
        plane_b = plane[3]
        
        for index, row in calibration_data.iterrows():
            image_path = base_path + row['file_name']
            true_point_on_screen = eval(row['point_on_screen'])  # Convert string to tuple

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
                eye_center = (left_eye_center + right_eye_center) / 2.0  # average to get the overall eye center
        
                # true_gaze_vector = get_true_gaze_vector(true_point_on_screen,eye_center, monitor_mm, monitor_pixels, plane_w, plane_b)

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


                result = ray_plane_intersection(face_center.reshape(3), gaze_vector, plane_w, plane_b)
                point_on_screen = get_point_on_screen(monitor_mm, monitor_pixels, result)
                
                writer.writerow([image_path, true_point_on_screen, point_on_screen, gaze_vector.tolist()])


    # Ensure the bias tensor is on the same device as the model
    return "success"