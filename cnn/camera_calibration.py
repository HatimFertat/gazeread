import cv2
import glob
from datetime import datetime
import numpy as np
import yaml


def capture_images(camera_id=0, save_dir='images', image_prefix='img', max_images=20):
    """
    Capture images from the webcam and save them to the specified directory.

    :param camera_id: ID of the camera (default is 0)
    :param save_dir: Directory to save images
    :param image_prefix: Prefix for image filenames
    :param max_images: Maximum number of images to capture
    :return: None
    """
    cap = cv2.VideoCapture(camera_id)
    num = 0

    while cap.isOpened() and num < max_images:
        success, img = cap.read()
        if not success:
            break

        k = cv2.waitKey(1)
        if k == 27:  # Press 'ESC' to exit
            break
        elif k == ord('s'):  # Press 's' to save the image
            filename = f'{save_dir}/{image_prefix}{num}.png'
            cv2.imwrite(filename, img)
            print(f"Image saved: {filename}")
            num += 1

        cv2.imshow('Capture Image', img)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {num} images.")


def calibration(image_path, every_nth: int = 1, debug: bool = False, chessboard_grid_size=(7,7)):
    """
    Perform camera calibration on the previously collected images.
    Creates `calibration_matrix.yaml` with the camera intrinsic matrix and the distortion coefficients.

    :param image_path: Path to all png images
    :param every_nth: Only use every nth image
    :param debug: Preview the matched chess patterns
    :param chessboard_grid_size: Size of chess pattern
    :return: None
    """

    x, y = chessboard_grid_size

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
    objp = np.zeros((y * x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:x, 0:y].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    images = glob.glob(f'{image_path}/*.png')[::every_nth]

    found = 0
    gray = None  # Initialize gray to None to ensure it is defined

    for fname in images:
        img = cv2.imread(fname)  # Capture frame-by-frame
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (x, y), None)

        # If found, add object points and image points (after refining them)
        if ret:
            objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            found += 1
            print(f"Chessboard corners found in {fname}")

            if debug:
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, chessboard_grid_size, corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(100)
        else:
            print(f"Chessboard corners not found in {fname}")

    print("Number of images used for calibration: ", found)

    # When everything done, release the capture
    if debug:
        cv2.destroyAllWindows()

    # Ensure there are valid images to calibrate
    if found == 0 or gray is None:
        print("No valid images found for calibration.")
        return

    # Calibration
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('rms', rms)

    # Transform the matrix and distortion coefficients to writable lists
    data = {
        'rms': np.asarray(rms).tolist(),
        'camera_matrix': np.asarray(mtx).tolist(),
        'dist_coeff': np.asarray(dist).tolist()
    }

    # Save it to a file
    with open("calibration_matrix.yaml", "w") as f:
        yaml.dump(data, f)

    print(data)


if __name__ == '__main__':
    # 1. Capture images manually
    capture_images(camera_id=0, save_dir='./cnn/images', image_prefix='img', max_images=20)
    # 2. Run calibration on the captured images
    calibration('./cnn/images', every_nth=1, debug=True)