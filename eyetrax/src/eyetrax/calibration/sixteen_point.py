import cv2
import numpy as np

from eyetrax.calibration.common import (
    _pulse_and_capture,
    compute_grid_points,
    wait_for_face_and_countdown,
)
from eyetrax.utils.screen import get_screen_size
import random

def run_16_point_calibration(gaze_estimator, camera_index: int = 0):
    """
    Standard sixteen-point calibration
    """
    sw, sh = get_screen_size()

    cap = cv2.VideoCapture(camera_index)
    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2):
        cap.release()
        cv2.destroyAllWindows()
        return

    order = [(row, col) for row in range(6) for col in range(3)]
    random.shuffle(order)
    pts = compute_grid_points(order, sw, sh)

    res = _pulse_and_capture(gaze_estimator, cap, pts, sw, sh, 1, 0.5)
    cap.release()
    cv2.destroyAllWindows()
    if res is None:
        return
    feats, targs = res
    if feats:
        gaze_estimator.train(np.array(feats), np.array(targs))
