import cv2
import numpy as np

from eyetrax.calibration.common import (
    _pulse_and_capture,
    compute_grid_points,
    wait_for_face_and_countdown,
)
from eyetrax.utils.screen import get_screen_size


def run_24_point_calibration(gaze_estimator, camera_index: int = 0):
    """
    24-point calibration using an 8x3 grid (8 rows, 3 columns).
    This provides more vertical calibration points which is useful for reading tasks.
    The grid is centered and only spans the middle 50% of the screen width since PDFs
    are typically centered.
    """
    sw, sh = get_screen_size()

    cap = cv2.VideoCapture(camera_index)
    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2):
        cap.release()
        cv2.destroyAllWindows()
        return

    # Define calibration points order
    # We use a snake pattern to minimize eye movement distance between points
    # Starting from top-left, going right, then down and left, etc.
    order = []
    for row in range(8):
        if row % 2 == 0:  # Even rows: left to right
            for col in range(3):
                order.append((row, col))
        else:  # Odd rows: right to left
            for col in range(2, -1, -1):
                order.append((row, col))

    # Custom grid points calculation for centered PDF reading
    quarter_w = sw // 4  # 25% of screen width
    margin_h = int(sh * 0.10)  # 10% vertical margin
    usable_h = sh - 2 * margin_h
    
    # Calculate step sizes
    step_x = quarter_w  # Points will be at 25%, 50%, and 75% of screen width
    step_y = usable_h / 7  # Divide remaining height into 7 gaps for 8 points
    
    # Generate points
    pts = []
    for row, col in order:
        x = quarter_w * (1 + col)  # This puts points at 25%, 50%, 75% of screen width
        y = margin_h + row * step_y
        pts.append((int(x), int(y)))

    res = _pulse_and_capture(gaze_estimator, cap, pts, sw, sh)
    cap.release()
    cv2.destroyAllWindows()
    if res is None:
        return
    feats, targs = res
    if feats:
        gaze_estimator.train(np.array(feats), np.array(targs)) 