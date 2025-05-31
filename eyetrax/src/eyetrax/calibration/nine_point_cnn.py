import cv2
import numpy as np
import torch
import logging

from eyetrax.calibration.common import (
    _pulse_and_capture,
    compute_grid_points,
    wait_for_face_and_countdown,
)
from eyetrax.utils.screen import get_screen_size


def run_9_point_calibration_cnn(gaze_estimator, camera_index: int = 0):
    """
    Nine-point calibration for CNN model that trains a wrapper model
    to adjust the CNN output.
    
    Args:
        gaze_estimator: The gaze estimator with CNN model
        camera_index: Camera to use for capturing
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting CNN 9-point calibration with wrapper model")
    
    sw, sh = get_screen_size()
    logger.info(f"Screen size: {sw}x{sh}")

    cap = cv2.VideoCapture(camera_index)
    if not wait_for_face_and_countdown(cap, gaze_estimator, sw, sh, 2):
        logger.warning("Face not detected or calibration canceled")
        cap.release()
        cv2.destroyAllWindows()
        return False

    order = [
        (1, 1),
        (0, 0),
        (2, 0),
        (0, 2),
        (2, 2),
        (1, 0),
        (0, 1),
        (2, 1),
        (1, 2),
    ]
    pts = compute_grid_points(order, sw, sh)
    logger.info(f"Generated {len(pts)} calibration points")

    res = _pulse_and_capture(gaze_estimator, cap, pts, sw, sh)
    cap.release()
    cv2.destroyAllWindows()
    if res is None:
        logger.warning("Calibration canceled during capture")
        return False
    
    features, targets = res
    logger.info(f"Collected {len(features)} feature samples and {len(targets)} targets")
    
    if not features:
        logger.warning("No features collected during calibration")
        return False
    
    # Use the wrapper model calibration instead of directly modifying subject bias
    success = gaze_estimator.calibrate_wrapper_model(features, targets)
    
    if success:
        logger.info("Wrapper model calibration successful")
        print("Calibration complete. Wrapper model trained to adjust CNN output.")
        return True
    else:
        logger.warning("Wrapper model calibration failed")
        print("Calibration failed. Could not train wrapper model.")
        return False