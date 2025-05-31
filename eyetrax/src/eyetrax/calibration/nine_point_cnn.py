import cv2
import numpy as np
import torch
import logging
import time

from eyetrax.calibration.common import (
    compute_grid_points,
    wait_for_face_and_countdown,
)
from eyetrax.utils.screen import get_screen_size

def _pulse_and_capture(
    gaze_estimator,
    cap,
    pts,
    sw: int,
    sh: int,
    pulse_d: float = 1.0,
    cd_d: float = 1.0,
):
    """
    Shared pulse-and-capture loop for each calibration point
    """
    feats, targs = [], []

    for x, y in pts:
        # pulse
        ps = time.time()
        final_radius = 20
        while True:
            e = time.time() - ps
            if e > pulse_d:
                break
            ok, frame = cap.read()
            if not ok:
                continue
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            radius = 15 + int(15 * abs(np.sin(2 * np.pi * e)))
            final_radius = radius
            cv2.circle(canvas, (x, y), radius, (0, 255, 0), -1)
            cv2.imshow("Calibration", canvas)
            if cv2.waitKey(1) == 27:
                return None
        # capture
        cs = time.time()
        while True:
            e = time.time() - cs
            if e > cd_d:
                break
            ok, frame = cap.read()
            if not ok:
                continue
            canvas = np.zeros((sh, sw, 3), dtype=np.uint8)
            cv2.circle(canvas, (x, y), final_radius, (0, 255, 0), -1)
            t = e / cd_d
            ease = t * t * (3 - 2 * t)
            ang = 360 * (1 - ease)
            cv2.ellipse(canvas, (x, y), (40, 40), 0, -90, -90 + ang, (255, 255, 255), 4)
            cv2.imshow("Calibration", canvas)
            if cv2.waitKey(1) == 27:
                return None
            ft, blink = gaze_estimator.extract_features(frame)
            point_on_screen = gaze_estimator.predict([ft])[0]

            if ft is not None and not blink:
                feats.append(point_on_screen)
                targs.append([x, y])

    return feats, targs


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