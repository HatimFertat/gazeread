#!/usr/bin/env python3
"""
Test script for the CNN wrapper model calibration.
This demonstrates how the wrapper model adjusts the raw CNN output.
"""

import os
import sys
import cv2
import numpy as np
import logging
import time
import argparse
import matplotlib.pyplot as plt

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from eyetrax.gaze import CNNGazeEstimator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

logger = logging.getLogger(__name__)

def plot_adjustment(raw_points, adjusted_points, targets):
    """Plot raw predictions vs adjusted predictions vs targets."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Convert lists to numpy arrays
    raw_points = np.array(raw_points)
    adjusted_points = np.array(adjusted_points)
    targets = np.array(targets)
    
    # Plot the points
    ax.scatter(raw_points[:, 0], raw_points[:, 1], color='red', label='Raw CNN', alpha=0.7)
    ax.scatter(adjusted_points[:, 0], adjusted_points[:, 1], color='green', label='Adjusted', alpha=0.7)
    ax.scatter(targets[:, 0], targets[:, 1], color='blue', label='Target', alpha=0.7)
    
    # Connect each set of points with lines
    for i in range(len(raw_points)):
        ax.plot([raw_points[i, 0], adjusted_points[i, 0]], 
                [raw_points[i, 1], adjusted_points[i, 1]], 
                'k-', alpha=0.3)
        ax.plot([adjusted_points[i, 0], targets[i, 0]], 
                [adjusted_points[i, 1], targets[i, 1]], 
                'g--', alpha=0.3)
    
    ax.set_title("CNN Output Adjustment")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.legend()
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='Test CNN wrapper model calibration')
    parser.add_argument('--model-path', type=str, default='cnn/p00.ckpt', 
                        help='Path to CNN model')
    parser.add_argument('--calibration-matrix', type=str, default='cnn/calibration_matrix.yaml',
                       help='Path to camera calibration matrix')
    parser.add_argument('--wrapper-model', type=str, default='ridge',
                       choices=['ridge', 'svr', 'tiny_mlp', 'elastic_net'],
                       help='Wrapper model to use')
    parser.add_argument('--save-plot', type=str, help='Save adjustment plot to file')
    args = parser.parse_args()
    
    logger.info("Initializing CNN gaze estimator...")
    estimator = CNNGazeEstimator(
        model_path=args.model_path,
        calibration_matrix_path=args.calibration_matrix,
        use_direct_camera=True,
        wrapper_model_name=args.wrapper_model
    )
    
    # Grid points for calibration (simplified for test)
    def generate_grid_points(width, height, rows=3, cols=3):
        points = []
        for r in range(rows):
            for c in range(cols):
                x = int(width * (c + 0.5) / cols)
                y = int(height * (r + 0.5) / rows)
                points.append((x, y))
        return points
    
    # Capture camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Failed to open camera")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Generate calibration points
    grid_points = generate_grid_points(width, height)
    
    # Data storage
    raw_predictions = []
    adjusted_predictions = []
    
    # Calibration
    logger.info("Starting calibration process...")
    logger.info("Look at each point as it appears")
    
    # First collect raw predictions
    for i, point in enumerate(grid_points):
        # Display point
        img = np.zeros((height, width, 3), np.uint8)
        cv2.circle(img, point, 20, (0, 255, 0), -1)
        cv2.putText(img, f"Point {i+1}/{len(grid_points)}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Calibration', img)
        
        # Wait for gaze to stabilize
        time.sleep(1.0)
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Failed to capture frame for point {i+1}")
            continue
        
        # Get raw prediction
        features, _ = estimator.extract_features(frame)
        if features is None:
            logger.warning(f"No features detected for point {i+1}")
            continue
            
        raw_pred = estimator.get_raw_prediction(features)
        logger.info(f"Point {i+1}: Target={point}, Raw={raw_pred}")
        
        # Store for training
        estimator.wrapper_model.collect_training_sample(raw_pred, point)
        raw_predictions.append(raw_pred)
        
        # Wait for key press to continue
        key = cv2.waitKey(500) & 0xFF
        if key == 27:  # ESC
            break
    
    # Train the model
    logger.info("Training wrapper model...")
    if estimator.wrapper_model.train_from_collected():
        estimator.is_calibrated = True
        logger.info("Wrapper model trained successfully")
    else:
        logger.error("Failed to train wrapper model")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    # Now test the adjusted predictions
    logger.info("Testing adjusted predictions...")
    for i, point in enumerate(grid_points):
        # Display point
        img = np.zeros((height, width, 3), np.uint8)
        cv2.circle(img, point, 20, (0, 0, 255), -1)
        cv2.putText(img, f"Test {i+1}/{len(grid_points)}", (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Testing', img)
        
        # Wait for gaze to stabilize
        time.sleep(1.0)
        
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Failed to capture frame for test {i+1}")
            continue
        
        # Get adjusted prediction
        features, _ = estimator.extract_features(frame)
        if features is None:
            logger.warning(f"No features detected for test {i+1}")
            continue
            
        # Get raw prediction first (for comparison)
        raw_pred = estimator.get_raw_prediction(features)
        
        # Now get adjusted prediction
        adjusted_pred = estimator.predict([features])[0]
        
        logger.info(f"Test {i+1}: Target={point}, Raw={raw_pred}, Adjusted={adjusted_pred}")
        adjusted_predictions.append(adjusted_pred)
        
        # Wait for key press to continue
        key = cv2.waitKey(500) & 0xFF
        if key == 27:  # ESC
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    # Plot results
    if len(raw_predictions) == len(adjusted_predictions) == len(grid_points):
        fig = plot_adjustment(raw_predictions, adjusted_predictions, grid_points)
        
        if args.save_plot:
            fig.savefig(args.save_plot)
            logger.info(f"Plot saved to {args.save_plot}")
        else:
            plt.show()

if __name__ == "__main__":
    main() 