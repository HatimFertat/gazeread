# ReadGaze - Eye Tracking for Reading

An eye tracking application for reading assistance, with support for both traditional calibration-based gaze estimation and CNN-based gaze estimation.

## Features

- Track user's gaze in real-time using either:
  - Traditional calibration-based approach
  - CNN-based model (no calibration required)
- Display reading content with visual aids
- Apply various filters to gaze data for smoother tracking
- Debug mode for troubleshooting

## Requirements

- Python 3.8+
- Webcam
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

To run the application with the traditional calibration-based gaze tracking:

```
python -m readgaze
```

To run with the CNN-based gaze tracking (no calibration needed):

```
python -m readgaze --use-cnn
```

### Command Line Arguments

- `--use-cnn`: Use the CNN model for gaze estimation (no calibration required)
- `--model-path PATH`: Path to a custom CNN model checkpoint (default: cnn/p00.ckpt)
- `--calibration-path PATH`: Path to a custom camera calibration file (default: cnn/calibration_matrix.yaml)
- `--debug`: Enable debug logging for troubleshooting

### Debug Mode

To run with debug logging enabled:

```
python -m readgaze --debug
```

Or with CNN model and debug logging:

```
python -m readgaze --use-cnn --debug
```

## CNN Gaze Estimation

ReadGaze now supports a CNN-based gaze estimation approach that doesn't require calibration. This alternative approach uses a deep learning model to predict gaze direction based on facial landmarks.

### Features
- No calibration required - works immediately
- Uses MediaPipe Face Mesh for reliable facial landmark detection
- Predicts gaze direction using a pre-trained CNN model

### Usage
To use the CNN gaze estimator:

```bash
# Run with CNN gaze estimation
python -m readgaze --use-cnn

# Specify a custom model path
python -m readgaze --use-cnn --model-path path/to/model.ckpt

# Enable debug logging
python -m readgaze --use-cnn --debug
```

### Technical Details
- The CNN implementation uses MediaPipe Face Mesh for facial landmark detection
- Pose estimation is performed using OpenCV's solvePnP with fallback methods
- For best results, start by looking directly at the camera with your head facing forward

## Testing

To test the CNN gaze estimator directly:

```
python test_cnn_gaze.py
```

This will display a simple window showing:
- Live webcam feed
- Detected gaze point
- FPS and processing times
- Visual indicators of gaze movement

## Troubleshooting

If you encounter issues with the CNN model:

1. Ensure the webcam is working correctly
2. Try running with the `--debug` flag for more detailed logging
3. Check lighting conditions - good, even lighting improves accuracy
4. Position yourself within 50-80cm from the camera
5. Ensure your face is fully visible to the camera

## License

[MIT License](LICENSE)
