import sys
import logging
import argparse
import os
from PyQt6.QtWidgets import QApplication
from .gui.main_window import MainWindow

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ReadGaze - Eye tracking for reading')
    parser.add_argument('--use-cnn', action='store_true', help='Use the CNN model for gaze estimation')
    parser.add_argument('--model-path', type=str, help='Path to the CNN model checkpoint')
    parser.add_argument('--calibration-path', type=str, help='Path to the camera calibration file')
    parser.add_argument('--wrapper-model-path', type=str, help='Path to the wrapper model for CNN adjustment')
    parser.add_argument('--wrapper-model-name', type=str, default='ridge', 
                        choices=['ridge', 'svr', 'elastic_net', 'tiny_mlp'], 
                        help='Name of the base model to use for CNN adjustment')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Enable debug logging for specific modules
    if args.debug:
        logging.getLogger('readgaze.eye_tracking').setLevel(logging.DEBUG)
    
    # Set default paths for CNN model if using CNN but paths not specified
    if args.use_cnn:
        # Get the project root directory
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cnn_dir = os.path.join(project_root, "cnn")
        
        if args.model_path is None:
            args.model_path = os.path.join(cnn_dir, "p00.ckpt")
            logging.info(f"Using default CNN model path: {args.model_path}")
            
        if args.calibration_path is None:
            args.calibration_path = os.path.join(cnn_dir, "calibration_matrix.yaml")
            logging.info(f"Using default calibration path: {args.calibration_path}")
        
        # Default wrapper model path is in the weights directory
        if args.wrapper_model_path is None:
            weights_dir = os.path.join(project_root, "weights")
            os.makedirs(weights_dir, exist_ok=True)
            default_wrapper_path = os.path.join(weights_dir, "cnn_wrapper_model.pkl")
            
            # Only set as default if file exists
            if os.path.exists(default_wrapper_path):
                args.wrapper_model_path = default_wrapper_path
                logging.info(f"Using existing wrapper model: {args.wrapper_model_path}")
            
        # Verify files exist
        if not os.path.exists(args.model_path):
            logging.error(f"CNN model file not found: {args.model_path}")
            sys.exit(1)
            
        if not os.path.exists(args.calibration_path):
            logging.error(f"Calibration file not found: {args.calibration_path}")
            sys.exit(1)
    
    app = QApplication(sys.argv)
    window = MainWindow(use_cnn=args.use_cnn, 
                        model_path=args.model_path, 
                        calibration_path=args.calibration_path,
                        wrapper_model_path=args.wrapper_model_path,
                        wrapper_model_name=args.wrapper_model_name)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 