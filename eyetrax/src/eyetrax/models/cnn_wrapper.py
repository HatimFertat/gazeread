from __future__ import annotations

import numpy as np
import torch
import logging
from typing import Optional, Dict, Any, List

from . import register_model, create_model
from .base import BaseModel


class CNNWrapperModel(BaseModel):
    """
    Wrapper model that uses CNN gaze predictions as input features for 
    traditional models (Ridge, SVR, etc.)
    
    This model doesn't modify the CNN itself, but uses another model
    to refine/adjust the raw CNN output.
    """
    
    def __init__(self, 
                 base_model: str = "ridge", 
                 base_model_kwargs: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialize the CNN wrapper model.
        
        Args:
            base_model: The name of the base model to use (e.g., "ridge", "svr")
            base_model_kwargs: Keyword arguments to pass to the base model
        """
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing CNN wrapper with base model: {base_model}")
        
        # Initialize the base model that will adjust CNN output
        self.base_model_name = base_model
        self.base_model_kwargs = base_model_kwargs or {}
        self._init_native()
        
        # These will store raw CNN outputs and target screen coordinates during training
        self.cnn_outputs: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []
        
        # Flag to track if dimensions changed
        self.input_dim = None
        
        # Override the scaler with identity scaler since we're working with screen coordinates
        self.use_scaling = False
        
    def _init_native(self, **kwargs):
        """Initialize the base model."""
        self.model = create_model(self.base_model_name, **self.base_model_kwargs)
        self.logger.info(f"Created base model: {self.base_model_name}")
        
    def _scale_features(self, X: np.ndarray) -> np.ndarray:
        """Override to bypass scaling since inputs are already screen coordinates."""
        return X
        
    def _unscale_predictions(self, y: np.ndarray) -> np.ndarray:
        """Override to bypass unscaling since outputs should be screen coordinates."""
        return y
        
    def _native_train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the base model using CNN outputs as features.
        For this wrapper, X should be CNN outputs and y should be the target coordinates.
        """
        # Store the input dimension for future reference
        self.input_dim = X.shape[1]
        self.logger.info(f"Training base model with {len(X)} samples, input dim: {self.input_dim}")
        
        # Train the model with raw coordinates
        self.model.train(X, y)
    
    def _native_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the base model, with CNN outputs as input features.
        """
        # Check if dimensions match what we trained with
        if self.input_dim is not None and X.shape[1] != self.input_dim:
            self.logger.warning(f"Input dimension mismatch: got {X.shape[1]}, expected {self.input_dim}")
            # Adapt input to match expected dimensions
            if X.shape[1] == 2 and self.input_dim == 3:
                # Add a zero column for compatibility with older models
                X = np.hstack([X, np.zeros((X.shape[0], 1))])
            elif X.shape[1] == 3 and self.input_dim == 2:
                # Remove the last column
                X = X[:, :2]
        
        return self.model.predict(X)
    
    def collect_training_sample(self, cnn_output: np.ndarray, target: np.ndarray):
        """
        Collect a single training sample during calibration.
        
        Args:
            cnn_output: Raw CNN model output (x, y coordinates)
            target: Target screen coordinates (ground truth)
        """
        self.cnn_outputs.append(cnn_output)
        self.targets.append(target)
        
    def train_from_collected(self):
        """Train the model from collected samples."""
        if not self.cnn_outputs or not self.targets:
            self.logger.warning("No training data collected")
            return False
            
        X = np.array(self.cnn_outputs)
        y = np.array(self.targets)
        
        self.logger.info(f"Training model with {len(X)} collected samples, input shape: {X.shape}")
        self.train(X, y)
        
        # Clear collected data after training
        self.cnn_outputs = []
        self.targets = []
        
        return True
    
    def predict_adjusted(self, cnn_output: np.ndarray) -> np.ndarray:
        """
        Make a prediction with the adjustment model.
        
        Args:
            cnn_output: Raw CNN model output
            
        Returns:
            Adjusted screen coordinates
        """
        return self.predict(np.array([cnn_output]))[0]


# Register this model
register_model("cnn_wrapper", CNNWrapperModel) 