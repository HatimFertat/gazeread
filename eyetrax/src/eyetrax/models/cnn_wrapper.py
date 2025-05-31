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
        
    def _init_native(self, **kwargs):
        """Initialize the base model."""
        self.model = create_model(self.base_model_name, **self.base_model_kwargs)
        self.logger.info(f"Created base model: {self.base_model_name}")
        
    def _native_train(self, X: np.ndarray, y: np.ndarray):
        """
        Train the base model using CNN outputs as features.
        For this wrapper, X should be CNN outputs and y should be the target coordinates.
        """
        self.logger.info(f"Training base model with {len(X)} samples")
        self.model.train(X, y)
    
    def _native_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the base model, with CNN outputs as input features.
        """
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
        
        self.logger.info(f"Training model with {len(X)} collected samples")
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