"""
Base Interface for Baseline Methods

Defines the common interface that all baseline methods must implement
for consistent evaluation and comparison.
"""

import torch
from torch.utils.data import DataLoader
from typing import Dict, Any


class BaselineMethodInterface:
    """Interface for baseline methods to ensure consistent evaluation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
        """Train the baseline method."""
        raise NotImplementedError("Baseline methods must implement train()")
    
    def evaluate(self, test_loader: DataLoader, device: torch.device) -> Dict[str, float]:
        """Evaluate the baseline method."""
        raise NotImplementedError("Baseline methods must implement evaluate()")
    
    def predict(self, batch: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        """Make predictions on a batch."""
        raise NotImplementedError("Baseline methods must implement predict()")