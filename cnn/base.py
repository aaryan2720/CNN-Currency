"""
Base Layer Interface for From-Scratch CNN
==========================================
Provides the abstract base class for all neural network layers.
"""

from abc import ABC, abstractmethod
import numpy as np


class Layer(ABC):
    """
    Abstract Base Class for all network layers.
    
    Every layer must implement:
      - forward(x): Computes layer output and caches intermediate states.
      - backward(grad_output): Computes and returns input gradients (dX),
                               and calculates parameter gradients (dW, db) if applicable.
    """
    def __init__(self):
        self.trainable = False
        self.params = {}
        self.grads = {}

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor.
        """
        raise NotImplementedError("Subclasses must implement forward()")

    @abstractmethod
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass.
        
        Args:
            grad_output: Upstream gradient dL/dOutput flowing back from subsequent layer.
            
        Returns:
            Gradient with respect to layer input dL/dX.
        """
        raise NotImplementedError("Subclasses must implement backward()")

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)
