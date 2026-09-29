"""
Activation Functions for From-Scratch CNN
==========================================
Pure NumPy implementations of non-linear activation layers: ReLU and Softmax.
"""

import numpy as np
from .base import Layer


class ReLU(Layer):
    """
    Rectified Linear Unit (ReLU) Activation Layer.
    
    Mathematical Definition:
        Forward:
            y = max(0, x)
        Backward:
            dL/dx = (dL/dy) * (x > 0)
            
    Note: At x = 0, the subgradient is defined as 0.
    """
    def __init__(self):
        super().__init__()
        self.cached_x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for ReLU.
        
        Args:
            x: Input NumPy tensor of arbitrary shape.
            
        Returns:
            Output tensor of the same shape with negative values zeroed out.
        """
        self.cached_x = x
        return np.maximum(0.0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for ReLU.
        
        Args:
            grad_output: Upstream gradient dL/dy of the same shape as forward output.
            
        Returns:
            dL/dx: Gradient with respect to layer input.
        """
        if self.cached_x is None:
            raise RuntimeError("Cannot run backward() before calling forward().")
            
        # Derivative is 1 for x > 0, and 0 for x <= 0
        relu_grad = (self.cached_x > 0).astype(grad_output.dtype)
        return grad_output * relu_grad

    def __repr__(self) -> str:
        return "ReLU()"


class Softmax(Layer):
    """
    Numerically Stable Softmax Activation Layer.
    
    Transforms unnormalized logits into normalized probability distributions.
    
    Mathematical Formulation:
        Forward:
            z_shifted = z - max(z, axis=-1, keepdims=True)
            p_i = exp(z_shifted_i) / sum_j exp(z_shifted_j)
            
        Backward (Standalone Jacobian-Vector Product):
            dp_i / dz_j = p_i * (delta_{ij} - p_j)
            dL / dz_i = p_i * (dL/dp_i - sum_k (dL/dp_k * p_k))
    """
    def __init__(self):
        super().__init__()
        self.cached_out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for Softmax with numerical overflow protection.
        
        Args:
            x: Logit tensor of shape (N, num_classes) or arbitrary (..., C).
            
        Returns:
            Probability tensor of the same shape where rows sum to 1.0.
        """
        # Subtract max per sample to prevent exp(large_positive) overflow
        x_max = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - x_max)
        probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        self.cached_out = probs
        return probs

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for standalone Softmax.
        
        Args:
            grad_output: Upstream gradient dL/dP of the same shape as forward output.
            
        Returns:
            dL/dZ: Gradient with respect to unnormalized logits Z.
        """
        if self.cached_out is None:
            raise RuntimeError("Cannot run backward() before calling forward().")
            
        P = self.cached_out
        # Compute dot product sum_k (dL/dp_k * p_k) per sample
        sum_p_grad = np.sum(grad_output * P, axis=-1, keepdims=True)
        return P * (grad_output - sum_p_grad)

    def __repr__(self) -> str:
        return "Softmax()"
