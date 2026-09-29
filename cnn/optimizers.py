"""
Optimizers for From-Scratch CNN
===============================
Pure NumPy implementations of gradient-based optimization algorithms.
"""

from typing import List, Dict
import numpy as np


class Optimizer:
    """Abstract base class for optimizers."""
    def step(self, layers: List):
        raise NotImplementedError

    def zero_grad(self, layers: List):
        for layer in layers:
            if hasattr(layer, "grads") and layer.grads:
                for k in layer.grads:
                    layer.grads[k].fill(0.0)


class SGD(Optimizer):
    """
    Vanilla Stochastic Gradient Descent (SGD) Optimizer.
    
    Update Rule:
        theta = theta - lr * grad_theta
        
    Args:
        lr: Learning rate (step size).
    """
    def __init__(self, lr: float = 0.001):
        self.lr = float(lr)

    def step(self, layers: List):
        """
        Applies a gradient descent update step across all trainable layers.
        
        Args:
            layers: List of Layer instances.
        """
        for layer in layers:
            if getattr(layer, "trainable", False) and hasattr(layer, "params") and hasattr(layer, "grads"):
                for name in layer.params:
                    if name in layer.grads:
                        # In-place parameter update: param -= lr * grad
                        layer.params[name] -= self.lr * layer.grads[name]

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr})"
