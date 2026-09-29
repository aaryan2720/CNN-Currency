"""
Loss Functions for From-Scratch CNN
===================================
Pure NumPy implementations of Cross-Entropy and combined SoftmaxCrossEntropy loss functions.
"""

from typing import Union, Tuple
import numpy as np


class CrossEntropyLoss:
    """
    Multiclass Cross-Entropy Loss (Operating on Probability Distributions).
    
    Mathematical Formulation:
        Forward:
            L = - (1 / N) * sum_{n=1}^N log(P[n, y[n]] + eps)
            
        Backward:
            dL/dP[n, c] = - (1 / N) * (I(y[n] == c) / (P[n, c] + eps))
            
    Note:
        eps is added for numerical stability to prevent log(0) -> NaN or -inf.
    """
    def __init__(self, eps: float = 1e-15):
        self.eps = eps
        self.cached_P = None
        self.cached_y = None

    def forward(self, y_pred_prob: np.ndarray, y_true: np.ndarray) -> float:
        """
        Forward pass for Cross-Entropy loss from probabilities.
        
        Args:
            y_pred_prob: Predicted probabilities of shape (N, num_classes).
            y_true: Integer class labels of shape (N,) with values in [0, num_classes-1],
                    or one-hot encoded matrix of shape (N, num_classes).
                    
        Returns:
            Scalar cross-entropy loss value.
        """
        self.cached_P = y_pred_prob
        self.cached_y = y_true
        
        N = y_pred_prob.shape[0]
        
        # Clip probabilities to prevent log(0) and log(1) edge cases
        probs_clipped = np.clip(y_pred_prob, self.eps, 1.0 - self.eps)
        
        if y_true.ndim == 1:
            # Integer labels
            correct_log_probs = np.log(probs_clipped[np.arange(N), y_true])
        else:
            # One-hot encoded matrix
            correct_log_probs = np.sum(y_true * np.log(probs_clipped), axis=-1)
            
        loss = -float(np.mean(correct_log_probs))
        return loss

    def backward(self) -> np.ndarray:
        """
        Backward pass for Cross-Entropy loss with respect to probabilities P.
        
        Returns:
            dL/dP of shape (N, num_classes).
        """
        if self.cached_P is None or self.cached_y is None:
            raise RuntimeError("Cannot run backward() before calling forward().")
            
        N, num_classes = self.cached_P.shape
        probs_clipped = np.clip(self.cached_P, self.eps, 1.0 - self.eps)
        
        if self.cached_y.ndim == 1:
            Y_one_hot = np.zeros((N, num_classes), dtype=self.cached_P.dtype)
            Y_one_hot[np.arange(N), self.cached_y] = 1.0
        else:
            Y_one_hot = self.cached_y
            
        # dL/dP = - (1 / N) * (Y_one_hot / P)
        return - (Y_one_hot / (probs_clipped * N))

    def __call__(self, y_pred_prob: np.ndarray, y_true: np.ndarray) -> float:
        return self.forward(y_pred_prob, y_true)


class SoftmaxCrossEntropyLoss:
    """
    Combined Softmax + Multiclass Cross-Entropy Loss.
    
    Computes cross-entropy directly from unnormalized logits with log-sum-exp numerical
    stabilization and exact analytical gradient cancellation:
    
    Mathematical Formulation:
        Forward:
            log_sum_exp = max(Z) + log(sum(exp(Z - max(Z))))
            log_p[n, c] = Z[n, c] - log_sum_exp[n]
            L = - (1 / N) * sum_{n=1}^N log_p[n, y[n]]
            
        Backward:
            dL/dZ = (P - Y_one_hot) / N
            
    Why the gradient simplifies:
        dL/dZ_i = sum_k (dL/dP_k * dP_k/dZ_i)
        Combining dL/dP_k = -y_k / P_k with dP_k/dZ_i = P_k(delta_{ki} - P_i)
        yields:
            dL/dZ_i = sum_k (-y_k / P_k) * P_k(delta_{ki} - P_i)
                    = -y_i + P_i * sum_k y_k
                    = P_i - y_i   (since sum_k y_k = 1 for one-hot labels)
        Dividing by batch size N gives (P - Y) / N.
    """
    def __init__(self):
        self.cached_probs = None
        self.cached_y = None

    def forward(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        """
        Forward pass from unnormalized logits.
        
        Args:
            logits: Logits tensor of shape (N, num_classes).
            y_true: Integer class labels of shape (N,) or one-hot matrix (N, num_classes).
            
        Returns:
            Scalar loss value.
        """
        N = logits.shape[0]
        self.cached_y = y_true
        
        # 1. Numerically stable Log-Sum-Exp trick
        max_logits = np.max(logits, axis=-1, keepdims=True)
        shifted_logits = logits - max_logits
        exp_logits = np.exp(shifted_logits)
        sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Softmax probabilities
        probs = exp_logits / sum_exp
        self.cached_probs = probs
        
        # Log probabilities: log(p_c) = (z_c - max(z)) - log(sum(exp(z - max(z))))
        log_probs = shifted_logits - np.log(sum_exp)
        
        if y_true.ndim == 1:
            correct_log_probs = log_probs[np.arange(N), y_true]
        else:
            correct_log_probs = np.sum(y_true * log_probs, axis=-1)
            
        loss = -float(np.mean(correct_log_probs))
        return loss

    def backward(self) -> np.ndarray:
        """
        Backward pass directly computing dL/dLogits.
        
        Returns:
            dL/dZ: Gradient with respect to logits of shape (N, num_classes).
        """
        if self.cached_probs is None or self.cached_y is None:
            raise RuntimeError("Cannot run backward() before calling forward().")
            
        N, num_classes = self.cached_probs.shape
        P = self.cached_probs.copy()
        
        if self.cached_y.ndim == 1:
            # Subtract 1 from the correct class probability
            P[np.arange(N), self.cached_y] -= 1.0
        else:
            P -= self.cached_y
            
        # Divide by batch size N
        dZ = P / N
        return dZ

    def __call__(self, logits: np.ndarray, y_true: np.ndarray) -> float:
        return self.forward(logits, y_true)
