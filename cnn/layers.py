"""
Neural Network Layers for From-Scratch CNN
===========================================
Pure NumPy implementation of Conv2D, MaxPool2D, Flatten, and Dense layers.
"""

from typing import Union, Tuple, Optional
import numpy as np
from .base import Layer


class Conv2D(Layer):
    """
    2D Convolutional Layer.
    
    Mathematical Formulation:
        Forward:
            Z[n, f, i, j] = sum_{c, u, v} (X_pad[n, c, i*S_h + u, j*S_w + v] * W[f, c, u, v]) + b[f]
            
        Backward:
            dL/db[f] = sum_{n, i, j} dZ[n, f, i, j]
            dL/dW[f, c, u, v] = sum_{n, i, j} dZ[n, f, i, j] * X_pad[n, c, i*S_h + u, j*S_w + v]
            dL/dX_pad[n, c, i*S_h + u, j*S_w + v] += sum_f dZ[n, f, i, j] * W[f, c, u, v]
            dL/dX = dL/dX_pad[:, :, P_h : P_h + H, P_w : P_w + W]
            
    Attributes:
        in_channels (int): Number of input feature channels.
        out_channels (int): Number of output filters/feature maps.
        kernel_size (Tuple[int, int]): (K_h, K_w) dimensions of the convolution kernel.
        stride (Tuple[int, int]): (S_h, S_w) convolution stride.
        padding (Tuple[int, int]): (P_h, P_w) symmetric zero-padding on height and width.
        W (np.ndarray): Filter weights of shape (out_channels, in_channels, K_h, K_w).
        b (np.ndarray): Biases of shape (out_channels,).
        dW (np.ndarray): Gradient of loss with respect to weights.
        db (np.ndarray): Gradient of loss with respect to biases.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 1,
        seed: Optional[int] = None
    ):
        super().__init__()
        self.trainable = True
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Parse tuples
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        
        kh, kw = self.kernel_size
        
        # He (Kaiming) Weight Initialization: std = sqrt(2 / fan_in)
        fan_in = in_channels * kh * kw
        std = np.sqrt(2.0 / fan_in)
        
        rng = np.random.default_rng(seed)
        self.W = rng.normal(loc=0.0, scale=std, size=(out_channels, in_channels, kh, kw)).astype(np.float64)
        self.b = np.zeros(out_channels, dtype=np.float64)
        
        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        self.params = {"W": self.W, "b": self.b}
        self.grads = {"W": self.dW, "b": self.db}
        
        # Caches for backward propagation
        self.cached_x = None
        self.cached_x_pad = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for 2D convolution.
        
        Args:
            x: Input tensor of shape (N, C_in, H, W).
            
        Returns:
            Output tensor of shape (N, C_out, H_out, W_out).
        """
        self.cached_x = x
        N, C, H, W = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        
        if C != self.in_channels:
            raise ValueError(f"Input channel mismatch: expected {self.in_channels}, got {C}")
            
        # 1. Apply zero-padding
        if ph > 0 or pw > 0:
            x_pad = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant", constant_values=0.0)
        else:
            x_pad = x
            
        self.cached_x_pad = x_pad
        
        # 2. Compute output spatial dimensions
        H_out = (H + 2 * ph - kh) // sh + 1
        W_out = (W + 2 * pw - kw) // sw + 1
        
        # 3. Vectorized spatial sliding window
        Z = np.zeros((N, self.out_channels, H_out, W_out), dtype=x.dtype)
        
        for i in range(H_out):
            h_start = i * sh
            h_end = h_start + kh
            for j in range(W_out):
                w_start = j * sw
                w_end = w_start + kw
                
                # Extract receptive field patch: shape (N, C_in, K_h, K_w)
                x_slice = x_pad[:, :, h_start:h_end, w_start:w_end]
                
                # Tensor contraction: (N, C_in, K_h, K_w) . (C_out, C_in, K_h, K_w) -> (N, C_out)
                Z[:, :, i, j] = np.tensordot(x_slice, self.W, axes=([1, 2, 3], [1, 2, 3])) + self.b
                
        return Z

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for 2D convolution.
        
        Args:
            grad_output: Upstream gradient dL/dZ of shape (N, C_out, H_out, W_out).
            
        Returns:
            dX: Gradient with respect to layer input dL/dX of shape (N, C_in, H, W).
        """
        if self.cached_x is None or self.cached_x_pad is None:
            raise RuntimeError("Cannot run backward() before calling forward().")
            
        N, C, H, W = self.cached_x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding
        _, _, H_out, W_out = grad_output.shape
        
        # 1. Bias gradient: sum upstream gradients over (batch, height, width)
        self.db = np.sum(grad_output, axis=(0, 2, 3))
        self.grads["b"] = self.db
        
        # 2. Initialize dW and dX_pad
        self.dW = np.zeros_like(self.W)
        dX_pad = np.zeros_like(self.cached_x_pad)
        
        # 3. Accumulate gradients over spatial positions
        for i in range(H_out):
            h_start = i * sh
            h_end = h_start + kh
            for j in range(W_out):
                w_start = j * sw
                w_end = w_start + kw
                
                x_slice = self.cached_x_pad[:, :, h_start:h_end, w_start:w_end] # (N, C_in, K_h, K_w)
                dZ_slice = grad_output[:, :, i, j]                               # (N, C_out)
                
                # dW accumulation: (N, C_out)^T . (N, C_in, K_h, K_w) -> (C_out, C_in, K_h, K_w)
                self.dW += np.tensordot(dZ_slice, x_slice, axes=([0], [0]))
                
                # dX_pad accumulation: (N, C_out) . (C_out, C_in, K_h, K_w) -> (N, C_in, K_h, K_w)
                dX_pad[:, :, h_start:h_end, w_start:w_end] += np.tensordot(dZ_slice, self.W, axes=([1], [0]))
                
        self.grads["W"] = self.dW
        
        # 4. Remove padding to yield gradient with respect to original input X
        if ph > 0 or pw > 0:
            dX = dX_pad[:, :, ph:ph + H, pw:pw + W]
        else:
            dX = dX_pad
            
        return dX

    def __repr__(self) -> str:
        return (
            f"Conv2D(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding})"
        )


class MaxPool2D(Layer):
    """
    2D Max-Pooling Layer.
    
    Mathematical Formulation:
        Forward:
            Y[n, c, i, j] = max_{u, v} X[n, c, i*S_h + u, j*S_w + v]
            
        Backward:
            dL/dX[n, c, i*S_h + u, j*S_w + v] = (dL/dY[n, c, i, j]) * I(X == max) / sum(I(X == max))
            
    Attributes:
        pool_size (Tuple[int, int]): (K_h, K_w) dimensions of the pooling window.
        stride (Tuple[int, int]): (S_h, S_w) stride of the pooling window.
    """
    def __init__(
        self,
        pool_size: Union[int, Tuple[int, int]] = 2,
        stride: Union[int, Tuple[int, int]] = 2
    ):
        super().__init__()
        self.pool_size = (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.cached_x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for 2D max pooling.
        
        Args:
            x: Input tensor of shape (N, C, H, W).
            
        Returns:
            Pooled tensor of shape (N, C, H_out, W_out).
        """
        self.cached_x = x
        N, C, H, W = x.shape
        kh, kw = self.pool_size
        sh, sw = self.stride
        
        H_out = (H - kh) // sh + 1
        W_out = (W - kw) // sw + 1
        
        Y = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
        
        for i in range(H_out):
            h_start = i * sh
            h_end = h_start + kh
            for j in range(W_out):
                w_start = j * sw
                w_end = w_start + kw
                
                x_slice = x[:, :, h_start:h_end, w_start:w_end]
                Y[:, :, i, j] = np.max(x_slice, axis=(2, 3))
                
        return Y

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for 2D max pooling.
        
        Routes upstream gradients strictly to the spatial argmax locations from forward pass.
        
        Args:
            grad_output: Upstream gradient dL/dY of shape (N, C, H_out, W_out).
            
        Returns:
            dX: Gradient with respect to layer input dL/dX of shape (N, C, H, W).
        """
        if self.cached_x is None:
            raise RuntimeError("Cannot run backward() before calling forward().")
            
        N, C, H, W = self.cached_x.shape
        kh, kw = self.pool_size
        sh, sw = self.stride
        _, _, H_out, W_out = grad_output.shape
        
        dX = np.zeros_like(self.cached_x)
        
        for i in range(H_out):
            h_start = i * sh
            h_end = h_start + kh
            for j in range(W_out):
                w_start = j * sw
                w_end = w_start + kw
                
                x_slice = self.cached_x[:, :, h_start:h_end, w_start:w_end]
                # Compute maximum per batch and channel in this window
                max_val = np.max(x_slice, axis=(2, 3), keepdims=True)
                
                # Binary mask for maximum positions
                mask = (x_slice == max_val).astype(grad_output.dtype)
                
                # Normalize mask in case of multiple identical tie values
                mask_sum = np.sum(mask, axis=(2, 3), keepdims=True)
                mask = mask / np.maximum(mask_sum, 1.0)
                
                # Upstream gradient for this window: shape (N, C, 1, 1)
                dY_slice = grad_output[:, :, i:i+1, j:j+1]
                
                dX[:, :, h_start:h_end, w_start:w_end] += mask * dY_slice
                
        return dX

    def __repr__(self) -> str:
        return f"MaxPool2D(pool_size={self.pool_size}, stride={self.stride})"


class Flatten(Layer):
    """
    Flatten Layer.
    
    Transforms multi-dimensional spatial feature maps into a 2D matrix
    while strictly preserving the batch dimension.
    
    Mathematical Formulation:
        Forward:
            Input shape:  (N, d_1, d_2, ..., d_k)
            Output shape: (N, D) where D = prod(d_1, ..., d_k)
            
        Backward:
            Reshapes incoming gradient dL/dOutput back to (N, d_1, d_2, ..., d_k).
            
    Learnable Parameters:
        None (0 parameters). Flatten is a pure structural tensor transformation.
    """
    def __init__(self):
        super().__init__()
        self.cached_shape = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for Flatten.
        
        Args:
            x: Input tensor of shape (N, C, H, W) or arbitrary (N, ...).
            
        Returns:
            2D tensor of shape (N, C*H*W).
        """
        self.cached_shape = x.shape
        N = x.shape[0]
        return x.reshape(N, -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for Flatten.
        
        Args:
            grad_output: Upstream gradient of shape (N, D).
            
        Returns:
            Reshaped gradient matching original input shape (N, C, H, W).
        """
        if self.cached_shape is None:
            raise RuntimeError("Cannot run backward() before calling forward().")
        return grad_output.reshape(self.cached_shape)

    def __repr__(self) -> str:
        return "Flatten()"


class Dense(Layer):
    """
    Fully Connected (Dense / Linear) Layer.
    
    Mathematical Formulation:
        Forward:
            Y = X @ W + b
            where:
                X in R^(N x D_in)
                W in R^(D_in x D_out)
                b in R^(D_out)
                Y in R^(N x D_out)
                
        Backward:
            dL/dW = X^T @ dY       in R^(D_in x D_out)
            dL/db = sum(dY, axis=0) in R^(D_out)
            dL/dX = dY @ W^T       in R^(N x D_in)
            
    Attributes:
        in_features (int): Dimensionality of input feature vectors (D_in).
        out_features (int): Number of output units / neurons (D_out).
        W (np.ndarray): Weight matrix of shape (D_in, D_out).
        b (np.ndarray): Bias vector of shape (D_out,).
        dW (np.ndarray): Gradient of loss with respect to W.
        db (np.ndarray): Gradient of loss with respect to b.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        seed: Optional[int] = None
    ):
        super().__init__()
        self.trainable = True
        self.in_features = in_features
        self.out_features = out_features
        
        # He (Kaiming) Weight Initialization: std = sqrt(2 / fan_in)
        std = np.sqrt(2.0 / in_features)
        
        rng = np.random.default_rng(seed)
        self.W = rng.normal(loc=0.0, scale=std, size=(in_features, out_features)).astype(np.float64)
        self.b = np.zeros(out_features, dtype=np.float64)
        
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        
        self.params = {"W": self.W, "b": self.b}
        self.grads = {"W": self.dW, "b": self.db}
        
        self.cached_x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for Dense layer.
        
        Args:
            x: Input 2D tensor of shape (N, D_in).
            
        Returns:
            Linear output tensor of shape (N, D_out).
        """
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected 2D input with shape (N, {self.in_features}), got {x.shape}"
            )
        self.cached_x = x
        return np.dot(x, self.W) + self.b

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass for Dense layer.
        
        Args:
            grad_output: Upstream gradient dL/dY of shape (N, D_out).
            
        Returns:
            dX: Gradient with respect to input dL/dX of shape (N, D_in).
        """
        if self.cached_x is None:
            raise RuntimeError("Cannot run backward() before calling forward().")
            
        # 1. Parameter gradients
        self.dW = np.dot(self.cached_x.T, grad_output)  # (D_in, N) @ (N, D_out) -> (D_in, D_out)
        self.db = np.sum(grad_output, axis=0)           # sum over batch N -> (D_out,)
        
        self.grads["W"] = self.dW
        self.grads["b"] = self.db
        
        # 2. Input gradient
        dX = np.dot(grad_output, self.W.T)              # (N, D_out) @ (D_out, D_in) -> (N, D_in)
        return dX

    def __repr__(self) -> str:
        return f"Dense(in_features={self.in_features}, out_features={self.out_features})"
